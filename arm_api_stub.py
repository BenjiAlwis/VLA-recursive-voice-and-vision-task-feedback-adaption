"""
DEV-ONLY stand-in for arm_api.py.  Owner: Aaryan.

DELETE THIS FILE the moment Benji ships the real arm_api.py. teach.py
imports arm_api first and only falls back to this, so when the real
contract lands this stub stops being used with zero edits to teach.py.

It exists for one reason: teach.py has to be developable and testable
with no hardware and no calibration, and arm_api.py is not in this tree.

Contract mirrored here:
    get_arm(role="follower"|"leader") -> Arm
    Arm.get_joints()   -> {"shoulder_pan.pos": float, ...}   6 keys
    Arm.set_joints(d)  -> None
    Arm.grip(closed)   -> None
    Arm.get_frame()    -> frame or None
    Arm.reconnect()    -> None

The mock LEADER does not sit still. It plays a scripted reach → grasp →
lift trajectory, where the grasp is a deliberately fast gripper snap.
That shape is the point: a downsampler that keeps waypoints where joints
move fastest must put visibly more of them in the snap than in the smooth
sweep. Against a constant or purely random mock, velocity weighting and
uniform sampling are indistinguishable and the logic is untested.

The mock FOLLOWER drops its connection at random, because the real one
does (see the KNOWN ISSUE in the project brief). That keeps teach._send's
reconnect path exercised in development instead of first firing on stage.
"""
import math
import os
import random
import time
from typing import Dict, Optional

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper"]
KEYS = [j + ".pos" for j in JOINTS]

# Fraction of set_joints calls that fake a mid-teleop disconnect.
DISCONNECT_RATE = float(os.getenv("MOCK_DISCONNECT_RATE", "0.02"))

# How long one full scripted demonstration takes before it loops.
SCRIPT_SECONDS = float(os.getenv("MOCK_SCRIPT_SECONDS", "10.0"))

NOISE = 0.015


def _frac(p: float, lo: float, hi: float) -> float:
    """Progress through the sub-interval [lo, hi], clamped to 0..1."""
    return max(0.0, min(1.0, (p - lo) / (hi - lo)))


def _smooth(t: float) -> float:
    """Smoothstep. Gives the sweeps a soft start/stop so their velocity
    genuinely varies — the snap below is left un-eased on purpose."""
    return t * t * (3.0 - 2.0 * t)


def _script(p: float) -> Dict[str, float]:
    """Scripted leader pose at normalised time p in [0, 1].

    0.00–0.45  reach toward the block   (smooth, slow)
    0.45–0.50  GRASP — gripper snaps    (fast, un-eased: this is the moment
                                         uniform sampling throws away)
    0.50–1.00  lift and transport       (smooth, slow)
    """
    reach = _smooth(_frac(p, 0.00, 0.45))
    snap = _frac(p, 0.45, 0.50)
    move = _smooth(_frac(p, 0.50, 1.00))
    return {
        "shoulder_pan.pos": 35.0 * reach - 55.0 * move,
        "shoulder_lift.pos": -35.0 * reach + 25.0 * move,
        "elbow_flex.pos": 45.0 * reach - 15.0 * move,
        "wrist_flex.pos": 10.0 * reach,
        "wrist_roll.pos": 25.0 * move,
        "gripper.pos": 80.0 * snap,
    }


class MockArm:
    """Same five methods as the real backend. No hardware, no calibration."""

    def __init__(self, role: str = "follower"):
        self.role = role
        self._t0 = time.monotonic()
        self._joints = {k: 0.0 for k in KEYS}
        self._connected = True

    # ---- contract ----

    def get_joints(self) -> Dict[str, float]:
        if self.role == "leader":
            # A human is "moving" it: play the script once, then HOLD the
            # final pose. Deliberately not looping — wrapping back to the
            # start would teleport every joint at once, and that fake
            # discontinuity would dominate any motion-weighted waypoint
            # selection downstream. Humans finish and hold still.
            p = min(1.0, (time.monotonic() - self._t0) / SCRIPT_SECONDS)
            self._joints = {k: v + random.gauss(0, NOISE)
                            for k, v in _script(p).items()}
        return dict(self._joints)

    def set_joints(self, joints: Dict[str, float]) -> None:
        if not self._connected:
            raise ConnectionError("mock follower is disconnected")
        if random.random() < DISCONNECT_RATE:
            self._connected = False
            raise ConnectionError("mock follower dropped mid-teleop")
        for k, v in joints.items():
            if k in self._joints:
                self._joints[k] = float(v)

    def grip(self, closed: bool) -> None:
        self.set_joints({"gripper.pos": 80.0 if closed else 0.0})

    def get_frame(self) -> Optional[object]:
        return None

    def reconnect(self) -> None:
        time.sleep(0.05)
        self._connected = True

    # ---- niceties so `with get_arm() as a:` works either way ----

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._connected = False
        return False


def get_arm(role: str = "follower") -> MockArm:
    """Role defaults to 'follower' so single-arm call sites are unaffected."""
    if role not in ("leader", "follower"):
        raise ValueError(f"unknown arm role {role!r}")
    return MockArm(role)
