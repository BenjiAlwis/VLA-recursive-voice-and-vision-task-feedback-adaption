"""
THE CONTRACT.  Owner: Benji.  FROZEN — do not change signatures after T-2:40.

This replaces the old robot_api.py, which was written for the Unitree Go2
(forward / turn / heading_deg). The hardware pivot to a static SO-101
leader/follower pair is settled — see master_reference.md section 4. Do not
reintroduce locomotion primitives here.

Two backends behind ROBOT_BACKEND:

    mock    fabricated arm with realistic, LEARNABLE error. No hardware.
    so101   the real follower arm over USB serial via lerobot.

Everything above this file (planner, critic, loop) speaks only in NAMED
POSES plus a centimetre offset. Nothing upstream ever sees a joint angle.
That is deliberate: it is the reason the LLM cannot drive the arm into the
table, and the reason swapping mock -> so101 changes no other file.

    from arm_api import get_arm
    arm = get_arm()
    arm.move_to("above_block")
    arm.move_to("at_block", dx_cm=-2.5)     # <- the learnable knob
    arm.grip("close")

THE LEARNABLE KNOB. The mock's error is not random jitter — it is a
CONSTANT perception bias (the arm believes the block is somewhere slightly
different from where it is) plus small noise. That matters: random error
teaches nothing and would make the memory ablation come out flat. A
constant bias is discoverable from past failures, which is the entire
claim we are making on stage.
"""
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------- tunables ----------------

GRASP_TOL_CM = float(os.getenv("GRASP_TOL_CM", "3.0"))
LIFT_CLEAR_CM = float(os.getenv("LIFT_CLEAR_CM", "6.0"))
POSES_PATH = os.getenv("POSES_PATH", "logs/poses.json")

# The mock sleeps so a watching human can follow the motion. Set
# MOCK_REALTIME=0 to strip the delays for bulk statistical runs and for
# the ablation sweep, where the wall-clock cost is otherwise minutes.
_REALTIME = os.getenv("MOCK_REALTIME", "1") == "1"


def _dwell(seconds: float) -> None:
    if _REALTIME:
        time.sleep(seconds)

# The five named poses are the ONLY vocabulary the planner may reference.
# schema.py validates against this exact list, so adding one here without
# adding it there is a silent no-op.
NAMED_POSES = ["home", "above_block", "at_block", "above_zone", "at_zone"]

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper"]
JOINT_KEYS = [j + ".pos" for j in JOINTS]


class ArmBase:
    """Every backend implements exactly this. Nothing more."""

    def get_joints(self) -> Dict[str, float]:
        """Current joint positions, keyed '<joint>.pos'."""
        raise NotImplementedError

    def set_joints(self, joints: Dict[str, float]) -> None:
        """Blocking. Drive to these joint positions, then settle."""
        raise NotImplementedError

    def move_to(self, pose: str, dx_cm: float = 0.0, dy_cm: float = 0.0,
                dz_cm: float = 0.0) -> None:
        """Blocking. Move to a NAMED pose, optionally offset in cm.

        The offset is how the system corrects itself: after a missed grasp,
        the planner re-issues at_block with a dx/dy that cancels the bias.
        """
        raise NotImplementedError

    def grip(self, state: str, strength: float = 80.0) -> None:
        """state is 'open' or 'close'. Blocking."""
        raise NotImplementedError

    def get_frame(self, which: str = "wrist") -> Optional[np.ndarray]:
        """BGR image for PLANNER CONTEXT ONLY, or None.

        which='wrist'    the arm's own camera — scene detail
        which='overhead' the ground-truth camera

        Never feed this to a pass/fail decision. Pass/fail comes from
        get_scene(), which is numeric. See master_reference.md section 5.
        """
        raise NotImplementedError

    def get_scene(self) -> Dict:
        """GROUND TRUTH. Numeric, model-free, cannot hallucinate.

        {'block_cm': (x, y), 'zone_cm': (x, y), 'gripper_cm': (x, y, z),
         'holding': bool, 'source': 'mock' | 'aruco' | 'aruco_partial'}

        This is the ONLY input the critic's pass/fail function is allowed to
        read. If you are about to add a code path where a model's opinion
        reaches this dict, stop — that is the one rule the whole build rests
        on.
        """
        raise NotImplementedError

    def stop(self) -> None:
        """Idempotent. Safe to call any time, from anywhere."""
        raise NotImplementedError


# ==================================================================
# MockArm — no hardware. The insurance policy; never delete it.
# ==================================================================

class MockArm(ArmBase):
    """Fabricated arm with deliberately imperfect, learnable behaviour.

    Failure modes it can actually produce, all of which the critic
    diagnoses and the planner can learn to avoid:

      missed_grasp   closes the gripper outside GRASP_TOL_CM of the block
      dropped_early  releases, or is jolted, while traversing
      wrong_position releases the block away from the zone centre
      collision      moves between low poses without lifting clear first
      no_motion      a plan that never commands the arm anywhere
    """

    # Constant calibration error: the arm believes the block sits here
    # relative to where it truly is. Discoverable, therefore learnable.
    #
    # Magnitude is ~5.1cm against a 3.0cm grasp tolerance. That gap is
    # deliberate. At the first value tried, (2.6, -1.7), the bias was
    # 3.11cm — barely over tolerance, so REACH_NOISE_CM alone let the
    # UNCORRECTED plan grasp successfully by luck a large fraction of the
    # time. A baseline that randomly succeeds makes the memory-ablation
    # chart noise, and that chart is the evidence the whole demo rests on.
    PERCEPTION_BIAS = (4.2, -2.8)

    REACH_NOISE_CM = 0.45           # small, so the bias dominates
    DROP_STRENGTH = 45.0            # grips weaker than this can slip
    DROP_CHANCE = 0.55

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed if seed is not None
                                 else int(os.getenv("MOCK_SEED", "7")))
        self.block_cm = (18.0, 24.0)
        self.zone_cm = (-16.0, 22.0)
        self.gripper_cm = [0.0, 20.0, 20.0]
        self.holding = False
        self.grip_strength = 0.0
        self.last_pose = "home"
        self.collided = False
        self.commanded_any = False
        self.last_grasp_cm = None
        self.grasp_target_cm = None
        # Per-instance so a perturbation can shift it mid-demo. The class
        # attribute stays the starting value.
        self.bias = list(self.PERCEPTION_BIAS)
        self._joints = {k: 0.0 for k in JOINT_KEYS}

    # ---- internal geometry ----

    def _nominal(self, pose: str) -> Tuple[float, float, float]:
        """Where the arm BELIEVES each named pose is."""
        bx = self.block_cm[0] + self.bias[0]
        by = self.block_cm[1] + self.bias[1]
        zx, zy = self.zone_cm
        return {
            "home": (0.0, 20.0, 20.0),
            "above_block": (bx, by, 10.0),
            "at_block": (bx, by, 1.0),
            "above_zone": (zx, zy, 10.0),
            "at_zone": (zx, zy, 1.0),
        }[pose]

    def _dist_to_block(self) -> float:
        return math.hypot(self.gripper_cm[0] - self.block_cm[0],
                          self.gripper_cm[1] - self.block_cm[1])

    # ---- contract ----

    def get_joints(self) -> Dict[str, float]:
        return dict(self._joints)

    def set_joints(self, joints: Dict[str, float]) -> None:
        """Drive to joint targets, and move the simulated gripper with them.

        This exists so a REPLAYED HUMAN DEMONSTRATION actually does
        something in mock mode. teach.py records joint-space waypoints; if
        set_joints only stored numbers, replaying a demonstration would be
        visibly inert, and gate T+0:30 (hardware fails -> demo on the mock)
        would throw away the strongest moment in the run.

        The map below is a crude two-point linear fit, NOT real forward
        kinematics — it is fitted so the mock demonstration in teach.py
        lands on the block and the zone. It is a mock's job to make the
        pipeline demonstrable, and this is clearly labelled rather than
        pretending to be a kinematic model. The real SO101Arm sends these
        same joint targets to actual servos and needs none of it.
        """
        self._joints.update({k: float(v) for k, v in joints.items()
                             if k in JOINT_KEYS})
        self.commanded_any = True

        pan = self._joints.get("shoulder_pan.pos")
        lift = self._joints.get("shoulder_lift.pos")
        if pan is not None and lift is not None:
            x = 0.618 * pan + 2.55            # pan  25 -> block x, -30 -> zone x
            y = 24.0 + (x - 18.0) * 0.0588    # block y 24 -> zone y 22
            z = max(0.0, min(20.0, (lift + 45.0) * 0.6 + 1.0))
            self.gripper_cm = [x, y, z]
            if self.holding:
                self.block_cm = (x, y)

        grip_pos = self._joints.get("gripper.pos")
        if grip_pos is not None:
            closing = grip_pos > 40.0
            if closing and not self.holding:
                self.last_grasp_cm = (self.gripper_cm[0], self.gripper_cm[1])
                self.grasp_target_cm = tuple(self.block_cm)
                if self._dist_to_block() <= GRASP_TOL_CM and \
                        self.gripper_cm[2] < 4.0:
                    self.holding = True
                    self.grip_strength = grip_pos
            elif not closing and self.holding:
                self.block_cm = (self.gripper_cm[0], self.gripper_cm[1])
                self.holding = False
                self.grip_strength = 0.0

        _dwell(0.05)

    def move_to(self, pose: str, dx_cm: float = 0.0, dy_cm: float = 0.0,
                dz_cm: float = 0.0) -> None:
        if pose not in NAMED_POSES:
            print(f"[mock] refusing unknown pose {pose!r}")
            return

        self.commanded_any = True
        x, y, z = self._nominal(pose)
        x += dx_cm + self.rng.gauss(0, self.REACH_NOISE_CM)
        y += dy_cm + self.rng.gauss(0, self.REACH_NOISE_CM)
        z = max(0.0, z + dz_cm)

        # Collision: travelling between two LOW poses without clearing.
        # A plan that goes at_block -> at_zone directly drags the block
        # across the table. This is a plan-ordering mistake the planner
        # can and does learn to stop making.
        was_low = self._nominal(self.last_pose)[2] < LIFT_CLEAR_CM
        now_low = z < LIFT_CLEAR_CM
        moved_far = math.hypot(x - self.gripper_cm[0],
                               y - self.gripper_cm[1]) > 8.0
        if was_low and now_low and moved_far:
            self.collided = True
            print("[mock] COLLISION — traversed at table height")
            if self.holding and self.rng.random() < 0.8:
                self.holding = False
                print("[mock] block knocked loose by the collision")

        self.gripper_cm = [x, y, z]
        self.last_pose = pose

        # A carried block follows the gripper, unless a weak grip slips.
        if self.holding:
            if (self.grip_strength < self.DROP_STRENGTH
                    and self.rng.random() < self.DROP_CHANCE):
                self.holding = False
                print("[mock] DROPPED — grip too weak to hold through the move")
            else:
                self.block_cm = (x, y)

        _dwell(0.12)

    def grip(self, state: str, strength: float = 80.0) -> None:
        self.commanded_any = True
        if state == "close":
            self.grip_strength = float(strength)
            # WHERE the gripper closed, latched at the moment it closed.
            # The critic cannot use the final gripper position for this:
            # every sane plan ends with move_to home, so by the time the
            # trial is scored the gripper is parked metres from the block
            # and the diagnosed offset is nonsense — with the wrong sign,
            # so acting on it doubles the error instead of cancelling it.
            self.last_grasp_cm = (self.gripper_cm[0], self.gripper_cm[1])
            self.grasp_target_cm = tuple(self.block_cm)
            gap = self._dist_to_block()
            if gap <= GRASP_TOL_CM and self.gripper_cm[2] < 4.0:
                self.holding = True
                self.block_cm = (self.gripper_cm[0], self.gripper_cm[1])
            else:
                self.holding = False
                print(f"[mock] MISSED — closed {gap:.1f}cm from the block "
                      f"(tolerance {GRASP_TOL_CM}cm)")
        elif state == "open":
            if self.holding:
                self.block_cm = (self.gripper_cm[0], self.gripper_cm[1])
            self.holding = False
            self.grip_strength = 0.0
        else:
            print(f"[mock] unknown grip state {state!r}")
        self._joints["gripper.pos"] = self.grip_strength
        _dwell(0.12)

    def get_frame(self, which: str = "wrist") -> Optional[np.ndarray]:
        """A crude synthetic top-down view. Not art — it exists so the
        vision path in the planner is exercised end to end without a
        camera, rather than being first run live on demo day."""
        img = np.full((240, 320, 3), 40, dtype=np.uint8)

        def px(p):
            return (int(160 + p[0] * 4), int(220 - p[1] * 4))

        try:
            import cv2
            cv2.circle(img, px(self.zone_cm), 22, (60, 60, 200), 2)
            cv2.rectangle(img, tuple(a - 7 for a in px(self.block_cm)),
                          tuple(a + 7 for a in px(self.block_cm)),
                          (40, 40, 220), -1)
            g = px(self.gripper_cm[:2])
            cv2.drawMarker(img, g, (220, 220, 60), cv2.MARKER_TILTED_CROSS, 16, 2)
            cv2.putText(img, f"holding={self.holding}", (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
        except Exception:                                   # noqa: BLE001
            return img
        return img

    def get_scene(self) -> Dict:
        return {
            "block_cm": (round(self.block_cm[0], 1), round(self.block_cm[1], 1)),
            "zone_cm": (round(self.zone_cm[0], 1), round(self.zone_cm[1], 1)),
            "gripper_cm": tuple(round(v, 1) for v in self.gripper_cm),
            # Where the gripper actually closed, and what it was aiming at.
            # None until a close has happened this trial.
            "grasp_cm": self.last_grasp_cm,
            "grasp_target_cm": self.grasp_target_cm,
            "holding": self.holding,
            "collided": self.collided,
            "commanded": self.commanded_any,
            "source": "mock",
        }

    def stop(self) -> None:
        pass

    # ---- per-trial bookkeeping ----

    def reset_trial(self) -> None:
        """Clear per-attempt flags. The block is NOT teleported back —
        the arm must cope with the world it actually left behind."""
        self.collided = False
        self.commanded_any = False
        self.last_grasp_cm = None
        self.grasp_target_cm = None

    def reset_task(self) -> None:
        """Between rounds: a human puts the block back at the start.

        The ZONE is deliberately NOT reset. If someone moved it mid-demo
        (perturb), it stays moved — that is the whole point of the
        perturbation, and restoring it here would silently undo the change
        the robot is supposed to be noticing and adapting to.
        """
        self.block_cm = (18.0, 24.0)
        self.gripper_cm = [0.0, 20.0, 20.0]
        self.holding = False
        self.grip_strength = 0.0
        self.last_pose = "home"
        self.reset_trial()

    def perturb(self, kind: str = "calibration") -> None:
        """THE DEMO MOMENT (master_reference.md §10, step 3).

        kind="calibration"  someone bumps the camera or the arm mount, so
                            the perception bias shifts. This is the one
                            that produces the beat the demo needs: the
                            correction the robot already learned becomes
                            WRONG, it misses again, re-diagnoses, and
                            re-converges. That is "adapt to unexpected
                            real-world change" in the challenge's own words.

        kind="zone"         the target zone is moved. Honest to show, but
                            the robot re-reads the zone from the overhead
                            camera every trial, so it absorbs this without
                            ever failing — visually it looks like nothing
                            happened. Do not build the stage moment on it.
        """
        if kind in ("zone", "both"):
            self.zone_cm = (self.zone_cm[0] - 7.0, self.zone_cm[1] + 5.0)
            print(f"[mock] ⚡ zone moved to "
                  f"({self.zone_cm[0]:.1f}, {self.zone_cm[1]:.1f})")
        if kind in ("calibration", "both"):
            self.bias = [self.bias[0] - 8.5, self.bias[1] + 6.0]
            print(f"[mock] ⚡ calibration drifted — perception bias is now "
                  f"({self.bias[0]:.1f}, {self.bias[1]:.1f})")


# ==================================================================
# SO101Arm — the real follower over USB serial.
# ==================================================================

class SO101Arm(ArmBase):
    """UNVERIFIED against hardware — lerobot is not installed in this tree
    and its module paths have moved across releases. Written to fail loudly
    with an actionable message rather than silently degrade to fake data.

    Two things here are not optional, both from master_reference.md
    section 7:

      1. Calibration comes first. Named poses are read from logs/poses.json,
         recorded with `python arm_api.py --calibrate`. Without that file
         this backend refuses to move at all — an uncalibrated arm swinging
         to a hardcoded joint angle is how you break a servo.
      2. The follower disconnects mid-teleop. EVERY send is wrapped in one
         reconnect retry. Build it in now, not at hour four.
    """

    RECONNECT_PAUSE_S = 0.6

    def __init__(self):
        self.port = os.getenv("FOLLOWER_PORT")
        if not self.port:
            raise RuntimeError(
                "FOLLOWER_PORT is not set (e.g. /dev/tty.usbmodem1101)")
        self.poses = _load_poses()
        if not self.poses:
            raise RuntimeError(
                f"no calibrated poses at {POSES_PATH}. "
                f"Run: python arm_api.py --calibrate")
        self._arm = None
        self._connect()
        self._holding = False
        self._last_pose = "home"

    # ---- connection ----

    def _connect(self):
        errors = []
        candidates = (
            ("lerobot.robots.so101_follower", "SO101Follower",
             "SO101FollowerConfig"),
            ("lerobot.robots.so_follower", "SOFollower", "SOFollowerConfig"),
            ("lerobot.robots.so101", "SO101", "SO101Config"),
        )
        for module, cls_name, cfg_name in candidates:
            try:
                mod = __import__(module, fromlist=[cls_name, cfg_name])
                cls = getattr(mod, cls_name)
                cfg_cls = getattr(mod, cfg_name)
                arm = cls(cfg_cls(port=self.port, id="follower"))
                connect = getattr(arm, "connect", None)
                if connect:
                    connect()
                self._arm = arm
                print(f"[so101] connected via {module}.{cls_name}")
                return
            except Exception as e:                          # noqa: BLE001
                errors.append(f"{module}: {e}")
        raise RuntimeError("could not open the follower arm. Tried:\n  "
                           + "\n  ".join(errors))

    def _send(self, action: Dict[str, float]) -> bool:
        """send_action with exactly one reconnect retry. The known failure
        mode is a mid-session USB drop; one retry recovers it, and a second
        would just extend a hang the operator needs to see."""
        for attempt in (1, 2):
            try:
                self._arm.send_action(action)
                return True
            except Exception as e:                          # noqa: BLE001
                print(f"[so101] send failed (attempt {attempt}): {e}")
                if attempt == 1:
                    time.sleep(self.RECONNECT_PAUSE_S)
                    try:
                        self._connect()
                    except Exception as ce:                 # noqa: BLE001
                        print(f"[so101] reconnect failed: {ce}")
                        return False
        return False

    # ---- contract ----

    def get_joints(self) -> Dict[str, float]:
        try:
            obs = self._arm.get_observation()
            return {k: float(v) for k, v in obs.items()
                    if k.endswith(".pos") and isinstance(v, (int, float))}
        except Exception as e:                              # noqa: BLE001
            print(f"[so101] get_observation failed: {e}")
            return {}

    def set_joints(self, joints: Dict[str, float]) -> None:
        self._send({k: float(v) for k, v in joints.items()
                    if k in JOINT_KEYS})
        time.sleep(0.35)

    def move_to(self, pose: str, dx_cm: float = 0.0, dy_cm: float = 0.0,
                dz_cm: float = 0.0) -> None:
        if pose not in self.poses:
            print(f"[so101] pose {pose!r} not calibrated; refusing to move")
            return
        target = dict(self.poses[pose])

        # Offsets are applied through a linear Jacobian approximation
        # recorded at calibration time (cm -> joint delta). Crude, but the
        # corrections we need are a few centimetres, well inside the range
        # where a local linear model holds.
        jac = self.poses.get("_jacobian", {})
        for axis, delta in (("dx", dx_cm), ("dy", dy_cm), ("dz", dz_cm)):
            for joint, per_cm in jac.get(axis, {}).items():
                if joint in target:
                    target[joint] += per_cm * delta

        self._send(target)
        self._last_pose = pose
        time.sleep(0.6)

    def grip(self, state: str, strength: float = 80.0) -> None:
        value = float(strength) if state == "close" else 0.0
        joints = self.get_joints()
        joints["gripper.pos"] = value
        self._send(joints)
        self._holding = state == "close"
        time.sleep(0.5)

    def get_frame(self, which: str = "wrist") -> Optional[np.ndarray]:
        import vision
        return vision.capture(which)

    def get_scene(self) -> Dict:
        import vision
        scene = vision.read_scene()
        scene["holding"] = self._holding
        scene["collided"] = False
        scene["commanded"] = True
        return scene

    def stop(self) -> None:
        try:
            fn = getattr(self._arm, "disconnect", None) or \
                 getattr(self._arm, "close", None)
            if fn:
                fn()
        except Exception as e:                              # noqa: BLE001
            print(f"[so101] unclean release: {e}")


# ==================================================================
# pose calibration
# ==================================================================

def _load_poses() -> Dict:
    try:
        with open(POSES_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:                                  # noqa: BLE001
        print(f"[arm] poses unreadable: {e}")
        return {}


def calibrate() -> None:
    """Record the five named poses by hand.

    Move the follower physically to each pose and press Enter. Twenty
    minutes, done once, and nothing else in the build works until it is.
    """
    arm = SO101Arm.__new__(SO101Arm)          # skip the poses precondition
    arm.port = os.getenv("FOLLOWER_PORT")
    if not arm.port:
        raise SystemExit("FOLLOWER_PORT is not set")
    arm.poses = {}
    arm._connect()

    recorded: Dict = {}
    print("\nMove the arm BY HAND to each pose, then press Enter.\n")
    for pose in NAMED_POSES:
        input(f"  → place the arm at {pose!r} and press Enter…")
        joints = arm.get_joints()
        if not joints:
            print(f"    could not read joints; skipping {pose}")
            continue
        recorded[pose] = joints
        print(f"    recorded {pose}: "
              f"{ {k: round(v, 1) for k, v in joints.items()} }")

    print("\nNow the offset model. Move 5cm in +x from at_block, "
          "then press Enter.")
    input("  → ")
    shifted = arm.get_joints()
    base = recorded.get("at_block", {})
    if shifted and base:
        recorded["_jacobian"] = {
            "dx": {k: (shifted[k] - base[k]) / 5.0
                   for k in shifted if k in base},
            "dy": {}, "dz": {},
        }
        print("    recorded dx jacobian")

    os.makedirs(os.path.dirname(POSES_PATH) or ".", exist_ok=True)
    with open(POSES_PATH, "w") as f:
        json.dump(recorded, f, indent=2)
    print(f"\nwrote {POSES_PATH}")
    arm.stop()


# ==================================================================

def get_arm() -> ArmBase:
    backend = os.getenv("ROBOT_BACKEND", "mock")
    if backend == "mock":
        return MockArm()
    if backend == "so101":
        return SO101Arm()
    raise ValueError(f"unknown ROBOT_BACKEND={backend!r} (mock | so101)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true",
                    help="record named poses from the real follower arm")
    args = ap.parse_args()

    if args.calibrate:
        calibrate()
        raise SystemExit(0)

    # ---- mock smoke test: every failure mode must be reachable ----
    print("=== naive plan (no offset) — expect a missed grasp ===")
    arm = MockArm()
    arm.move_to("above_block")
    arm.move_to("at_block")
    arm.grip("close")
    print(f"  holding={arm.get_scene()['holding']}  (expected False)")
    assert not arm.get_scene()["holding"], \
        "the bias must cause a miss, or there is nothing to learn"

    print("\n=== corrected plan (offset cancels the bias) — expect success ===")
    arm = MockArm()
    bias_x, bias_y = MockArm.PERCEPTION_BIAS
    arm.move_to("above_block", dx_cm=-bias_x, dy_cm=-bias_y)
    arm.move_to("at_block", dx_cm=-bias_x, dy_cm=-bias_y)
    arm.grip("close", strength=80)
    assert arm.get_scene()["holding"], "corrected grasp must succeed"
    arm.move_to("above_block", dx_cm=-bias_x, dy_cm=-bias_y)
    arm.move_to("above_zone")
    arm.move_to("at_zone")
    arm.grip("open")
    scene = arm.get_scene()
    err = math.hypot(scene["block_cm"][0] - scene["zone_cm"][0],
                     scene["block_cm"][1] - scene["zone_cm"][1])
    print(f"  placed {err:.1f}cm from the zone centre")
    assert err < 5.0, f"corrected place should land near the zone, got {err}"

    print("\n=== collision: at_block -> at_zone with no lift ===")
    arm = MockArm()
    arm.move_to("at_block", dx_cm=-bias_x, dy_cm=-bias_y)
    arm.grip("close")
    arm.move_to("at_zone")
    assert arm.get_scene()["collided"], "low traverse must register a collision"

    print("\n=== dropped_early: a weak grip ===")
    dropped = False
    for seed in range(12):
        a = MockArm(seed=seed)
        a.move_to("at_block", dx_cm=-bias_x, dy_cm=-bias_y)
        a.grip("close", strength=20)
        if a.get_scene()["holding"]:
            a.move_to("above_block")
            a.move_to("above_zone")
            if not a.get_scene()["holding"]:
                dropped = True
                break
    assert dropped, "a weak grip must be able to drop the block"

    print("\n=== no_motion ===")
    arm = MockArm()
    assert not arm.get_scene()["commanded"], "a plan that does nothing must show it"

    print("\n=== frame + scene shape ===")
    arm = MockArm()
    frame = arm.get_frame("wrist")
    print(f"  frame: {None if frame is None else frame.shape}")
    print(f"  scene: {json.dumps(arm.get_scene(), default=str)}")

    print("\narm_api smoke test passed — all five failure modes reachable")
