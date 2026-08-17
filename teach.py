"""
Leader-arm demonstration recorder.  Owner: Aaryan (RED team).

After two consecutive failures a human physically moves the LEADER arm
through the correct motion. We record that trajectory, compress it to the
waypoints that matter, and save it as a named skill.

=== BOUNDARY — READ THIS BEFORE EDITING ===

Red never commands a motor. This file only ever:
  - READS joint states (lerobot get_observation, or a fabricated mock)
  - WRITES JSON to logs/skills/ and shared/red_to_blue.json

It does NOT mirror the leader to the follower, does NOT replay, does NOT
call send_action. Mirroring and replay are BLUE actions. We ask for a
replay by writing `replay_skill` into the Red->Blue message and Blue
decides when to run it.

An earlier version of this file did drive the follower. That was removed
deliberately — if you are reintroducing a write path to any arm here,
you are crossing the team boundary.

    import teach
    teach.record_demonstration("pick_block", seconds=10, source="mock")
    teach.list_skills()
    teach.request_replay("pick_block")

Benji: call record_demonstration() at the escalation branch, then
request_replay() to hand it to Blue. I never touch the shared loop.
"""
import glob
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional

try:
    import narrate

    def _say(text: str, block: bool = True) -> None:
        narrate.speak(text, block=block)
except Exception as _e:                                     # noqa: BLE001
    print(f"[teach] narrate unavailable ({_e}); speech falls back to print")

    def _say(text: str, block: bool = True) -> None:
        print(f"  🗣  {text}")


SKILL_DIR = "logs/skills"
SHARED_DIR = os.getenv("SHARED_DIR", "shared")
RED_TO_BLUE = os.path.join(SHARED_DIR, "red_to_blue.json")

HZ = float(os.getenv("TEACH_HZ", "30"))
WAYPOINTS = int(os.getenv("TEACH_WAYPOINTS", "20"))

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper"]
KEYS = [j + ".pos" for j in JOINTS]

# SO-101 ".pos" values are normalised to roughly +/-100. Waypoint
# selection scores motion against this FIXED scale rather than each
# joint's observed range: range-normalising amplifies sensor noise on
# joints that barely moved, which is exactly the wrong thing to weight on.
JOINT_SCALE = 100.0


# ---------------- mock source ----------------
# A scripted pick-and-place, used when there is no hardware. It is not
# decoration: the mock has to contain fast moments (the grasp and the
# release) or the motion-weighted downsampler below is indistinguishable
# from uniform sampling and its logic goes untested.
#
# Keyframes are (progress, pose). Two of them are deliberately only 5% of
# the run apart with the gripper swinging 80 units — those are the grasp
# and the release, the parts a replay must not miss.

_KEYFRAMES = [
    (0.00, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (0.30, (25.0, -30.0, 40.0, 10.0, 0.0, 0.0)),    # reach over the block
    (0.38, (25.0, -45.0, 55.0, 15.0, 0.0, 0.0)),    # descend onto it
    (0.43, (25.0, -45.0, 55.0, 15.0, 0.0, 80.0)),   # GRASP  <- fast
    (0.55, (25.0, -20.0, 35.0, 10.0, 0.0, 80.0)),   # lift
    (0.80, (-30.0, -20.0, 35.0, 10.0, 20.0, 80.0)),  # traverse to the zone
    (0.86, (-30.0, -42.0, 52.0, 14.0, 20.0, 80.0)),  # descend
    (0.91, (-30.0, -42.0, 52.0, 14.0, 20.0, 0.0)),  # RELEASE <- fast
    (1.00, (-10.0, 0.0, 0.0, 0.0, 0.0, 0.0)),       # retreat
]

MOCK_NOISE = 0.015


def _smooth(t: float) -> float:
    """Smoothstep — soft start/stop, so velocity genuinely varies."""
    return t * t * (3.0 - 2.0 * t)


def _mock_pose(p: float) -> Dict[str, float]:
    """Scripted leader pose at normalised progress p in [0, 1]."""
    p = max(0.0, min(1.0, p))
    for (p0, a), (p1, b) in zip(_KEYFRAMES, _KEYFRAMES[1:]):
        if p <= p1 or p1 == 1.0:
            span = p1 - p0
            f = _smooth((p - p0) / span) if span > 0 else 0.0
            return {k: a[i] + (b[i] - a[i]) * f + random.gauss(0, MOCK_NOISE)
                    for i, k in enumerate(KEYS)}
    return {k: _KEYFRAMES[-1][1][i] for i, k in enumerate(KEYS)}


# ---------------- leader source ----------------

def _open_leader():
    """Open the LEADER arm READ-ONLY and return something with
    get_observation().

    UNVERIFIED: lerobot is not installed in this tree, and its module
    paths have moved across releases. The candidates below are tried in
    order. If none import, this raises with a clear message rather than
    silently degrading to fabricated data — a mock trajectory recorded
    while a human is physically demonstrating is worse than a visible
    failure, because it looks like it worked.

    Only get_observation() is ever called. No send_action, ever.
    """
    port = os.getenv("LEADER_PORT")
    if not port:
        raise RuntimeError("LEADER_PORT is not set (e.g. /dev/tty.usbmodem1101)")

    errors = []
    candidates = (
        ("lerobot.teleoperators.so101_leader", "SO101Leader", "SO101LeaderConfig"),
        ("lerobot.teleoperators.so_leader", "SOLeader", "SOLeaderConfig"),
        ("lerobot.robots.so_leader", "SOLeader", "SOLeaderConfig"),
    )
    for module, cls_name, cfg_name in candidates:
        try:
            mod = __import__(module, fromlist=[cls_name, cfg_name])
            cls = getattr(mod, cls_name)
            cfg_cls = getattr(mod, cfg_name)
            arm = cls(cfg_cls(port=port, id="leader"))
            connect = getattr(arm, "connect", None)
            if connect:
                connect()
            if not hasattr(arm, "get_observation"):
                raise AttributeError(f"{cls_name} has no get_observation()")
            print(f"[teach] leader opened via {module}.{cls_name}")
            return arm
        except Exception as e:                              # noqa: BLE001
            errors.append(f"{module}: {e}")

    raise RuntimeError("could not open the leader arm. Tried:\n  "
                       + "\n  ".join(errors))


def _release(arm) -> None:
    """Close the serial port. Left open, the NEXT run cannot connect —
    which on demo day looks like broken hardware."""
    try:
        fn = getattr(arm, "disconnect", None) or getattr(arm, "close", None)
        if fn:
            fn()
    except Exception as e:                                  # noqa: BLE001
        print(f"[teach] could not release the leader cleanly: {e}")


def _only_positions(obs: Dict) -> Dict[str, float]:
    """Keep the six '.pos' keys; drop images and anything else."""
    return {k: float(v) for k, v in obs.items()
            if k.endswith(".pos") and isinstance(v, (int, float))}


# ---------------- waypoint selection ----------------

def _motion(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Joint-space distance between two samples, on the fixed scale."""
    return sum(abs(b[k] - a.get(k, b[k])) for k in b) / JOINT_SCALE


def downsample(samples: List[Dict], n: int = WAYPOINTS) -> List[Dict]:
    """Compress a ~30Hz recording to ~n waypoints.

    Resamples uniformly in CUMULATIVE JOINT MOTION, not in time. Where
    the arm moves fast, waypoints land close together in time; where it
    dwells, they thin out.

    That is the whole point. Ten seconds of teaching is mostly slow
    approach plus one fast critical grasp. Uniform time sampling spends
    its budget on the approach and smears the grasp across two distant
    waypoints, so the replay closes the gripper in the wrong place and
    misses the block.
    """
    if len(samples) <= n:
        return list(samples)

    cum = [0.0]
    for a, b in zip(samples, samples[1:]):
        cum.append(cum[-1] + _motion(a["joints"], b["joints"]))
    total = cum[-1]

    if total <= 1e-9:                   # never moved; nothing to weight on
        step = (len(samples) - 1) / (n - 1)
        return [samples[round(i * step)] for i in range(n)]

    picked, j = [], 0
    for i in range(n):
        target = total * i / (n - 1)
        while j < len(cum) - 1 and cum[j] < target:
            j += 1
        picked.append(j)

    seen, out = set(), []
    for i in picked:
        if i not in seen:
            seen.add(i)
            out.append(samples[i])
    if out[-1] is not samples[-1]:      # the final pose always matters
        out.append(samples[-1])
    return out


# ---------------- public API ----------------

def record_demonstration(name: str, seconds: int = 10, source: str = "mock",
                         task: str = "unknown") -> Optional[Dict]:
    """Record a human demonstration and save it as a named skill.

    source="mock"   fabricate a plausible pick-and-place. No hardware.
    source="leader" read the real leader arm (READ ONLY).

    Returns the skill dict, or None. Never raises.
    """
    if source not in ("mock", "leader"):
        print(f"[teach] unknown source {source!r}")
        return None

    arm = None
    try:
        if source == "leader":
            arm = _open_leader()        # raises with a clear message

        _say("Show me how to do this.")
        for count in ("Three", "Two", "One"):
            _say(count)
            time.sleep(0.7)
        _say("Go.")

        samples: List[Dict] = []
        period = 1.0 / HZ
        t0 = time.monotonic()
        next_t = t0

        while True:
            elapsed = time.monotonic() - t0
            if elapsed >= seconds:
                break

            if source == "mock":
                joints = _mock_pose(elapsed / seconds)
            else:
                try:
                    joints = _only_positions(arm.get_observation())
                except Exception as e:                      # noqa: BLE001
                    print(f"[teach] leader read failed: {e}")
                    joints = None

            if joints:
                samples.append({"t": round(elapsed, 4),
                                "ts": time.time(),
                                "joints": joints})

            # Deadline-based rather than sleep(period): a serial read per
            # tick would otherwise let the clock drift and make the
            # recorded duration a lie.
            next_t += period
            time.sleep(max(0.0, next_t - time.monotonic()))

        if len(samples) < 2:
            _say("I did not see anything. Let us try that again.")
            return None

        waypoints = downsample(samples)
        skill = {
            "name": name,
            "type": "demonstration",
            "source": source,
            "taught_for": task,
            "ts": time.time(),
            "duration_s": round(samples[-1]["t"], 3),
            "raw_samples": len(samples),
            "sample_hz": round(len(samples) / max(samples[-1]["t"], 1e-6), 1),
            "waypoint_count": len(waypoints),
            "joint_names": KEYS,
            "waypoints": waypoints,
        }
        if not _write_skill(name, skill):
            return None
        _say("Got it.")
        return skill

    except Exception as e:                                  # noqa: BLE001
        print(f"[teach] record_demonstration failed: {e}")
        return None
    finally:
        if arm is not None:
            _release(arm)


def list_skills() -> List[str]:
    """Names of every taught demonstration on disk."""
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(f"{SKILL_DIR}/*.json"))


def load_skill(name: str) -> Optional[Dict]:
    """The saved skill dict, or None if missing or unreadable."""
    path = _skill_path(name)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:                                  # noqa: BLE001
        print(f"[teach] could not read {path}: {e}")
        return None


def request_replay(name: str, reason: str = "", confidence: float = 0.8) -> bool:
    """Ask BLUE to replay a taught skill. We never replay it ourselves.

    Read-modify-write, because `replay_skill` is one field of a message
    Red also uses for prompt_update/reason/confidence. Writing a fresh
    object here would wipe whatever directive Reasoning last published.
    """
    if load_skill(name) is None:
        print(f"[teach] refusing to request replay of unknown skill {name!r}")
        return False
    return _update_red_to_blue({
        "replay_skill": name,
        "reason": reason or f"replaying taught skill '{name}'",
        "confidence": float(confidence),
    })


# ---------------- persistence ----------------

def _skill_path(name: str) -> str:
    safe = "".join(c for c in name if c.isalnum() or c in "._-")
    return f"{SKILL_DIR}/{safe or 'unnamed'}.json"


def _write_json_atomic(path: str, payload: Dict) -> bool:
    """Write via temp file + os.replace.

    Blue polls these files while we write them. json.dump into an open
    handle is not atomic — a poll landing mid-write reads truncated JSON
    and raises on their side. os.replace on the same filesystem is atomic.
    """
    try:
        parent = os.path.dirname(path) or "."
        os.makedirs(parent, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return True
    except Exception as e:                                  # noqa: BLE001
        print(f"[teach] could not write {path}: {e}")
        return False


def _read_json_tolerant(path: str) -> Dict:
    """Existing message, or {} if missing/corrupt/mid-write."""
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:                                  # noqa: BLE001
        print(f"[teach] ignoring unreadable {path}: {e}")
        return {}


def _update_red_to_blue(fields: Dict) -> bool:
    """Merge `fields` into the Red->Blue message, preserving the rest."""
    msg = _read_json_tolerant(RED_TO_BLUE)
    msg.setdefault("prompt_update", None)
    msg.setdefault("reason", "")
    msg.setdefault("confidence", 0.0)
    msg.setdefault("replay_skill", None)
    msg.update(fields)
    msg["ts"] = time.time()
    if _write_json_atomic(RED_TO_BLUE, msg):
        print(f"[teach] -> blue: replay_skill={msg.get('replay_skill')!r}")
        return True
    return False


def _write_skill(name: str, skill: Dict) -> bool:
    if not _write_json_atomic(_skill_path(name), skill):
        return False
    print(f"[teach] saved {_skill_path(name)} "
          f"({skill['waypoint_count']} waypoints from "
          f"{skill['raw_samples']} samples)")
    return True


# ---------------- standalone smoke test ----------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="record a demonstration")
    ap.add_argument("--name", default="smoke_demo")
    ap.add_argument("--seconds", type=int, default=8)
    ap.add_argument("--source", default="mock", choices=["mock", "leader"])
    ap.add_argument("--task", default="pick up the block")
    args = ap.parse_args()

    skill = record_demonstration(args.name, seconds=args.seconds,
                                 source=args.source, task=args.task)
    if not skill:
        sys.exit("record_demonstration returned nothing")

    # Did motion weighting actually happen, or did it degenerate into
    # uniform sampling? Measured from the data, with no assumption about
    # when the grasp occurred: waypoints spanning gripper motion must sit
    # CLOSER together in time than the rest. Equal medians would mean the
    # grasp is sampled no better than the idle approach, which is the one
    # thing this file exists to prevent.
    wps = skill["waypoints"]
    gaps = [(b["t"] - a["t"],
             abs(b["joints"]["gripper.pos"] - a["joints"]["gripper.pos"]) > 1.0)
            for a, b in zip(wps, wps[1:])]
    grip = sorted(g for g, moving in gaps if moving)
    idle = sorted(g for g, moving in gaps if not moving)

    def _median(xs):
        return xs[len(xs) // 2] if xs else float("nan")

    print(f"\nwaypoints: {skill['waypoint_count']} over "
          f"{skill['duration_s']}s ({skill['sample_hz']}Hz raw)")
    print(f"  spanning gripper motion : {len(grip):2d}  median gap "
          f"{_median(grip):.3f}s")
    print(f"  elsewhere               : {len(idle):2d}  median gap "
          f"{_median(idle):.3f}s")
    if not grip:
        sys.exit("FAIL: no waypoint captured the gripper moving")
    if _median(grip) >= _median(idle):
        sys.exit("FAIL: grasp sampled no denser than idle motion — "
                 "downsampling degenerated to uniform")

    print(f"\nskills on disk: {list_skills()}")
    print(f"load_skill round-trip: "
          f"{load_skill(args.name)['waypoint_count']} waypoints")
    print(f"load_skill(missing):   {load_skill('nope__')}")

    # Requesting replay must not clobber a directive Reasoning wrote.
    _update_red_to_blue({"prompt_update": "move 2cm left before grasping",
                         "reason": "previous attempt missed the grasp"})
    print(f"request_replay:        {request_replay(args.name)}")
    print(f"request_replay(bogus): {request_replay('nope__')}")
    print("\nred_to_blue.json now:")
    print(json.dumps(_read_json_tolerant(RED_TO_BLUE), indent=2))
    if _read_json_tolerant(RED_TO_BLUE).get("prompt_update") is None:
        sys.exit("FAIL: request_replay clobbered prompt_update")
