"""
Learning from demonstration.  Owner: Aaryan.

After two consecutive failures the arm stops guessing and asks a human.
The human physically moves the LEADER arm through the correct motion;
we record it, compress it to a handful of waypoints, and save it as a
named skill the arm can replay on demand.

    import teach
    teach.record_demonstration("pick_block", seconds=10)
    teach.replay_skill("pick_block")
    teach.list_skills()

Benji: I expose the three functions above and never touch loop.py.
Wire them in at the escalation branch — where narrate.request_help()
is called today, call record_demonstration() instead when
ESCALATE_BACKEND=teach.

BENJI — TWO CHANGES NEEDED IN YOUR FILES TO MAKE REPLAY REACHABLE

record_demonstration() already registers the taught skill via
schema.add_skill(), so its name shows up in skills_for_prompt() and the
planner can see it exists. But the planner may only emit PRIMITIVES, and
there is no primitive that replays one. Two edits close that:

  schema.py — add the primitive:
      "replay_skill": {"name": (str, None, None)}
    NOTE: validate_step unpacks every spec as (type, lo, hi) and then
    does `lo <= val <= hi`, which blows up on a str param. It needs a
    skip when lo/hi are None. This is the only reason it is your change
    and not mine.

  loop.py — add the execute() branch:
      elif action == "replay_skill":
          teach.replay_skill(params["name"])

Until both land, demonstrations record and save correctly but nothing
ever plays them back.

TWO MORE NOTES FOR THE INTEGRATOR
  1. This needs the leader AND the follower at the same time, which the
     original single-arm contract couldn't express. Agreed extension:
     arm_api.get_arm(role="leader"|"follower"), role defaulting to
     "follower" so every existing call site keeps working untouched.
  2. Demos are written to logs/demos/{name}.json, NOT logs/skills/. The
     latter collides with schema.SKILLS_PATH ("logs/skills.json") — a
     path cannot be both a file and a directory.

UNVERIFIED: the LeRobot module paths in the brief could not be checked
against an installed version (lerobot is not installed in this tree).
Nothing here imports lerobot directly — it all goes through arm_api — so
that risk lands in Benji's backend, not in this file.
"""
import glob
import json
import math
import os
import time
from typing import Dict, List, Optional

try:                                    # the real contract, once it exists
    from arm_api import get_arm
except ImportError:                     # DEV ONLY — delete with arm_api_stub.py
    from arm_api_stub import get_arm

try:
    import narrate

    def _say(text: str, block: bool = True) -> None:
        narrate.speak(text, block=block)
except Exception as _e:                                     # noqa: BLE001
    print(f"[teach] narrate unavailable ({_e}); speech falls back to print")

    def _say(text: str, block: bool = True) -> None:
        print(f"  🗣  {text}")


DEMO_DIR = "logs/demos"
HZ = 30.0
WAYPOINTS = 20

# Never command a joint further than this in one message. Mirrors
# SOFollowerRobotConfig(max_relative_target=10.0) — the config clamps
# silently, so we interpolate to stay under it rather than let motion
# get quietly truncated.
MAX_REL_TARGET = float(os.getenv("ARM_MAX_REL_TARGET", "10.0"))

# SO-101 ".pos" values are normalised roughly to +/-100. Waypoint
# selection scores motion against this FIXED scale rather than each
# joint's observed range: range-normalising amplifies sensor noise on
# joints that barely moved, which is precisely the wrong thing to weight.
JOINT_SCALE = 100.0

# Where to leave the arm on every exit path.
#
# The default is deliberately NOT a hardcoded joint vector. On an
# uncalibrated SO-101 all-zeros is not a rest pose — it is roughly fully
# extended — so gliding there would sweep straight through the workspace
# where the block and the human's hands are. Holding the current measured
# position is the only pose that is safe without knowing the calibration.
#
# The gripper is left ALONE for the same reason: forcing it open at the
# end of record_demonstration would drop the block the arm was just
# taught to hold.
#
# After calibration, set a real rest pose to park in via either:
#   ARM_SAFE_POSE='{"shoulder_pan.pos": 0.0, ...}'   (env, JSON)
#   logs/safe_pose.json                              (file, JSON)
SAFE_POSE_FILE = "logs/safe_pose.json"


def _safe_pose_override() -> Optional[Dict[str, float]]:
    """A calibrated rest pose, if anyone has supplied one. Else None."""
    raw = os.getenv("ARM_SAFE_POSE")
    try:
        if raw:
            return json.loads(raw)
        if os.path.exists(SAFE_POSE_FILE):
            with open(SAFE_POSE_FILE) as f:
                return json.load(f)
    except Exception as e:                                  # noqa: BLE001
        print(f"[teach] ignoring unreadable safe pose: {e}")
    return None

# Populated by replay_skill so callers (and the smoke test) can assert on
# what actually went over the wire.
LAST_REPLAY_STATS: Dict[str, float] = {}

# Total successful set_joints calls, counted in _send so that approach
# moves are included too — not just the waypoint-to-waypoint steps.
_SENT = 0


# ---------------- transport ----------------

def _send(arm, joints: Dict[str, float]) -> bool:
    """Every command to the follower goes through here.

    The follower drops its connection mid-teleop; that is a known defect,
    not an exceptional case. One reconnect, one retry, then give up
    quietly — a dropped frame at 30Hz is invisible, a crash is not.
    """
    global _SENT
    try:
        arm.set_joints(joints)
        _SENT += 1
        return True
    except Exception as first:                              # noqa: BLE001
        try:
            reconnect = getattr(arm, "reconnect", None)
            if reconnect is not None:
                reconnect()
            else:
                arm.connect()
            arm.set_joints(joints)
            _SENT += 1
            return True
        except Exception as second:                         # noqa: BLE001
            print(f"[teach] send failed after reconnect: {first} / {second}")
            return False


def _read(arm) -> Optional[Dict[str, float]]:
    try:
        return arm.get_joints()
    except Exception as e:                                  # noqa: BLE001
        print(f"[teach] read failed: {e}")
        return None


def _clamp_toward(current: Dict[str, float], goal: Dict[str, float],
                  limit: float = MAX_REL_TARGET) -> Dict[str, float]:
    """Goal, with each joint pulled back to within `limit` of current."""
    out = {}
    for k, target in goal.items():
        now = current.get(k, target)
        out[k] = now + max(-limit, min(limit, target - now))
    return out


def _steps_needed(a: Dict[str, float], b: Dict[str, float]) -> int:
    """Interpolation steps so no single message exceeds MAX_REL_TARGET.

    Derived from the largest per-joint jump, not a fixed count: a fixed
    5-step interpolation between two distant waypoints still violates the
    limit, which is the bug this function exists to prevent.

    VERIFY AT BRING-UP: this guarantees a bound on the delta between
    consecutive COMMANDED frames. Check what LeRobot's
    ensure_safe_goal_position actually clamps against — if it clamps
    against the present MEASURED position, a servo lagging behind the
    command stream can sit >MAX_REL_TARGET from the next command even
    when consecutive commands are well inside it, and LeRobot will clamp
    and silently under-travel. If so, interpolate from _read(follower)
    each step instead of from the previous commanded frame. The mock
    cannot catch this — it applies commands instantly with no lag.
    """
    biggest = max((abs(b[k] - a.get(k, b[k])) for k in b), default=0.0)
    return max(1, math.ceil(biggest / MAX_REL_TARGET))


def _glide(arm, goal: Dict[str, float], duration: float = 1.0) -> float:
    """Move from wherever the arm is to `goal`, smoothly and within limit.
    Returns the largest per-message joint delta actually sent."""
    start = _read(arm) or dict(goal)
    steps = _steps_needed(start, goal)
    dt = duration / steps if duration > 0 else 0.0
    worst = 0.0
    prev = start
    for i in range(1, steps + 1):
        f = i / steps
        frame = {k: start.get(k, v) + (v - start.get(k, v)) * f
                 for k, v in goal.items()}
        worst = max(worst, max((abs(frame[k] - prev.get(k, frame[k]))
                                for k in frame), default=0.0))
        _send(arm, frame)
        prev = frame
        if dt:
            time.sleep(dt)
    return worst


def _park(arm) -> None:
    """Leave the arm safe. Called from `finally`, so it must not raise."""
    try:
        override = _safe_pose_override()
        if override:
            _glide(arm, override, duration=1.2)
            return
        # No calibrated rest pose known: hold exactly where we are. Commanding
        # the measured position stops motion without inventing coordinates.
        here = _read(arm)
        if here:
            _send(arm, here)
    except Exception as e:                                  # noqa: BLE001
        print(f"[teach] could not reach safe pose: {e}")


def _release(arm) -> None:
    """Close the serial connection. Leaving the port open means the NEXT
    run cannot connect, which on demo day looks like broken hardware."""
    try:
        for name in ("disconnect", "close", "__exit__"):
            fn = getattr(arm, name, None)
            if fn is None:
                continue
            fn(None, None, None) if name == "__exit__" else fn()
            return
    except Exception as e:                                  # noqa: BLE001
        print(f"[teach] could not release arm cleanly: {e}")


# ---------------- waypoint selection ----------------

def _motion(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Joint-space distance between two samples, on the fixed scale."""
    return sum(abs(b[k] - a.get(k, b[k])) for k in b) / JOINT_SCALE


def downsample(samples: List[Dict], n: int = WAYPOINTS) -> List[Dict]:
    """Compress a 30Hz recording to ~n waypoints.

    Resamples uniformly in CUMULATIVE MOTION rather than in time. Where
    the arm moves fast, waypoints land close together in time; where it
    dwells, they thin out. That is deliberate — uniform time sampling
    spends most of its budget on the slow approach and can miss the grasp
    entirely, and the grasp is the only part of the demonstration that
    has to be right.
    """
    if len(samples) <= n:
        return list(samples)

    cum = [0.0]
    for a, b in zip(samples, samples[1:]):
        cum.append(cum[-1] + _motion(a["joints"], b["joints"]))
    total = cum[-1]

    if total <= 1e-9:                   # arm never moved; nothing to weight on
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

def record_demonstration(name: str, seconds: int = 10,
                         task: str = "unknown") -> Optional[Dict]:
    """Human moves the leader; the follower mirrors; we keep the shape.

    Returns the saved skill dict, or None if it could not be recorded.
    Never raises.
    """
    leader = follower = None
    try:
        leader = get_arm(role="leader")
        follower = get_arm(role="follower")

        _say("Show me how to do this.")

        # Bring the follower to the leader's pose before mirroring starts,
        # so the first tracked command isn't a jump across the workspace.
        start = _read(leader)
        if start:
            _glide(follower, start, duration=1.5)

        for count in ("Three", "Two", "One"):
            _say(count)
            time.sleep(0.7)
        _say("Go.")

        samples: List[Dict] = []
        period = 1.0 / HZ
        t0 = time.monotonic()
        next_t = t0
        mirrored = dict(start) if start else {}

        while True:
            now = time.monotonic()
            elapsed = now - t0
            if elapsed >= seconds:
                break

            joints = _read(leader)
            if joints:
                samples.append({"t": round(elapsed, 4), "joints": joints})
                # Mirror so the human sees the follower copying them.
                mirrored = _clamp_toward(mirrored or joints, joints)
                _send(follower, mirrored)

            # Deadline-based, not sleep(period): two serial round-trips per
            # tick would otherwise let the clock drift and make the recorded
            # duration a lie.
            next_t += period
            time.sleep(max(0.0, next_t - time.monotonic()))

        if len(samples) < 2:
            _say("I did not see anything. Let us try that again.")
            return None

        waypoints = downsample(samples)
        skill = {
            "name": name,
            "type": "demonstration",
            "taught_for": task,
            "ts": time.time(),
            "duration_s": round(samples[-1]["t"], 3),
            "raw_samples": len(samples),
            "sample_hz": round(len(samples) / max(samples[-1]["t"], 1e-6), 1),
            "waypoint_count": len(waypoints),
            "waypoints": waypoints,
        }
        _write_demo(name, skill)
        _register(name, task)
        _say("Got it. Let me try that.")
        return skill

    except Exception as e:                                  # noqa: BLE001
        print(f"[teach] record_demonstration failed: {e}")
        return None
    finally:
        if follower is not None:
            _park(follower)
            _release(follower)
        if leader is not None:
            _release(leader)


def replay_skill(name: str, speed: float = 1.0) -> bool:
    """Replay a taught demonstration on the follower. Never raises."""
    LAST_REPLAY_STATS.clear()
    skill = load_skill(name)
    if not skill:
        print(f"[teach] no such skill: {name}")
        return False

    waypoints = skill.get("waypoints") or []
    if len(waypoints) < 2:
        print(f"[teach] skill '{name}' has too few waypoints")
        return False

    follower = None
    worst = 0.0
    sent_before = _SENT
    try:
        follower = get_arm(role="follower")
        speed = max(0.1, float(speed))

        # Approach the first waypoint from wherever we happen to be.
        worst = max(worst, _glide(follower, waypoints[0]["joints"],
                                  duration=1.5 / speed))

        for prev, wp in zip(waypoints, waypoints[1:]):
            a, b = prev["joints"], wp["joints"]
            steps = _steps_needed(a, b)
            dt = max(0.0, (wp["t"] - prev["t"]) / speed) / steps
            for i in range(1, steps + 1):
                f = i / steps
                frame = {k: a.get(k, v) + (v - a.get(k, v)) * f
                         for k, v in b.items()}
                ref = a if i == 1 else last
                worst = max(worst, max(abs(frame[k] - ref.get(k, frame[k]))
                                       for k in frame))
                _send(follower, frame)
                last = frame
                if dt:
                    time.sleep(dt)

        LAST_REPLAY_STATS.update(
            {"max_step_delta": round(worst, 4),
             "messages": _SENT - sent_before,
             "limit": MAX_REL_TARGET})
        return True

    except Exception as e:                                  # noqa: BLE001
        print(f"[teach] replay_skill failed: {e}")
        return False
    finally:
        if follower is not None:
            _park(follower)
            _release(follower)


def list_skills() -> List[str]:
    """Names of every taught demonstration on disk."""
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(f"{DEMO_DIR}/*.json"))


# ---------------- persistence ----------------

def _demo_path(name: str) -> str:
    safe = "".join(c for c in name if c.isalnum() or c in "._-") or "unnamed"
    return f"{DEMO_DIR}/{safe}.json"


def _register(name: str, task: str) -> bool:
    """Announce the taught skill to Benji's skill library.

    Without this the demo is unreachable: the planner only ever sees
    schema.skills_for_prompt(), which reads logs/skills.json, so a file
    sitting in logs/demos/ might as well not exist. Registering an entry
    with composed_of=[] validates trivially and puts the name in front of
    the model. Uses schema's public API — schema.py itself is untouched.

    Best-effort by design: a skill library hiccup must not lose a
    demonstration we already have safely on disk.
    """
    try:
        import schema
        return schema.add_skill({
            "name": name,
            "params": {"name": name},
            "composed_of": [],
            "learned_from": f"demonstration/{task}",
        })
    except Exception as e:                                  # noqa: BLE001
        print(f"[teach] could not register '{name}' in the skill library: {e}")
        return False


def _write_demo(name: str, skill: Dict) -> None:
    os.makedirs(DEMO_DIR, exist_ok=True)
    with open(_demo_path(name), "w") as f:
        json.dump(skill, f, indent=2)
    print(f"[teach] saved {_demo_path(name)} "
          f"({skill['waypoint_count']} waypoints from {skill['raw_samples']} samples)")


def load_skill(name: str) -> Optional[Dict]:
    path = _demo_path(name)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:                                  # noqa: BLE001
        print(f"[teach] could not read {path}: {e}")
        return None


# ---------------- standalone smoke test ----------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="record a demo, then replay it")
    ap.add_argument("--name", default="smoke_demo")
    ap.add_argument("--seconds", type=int, default=6)
    ap.add_argument("--speed", type=float, default=2.0)
    args = ap.parse_args()

    skill = record_demonstration(args.name, seconds=args.seconds,
                                 task="smoke test")
    if not skill:
        raise SystemExit("record_demonstration returned nothing")

    # Did velocity weighting actually happen, or did it degenerate into
    # uniform sampling? Measured from the data, not from any assumption
    # about when the grasp occurred: waypoints spanning gripper motion
    # should sit CLOSER together in time than the rest. If the two median
    # gaps match, the grasp is being sampled no better than the idle sweep
    # and this whole function is pointless.
    wps = skill["waypoints"]
    gaps = [(b["t"] - a["t"],
             abs(b["joints"]["gripper.pos"] - a["joints"]["gripper.pos"]) > 1.0)
            for a, b in zip(wps, wps[1:])]
    grasp = sorted(g for g, is_grasp in gaps if is_grasp)
    idle = sorted(g for g, is_grasp in gaps if not is_grasp)

    def _median(xs):
        return xs[len(xs) // 2] if xs else float("nan")

    print(f"\nwaypoints: {skill['waypoint_count']} over "
          f"{skill['duration_s']}s ({skill['sample_hz']}Hz raw)")
    print(f"  spanning the grasp : {len(grasp):2d}  median gap "
          f"{_median(grasp):.3f}s")
    print(f"  elsewhere          : {len(idle):2d}  median gap "
          f"{_median(idle):.3f}s")
    if not grasp:
        raise SystemExit("FAIL: no waypoint captured the gripper moving")
    if _median(grasp) >= _median(idle):
        raise SystemExit("FAIL: grasp sampled no denser than idle motion "
                         "— downsampling degenerated to uniform")

    print(f"\nskills on disk: {list_skills()}")

    ok = replay_skill(args.name, speed=args.speed)
    print(f"replay ok={ok}  stats={LAST_REPLAY_STATS}")
    if ok and LAST_REPLAY_STATS["max_step_delta"] > MAX_REL_TARGET + 1e-6:
        raise SystemExit("FAIL: a replay step exceeded max_relative_target")
