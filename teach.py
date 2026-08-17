"""
Leader-arm demonstration recorder.  Owner: Aaryan (RED team).

After two consecutive failures a human physically moves the LEADER arm
through the correct motion. We record that trajectory, compress it to the
waypoints that matter, and save it as a named skill.

=== BOUNDARY — READ THIS BEFORE EDITING ===

Red never commands a motor. This file only ever:
  - READS joint states (lerobot get_observation, or a fabricated mock)
  - WRITES JSON to logs/skills/ only

It does NOT mirror the leader to the follower, does NOT replay, does NOT
call send_action. Mirroring and replay belong to the BLUE team (a separate
team running the simulation and the arm); we only ask for them.

It also does NOT write shared/red_to_blue.json. Benji's red loop owns the
"emit prompt update" step and is the single writer of that file. Two
writers would each need merge logic to avoid clobbering the other's
prompt_update; one writer needs none. We queue a request and the harness
emits it.

An earlier version of this file drove the follower AND wrote the shared
file. Both were removed deliberately — if you are reintroducing a write
path to an arm or to red_to_blue.json here, you are crossing a boundary.

    import teach
    skill = teach.record_demonstration("pick_block", seconds=10)
    teach.request_replay("pick_block")     # queues, does not write

Benji, in the red loop:
    pending = teach.take_pending_replay()  # consume-once
    if pending:
        msg.update(pending)                # your writer emits it
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

# A replay request waiting for the harness to emit it. This file never
# writes shared/red_to_blue.json — Benji's red loop owns that write.
_pending_replay: Optional[Dict] = None

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
    """Scripted leader pose at normalised progress p in [0, 1].

    Human motion is eased, EXCEPT across the gripper transitions. A grasp
    is abrupt, and smoothstepping it ramps velocity from zero at both ends,
    which lowers peak speed and costs the snap waypoints in the
    motion-weighted downsampler. Measured over a 240-sample run: easing the
    snap gives a 1.88x density ratio against idle motion, linear gives
    2.43x. Easing it here was fighting the design intent.
    """
    p = max(0.0, min(1.0, p))
    for (p0, a), (p1, b) in zip(_KEYFRAMES, _KEYFRAMES[1:]):
        if p <= p1 or p1 == 1.0:
            span = p1 - p0
            raw = (p - p0) / span if span > 0 else 0.0
            snap = abs(b[5] - a[5]) > 1.0          # index 5 is the gripper
            f = raw if snap else _smooth(raw)
            return {k: a[i] + (b[i] - a[i]) * f + random.gauss(0, MOCK_NOISE)
                    for i, k in enumerate(KEYS)}
    return {k: _KEYFRAMES[-1][1][i] for i, k in enumerate(KEYS)}


# ---------------- leader source ----------------

def _open_leader():
    """Open the LEADER arm READ-ONLY and return something with
    get_observation().

    The leader is a lerobot TELEOPERATOR of type `so101_leader` — confirmed
    against Team Yellow's setup in so101_yellow/scripts/05_teleoperate.sh,
    which drives it as `--teleop.type=so101_leader`. The class name is still
    inferred, so the candidates below are tried in order; if none import this
    raises with a clear message rather than silently degrading to fabricated
    data. A mock trajectory recorded while a human is physically
    demonstrating looks like it worked, which is worse than an error.

    THE ID MATTERS AS MUCH AS THE PORT. lerobot loads calibration by `id`,
    and a mismatched one silently loads the wrong calibration or none at
    all — at which point every joint value is meaningless and the recorded
    demonstration is junk that looks fine. So the id is read from the SAME
    env vars Team Yellow calibrates with (SO101_LEADER_ID, e.g.
    "yellow_leader") rather than hardcoded.

    NOTE: lerobot needs Python >= 3.12 and the [feetech] extra — SO-101 uses
    Feetech STS3215 servos, not Dynamixel.

    Only get_observation() is ever called. No send_action, ever.
    """
    port = os.getenv("LEADER_PORT") or os.getenv("SO101_LEADER_PORT")
    if not port:
        raise RuntimeError(
            "no leader port. Set SO101_LEADER_PORT (or LEADER_PORT) — see "
            "so101_yellow/configs/leader.env, or run "
            "so101_yellow/scripts/01_find_ports.sh")

    arm_id = os.getenv("LEADER_ID") or os.getenv("SO101_LEADER_ID")
    if not arm_id:
        arm_id = "leader"
        print("[teach] WARNING: no SO101_LEADER_ID set, falling back to "
              f"{arm_id!r}. If the arm was calibrated under a different id "
              "lerobot will load the wrong calibration or none, and the "
              "recorded joint values will be meaningless. Check "
              "so101_yellow/configs/leader.env.")
    print(f"[teach] leader port={port} id={arm_id}")

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
            arm = cls(cfg_cls(port=port, id=arm_id))
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


def request_replay(name: str, reason: str = "",
                   confidence: Optional[float] = None) -> Optional[Dict]:
    """Register a request for Blue to replay a taught skill.

    Does NOT touch shared/red_to_blue.json. Benji's harness owns the emit
    step and is the single writer of that file; two writers would each need
    merge logic to avoid clobbering the other's prompt_update, and one
    writer needs none.

    Returns the fields to merge into the Red->Blue message, or None if the
    skill does not exist. The request is also queued for
    take_pending_replay() so a polling loop can pick it up without holding
    the return value.

    The payload carries ONLY replay_skill unless you explicitly pass reason
    or confidence. `reason` and `confidence` are shared fields that
    Reasoning also sets, and merging our defaults over them would silently
    replace Rikin's explanation of his own prompt_update with boilerplate
    about a replay.
    """
    global _pending_replay

    if load_skill(name) is None:
        print(f"[teach] refusing to request replay of unknown skill {name!r}")
        return None

    payload: Dict = {"replay_skill": name}
    if reason:
        payload["reason"] = reason
    if confidence is not None:
        payload["confidence"] = float(confidence)
    _pending_replay = dict(payload)
    print(f"[teach] replay requested: {name!r} (awaiting the harness)")
    return payload


def take_pending_replay() -> Optional[Dict]:
    """Pop the queued replay request, or None if there isn't one.

    For Benji: call this each tick of the red loop and merge the result into
    the Red->Blue message. Consume-once, so a skill is not replayed forever
    once it has been handed over.

        pending = teach.take_pending_replay()
        if pending:
            msg.update(pending)
    """
    global _pending_replay
    payload, _pending_replay = _pending_replay, None
    return payload


def peek_pending_replay() -> Optional[Dict]:
    """The queued request without consuming it. For logging and tests."""
    return dict(_pending_replay) if _pending_replay else None


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

    # Replay is queued for the harness, never written to the shared file.
    payload = request_replay(args.name)
    print(f"\nrequest_replay ->      {payload}")
    print(f"request_replay(bogus): {request_replay('nope__')}")

    assert payload and payload["replay_skill"] == args.name
    assert request_replay("nope__") is None, "unknown skill must not queue"
    assert peek_pending_replay() is not None, "request should be queued"

    # This is the integration Benji wires: pop, merge, emit. Consume-once,
    # so a skill is not replayed on every subsequent tick.
    msg = {"prompt_update": "move 2cm left before grasping",
           "reason": "previous attempt missed the grasp",
           "confidence": 0.6, "replay_skill": None}
    pending = take_pending_replay()
    if pending:
        msg.update(pending)
    print("\nwhat Benji's writer would emit:")
    print(json.dumps(msg, indent=2))

    assert msg["replay_skill"] == args.name
    assert msg["prompt_update"] == "move 2cm left before grasping", \
        "merging must not drop the directive Reasoning wrote"
    # Reasoning's own reason and confidence must survive the merge too.
    assert msg["reason"] == "previous attempt missed the grasp", \
        "merging must not overwrite Reasoning's reason"
    assert msg["confidence"] == 0.6, \
        "merging must not overwrite Reasoning's confidence"
    assert take_pending_replay() is None, "consume-once failed"

    # ...but an explicit reason IS allowed to be sent when the caller means it.
    assert request_replay(args.name, reason="human taught it",
                          confidence=0.9) == {
        "replay_skill": args.name, "reason": "human taught it",
        "confidence": 0.9}
    take_pending_replay()

    # The whole point of the refactor: this file writes nothing to shared/.
    assert not os.path.exists("shared/red_to_blue.json"), \
        "teach.py must not write the Red->Blue file — that is Benji's writer"
    print("\nteach smoke test passed (wrote nothing to shared/)")
