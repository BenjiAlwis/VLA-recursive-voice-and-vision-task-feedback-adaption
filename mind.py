"""
One command to run the whole Red side.  Owner: Aaryan (RED team).

    python3 mind.py                       # ask for the task out loud
    python3 mind.py --task "pick up the red block and place it in the zone"
    python3 mind.py --emit                # also write the Red->Blue message

Composes what already exists — nothing new architecturally, one process to
launch on stage instead of three terminals:

    voice.listen_for_task()        you speak the task into the glasses
    glasses_bridge (in-process)    receives frames the phone pushes
    supervise.Supervisor           examines each frame, corrects the arm
    narrate.speak                  the diagnosis comes back in your ears
    teach.record_demonstration     after N corrections that did not help,
                                   it stops guessing and asks you to show it

Every layer degrades: no glasses, no mic, no model, no Blue team — it still
runs and still tells you what it would have done.

=== --emit AND THE SINGLE-WRITER RULE ===

Benji's red loop owns shared/red_to_blue.json. By default this script does
NOT write it: corrections are drained from the queue and PRINTED, so he
merges them in his loop and there is exactly one writer.

--emit turns on a writer here for running the demo without his loop. Do not
use it while his loop is running — two writers on one file will clobber each
other's fields. It read-modify-writes and warns loudly at startup.
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, Optional

import glasses
import reason
import supervise

_OPTIONAL: Dict[str, Optional[object]] = {}
for _name in ("narrate", "voice", "teach", "memory", "glasses_bridge"):
    try:
        _OPTIONAL[_name] = __import__(_name)
    except Exception as _e:                                 # noqa: BLE001
        _OPTIONAL[_name] = None
        print(f"[mind] {_name} unavailable: {_e}")

narrate = _OPTIONAL["narrate"]
voice = _OPTIONAL["voice"]
teach = _OPTIONAL["teach"]
memory = _OPTIONAL["memory"]
glasses_bridge = _OPTIONAL["glasses_bridge"]

RED_TO_BLUE = os.path.join(os.getenv("SHARED_DIR", "shared"),
                           "red_to_blue.json")
DEFAULT_TASK = "pick up the red block and place it in the target zone"


def _say(text: str, block: bool = False) -> None:
    if narrate is not None:
        narrate.speak(text, block=block)
    else:
        print(f"  🗣  {text}")


# ---------------- audio routing ----------------

def route_audio(device: str) -> Dict[str, bool]:
    """Point speech and the mic at the glasses. Failure is not fatal —
    a missing headset must fall back to the laptop, never stop the run."""
    routed = {"out": False, "in": False}
    if not device:
        return routed
    try:
        if narrate is not None:
            routed["out"] = bool(narrate.set_output_device(device))
    except Exception as e:                                  # noqa: BLE001
        print(f"[mind] output routing failed: {e}")
    try:
        if voice is not None:
            routed["in"] = bool(voice.set_input_device(device))
    except Exception as e:                                  # noqa: BLE001
        print(f"[mind] input routing failed: {e}")
    return routed


# ---------------- the standalone Red->Blue writer ----------------

def emit(fields: Dict) -> bool:
    """STANDALONE ONLY. Merge `fields` into the Red->Blue message.

    Read-modify-write and atomic: the message carries prompt_update from
    reasoning AND replay_skill from a demonstration, and clobbering one
    while setting the other is the exact bug this guards against.
    """
    try:
        msg: Dict = {}
        try:
            with open(RED_TO_BLUE) as f:
                loaded = json.load(f)
            msg = loaded if isinstance(loaded, dict) else {}
        except FileNotFoundError:
            pass
        except Exception as e:                              # noqa: BLE001
            print(f"[mind] existing message unreadable, starting fresh: {e}")

        for key, default in (("prompt_update", None), ("reason", ""),
                             ("confidence", 0.0), ("replay_skill", None)):
            msg.setdefault(key, default)
        msg.update(fields)
        msg["ts"] = time.time()

        parent = os.path.dirname(RED_TO_BLUE) or "."
        os.makedirs(parent, exist_ok=True)
        tmp = f"{RED_TO_BLUE}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(msg, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, RED_TO_BLUE)
        return True
    except Exception as e:                                  # noqa: BLE001
        print(f"[mind] could not emit: {e}")
        return False


# ---------------- task capture ----------------

def ask_for_task(explicit: Optional[str], timeout_s: int = 8) -> str:
    """Spoken task if we can hear one, else the flag, else the default.

    Never blocks without a tty — voice.listen_for_task already guarantees
    that, which is what makes this safe to run unattended.
    """
    if explicit:
        return explicit
    if voice is not None:
        try:
            heard = voice.listen_for_task(timeout_s=timeout_s)
            if heard:
                return heard
        except Exception as e:                              # noqa: BLE001
            print(f"[mind] could not hear a task: {e}")
    print(f"[mind] using the default task: {DEFAULT_TASK!r}")
    return DEFAULT_TASK


# ---------------- the run ----------------

def run(task: Optional[str] = None, interval: float = 4.0,
        duration: Optional[float] = None, do_emit: bool = False,
        escalate_after: int = 3, teach_source: str = "leader",
        teach_seconds: int = 10, device: str = "Ray-Ban",
        start_bridge: bool = True, bridge_port: Optional[int] = None) -> Dict:
    """The whole Red loop in one call. Never raises."""
    routed = route_audio(device)
    print(f"[mind] audio out->glasses={routed['out']} "
          f"in->glasses={routed['in']}")

    srv = None
    if start_bridge and glasses_bridge is not None:
        try:
            srv = glasses_bridge.serve_in_background(
                bridge_port or glasses_bridge.PORT)
        except Exception as e:                              # noqa: BLE001
            print(f"[mind] could not start the frame receiver: {e}")

    task = ask_for_task(task)
    print(f"\n[mind] task: {task}")
    _say(f"Working on it. {task}")

    sup = supervise.Supervisor(task)
    state = {"since_help": 0, "demos": 0, "emitted": 0}

    def on_correction(supervisor, result) -> None:
        """Called once per accepted correction."""
        payload = supervisor.take_pending()
        if payload:
            if do_emit:
                if emit(payload):
                    state["emitted"] += 1
                    print(f"[mind] emitted -> {RED_TO_BLUE}")
            else:
                print(f"[mind] for Benji's writer: {json.dumps(payload)}")

        state["since_help"] += 1
        if escalate_after and state["since_help"] >= escalate_after:
            state["since_help"] = 0
            _escalate(task, teach_source, teach_seconds, do_emit, state)

    stats = sup.run(interval=interval, duration=duration,
                    on_correction=on_correction)

    if srv is not None:
        try:
            srv.shutdown()
        except Exception:                                   # noqa: BLE001
            pass

    stats.update(state)
    if memory is not None:
        try:
            stats["memory"] = memory.stats()
        except Exception:                                   # noqa: BLE001
            pass
    return stats


def _escalate(task: str, source: str, seconds: int, do_emit: bool,
              state: Dict) -> None:
    """Corrections are not landing. Stop guessing and ask to be shown.

    This is the beat that makes the system look like it is learning rather
    than retrying: it admits the loop is stuck and asks for a demonstration.
    """
    if teach is None:
        _say("I am still getting this wrong, and I cannot record a "
             "demonstration right now.")
        return

    _say("My corrections are not working. Please show me how to do this.")
    name = f"taught_{int(time.time())}"
    try:
        skill = teach.record_demonstration(name, seconds=seconds,
                                           source=source, task=task)
    except Exception as e:                                  # noqa: BLE001
        print(f"[mind] demonstration failed: {e}")
        return

    if not skill:
        print("[mind] no demonstration recorded")
        return

    state["demos"] += 1
    payload = teach.request_replay(name, reason="human demonstrated it")
    if not payload:
        return
    if do_emit:
        if emit(payload):
            print(f"[mind] emitted replay request for {name}")
    else:
        print(f"[mind] for Benji's writer: {json.dumps(payload)}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="glasses -> reasoning -> corrected prompt, in one process")
    ap.add_argument("--task", default=None,
                    help="skip the spoken prompt and use this task")
    ap.add_argument("--interval", type=float, default=4.0)
    ap.add_argument("--duration", type=float, default=None,
                    help="seconds to run; default forever")
    ap.add_argument("--emit", action="store_true",
                    help="ALSO write shared/red_to_blue.json. Standalone "
                         "only — never with Benji's red loop running.")
    ap.add_argument("--escalate-after", type=int, default=3,
                    help="corrections before asking for a demonstration; "
                         "0 disables")
    ap.add_argument("--teach-source", default="leader",
                    choices=["leader", "mock"])
    ap.add_argument("--teach-seconds", type=int, default=10)
    ap.add_argument("--device", default="Ray-Ban",
                    help="audio device substring; empty to leave defaults")
    ap.add_argument("--no-bridge", action="store_true",
                    help="do not start the frame receiver in this process")
    args = ap.parse_args()

    if args.emit:
        print("[mind] WARNING: --emit writes shared/red_to_blue.json. "
              "Turn it OFF if Benji's red loop is running, or you will "
              "clobber each other's fields.\n")

    stats = run(task=args.task, interval=args.interval,
                duration=args.duration, do_emit=args.emit,
                escalate_after=args.escalate_after,
                teach_source=args.teach_source,
                teach_seconds=args.teach_seconds, device=args.device,
                start_bridge=not args.no_bridge)
    print(f"\nstats: {json.dumps(stats, indent=2, default=str)}")


if __name__ == "__main__":
    if "--selftest" not in sys.argv:
        main()
        raise SystemExit(0)

    # ---------------- self test: no glasses, no model, no Blue ----------
    import shutil
    import tempfile

    tmp = os.path.join(tempfile.gettempdir(), "mind_smoke")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)

    glasses.GLASSES_DIR = tmp
    RED_TO_BLUE = os.path.join(tmp, "red_to_blue.json")
    supervise.MIN_EMIT_GAP_S = 0.0          # so the test is not gated on time
    supervise.BLUE_TO_RED = os.path.join(tmp, "blue_to_red.json")
    if memory is not None:
        memory.MEM_PATH = os.path.join(tmp, "failures.json")
        memory.wipe()
    if teach is not None:
        teach.SKILL_DIR = os.path.join(tmp, "skills")

    JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64 + b"\xff\xd9"
    counter = {"n": 0}

    def drop():
        counter["n"] += 1
        p = os.path.join(tmp, f"f{counter['n']}.jpg")
        with open(p, "wb") as f:
            f.write(JPEG)
        time.sleep(0.02)
        return p

    # ---- a directive already in the message must survive our writes ----
    with open(RED_TO_BLUE, "w") as f:
        json.dump({"prompt_update": "old directive", "reason": "earlier",
                   "confidence": 0.5, "replay_skill": "existing_skill"}, f)
    assert emit({"prompt_update": "new directive", "confidence": 0.9})
    got = json.load(open(RED_TO_BLUE))
    assert got["prompt_update"] == "new directive", got
    assert got["replay_skill"] == "existing_skill", \
        "emit must not clobber a queued demonstration"
    assert "ts" in got
    print("PASS  emit() merges without clobbering replay_skill")

    # ---- no tty and no --task: falls back, never blocks ----
    t0 = time.monotonic()
    task = ask_for_task(None, timeout_s=1)
    assert task == DEFAULT_TASK, task
    assert time.monotonic() - t0 < 10, "ask_for_task must not hang"
    assert ask_for_task("explicit task") == "explicit task"
    print("PASS  task capture falls back without a tty")

    # ---- corrections flow through to the message ----
    fired = {"n": 0}

    def fake_correction(task, frame, memories=None, surface="unknown",
                        verdict=None):
        fired["n"] += 1
        return {"needs_correction": True, "failure_mode": "wrong_position",
                "diagnosis": "The block is left of the zone.",
                "prompt_update": f"Move right, attempt {fired['n']}.",
                "confidence": 0.85, "source": "glasses"}

    reason.suggest_correction = fake_correction

    sup = supervise.Supervisor("pick up the red block")
    state = {"since_help": 0, "demos": 0, "emitted": 0}

    for _ in range(3):
        drop()
        result = sup.tick()
        assert result is not None
        payload = sup.take_pending()
        assert emit(payload)
        state["emitted"] += 1
    msg = json.load(open(RED_TO_BLUE))
    assert msg["prompt_update"] == "Move right, attempt 3."
    assert msg["replay_skill"] == "existing_skill", \
        "three corrections must still not disturb the queued skill"
    print(f"PASS  {state['emitted']} corrections emitted, "
          f"replay_skill intact")

    # ---- escalation records a demonstration and requests a replay ----
    if teach is not None:
        _escalate("pick up the red block", "mock", 1, True, state)
        assert state["demos"] == 1, state
        msg = json.load(open(RED_TO_BLUE))
        assert msg["replay_skill"] != "existing_skill", \
            "the demonstration should now be the requested skill"
        assert msg["prompt_update"] == "Move right, attempt 3.", \
            "requesting a replay must not wipe the correction"
        print(f"PASS  escalation taught {msg['replay_skill']!r} and kept "
              f"the prompt update")

    # ---- full run(), unattended, with the bridge disabled ----
    reason.suggest_correction = fake_correction
    stats = run(task="pick up the red block", interval=0.05, duration=0.6,
                do_emit=True, escalate_after=0, device="",
                start_bridge=False)
    print(f"\nstats: {json.dumps({k: v for k, v in stats.items() if k != 'memory'}, indent=2)}")
    assert stats["corrections"] >= 1, stats
    assert stats["emitted"] >= 1, stats
    print("PASS  run() completed unattended and emitted corrections")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\nall mind.py assertions passed")
