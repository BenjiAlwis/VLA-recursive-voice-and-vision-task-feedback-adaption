"""
The glasses watch the arm and correct it.  Owner: Aaryan (RED team).

A person wears the Ray-Bans. The phone app samples the video stream every
few seconds and POSTs a JPEG to glasses_bridge. This loop picks up each new
frame, asks what looks wrong, speaks the answer, remembers it, and queues a
corrected prompt for the arm.

    python3 supervise.py --task "pick up the red block and place it in the
                                 target zone" --interval 4

    ┌── you, wearing the glasses ──┐
    │  phone app, timer @ interval │
    │  videoFramePublisher ──jpeg──┼──> glasses_bridge.py
    └──────────────────────────────┘          │
                                              v
             reason.suggest_correction  <──  new frame
                        │
          ┌─────────────┼──────────────┬──────────────┐
          v             v              v              v
     narrate.speak  memory.record  prompt_update   (nothing, if
     "I see the      (learns from   queued for      nothing is
      block is on     it)           Benji's writer  wrong)
      its side")                    -> Blue -> arm corrects

=== WHAT THIS LOOP MAY AND MAY NOT DO ===

It may only ever emit CORRECTIONS. It can never report success.

That is not a stylistic choice. The camera decides pass/fail; a model that
can say "that looks right, stop trying" can end a trial that actually
failed, and the arm then looks broken while everyone watches. So:

  - reason.suggest_correction() runs against an explicit "you are not a
    judge of success" instruction, and validate() strips any success-like
    key out of the reply
  - if Blue publishes a verdict saying the trial PASSED, this loop goes
    quiet and emits nothing — the fixed camera outranks the glasses
  - nothing here ever writes `passed`, and nothing here ever ends a trial

It also must not spam. A new prompt on every frame makes the controller
thrash between contradictory instructions, so identical consecutive
corrections are suppressed and there is a hard floor between emissions.
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, Optional

import glasses
import reason

try:
    import memory
except Exception as _e:                                     # noqa: BLE001
    memory = None
    print(f"[supervise] memory unavailable: {_e}")

try:
    import narrate

    def _say(text: str) -> None:
        narrate.speak(text)
except Exception as _e:                                     # noqa: BLE001
    print(f"[supervise] narrate unavailable ({_e}); printing instead")

    def _say(text: str) -> None:
        print(f"  🗣  {text}")


BLUE_TO_RED = os.path.join(os.getenv("SHARED_DIR", "shared"),
                           "blue_to_red.json")

# Never emit two corrections closer together than this, however fast frames
# arrive. The arm needs time to act on one instruction before being handed
# another, and a VLA fed contradictory prompts a second apart just jitters.
MIN_EMIT_GAP_S = float(os.getenv("SUPERVISE_MIN_GAP_S", "6"))

# A frame older than this is not evidence about the present.
MAX_FRAME_AGE_S = float(os.getenv("SUPERVISE_MAX_FRAME_AGE_S", "30"))


def read_blue() -> Dict:
    """Latest Blue->Red message, or {} if absent, corrupt or mid-write.

    Benji owns this file; we only read it, and a missing one is the normal
    case when Blue is not running.
    """
    try:
        with open(BLUE_TO_RED) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:                                  # noqa: BLE001
        print(f"[supervise] ignoring unreadable {BLUE_TO_RED}: {e}")
        return {}


class Supervisor:
    """Holds the state that keeps the loop from repeating itself."""

    def __init__(self, task: str, surface: str = "unknown",
                 min_confidence: float = 0.4):
        self.task = task
        self.surface = surface
        self.min_confidence = min_confidence
        self.pending: Optional[Dict] = None
        self._seen_frame: Optional[str] = None
        self._last_prompt: Optional[str] = None
        self._last_emit = 0.0
        self.stats = {"frames": 0, "corrections": 0, "suppressed": 0,
                      "quiet": 0, "skipped_passed": 0, "low_confidence": 0}

    # ---- the one method the loop calls ----

    def tick(self) -> Optional[Dict]:
        """Examine the newest glasses frame. Returns a correction or None."""
        blue = read_blue()

        # The fixed camera outranks the glasses. If it says this trial
        # passed, there is nothing to correct and we stay silent.
        verdict = blue.get("verdict") if isinstance(blue.get("verdict"), dict) else blue
        if verdict.get("passed"):
            self.stats["skipped_passed"] += 1
            return None

        frame = glasses.latest_frame(max_age_s=MAX_FRAME_AGE_S)
        if not frame or frame == self._seen_frame:
            return None                     # nothing new to look at
        self._seen_frame = frame
        self.stats["frames"] += 1

        result = reason.suggest_correction(
            self.task, frame, surface=self.surface,
            verdict=verdict if verdict.get("passed") is not None else None)

        if result is None:
            self.stats["quiet"] += 1
            return None

        if result["confidence"] < self.min_confidence:
            # A hesitant correction is worse than none: it moves the arm on
            # a guess and pollutes memory with a mode nobody observed.
            self.stats["low_confidence"] += 1
            print(f"[supervise] ignoring low-confidence "
                  f"({result['confidence']:.2f}) {result['failure_mode']}")
            return None

        now = time.monotonic()
        if result["prompt_update"] == self._last_prompt:
            self.stats["suppressed"] += 1
            print("[supervise] same correction as last time; not re-emitting")
            return None
        if now - self._last_emit < MIN_EMIT_GAP_S:
            self.stats["suppressed"] += 1
            print(f"[supervise] holding correction "
                  f"({MIN_EMIT_GAP_S - (now - self._last_emit):.1f}s to go)")
            return None

        # Commit: speak it, remember it, queue it.
        self._last_prompt = result["prompt_update"]
        self._last_emit = now
        self.stats["corrections"] += 1

        _say(result["diagnosis"])

        if memory is not None:
            try:
                memory.record(self.task, ["glasses_observation"],
                              error_cm=float(verdict.get("error_cm") or 0.0),
                              failure_mode=result["failure_mode"],
                              diagnosis=result["diagnosis"],
                              surface=self.surface)
            except Exception as e:                          # noqa: BLE001
                print(f"[supervise] could not record: {e}")

        self.pending = reason.to_prompt_update(result)
        print(f"[supervise] -> {result['prompt_update']}")
        return result

    # ---- handoff, same contract as teach.take_pending_replay ----

    def take_pending(self) -> Optional[Dict]:
        """Pop the queued prompt update for Benji's writer to emit.

        Consume-once. We never write shared/red_to_blue.json ourselves —
        one writer, no merge races.
        """
        payload, self.pending = self.pending, None
        return payload

    def run(self, interval: float = 4.0, duration: Optional[float] = None,
            on_correction=None) -> Dict:
        """Poll every `interval` seconds until `duration` elapses.

        Never raises. Ctrl-C exits cleanly with the stats.
        """
        started = time.monotonic()
        print(f"[supervise] watching every {interval}s "
              f"(min gap {MIN_EMIT_GAP_S}s, frames from "
              f"{os.path.abspath(glasses.GLASSES_DIR)})")
        try:
            while duration is None or time.monotonic() - started < duration:
                try:
                    result = self.tick()
                    if result and on_correction:
                        on_correction(self, result)
                except Exception as e:                      # noqa: BLE001
                    print(f"[supervise] tick failed, continuing: {e}")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n[supervise] stopped")
        return dict(self.stats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="watch the arm through the glasses and correct it")
    ap.add_argument("--task",
                    default="pick up the red block and place it in the "
                            "target zone")
    ap.add_argument("--interval", type=float, default=4.0)
    ap.add_argument("--duration", type=float, default=None,
                    help="seconds to run; default forever")
    ap.add_argument("--surface", default="unknown")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if not args.selftest:
        sup = Supervisor(args.task, surface=args.surface)
        print("Start glasses_bridge.py in another terminal, and point the "
              "phone app at it.\n")
        stats = sup.run(interval=args.interval, duration=args.duration)
        print(f"\nstats: {json.dumps(stats, indent=2)}")
        raise SystemExit(0)

    # ---------------- self test, no model and no glasses ----------------
    import shutil
    import tempfile

    tmp = os.path.join(tempfile.gettempdir(), "supervise_smoke")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    glasses.GLASSES_DIR = tmp
    BLUE_TO_RED = os.path.join(tmp, "blue_to_red.json")
    if memory is not None:
        memory.MEM_PATH = os.path.join(tmp, "failures.json")
        memory.wipe()

    JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64 + b"\xff\xd9"

    def drop(name="f.jpg"):
        p = os.path.join(tmp, name)
        with open(p, "wb") as f:
            f.write(JPEG)
        return p

    sup = Supervisor("pick up the red block")

    # ---- with no model, the loop must stay silent, not invent advice ----
    drop("frame1.jpg")
    assert sup.tick() is None, "no model must produce no correction"
    assert sup.stats["quiet"] == 1
    print("PASS  no model -> silent, never a fabricated correction")

    # ---- a frame is examined once, not on every tick ----
    assert sup.tick() is None
    assert sup.stats["frames"] == 1, sup.stats
    print("PASS  the same frame is not re-examined")

    # ---- now stub the model so the rest of the path is exercised ----
    calls = {"n": 0}

    def fake(task, frame, memories=None, surface="unknown", verdict=None):
        if verdict is not None and verdict.get("passed"):
            return None
        calls["n"] += 1
        return {"needs_correction": True, "failure_mode": "wrong_position",
                "diagnosis": "I can see the block is left of the zone.",
                "prompt_update": "Move the gripper 3 centimetres right.",
                "confidence": 0.8, "source": "glasses"}

    reason.suggest_correction = fake

    time.sleep(0.02)
    drop("frame2.jpg")
    r = sup.tick()
    assert r and r["failure_mode"] == "wrong_position", r
    payload = sup.take_pending()
    assert payload["prompt_update"] == "Move the gripper 3 centimetres right."
    assert "replay_skill" not in payload, \
        "a correction must not disturb a queued demonstration"
    assert sup.take_pending() is None, "consume-once failed"
    print(f"PASS  correction emitted and queued: {payload['prompt_update']}")

    # ---- it was recorded, so the arm learns from what the glasses saw ----
    if memory is not None:
        assert memory.stats()["total"] == 1, memory.stats()
        print("PASS  the observation was written to failure memory")

    # ---- identical advice is not re-emitted ----
    time.sleep(0.02)
    drop("frame3.jpg")
    assert sup.tick() is None
    assert sup.stats["suppressed"] == 1
    print("PASS  an identical correction is suppressed")

    # ---- and a different one still respects the rate floor ----
    def fake2(task, frame, memories=None, surface="unknown", verdict=None):
        return {"needs_correction": True, "failure_mode": "missed_grasp",
                "diagnosis": "The gripper closed early.",
                "prompt_update": "Lower the gripper before closing.",
                "confidence": 0.9, "source": "glasses"}

    reason.suggest_correction = fake2
    time.sleep(0.02)
    drop("frame4.jpg")
    assert sup.tick() is None, "must respect MIN_EMIT_GAP_S"
    assert sup.stats["suppressed"] == 2
    print(f"PASS  rate floor held ({MIN_EMIT_GAP_S}s between corrections)")

    # ---- the camera outranks the glasses ----
    with open(BLUE_TO_RED, "w") as f:
        json.dump({"passed": True, "error_cm": 1.0}, f)
    globals()["BLUE_TO_RED"] = BLUE_TO_RED
    before = calls["n"]
    time.sleep(0.02)
    drop("frame5.jpg")
    assert sup.tick() is None, "a passing verdict must silence the loop"
    assert sup.stats["skipped_passed"] >= 1
    assert calls["n"] == before, "the model must not even be consulted"
    print("PASS  a passing camera verdict silences the glasses entirely")

    # ---- low confidence is not acted on ----
    # Clear the passing verdict first, or the check above short-circuits
    # this one and it silently proves nothing.
    with open(BLUE_TO_RED, "w") as f:
        json.dump({"passed": False, "error_cm": 9.0}, f)
    sup2 = Supervisor("pick up the red block", min_confidence=0.6)
    reason.suggest_correction = lambda *a, **k: {
        "needs_correction": True, "failure_mode": "collision",
        "diagnosis": "Maybe it clipped something.",
        "prompt_update": "Lift higher.", "confidence": 0.2,
        "source": "glasses"}
    time.sleep(0.02)
    drop("frame6.jpg")
    assert sup2.tick() is None
    assert sup2.stats["low_confidence"] == 1
    print("PASS  a low-confidence guess is not acted on")

    # ---- run() survives a tick that explodes ----
    reason.suggest_correction = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("boom"))
    sup3 = Supervisor("t")
    time.sleep(0.02)
    drop("frame7.jpg")
    stats = sup3.run(interval=0.05, duration=0.2)
    print(f"PASS  run() survived a failing tick: {stats['frames']} frame(s)")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\nall supervise.py assertions passed")
