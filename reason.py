"""
Reasoning: why did it fail, and what should change next.  Owner: Rikin (RED).

Scaffolded by Aaryan — Rikin, this is yours to take over. The prompts are
the part worth your time; the validation and fallback below exist so a bad
model response can never reach the arm or stop the run.

    camera says FAILED (Benji)  ->  reason.diagnose(...)  ->  prompt_update
                                          |                       |
                                          v                       v
                                  narrate.announce_failure   Benji's writer
                                  memory.record                 -> Blue

=== THE TWO HARD RULES, AND HOW THEY ARE ENFORCED HERE ===

1. The model emits JSON only, validated against a fixed schema. Anything
   that fails validation is discarded and the heuristic fallback is used.
   Nothing from the model is ever executed.

2. THE CAMERA DECIDES PASS/FAIL. THE MODEL ONLY EXPLAINS WHY.
   This is enforced, not just documented:
     - diagnose() refuses to run at all on a passing verdict
     - any success-like key in the model's reply (passed, success, ok...)
       is STRIPPED and logged before validation
     - the returned dict never contains a pass/fail field
   A critic that hallucinates success stops the learning loop and makes the
   arm look broken on stage. There is deliberately no code path from a
   model's opinion to a pass.

Failure modes are imported from memory.py rather than redeclared, so the
diagnosis, the stored record and the evidence chart cannot drift apart.
"""
import base64
import json
import os
import time
from typing import Any, Dict, List, Optional

MODEL = os.getenv("NEBIUS_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct")
VISION = os.getenv("NEBIUS_VISION", "1") == "1"
NEBIUS_BASE = "https://api.studio.nebius.com/v1/"
RETRIES = int(os.getenv("REASON_RETRIES", "1"))

DISAGREE_LOG = "logs/reason_disagreements.jsonl"
MAX_DIAGNOSIS_CHARS = 240
MAX_PROMPT_CHARS = 240

# One taxonomy, defined in memory.py. Importing it means a mode this file
# invents can never end up as a category the ablation chart cannot plot.
try:
    from memory import FAILURE_MODES
except Exception:                                           # noqa: BLE001
    FAILURE_MODES = ["missed_grasp", "dropped_early", "wrong_position",
                     "collision", "no_motion"]

# Keys that would let the model claim the attempt succeeded. Stripped on
# sight — see hard rule 2.
_VERDICT_KEYS = {"passed", "pass", "success", "succeeded", "ok", "solved",
                 "complete", "completed", "achieved", "done", "result"}

# Below this much arm movement we call it no_motion rather than a miss.
NO_MOTION_CM = float(os.getenv("NO_MOTION_CM", "2.0"))

_client = None


def client():
    """Lazily built OpenAI-compatible client. Raises if the key is absent —
    diagnose() catches that and falls back."""
    global _client
    if _client is None:
        from openai import OpenAI
        key = os.getenv("NEBIUS_API_KEY")
        if not key:
            raise RuntimeError("NEBIUS_API_KEY not set")
        _client = OpenAI(base_url=NEBIUS_BASE, api_key=key)
    return _client


SYSTEM = f"""You diagnose a robot arm's FAILED attempt to pick up a block \
and place it in a target zone.

You are NOT deciding whether it succeeded. A camera has already measured \
that and it failed. Do not comment on success. Explain WHY it failed and \
what should change on the next attempt.

Output ONLY a JSON object, no prose and no markdown fences:
{{"failure_mode": one of {FAILURE_MODES},
  "diagnosis": "<one sentence, first person, spoken aloud by the robot>",
  "prompt_update": "<one imperative instruction for the controller>",
  "confidence": <float 0.0 to 1.0>}}

failure_mode meanings:
  missed_grasp    the gripper closed but the block was not in it
  dropped_early   it held the block and released before the target zone
  wrong_position  the block moved but ended outside the target zone
  collision       it struck the block, the zone or a fixture
  no_motion       nothing meaningfully moved

"diagnosis" is read aloud, so write it as the robot speaking:
  "I closed my gripper two centimetres short of the block."

"prompt_update" is fed to a controller that only understands natural \
language instructions. Make it a single concrete correction, not a \
restatement of the task. Good: "Move 2cm closer to the block before \
closing the gripper." Bad: "Pick up the block."

"confidence" is how sure you are of the diagnosis given the images."""


# ---------------- frames ----------------

def _encode_frame(frame: Any) -> Optional[str]:
    """Frame -> base64 data URL, or None.

    Accepts a path (from blue_to_red.json's "video_frame") or a decoded
    array. A path needs no OpenCV at all — the bytes are already a JPEG, so
    we read and encode them directly. Only an in-memory array needs cv2, and
    that import is lazy so a machine without OpenCV can still reason from
    paths and text.
    """
    if frame is None:
        return None
    try:
        if isinstance(frame, (str, bytes, os.PathLike)):
            path = os.fspath(frame)
            if not os.path.exists(path):
                print(f"[reason] frame not found: {path}")
                return None
            # Mime comes from an explicit map, never a default. Labelling an
            # unknown file "image/jpeg" is how a HEIC from an iPhone reaches
            # the endpoint and fails as an opaque API error instead of an
            # actionable "convert this" message.
            ext = os.path.splitext(path)[1].lower()
            mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".png": "image/png", ".webp": "image/webp"}.get(ext)
            if mime is None:
                print(f"[reason] refusing frame {os.path.basename(path)}: "
                      f"{ext or 'no extension'} is not a format the vision "
                      f"endpoint accepts (use JPEG or PNG)")
                return None
            with open(path, "rb") as f:
                return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"
        import cv2
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            return None
        return f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"
    except Exception as e:                                  # noqa: BLE001
        print(f"[reason] could not encode frame: {e}")
        return None


# ---------------- heuristic fallback ----------------

def heuristic_mode(verdict: Dict) -> str:
    """Best guess at the failure mode from geometry alone, no model.

    Used when the model is unavailable or returns junk, and as the
    comparison baseline for disagreement logging.

    Reads these keys from Benji's verdict when present. All are optional —
    the more he supplies, the sharper this gets:
        arm_moved_cm    how far the gripper travelled
        block_moved_cm  how far the block travelled
        grasped         bool, did the gripper end up holding the block
        in_zone         bool, did the block end inside the target zone
        collision       bool, from his geometry
    """
    if verdict.get("collision"):
        return "collision"

    arm_moved = verdict.get("arm_moved_cm", verdict.get("moved_cm"))
    if arm_moved is not None and float(arm_moved) < NO_MOTION_CM:
        return "no_motion"

    block_moved = verdict.get("block_moved_cm")
    if block_moved is not None and float(block_moved) < NO_MOTION_CM:
        # The arm moved but the block did not: it never had it.
        return "missed_grasp"

    if verdict.get("grasped") is False:
        return "missed_grasp"

    if verdict.get("grasped") and verdict.get("in_zone") is False:
        return "dropped_early"

    # It moved the block somewhere, just not the right somewhere.
    return "wrong_position"


def _fallback(task: str, verdict: Dict, why: str) -> Dict:
    """A valid result with no model involved. Never fails."""
    mode = heuristic_mode(verdict)
    error_cm = verdict.get("error_cm")
    where = (f" I ended up {float(error_cm):.0f} centimetres off."
             if isinstance(error_cm, (int, float)) else "")

    spoken = {
        "missed_grasp": "I closed my gripper without the block in it.",
        "dropped_early": "I let go of the block before reaching the zone.",
        "wrong_position": "I moved the block but not into the target zone.",
        "collision": "I hit something on the way.",
        "no_motion": "I did not move.",
    }[mode]

    fix = {
        "missed_grasp": "Move the gripper closer to the block and lower it "
                        "before closing.",
        "dropped_early": "Keep the gripper closed until the block is over "
                         "the target zone.",
        "wrong_position": "Carry the block further into the centre of the "
                          "target zone before releasing.",
        "collision": "Lift higher before moving across, and approach more "
                     "slowly.",
        "no_motion": "Begin by moving the gripper toward the block.",
    }[mode]

    return {
        "failure_mode": mode,
        "diagnosis": (spoken + where)[:MAX_DIAGNOSIS_CHARS],
        "prompt_update": fix[:MAX_PROMPT_CHARS],
        "confidence": 0.3,          # honest: this is geometry, not insight
        "source": "fallback",
        "fallback_reason": why,
    }


# ---------------- validation ----------------

class InvalidDiagnosis(Exception):
    pass


def _strip_verdict_claims(raw: Dict) -> List[str]:
    """Remove any key by which the model could claim success. Hard rule 2.

    Returns the names removed, so the attempt is visible in the logs rather
    than silently tolerated.
    """
    removed = []
    for key in list(raw.keys()):
        if key.strip().lower().replace(" ", "_") in _VERDICT_KEYS:
            raw.pop(key)
            removed.append(key)
    return removed


def _strip_fences(text: str) -> str:
    """Models wrap JSON in ```json fences roughly half the time."""
    t = text.strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) > 1:
            t = parts[1]
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    return t.strip()


def validate(raw: Any) -> Dict:
    """Coerce a model reply into the fixed schema, or raise.

    Deliberately strict about failure_mode and non-empty text, forgiving
    about confidence — a missing number is not worth discarding a good
    diagnosis over.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(_strip_fences(raw))
        except json.JSONDecodeError as e:
            raise InvalidDiagnosis(f"not valid JSON: {e}")
    if not isinstance(raw, dict):
        raise InvalidDiagnosis("reply is not a JSON object")

    raw = dict(raw)
    removed = _strip_verdict_claims(raw)
    if removed:
        print(f"[reason] model tried to return a verdict {removed}; "
              f"stripped — the camera decides pass/fail, not the model")

    mode = str(raw.get("failure_mode", "")).strip().lower()
    if mode not in FAILURE_MODES:
        raise InvalidDiagnosis(
            f"failure_mode {mode!r} not in {FAILURE_MODES}")

    diagnosis = str(raw.get("diagnosis", "")).strip()
    if not diagnosis:
        raise InvalidDiagnosis("empty diagnosis")

    prompt_update = str(raw.get("prompt_update", "")).strip()
    if not prompt_update:
        raise InvalidDiagnosis("empty prompt_update")
    if any(tok in prompt_update for tok in ("```", "def ", "import ",
                                            "exec(", "eval(")):
        raise InvalidDiagnosis("prompt_update looks like code, not an "
                               "instruction")

    try:
        confidence = min(1.0, max(0.0, float(raw.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5

    return {
        "failure_mode": mode,
        "diagnosis": diagnosis[:MAX_DIAGNOSIS_CHARS],
        "prompt_update": prompt_update[:MAX_PROMPT_CHARS],
        "confidence": confidence,
        "source": "vlm",
    }


# ---------------- the one function the loop calls ----------------

def diagnose(task: str, verdict: Dict, frame_before: Any = None,
             frame_after: Any = None, actions: Optional[List] = None,
             memories: Optional[str] = None,
             surface: str = "unknown") -> Optional[Dict]:
    """Explain a failed attempt and say what to change.

    `verdict` is Benji's camera measurement. It MUST carry passed and
    error_cm; the optional keys listed in heuristic_mode() sharpen the
    fallback.

    Returns {failure_mode, diagnosis, prompt_update, confidence, source} —
    with no pass/fail field, by design — or None if the verdict says the
    attempt passed. Never raises.
    """
    if verdict is None:
        print("[reason] no verdict supplied")
        return None
    if verdict.get("passed"):
        # Hard rule 2: there is nothing to diagnose, and asking a model
        # about a success invites it to volunteer an opinion on one.
        print("[reason] verdict passed — nothing to diagnose")
        return None

    if memories is None:
        try:
            import memory
            memories = memory.for_prompt(task, surface)
        except Exception:                                   # noqa: BLE001
            memories = "(memory unavailable)"

    user_text = f"""TASK: {task}

CAMERA MEASUREMENT (ground truth, already decided this FAILED):
{json.dumps(verdict, indent=2, default=str)}

ACTIONS ATTEMPTED: {actions if actions else "(not recorded)"}

RELEVANT PAST FAILURES:
{memories}

The images are before and after the attempt, in that order.
Diagnose the failure and give one correction."""

    content: List[Dict] = [{"type": "text", "text": user_text}]
    if VISION:
        for frame in (frame_before, frame_after):
            url = _encode_frame(frame)
            if url:
                content.append({"type": "image_url",
                                "image_url": {"url": url}})

    # Build the client BEFORE the retry loop. A missing package or key is
    # not a transient error, and retrying it just spends the demo's time
    # twice over before falling back to the same answer.
    try:
        api = client()
    except Exception as e:                                  # noqa: BLE001
        print(f"[reason] no model available ({e}); using geometry")
        return _fallback(task, verdict, why=f"no client: {e}")

    last_err = None
    for attempt in range(RETRIES + 1):
        try:
            resp = api.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": content}],
                temperature=0.3,
                max_tokens=300,
            )
            result = validate(resp.choices[0].message.content)
            _log_disagreement(result, verdict)
            return result
        except Exception as e:                              # noqa: BLE001
            last_err = e
            print(f"[reason] attempt {attempt} unusable: {e}")

    return _fallback(task, verdict, why=str(last_err))


def to_prompt_update(result: Optional[Dict]) -> Optional[Dict]:
    """The Red->Blue fields for Benji's writer to merge.

    Mirrors teach.request_replay's contract: we return, he writes. Carries
    no replay_skill, so merging this cannot disturb a queued demonstration.
    """
    if not result:
        return None
    return {
        "prompt_update": result["prompt_update"],
        "reason": result["diagnosis"],
        "confidence": result["confidence"],
    }


def _log_disagreement(result: Dict, verdict: Dict) -> None:
    """Where the model and the geometry disagree, record it.

    Every row is a labelling task: 30-50 of these, hand-labelled, turn the
    weakest component in the system into a measured accuracy number instead
    of a claim. Failing to log must never break a run.
    """
    heur = heuristic_mode(verdict)
    if result["failure_mode"] == heur:
        return
    try:
        os.makedirs(os.path.dirname(DISAGREE_LOG), exist_ok=True)
        with open(DISAGREE_LOG, "a") as f:
            f.write(json.dumps({
                "ts": time.time(),
                "vlm_mode": result["failure_mode"],
                "heuristic_mode": heur,
                "vlm_confidence": result["confidence"],
                "diagnosis": result["diagnosis"],
                "verdict": verdict,
            }, default=str) + "\n")
    except Exception as e:                                  # noqa: BLE001
        print(f"[reason] could not log disagreement: {e}")


if __name__ == "__main__":
    print(f"model={MODEL}  key_present={bool(os.getenv('NEBIUS_API_KEY'))}\n")

    # ---- hard rule 2: a pass is never diagnosed ----
    assert diagnose("t", {"passed": True, "error_cm": 1.0}) is None
    assert diagnose("t", None) is None
    print("PASS  a passing verdict is never sent to the model")

    # ---- hard rule 2: the model cannot claim success ----
    sneaky = json.dumps({"passed": True, "success": True,
                         "failure_mode": "missed_grasp",
                         "diagnosis": "Actually I did great.",
                         "prompt_update": "Do nothing.", "confidence": 0.9})
    got = validate(sneaky)
    assert "passed" not in got and "success" not in got, got
    print("PASS  success claims are stripped from the model reply")

    # ---- hard rule 1: junk is rejected, never executed ----
    bad = [
        ("not json at all", "prose instead of JSON"),
        ('{"failure_mode":"undershoot","diagnosis":"d",'
         '"prompt_update":"p"}', "old quadruped taxonomy"),
        ('{"failure_mode":"missed_grasp","diagnosis":"",'
         '"prompt_update":"p"}', "empty diagnosis"),
        ('{"failure_mode":"missed_grasp","diagnosis":"d",'
         '"prompt_update":""}', "empty prompt_update"),
        ('{"failure_mode":"missed_grasp","diagnosis":"d",'
         '"prompt_update":"import os; os.system(\'rm -rf /\')"}',
         "code in prompt_update"),
    ]
    for payload, label in bad:
        try:
            validate(payload)
        except InvalidDiagnosis:
            print(f"PASS  rejected: {label}")
        else:
            raise SystemExit(f"FAIL: accepted {label}")

    # ---- good replies survive, fences and all ----
    ok = validate('```json\n{"failure_mode":"dropped_early",'
                  '"diagnosis":"I let go too soon.",'
                  '"prompt_update":"Hold until over the zone.",'
                  '"confidence":1.7}\n```')
    assert ok["failure_mode"] == "dropped_early"
    assert ok["confidence"] == 1.0, "confidence must clamp to 0..1"
    print("PASS  fenced JSON parsed and confidence clamped")

    # ---- the fallback covers every mode, with no model at all ----
    print("\nfallback, by what the camera saw:")
    cases = [
        ("gripper never moved", {"passed": False, "error_cm": 30.0,
                                 "arm_moved_cm": 0.4}),
        ("arm moved, block did not", {"passed": False, "error_cm": 22.0,
                                      "arm_moved_cm": 18.0,
                                      "block_moved_cm": 0.2}),
        ("held it, released early", {"passed": False, "error_cm": 14.0,
                                     "arm_moved_cm": 25.0,
                                     "block_moved_cm": 12.0,
                                     "grasped": True, "in_zone": False}),
        ("placed, but off target", {"passed": False, "error_cm": 9.0,
                                    "arm_moved_cm": 30.0,
                                    "block_moved_cm": 25.0}),
        ("hit the fixture", {"passed": False, "error_cm": 5.0,
                             "collision": True}),
    ]
    seen = set()
    for label, verdict in cases:
        r = diagnose("pick up the red block and place it in the target zone",
                     verdict)
        assert r is not None and r["failure_mode"] in FAILURE_MODES
        assert r["source"] == "fallback"          # no key in this env
        seen.add(r["failure_mode"])
        print(f"  {label:<26} -> {r['failure_mode']:<14} "
              f"\"{r['prompt_update']}\"")
    assert seen == set(FAILURE_MODES), f"modes not covered: "\
        f"{set(FAILURE_MODES) - seen}"
    print(f"\nPASS  all {len(FAILURE_MODES)} failure modes reachable "
          f"without a model")

    # ---- the handoff to Benji's writer ----
    result = diagnose("pick up the red block",
                      {"passed": False, "error_cm": 12.0,
                       "arm_moved_cm": 20.0, "block_moved_cm": 0.1})
    msg = {"prompt_update": None, "reason": "", "confidence": 0.0,
           "replay_skill": "human_grasp"}
    msg.update(to_prompt_update(result))
    print("\nwhat Benji's writer would emit:")
    print(json.dumps(msg, indent=2))
    assert msg["replay_skill"] == "human_grasp", \
        "merging reasoning must not disturb a queued demonstration"
    assert to_prompt_update(None) is None

    # ---- what narrate would actually say ----
    try:
        import narrate
        narrate.announce_failure(result)
    except Exception as e:                                  # noqa: BLE001
        print(f"(narrate unavailable: {e})")

    print("\nall reason.py assertions passed")
