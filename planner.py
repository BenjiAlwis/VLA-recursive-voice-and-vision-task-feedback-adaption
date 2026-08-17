"""
The Nebius call.  Owner: Rikin.

Nebius AI Studio is OpenAI-compatible:
    pip install openai
    export NEBIUS_API_KEY=...
    base_url = https://api.studio.nebius.com/v1/

VERIFY THE MODEL ID FIRST — `python planner.py --verify`. master_reference
§12 flags this as a real risk: a wrong model string fails in a way that
looks exactly like an auth error and eats twenty minutes.

This is the Reasoning box in the diagram. It takes:
    wrist camera frame     scene context (never a verdict)
    the goal               spoken by the human, via glasses.py
    the skill list         schema.py, the fixed action menu
    past failures          memory.py retrieval
    voice corrections      the human talking mid-run
and returns a VALIDATED plan. It cannot return anything else: every exit
path from plan() goes through schema.validate_plan or the fixed fallback.
"""
import base64
import json
import os
from typing import Dict, List, Optional

import schema

MODEL = os.getenv("NEBIUS_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct")
VISION = os.getenv("NEBIUS_VISION", "1") == "1"
BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1/")
TIMEOUT_S = float(os.getenv("NEBIUS_TIMEOUT_S", "25"))

_client = None


def client():
    global _client
    if _client is None:
        from openai import OpenAI
        key = os.getenv("NEBIUS_API_KEY")
        if not key:
            raise RuntimeError("NEBIUS_API_KEY is not set — fix this FIRST")
        _client = OpenAI(base_url=BASE_URL, api_key=key, timeout=TIMEOUT_S)
    return _client


def strip_fences_safe(text: str) -> str:
    """Re-exported so critic.py need not import schema directly."""
    return schema.strip_fences(text)


SYSTEM = """You plan for a stationary SO-101 robot arm doing pick-and-place.
You output ONLY a JSON object. No prose, no markdown fences, nothing else.

THE ONLY ACTIONS THAT EXIST:
  {"action":"move_to","params":{"pose":<name>,"dx_cm":<float>,"dy_cm":<float>,"dz_cm":<float>}}
  {"action":"grip","params":{"state":"open"|"close","strength":<0-100>}}
  {"action":"replay_skill","params":{"name":<a skill a human taught you>}}

POSES: home, above_block, at_block, above_zone, at_zone.
Offsets are in centimetres, capped at +/-15, and default to 0.

Output format:
{
  "steps": [ <1 to 10 action objects> ],
  "rationale": "<one short sentence>",
  "new_skill": null
}

HOW THIS ARM FAILS, and what you can do about it:

* Its idea of where the block is carries a CONSTANT offset. If you closed
  the gripper and came away with nothing, you were not at the block: apply
  dx_cm / dy_cm on BOTH above_block and at_block to cancel that offset, and
  keep applying it on later attempts. The past-failure notes tell you the
  direction and size. A reported "gripper finished +4.2cm x" means you must
  move -4.2cm in x.
* ALWAYS lift before traversing. Going from at_block straight to at_zone
  drags the block across the table and counts as a collision. The safe
  order is: above_block, at_block, grip close, above_block, above_zone,
  at_zone, grip open.
* A grip strength below 45 can slip during the move. 80 is safe.
* If a human has taught you a skill, replay_skill is usually better than
  rebuilding the motion yourself.

If a sequence worked and is worth keeping, set new_skill to
{"name":"snake_case_name","composed_of":[<action objects>]}. Otherwise null.
"""


def build_user_msg(task: str, scene: Dict, memories: str, skills: str,
                   corrections: Optional[List[str]] = None) -> str:
    block = scene.get("block_cm", (0, 0))
    zone = scene.get("zone_cm", (0, 0))
    gripper = scene.get("gripper_cm", (0, 0, 0))
    dx = zone[0] - block[0]
    dy = zone[1] - block[1]

    msg = f"""TASK: {task}

MEASURED SCENE (overhead camera, centimetres, ground truth):
  block:   x={block[0]} y={block[1]}
  zone:    x={zone[0]} y={zone[1]}
  gripper: x={gripper[0]} y={gripper[1]} z={gripper[2] if len(gripper) > 2 else 0}
  block -> zone offset: dx={dx:.1f} dy={dy:.1f}
  holding the block: {scene.get('holding', False)}

SKILL LIBRARY:
{skills}

RELEVANT PAST FAILURES (yours, on this task):
{memories}"""

    if corrections:
        # The human's own words, verbatim. This is the "Voice Feedback
        # (Ray Bans) -> Reasoning" arrow in the diagram. Placed last
        # because it is the highest-authority signal in the prompt: a human
        # looking at the table knows things the cameras did not capture.
        msg += "\n\nA HUMAN JUST TOLD YOU (obey this over your own guess):\n"
        msg += "\n".join(f"  \"{c}\"" for c in corrections[-3:])

    return msg + "\n\nEmit the JSON plan."


def _encode(frame) -> Optional[str]:
    try:
        import cv2
        ok, buf = cv2.imencode(".jpg", frame)
        return base64.b64encode(buf).decode() if ok else None
    except Exception as e:                                  # noqa: BLE001
        print(f"[planner] could not encode frame: {e}")
        return None


def plan(task: str, scene: Dict, memories: str, frame=None,
         corrections: Optional[List[str]] = None, retries: int = 1) -> Dict:
    """Returns a VALIDATED plan dict. Never raises, never returns junk.

    On the retry, the validation error is fed back to the model. That is
    worth the extra call: "move_to.pose='grab_it' not in [...]" is a
    correctable mistake, and a blind resample at the same temperature tends
    to reproduce it.
    """
    user_msg = build_user_msg(task, scene, memories,
                              schema.skills_for_prompt(), corrections)
    content: List[Dict] = [{"type": "text", "text": user_msg}]

    if frame is not None and VISION:
        encoded = _encode(frame)
        if encoded:
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{encoded}"}})

    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": content}]

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client().chat.completions.create(
                model=MODEL, messages=messages,
                temperature=0.2, max_tokens=700,
            )
            raw = resp.choices[0].message.content
            try:
                return {**schema.validate_plan(raw), "source": "nebius"}
            except schema.ValidationError as ve:
                last_err = ve
                print(f"[planner] attempt {attempt} invalid: {ve}")
                messages += [
                    {"role": "assistant", "content": str(raw)[:1500]},
                    {"role": "user", "content":
                        f"That was rejected: {ve}. Emit ONLY corrected JSON."},
                ]
        except Exception as e:                              # noqa: BLE001
            last_err = e
            print(f"[planner] attempt {attempt} failed: {e}")

    print(f"[planner] FALLBACK after {last_err}")
    return {**fallback_plan(scene), "source": "fallback"}


def fallback_plan(scene: Dict = None, **kw) -> Dict:
    """The fixed pick-and-place. Lives in schema.py; re-exported here so
    loop.py has one import for both planning paths."""
    return schema.fallback_plan(scene, **kw)


# ---------------- the reflex arm ----------------

OFFSET_RE = r"dx_cm=([-+]?\d+\.?\d*)\s+dy_cm=([-+]?\d+\.?\d*)"


def reflex_plan(scene: Dict, memories: str) -> Dict:
    """Closed-loop correction with NO model at all.

    Reads the most recent stored correction out of the retrieved failures
    and applies it. This exists for two reasons, both of which matter more
    than it might look:

    1. It makes the whole demo rehearsable with no API key. Without it,
       every ablation arm silently collapses to the same fixed fallback
       when Nebius is unreachable, and the evidence chart shows three
       identical flat lines — which reads on stage as "nothing works".

    2. It is the HONEST baseline for the headline claim. "Our robot learns
       from failure" is much weaker if a regex over the same measurements
       learns just as fast. Reporting that comparison is the difference
       between a demo and a result. If the LLM arm does not beat this arm,
       say so on the slide — see the README.
    """
    import re

    matches = re.findall(OFFSET_RE, memories or "")
    if not matches:
        return {**schema.fallback_plan(scene),
                "rationale": "reflex: no correction stored yet"}
    dx, dy = (float(v) for v in matches[0])
    return {**schema.fallback_plan(scene, dx_cm=dx, dy_cm=dy),
            "rationale": f"reflex: applying stored correction "
                         f"dx={dx:+.1f} dy={dy:+.1f}"}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="check the model id resolves before anything else")
    args = ap.parse_args()

    if args.verify:
        # master_reference §12: do this before the first live planner call.
        print(f"base_url = {BASE_URL}")
        print(f"model    = {MODEL}")
        if not os.getenv("NEBIUS_API_KEY"):
            raise SystemExit("NEBIUS_API_KEY is not set")
        try:
            r = client().chat.completions.create(
                model=MODEL, max_tokens=10,
                messages=[{"role": "user", "content": "Reply with: ok"}])
            print(f"✅ model responded: {r.choices[0].message.content!r}")
        except Exception as e:                              # noqa: BLE001
            raise SystemExit(
                f"❌ {type(e).__name__}: {e}\n\n"
                "If this says 'model not found', the id is wrong — check the "
                "Nebius catalogue or ask a rep. It is NOT necessarily auth.")
        raise SystemExit(0)

    # ---- offline test: prompt assembly and the fallback path ----
    scene = {"block_cm": (18.0, 24.0), "zone_cm": (-16.0, 22.0),
             "gripper_cm": (22.2, 21.2, 1.0), "holding": False,
             "source": "mock"}

    print("=== assembled prompt ===")
    msg = build_user_msg(
        task="pick up the red block and place it in the target zone",
        scene=scene,
        memories="- tried ['move_to','grip'] -> missed_grasp, off by 34.1cm. "
                 "I closed the gripper 4.2cm right and 2.8cm short of the block.",
        skills=schema.skills_for_prompt(),
        corrections=["it's further left than you think"],
    )
    print(msg)
    assert "further left" in msg, "voice corrections must reach the prompt"
    assert "missed_grasp" in msg, "past failures must reach the prompt"

    print("\n=== no API key -> fixed fallback, never a crash ===")
    saved, os.environ["NEBIUS_API_KEY"] = os.getenv("NEBIUS_API_KEY", ""), ""
    _client = None
    p = plan("pick up the red block", scene, "(none)", retries=0)
    if saved:
        os.environ["NEBIUS_API_KEY"] = saved
    print(json.dumps(p["steps"], indent=1)[:400])
    assert p["rationale"] == "fixed pick-and-place (no model)"
    assert schema.validate_plan(p)["steps"] == p["steps"], \
        "the fallback must satisfy our own validator"

    print("\nplanner offline test passed — run --verify with a key before the demo")
