"""
Hybrid critic.  Owner: Rikin (with Benji).

THE ARCHITECTURAL COMMITMENT OF THIS PROJECT:

    geometry decides PASS/FAIL.        <- cannot hallucinate
    the VLM decides WHY it failed.     <- can hallucinate, doesn't matter

Do not add a code path where the model returns a boolean. A critic that
claims success on a failure stops the learning loop and the robot looks
broken on stage. This split is what makes the demo robust.

Side effect: every disagreement between the two is a data point.
Log them — they are the Toloka payload and they turn our weakest
component into a slide about rigour.
"""
import base64
import json
import math
import os
import time
from typing import Dict, Optional

import cv2

import planner

SUCCESS_CM = float(os.getenv("SUCCESS_CM", "12"))
DISAGREE_LOG = "logs/critic_disagreements.jsonl"

FAILURE_MODES = ["overshoot", "undershoot", "misalignment",
                 "wrong_skill", "physics_surprise", "no_motion"]


def measure_error(pose: Dict, target: Dict) -> float:
    return math.hypot(target["x_cm"] - pose["x_cm"],
                      target["y_cm"] - pose["y_cm"])


def geometric_verdict(pose_before: Dict, pose_after: Dict,
                      target: Dict) -> Dict:
    """Ground truth. No model involved."""
    err_before = measure_error(pose_before, target)
    err_after = measure_error(pose_after, target)
    moved = math.hypot(pose_after["x_cm"] - pose_before["x_cm"],
                       pose_after["y_cm"] - pose_before["y_cm"])
    return {
        "passed": err_after < SUCCESS_CM,
        "error_cm": round(err_after, 1),
        "improved": err_after < err_before,
        "moved_cm": round(moved, 1),
        "no_motion": moved < 3.0,
    }


def heuristic_mode(verdict: Dict, pose_after: Dict, target: Dict) -> str:
    """Cheap fallback so the loop never blocks on the LLM."""
    if verdict["no_motion"]:
        return "no_motion"
    err = verdict["error_cm"]
    d_target = math.hypot(target["x_cm"], target["y_cm"])
    d_robot = math.hypot(pose_after["x_cm"], pose_after["y_cm"])
    if d_robot > d_target + SUCCESS_CM:
        return "overshoot"
    if err > SUCCESS_CM:
        return "undershoot"
    return "misalignment"


DIAGNOSE_SYSTEM = f"""You diagnose a quadruped robot's failed attempt.
You are NOT deciding whether it succeeded — that is already measured.
You explain WHY it failed and what to change.

Output ONLY JSON:
{{"failure_mode": one of {FAILURE_MODES},
  "diagnosis": "<one sentence, first person, spoken aloud by the robot>",
  "suggested_fix": "<one short imperative>"}}

The diagnosis is read out loud, so write it as the robot speaking:
"I overshot the target by nine centimetres because the floor is slicker
than I estimated."
"""


def diagnose(verdict: Dict, pose_before: Dict, pose_after: Dict,
             target: Dict, plan: Dict,
             frame_before=None, frame_after=None) -> Dict:
    """VLM explains the failure. Falls back to heuristics if it errors."""
    fallback = {
        "failure_mode": heuristic_mode(verdict, pose_after, target),
        "diagnosis": f"I ended up {verdict['error_cm']} centimetres "
                     f"from the target.",
        "suggested_fix": "adjust distance estimate",
        "source": "heuristic",
    }

    content = [{"type": "text", "text": f"""Attempted: {json.dumps(plan['steps'])}
Before: {pose_before}
After:  {pose_after}
Target: {target}
Measured error: {verdict['error_cm']}cm. Moved {verdict['moved_cm']}cm.
Diagnose."""}]

    for frame in (frame_before, frame_after):
        if frame is not None:
            ok, buf = cv2.imencode(".jpg", frame)
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"}})

    try:
        resp = planner.client().chat.completions.create(
            model=planner.MODEL,
            messages=[{"role": "system", "content": DIAGNOSE_SYSTEM},
                      {"role": "user", "content": content}],
            temperature=0.3,
            max_tokens=250,
        )
        raw = json.loads(planner.strip_fences_safe(resp.choices[0].message.content))
        if raw.get("failure_mode") not in FAILURE_MODES:
            raw["failure_mode"] = fallback["failure_mode"]
        raw["source"] = "vlm"
        _log_if_disagree(raw, fallback, verdict)
        return raw
    except Exception as e:                          # noqa: BLE001
        print(f"[critic] diagnose fell back: {e}")
        return fallback


def _log_if_disagree(vlm: Dict, heur: Dict, verdict: Dict) -> None:
    """Toloka payload: cases where model and heuristic disagree."""
    if vlm["failure_mode"] == heur["failure_mode"]:
        return
    os.makedirs("logs", exist_ok=True)
    with open(DISAGREE_LOG, "a") as f:
        f.write(json.dumps({
            "ts": time.time(),
            "vlm_mode": vlm["failure_mode"],
            "heuristic_mode": heur["failure_mode"],
            "error_cm": verdict["error_cm"],
        }) + "\n")


def critique(pose_before: Dict, pose_after: Dict, target: Dict, plan: Dict,
             frame_before=None, frame_after=None) -> Dict:
    """The one function the loop calls."""
    verdict = geometric_verdict(pose_before, pose_after, target)
    if verdict["passed"]:
        return {**verdict, "failure_mode": None,
                "diagnosis": "I reached the target.", "suggested_fix": None}
    return {**verdict, **diagnose(verdict, pose_before, pose_after,
                                  target, plan, frame_before, frame_after)}
