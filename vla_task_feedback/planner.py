"""
Nebius planner.  Owner: Rikin.

Nebius AI Studio is OpenAI-compatible:
    pip install openai
    export NEBIUS_API_KEY=...
    base_url = https://api.studio.nebius.com/v1/

VERIFY THE MODEL ID FIRST. Run `python planner.py` standalone before
Benji wires this in. Model catalogues change; don't assume.
"""
import base64
import json
import os
from typing import Optional, Dict, List

import cv2
from openai import OpenAI

import schema

MODEL = os.getenv("NEBIUS_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct")
VISION = os.getenv("NEBIUS_VISION", "1") == "1"

_client = None


def client() -> OpenAI:
    global _client
    if _client is None:
        key = os.getenv("NEBIUS_API_KEY")
        if not key:
            raise RuntimeError("NEBIUS_API_KEY not set — fix this FIRST")
        _client = OpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=key,
        )
    return _client


SYSTEM = """You control a quadruped robot. You output ONLY a JSON object. \
No prose, no markdown fences, no explanation outside the JSON.

Available actions (the ONLY actions that exist):
  {"action":"forward","params":{"meters":<float -2.0..2.0>}}
  {"action":"turn","params":{"degrees":<float -180..180>}}   // + = left
  {"action":"stop","params":{}}

Output format:
{
  "steps": [ <1 to 6 action objects> ],
  "rationale": "<one short sentence>",
  "new_skill": null
}

If a sequence you just used worked well and is worth reusing, you may set
new_skill to {"name":"snake_case_name","composed_of":[<action objects>]}.
Otherwise leave it null.

Key fact: the robot UNDER-shoots. Commanded distance is not actual distance.
Past-failure notes tell you by how much. Compensate."""


def strip_fences_safe(text: str) -> str:
    """Re-exported for critic.py so it doesn't import schema directly."""
    return schema.strip_fences(text)


def _encode(frame) -> str:
    ok, buf = cv2.imencode(".jpg", frame)
    return base64.b64encode(buf).decode()


def build_user_msg(task: str, pose: Dict, target: Dict,
                   memories: str, skills: str) -> str:
    dx = target["x_cm"] - pose["x_cm"]
    dy = target["y_cm"] - pose["y_cm"]
    return f"""TASK: {task}

CURRENT POSE: x={pose['x_cm']}cm y={pose['y_cm']}cm heading={pose['heading_deg']}deg
TARGET:       x={target['x_cm']}cm y={target['y_cm']}cm
OFFSET:       dx={dx:.1f}cm dy={dy:.1f}cm

SKILL LIBRARY:
{skills}

RELEVANT PAST FAILURES:
{memories}

Emit the JSON plan."""


def plan(task: str, pose: Dict, target: Dict, memories: str,
         frame=None, retries: int = 1) -> Dict:
    """Returns a VALIDATED plan dict. Falls back to a safe nudge on failure."""
    content = [{"type": "text",
                "text": build_user_msg(task, pose, target, memories,
                                       schema.skills_for_prompt())}]
    if frame is not None and VISION:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{_encode(frame)}"}
        })

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client().chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": content}],
                temperature=0.2,
                max_tokens=500,
            )
            return schema.validate_plan(resp.choices[0].message.content)
        except Exception as e:                      # noqa: BLE001
            last_err = e
            print(f"[planner] attempt {attempt} failed: {e}")

    # NEVER crash the demo. Degrade to a conservative closed-form nudge.
    print(f"[planner] FALLBACK after {last_err}")
    return fallback_plan(pose, target)


def fallback_plan(pose: Dict, target: Dict) -> Dict:
    """Deterministic geometry. Also serves as the no-LLM baseline
    for the ablation chart."""
    import math
    dx = target["x_cm"] - pose["x_cm"]
    dy = target["y_cm"] - pose["y_cm"]
    dist_m = math.hypot(dx, dy) / 100.0
    want = math.degrees(math.atan2(dy, dx))
    delta = (want - pose["heading_deg"] + 180) % 360 - 180
    steps = []
    if abs(delta) > 5:
        steps.append({"action": "turn",
                      "params": {"degrees": max(-180.0, min(180.0, delta))}})
    # NOTE: naive — assumes commanded == actual. It doesn't know about slip.
    # That is the point: this is the no-learning baseline for the chart.
    steps.append({"action": "forward",
                  "params": {"meters": max(-2.0, min(2.0, dist_m))}})
    return {"steps": steps, "rationale": "geometric fallback", "new_skill": None}


if __name__ == "__main__":
    # STANDALONE SMOKE TEST — run this before integrating.
    p = plan(
        task="walk to the target marker and stop on it",
        pose={"x_cm": 0, "y_cm": 0, "heading_deg": 0},
        target={"x_cm": 150, "y_cm": 0},
        memories="- attempted ['forward'] -> undershoot, off by 32cm. "
                 "Carpet absorbs stride; commanded 1.0m gave 0.68m.",
    )
    print(json.dumps(p, indent=2))
