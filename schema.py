"""
Skill library + validator.  Owner: Benji.

HARD RULE: the LLM emits JSON conforming to Plan. It never emits code.
Anything that fails validation is rejected and we fall back. This is the
difference between a demo that runs and an hour lost to eval() debugging.
"""
import json
import os
from typing import List, Dict, Any

SKILLS_PATH = "logs/skills.json"

# Primitives map 1:1 onto robot_api. The LLM may only compose these.
PRIMITIVES = {
    "forward": {"meters": (float, -2.0, 2.0)},
    "turn":    {"degrees": (float, -180.0, 180.0)},
    "stop":    {},
}

SEED_SKILLS = [
    {"name": "forward", "primitive": True, "params": {"meters": 0.5},
     "composed_of": [], "success_count": 0, "learned_from": "seed"},
    {"name": "turn", "primitive": True, "params": {"degrees": 15.0},
     "composed_of": [], "success_count": 0, "learned_from": "seed"},
]


class ValidationError(Exception):
    pass


def validate_step(step: Dict[str, Any]) -> Dict[str, Any]:
    """One step = one primitive call with in-range params."""
    if not isinstance(step, dict):
        raise ValidationError("step is not an object")
    action = step.get("action")
    if action not in PRIMITIVES:
        raise ValidationError(f"unknown action '{action}'")
    spec = PRIMITIVES[action]
    params = step.get("params", {}) or {}
    clean = {}
    for key, (typ, lo, hi) in spec.items():
        if key not in params:
            raise ValidationError(f"{action} missing param '{key}'")
        try:
            val = typ(params[key])
        except (TypeError, ValueError):
            raise ValidationError(f"{action}.{key} not a {typ.__name__}")
        if not (lo <= val <= hi):
            raise ValidationError(
                f"{action}.{key}={val} outside safe range [{lo},{hi}]")
        clean[key] = val
    return {"action": action, "params": clean}


def validate_plan(raw: Any) -> Dict[str, Any]:
    """Full plan from the LLM. Raises ValidationError on anything odd."""
    if isinstance(raw, str):
        raw = strip_fences(raw)
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValidationError(f"not valid JSON: {e}")
    if not isinstance(raw, dict):
        raise ValidationError("plan is not an object")

    steps = raw.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValidationError("plan.steps must be a non-empty list")
    if len(steps) > 6:
        raise ValidationError("plan.steps too long (max 6) — likely a runaway")

    return {
        "steps": [validate_step(s) for s in steps],
        "rationale": str(raw.get("rationale", ""))[:400],
        "new_skill": raw.get("new_skill"),
    }


def strip_fences(text: str) -> str:
    """LLMs wrap JSON in ```json fences roughly half the time."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.startswith("json"):
            t = t[4:]
    return t.strip()


# ---------- skill library persistence ----------

def load_skills() -> List[Dict]:
    if not os.path.exists(SKILLS_PATH):
        save_skills(SEED_SKILLS)
        return list(SEED_SKILLS)
    with open(SKILLS_PATH) as f:
        return json.load(f)


def save_skills(skills: List[Dict]) -> None:
    os.makedirs(os.path.dirname(SKILLS_PATH), exist_ok=True)
    with open(SKILLS_PATH, "w") as f:
        json.dump(skills, f, indent=2)


def add_skill(skill: Dict) -> bool:
    """LLM proposes a composed skill. Validate its steps before storing."""
    if not skill or not skill.get("name"):
        return False
    for step in skill.get("composed_of", []):
        try:
            validate_step(step)
        except ValidationError:
            return False
    skills = load_skills()
    if any(s["name"] == skill["name"] for s in skills):
        return False
    skills.append({
        "name": skill["name"],
        "primitive": False,
        "params": skill.get("params", {}),
        "composed_of": skill.get("composed_of", []),
        "success_count": 0,
        "learned_from": skill.get("learned_from", "unknown"),
    })
    save_skills(skills)
    return True


def bump_success(name: str) -> None:
    skills = load_skills()
    for s in skills:
        if s["name"] == name:
            s["success_count"] += 1
    save_skills(skills)


def skills_for_prompt() -> str:
    """Compact rendering for the planner prompt. Keep it short — tokens cost latency."""
    lines = []
    for s in load_skills():
        kind = "primitive" if s.get("primitive") else "learned"
        lines.append(f"- {s['name']} ({kind}, used_ok={s['success_count']}) "
                     f"params={list(s.get('params', {}).keys())}")
    return "\n".join(lines)
