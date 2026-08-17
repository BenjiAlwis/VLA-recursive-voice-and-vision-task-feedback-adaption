"""
The action menu + the JSON contract.  Owner: Rikin.

HARD RULE #1 (master_reference.md section 6): the LLM emits JSON from a
FIXED action menu. It never emits code, and it never names a pose or a
skill that does not already exist. Anything failing validation is rejected
and we fall back to the fixed sequence below.

This is not defensive programming for its own sake. It is the difference
between a demo that runs and an hour lost to debugging whatever a 72B model
decided `exec()` should mean at 2pm.

Rewritten from the Go2 version: forward/turn/stop are gone, because the
hardware is a static arm now. See master_reference.md section 4.
"""
import json
import os
from typing import Any, Dict, List

from arm_api import NAMED_POSES

SKILLS_PATH = os.getenv("SKILLS_PATH", "logs/skills.json")
TAUGHT_DIR = os.getenv("TAUGHT_DIR", "logs/skills")

# Offsets are capped at +/-15cm. The corrections this system needs are a
# few centimetres; anything larger is a model that has lost the plot, and
# on real hardware it is the arm hitting the table edge.
MAX_OFFSET_CM = 15.0

PRIMITIVES: Dict[str, Dict] = {
    "move_to": {
        "pose":   ("enum", NAMED_POSES),
        "dx_cm":  ("float", -MAX_OFFSET_CM, MAX_OFFSET_CM, 0.0),
        "dy_cm":  ("float", -MAX_OFFSET_CM, MAX_OFFSET_CM, 0.0),
        "dz_cm":  ("float", -MAX_OFFSET_CM, MAX_OFFSET_CM, 0.0),
    },
    "grip": {
        "state":    ("enum", ["open", "close"]),
        "strength": ("float", 0.0, 100.0, 80.0),
    },
    "replay_skill": {
        "name": ("skill", None),
    },
}


class ValidationError(Exception):
    pass


# ---------------- validation ----------------

def _taught_skill_names() -> List[str]:
    """Demonstrations teach.py has written to disk."""
    import glob
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(f"{TAUGHT_DIR}/*.json"))


def validate_step(step: Any) -> Dict[str, Any]:
    """One action, with every param type-checked and range-clamped.

    Missing OPTIONAL params take their default rather than raising: models
    routinely omit dx_cm when they mean zero, and rejecting the whole plan
    over that would burn a retry for nothing. Missing REQUIRED params (the
    enums) still raise, because guessing which pose was meant is exactly
    the kind of silent invention this validator exists to stop.
    """
    if not isinstance(step, dict):
        raise ValidationError("step is not an object")

    action = step.get("action")
    if action not in PRIMITIVES:
        raise ValidationError(
            f"unknown action {action!r}; allowed: {sorted(PRIMITIVES)}")

    spec = PRIMITIVES[action]
    params = step.get("params") or {}
    if not isinstance(params, dict):
        raise ValidationError(f"{action}.params is not an object")

    clean: Dict[str, Any] = {}
    for key, rule in spec.items():
        kind = rule[0]

        if kind == "enum":
            allowed = rule[1]
            if key not in params:
                raise ValidationError(f"{action} missing required '{key}'")
            val = str(params[key])
            if val not in allowed:
                raise ValidationError(
                    f"{action}.{key}={val!r} not in {allowed}")
            clean[key] = val

        elif kind == "float":
            _, lo, hi, default = rule
            if key not in params or params[key] is None:
                clean[key] = default
                continue
            try:
                val = float(params[key])
            except (TypeError, ValueError):
                raise ValidationError(f"{action}.{key} is not a number")
            if not (lo <= val <= hi):
                raise ValidationError(
                    f"{action}.{key}={val} outside safe range [{lo},{hi}]")
            clean[key] = round(val, 2)

        elif kind == "skill":
            if key not in params:
                raise ValidationError(f"{action} missing required '{key}'")
            val = str(params[key])
            known = _taught_skill_names()
            if val not in known:
                raise ValidationError(
                    f"replay_skill {val!r} was never taught; known: {known}")
            clean[key] = val

    return {"action": action, "params": clean}


def validate_plan(raw: Any) -> Dict[str, Any]:
    """Full plan from the model. Raises ValidationError on anything odd."""
    if isinstance(raw, str):
        try:
            raw = json.loads(strip_fences(raw))
        except json.JSONDecodeError as e:
            raise ValidationError(f"not valid JSON: {e}")
    if not isinstance(raw, dict):
        raise ValidationError("plan is not an object")

    steps = raw.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValidationError("plan.steps must be a non-empty list")
    if len(steps) > 10:
        raise ValidationError(
            f"plan.steps has {len(steps)} entries (max 10) — likely a runaway")

    return {
        "steps": [validate_step(s) for s in steps],
        "rationale": str(raw.get("rationale", ""))[:400],
        "new_skill": raw.get("new_skill"),
    }


def strip_fences(text: str) -> str:
    """Models wrap JSON in ```json fences roughly half the time."""
    t = str(text).strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) >= 2:
            t = parts[1]
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    return t.strip()


# ---------------- the fallback ----------------

def fallback_plan(scene: Dict = None, dx_cm: float = 0.0,
                  dy_cm: float = 0.0) -> Dict[str, Any]:
    """The fixed pick-and-place used with --no-llm, with no API key, and
    whenever the model returns something unusable.

    NOTE: the default offsets are ZERO — this plan does not know about the
    arm's perception bias. That is deliberate. This is the no-learning
    baseline in the ablation chart, and a fallback that silently included
    the correction would make the baseline look identical to the full
    system and destroy the only evidence we have that learning happened.
    """
    return {
        "steps": [
            {"action": "move_to", "params": {"pose": "above_block",
                                             "dx_cm": dx_cm, "dy_cm": dy_cm,
                                             "dz_cm": 0.0}},
            {"action": "move_to", "params": {"pose": "at_block",
                                             "dx_cm": dx_cm, "dy_cm": dy_cm,
                                             "dz_cm": 0.0}},
            {"action": "grip", "params": {"state": "close", "strength": 80.0}},
            {"action": "move_to", "params": {"pose": "above_block",
                                             "dx_cm": dx_cm, "dy_cm": dy_cm,
                                             "dz_cm": 0.0}},
            {"action": "move_to", "params": {"pose": "above_zone",
                                             "dx_cm": 0.0, "dy_cm": 0.0,
                                             "dz_cm": 0.0}},
            {"action": "move_to", "params": {"pose": "at_zone",
                                             "dx_cm": 0.0, "dy_cm": 0.0,
                                             "dz_cm": 0.0}},
            {"action": "grip", "params": {"state": "open", "strength": 0.0}},
            {"action": "move_to", "params": {"pose": "home",
                                             "dx_cm": 0.0, "dy_cm": 0.0,
                                             "dz_cm": 0.0}},
        ],
        "rationale": "fixed pick-and-place (no model)",
        "new_skill": None,
    }


# ---------------- skill library ----------------

SEED_SKILLS = [
    {"name": "pick_and_place", "primitive": False,
     "composed_of": fallback_plan()["steps"],
     "success_count": 0, "learned_from": "seed"},
]


def load_skills() -> List[Dict]:
    try:
        with open(SKILLS_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, list) else list(SEED_SKILLS)
    except FileNotFoundError:
        save_skills(SEED_SKILLS)
        return list(SEED_SKILLS)
    except Exception as e:                                  # noqa: BLE001
        print(f"[schema] skills unreadable ({e}); using seeds")
        return list(SEED_SKILLS)


def save_skills(skills: List[Dict]) -> bool:
    try:
        os.makedirs(os.path.dirname(SKILLS_PATH) or ".", exist_ok=True)
        tmp = f"{SKILLS_PATH}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(skills, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, SKILLS_PATH)
        return True
    except Exception as e:                                  # noqa: BLE001
        print(f"[schema] could not save skills: {e}")
        return False


def add_skill(skill: Dict) -> bool:
    """The model proposes a composed skill. Every step is validated before
    it is stored, so a bad proposal cannot poison the library for the rest
    of the run."""
    if not skill or not skill.get("name"):
        return False
    steps = skill.get("composed_of") or []
    if not steps:
        return False
    try:
        clean = [validate_step(s) for s in steps]
    except ValidationError as e:
        print(f"[schema] rejected proposed skill {skill.get('name')!r}: {e}")
        return False

    skills = load_skills()
    if any(s["name"] == skill["name"] for s in skills):
        return False
    skills.append({
        "name": str(skill["name"]),
        "primitive": False,
        "composed_of": clean,
        "success_count": 0,
        "learned_from": skill.get("learned_from", "unknown"),
    })
    save_skills(skills)
    print(f"[schema] learned new skill {skill['name']!r}")
    return True


def bump_success(name: str) -> None:
    skills = load_skills()
    hit = False
    for s in skills:
        if s["name"] == name:
            s["success_count"] += 1
            hit = True
    if hit:
        save_skills(skills)


def skills_for_prompt() -> str:
    """Compact rendering for the planner prompt. Keep it short — every
    token here is latency on a call the demo waits for."""
    lines = [f"POSES (the only ones that exist): {', '.join(NAMED_POSES)}"]
    for s in load_skills():
        lines.append(f"- {s['name']} (composed, succeeded {s['success_count']}x)")
    taught = _taught_skill_names()
    if taught:
        lines.append(f"HUMAN-TAUGHT (usable via replay_skill): "
                     f"{', '.join(taught)}")
    return "\n".join(lines)


# ---------------- standalone check ----------------

if __name__ == "__main__":
    SKILLS_PATH = "logs/skills_smoke.json"

    print("=== the fallback must validate against our own schema ===")
    fb = fallback_plan()
    assert validate_plan(fb)["steps"] == fb["steps"]
    print(f"  ok — {len(fb['steps'])} steps")

    print("\n=== accepts a fenced, partially-specified model response ===")
    ok = validate_plan("""```json
    {"steps": [{"action": "move_to", "params": {"pose": "at_block", "dx_cm": -4.2}},
               {"action": "grip", "params": {"state": "close"}}],
     "rationale": "compensating for the bias I saw last time"}
    ```""")
    assert ok["steps"][0]["params"]["dy_cm"] == 0.0, "omitted param -> default"
    assert ok["steps"][0]["params"]["dx_cm"] == -4.2
    assert ok["steps"][1]["params"]["strength"] == 80.0, "default strength"
    print(f"  ok — {json.dumps(ok['steps'])}")

    print("\n=== rejects everything it must ===")
    bad = [
        ('{"steps":[{"action":"forward","params":{"meters":1}}]}',
         "a Go2 primitive that no longer exists"),
        ('{"steps":[{"action":"move_to","params":{"pose":"at_the_thing"}}]}',
         "invented pose"),
        ('{"steps":[{"action":"move_to","params":{"dx_cm":1}}]}',
         "missing required pose"),
        ('{"steps":[{"action":"move_to","params":{"pose":"at_block","dx_cm":95}}]}',
         "offset outside the safe range"),
        ('{"steps":[{"action":"grip","params":{"state":"crush"}}]}',
         "invented enum value"),
        ('{"steps":[{"action":"replay_skill","params":{"name":"never_taught"}}]}',
         "skill that was never demonstrated"),
        ('{"steps":[]}', "empty plan"),
        ('not json at all', "unparseable"),
        ('{"steps":"move the arm"}', "steps is not a list"),
        ('{"steps":' + json.dumps([fb["steps"][0]] * 11) + '}', "runaway length"),
    ]
    for payload, why in bad:
        try:
            validate_plan(payload)
        except ValidationError as e:
            print(f"  ✓ rejected ({why}): {str(e)[:64]}")
        else:
            raise SystemExit(f"FAIL: accepted {why} — {payload[:60]}")

    print("\n=== skill library ===")
    save_skills(list(SEED_SKILLS))
    assert add_skill({"name": "corrected_grasp", "composed_of": [
        {"action": "move_to", "params": {"pose": "at_block", "dx_cm": -4.2,
                                         "dy_cm": 2.8}},
        {"action": "grip", "params": {"state": "close", "strength": 85}},
    ], "learned_from": "trial2"}), "a valid skill must be accepted"
    assert not add_skill({"name": "bogus", "composed_of": [
        {"action": "teleport", "params": {}}]}), \
        "a skill containing an invalid step must be rejected wholesale"
    assert not add_skill({"name": "corrected_grasp", "composed_of":
                          fb["steps"]}), "duplicate names must be rejected"
    bump_success("corrected_grasp")
    names = {s["name"]: s["success_count"] for s in load_skills()}
    print(f"  library: {names}")
    assert names["corrected_grasp"] == 1
    assert "bogus" not in names

    print("\n=== prompt rendering ===")
    print("\n".join("  " + l for l in skills_for_prompt().splitlines()))

    os.remove(SKILLS_PATH)
    print("\nschema smoke test passed")
