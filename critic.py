"""
The hybrid judge.  Owner: Rikin (with Benji).

THE ARCHITECTURAL COMMITMENT OF THIS PROJECT (master_reference.md §5, §6):

    ArUco geometry decides PASS / FAIL.     <- cannot hallucinate
    the VLM decides WHY it failed.          <- can hallucinate; harmless

`geometric_verdict()` is the one function in this repo that is never
allowed to be influenced by a model call. A VLM asked "did it succeed?"
will sometimes say yes when it plainly failed. That stops the learning
loop, and on stage it looks like the robot is broken. If you are adding a
path where a model returns a boolean, you are removing the thing that makes
this build work.

FAILURE MODES ARE SHARED STATE. The list below must stay identical to
memory.FAILURE_MODES. It did not used to be: this file carried the Go2
vocabulary (overshoot / undershoot / misalignment) while memory.py had
already moved to the arm's, so every recorded failure hit memory's
"unexpected failure_mode" branch and the retrieval that feeds the planner
was being fed labels the prompt never mentions. The assertion at import
time makes that specific drift impossible to reintroduce quietly.

Side effect worth having: every VLM/heuristic disagreement is logged.
That file is the Toloka payload — it turns our least trustworthy component
into a measured number on a slide.
"""
import base64
import json
import math
import os
import time
from typing import Dict, Optional

import memory

SUCCESS_CM = float(os.getenv("SUCCESS_CM", "5.0"))
DISAGREE_LOG = os.getenv("DISAGREE_LOG", "logs/critic_disagreements.jsonl")

# Single source of truth: memory.py owns the vocabulary, we import it.
FAILURE_MODES = list(memory.FAILURE_MODES)
assert FAILURE_MODES == ["missed_grasp", "dropped_early", "wrong_position",
                         "collision", "no_motion"], \
    ("critic and memory disagree about the failure vocabulary — fix "
     "memory.FAILURE_MODES, do not fork the list here")


# ---------------- geometry: the ONLY pass/fail authority ----------------

def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def geometric_verdict(scene_before: Dict, scene_after: Dict) -> Dict:
    """Ground truth. No model is involved, now or ever.

    Success is the block resting within SUCCESS_CM of the zone centre and
    NOT still in the gripper. Both halves matter: an arm holding the block
    over the zone has not placed it, and scoring that as success is exactly
    the kind of near-miss a vision model would wave through.
    """
    block_after = scene_after.get("block_cm", (0.0, 0.0))
    zone_after = scene_after.get("zone_cm", (0.0, 0.0))
    block_before = scene_before.get("block_cm", (0.0, 0.0))
    zone_before = scene_before.get("zone_cm", (0.0, 0.0))

    err_after = _dist(block_after, zone_after)
    err_before = _dist(block_before, zone_before)
    block_moved = _dist(block_before, block_after)
    holding = bool(scene_after.get("holding", False))
    collided = bool(scene_after.get("collided", False))

    grip_before = scene_before.get("gripper_cm", (0.0, 0.0, 0.0))
    grip_after = scene_after.get("gripper_cm", (0.0, 0.0, 0.0))
    arm_moved = _dist(grip_before, grip_after)

    return {
        "passed": err_after <= SUCCESS_CM and not holding,
        "error_cm": round(err_after, 1),
        "error_before_cm": round(err_before, 1),
        "improved": err_after < err_before - 0.5,
        "block_moved_cm": round(block_moved, 1),
        "still_holding": holding,
        "collided": collided,
        "commanded": bool(scene_after.get("commanded", True)),
        "measured_by": scene_after.get("source", "unknown"),

        # The names reason.heuristic_mode() documents. These are NOT
        # redundant decoration — reason.py reads 'collision', 'grasped',
        # 'in_zone' and 'arm_moved_cm', while this function historically
        # emitted 'collided' and 'still_holding'. The two files were
        # written in parallel and the mismatch meant Rikin's collision
        # branch could never fire and his grasped check always saw None.
        # Same class of drift as the failure-mode vocabulary split; caught
        # by the assertion in the self-test below.
        "collision": collided,
        "arm_moved_cm": round(arm_moved, 1),
        # "did the gripper end up holding the block", per reason.py's
        # docstring. Either it still has it, or it moved it — which it
        # could only do by having held it. NOT keyed off grasp_cm: that
        # merely records that a close was ATTEMPTED, which is true even
        # for a grasp that closed on empty air.
        "grasped": holding or block_moved >= 1.0,
        "in_zone": err_after <= SUCCESS_CM,
    }


def heuristic_diagnosis(mode: str, verdict: Dict, scene_after: Dict) -> str:
    """A spoken diagnosis built from measurements alone — no model.

    This carries the ACTUAL NUMBERS, not just the final error. That matters
    more than it looks: memory.py stores this sentence verbatim and hands it
    back to the planner next attempt, so "the block ended up 34cm from the
    zone" teaches nothing (the planner already knows the error), whereas
    "I closed 4.2cm right and 2.8cm short of the block" is directly
    invertible into the dx_cm/dy_cm that fixes it.

    It is also what makes --no-llm a fair ablation arm rather than a
    strawman: the geometric baseline gets a genuinely useful diagnosis too,
    so the chart measures what the MODEL adds, not what numbers add.
    """
    block = scene_after.get("block_cm", (0.0, 0.0))
    zone = scene_after.get("zone_cm", (0.0, 0.0))

    if mode == "missed_grasp":
        err = _grasp_error(scene_after)
        if err is None:
            return "I closed my gripper somewhere other than on the block."
        gx, gy = err
        return (f"I closed my gripper {abs(gx):.1f} centimetres "
                f"{'right of' if gx > 0 else 'left of'} and "
                f"{abs(gy):.1f} centimetres "
                f"{'beyond' if gy > 0 else 'short of'} the block, "
                f"so I came away with nothing.")
    if mode == "collision":
        return ("I dragged the block across the table instead of lifting "
                "it clear first.")
    if mode == "dropped_early":
        return ("I never let go of the block, so the placement did not "
                "finish.")
    if mode == "no_motion":
        return "I did not move at all."
    return (f"I put the block down {verdict['error_cm']} centimetres from "
            f"the zone centre, {abs(block[0] - zone[0]):.1f} across and "
            f"{abs(block[1] - zone[1]):.1f} away.")


def _grasp_error(scene_after: Dict):
    """(dx, dy) of where the gripper CLOSED relative to the block, or None.

    Read from 'grasp_cm', latched by the arm at the moment the gripper
    closed — NOT from the final gripper position. Every sensible plan ends
    with move_to home, so the gripper's resting place is tens of
    centimetres from the block and produces a correction with the wrong
    sign and magnitude. Applying that would drive the arm further from the
    block on every retry, which is worse than not learning at all.
    """
    grasp = scene_after.get("grasp_cm")
    target = scene_after.get("grasp_target_cm") or scene_after.get("block_cm")
    if not grasp or not target:
        return None
    return (grasp[0] - target[0], grasp[1] - target[1])


def _suggested_fix(mode: str, scene_after: Dict) -> str:
    """The correction, stated as the exact offset that cancels the error.

    For a missed grasp this is literally the dx_cm/dy_cm the planner should
    emit next attempt — the sign is already inverted here so that neither
    the model nor a reader has to do it. Getting that sign backwards
    doubles the error instead of cancelling it, and it is the single
    easiest thing in this system to get wrong.
    """
    if mode == "missed_grasp":
        err = _grasp_error(scene_after)
        if err is None:
            return "close the gripper closer to the block"
        return (f"next attempt use dx_cm={-err[0]:+.1f} "
                f"dy_cm={-err[1]:+.1f} on above_block and at_block")
    if mode == "collision":
        return "lift to above_block before traversing to above_zone"
    if mode == "dropped_early":
        return "open the gripper at at_zone, and grip at strength 80"
    if mode == "no_motion":
        return "emit at least one move_to step"
    return "re-check the zone position before releasing"


def heuristic_mode(verdict: Dict, scene_before: Dict,
                   scene_after: Dict) -> str:
    """Cheap, deterministic diagnosis so the loop never blocks on the LLM.

    Order matters — the checks run most-specific first. A collision that
    also dropped the block is a collision, because that is the thing the
    planner has to stop doing.
    """
    if not verdict["commanded"]:
        return "no_motion"
    if verdict["collided"]:
        return "collision"
    if verdict["block_moved_cm"] < 1.0:
        # The arm moved but the block never did: it closed on empty air.
        return "missed_grasp"
    if verdict["still_holding"]:
        return "dropped_early"      # never released; the place did not finish
    return "wrong_position"


# ---------------- the VLM: diagnosis only, never a verdict ----------------

DIAGNOSE_SYSTEM = f"""You diagnose a robot arm's FAILED attempt at a \
pick-and-place task.

You are NOT deciding whether it succeeded. That has already been measured \
by an overhead camera and is not in question. Your only job is to explain \
WHY it failed and what to change next time.

Output ONLY JSON, no prose and no markdown fences:
{{"failure_mode": one of {FAILURE_MODES},
  "diagnosis": "<one sentence, first person, spoken aloud by the robot>",
  "suggested_fix": "<one short imperative, mention centimetres if relevant>"}}

The diagnosis is read out loud to an audience, so write it as the robot \
speaking about itself:
"I closed my gripper four centimetres to the right of the block, so I came \
away with nothing."
"""


def diagnose(verdict: Dict, scene_before: Dict, scene_after: Dict,
             plan: Dict, frame=None, task: str = "") -> Dict:
    """The VLM explains the failure. Falls back to heuristics on any error.

    DELEGATES to reason.diagnose(), which is Rikin's and is the better
    implementation: it refuses to run on a passing verdict, strips every
    success-like key out of the model's reply, rejects a prompt_update
    containing code, and logs its own disagreements. This file used to
    carry a parallel VLM call doing the same job less carefully — two
    prompts to keep in sync, two places to leak a verdict.

    Returns a dict WITHOUT a 'passed' key. That omission is load-bearing:
    no shape of this return value can overwrite the geometric verdict when
    the caller merges them.
    """
    mode = heuristic_mode(verdict, scene_before, scene_after)
    heur = {
        "failure_mode": mode,
        "diagnosis": heuristic_diagnosis(mode, verdict, scene_after),
        "suggested_fix": _suggested_fix(mode, scene_after),
        "source": "heuristic",
    }

    try:
        import reason
    except Exception as e:                                  # noqa: BLE001
        print(f"[critic] reason.py unavailable, using the heuristic: {e}")
        return heur

    try:
        result = reason.diagnose(
            task or "pick up the block and place it in the target zone",
            verdict,
            frame_after=frame,
            actions=[s.get("action") for s in plan.get("steps", [])],
        )
    except Exception as e:                                  # noqa: BLE001
        print(f"[critic] reason.diagnose raised, using the heuristic: {e}")
        return heur

    if not result:
        return heur

    # reason.py returns prompt_update; this file's callers (narrate, memory)
    # expect suggested_fix. Carry the measured offset through as the fix
    # when we have one, because it is strictly more actionable than prose.
    merged = dict(result)
    merged.setdefault("diagnosis", heur["diagnosis"])
    merged["suggested_fix"] = (result.get("prompt_update")
                               or heur["suggested_fix"])
    if mode == "missed_grasp" and _grasp_error(scene_after):
        merged["suggested_fix"] = heur["suggested_fix"]

    # Belt and braces. reason.py already strips these; if that ever
    # regresses, it must not reach the merge in critique().
    for banned in ("passed", "success", "succeeded", "ok", "solved"):
        merged.pop(banned, None)

    _log_if_disagree(merged, heur, verdict)
    return merged


def _log_if_disagree(vlm: Dict, heur: Dict, verdict: Dict) -> None:
    """Toloka payload: every case where the model and the heuristic
    disagree. Ship 30-50 of these for human labels and report critic
    accuracy as a real number."""
    if vlm.get("failure_mode") == heur.get("failure_mode"):
        return
    try:
        os.makedirs(os.path.dirname(DISAGREE_LOG) or ".", exist_ok=True)
        with open(DISAGREE_LOG, "a") as f:
            f.write(json.dumps({
                "ts": time.time(),
                "vlm_mode": vlm.get("failure_mode"),
                "heuristic_mode": heur.get("failure_mode"),
                "vlm_diagnosis": vlm.get("diagnosis"),
                "error_cm": verdict["error_cm"],
                "block_moved_cm": verdict["block_moved_cm"],
                "still_holding": verdict["still_holding"],
            }) + "\n")
    except Exception as e:                                  # noqa: BLE001
        print(f"[critic] could not log disagreement: {e}")


# ---------------- the one function the loop calls ----------------

def critique(scene_before: Dict, scene_after: Dict, plan: Dict,
             frame=None, use_llm: bool = True, task: str = "") -> Dict:
    """Geometry first, always. The model only ever adds prose.

    The merge order below is the safety property: `verdict` is spread
    LAST, so even a diagnosis dict that somehow contained 'passed' could
    not override the camera.
    """
    verdict = geometric_verdict(scene_before, scene_after)

    # The raw residual, exposed so the harness can accumulate corrections
    # across trials without re-parsing the English sentence.
    verdict["grasp_error_cm"] = _grasp_error(scene_after)

    if verdict["passed"]:
        return {**verdict, "failure_mode": None,
                "diagnosis": "I placed the block in the zone.",
                "suggested_fix": None, "source": "geometry"}

    if use_llm:
        explanation = diagnose(verdict, scene_before, scene_after, plan,
                               frame, task=task)
    else:
        mode = heuristic_mode(verdict, scene_before, scene_after)
        explanation = {
            "failure_mode": mode,
            "diagnosis": heuristic_diagnosis(mode, verdict, scene_after),
            "suggested_fix": _suggested_fix(mode, scene_after),
            "source": "heuristic",
        }

    return {**explanation, **verdict}


# ---------------- standalone check ----------------

if __name__ == "__main__":
    import arm_api

    def scene(block, zone=(-16.0, 22.0), gripper=(0, 20, 20),
              holding=False, collided=False, commanded=True):
        return {"block_cm": block, "zone_cm": zone, "gripper_cm": gripper,
                "holding": holding, "collided": collided,
                "commanded": commanded, "source": "mock"}

    start = scene((18.0, 24.0))

    print("=== geometry decides pass/fail, with no model anywhere ===")
    cases = [
        ("placed on target",   scene((-16.0, 22.0)),                 True),
        ("placed 2cm off",     scene((-14.5, 21.0)),                 True),
        ("placed 9cm off",     scene((-8.0, 20.0)),                  False),
        ("held over the zone", scene((-16.0, 22.0), holding=True),   False),
        ("never moved",        scene((18.0, 24.0)),                  False),
    ]
    for label, after, expect in cases:
        v = geometric_verdict(start, after)
        print(f"  {label:<22} err={v['error_cm']:>5.1f}cm  "
              f"holding={v['still_holding']!s:<5} -> passed={v['passed']}")
        assert v["passed"] is expect, f"{label}: expected passed={expect}"

    print("\n=== heuristic modes, all five reachable ===")
    # The gripper position matters here. critic.heuristic_mode keys off the
    # arm's `commanded` flag, reason.heuristic_mode keys off arm_moved_cm —
    # so a fixture that leaves the gripper parked reads as "no motion" to
    # reason.py even when the case under test is a missed grasp. Every
    # non-no_motion case therefore moves the arm, as a real trial does.
    reached = (18.0, 24.0, 1.0)         # arm travelled to the block
    placed = (-16.0, 22.0, 1.0)         # arm travelled to the zone
    expectations = [
        ("no_motion",      start, scene((18.0, 24.0), commanded=False)),
        ("collision",      start, scene((5.0, 23.0), gripper=reached,
                                        collided=True)),
        ("missed_grasp",   start, scene((18.0, 24.0), gripper=reached)),
        ("dropped_early",  start, scene((-16.0, 22.0), gripper=placed,
                                        holding=True)),
        ("wrong_position", start, scene((-6.0, 18.0), gripper=placed)),
    ]
    seen = set()
    for expect, before, after in expectations:
        v = geometric_verdict(before, after)
        mode = heuristic_mode(v, before, after)
        print(f"  {expect:<15} -> {mode}")
        assert mode == expect, f"expected {expect}, got {mode}"
        seen.add(mode)
    assert seen == set(FAILURE_MODES), f"unreachable modes: {set(FAILURE_MODES) - seen}"

    print("\n=== every mode the critic emits is storable by memory.py ===")
    for mode in FAILURE_MODES:
        assert mode in memory.FAILURE_MODES, mode
    print(f"  shared vocabulary: {FAILURE_MODES}")

    print("\n=== a rogue model cannot flip the verdict ===")
    v = geometric_verdict(start, scene((-8.0, 20.0)))
    rogue = {"failure_mode": "wrong_position", "diagnosis": "I nailed it.",
             "passed": True, "error_cm": 0.0, "source": "vlm"}
    merged = {**rogue, **v}
    print(f"  model claimed passed=True, merged verdict passed={merged['passed']}")
    assert merged["passed"] is False, \
        "MERGE ORDER BROKEN — the model overrode the camera"
    assert merged["error_cm"] == v["error_cm"], "the model overrode the measurement"

    print("\n=== end-to-end against the real mock arm, no LLM ===")
    arm = arm_api.MockArm()
    before = arm.get_scene()
    import schema
    for step in schema.fallback_plan()["steps"]:
        if step["action"] == "move_to":
            arm.move_to(step["params"]["pose"], step["params"]["dx_cm"],
                        step["params"]["dy_cm"], step["params"]["dz_cm"])
        else:
            arm.grip(step["params"]["state"], step["params"].get("strength", 80))
    result = critique(before, arm.get_scene(), schema.fallback_plan(),
                      use_llm=False)
    print(f"  uncorrected fallback -> passed={result['passed']} "
          f"mode={result['failure_mode']} err={result['error_cm']}cm")
    assert not result["passed"], "the naive plan must fail, or nothing is learned"
    assert result["failure_mode"] == "missed_grasp"

    print("\n=== the verdict carries every key reason.py reads ===")
    # reason.heuristic_mode() documents exactly these. They were missing
    # once, which silently disabled Rikin's collision and grasp branches.
    import reason
    v = geometric_verdict(start, scene((5.0, 23.0), collided=True))
    for key in ("collision", "grasped", "in_zone", "arm_moved_cm",
                "block_moved_cm", "error_cm", "passed"):
        assert key in v, f"reason.py reads {key!r}, verdict does not carry it"
    assert reason.heuristic_mode(v) == "collision", \
        (f"reason.py could not see the collision: got "
         f"{reason.heuristic_mode(v)}. The two files' key names have drifted.")
    print(f"  reason.heuristic_mode agrees: {reason.heuristic_mode(v)}")

    # The contract is that reason.py can READ our verdict and return a
    # VALID mode — not that it returns the SAME mode. The two estimators
    # see different data on purpose: this file has still_holding and the
    # latched grasp position, reason.py does not, so it cannot separate
    # "dropped mid-traverse" from "placed but off target". Forcing them to
    # agree would throw away the finer signal, and their disagreements are
    # exactly what logs/critic_disagreements.jsonl exists to collect.
    print("\n  mode-by-mode, critic vs reason (disagreement is expected):")
    for expect, before, after in expectations:
        vv = geometric_verdict(before, after)
        theirs = reason.heuristic_mode(vv)
        assert theirs in FAILURE_MODES, \
            f"reason.py returned {theirs!r}, outside the shared vocabulary"
        flag = "  same" if theirs == expect else "  <- differs, logged"
        print(f"    critic={expect:<15} reason={theirs:<15}{flag}")

    assert set(reason.FAILURE_MODES) == set(FAILURE_MODES), \
        "reason.py and memory.py disagree about the failure vocabulary"

    print("\ncritic smoke test passed")
