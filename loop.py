"""
THE HARNESS.  Owner: Benji.  Integrator — only person merging after T+0:30.

    plan -> execute -> measure -> critique -> remember -> narrate -> repeat

and, after N consecutive failures on the same task, escalate to a human:
ask out loud, listen for a spoken correction, and if that is not enough,
have them physically demonstrate the motion on the leader arm.

    python loop.py --no-llm                 # prove the mechanics, zero AI
    python loop.py --reflex                 # learns with no API key at all
    python loop.py                          # the full loop
    python loop.py --voice                  # human speaks the task
    python loop.py --rounds 5 --perturb-at 3   # the stage demo
    python loop.py --evidence               # all four ablation arms + chart

`python loop.py --no-llm` completing its trials IS the milestone. Everything
after that is swapping stubs for real parts.

SINGLE-WRITER RULE. This file is the ONLY writer of shared/red_to_blue.json.
teach.py deliberately queues replay requests instead of writing them (see
its docstring) — two writers would each need merge logic to avoid clobbering
the other's prompt_update, and one writer needs none.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

import critic
import memory
import narrate
import planner
import schema
import teach
from arm_api import get_arm

TRIAL_LOG = os.getenv("TRIAL_LOG", "logs/trials.jsonl")
SHARED_MSG = os.getenv("RED_TO_BLUE", "shared/red_to_blue.json")
TEACH_AFTER = int(os.getenv("TEACH_AFTER", "2"))


# ---------------- execution ----------------

def execute(arm, plan: Dict) -> List[str]:
    """Run a validated plan against the arm. Returns what was actually run.

    Every step is re-validated here even though the planner already did it.
    That is not paranoia about the model — it is protection against a
    hand-edited skill file or a taught skill written by an older version,
    both of which reach this function without passing through the planner.
    """
    done = []
    for step in plan.get("steps", []):
        try:
            step = schema.validate_step(step)
        except schema.ValidationError as e:
            print(f"  ⚠ skipping invalid step: {e}")
            continue

        action, params = step["action"], step["params"]
        print(f"  ▸ {action}({', '.join(f'{k}={v}' for k, v in params.items())})")

        if action == "move_to":
            arm.move_to(params["pose"], params["dx_cm"],
                        params["dy_cm"], params["dz_cm"])
        elif action == "grip":
            arm.grip(params["state"], params["strength"])
        elif action == "replay_skill":
            _replay(arm, params["name"])
        done.append(action)
    return done


def _replay(arm, name: str) -> None:
    """Replay a human demonstration recorded by teach.py.

    teach.py records but never moves an arm — that boundary is deliberate
    and documented there. Driving the follower is the harness's job, so the
    joint-level playback lives here.
    """
    skill = teach.load_skill(name)
    if not skill:
        print(f"  ⚠ no taught skill named {name!r}")
        return
    waypoints = skill.get("waypoints", [])
    print(f"  ▸ replaying {name!r} ({len(waypoints)} waypoints)")
    for wp in waypoints:
        joints = wp.get("joints") or {}
        if joints:
            arm.set_joints(joints)


# ---------------- the Red -> Blue channel ----------------

def emit_red_to_blue(payload: Dict) -> None:
    """Write the Reasoning -> Feedback Controller message.

    Atomic, because Blue polls this file while we write it: json.dump into
    an open handle is not atomic and a poll landing mid-write reads
    truncated JSON and raises on their side.
    """
    try:
        os.makedirs(os.path.dirname(SHARED_MSG) or ".", exist_ok=True)
        tmp = f"{SHARED_MSG}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, SHARED_MSG)
    except Exception as e:                                  # noqa: BLE001
        print(f"[loop] could not emit {SHARED_MSG}: {e}")


def _applied_offset(plan: Dict) -> tuple:
    """The dx/dy this plan actually used on at_block."""
    for step in plan.get("steps", []):
        if step.get("action") == "move_to" and \
                step.get("params", {}).get("pose") == "at_block":
            p = step["params"]
            return (float(p.get("dx_cm", 0.0)), float(p.get("dy_cm", 0.0)))
    return (0.0, 0.0)


def _next_offset(plan: Dict, verdict: Dict):
    """The ABSOLUTE offset to use next attempt, or None.

    Corrections must COMPOUND, not reset. The measured grasp error is a
    RESIDUAL — it is what remained after the offset this trial already
    applied. Storing the residual alone would make the arm oscillate: it
    corrects by 4cm, measures ~0 residual, concludes no correction is
    needed, and misses again on the next attempt. So the stored value is
    applied - residual, which converges.
    """
    residual = verdict.get("grasp_error_cm")
    if not residual:
        return None
    ax, ay = _applied_offset(plan)
    return (round(ax - residual[0], 1), round(ay - residual[1], 1))


def log_trial(rec: Dict) -> None:
    try:
        os.makedirs(os.path.dirname(TRIAL_LOG) or ".", exist_ok=True)
        with open(TRIAL_LOG, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception as e:                                  # noqa: BLE001
        print(f"[loop] could not log trial: {e}")


# ---------------- escalation ----------------

def escalate(arm, task: str, verdict: Dict, corrections: List[str],
             use_voice: bool, allow_teaching: bool) -> Dict:
    """Two consecutive failures. Ask a human.

    Escalates cheapest-first, because each rung costs more stage time than
    the last:

        1. a spoken correction        seconds, usually enough
        2. a glasses photo            ~20s, shows what the fixed camera cannot
        3. a leader-arm demonstration ~a minute, but teaches a whole skill

    Returns {"replay": <payload or None>, "view": <image path or None>}.
    Both may be None — every caller must cope, because with the glasses
    unpaired and teaching disabled this function has nothing to offer and
    the run must continue regardless.
    """
    mode = verdict.get("failure_mode", "the same mistake")

    # Say nothing if there is no channel to actually ask through. Announcing
    # "can you help me?" during a --no-teaching ablation sweep is a promise
    # to an audience that nothing will follow up on, and it makes the
    # control runs sound broken rather than controlled.
    out: Dict = {"replay": None, "view": None}

    if not use_voice and not allow_teaching:
        print(f"[loop] {TEACH_AFTER} consecutive failures; "
              f"escalation disabled for this run")
        return out

    narrate.speak(f"I have failed twice with {str(mode).replace('_', ' ')}. "
                  f"Can you help me?")

    if use_voice:
        try:
            import glasses
            said = glasses.listen_for_feedback(timeout_s=8)
        except Exception as e:                              # noqa: BLE001
            print(f"[loop] voice feedback unavailable: {e}")
            said = ""
        if said:
            corrections.append(said)
            narrate.speak("Understood. Let me try that.")
            return out                  # the words go into the next prompt

    # Rung 2: a photo through the glasses. master_reference section 5 routes
    # Meta Video to the PLANNER as scene context, never to the critic — a
    # human-held camera showing an angle the fixed camera cannot is exactly
    # the thing that must not be allowed near a pass/fail decision.
    #
    # Silent no-op when GLASSES_BACKEND=off, which is the T-1:00 cut.
    try:
        import glasses
        out["view"] = glasses.request_view(
            f"I keep getting {str(mode).replace('_', ' ')}.", timeout_s=20)
    except Exception as e:                                  # noqa: BLE001
        print(f"[loop] glasses view unavailable: {e}")
    if out["view"]:
        return out                      # try again with the human's view

    if not allow_teaching:
        return out

    narrate.speak("Please show me on the leader arm.")
    name = f"taught_{int(time.time())}"
    # Accept Team Yellow's variable name too. Gating on the bare
    # LEADER_PORT alone meant that with only SO101_LEADER_PORT set — which
    # is what their configs/leader.env actually defines — this silently
    # recorded a MOCK trajectory while a human stood there physically
    # demonstrating. It looks like it worked, which is the worst way to
    # fail.
    source = ("leader"
              if (os.getenv("LEADER_PORT") or os.getenv("SO101_LEADER_PORT"))
              else "mock")
    skill = teach.record_demonstration(
        name, seconds=int(os.getenv("TEACH_SECONDS", "8")),
        source=source, task=task)
    if not skill:
        narrate.speak("I could not record that. I will keep trying myself.")
        return out

    # teach.py queues; WE emit. Consume-once, so the skill is not replayed
    # forever once it has been handed over.
    teach.request_replay(name, reason=f"human demonstrated after {mode}")
    narrate.announce_learned(name)
    out["replay"] = teach.take_pending_replay()
    return out


# ---------------- the loop ----------------

def run_task(arm, task: str, max_trials: int = 5, use_memory: bool = True,
             use_llm: bool = True, use_reflex: bool = False,
             use_voice: bool = False,
             allow_teaching: bool = True, surface: str = "table",
             run_id: str = "default", perturb_at: int = 0) -> Dict:

    consecutive = 0
    corrections: List[str] = []
    pending_replay: Optional[Dict] = None
    help_view: Optional[str] = None     # a glasses photo, for ONE next trial

    for trial in range(1, max_trials + 1):
        print(f"\n─── trial {trial}/{max_trials} — {task}")
        narrate.announce_attempt(trial, task)

        if hasattr(arm, "reset_trial"):
            arm.reset_trial()

        scene_before = arm.get_scene()
        # The glasses photo, when a human just supplied one, outranks the
        # wrist camera as planner context — it was taken deliberately to
        # show what the fixed view could not. Consumed once: a photo of the
        # previous failure describes a world that no longer exists.
        frame = help_view or arm.get_frame("wrist")
        if help_view:
            print(f"  👓 planning with the glasses view: {help_view}")
        help_view = None
        mems = memory.for_prompt(task, surface) if use_memory \
            else "(memory disabled for this run)"

        if pending_replay:
            plan = {"steps": [{"action": "replay_skill",
                               "params": {"name": pending_replay["replay_skill"]}}],
                    "rationale": "replaying the human demonstration",
                    "new_skill": None}
        elif use_reflex:
            plan = planner.reflex_plan(scene_before, mems)
        elif use_llm:
            plan = planner.plan(task, scene_before, mems, frame=frame,
                                corrections=corrections)
            # The model was unreachable. Rather than fall all the way back
            # to a plan that ignores everything we have learned, drop to
            # the reflex correction — the robot keeps improving on stage
            # even if Nebius goes down mid-demo. Tagged in the trial log so
            # the evidence chart can tell this apart from a real model run.
            if plan.get("source") == "fallback" and use_memory:
                reflex = planner.reflex_plan(scene_before, mems)
                if "no correction stored" not in reflex["rationale"]:
                    print("  ↩ model unreachable — using the stored correction")
                    plan = {**reflex, "source": "reflex_fallback"}
        else:
            plan = planner.fallback_plan(scene_before)

        print(f"  plan: {plan.get('rationale', '')}")

        message = {
            "ts": time.time(), "run_id": run_id, "trial": trial, "task": task,
            "prompt_update": plan.get("rationale", ""),
            "reason": f"trial {trial}",
            "confidence": 0.5,
            "replay_skill": None,
            "steps": plan.get("steps", []),
        }
        if pending_replay:
            message.update(pending_replay)
        emit_red_to_blue(message)
        pending_replay = None

        actions = execute(arm, plan)

        scene_after = arm.get_scene()
        verdict = critic.critique(scene_before, scene_after, plan,
                                  frame=arm.get_frame("wrist"),
                                  use_llm=use_llm)

        log_trial({
            "run_id": run_id, "ts": time.time(), "task": task, "trial": trial,
            "use_memory": use_memory, "use_llm": use_llm, "surface": surface,
            "actions": actions, "rationale": plan.get("rationale", ""),
            "error_cm": verdict["error_cm"], "passed": verdict["passed"],
            "failure_mode": verdict.get("failure_mode"),
            "diagnosis": verdict.get("diagnosis"),
            "planned_by": plan.get("source", "fallback"),
            "diagnosed_by": verdict.get("source"),
            "measured_by": verdict.get("measured_by"),
            "corrections": list(corrections),
        })

        if verdict["passed"]:
            print(f"  ✅ PASS  error={verdict['error_cm']}cm")
            narrate.announce_success(trial, verdict["error_cm"])
            for name in {s["action"] for s in plan.get("steps", [])}:
                schema.bump_success(name)
            if plan.get("new_skill"):
                schema.add_skill({**plan["new_skill"],
                                  "learned_from": f"{task}/trial{trial}"})
            return {"solved": True, "trials": trial,
                    "error_cm": verdict["error_cm"]}

        print(f"  ❌ FAIL  error={verdict['error_cm']}cm  "
              f"mode={verdict['failure_mode']}  ({verdict.get('source')})")
        narrate.announce_failure(verdict)

        if use_memory:
            # What memory stores is deliberately RICHER than what is spoken
            # aloud: the audience wants the sentence, the planner wants the
            # numbers. Appending the fix here keeps narrate.py's line clean
            # while making the retrieved note directly actionable.
            note = verdict["diagnosis"]
            nxt = _next_offset(plan, verdict)
            if nxt:
                note += f" Next attempt use dx_cm={nxt[0]:+.1f} dy_cm={nxt[1]:+.1f}."
            elif verdict.get("suggested_fix"):
                note += f" Fix: {verdict['suggested_fix']}."
            memory.record(task, plan, verdict["error_cm"],
                          verdict["failure_mode"], note, surface)

        consecutive += 1
        if consecutive >= TEACH_AFTER and trial < max_trials:
            helped = escalate(arm, task, verdict, corrections,
                              use_voice, allow_teaching)
            pending_replay = helped.get("replay")
            help_view = helped.get("view")
            consecutive = 0

    return {"solved": False, "trials": max_trials,
            "error_cm": verdict["error_cm"]}


# ---------------- rounds: the actual demo shape ----------------

def run_rounds(arm, task: str, rounds: int, perturb_round: int = 0,
               perturb_kind: str = "calibration",
               run_id: str = "demo", **kw) -> List[Dict]:
    """Attempt the same task several times over, KEEPING memory between.

    This is the demo from master_reference §10, and it is why one call to
    run_task is not enough: the story is "round 1 took four attempts,
    round 2 took one" — which requires more than one round to exist. A
    single run_task returns the moment it succeeds, so skill reuse, the
    perturbation, and the teaching escalation never get a chance to happen.

    Between rounds a human puts the block back. The ZONE stays wherever it
    is, so a perturbation persists and the robot has to re-converge on the
    world as it now is rather than the one it learned.
    """
    results = []
    for rnd in range(1, rounds + 1):
        print(f"\n{'━' * 62}\n  ROUND {rnd}/{rounds}\n{'━' * 62}")

        if hasattr(arm, "reset_task"):
            arm.reset_task()

        if perturb_round and rnd == perturb_round and hasattr(arm, "perturb"):
            narrate.speak("Wait. Something just changed.")
            arm.perturb(perturb_kind)

        # Each round gets its own run_id so chart.py can plot them as
        # separate series — that IS the "attempts going down" evidence.
        result = run_task(arm, task, run_id=f"{run_id}_r{rnd}", **kw)
        result["round"] = rnd
        result["perturbed"] = bool(perturb_round and rnd == perturb_round)
        results.append(result)
        print(f"\n  round {rnd}: "
              f"{'solved in ' + str(result['trials']) + ' attempt(s)' if result['solved'] else 'not solved'}")

    print(f"\n{'━' * 62}\n  ATTEMPTS PER ROUND — this is the demo\n{'━' * 62}")
    for r in results:
        bar = "█" * r["trials"]
        flag = "  ← world changed" if r["perturbed"] else ""
        print(f"  round {r['round']}: {bar} {r['trials']}"
              f"{'' if r['solved'] else ' (unsolved)'}{flag}")
    return results


# ---------------- evidence sweep ----------------

def evidence_runs(args) -> None:
    """The three runs that produce the chart, plus the chart.

    Run as subprocesses so each gets a genuinely clean module state — the
    skill library and memory are module-level caches, and carrying them
    between ablation arms is precisely the contamination the ablation
    exists to rule out.
    """
    runs = [
        ("baseline", ["--no-llm", "--no-memory"]),
        ("reflex_mem", ["--no-llm", "--reflex"]),
        ("llm_nomem", ["--no-memory"]),
        ("full", []),
    ]
    for run_id, flags in runs:
        print(f"\n{'=' * 62}\n  {run_id}\n{'=' * 62}", flush=True)
        subprocess.run(
            [sys.executable, __file__, "--wipe-memory", "--run-id", run_id,
             "--trials", str(args.trials), "--task", args.task,
             "--no-teaching"] + flags,
            check=False)
    subprocess.run([sys.executable, "chart.py"], check=False)


# ---------------- entry ----------------

# ---------------- self-test ----------------

def self_test() -> None:
    """Offline check of the harness's own logic.

    Everything else in this repo self-tests; this file did not, which is
    backwards — it is the integrator, it owns the two arithmetic bugs that
    were hardest to see (the grasp latch and compounding corrections), and
    it is the single writer of the file the other team reads.
    """
    import tempfile
    import arm_api

    print("=== _applied_offset reads the offset off at_block ===")
    plan = schema.fallback_plan(dx_cm=-4.2, dy_cm=2.8)
    assert _applied_offset(plan) == (-4.2, 2.8), _applied_offset(plan)
    assert _applied_offset({"steps": []}) == (0.0, 0.0)
    assert _applied_offset({"steps": [{"action": "grip",
                                       "params": {"state": "close"}}]}) == (0.0, 0.0)
    print("  ok")

    print("\n=== corrections COMPOUND, they do not reset ===")
    # Trial 1 applies nothing and measures the full bias as residual.
    p1 = schema.fallback_plan()
    n1 = _next_offset(p1, {"grasp_error_cm": (4.2, -2.8)})
    assert n1 == (-4.2, 2.8), n1
    # Trial 2 applies that, and a small residual REMAINS. The new offset
    # must be the old one adjusted — not the residual on its own, which
    # would throw away the correction and make the arm oscillate.
    p2 = schema.fallback_plan(dx_cm=n1[0], dy_cm=n1[1])
    n2 = _next_offset(p2, {"grasp_error_cm": (0.4, -0.3)})
    assert n2 == (-4.6, 3.1), f"expected (-4.6, 3.1), got {n2}"
    assert abs(n2[0]) > abs(n1[0]), "the correction must accumulate"
    print(f"  trial1 -> {n1}   trial2 -> {n2}   (accumulated, not reset)")
    assert _next_offset(p1, {"grasp_error_cm": None}) is None
    assert _next_offset(p1, {}) is None
    print("  ok")

    print("\n=== the grasp offset is latched at CLOSE, not at rest ===")
    arm = arm_api.MockArm()
    for step in schema.fallback_plan()["steps"]:
        pr = step["params"]
        if step["action"] == "move_to":
            arm.move_to(pr["pose"], pr["dx_cm"], pr["dy_cm"], pr["dz_cm"])
        else:
            arm.grip(pr["state"], pr.get("strength", 80))
    scene = arm.get_scene()
    verdict = critic.critique(arm.get_scene(), scene,
                              schema.fallback_plan(), use_llm=False)
    residual = verdict["grasp_error_cm"]
    true_bias = arm_api.MockArm.PERCEPTION_BIAS
    print(f"  measured {tuple(round(v, 1) for v in residual)}  "
          f"true bias {true_bias}")
    assert abs(residual[0] - true_bias[0]) < 1.0 and \
           abs(residual[1] - true_bias[1]) < 1.0, \
        ("the measured offset must match the real bias. If this fails, the "
         "critic is reading the gripper's PARKED position again and every "
         "correction will have the wrong sign.")
    # And the correction it implies must actually solve the task.
    nxt = _next_offset(schema.fallback_plan(), verdict)
    arm2 = arm_api.MockArm()
    before2 = arm2.get_scene()
    fixed = schema.fallback_plan(dx_cm=nxt[0], dy_cm=nxt[1])
    for step in fixed["steps"]:
        pr = step["params"]
        if step["action"] == "move_to":
            arm2.move_to(pr["pose"], pr["dx_cm"], pr["dy_cm"], pr["dz_cm"])
        else:
            arm2.grip(pr["state"], pr.get("strength", 80))
    v2 = critic.critique(before2, arm2.get_scene(), fixed, use_llm=False)
    print(f"  applying {nxt} -> passed={v2['passed']} err={v2['error_cm']}cm")
    assert v2["passed"], "the derived correction must actually solve the task"
    print("  ok")

    print("\n=== execute() drops invalid steps and runs the rest ===")
    arm3 = arm_api.MockArm()
    ran = execute(arm3, {"steps": [
        {"action": "move_to", "params": {"pose": "above_block"}},
        {"action": "forward", "params": {"meters": 1.0}},      # stale Go2
        {"action": "move_to", "params": {"pose": "at_the_pub"}},  # invented
        {"action": "grip", "params": {"state": "close"}},
    ]})
    assert ran == ["move_to", "grip"], ran
    print(f"  ran {ran}, skipped 2 bad steps without raising")

    print("\n=== emit_red_to_blue writes atomically and completely ===")
    global SHARED_MSG
    saved = SHARED_MSG
    SHARED_MSG = os.path.join(tempfile.mkdtemp(), "red_to_blue.json")
    payload = {"ts": 1.0, "trial": 1, "prompt_update": "move 2cm left",
               "reason": "missed the grasp", "confidence": 0.6,
               "replay_skill": None, "steps": []}
    emit_red_to_blue(payload)
    with open(SHARED_MSG) as f:
        assert json.load(f) == payload, "round trip lost data"
    leftovers = [p for p in os.listdir(os.path.dirname(SHARED_MSG))
                 if p.endswith(".tmp")]
    assert not leftovers, f"temp files left behind: {leftovers}"
    print(f"  round-tripped, no .tmp left behind")

    print("\n=== a queued replay merges without clobbering Reasoning ===")
    # teach.py's contract: it queues, we merge and emit. Reasoning's own
    # prompt_update / reason / confidence must survive the merge.
    teach._pending_replay = {"replay_skill": "human_taught_grasp"}
    msg = dict(payload)
    pending = teach.take_pending_replay()
    if pending:
        msg.update(pending)
    emit_red_to_blue(msg)
    with open(SHARED_MSG) as f:
        out = json.load(f)
    assert out["replay_skill"] == "human_taught_grasp"
    assert out["prompt_update"] == "move 2cm left", "merge dropped the directive"
    assert out["reason"] == "missed the grasp", "merge overwrote the reason"
    assert out["confidence"] == 0.6, "merge overwrote the confidence"
    assert teach.take_pending_replay() is None, "consume-once failed"
    SHARED_MSG = saved
    print("  ok")

    print("\n=== escalation stays silent when it cannot actually ask ===")
    spoken = []
    real_speak, narrate.speak = narrate.speak, lambda t, **k: spoken.append(t)
    try:
        got = escalate(arm_api.MockArm(), "task", {"failure_mode": "missed_grasp"},
                       [], use_voice=False, allow_teaching=False)
        assert got == {"replay": None, "view": None}, got
        assert not spoken, \
            f"must not promise help it cannot deliver, said: {spoken}"
        print("  --no-teaching + no voice -> silent, {replay: None, view: None}")

        # Every escalation path must return both keys. A caller that reads
        # .get("view") on a None would crash the run at the exact moment a
        # human is trying to help, which is the worst possible time.
        os.environ["GLASSES_BACKEND"] = "off"
        import importlib
        import glasses as g
        importlib.reload(g)
        got2 = escalate(arm_api.MockArm(), "task",
                        {"failure_mode": "collision"}, [],
                        use_voice=False, allow_teaching=True)
        assert set(got2) == {"replay", "view"}, got2
        print(f"  glasses off + teaching on -> keys {sorted(got2)}")
    finally:
        narrate.speak = real_speak

    print("\n=== a full run returns the documented shape ===")
    memory.wipe()
    result = run_task(arm_api.MockArm(), "pick up the red block", max_trials=3,
                      use_llm=False, use_reflex=True, allow_teaching=False,
                      run_id="selftest")
    assert set(result) >= {"solved", "trials"}, result
    assert result["solved"] and result["trials"] == 2, result
    print(f"  {result}")

    print("\nloop self-test passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task",
                    default="pick up the red block and place it in the target zone")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--surface", default="table")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-memory", action="store_true",
                    help="ablation: no failure retrieval")
    ap.add_argument("--no-llm", action="store_true",
                    help="baseline: fixed plan every trial, no model")
    ap.add_argument("--reflex", action="store_true",
                    help="closed-loop correction from memory, no model "
                         "(the honest baseline for 'it learns')")
    ap.add_argument("--no-teaching", action="store_true",
                    help="disable the demonstration escalation")
    ap.add_argument("--voice", action="store_true",
                    help="human speaks the task and mid-run corrections")
    ap.add_argument("--wipe-memory", action="store_true")
    ap.add_argument("--rounds", type=int, default=1,
                    help="attempt the task N times over, keeping memory "
                         "(this is the demo shape)")
    ap.add_argument("--perturb-at", type=int, default=0,
                    help="perturb the world at the start of ROUND N "
                         "(the adapt-to-change demo moment)")
    ap.add_argument("--perturb-kind", default="calibration",
                    choices=["calibration", "zone", "both"],
                    help="calibration drift invalidates what it learned "
                         "(use this on stage); zone move it absorbs silently")
    ap.add_argument("--evidence", action="store_true",
                    help="run all four ablation arms and build the chart")
    ap.add_argument("--self-test", action="store_true",
                    help="offline check of the harness's own logic")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if args.evidence:
        evidence_runs(args)
        return

    if args.wipe_memory:
        memory.wipe()

    task = args.task
    if args.voice:
        try:
            import glasses
            print(f"[loop] glasses: {json.dumps(glasses.pair())}")
            spoken = glasses.listen_for_task(timeout_s=8)
            if spoken:
                task = spoken
                print(f"[loop] task from voice: {task!r}")
            else:
                print(f"[loop] nothing heard; using --task")
        except Exception as e:                              # noqa: BLE001
            print(f"[loop] voice unavailable: {e}")

    run_id = args.run_id or (
        f"{'nomem' if args.no_memory else 'mem'}"
        f"{'_reflex' if args.reflex else '_nollm' if args.no_llm else ''}"
        f"_{int(time.time())}")

    arm = get_arm()
    common = dict(
        max_trials=args.trials,
        use_memory=not args.no_memory, use_llm=not args.no_llm,
        use_reflex=args.reflex, use_voice=args.voice,
        allow_teaching=not args.no_teaching, surface=args.surface)
    try:
        if args.rounds > 1 or args.perturb_at:
            results = run_rounds(arm, task, args.rounds,
                                 perturb_round=args.perturb_at,
                                 perturb_kind=args.perturb_kind,
                                 run_id=run_id, **common)
            print(f"\n══ {run_id}: {json.dumps(results)}")
        else:
            result = run_task(arm, task, run_id=run_id, **common)
            print(f"\n══ {run_id}: {json.dumps(result)}")
    finally:
        arm.stop()


if __name__ == "__main__":
    main()
