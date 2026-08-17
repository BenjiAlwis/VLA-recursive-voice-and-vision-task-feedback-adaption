"""
THE HARNESS.  Owner: Benji.

Run this FIRST with everything stubbed. When it completes 5 trials and
writes logs/trials.jsonl, the product exists — everything after is
swapping stubs for real components.

    python loop.py --task "reach the target" --trials 5
    python loop.py --no-memory        # ablation control run
    python loop.py --no-llm           # geometric baseline

The three runs above produce the chart that is our actual evidence.
"""
import argparse
import json
import os
import time
from typing import Dict, List

import critic
import memory
import narrate
import planner
import schema
from robot_api import get_robot

TRIAL_LOG = "logs/trials.jsonl"
MAX_STEPS_GUARD = 6


def execute(robot, plan: Dict) -> None:
    """Run a validated plan. Stop between steps — never queue motion."""
    for step in plan["steps"][:MAX_STEPS_GUARD]:
        action, params = step["action"], step["params"]
        print(f"  ▸ {action}({params})")
        if action == "forward":
            robot.forward(params["meters"])
        elif action == "turn":
            robot.turn(params["degrees"])
        robot.stop()
        time.sleep(0.3)


def log_trial(rec: Dict) -> None:
    os.makedirs("logs", exist_ok=True)
    with open(TRIAL_LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def run_task(robot, task: str, target: Dict, max_trials: int = 5,
             use_memory: bool = True, use_llm: bool = True,
             surface: str = "hardfloor", run_id: str = "default") -> Dict:

    consecutive_failures = 0
    help_frame = None

    for trial in range(1, max_trials + 1):
        print(f"\n─── trial {trial}/{max_trials} — {task}")
        narrate.announce_attempt(trial, task)

        pose_before = robot.get_pose()
        frame_before = robot.get_front_frame()

        mems = memory.for_prompt(task, surface) if use_memory else "(memory disabled)"

        if use_llm:
            plan = planner.plan(task, pose_before, target, mems,
                                frame=help_frame or frame_before)
        else:
            plan = planner.fallback_plan(pose_before, target)
        help_frame = None
        print(f"  plan: {plan['rationale']}")

        execute(robot, plan)

        pose_after = robot.get_pose()
        frame_after = robot.get_front_frame()
        verdict = critic.critique(pose_before, pose_after, target, plan,
                                  frame_before, frame_after)

        log_trial({
            "run_id": run_id, "ts": time.time(), "task": task, "trial": trial,
            "use_memory": use_memory, "use_llm": use_llm, "surface": surface,
            "actions": [s["action"] for s in plan["steps"]],
            "rationale": plan["rationale"],
            "error_cm": verdict["error_cm"], "passed": verdict["passed"],
            "failure_mode": verdict.get("failure_mode"),
            "diagnosis": verdict.get("diagnosis"),
        })

        if verdict["passed"]:
            print(f"  ✅ PASS  error={verdict['error_cm']}cm")
            narrate.announce_success(trial, verdict["error_cm"])
            for s in plan["steps"]:
                schema.bump_success(s["action"])
            if plan.get("new_skill") and schema.add_skill(
                    {**plan["new_skill"], "learned_from": f"{task}/trial{trial}"}):
                narrate.announce_learned(plan["new_skill"]["name"])
            return {"solved": True, "trials": trial,
                    "error_cm": verdict["error_cm"]}

        print(f"  ❌ FAIL  error={verdict['error_cm']}cm  "
              f"mode={verdict['failure_mode']}")
        narrate.announce_failure(verdict)

        if use_memory:
            memory.record(task, plan, verdict["error_cm"],
                          verdict["failure_mode"], verdict["diagnosis"], surface)

        consecutive_failures += 1
        if consecutive_failures >= 2:
            help_frame = narrate.request_help(
                f"I keep getting {verdict['failure_mode']}.")
            consecutive_failures = 0

    return {"solved": False, "trials": max_trials}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="walk to the target marker and stop on it")
    ap.add_argument("--target-x", type=float, default=150.0)
    ap.add_argument("--target-y", type=float, default=0.0)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--surface", default="hardfloor")
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--wipe-memory", action="store_true",
                    help="clear failure memory before running (ablation)")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    if args.wipe_memory:
        memory.wipe()
        print("memory wiped (ablation run)")

    run_id = args.run_id or (
        f"{'nomem' if args.no_memory else 'mem'}"
        f"{'_nollm' if args.no_llm else ''}_{int(time.time())}")

    robot = get_robot()
    target = {"x_cm": args.target_x, "y_cm": args.target_y}

    result = run_task(robot, args.task, target, args.trials,
                      use_memory=not args.no_memory,
                      use_llm=not args.no_llm,
                      surface=args.surface, run_id=run_id)

    print(f"\n══ {run_id}: {result}")
    robot.stop()


if __name__ == "__main__":
    main()
