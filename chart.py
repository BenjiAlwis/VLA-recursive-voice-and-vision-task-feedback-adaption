"""
The evidence slide.  Owner: Aaryan (hand to Team A for the dashboard).

Reads logs/trials.jsonl and produces the one chart that makes the claim
falsifiable: attempts-to-success, with and without each component.

    python loop.py --evidence        # runs every arm, then calls this

THE ARMS, and what each one isolates:

    baseline     fixed plan, no memory, no model.   Must never improve.
    reflex_mem   memory + arithmetic, no model.     Does memory alone work?
    llm_nomem    model, no memory.                  Does the model alone work?
    full         model + memory.                    The system as pitched.

reflex_mem is the arm that makes this honest. "Our robot learns from its
failures" is a much weaker claim if a regex over the same measurements
learns just as fast — so we measure that directly instead of hoping nobody
asks. If `full` does not beat `reflex_mem`, PUT THAT ON THE SLIDE. A team
that reports "the LLM added interpretability and language grounding but no
accuracy over closed-form correction" reads as serious, and roughly nobody
else will do it.
"""
import json
import os
from collections import defaultdict

LOG = os.getenv("TRIAL_LOG", "logs/trials.jsonl")
OUT = os.getenv("EVIDENCE_PNG", "logs/evidence.png")

# Display order and colour, so the arms read consistently everywhere.
ARMS = [
    ("baseline",   "#9aa0a6", "no memory, no model"),
    ("reflex_mem", "#e8a33d", "memory only"),
    ("llm_nomem",  "#5b8ff9", "model only"),
    ("full",       "#2ca02c", "memory + model"),
]


def load():
    if not os.path.exists(LOG):
        return []
    rows = []
    with open(LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue          # a truncated final line is not fatal
    return rows


def summarise():
    runs = defaultdict(list)
    for r in load():
        runs[r.get("run_id", "unknown")].append(r)

    if not runs:
        print(f"no trials in {LOG} — run `python loop.py --evidence` first")
        return {}

    print(f"\n{'run':<14}{'label':<22}{'attempts':>9}{'solved':>8}"
          f"{'first_err':>11}{'final_err':>11}")
    print("─" * 75)

    known = {name: (colour, label) for name, colour, label in ARMS}
    order = [n for n, _, _ in ARMS if n in runs] + \
            [n for n in runs if n not in known]

    out = {}
    for run_id in order:
        trials = sorted(runs[run_id], key=lambda t: t.get("trial", 0))
        solved = any(t.get("passed") for t in trials)
        n = next((t["trial"] for t in trials if t.get("passed")), len(trials))
        errors = [t.get("error_cm", 0.0) for t in trials]
        label = known.get(run_id, ("", ""))[1]
        planners = {t.get("planned_by", "fallback") for t in trials}
        print(f"{run_id:<14}{label:<22}{n:>9}{str(solved):>8}"
              f"{errors[0]:>11.1f}{errors[-1]:>11.1f}")
        out[run_id] = {"attempts": n, "solved": solved, "errors": errors,
                       "label": label, "planners": planners}

    _verdict(out)
    return out


def _verdict(data):
    """State plainly whether the headline claim survived. Reporting a
    negative result here is worth more than a chart that quietly hides it."""
    full = data.get("full")
    reflex = data.get("reflex_mem")
    base = data.get("baseline")

    print()

    # An arm labelled "model" that never reached the model is not a result,
    # it is a broken run. Say so loudly rather than letting a keyless sweep
    # be screenshotted as evidence that the LLM did not help.
    for name in ("full", "llm_nomem"):
        arm = data.get(name)
        if arm and "nebius" not in arm.get("planners", set()):
            print(f"⚠  '{name}' never reached Nebius (planned by "
                  f"{sorted(arm.get('planners', {'?'}))}). This row is NOT a "
                  f"result — set NEBIUS_API_KEY and re-run before using it.")

    if base and base["solved"]:
        print("⚠  the baseline SOLVED it — the task is too easy to show "
              "learning. Increase the perception bias or tighten SUCCESS_CM.")
    if full and reflex:
        if full["attempts"] < reflex["attempts"]:
            print(f"✅ full ({full['attempts']}) beat memory-only "
                  f"({reflex['attempts']}) — the model earned its place.")
        elif full["attempts"] == reflex["attempts"]:
            print(f"➖ full tied memory-only at {full['attempts']} attempts. "
                  f"SAY THIS ON THE SLIDE: the LLM added language grounding "
                  f"and interpretability, not accuracy.")
        else:
            print(f"❌ memory-only ({reflex['attempts']}) beat full "
                  f"({full['attempts']}). Report it. It is a real finding.")
    elif full or reflex:
        print("(run `python loop.py --evidence` for the full comparison)")


def plot(data):
    if not data:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed — the table above is enough)")
        return

    known = {name: colour for name, colour, _ in ARMS}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

    for run_id, d in data.items():
        ax1.plot(range(1, len(d["errors"]) + 1), d["errors"],
                 marker="o", linewidth=2, color=known.get(run_id),
                 label=f"{run_id} — {d['label']}" if d["label"] else run_id)
    ax1.axhline(float(os.getenv("SUCCESS_CM", "5.0")), ls="--",
                c="grey", lw=1)
    ax1.annotate("success threshold", (0.02, 0.06), xycoords="axes fraction",
                 fontsize=7, color="grey")
    ax1.set_xlabel("attempt")
    ax1.set_ylabel("placement error (cm)")
    ax1.set_title("Error per attempt")
    ax1.legend(fontsize=7)

    names = list(data)
    ax2.bar(names, [data[n]["attempts"] for n in names],
            color=[known.get(n, "#777") for n in names])
    for i, n in enumerate(names):
        ax2.text(i, data[n]["attempts"] + 0.05,
                 "solved" if data[n]["solved"] else "never",
                 ha="center", fontsize=7)
    ax2.set_ylabel("attempts needed")
    ax2.set_title("Attempts to success (lower = learned more)")
    ax2.tick_params(axis="x", rotation=15, labelsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    plt.savefig(OUT, dpi=140)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    plot(summarise())
