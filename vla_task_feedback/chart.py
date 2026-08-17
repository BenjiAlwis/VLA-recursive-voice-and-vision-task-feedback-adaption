"""
The evidence slide.  Owner: Aaryan (hand to Team A for the dashboard).

Reads logs/trials.jsonl and produces the one chart that proves the claim.

Run all three, then chart:
    python loop.py --wipe-memory --no-llm  --run-id baseline
    python loop.py --wipe-memory --no-memory --run-id llm_nomem
    python loop.py --wipe-memory           --run-id full
    python chart.py

If the 'full' line is not below the others, SAY SO ON THE SLIDE.
Reporting a negative result reads as serious. Roughly nobody else will.
"""
import json
import os
from collections import defaultdict

LOG = "logs/trials.jsonl"


def load():
    if not os.path.exists(LOG):
        return []
    with open(LOG) as f:
        return [json.loads(l) for l in f if l.strip()]


def summarise():
    runs = defaultdict(list)
    for r in load():
        runs[r["run_id"]].append(r)

    print(f"\n{'run':<24}{'trials':>8}{'solved':>8}{'first_err':>11}{'final_err':>11}")
    print("─" * 62)
    out = {}
    for run_id, trials in runs.items():
        trials.sort(key=lambda t: t["trial"])
        solved = any(t["passed"] for t in trials)
        n = next((t["trial"] for t in trials if t["passed"]), len(trials))
        print(f"{run_id:<24}{n:>8}{str(solved):>8}"
              f"{trials[0]['error_cm']:>11.1f}{trials[-1]['error_cm']:>11.1f}")
        out[run_id] = {"trials_to_success": n, "solved": solved,
                       "errors": [t["error_cm"] for t in trials]}
    return out


def plot(data):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed — table above is enough for the slide)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for run_id, d in data.items():
        ax1.plot(range(1, len(d["errors"]) + 1), d["errors"],
                 marker="o", label=run_id)
    ax1.set_xlabel("attempt")
    ax1.set_ylabel("error (cm)")
    ax1.set_title("Error per attempt")
    ax1.axhline(12, ls="--", c="grey", lw=1)
    ax1.legend(fontsize=8)

    names = list(data)
    ax2.bar(names, [data[n]["trials_to_success"] for n in names])
    ax2.set_ylabel("attempts needed")
    ax2.set_title("Trials to success (lower = learned more)")
    ax2.tick_params(axis="x", rotation=20, labelsize=8)

    plt.tight_layout()
    plt.savefig("logs/evidence.png", dpi=140)
    print("\nwrote logs/evidence.png")


if __name__ == "__main__":
    plot(summarise())
