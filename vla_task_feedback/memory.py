"""
Failure memory + retrieval.  Owner: Aaryan (RED team).

Every failed attempt is written here with the camera's measurement and the
VLM's diagnosis. Retrieval feeds relevant past failures back into the next
prompt, which is the mechanism by which the arm stops repeating itself.

Design note: retrieval is keyword overlap, NOT embeddings. Embeddings add
an API call (latency) and a failure mode (an outage takes memory down with
it) for maybe 5% better recall over a corpus of ~20 entries. At this size
keyword overlap is fine. If there is time left over, swap in pgvector —
the interface below will not change.

wipe() is the ABLATION SWITCH. The chart comparing trials-to-success with
and without memory is our actual evidence that learning happened, and it
requires being able to zero this store between runs.
"""
import json
import os
import re
import time
from typing import Any, Dict, List

MEM_PATH = os.getenv("MEMORY_PATH", "logs/failures.json")

# The camera measures the failure; the VLM picks one of these to explain it.
FAILURE_MODES = ["missed_grasp", "dropped_early", "wrong_position",
                 "collision", "no_motion"]

# Words too common to carry signal. Note "target" and "block" are NOT in
# here: in a pick-and-place task the target zone is a meaningful
# distinction between one task and another, so stripping it would collapse
# "place it in the target zone" and "place it beside the block" into the
# same query.
STOP = {"the", "a", "an", "to", "of", "and", "is", "was", "it", "on",
        "at", "by", "for", "with", "i", "my", "then", "that", "arm"}

RECENCY_WINDOW_S = 600.0        # 10 min — about one hackathon session


def _tokens(text: Any) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", str(text).lower())
            if w not in STOP and len(w) > 2}


def _normalise_actions(actions: Any) -> List[str]:
    """Accept a list of names, a list of step dicts, or a whole plan dict.

    Benji may hand this either the raw action list or the plan it came
    from; losing a failure record to a shape mismatch would be a bad trade.
    """
    if isinstance(actions, dict):
        actions = actions.get("steps", [])
    if not isinstance(actions, (list, tuple)):
        return [str(actions)] if actions else []
    out = []
    for a in actions:
        out.append(str(a.get("action", a)) if isinstance(a, dict) else str(a))
    return out


def load() -> List[Dict]:
    """Every stored failure. [] if the store is missing or unreadable."""
    try:
        with open(MEM_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except FileNotFoundError:
        return []
    except Exception as e:                                  # noqa: BLE001
        print(f"[memory] store unreadable, treating as empty: {e}")
        return []


def _save(entries: List[Dict]) -> bool:
    """Atomic write — a crash mid-save must not truncate the whole store."""
    try:
        parent = os.path.dirname(MEM_PATH) or "."
        os.makedirs(parent, exist_ok=True)
        tmp = f"{MEM_PATH}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(entries, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, MEM_PATH)
        return True
    except Exception as e:                                  # noqa: BLE001
        print(f"[memory] could not save: {e}")
        return False


def record(task: str, actions: Any, error_cm: float, failure_mode: str,
           diagnosis: str, surface: str = "unknown") -> Dict:
    """Write one failure. Called after the camera has judged a trial failed.

    Returns the entry even if the save failed — the caller should not lose
    the diagnosis just because the disk did.
    """
    if failure_mode not in FAILURE_MODES:
        # Stored anyway. An unexpected label is still a real data point and
        # discarding it would silently lose the trial.
        print(f"[memory] unexpected failure_mode {failure_mode!r} "
              f"(expected one of {FAILURE_MODES})")

    entry = {
        "ts": time.time(),
        "task": str(task),
        "surface": str(surface),
        "actions": _normalise_actions(actions),
        "error_cm": round(float(error_cm), 1),
        "failure_mode": failure_mode,
        "diagnosis": str(diagnosis),
    }
    entries = load()
    entries.append(entry)
    _save(entries)
    return entry


def retrieve(task: str, surface: str = "unknown", k: int = 3) -> List[Dict]:
    """The k most relevant past failures.

    Scored on task-word overlap, same-surface match, and recency. Recency
    matters more than it looks: the arm gets recalibrated and the block
    gets moved between runs, so a failure from forty minutes ago may
    describe a world that no longer exists.
    """
    entries = load()
    if not entries:
        return []

    query = _tokens(task)
    now = time.time()
    scored = []
    for e in entries:
        overlap = len(query & _tokens(f"{e.get('task', '')} "
                                      f"{e.get('diagnosis', '')}"))
        score = overlap * 2.0
        if e.get("surface") == surface and surface != "unknown":
            score += 3.0
        age = max(0.0, now - e.get("ts", now))
        score += max(0.0, 2.0 - 2.0 * age / RECENCY_WINDOW_S)
        # Ties broken toward the more recent entry.
        scored.append((score, e.get("ts", 0.0), e))

    scored.sort(key=lambda p: (-p[0], -p[1]))
    return [e for _, _, e in scored[:max(0, k)]]


def for_prompt(task: str, surface: str = "unknown", k: int = 3) -> str:
    """Retrieved failures, rendered for the VLM prompt."""
    hits = retrieve(task, surface, k)
    if not hits:
        return "(no relevant past failures)"
    return "\n".join(
        f"- tried {h['actions']} -> {h['failure_mode']}, "
        f"off by {h['error_cm']}cm. {h['diagnosis']}"
        for h in hits
    )


def wipe() -> None:
    """ABLATION SWITCH. Call before a no-memory control run."""
    _save([])
    print("[memory] wiped")


def stats() -> Dict:
    """Totals for the evidence chart."""
    entries = load()
    modes: Dict[str, int] = {}
    for e in entries:
        mode = e.get("failure_mode", "unknown")
        modes[mode] = modes.get(mode, 0) + 1
    errors = [e["error_cm"] for e in entries if "error_cm" in e]
    return {
        "total": len(entries),
        "by_mode": modes,
        "mean_error_cm": round(sum(errors) / len(errors), 1) if errors else None,
        "worst_error_cm": max(errors) if errors else None,
    }


if __name__ == "__main__":
    MEM_PATH = "logs/failures_smoke.json"       # never touch the real store
    wipe()

    record("pick up the red block and place it in the target zone",
           ["reach", "grasp"], 7.4, "missed_grasp",
           "I closed the gripper two centimetres short of the block.",
           surface="hardfloor")
    record("pick up the red block and place it in the target zone",
           [{"action": "reach"}, {"action": "lift"}], 12.1, "dropped_early",
           "I let go before reaching the zone; my grip was too loose.",
           surface="hardfloor")
    record("stack the block on the tower",
           {"steps": [{"action": "reach"}]}, 3.2, "collision",
           "I clipped the tower on the way in.", surface="carpet")
    record("push the block", ["push"], 40.0, "sideways_slide",
           "Deliberately invalid mode — must warn but still store.",
           surface="carpet")

    print("\nstats:", json.dumps(stats(), indent=2))

    print("\nfor_prompt (same task, same surface):")
    print(for_prompt("pick up the red block and place it in the target zone",
                     surface="hardfloor"))

    print("\nfor_prompt (unrelated task):")
    print(for_prompt("open the drawer", surface="unknown"))

    assert stats()["total"] == 4, "invalid failure_mode must still be stored"
    hits = retrieve("pick up the red block and place it in the target zone",
                    surface="hardfloor", k=2)
    assert len(hits) == 2, hits
    assert all(h["surface"] == "hardfloor" for h in hits), \
        f"surface + overlap should win: {[h['surface'] for h in hits]}"
    assert _normalise_actions({"steps": [{"action": "reach"}]}) == ["reach"]
    assert _normalise_actions(["a", "b"]) == ["a", "b"]
    assert _normalise_actions(None) == []
    # "target" must remain a real token, else the two task phrasings below
    # would score identically.
    assert "target" in _tokens("place it in the target zone")

    wipe()
    assert load() == [] and stats()["total"] == 0
    assert for_prompt("anything") == "(no relevant past failures)"
    os.remove(MEM_PATH)
    print("\nall memory assertions passed")
