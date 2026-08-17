"""
Failure memory + retrieval.  Owner: Aaryan.

Design note: retrieval is keyword-overlap, NOT embeddings.
Embeddings add an API call (latency) and a dependency (risk) for maybe
5% better recall over a corpus of ~20 entries. At 20 entries, keyword
overlap is fine. If you finish early, swap in pgvector — the interface
below won't change.
"""
import json
import os
import re
import time
from typing import List, Dict

MEM_PATH = "logs/failures.json"

STOP = {"the", "a", "an", "to", "of", "and", "is", "was", "it", "on",
        "at", "by", "for", "with", "i", "my", "robot", "target"}


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower())
            if w not in STOP and len(w) > 2}


def load() -> List[Dict]:
    if not os.path.exists(MEM_PATH):
        return []
    with open(MEM_PATH) as f:
        return json.load(f)


def _save(entries: List[Dict]) -> None:
    os.makedirs(os.path.dirname(MEM_PATH), exist_ok=True)
    with open(MEM_PATH, "w") as f:
        json.dump(entries, f, indent=2)


def record(task: str, plan: Dict, error_cm: float,
           failure_mode: str, diagnosis: str, surface: str = "unknown") -> Dict:
    """Write one failure. Called by the loop after a failed trial."""
    entry = {
        "ts": time.time(),
        "task": task,
        "surface": surface,
        "actions": [s["action"] for s in plan.get("steps", [])],
        "error_cm": round(error_cm, 1),
        "failure_mode": failure_mode,
        "diagnosis": diagnosis,
    }
    entries = load()
    entries.append(entry)
    _save(entries)
    return entry


def retrieve(task: str, surface: str = "unknown", k: int = 3) -> List[Dict]:
    """Most relevant past failures. Scored on task-word overlap,
    surface match, and recency."""
    entries = load()
    if not entries:
        return []
    query = _tokens(task)
    now = time.time()
    scored = []
    for e in entries:
        overlap = len(query & _tokens(e["task"] + " " + e["diagnosis"]))
        score = overlap * 2.0
        if e.get("surface") == surface and surface != "unknown":
            score += 3.0
        score += max(0.0, 2.0 - (now - e["ts"]) / 600.0)   # recency, 10 min decay
        scored.append((score, e))
    scored.sort(key=lambda p: -p[0])
    return [e for _, e in scored[:k]]


def for_prompt(task: str, surface: str = "unknown", k: int = 3) -> str:
    """Render retrieved failures for the planner. Empty string if none."""
    hits = retrieve(task, surface, k)
    if not hits:
        return "(no relevant past failures)"
    return "\n".join(
        f"- attempted {h['actions']} -> {h['failure_mode']}, "
        f"off by {h['error_cm']}cm. {h['diagnosis']}"
        for h in hits
    )


def wipe() -> None:
    """ABLATION SWITCH. Call this before the no-memory control run.
    The chart comparing with/without is the proof that we learned."""
    _save([])


def stats() -> Dict:
    entries = load()
    modes = {}
    for e in entries:
        modes[e["failure_mode"]] = modes.get(e["failure_mode"], 0) + 1
    return {"total": len(entries), "by_mode": modes}
