<div align="center">
  <h1>The Arm That Knows It Failed</h1>
</div>

<div align="center">
  <h3>A robot arm that fails at a task, says out loud why it failed, and
  gets it right on the next attempt — no retraining, no teleoperation.</h3>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/arm-SO--101-black" alt="Arm: SO-101">
  <img src="https://img.shields.io/badge/platform-macOS-black" alt="Platform: macOS">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/inference-Nebius-6f42c1" alt="Inference: Nebius">
  <img src="https://img.shields.io/badge/training-none-brightgreen" alt="No training">
</div>

<br>

An SO-101 arm tries to pick up a block and place it in a target zone. A
fixed camera measures what actually happened and decides pass or fail. On a
failure a vision-language model explains *why* — "I closed my gripper two
centimetres short of the block" — the arm speaks that diagnosis aloud, and
the explanation is rewritten into a new instruction for the controller. The
next attempt is a different attempt. Nothing is retrained; the improvement
lives entirely in the prompt and in a memory of past failures.

When it fails twice in a row it stops guessing and asks for help. A human
moves the **leader** arm through the motion by hand, that trajectory is
saved as a named skill, and the arm replays it.

```
camera  →  sensing  →  reasoning  →  prompt update  →  VLA  →  arm moves
(fixed)   (pass/fail)  (why + fix)     (Red→Blue)     (Blue)      │
                            ↑                                     │
                         memory  ←──── failure recorded ←──────────┘
                     (past failures shape the next prompt)

    after 2 failures:  human moves the leader arm  →  skill saved  →  replayed
```

Two rules hold the whole thing up, and both are enforced in code rather
than trusted:

- **The camera decides pass/fail — the model only explains why.** A critic
  that hallucinates success stops the learning loop and makes the arm look
  broken on stage. `reason.diagnose()` refuses to run on a passing verdict,
  strips any success-like key out of the model's reply, and returns a dict
  with no pass/fail field at all.
- **The model emits JSON only, against a fixed schema.** Non-JSON, an
  unknown failure mode, an empty diagnosis, or a `prompt_update` containing
  code is discarded and a geometry-only fallback is used instead. The
  demo never depends on a model behaving.

The self-improvement is measurable, not asserted: `memory.wipe()` is an
ablation switch, so the same task can be run with and without memory and
the trials-to-success compared.

---

## Quick start (Team B, right now)

```bash
pip install openai opencv-python numpy pyttsx3 matplotlib
export NEBIUS_API_KEY=...          # DO THIS FIRST. Verify it works.
export ROBOT_BACKEND=temp          # fake robot until Team A ships mock

python planner.py                  # standalone smoke test — Rikin
python loop.py --no-llm            # harness runs with zero AI — Benji
python narrate.py                  # TTS check — Aaryan
```

`loop.py --no-llm` completing 5 trials **is the milestone.** Everything
after is swapping stubs for real parts.

---

## Ownership

| File | Owner | What it is |
|---|---|---|
| `robot_api.py` | Benji (contract) / Team A (impls) | **FROZEN.** 5 methods. The only surface between teams. |
| `schema.py` | Benji | Skill library + strict JSON validator |
| `loop.py` | Benji | The harness. Plan → execute → critique → remember |
| `planner.py` | Rikin | Nebius call, prompt, JSON extraction, fallback |
| `critic.py` | Rikin + Benji | Hybrid verdict: geometry passes/fails, VLM diagnoses |
| `memory.py` | Aaryan | Failure store, keyword retrieval, ablation wipe |
| `narrate.py` | Aaryan | TTS + human escalation (phone photo / glasses) |
| `chart.py` | Aaryan → Team A | The evidence slide |
| `mock_robot.py` | **Team A** | MockSportClient with slip + ArUco pose |
| `go2_robot.py` | **Team A** | Real Go2 via go2-webrtc or SportClient |

---

## The two non-negotiable design rules

**1. The LLM never emits code.** It emits JSON matching `schema.PRIMITIVES`.
Invalid JSON → one retry → geometric fallback. The demo never crashes
because a model got creative.

**2. Geometry decides pass/fail. The VLM only decides *why*.**
A VLM asked "did it succeed?" will sometimes say yes when it failed. That
stops the learning loop and the robot looks broken on stage. Never add a
code path where the model returns a boolean.

Disagreements between VLM diagnosis and heuristic diagnosis are appended to
`logs/critic_disagreements.jsonl` — that file is the **Toloka payload**.
Ship 30–50 of them for human labels, report critic accuracy as a number.

---

## Team A: what you must implement

```python
# mock_robot.py
class MockSportClient(RobotBase):
    """Same 5 methods. Inject slip ~0.75 and turn drift ~2°.
    get_pose() should read ArUco when the camera sees markers,
    and fall back to dead reckoning when it doesn't."""

# go2_robot.py
class Go2Robot(RobotBase):
    """Go2 Move() is a VELOCITY command that persists until replaced.
    forward(m) = Move(v,0,0); sleep(m/v); Move(0,0,0)
    ALWAYS send the stop. Add a 2s watchdog that zeroes velocity."""
```

Then Team B changes nothing but `ROBOT_BACKEND=go2`.

---

## Evidence runs (do these before the freeze)

```bash
python loop.py --wipe-memory --no-llm    --run-id baseline    # no learning
python loop.py --wipe-memory --no-memory --run-id llm_nomem   # LLM, no memory
python loop.py --wipe-memory             --run-id full        # everything
python chart.py
```

If `full` doesn't beat `baseline`, put that on the slide honestly. A team
that reports "the LLM added interpretability but no accuracy over geometric
regression" reads as serious. Nobody else will do it.

---

## Gates — write these on the whiteboard now

| When | Check | If it fails |
|---|---|---|
| T−1:00 | Glasses streaming a frame into Python? | **Cut.** `ESCALATE_BACKEND=phone` |
| T+0:30 | Go2 standing from our code? | **Demo the mock.** Still a real system |
| T+1:30 | — | **HARD FREEZE.** Record backup video. Rehearse only |

**Integrator:** Benji. Only person pushing to `main` after T+0:30.
**Safety:** one person on the remote, thumb on damp, that is their only job.
3m taped perimeter, nobody inside it while the robot moves.

---

## Demo script (90 seconds)

1. Task 1. It fails. **It says why out loud.** It retries. It succeeds.
2. Task 2. Fewer attempts — it reuses a skill learned in Task 1.
3. **Someone throws a towel across the floor mid-run.** It fails, notices
   the physics changed, adapts.
4. Show `logs/evidence.png`: trials-to-success, with and without memory.

Step 3 is what they'll remember. Step 4 is what makes it defensible.

---

## Env vars

```
NEBIUS_API_KEY      required
NEBIUS_MODEL        default Qwen/Qwen2.5-VL-72B-Instruct — VERIFY THIS
NEBIUS_VISION       1|0    send frames to the planner
ROBOT_BACKEND       temp|mock|go2
SUCCESS_CM          12     pass threshold
ESCALATE_BACKEND    phone|glasses|off
HELP_DIR            help_frames/   drop a photo here to help the robot
```
