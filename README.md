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

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# The milestone. Zero AI, zero hardware, proves the whole loop's mechanics.
.venv/bin/python loop.py --no-llm

# The loop actually learning, still with no API key and no hardware.
.venv/bin/python loop.py --reflex --rounds 5 --perturb-at 3

# The full system.
export NEBIUS_API_KEY=...
.venv/bin/python planner.py --verify        # DO THIS FIRST. See §Risks.
.venv/bin/python loop.py --rounds 3 --voice
```

Every module runs standalone as its own smoke test — `python critic.py`,
`python vision.py`, and so on. Nine of them, all passing, all offline.

---

## The two non-negotiable rules

**1. The LLM emits JSON from a fixed menu. Never code.**
It picks from `move_to` / `grip` / `replay_skill` over five named poses.
Invalid JSON → one retry with the validation error fed back → fixed
fallback. `schema.py` rejects unknown actions, invented poses, offsets
outside ±15cm, and skills that were never taught.

**2. The camera decides pass/fail. The model only decides *why*.**
`critic.geometric_verdict()` is the one function that no model call may
influence. A VLM asked "did it succeed?" sometimes says yes when it
failed — that stops the learning loop and looks broken on stage. The merge
in `critique()` spreads the geometric verdict **last**, so a model that
returns `passed: true` cannot override the camera. There is a test for
exactly that.

Every VLM/heuristic disagreement is appended to
`logs/critic_disagreements.jsonl` — that file is the **Toloka payload**.
Ship 30–50 for human labels and report critic accuracy as a number.

---

## Ownership

| File | Owner | What it is |
|---|---|---|
| `arm_api.py` | Benji | **THE CONTRACT.** 7 methods, `MockArm` + `SO101Arm`. |
| `vision.py` | Benji | ArUco ground truth in cm. The critic's only input. |
| `loop.py` | Benji | The harness. Sole writer of `shared/red_to_blue.json`. |
| `schema.py` | Rikin | Action menu + validator + the fixed fallback. |
| `planner.py` | Rikin | Nebius call, prompt, retry-with-error, reflex arm. |
| `reason.py` | Rikin | VLM diagnosis → `prompt_update`. Strips verdict keys. |
| `critic.py` | Rikin + Benji | Geometry passes/fails; delegates *why* to `reason`. |
| `memory.py` | Aaryan | Failure store, keyword retrieval, ablation wipe. |
| `narrate.py` | Aaryan | TTS routed to the glasses via `say -a`. |
| `voice.py` | Aaryan | Speech in, with a terminating fallback chain. |
| `glasses.py` | Aaryan | Human interface — audio routing + photo ingest. |
| `glasses_bridge.py` | Aaryan | Wearables SDK app side of the photo ingest. |
| `supervise.py` | Aaryan | Interval glasses-correction loop. |
| `teach.py` | Aaryan | Leader-arm demonstration recorder. Records only. |
| `chart.py` | Aaryan | The evidence slide. |
| `so101_yellow/` | Team Yellow | Real hardware bring-up: ports, motors, calibration. |

**Integrator:** Benji. Only person merging to `main` after T+0:30.

### Where the two reasoning paths meet

`critic.py` and `reason.py` were written in parallel and both diagnose
failures. They are **not** duplicates any more:

- `critic.geometric_verdict()` — the camera. The sole pass/fail authority.
- `reason.diagnose()` — the model. Refuses to run on a passing verdict,
  strips any success-like key out of the reply, returns no pass/fail field.

`critic.critique()` calls `reason.diagnose()` for the explanation and keeps
its own heuristic as the offline fallback. One camera, one explainer.

---

## How it actually learns

The mock arm has a **constant perception bias** — it believes the block is
~5cm from where it truly is, against a 3cm grasp tolerance. So the naive
plan misses 300/300 times, and the corrected plan succeeds 300/300. That
separation is deliberate: random noise would teach nothing and would make
the ablation chart flap.

The loop closes like this:

1. Gripper closes 4.1cm right and 2.9cm short of the block. **Measured**,
   at the moment it closed — not where the arm parked afterwards.
2. `critic` writes that as a sentence *and* as `dx_cm=-4.1 dy_cm=+2.9`.
3. `memory` stores it. Next attempt, retrieval hands it to the planner.
4. The planner applies the offset. It succeeds.

Corrections **compound** rather than reset — the measured error is a
residual on top of whatever offset was already applied, so the stored value
is `applied - residual`. Storing the raw residual makes the arm oscillate.

---

## Evidence — four arms, one of them adversarial to our own claim

```bash
python loop.py --evidence        # runs all four, then charts them
```

| arm | what it isolates |
|---|---|
| `baseline` | fixed plan, no memory, no model. Must never improve. |
| `reflex_mem` | memory + arithmetic, **no model**. |
| `llm_nomem` | model, no memory. |
| `full` | the system as pitched. |

`reflex_mem` exists to attack our own headline. "Our robot learns from
failure" is much weaker if a regex over the same measurements learns just
as fast — so we measure it instead of hoping nobody asks. **If `full` does
not beat `reflex_mem`, that goes on the slide.** `chart.py` prints the
comparison in plain language and will say so itself.

It also refuses to let a keyless run be mistaken for a result: if an arm
labelled "model" never actually reached Nebius, the chart says so.

---

## Demo (90 seconds)

```bash
python loop.py --rounds 5 --perturb-at 3 --voice
```

1. Round 1: it misses, **says why out loud**, corrects, succeeds.
2. Round 2: one attempt. It reused what it learned.
3. Round 3: **someone bumps the camera.** Its learned correction is now
   wrong. It fails, re-diagnoses, re-converges.
4. Two failures in a row → it asks aloud → a human shows it on the leader
   arm → it replays the demonstration and succeeds.
5. `logs/evidence.png`.

Step 3 is what they remember. Step 5 is why they believe it.

Use `--perturb-kind calibration` (the default), not `zone`. Moving the
zone is absorbed silently — the arm re-reads it from the overhead camera
every trial, so nothing visibly happens.

---

## Gates — decide these now, while everyone is calm

| When | Check | If it fails |
|---|---|---|
| T−1:00 | `python glasses.py` — video streaming? | **Cut it.** Audio is the core path; video is off by default and every caller already handles `None`. |
| T+0:30 | Arm moving from our code? | **Demo the mock.** Still a real system, teaching included. |
| T+1:30 | — | **HARD FREEZE.** Record a backup video. Rehearse only. |

The mock backend is never deleted, even after the real arm works.

**Safety:** emergency stop is unplugging the USB. One person owns that and
nothing else during any powered replay. Hands clear when powered.

---

## Bringing up the real arm

```bash
export FOLLOWER_PORT=/dev/tty.usbmodem1101
python arm_api.py --calibrate     # ~20 min, MANDATORY, do it first
export ROBOT_BACKEND=so101
```

Calibration records the five named poses plus a cm→joint offset model.
`SO101Arm` refuses to move without it — an uncalibrated arm swinging to a
hardcoded joint angle is how you break a servo. Every `send_action` carries
one reconnect retry, because the follower drops mid-session.

Then Team B changes nothing but `ROBOT_BACKEND`.

---

## Known risks

- **The Nebius model id is unverified.** `python planner.py --verify`
  before the first live call. A wrong id fails in a way that looks exactly
  like an auth error and eats 20 minutes.
- **`SO101Arm` and `teach.py --source leader` are untested against
  hardware.** lerobot is not installed in this tree and its module paths
  move between releases; both try several import paths and fail loudly.
- **The mock's `set_joints` uses a crude linear map, not real forward
  kinematics.** It exists so replayed demonstrations visibly work in mock
  mode. Labelled as such in the code.
- **Meta video needs a native app bridge.** Off by default; treat as a
  stretch goal, never the critical path.

---

## Env vars

```
NEBIUS_API_KEY      required for the model arms
NEBIUS_MODEL        default Qwen/Qwen2.5-VL-72B-Instruct — VERIFY IT
NEBIUS_VISION       1|0    send frames to the planner
ROBOT_BACKEND       mock|so101
FOLLOWER_PORT       serial port of the follower arm
LEADER_PORT         serial port of the leader arm (teaching)
SUCCESS_CM          5.0    pass threshold
GRASP_TOL_CM        3.0    how close the gripper must be
TEACH_AFTER         2      consecutive failures before escalating
MOCK_REALTIME       1|0    0 strips the mock's sleeps (fast sweeps)
GLASSES_VIDEO       0|1    the gated Meta video path
OVERHEAD_CAM        0      ArUco ground-truth camera index
WRIST_CAM           1      planner context camera index
```
