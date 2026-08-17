# SO-101 Yellow — LeRobot setup

This directory covers "Team Yellow's" scope in the project's Red/Blue/Yellow
pipeline: the physical SO-101 leader+follower robot arm, teleoperated data
collection, and fine-tuning SmolVLA so the arm embodies natural-language
pick-and-place instructions.

## Where this fits in the big picture

- **Red** (perception/reasoning) and **Blue** (planning/control loop) are out
  of scope here — built by teammates.
- **Yellow** is the diagram's `Dynamic System (Robot)` node (colored yellow =
  "Action" in the legend): it receives a **text prompt** (initially from
  Blue's Feedback Controller, eventually corrections resolved by Red, e.g.
  "pick the red one, not green" or "don't pick it up like that"), and must
  turn that into robot actions. This directory's job is to produce that
  capability — a fine-tuned SmolVLA policy driving a real SO-101 arm — via
  teleoperated demonstrations recorded with the leader arm.
- State feedback for the wider system comes from video (handled by Red's
  Sensing/Reasoning loop), so Yellow doesn't need to emit any custom signal
  beyond the standard LeRobot policy action/observation loop.

## Prerequisites

- SO-101 leader + follower arm kit, USB cables, power supplies
- At least one USB webcam (or phone camera app exposing an OpenCV-compatible
  camera)
- Python >= 3.12
- A GPU for fine-tuning (local, or see the HF Jobs fallback in step 8)
- A Hugging Face account + access token (for dataset/policy push to Hub)

**A note on currency:** LeRobot's CLI changes across releases. If a command
below doesn't match what's installed, run `lerobot-<command> --help` and
cross-check https://huggingface.co/docs/lerobot. See `docs/troubleshooting.md`.

## Workflow

Run everything from this directory (`so101_yellow/`) unless noted.

### 1. Install

```bash
./scripts/00_install.sh
source .venv/bin/activate
```

This creates a Python 3.12 venv, installs
`lerobot[feetech,smolvla,core_scripts,training]` (the `feetech` extra is
required — SO-101 uses Feetech STS3215 servos, not Dynamixel), and copies
`.env.example` to `.env`. Fill in `HF_USER` and `HUGGINGFACE_TOKEN` in `.env`,
and run `hf auth login` if you haven't already.

### 2. Find each arm's USB port

```bash
./scripts/01_find_ports.sh
```

Run once, follow the prompts (it'll ask you to unplug the arm in question to
diff the port list). Do this for the leader, then again for the follower.
Copy the resulting ports into `configs/leader.env` and `configs/follower.env`.

### 3. Set up motor IDs (only for brand-new/repurposed motors)

```bash
./scripts/02_setup_motors.sh leader
./scripts/02_setup_motors.sh follower
```

Walks you through connecting each of the 6 servos individually so the script
can burn a unique ID (1-6) and common baudrate into each motor. Skip this if
your arms' motors are already configured.

### 4. Calibrate both arms

```bash
./scripts/03_calibrate.sh leader
./scripts/03_calibrate.sh follower
```

For each arm: move all joints to the middle of their range, press Enter, then
physically sweep each joint through its full range of motion. This records
homing offsets + limits to a calibration file keyed by the arm's `id`
(`SO101_LEADER_ID` / `SO101_FOLLOWER_ID` in `configs/*.env`).

**Keep these ids fixed** — every later step (teleop, record, rollout) must
use the exact same id or it will silently load the wrong calibration.

### 5. Find your camera

```bash
./scripts/04_find_cameras.sh
```

Lists available OpenCV-compatible cameras with their index. Put the working
index into `configs/cameras.env`.

### 6. Verify teleoperation

```bash
./scripts/05_teleoperate.sh
```

The follower arm should smoothly mirror the leader arm, with a live camera +
joint-position view. **This is a gate** — don't move on to recording until
this looks good.

### 7. Record demonstration data

See `prompts/task_prompts.md` for the variation matrix (object color/type,
grasp style). For each row:

```bash
./scripts/06_record_dataset.sh "Pick up the red cube and place it in the bin" 10
```

Each call appends `<episodes>` episodes labeled with that prompt to the same
dataset (`DATASET_REPO_ID` in `.env`). During recording: → / `n` = next
episode, ← / `r` = re-record, Esc / `q` = stop and finalize/upload. Aim for
~10 episodes per variation and ≥50 episodes total before fine-tuning — don't
cut this short even under time pressure, it matters more than training steps.

### 8. Fine-tune SmolVLA

```bash
./scripts/07_train_smolvla.sh
```

Fine-tunes the pretrained `lerobot/smolvla_base` (450M param) checkpoint on
your recorded dataset — this is the recommended path, not training from
scratch. Defaults to `STEPS=20000` (~4hrs on an A100); override via `.env` or
inline env vars. **Run a smoke test first** with `STEPS=100 ./scripts/07_train_smolvla.sh`
to confirm the pipeline runs end-to-end before committing GPU time.

No local GPU? Add `--job.target=a10g-small` (or similar) to the
`lerobot-train` call in the script to run on HF Jobs.

Checkpoints land in `outputs/train/yellow_smolvla/checkpoints/`.

### 9. Run inference / hand off to Blue

```bash
./scripts/08_infer_rollout.sh "Pick up the red cube and place it in the bin"
```

Runs the fine-tuned policy against the live follower arm with a given text
prompt.

**Open integration question for the team:** the current LeRobot CLI takes a
fixed prompt per process invocation — there's no built-in hot mid-run prompt
swap. Two options:

1. **v1 (implemented here)**: Blue/Red re-invoke `08_infer_rollout.sh` with
   the newly resolved prompt whenever the instruction changes; the process
   restarts between corrections. Fine if corrections land between pick
   attempts.
2. **v2 (stretch goal, not built)**: a small Python service using LeRobot's
   policy API directly, polling a shared queue/file Blue writes to, so the
   conditioning text can update between action chunks without restarting —
   needed only if a correction must interrupt an in-flight motion.

## Directory layout

```
so101_yellow/
├── README.md              this file
├── requirements.txt        pip install line (extras-based)
├── .env.example             copy to .env (gitignored) — HF creds, dataset/policy repo ids
├── configs/                 arm ports/ids, camera settings — filled in during steps 2, 4, 5
├── scripts/                 00-08, run in order (see Workflow above)
├── prompts/task_prompts.md  recording variation matrix
└── docs/troubleshooting.md  common issues
```

## Troubleshooting

See `docs/troubleshooting.md` for port/camera drift, calibration id
mismatches, Linux permissions, and slow training.
