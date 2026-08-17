# Agent notes: so101_yellow

This directory is "Team Yellow" in the project's Red/Blue/Yellow pipeline —
the physical SO-101 leader+follower arm, teleop data collection, and SmolVLA
fine-tuning. Full context and step-by-step usage: `README.md`. Read that
first; this file is only about the state of the code, not the workflow.

## Project reality check

This whole repo is **vibe-coded** — built fast, by AI agents, for a
hackathon. Nothing here has an established track record. Treat existing code
(including this directory) as a first draft, not ground truth. Read it
critically before building on it, and don't assume a script works just
because it's checked in.

## What has and hasn't been verified

- All 9 scripts in `scripts/` were written against LeRobot's docs, then
  **checked against the actual installed CLI** (`lerobot==0.6.1`, in this
  dir's own `.venv`) via `--help` output and one partial dry-run of
  `lerobot-train` (confirmed it parses args and reaches config validation).
- **Nothing has been run against real SO-101 hardware yet.** Calibration,
  teleop, recording, and rollout are all unverified beyond flag-name
  correctness. Don't trust that they work end-to-end until someone has
  actually run them with the arms connected.
- LeRobot's CLI changes across releases (confirmed while building this: the
  install pip pkg differs meaningfully from a source checkout — see below).
  If a script errors on an unrecognized flag, don't guess-fix it — run
  `.venv/bin/lerobot-<cmd> --help` (or with a specific `--policy.type=`/
  `--robot.type=` set, since some fields only appear once the polymorphic
  type is chosen) and compare against what the script passes.

## A real bug that was already caught and fixed here

The official docs (and a first draft of these scripts) used
`--policy.path=lerobot/smolvla_base` for both training and rollout. That flag
**does not exist** in the installed CLI — the real field is
`--policy.pretrained_path`, and it must be paired with `--policy.type=smolvla`
(the type isn't inferred from the path). This was caught by actually invoking
`lerobot-train --help` with `--policy.type=smolvla` set, not by trusting
prose docs. If you're modifying `07_train_smolvla.sh` or
`08_infer_rollout.sh`, keep both flags together.

## Do not touch `~/lerobot`

There is an unrelated LeRobot source checkout at `~/lerobot` on this machine
with its own gamepad/hand-tracking/kinematics experiments (see its own
`.claude/` and loose `.py` files at its root). It is a **different,
pre-existing project** — explicitly out of scope here per direct user
instruction. This directory (`so101_yellow/`) has its own isolated `.venv`
installed from PyPI (`pip install lerobot[...]`) and does not depend on or
reference `~/lerobot` in any way. Don't import from it, don't copy its
calibration files, don't suggest merging the two.

## Conventions if you extend this

- Scripts are numbered and meant to run in order (`00` install through `08`
  inference). Keep new scripts in that ordering scheme rather than bolting
  extra steps on ad hoc.
- Config lives in `configs/*.env` (ports/ids/camera settings) and `.env`
  (HF/dataset/policy identifiers) — sourced by scripts, not hardcoded. Add
  new tunables there, not as literals in scripts.
- The `--robot.id`/`--teleop.id` values in `configs/*.env` are calibration
  file keys, not free-form labels — changing them orphans any existing
  calibration.
- `calibration/` is checked into this repo (mirroring LeRobot's cache layout)
  and restored via `scripts/restore_calibration.sh`. It is physically
  calibrated to this project's specific two SO-101 arms — it's not portable
  to a different arm pair, and it's not a template to copy from.
