#!/usr/bin/env bash
# Fine-tunes the pretrained lerobot/smolvla_base checkpoint on the recorded
# dataset. Run a smoke test first with STEPS=100 to confirm the pipeline
# works end-to-end before committing GPU time to a full run.
#
# No local GPU? Use HF Jobs instead, e.g. add: --job.target=a10g-small
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
source .env

lerobot-train \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --batch_size="${BATCH_SIZE:-64}" \
  --steps="${STEPS:-20000}" \
  --output_dir=outputs/train/yellow_smolvla \
  --job_name=yellow_smolvla_training \
  --policy.device="${POLICY_DEVICE:-cuda}" \
  --policy.repo_id="$POLICY_REPO_ID" \
  --wandb.enable="${WANDB_ENABLE:-false}"
