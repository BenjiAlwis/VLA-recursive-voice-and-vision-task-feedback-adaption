#!/usr/bin/env bash
# Records teleoperated demonstration episodes for one task-prompt variation.
# Run once per row of prompts/task_prompts.md — episodes accumulate into the
# same DATASET_REPO_ID across runs via --resume=true.
#
# Usage: ./06_record_dataset.sh "<task prompt>" [num_episodes]
#
# Controls during recording: -> / n = next episode, <- / r = re-record episode,
# Esc / q = stop and finalize/upload.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
source .env
source configs/leader.env
source configs/follower.env
source configs/cameras.env

TASK_PROMPT="${1:?Usage: $0 \"<task prompt>\" [num_episodes]}"
NUM_EPISODES="${2:-10}"

lerobot-record \
  --robot.type=so101_follower --robot.port="$SO101_FOLLOWER_PORT" --robot.id="$SO101_FOLLOWER_ID" \
  --robot.cameras="{wrist: {type: opencv, index_or_path: $CAM_WRIST_INDEX, width: $CAM_WIDTH, height: $CAM_HEIGHT, fps: $CAM_FPS}}" \
  --teleop.type=so101_leader --teleop.port="$SO101_LEADER_PORT" --teleop.id="$SO101_LEADER_ID" \
  --display_data=true \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.single_task="$TASK_PROMPT" \
  --resume=true
