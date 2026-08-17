#!/usr/bin/env bash
# Runs the fine-tuned SmolVLA policy against the live follower arm with a
# given text prompt. This is the Yellow <-> Blue/Red inference handoff point.
#
# NOTE: the prompt is fixed for the duration of this process (no hot mid-run
# prompt swap in the current CLI). v1 integration: Blue/Red re-invoke this
# script fresh each time the resolved instruction changes. See README.md
# "Inference handoff to Blue" for the v2 (hot-swap) stretch goal.
#
# Usage: ./08_infer_rollout.sh "<current text prompt from Blue>"
#
# Verify `lerobot-rollout` is the correct entry point for your installed
# lerobot version (`lerobot-rollout --help`) — this evolves; older versions
# may expose real-hardware policy eval differently.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
source .env
source configs/follower.env
source configs/cameras.env

TASK_PROMPT="${1:?Usage: $0 \"<current text prompt from Blue>\"}"

lerobot-rollout \
  --strategy.type=base \
  --robot.type=so101_follower --robot.port="$SO101_FOLLOWER_PORT" --robot.id="$SO101_FOLLOWER_ID" \
  --robot.cameras="{wrist: {type: opencv, index_or_path: $CAM_WRIST_INDEX, width: $CAM_WIDTH, height: $CAM_HEIGHT, fps: $CAM_FPS}}" \
  --task="$TASK_PROMPT" \
  --policy.type=smolvla \
  --policy.pretrained_path="$POLICY_REPO_ID" \
  --duration="${ROLLOUT_DURATION:-60}"
