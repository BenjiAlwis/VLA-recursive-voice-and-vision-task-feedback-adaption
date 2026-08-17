#!/usr/bin/env bash
# Teleop verification gate: leader arm should smoothly drive the follower arm,
# with a live camera + joint-position view (via rerun). Do not move on to
# recording until this looks good.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
source configs/leader.env
source configs/follower.env
source configs/cameras.env

lerobot-teleoperate \
  --robot.type=so101_follower --robot.port="$SO101_FOLLOWER_PORT" --robot.id="$SO101_FOLLOWER_ID" \
  --robot.cameras="{wrist: {type: opencv, index_or_path: $CAM_WRIST_INDEX, width: $CAM_WIDTH, height: $CAM_HEIGHT, fps: $CAM_FPS}}" \
  --teleop.type=so101_leader --teleop.port="$SO101_LEADER_PORT" --teleop.id="$SO101_LEADER_ID" \
  --display_data=true
