#!/usr/bin/env bash
# Calibrates an SO-101 arm: move all joints to mid-range, press Enter, then
# sweep each joint through its full range of motion. Writes calibration JSON
# automatically to ~/.cache/huggingface/lerobot/calibration/...
#
# IMPORTANT: the --id here must match SO101_LEADER_ID / SO101_FOLLOWER_ID used
# everywhere else (teleoperate, record, rollout). Mismatched ids silently load
# the wrong (or no) calibration.
#
# Usage: ./03_calibrate.sh {leader|follower}
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

ROLE="${1:?Usage: $0 {leader|follower}}"

case "$ROLE" in
  leader)
    source configs/leader.env
    lerobot-calibrate --teleop.type=so101_leader --teleop.port="$SO101_LEADER_PORT" --teleop.id="$SO101_LEADER_ID"
    ;;
  follower)
    source configs/follower.env
    lerobot-calibrate --robot.type=so101_follower --robot.port="$SO101_FOLLOWER_PORT" --robot.id="$SO101_FOLLOWER_ID"
    ;;
  *)
    echo "Unknown role: $ROLE (expected 'leader' or 'follower')" >&2
    exit 1
    ;;
esac
