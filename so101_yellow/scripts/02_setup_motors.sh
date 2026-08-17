#!/usr/bin/env bash
# One-time motor ID/baudrate setup for a fresh (or repurposed) SO-101 arm's
# Feetech servos. NOT needed if the arm's motors were already configured.
#
# Usage: ./02_setup_motors.sh {leader|follower}
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

ROLE="${1:?Usage: $0 {leader|follower}}"

case "$ROLE" in
  leader)
    source configs/leader.env
    lerobot-setup-motors --teleop.type=so101_leader --teleop.port="$SO101_LEADER_PORT"
    ;;
  follower)
    source configs/follower.env
    lerobot-setup-motors --robot.type=so101_follower --robot.port="$SO101_FOLLOWER_PORT"
    ;;
  *)
    echo "Unknown role: $ROLE (expected 'leader' or 'follower')" >&2
    exit 1
    ;;
esac
