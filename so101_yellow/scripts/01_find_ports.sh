#!/usr/bin/env bash
# Identifies the USB serial port for an SO-101 arm.
# Run this ONCE PER ARM (leader, then follower) — it will prompt you to
# unplug/replug the arm in question so it can diff the port list.
#
# Copy the printed port into configs/leader.env or configs/follower.env.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "Finding port — follow the on-screen prompts (unplug the arm you're identifying when asked)."
lerobot-find-port
