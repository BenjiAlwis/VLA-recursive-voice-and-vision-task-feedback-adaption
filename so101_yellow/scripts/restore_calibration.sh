#!/usr/bin/env bash
# Restores the checked-in yellow_leader/yellow_follower calibration files
# into LeRobot's expected cache location, so a teammate cloning this repo
# doesn't have to physically recalibrate the arms from scratch.
#
# LeRobot's default calibration lookup (--robot.id / --teleop.id with no
# --robot.calibration_dir / --teleop.calibration_dir override) resolves to
# ~/.cache/huggingface/lerobot/calibration/{robots|teleoperators}/<type>/<id>.json
# — this script copies our checked-in files there. It does NOT overwrite an
# existing file with the same name unless --force is passed, since a
# teammate's own freshly-calibrated file for the same id should win by
# default.
#
# Usage: ./restore_calibration.sh [--force]
set -euo pipefail
cd "$(dirname "$0")/.."

FORCE=false
[ "${1:-}" = "--force" ] && FORCE=true

CACHE_ROOT="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}/calibration"

copy_one() {
  local src="$1" dest="$2"
  if [ -f "$dest" ] && [ "$FORCE" != "true" ]; then
    echo "skip (exists): $dest — pass --force to overwrite"
    return
  fi
  mkdir -p "$(dirname "$dest")"
  cp "$src" "$dest"
  echo "restored: $dest"
}

copy_one calibration/robots/so_follower/yellow_follower.json \
         "$CACHE_ROOT/robots/so_follower/yellow_follower.json"
copy_one calibration/teleoperators/so_leader/yellow_leader.json \
         "$CACHE_ROOT/teleoperators/so_leader/yellow_leader.json"
