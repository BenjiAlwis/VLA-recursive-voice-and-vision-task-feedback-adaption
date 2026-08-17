#!/usr/bin/env bash
# Lists OpenCV-compatible cameras (webcams, phone via Continuity Camera/DroidCam)
# with their index_or_path. Copy the working index into configs/cameras.env.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

lerobot-find-cameras opencv
