#!/usr/bin/env bash
# One-time environment setup. Requires Python >= 3.12.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  python3.12 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example — fill in HF_USER / HUGGINGFACE_TOKEN before recording/training."
fi

echo "Install complete. Activate with: source so101_yellow/.venv/bin/activate"
