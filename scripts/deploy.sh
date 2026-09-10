#!/usr/bin/env bash
# Copy this tree to Atlas and Uranus. Does not start jobs and does not touch GPUs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOSTS=("atlas:~/Troiani-Platform" "uranus:~/Troiani-Platform")
for dest in "${HOSTS[@]}"; do
  echo "sync -> $dest"
  rsync -az --delete \
    --exclude '.venv' \
    --exclude 'var/' \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude '.pytest_cache/' \
    "$ROOT/" "$dest/"
done
echo "install venv on atlas"
ssh atlas 'cd ~/Troiani-Platform && python3 -m venv .venv && . .venv/bin/activate && pip install -q -U pip && pip install -q -e ".[dev]"'
echo "install venv on uranus"
ssh uranus 'cd ~/Troiani-Platform && python3 -m venv .venv && . .venv/bin/activate && pip install -q -U pip && pip install -q -e ".[dev]"'
echo "deployed"
