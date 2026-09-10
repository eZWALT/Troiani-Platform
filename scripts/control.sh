#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
export TROIANI_PLATFORM_CONFIG="${TROIANI_PLATFORM_CONFIG:-$ROOT/config/platform.yaml}"
export TROIANI_NODE="${TROIANI_NODE:-atlas}"
exec troiani-platform control --with-tracking
