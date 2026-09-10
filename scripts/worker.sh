#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
NODE="${1:-${TROIANI_NODE:-$(hostname -s)}}"
export TROIANI_PLATFORM_CONFIG="${TROIANI_PLATFORM_CONFIG:-$ROOT/config/platform.yaml}"
export TROIANI_NODE="$NODE"
exec troiani-platform worker --node "$NODE"
