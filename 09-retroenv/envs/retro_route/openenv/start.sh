#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Fetch or check the private task bundle, then point the server at it.
python prepare.py
export RETROENV_BENCHMARK_DIR="${RETROENV_BENCHMARK_DIR:-${RETROENV_PREPARED_DIR:-prepared}}"
exec uvicorn retroenv_openenv.server:app --host 0.0.0.0 --port "${PORT:-8000}" --ws-ping-timeout 600
