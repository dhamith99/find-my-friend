#!/usr/bin/env bash
# Start the Find My Friend API (all three @champion models) on http://127.0.0.1:8000
# Docs: http://127.0.0.1:8000/docs     Needs: bash scripts/start_mlflow.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ../ml02/bin/activate ]; then
  # shellcheck disable=SC1091
  source ../ml02/bin/activate
fi

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
curl -sf "$MLFLOW_TRACKING_URI/health" >/dev/null || {
  echo "MLflow server is not running. Start it first: bash scripts/start_mlflow.sh"; exit 1; }

exec uvicorn serving.api:app --host 127.0.0.1 --port "${API_PORT:-8000}"
