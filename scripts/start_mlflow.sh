#!/usr/bin/env bash
# Start the MLflow tracking server + model registry on http://127.0.0.1:5000
#   runs/metrics/registry -> mlflow.db (SQLite)
#   models and files      -> mlartifacts/
# Keep this terminal open while training. Stop with Ctrl+C.
# (In Phase 7 the Docker MLflow service replaces this script, using the same files.)
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ../ml02/bin/activate ]; then
  # shellcheck disable=SC1091
  source ../ml02/bin/activate
fi

exec mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  --host 127.0.0.1 \
  --port "${MLFLOW_PORT:-5000}"
