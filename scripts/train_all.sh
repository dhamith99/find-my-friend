#!/usr/bin/env bash
# Train all three models with the default settings in params.yaml (baseline runs).
# Start MLflow first in another terminal:  bash scripts/start_mlflow.sh
# Extra flags are passed to every run, e.g.:  bash scripts/train_all.sh --epochs 5
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ../ml02/bin/activate ]; then
  # shellcheck disable=SC1091
  source ../ml02/bin/activate
fi

curl -sf http://127.0.0.1:5000/health >/dev/null || {
  echo "MLflow server is not running. Start it first: bash scripts/start_mlflow.sh"; exit 1; }

for model in custom_cnn mobilenet_v3_small resnet18; do
  echo -e "\n==================== $model ===================="
  python -m src.train --model "$model" --run-name "${model}_baseline" "$@"
done

echo -e "\nDone. Compare the runs at http://127.0.0.1:5000"
