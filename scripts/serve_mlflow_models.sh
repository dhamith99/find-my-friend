#!/usr/bin/env bash
# Serve each registered @champion model with MLflow's own model server:
#   fmf-custom_cnn          -> http://127.0.0.1:5001/invocations
#   fmf-mobilenet_v3_small  -> http://127.0.0.1:5002/invocations
#   fmf-resnet18            -> http://127.0.0.1:5003/invocations
# Needs the MLflow tracking server running (bash scripts/start_mlflow.sh).
# Test with:  python -m serving.mlflow_client <image.jpg>
# Stop with Ctrl+C.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ../ml02/bin/activate ]; then
  # shellcheck disable=SC1091
  source ../ml02/bin/activate
fi

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
curl -sf "$MLFLOW_TRACKING_URI/health" >/dev/null || {
  echo "MLflow server is not running. Start it first: bash scripts/start_mlflow.sh"; exit 1; }

mkdir -p logs
PIDS=()
cleanup() { echo "Stopping model servers..."; kill "${PIDS[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

port=5001
for model in custom_cnn mobilenet_v3_small resnet18; do
  echo "Starting fmf-$model@champion on port $port (log: logs/serve_$model.log)"
  mlflow models serve -m "models:/fmf-$model@champion" -h 127.0.0.1 -p "$port" \
    --env-manager local > "logs/serve_$model.log" 2>&1 &
  PIDS+=($!)
  port=$((port + 1))
done

for port in 5001 5002 5003; do
  for _ in $(seq 90); do curl -sf "http://127.0.0.1:$port/ping" >/dev/null && break; sleep 1; done
  if curl -sf "http://127.0.0.1:$port/ping" >/dev/null; then
    echo "  port $port ready"
  else
    echo "  port $port did not start - check logs/"
  fi
done

echo "All model servers running. Press Ctrl+C to stop."
wait
