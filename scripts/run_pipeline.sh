#!/usr/bin/env bash
# Run the full Find My Friend pipeline with DVC.
# Only stages whose inputs changed are re-run (code, params.yaml or data).
#
# Usage (from anywhere):
#   bash scripts/run_pipeline.sh                 run pipeline + push data to DVC remote
#   bash scripts/run_pipeline.sh -m "message"    also commit and push to GitHub
#   bash scripts/run_pipeline.sh -s tune         run only one stage (and what it needs)
set -euo pipefail
cd "$(dirname "$0")/.."            # always run from the project root

MSG=""
STAGE=""
while getopts "m:s:" opt; do
  case $opt in
    m) MSG="$OPTARG" ;;
    s) STAGE="$OPTARG" ;;
    *) echo "Usage: $0 [-m \"commit message\"] [-s stage]"; exit 1 ;;
  esac
done

step() { echo -e "\n==> $*"; }

# Activate the ml02 venv (one folder up) if no env is active
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ../ml02/bin/activate ]; then
  # shellcheck disable=SC1091
  source ../ml02/bin/activate
fi

step "Checking tools"
command -v dvc >/dev/null || { echo "dvc not found. Run: pip install -r requirements.txt"; exit 1; }
python -c "import torch, cv2, yaml, mlflow, optuna" || {
  echo "Python packages missing. Run: pip install -r requirements.txt"; exit 1; }

step "Checking face detector model"
bash scripts/download_face_model.sh

if [ ! -d raw ]; then
  step "raw/ not found - pulling photos from DVC remote"
  dvc pull raw.dvc
fi

# Training stages log to MLflow - start the server if it is not already running
MLFLOW_PID=""
if ! curl -sf http://127.0.0.1:5000/health >/dev/null; then
  step "Starting MLflow server (log: mlflow_server.log)"
  bash scripts/start_mlflow.sh > mlflow_server.log 2>&1 &
  MLFLOW_PID=$!
  for _ in $(seq 60); do
    curl -sf http://127.0.0.1:5000/health >/dev/null && break
    sleep 1
  done
  curl -sf http://127.0.0.1:5000/health >/dev/null || {
    echo "MLflow did not start - see mlflow_server.log"; exit 1; }
fi
stop_mlflow() { [ -n "$MLFLOW_PID" ] && kill "$MLFLOW_PID" 2>/dev/null || true; }
trap stop_mlflow EXIT

step "Running pipeline (dvc repro ${STAGE})"
dvc repro ${STAGE:+"$STAGE"}

step "Metrics"
dvc metrics show

step "Pushing data and outputs to DVC remote"
dvc push

if [ -n "$MSG" ]; then
  step "Committing to Git"
  git add -A
  git commit -m "$MSG" || echo "Nothing new to commit"
  git push
fi

echo -e "\nDone. MLflow runs: start 'bash scripts/start_mlflow.sh' and open http://127.0.0.1:5000"
