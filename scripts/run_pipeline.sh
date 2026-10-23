#!/usr/bin/env bash
# Run the full Find My Friend pipeline with DVC.
# Only stages whose inputs changed are re-run (code, params.yaml or data).
#
# Usage (from anywhere):
#   bash scripts/run_pipeline.sh                 run pipeline + push data to DVC remote
#   bash scripts/run_pipeline.sh -m "message"    also commit and push to GitHub
set -euo pipefail
cd "$(dirname "$0")/.."            # always run from the project root

MSG=""
while getopts "m:" opt; do
  case $opt in
    m) MSG="$OPTARG" ;;
    *) echo "Usage: $0 [-m \"commit message\"]"; exit 1 ;;
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
python -c "import torch, cv2, yaml" || { echo "Python packages missing. Run: pip install -r requirements.txt"; exit 1; }

step "Checking face detector model"
bash scripts/download_face_model.sh

if [ ! -d raw ]; then
  step "raw/ not found - pulling photos from DVC remote"
  dvc pull raw.dvc
fi

step "Running pipeline (dvc repro)"
dvc repro

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

echo -e "\nDone."
