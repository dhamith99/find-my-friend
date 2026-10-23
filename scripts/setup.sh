#!/usr/bin/env bash
# First-time setup after cloning the repo on a machine.
# Install torch + torchvision first (GPU build), then run:
#   bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Installing Python packages"
pip install -r requirements.txt

echo "==> Face detector model"
bash scripts/download_face_model.sh

echo "==> Pulling data from DVC remote"
dvc pull || echo "Could not pull. Check the remote with: dvc remote list"

echo "Setup done. Run the pipeline with: bash scripts/run_pipeline.sh"
