#!/usr/bin/env bash
# Stop and remove the Find My Friend containers (images, mlflow.db and logs are kept).
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose down
echo "Stopped. Start again with: bash scripts/docker_up.sh"
