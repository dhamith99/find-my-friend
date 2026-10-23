#!/usr/bin/env bash
# Build and start MLflow + API + app in Docker.
#   bash scripts/docker_up.sh            build (if needed) and start
#   bash scripts/docker_up.sh --rebuild  force a fresh build
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f mlflow.db ] || { echo "mlflow.db not found - train the models first (Phases 2-4)."; exit 1; }
[ -d mlartifacts ] || { echo "mlartifacts/ not found - train the models first (Phases 2-4)."; exit 1; }
mkdir -p logs

# Use the same MLflow version as the training environment, so the containers
# can open the existing mlflow.db; run containers as you, so files stay yours.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ../ml02/bin/activate ]; then
  # shellcheck disable=SC1091
  source ../ml02/bin/activate
fi
MLFLOW_VERSION=$(python -c "import mlflow; print(mlflow.__version__)" 2>/dev/null || echo "3.17.2")
cat > .env << EOF
MLFLOW_VERSION=$MLFLOW_VERSION
HOST_UID=$(id -u)
HOST_GID=$(id -g)
EOF
echo "Using MLflow $MLFLOW_VERSION, user $(id -u):$(id -g)  (written to .env)"

# The Docker MLflow server needs port 5000 - stop a local one first
if curl -sf http://127.0.0.1:5000/health >/dev/null && ! docker ps --format '{{.Names}}' | grep -q '^fmf-mlflow$'; then
  echo "Port 5000 is used by a local MLflow server. Stop it (Ctrl+C in its terminal) and run this again."
  exit 1
fi
for port in 8000 8501; do
  if curl -s -o /dev/null "http://127.0.0.1:$port" && ! docker ps --format '{{.Ports}}' | grep -q ":$port->"; then
    echo "Port $port is in use by a local process (serve_api.sh / serve_app.sh?). Stop it first."
    exit 1
  fi
done

BUILD_FLAGS="--build"
[ "${1:-}" = "--rebuild" ] && docker compose build --no-cache
docker compose up -d $BUILD_FLAGS

echo "Waiting for services to become healthy..."
for svc in fmf-mlflow fmf-api fmf-app; do
  for _ in $(seq 120); do
    state=$(docker inspect -f '{{.State.Health.Status}}' "$svc" 2>/dev/null || echo "missing")
    [ "$state" = "healthy" ] && break
    sleep 2
  done
  echo "  $svc: $state"
done

docker compose ps
cat << EOF

MLflow UI + registry : http://127.0.0.1:5000
Model API docs       : http://127.0.0.1:8000/docs
Real-time app        : http://127.0.0.1:8501
Logs                 : docker compose logs -f api
Stop                 : bash scripts/docker_down.sh
EOF
