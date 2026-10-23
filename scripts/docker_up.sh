#!/usr/bin/env bash
# Build and start MLflow + API + app in Docker.
#   bash scripts/docker_up.sh                     build (if needed) and start
#   bash scripts/docker_up.sh --rebuild           force a fresh build
#   MLFLOW_PORT=5050 bash scripts/docker_up.sh    use another laptop port if 5000 is taken
#   (API_PORT and APP_PORT work the same way; inside Docker the ports never change)
set -euo pipefail
cd "$(dirname "$0")/.."

MLFLOW_PORT="${MLFLOW_PORT:-5000}"
API_PORT="${API_PORT:-8000}"
APP_PORT="${APP_PORT:-8501}"

[ -f mlflow.db ] || { echo "mlflow.db not found - train the models first (Phases 2-4)."; exit 1; }
[ -d mlartifacts ] || { echo "mlartifacts/ not found - train the models first (Phases 2-4)."; exit 1; }
mkdir -p logs

# Same MLflow version as the training environment (so the containers can open
# mlflow.db), and run the containers as you (so files stay yours).
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ../ml02/bin/activate ]; then
  # shellcheck disable=SC1091
  source ../ml02/bin/activate
fi
MLFLOW_VERSION=$(python -c "import mlflow; print(mlflow.__version__)" 2>/dev/null || echo "3.17.2")
cat > .env << EOF
MLFLOW_VERSION=$MLFLOW_VERSION
HOST_UID=$(id -u)
HOST_GID=$(id -g)
MLFLOW_PORT=$MLFLOW_PORT
API_PORT=$API_PORT
APP_PORT=$APP_PORT
EOF
echo "MLflow $MLFLOW_VERSION, user $(id -u):$(id -g), ports mlflow=$MLFLOW_PORT api=$API_PORT app=$APP_PORT (saved in .env)"

# Stop if a laptop port is already taken by something that is not our container
for port in "$MLFLOW_PORT" "$API_PORT" "$APP_PORT"; do
  if ss -ltn "sport = :$port" 2>/dev/null | grep -q LISTEN \
     && ! docker ps --format '{{.Names}} {{.Ports}}' | grep '^fmf-' | grep -q ":$port->"; then
    echo
    echo "Port $port is already in use on this laptop:"
    ss -ltnp "sport = :$port" 2>/dev/null | tail -n +2 || true
    echo "Either stop that program (if it is scripts/start_mlflow.sh, serve_api.sh or serve_app.sh,"
    echo "Docker replaces it), or pick another port, e.g.:  MLFLOW_PORT=5050 bash scripts/docker_up.sh"
    exit 1
  fi
done

[ "${1:-}" = "--rebuild" ] && docker compose build --no-cache
docker compose up -d --build

echo "Waiting for services to become healthy..."
for svc in fmf-mlflow fmf-api fmf-app; do
  state="missing"
  for _ in $(seq 120); do
    state=$(docker inspect -f '{{.State.Health.Status}}' "$svc" 2>/dev/null || echo "missing")
    [ "$state" = "healthy" ] && break
    sleep 2
  done
  echo "  $svc: $state"
done

docker compose ps
cat << EOF

MLflow UI + registry : http://127.0.0.1:$MLFLOW_PORT
Model API docs       : http://127.0.0.1:$API_PORT/docs
Real-time app        : http://127.0.0.1:$APP_PORT
Logs                 : docker compose logs -f api
Stop                 : bash scripts/docker_down.sh
EOF
if [ "$MLFLOW_PORT" != "5000" ]; then
  echo "Note: to train or run scripts on the laptop against this MLflow, first run:"
  echo "  export MLFLOW_TRACKING_URI=http://127.0.0.1:$MLFLOW_PORT"
fi
