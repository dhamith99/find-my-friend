#!/usr/bin/env bash
# Start the real-time web app on http://127.0.0.1:8501
# Needs the model API running:  bash scripts/serve_api.sh  (which needs MLflow)
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ../ml02/bin/activate ]; then
  # shellcheck disable=SC1091
  source ../ml02/bin/activate
fi

export API_URL="${API_URL:-http://127.0.0.1:8000}"
curl -sf "$API_URL/health" >/dev/null || echo "Warning: model API not reachable at $API_URL yet"

echo "Open http://127.0.0.1:8501 in Chrome or Firefox and press 'Start camera'"
exec uvicorn app.server:app --host 127.0.0.1 --port "${APP_PORT:-8501}"
