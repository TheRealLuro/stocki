#!/usr/bin/env bash
# Starts the full Stocki stack:
#   1. Postgres + backend API   (docker compose, port 8000)
#   2. ONNX model API           (uvicorn, port 8001)
#   3. Frontend dev server      (Vite, port 5173)
#
# Ctrl+C stops the model API and frontend. The backend stack keeps running;
# stop it with `docker compose down` (or `docker compose stop`).

set -euo pipefail
cd "$(dirname "$0")"

API_PORT="${STOCKI_API_PORT:-8000}"
MODEL_PORT="${MODEL_PORT:-8001}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

port_in_use() {
  python3 - "$1" <<'PY'
import socket, sys
port = int(sys.argv[1])
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(('127.0.0.1', port))
    print('free')
except OSError:
    print('in_use')
finally:
    s.close()
PY
}

pick_free_port() {
  local port="$1"
  while [ "$(port_in_use "$port")" = "in_use" ]; do
    port=$((port + 1))
  done
  echo "$port"
}

if [ "$(port_in_use "$MODEL_PORT")" = "in_use" ]; then
  MODEL_PORT="$(pick_free_port "$MODEL_PORT")"
  echo "!! Port $MODEL_PORT was already in use; using $MODEL_PORT for the model API instead."
fi
if [ "$(port_in_use "$FRONTEND_PORT")" = "in_use" ]; then
  FRONTEND_PORT="$(pick_free_port "$FRONTEND_PORT")"
  echo "!! Port $FRONTEND_PORT was already in use; using $FRONTEND_PORT for the frontend instead."
fi

# Dev-friendly rate limits. The backend defaults (120/min, 20/min for
# /dataset/*) are production-conservative; an interactive dashboard browsing
# sessions blows through them. Export these yourself to override.
export STOCKI_RATE_LIMIT="${STOCKI_RATE_LIMIT:-600}"
export STOCKI_DATASET_RATE_LIMIT="${STOCKI_DATASET_RATE_LIMIT:-60}"
export MODEL_PORT FRONTEND_PORT

pids=()
cleanup() {
  if [ "${#pids[@]}" -gt 0 ]; then
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

kill_stale_model_api() {
  local port="$1"
  local pid
  pid=$(ss -lntp "sport = :$port" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -n 1 | cut -d= -f2 || true)
  if [ -n "${pid:-}" ]; then
    local cmd
    cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
    if printf '%s\n' "$cmd" | grep -Eq 'uvicorn.*main:app|python.*main:app'; then
      echo "==> Stopping stale model API on :$port (pid $pid)"
      kill "$pid" 2>/dev/null || true
      sleep 1
    fi
  fi
}

echo "==> Starting Postgres + backend API (docker compose)"
docker compose up -d

echo "==> Waiting for the backend API on :$API_PORT"
healthy=0
for _ in $(seq 1 60); do
  if curl -sf "http://localhost:$API_PORT/health" >/dev/null; then
    healthy=1
    break
  fi
  sleep 1
done
if [ "$healthy" -ne 1 ]; then
  echo "!! Backend API did not become healthy after 60s -- check 'docker compose logs api'" >&2
  exit 1
fi

# Prefer the project venv (created once with the model requirements); fall back
# to whatever uvicorn is on PATH.
ROOT="$(pwd)"
UVICORN="$ROOT/.venv/bin/uvicorn"
if [ ! -x "$UVICORN" ]; then
  if command -v uvicorn >/dev/null; then
    UVICORN="uvicorn"
  else
    echo "!! uvicorn not found -- run:" >&2
    echo "     python3 -m venv .venv && .venv/bin/pip install -r model/requirements.txt" >&2
    exit 1
  fi
fi

kill_stale_model_api "$MODEL_PORT"

echo "==> Starting model API on :$MODEL_PORT"
(cd model && exec "$UVICORN" main:app --port "$MODEL_PORT") &
pids+=($!)

for _ in $(seq 1 30); do
  if curl -sf "http://localhost:$MODEL_PORT/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -sf "http://localhost:$MODEL_PORT/health" | grep -q '"model_loaded":true'; then
  echo "!! Model API is still not healthy on :$MODEL_PORT" >&2
  exit 1
fi

if command -v npm >/dev/null 2>&1; then
  if [ ! -d frontend/node_modules ]; then
    echo "==> Installing frontend dependencies"
    (cd frontend && npm install --no-audit --no-fund)
  fi

  echo "==> Starting frontend dev server on :$FRONTEND_PORT"
  (cd frontend && npm run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT") &
  pids+=($!)
else
  echo "!! npm not found; skipping frontend startup. Install Node.js/npm to run the dashboard."
fi

echo
echo "Dashboard:  http://localhost:$FRONTEND_PORT"
echo "Backend:    http://localhost:$API_PORT/docs"
echo "Model API:  http://localhost:$MODEL_PORT/health"
echo
echo "Ctrl+C stops the model API and frontend. 'docker compose down' stops the backend stack."

wait
