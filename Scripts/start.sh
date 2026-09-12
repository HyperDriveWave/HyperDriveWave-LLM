#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$ROOT/Configs/docker-compose.yml"
ENV_FILE="$ROOT/Configs/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "missing $ENV_FILE; create it from Configs/.env.example first" >&2
  exit 1
fi

cd "$ROOT"
USER_PROFILES="${HDW_COMPOSE_PROFILES:-}"
set -a
. "$ENV_FILE"
set +a

if [ -n "$USER_PROFILES" ]; then
  HDW_COMPOSE_PROFILES="$USER_PROFILES"
elif [ -z "${HDW_COMPOSE_PROFILES:-}" ]; then
  HDW_COMPOSE_PROFILES="base knowledge web"
fi

bash "$SCRIPT_DIR/prepare_dirs.sh"

if ! command -v systemctl >/dev/null 2>&1 || ! systemctl --user show-environment >/dev/null 2>&1; then
  echo "systemd user manager is required for the GPU resource coordinator" >&2
  exit 1
fi
systemctl --user link "$SCRIPT_DIR/hyperdrivewave-resource-coordinator.service" >/dev/null 2>&1 || true
systemctl --user daemon-reload
systemctl --user enable --now hyperdrivewave-resource-coordinator.service >/dev/null
RUNTIME_ROOT="${HDW_RUNTIME_ROOT:-$ROOT/HDW_Runtime}"
for _ in {1..15}; do
  [ -S "$RUNTIME_ROOT/maintenance/maintenance.sock" ] && break
  sleep 1
done
if [ ! -S "$RUNTIME_ROOT/maintenance/maintenance.sock" ]; then
  echo "GPU resource coordinator failed to start; see: journalctl --user -u hyperdrivewave-resource-coordinator.service -n 120" >&2
  exit 1
fi

if [[ "${HDW_LOCAL_LLM_BASE_URL:-}" == *"host.docker.internal:1919"* ]]; then
  if ! command -v systemctl >/dev/null 2>&1 || ! systemctl --user show-environment >/dev/null 2>&1; then
    echo "systemd user manager is required for the resident llama.cpp service" >&2
    exit 1
  fi
fi

read -r -a profiles <<< "$HDW_COMPOSE_PROFILES"
args=()
for profile in "${profiles[@]}"; do
  args+=(--profile "$profile")
done

echo "starting HyperDriveWave with profiles: ${profiles[*]}"

# 必须先单独建 hdw-rag：hdw-mineru 的 Dockerfile 第一行是
# `FROM hyperdrivewave-hdw-rag:latest`，但 compose 里 mineru 没有声明对 rag 的
# depends_on，所以 `up --build` 的构建顺序没有保证——全新机器上会随机失败。
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" build hdw-rag

docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" \
  "${args[@]}" up -d --build --remove-orphans

if [[ "${HDW_LOCAL_LLM_BASE_URL:-}" == *"host.docker.internal:1919"* ]]; then
  systemctl --user link "$ROOT/HDW_Inference/llama/hyperdrivewave-llama.service" >/dev/null 2>&1 || true
  systemctl --user daemon-reload
  systemctl --user enable --now hyperdrivewave-llama.service >/dev/null
  for _ in {1..180}; do
    curl -fsS http://127.0.0.1:1919/health >/dev/null 2>&1 && break
    sleep 1
  done
  if ! curl -fsS http://127.0.0.1:1919/health >/dev/null 2>&1; then
    echo "llama.cpp local inference failed to start; see: journalctl --user -u hyperdrivewave-llama.service -n 120" >&2
    exit 1
  fi
fi

docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" ps

WEBUI_HOST="${HDW_WEBUI_BIND:-127.0.0.1}"
if [ "$WEBUI_HOST" = "0.0.0.0" ]; then
  WEBUI_HOST="127.0.0.1"
fi
echo "WebUI: http://${WEBUI_HOST}:${HDW_WEBUI_PORT:-3000}"
