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

# ── 本地推理走不走宿主 llama ──────────────────────────────────
#
# 原来这里写的是 `[[ "$HDW_LOCAL_LLM_BASE_URL" == *"host.docker.internal:1919"* ]]`。
# 那是个**字符串匹配**，后果很隐蔽：把 LLM 端口从 1919 改成别的，base_url 跟着变，
# 匹配就失败了 → 下面 enable llama 的整段被跳过 → **llama 根本不启动，
# 而且不打印任何提示**，只在最后报"服务已启动"。
#
# 改成两步，都不依赖硬编码端口：
#   1. base_url 是否指向宿主（判断"用不用本地推理"）
#   2. base_url 里的端口是否与 HDW_LLAMA_PORT 一致（判断"配置是否自洽"）
LLAMA_PORT="${HDW_LLAMA_PORT:-1919}"
LLAMA_HEALTH_URL="http://127.0.0.1:${LLAMA_PORT}/health"

wants_host_llama() {
  # 显式关掉：无 GPU 的部署常见（纯 CPU 跑 27B 约 1 token/s，实际不可用），
  # 由 deploy.sh 在检测到无卡并切在线 API 时写入。
  [ "${HDW_SKIP_LOCAL_LLM:-false}" = "true" ] && return 1
  case "${HDW_LOCAL_LLM_BASE_URL:-}" in
    *host.docker.internal*) return 0 ;;
    *) return 1 ;;
  esac
}

if [ "${HDW_SKIP_LOCAL_LLM:-false}" = "true" ]; then
  echo "本地推理已停用（HDW_SKIP_LOCAL_LLM=true）——不发 enable llama 服务。"
  echo "  问答走在线 API：${HDW_ONLINE_LLM_BASE_URL:-<未配置>} / ${HDW_ONLINE_LLM_MODEL:-<未配置>}"
  if [ -z "${HDW_ONLINE_LLM_API_KEY:-}" ]; then
    echo "  ⚠ 但没有配 HDW_ONLINE_LLM_API_KEY，在线也调不通——提问会返回 503。" >&2
  fi
fi

if wants_host_llama; then
  if ! command -v systemctl >/dev/null 2>&1 || ! systemctl --user show-environment >/dev/null 2>&1; then
    echo "systemd user manager is required for the resident llama.cpp service" >&2
    exit 1
  fi
  # 端口不一致是最容易踩的坑：llama 听 A，qa-api 连 B。
  # 表现是容器 /health 全绿、模型页正常，一提问就 connection refused。
  # 与其让人去猜，不如在这里直接拦下来。
  _url_port="$(printf '%s' "${HDW_LOCAL_LLM_BASE_URL:-}" | sed -nE 's#.*host\.docker\.internal:([0-9]+).*#\1#p')"
  if [ -n "$_url_port" ] && [ "$_url_port" != "$LLAMA_PORT" ]; then
    echo "配置不自洽：HDW_LOCAL_LLM_BASE_URL 指向端口 $_url_port，但 HDW_LLAMA_PORT=$LLAMA_PORT。" >&2
    echo "  llama 会监听 $LLAMA_PORT，而 qa-api 会去连 $_url_port，提问必然失败。" >&2
    echo "  重跑一次 bash Scripts/deploy.sh 会自动同步这两处（含 model-config/config.json）。" >&2
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

if wants_host_llama; then
  systemctl --user link "$ROOT/HDW_Inference/llama/hyperdrivewave-llama.service" >/dev/null 2>&1 || true
  systemctl --user daemon-reload
  systemctl --user enable --now hyperdrivewave-llama.service >/dev/null
  # 轮询用 LLAMA_HEALTH_URL（由 HDW_LLAMA_PORT 推导），不再写死 1919。
  # 写死的话改了端口会在这里空等 180 秒，然后报"llama 启动失败"，
  # 把人引去查 journalctl，而日志里其实什么问题都没有。
  for _ in {1..180}; do
    curl -fsS "$LLAMA_HEALTH_URL" >/dev/null 2>&1 && break
    sleep 1
  done
  if ! curl -fsS "$LLAMA_HEALTH_URL" >/dev/null 2>&1; then
    echo "llama.cpp local inference failed to start; see: journalctl --user -u hyperdrivewave-llama.service -n 120" >&2
    echo "  （探针地址：$LLAMA_HEALTH_URL，取自 HDW_LLAMA_PORT=$LLAMA_PORT）" >&2
    exit 1
  fi
fi

docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" ps

WEBUI_HOST="${HDW_WEBUI_BIND:-127.0.0.1}"
if [ "$WEBUI_HOST" = "0.0.0.0" ]; then
  WEBUI_HOST="127.0.0.1"
fi
echo "WebUI: http://${WEBUI_HOST}:${HDW_WEBUI_PORT:-3000}"
