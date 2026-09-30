#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$ROOT/Configs/docker-compose.yml"
ENV_FILE="$ROOT/Configs/.env"

# 单元安装要用到 common.sh 里的 install_hdw_units / systemd_path /
# write_if_different。它被 source 时不依赖项目变量（自己带加载保护），
# 所以放在最前面也不会和下面的 .env 打架。
# shellcheck source=lib/common.sh
. "$SCRIPT_DIR/lib/common.sh"

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
# 按**本机实际路径**生成单元，而不是 link 仓库里那份。
# 原来那行 `systemctl --user link "$SCRIPT_DIR/...service"` 只在项目位于
# ~/桌面/HyperDriveWave 时才对——单元里的路径是写死的，换个目录就指向不存在的
# 文件，而 EnvironmentFile 的 `-` 会让 systemd 静默略过缺失的 .env。
install_hdw_units "$ROOT"
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

# ── compose 文件清单 ────────────────────────────────────────────
# 默认只有主文件；要让 RAG/MinerU 上 GPU 再叠一个覆盖文件。
#
# 开闸条件就是 HDW_RAG_DEVICE=cuda，**不另设布尔开关**：那个键本来就是
# 「RAG 跑在哪个设备上」，再引一个开关就会出现「开关说上 GPU、键却说 cpu」
# 这种自相矛盾的状态，而且两种都不算错，排查时非常费劲。
#
# 覆盖文件给 RAG 和 MinerU 发卡，两者**默认共用 1 号**（llama 占 0 号），
# 所以机器上两张卡就够。想拆开就把 HDW_MINERU_GPU 设成别的号。
FILES=(-f "$COMPOSE_FILE")
if [ "${HDW_RAG_DEVICE:-cpu}" = "cuda" ]; then
  FILES+=(-f "$ROOT/Configs/docker-compose.rebuild-gpu.yml")
  dim "RAG/MinerU 启用 GPU（叠加 docker-compose.rebuild-gpu.yml，卡号取自 HDW_RAG_GPU / HDW_MINERU_GPU）"
fi

echo "starting HyperDriveWave with profiles: ${profiles[*]}"

# ── 要不要重新构建镜像 ──────────────────────────────────────────
#
# 默认构建（改了 app.py 之类重启就能生效）。但**无外网的机器构建不了**：
# 拉不到基础镜像、pip 装不了包，`docker compose build` 会卡在第一步且
# **一行输出都没有**，看起来像死锁。这类机器靠
#     bash Scripts/pack_hdw.sh          # 在有网的机器上打包
#     docker load < xxx.tar             # 搬到目标机
# 把镜像运过去，启动时只需要 up，不需要 build。
#
# 所以给一个显式开关。**不做"镜像已存在就跳过"的自动判断**——那会让
# `改了代码 → start.sh` 这个日常动作悄悄不再重建，比构建失败更难发现。
if [ "${HDW_SKIP_BUILD:-false}" = "true" ]; then
  dim "HDW_SKIP_BUILD=true —— 跳过镜像构建，直接用本地已有镜像"
  BUILD_ARGS=(--no-build)
else
  # 必须先单独建 hdw-rag：hdw-mineru 的 Dockerfile 第一行是
  # `FROM hyperdrivewave-hdw-rag:latest`，但 compose 里 mineru 没有声明对 rag 的
  # depends_on，所以 `up --build` 的构建顺序没有保证——全新机器上会随机失败。
  docker compose --env-file "$ENV_FILE" "${FILES[@]}" build hdw-rag
  BUILD_ARGS=(--build)
fi

docker compose --env-file "$ENV_FILE" "${FILES[@]}" \
  "${args[@]}" up -d "${BUILD_ARGS[@]}" --remove-orphans

if wants_host_llama; then
  install_hdw_units "$ROOT"
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

docker compose --env-file "$ENV_FILE" "${FILES[@]}" ps

WEBUI_HOST="${HDW_WEBUI_BIND:-127.0.0.1}"
if [ "$WEBUI_HOST" = "0.0.0.0" ]; then
  WEBUI_HOST="127.0.0.1"
fi
echo "WebUI: http://${WEBUI_HOST}:${HDW_WEBUI_PORT:-3000}"
