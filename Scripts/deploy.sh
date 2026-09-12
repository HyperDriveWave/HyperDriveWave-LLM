#!/usr/bin/env bash
# HyperDriveWave 一键部署。
#
# 用法：
#   bash Scripts/deploy.sh                        # 交互式，自动探测
#   bash Scripts/deploy.sh --network offline      # 离线：只报缺，不下载
#   bash Scripts/deploy.sh --network mirror --proxy http://10.0.0.1:7890
#   bash Scripts/deploy.sh --bind 10.0.0.5        # 指定 WebUI 绑定的网卡地址
#   bash Scripts/deploy.sh --skip-models          # 模型已备好，跳过检查与下载
#   bash Scripts/deploy.sh --prefer gitee         # 第三方依赖优先走 gitee（国内网络建议）
#   bash Scripts/deploy.sh --setup-mirrors       # 先配好 Docker/pip/npm/apt 国内镜像源
#
# 设计原则：
#   1. **路径一律相对**。代码本身已经自定位（脚本用 BASH_SOURCE 推根目录、
#      compose 用 ../ 相对挂载），所以部署脚本的工作是**删掉 .env 里的绝对路径覆盖**，
#      把控制权还给这些默认值——而不是写一批新的绝对路径进去。
#      这样项目在家目录里移动后重跑一次即可恢复，不会"换机器就坏、再移动又坏"。
#   2. 启动逻辑不重写，末尾 exec Scripts/start.sh。
#   3. 能降级的降级并明确报告，不能降级的立刻失败。

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HDW_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export HDW_ROOT

# shellcheck source=lib/common.sh
. "$SCRIPT_DIR/lib/common.sh"
# shellcheck source=lib/detect.sh
. "$SCRIPT_DIR/lib/detect.sh"
# shellcheck source=lib/models.sh
. "$SCRIPT_DIR/lib/models.sh"

ENV_FILE="$HDW_ROOT/Configs/.env"
ENV_EXAMPLE="$HDW_ROOT/Configs/.env.example"
COMPOSE_FILE="$HDW_ROOT/Configs/docker-compose.yml"

NET_MODE=""
PROXY=""
BIND_IP=""
SKIP_MODELS=0
SKIP_VERIFY=0
WITH_FRP=0
DRY_RUN=0
PREFER_HOST=""
SETUP_MIRRORS=0

usage() {
  sed -n "2,21p" "${BASH_SOURCE[0]}" | sed 's/^# \?//'
  exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    --network)    NET_MODE="${2:-}"; shift 2 ;;
    --proxy)      PROXY="${2:-}"; shift 2 ;;
    --bind)       BIND_IP="${2:-}"; shift 2 ;;
    --skip-models) SKIP_MODELS=1; shift ;;
    --skip-verify) SKIP_VERIFY=1; shift ;;
    --with-frp)   WITH_FRP=1; shift ;;
    --dry-run)    DRY_RUN=1; shift ;;
    --prefer)     PREFER_HOST="${2:-}"; shift 2 ;;
    --setup-mirrors) SETUP_MIRRORS=1; shift ;;
    -h|--help)    usage ;;
    *) die "未知参数：$1（--help 看用法）" ;;
  esac
done

case "$NET_MODE" in
  ""|online|mirror|offline) ;;
  *) die "--network 只能是 online / mirror / offline，收到：$NET_MODE" ;;
esac

# ═══ 阶段 0：前置检查与探测 ═══════════════════════════════════

step "前置检查"

need_cmd docker "安装 Docker Engine"
docker compose version >/dev/null 2>&1 || die "需要 docker compose v2（docker-compose v1 不支持 profiles）"
need_cmd curl ""; need_cmd tar ""; need_cmd awk ""; need_cmd sed ""
need_cmd python3 "resource_coordinator 与模型配置读写都依赖它"

if ! systemctl --user show-environment >/dev/null 2>&1; then
  die "systemd 用户管理器不可用。llama 和资源协调器都是用户级服务，没有它无法启动。
     常见原因：在容器里跑、或没有登录会话。"
fi

[ -f "$COMPOSE_FILE" ] || die "找不到 $COMPOSE_FILE，项目结构不完整"

# 端口占用：早点发现比 compose up 失败后再回滚便宜。
# 被本项目自己的容器占着是正常状态（重跑部署时必然如此），只报别人的冲突。
for p in "${HDW_WEBUI_PORT:-3000}" "${HDW_QA_API_PORT:-8080}" "${HDW_RAG_PORT:-8001}" "${HDW_MINERU_PORT:-8002}"; do
  if port_busy "$p" && ! port_held_by_hdw "$p"; then
    warn "端口 $p 被本项目的进程之外的东西占用，compose 启动时可能失败"
  fi
done

FREE_GB="$(disk_free_gb "$HDW_ROOT")"
info "项目根：$HDW_ROOT"
info "磁盘可用：${FREE_GB}G"

step "环境探测"

detect_gpu
GPU_SUMMARY="无 NVIDIA 卡"
if [ "$HDW_GPU_COUNT" -gt 0 ]; then
  GPU_SUMMARY="$HDW_GPU_COUNT 张，选中 #$HDW_GPU_BEST_INDEX $HDW_GPU_NAME（cap $HDW_GPU_CAP，空闲 ${HDW_GPU_FREE_MIB}MiB）"
fi
info "GPU：$GPU_SUMMARY"

# 后端选择：看架构能不能在目标卡上跑，而不是看目录存不存在
LLAMA_DIR="$HDW_ROOT/HDW_Inference/llama"
BACKEND_REL="$(select_llama_backend_dir "$LLAMA_DIR" "$HDW_GPU_CAP" "$HDW_GPU_COUNT")"
[ -n "$BACKEND_REL" ] || die "找不到可用的 llama-server 二进制（$LLAMA_DIR 下没有任何构建产物）"

LLAMA_DEVICE="cpu"
case "$BACKEND_REL" in
  *build-cuda*) LLAMA_DEVICE="CUDA0" ;;
  *build*)      LLAMA_DEVICE="Vulkan0" ;;
esac

info "推理后端：$BACKEND_REL（device=$LLAMA_DEVICE）"

# 后端选错是"跑起来才炸"的典型：CUDA kernel 不匹配时 /health 照样通过。
case "$BACKEND_REL" in
  *build-cuda-multi*)
    : # 多架构，目标卡在任何支持的架构上都没问题
    ;;
  *build-cuda*)
    BACKEND_ARCHES="$(llama_backend_arches "$LLAMA_DIR/$BACKEND_REL")"
    if [ "$HDW_GPU_COUNT" -gt 0 ] && ! arch_covers_cap "$BACKEND_ARCHES" "$HDW_GPU_CAP"; then
      die "内部错误：选中的后端不覆盖本机架构（arch=$BACKEND_ARCHES cap=$HDW_GPU_CAP）"
    fi
    warn "当前 CUDA 后端只编译了 sm_${BACKEND_ARCHES:-未知}，换到别的显卡型号会失败。
     要支持多型号，在本机跑一次：
       cd $LLAMA_DIR/llama.cpp-upstream && cmake -B build-cuda-multi -S . \\
         -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_NATIVE=OFF \\
         -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON \\
         -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF \\
         -DCMAKE_CUDA_ARCHITECTURES='80;89;90;120' && \\
       cmake --build build-cuda-multi -j \"\$(nproc)\" --target llama-server"
    ;;
  *)
    warn "回退到 Vulkan 后端（$BACKEND_REL）。它不需要 NVIDIA 卡，但代价明确：
     - MTP 失效（旧版 llama.cpp 不支持 --spec-type），解码吞吐大约降 2-3 倍
     - 旧版可能不认当前模型架构，启动会直接失败——所以下面的推理冒烟测试必须通过
     要拿到 CUDA + MTP，请看上面重编多架构二进制的命令。"
    ;;
esac

# CUDA 二进制链接 /usr/local/cuda/.../libcudart.so.13，目标机只有驱动没有 toolkit 时会起不来
if [ "$LLAMA_DEVICE" = "CUDA0" ] && ! cuda_runtime_present; then
  warn "找不到 CUDA runtime（/usr/local/cuda/targets/x86_64-linux/lib）。
     如果二进制目录里没有随包携带 runtime 库，llama 会启动失败。"
fi

if [ -z "$BIND_IP" ]; then
  BIND_IP="$(detect_lan_ip)"
fi
info "WebUI 绑定地址：$BIND_IP"

# 网络模式：不指定就交互式问，非交互（无 tty）默认 online
if [ -z "$NET_MODE" ]; then
  if [ -t 0 ]; then
    log ""
    log "网络模式（决定模型怎么来）："
    log "  1) online  —— 直连魔搭 / Docker Hub（默认）"
    log "  2) mirror  —— 走代理或内网镜像源"
    log "  3) offline —— 不下载，只报告缺什么"
    printf '选择 [1]: '
    read -r _ans
    case "${_ans:-1}" in
      1|"") NET_MODE=online ;;
      2) NET_MODE=mirror ;;
      3) NET_MODE=offline ;;
      *) die "无效选择：$_ans" ;;
    esac
  else
    NET_MODE=online
  fi
fi
info "网络模式：$NET_MODE"

# 代理：mirror 模式下没有代理就只能靠已配好的 docker registry mirror
if [ "$NET_MODE" = "mirror" ]; then
  if [ -z "$PROXY" ] && [ -z "${HTTPS_PROXY:-}${https_proxy:-}" ]; then
    MIRRORS="$(docker_registry_mirrors)"
    if [ "$MIRRORS" = "[]" ] || [ -z "$MIRRORS" ]; then
      warn "mirror 模式但既没有 --proxy 也没有配置 docker registry-mirrors。
       注意：**镜像构建期（docker build 里的 pip）不读宿主机的镜像源配置**，
       只能通过代理生效。没有代理时，模型下载走镜像源可以，
       但 hdw-* 镜像的构建会去连公网 PyPI。"
    else
      info "docker registry mirrors：$MIRRORS"
    fi
  fi
fi

if [ "$NET_MODE" = "offline" ]; then
  info "离线模式：跳过下载与镜像构建，只校验本地是否齐备"
fi

# ── 镜像源 ──
# 拉基础镜像走 Docker Hub、构建期 pip 走 PyPI，在国内网络下都很慢甚至连不上。
# 显式 --setup-mirrors 才动手改 /etc（要 sudo、还会重启 docker），
# 其余情况只提示，不擅自改系统的包管理配置。
if [ "$SETUP_MIRRORS" = "1" ]; then
  step "配置镜像源"
  bash "$SCRIPT_DIR/setup_mirrors.sh" || warn "镜像源配置未全部成功，继续部署"
elif [ "$NET_MODE" = "mirror" ]; then
  _m="$(docker_registry_mirrors)"
  if [ "$_m" = "[]" ] || [ -z "$_m" ]; then
    warn "选了 mirror 模式但 Docker 还没配 registry-mirrors。
     拉基础镜像会走 Docker Hub（国内通常很慢）。先跑一次：
       bash Scripts/setup_mirrors.sh
     它会读本机现有配置并让你确认后再写。"
  fi
fi

# 决策快照。后续阶段只读它，不重复探测。
mkdir -p "$(deploy_state_dir)"
PLAN_FILE="$(deploy_state_dir)/plan.env"
{
  echo "# 部署决策快照，由 deploy.sh 生成。改了要重跑 deploy.sh。"
  echo "HDW_DEPLOY_NET=$NET_MODE"
  echo "HDW_DEPLOY_GPU_COUNT=$HDW_GPU_COUNT"
  echo "HDW_DEPLOY_GPU_CAP=$HDW_GPU_CAP"
  echo "HDW_DEPLOY_GPU_NAME=$HDW_GPU_NAME"
  echo "HDW_DEPLOY_LLAMA_BACKEND=$BACKEND_REL"
  echo "HDW_DEPLOY_LLAMA_DEVICE=$LLAMA_DEVICE"
  echo "HDW_DEPLOY_LAN_IP=$BIND_IP"
  echo "HDW_DEPLOY_START_TS=$(date +%s)"
} > "$PLAN_FILE"
dim "  决策快照：$PLAN_FILE"

# ═══ 阶段 1：目录 ═════════════════════════════════════════════

step "准备运行目录"

bash "$SCRIPT_DIR/prepare_dirs.sh"

# prepare_dirs.sh 没建这三个，但 compose 要挂载它们。
# 不补建的话 Docker 会自动建成 root:root，之后普通用户读写都要 sudo。
RUNTIME_ROOT="$HDW_ROOT/HDW_Runtime"
for d in ingest model-config frp; do
  mkdir -p "$RUNTIME_ROOT/$d"
done
info "已补建 ingest / model-config / frp（prepare_dirs.sh 漏掉了这三个）"

# ═══ 阶段 2：配置渲染 ═════════════════════════════════════════

step "渲染机器相关配置"

# 先确保 .env 存在
if [ ! -f "$ENV_FILE" ]; then
  [ -f "$ENV_EXAMPLE" ] || die "既没有 $ENV_FILE 也没有 $ENV_EXAMPLE"
  cp "$ENV_EXAMPLE" "$ENV_FILE"
  chmod 600 "$ENV_FILE"
  info "已从 .env.example 生成 Configs/.env"
fi
backup_once "$ENV_FILE"

# ── 关键：删掉绝对路径覆盖，把控制权还给代码里的相对默认值 ──
# 这几个键的默认值本来就是自定位的：
#   resource_coordinator.py  ROOT = Path(__file__).resolve().parents[1]
#   prepare_dirs.sh          ROOT = dirname($BASH_SOURCE)/..
#   docker-compose.yml       ${HDW_RUNTIME_ROOT:-../HDW_Runtime}（相对 compose 文件）
#   llama/start.sh           BINARY 自选 + SCRIPT_DIR
# 一旦 .env 用绝对路径覆盖，项目就不能移动了。删掉它们。
#
# 用注释掉而不是删除：保留可读性，万一有人需要显式覆盖可以从这里恢复。
unset_abs_path_key() {
  local key="$1" cur
  cur="$(env_get "$ENV_FILE" "$key" || true)"
  case "$cur" in
    /*)   # 绝对路径才处理；相对路径或空值已经是可移植的
      local tmp
      tmp="$(mktemp "$ENV_FILE.tmp.XXXXXX")"
      awk -v k="$key" '
        $0 ~ "^" k "=" { print "# [deploy] 已注释以保持可移植（代码默认值自定位）：" $0; next }
        { print }
      ' "$ENV_FILE" > "$tmp"
      chmod --reference="$ENV_FILE" "$tmp" 2>/dev/null || true
      mv -f "$tmp" "$ENV_FILE"
      dim "  已注释绝对路径覆盖：$key=$cur"
      ;;
  esac
}
for key in HDW_PROJECT_ROOT HDW_RUNTIME_ROOT HDW_LLAMA_BINARY HDW_ZVEC_COLLECTION_PATH \
           HDW_MODEL_CONFIG_PATH HDW_CHATDATA_ROOT; do
  unset_abs_path_key "$key"
done

# 真正需要按机器改的（不是路径，是环境）
env_set "$ENV_FILE" HDW_WEBUI_BIND "$BIND_IP"
env_set_if_empty "$ENV_FILE" HDW_COMPOSE_PROFILES "base knowledge web" || true

# 远端 RAG 默认清空：新机器上那个远端地址多半不存在，
# 留着会让每次问答有一半请求吃 3 秒连接超时。
if [ -z "$(env_get "$ENV_FILE" HDW_RAG_REMOTE_URLS || true)" ]; then
  env_set "$ENV_FILE" HDW_RAG_REMOTE_URLS ""
fi

info ".env 已更新（HDW_WEBUI_BIND=$BIND_IP，绝对路径覆盖已清除）"

# ── model-config/config.json ──
# 隐藏硬依赖：llama/start.sh、resource_coordinator、qa-api 三方都读它，
# 但仓库里没有任何地方生成它。
CONFIG_JSON="$RUNTIME_ROOT/model-config/config.json"
if [ ! -f "$CONFIG_JSON" ]; then
  python3 - "$CONFIG_JSON" <<'PY'
import json, sys
from pathlib import Path
target = Path(sys.argv[1])
target.parent.mkdir(parents=True, exist_ok=True)
cfg = {
    "version": 2,
    "local": {
        "base_url": "http://host.docker.internal:1919/v1",
        "model": "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf",
        "engine": "llama.cpp",
        "context_window": 262144,
        "multimodal_enabled": False,
        "mtp_enabled": True,
        "thinking_enabled": True,
        "model_options": [],
        "engine_options": ["llama.cpp", "FreeToken"],
    },
    "online": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
        "context_window": 262144,
        "multimodal_enabled": False,
        "thinking_enabled": True,
        "enabled_for_users": True,
        "model_options": [],
    },
    "retrieval": {"embedding_model": "bge-m3", "reranker_model": "bge-reranker-v2-m3"},
    "vision_priority": [],
}
target.write_text(json.dumps(cfg, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
PY
  # 归属当前用户：qa-api 要写回它，root 属主会让 PATCH /model-config 失败
  info "已生成 model-config/config.json（此前是隐藏硬依赖，仓库里没有）"
else
  dim "  model-config/config.json 已存在，保留（含 UI 里改过的模型选择）"
fi

# ── systemd 单元 ──
# 装到 ~/.config/systemd/user/ 而**不写回仓库**：
# 仓库里的单元保持原样，拷包不会带上一台机器的路径。
# 用 cp 而不是 start.sh 的 `systemctl --user link`——link 之后无法 disable。
UNIT_DIR="$(user_unit_dir)"
mkdir -p "$UNIT_DIR"

install_unit() {
  local name="$1" content="$2"
  local target="$UNIT_DIR/$name"
  if write_if_different "$target" "$content"; then
    UNIT_CHANGED=1
    info "已更新单元：$name"
  else
    dim "  单元未变：$name"
  fi
}

SD_ROOT="$(systemd_path "$HDW_ROOT")"
case "$SD_ROOT" in
  %h/*) dim "  项目在家目录下，单元用 %h/ 相对形式（家目录内移动无需重装）" ;;
  *)    warn "项目不在 \$HOME 下，单元只能写绝对路径；移动后需要重跑 deploy.sh" ;;
esac

UNIT_CHANGED=0

install_unit "hyperdrivewave-llama.service" "[Unit]
Description=HyperDriveWave llama.cpp local inference
After=default.target

[Service]
Type=simple
WorkingDirectory=$SD_ROOT
EnvironmentFile=-$SD_ROOT/Configs/.env
ExecStart=$SD_ROOT/HDW_Inference/llama/start.sh
Restart=always
RestartSec=3
TimeoutStopSec=60
KillSignal=SIGINT

[Install]
WantedBy=default.target
"

install_unit "hyperdrivewave-resource-coordinator.service" "[Unit]
Description=HyperDriveWave GPU resource coordinator
After=default.target

[Service]
Type=simple
WorkingDirectory=$SD_ROOT
EnvironmentFile=-$SD_ROOT/Configs/.env
Environment=HDW_PROJECT_ROOT=$SD_ROOT
Environment=HDW_MODEL_CONFIG_PATH=$SD_ROOT/HDW_Runtime/model-config/config.json
Environment=HDW_MAINTENANCE_SOCKET=$SD_ROOT/HDW_Runtime/maintenance/maintenance.sock
ExecStart=/usr/bin/python3 $SD_ROOT/Scripts/resource_coordinator.py
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
"

# frpc 只在需要外网访问时装。装了也不 enable——启停由控制中心的
# 「外网访问」页控制（走 coordinator 的 /frp-enable），这里抢先 enable 会打架。
if [ "$WITH_FRP" = "1" ]; then
  install_unit "hyperdrivewave-frpc.service" "[Unit]
Description=HyperDriveWave public FRP client
After=default.target

[Service]
Type=simple
WorkingDirectory=$SD_ROOT
ExecStart=$SD_ROOT/HDW_Frontend/FRP/bin/frpc -c $SD_ROOT/HDW_Frontend/FRP/conf/frpc_hdw_public.toml
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
"
else
  dim "  跳过 frpc 单元（默认不部署外网访问，需要时加 --with-frp）"
fi

systemctl --user daemon-reload

# ── dry-run 到此为止 ──
# 后面的建镜像 / 启动会真的重启在跑的服务，不适合"只想看看会改什么"的场景。
if [ "$DRY_RUN" = "1" ]; then
  log ""
  info "--dry-run：已完成前置检查、探测与配置渲染，未构建镜像、未启动服务。"
  log ""
  _units="hyperdrivewave-llama.service hyperdrivewave-resource-coordinator.service"
  [ "$WITH_FRP" = "1" ] && _units="$_units hyperdrivewave-frpc.service"
  log "  决策快照   ：$PLAN_FILE"
  log "  推理后端   ：$BACKEND_REL（device=$LLAMA_DEVICE）"
  log "  系统单元   ：$UNIT_DIR/"
  for _u in $_units; do log "               $_u"; done
  log ""
  log "  去掉 --dry-run 重跑即真正部署。"
  log ""
  exit 0
fi

# ═══ 阶段 3：第三方依赖 ═══════════════════════════════════════
# 排在模型之前：镜像构建依赖源码（hdw-mineru 的 build.context 就是 MinerU 仓库），
# 源码不到位的话后面 build 一定失败。

step "第三方依赖"
bash "$SCRIPT_DIR/fetch_vendors.sh" --network "$NET_MODE" ${PREFER_HOST:+--prefer "$PREFER_HOST"} || {
  warn "第三方依赖未全部就绪。缺 MinerU 会导致 hdw-mineru 镜像构建失败；
     缺 aora-bot 会导致 WebUI 的情绪球加载不出来。"
}

# ═══ 阶段 4：模型 ═════════════════════════════════════════════

if [ "$SKIP_MODELS" = "1" ]; then
  step "模型检查（已跳过）"
else
  step "检查模型权重"
  bash "$SCRIPT_DIR/fetch_models.sh" --network "$NET_MODE" ${PROXY:+--proxy "$PROXY"}
fi

# ═══ 阶段 5：镜像 ═════════════════════════════════════════════
# 顺序有讲究：hdw-mineru 的 Dockerfile 第一行是 FROM hyperdrivewave-hdw-rag:latest，
# 但 compose 里 mineru 没声明对 rag 的 depends_on，`up --build` 的构建顺序无保证。

if [ "$NET_MODE" = "offline" ]; then
  step "准备镜像（离线）"
  bash "$SCRIPT_DIR/deploy_images_offline.sh" 2>/dev/null || \
    warn "离线镜像准备脚本不可用；请确保所需镜像已 docker load 过"
else
  step "构建镜像"
  info "先单独构建 hdw-rag（hdw-mineru 的 FROM 依赖它，无法靠 compose 排序保证）"
  ( cd "$HDW_ROOT/Configs" && \
    docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" build ${PROXY:+--build-arg "HTTPS_PROXY=$PROXY" --build-arg "HTTP_PROXY=$PROXY"} hdw-rag )
  ok "hdw-rag 镜像就绪"
fi

# ═══ 阶段 6：调用既有启动入口 ═══════════════════════════════
# 不在这里重写启动逻辑：项目的启动入口是 Scripts/start.sh（README §10.3 的契约），
# 它已经处理了 profile 解析、协调器先起、socket 等待、llama 健康轮询。
# 部署脚本重写一遍只会让两条路径逐渐分叉。

step "启动服务"
bash "$SCRIPT_DIR/start.sh"

# ═══ 阶段 7：验收 ═════════════════════════════════════════════

if [ "$SKIP_VERIFY" = "1" ]; then
  step "验收（已跳过）"
else
  step "验收"
  bash "$SCRIPT_DIR/deploy_verify.sh" || {
    log ""
    warn "验收未全部通过。上面标 [失败] 的项需要处理后再跑一次：bash Scripts/deploy_verify.sh"
  }
fi

# ═══ 摘要 ═════════════════════════════════════════════════════

log ""
log "════════════════ 部署摘要 ════════════════"
log "项目根    ：$HDW_ROOT"
log "推理后端  ：$BACKEND_REL（device=$LLAMA_DEVICE）"
log "GPU       ：$GPU_SUMMARY"
log "网络模式  ：$NET_MODE"
log "WebUI     ：http://$BIND_IP:${HDW_WEBUI_PORT:-3000}"
log ""

# 不开 linger 的话用户级服务只在登录会话存在时运行，重启后 llama/协调器不会自启。
# 这是「部署完看着好、重启就没了」的元凶，所以顺手做掉（需要 sudo 密码）。
if ! linger_enabled; then
  info "开启 linger（重启后用户级服务才能自启，需要 sudo）"
  if sudo loginctl enable-linger "$(whoami)"; then
    ok "linger 已开启"
  else
    warn "开启 linger 失败。手动执行：sudo loginctl enable-linger $(whoami)
     不开的话重启机器后 llama / 协调器不会自动起来，需要先登录一次。"
  fi
fi

log "不可用的能力（如有）："
if [ "$HDW_GPU_COUNT" -eq 0 ]; then
  log "  · 无 NVIDIA 卡：本地推理走 CPU，速度很低；知识入库的 GPU 加速路径（/prepare）不可用"
fi
case "$BACKEND_REL" in
  *build-cuda*) : ;;
  *) log "  · 非 CUDA 后端：MTP 已失效，解码吞吐明显下降" ;;
esac
if [ "$WITH_FRP" != "1" ]; then
  log "  · 外网访问：未部署 frpc（需要时重跑 deploy.sh --with-frp）"
fi

log ""
log "下一步："
log "  1. 灌知识库（若之前没灌过）：bash Scripts/ingest_knowledge.sh <文档目录>"
log "  2. 接远端 RAG 节点（若有）：bash Scripts/deploy_remote_rag.sh --help"
log "  3. 日常巡检：bash Scripts/healthcheck.sh    完整验收：bash Scripts/deploy_verify.sh"
log ""
if [ "${HDW_WARNINGS:-0}" -gt 0 ]; then
  log "本次有 $HDW_WARNINGS 条警告，请回看上面的 [警告]。"
fi
ok "部署流程结束"
