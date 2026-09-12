#!/usr/bin/env bash
# 在远端 GPU 服务器上只部署 RAG 节点（不需要主站那套 WebUI/数据库/llama）。
#
# 前提：本脚本必须位于 <REMOTE_ROOT>/Scripts/ 下，且 <REMOTE_ROOT>/HDW_Engines 与
# <REMOTE_ROOT>/HDW_Runtime 存在——也就是「把整个 HDW 文件夹拷过来」的那种布局。
# 脚本靠 BASH_SOURCE 自定位，没有别的假设。
#
# 用法：
#   bash Scripts/deploy_remote_rag.sh                      # 自动探测，交互选镜像来源
#   bash Scripts/deploy_remote_rag.sh --image-mode load --image-tar /media/hdw-rag.tar.gz
#   bash Scripts/deploy_remote_rag.sh --image-mode save-ssh --image-from user@<主站IP>
#   bash Scripts/deploy_remote_rag.sh --image-mode build
#   bash Scripts/deploy_remote_rag.sh --bind 10.0.0.9 --admin-token <token>
#
# 完成后**必须回到主站**跑一次 Scripts/sync_remote_rag.sh 把 chunks.jsonl 推过来，
# 否则远端 zvec 是空的，/search 会报 index not found。

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HDW_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export HDW_ROOT

# 放在 RUNTIME_ROOT 下而不是仓库里：它是**本机专属**的生成物，
# 写回仓库会让这个文件夹拷到下一台机器时带着上一台的 IP 和卡号。
RUNTIME_ROOT="$HDW_ROOT/HDW_Runtime"
OUT_DIR="$RUNTIME_ROOT/remote-rag"
COMPOSE_OUT="$OUT_DIR/remote-compose.yml"
ENV_OUT="$OUT_DIR/remote-rag.env"

. "$SCRIPT_DIR/lib/common.sh"
. "$SCRIPT_DIR/lib/detect.sh"

IMAGE_MODE=""
IMAGE_TAR=""
IMAGE_FROM=""
BIND_IP=""
ADMIN_TOKEN=""
RAG_IMAGE="hyperdrivewave-hdw-rag:latest"
PORT_BASE=8001
SKIP_MODELS=0
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --image-mode)  IMAGE_MODE="${2:-}"; shift 2 ;;
    --image-tar)   IMAGE_TAR="${2:-}"; shift 2 ;;
    --image-from)  IMAGE_FROM="${2:-}"; shift 2 ;;
    --bind)        BIND_IP="${2:-}"; shift 2 ;;
    --admin-token) ADMIN_TOKEN="${2:-}"; shift 2 ;;
    --skip-models) SKIP_MODELS=1; shift ;;
    --dry-run)     DRY_RUN=1; shift ;;
    -h|--help)     sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done

# ═══ 0. 前置检查 ══════════════════════════════════════════════

step "前置检查"

need_cmd docker ""
docker compose version >/dev/null 2>&1 || die "需要 docker compose v2"
need_cmd curl ""
# 只有"远端自己从魔搭下模型"这条路径需要 python3；下面按需再查
if ! docker info >/dev/null 2>&1; then
  die "当前用户没有 docker 权限。加进 docker 组：sudo usermod -aG docker $(whoami) 然后重新登录"
fi

FREE_GB="$(disk_free_gb "$HDW_ROOT")"
info "项目根：$HDW_ROOT"
info "磁盘可用：${FREE_GB}G"

# ═══ 1. 探测 GPU ══════════════════════════════════════════════

step "探测 GPU"

detect_gpu
info "GPU：$HDW_GPU_COUNT 张"

# 服务数量按实际卡数定，不硬凑成 2 个：
#   · 0 卡 —— 起 1 个 CPU 服务（**必须去掉 device reservation**，否则容器直接起不来）
#   · 1 卡 —— 起 1 个（在同一张卡上跑两个容器只会互相抢，吞吐反而更差）
#   · 2 卡 —— 起 2 个，各占一张
if [ "$HDW_GPU_COUNT" -eq 0 ]; then
  RAG_SERVICES=1
  USE_GPU=0
  warn "没有 NVIDIA 卡：将以 CPU 模式起 1 个 RAG 服务（嵌入/重排会明显变慢）"
elif [ "$HDW_GPU_COUNT" -eq 1 ]; then
  RAG_SERVICES=1
  USE_GPU=1
  info "单卡 → 起 1 个 RAG 服务（占满这张卡，不硬凑第二个）"
else
  RAG_SERVICES=2
  USE_GPU=1
  info "多卡（$HDW_GPU_COUNT）→ 起 2 个 RAG 服务，各占 GPU 0 / GPU 1"
fi

if [ "$USE_GPU" = "1" ]; then
  if ! docker_has_nvidia_runtime; then
    warn "Docker 里找不到 nvidia runtime（没装 nvidia-container-toolkit？）。
     降级为 CPU 模式。装好后重跑本脚本即可启用 GPU。
     安装：sudo apt install nvidia-container-toolkit && sudo systemctl restart docker"
    USE_GPU=0
  fi
  if [ "$HDW_GPU_VRAM_MIB" -lt 12288 ]; then
    warn "选中卡显存只有 ${HDW_GPU_VRAM_MIB}MiB：bge-m3 + 重排常驻约 7GB，可能吃紧"
  fi
fi

[ -n "$BIND_IP" ] || BIND_IP="$(detect_lan_ip)"
info "对外绑定地址：$BIND_IP"

# ═══ 2. 目录 ══════════════════════════════════════════════════
# 属主必须是当前用户：Docker 会自动以 root 建缺失的挂载点，
# 之后普通用户想清理或 rsync 同步都会很别扭。

step "准备目录"

for i in $(seq 0 $((RAG_SERVICES - 1))); do
  mkdir -p "$RUNTIME_ROOT/zvec-gpu$i"
done
mkdir -p "$RUNTIME_ROOT/rag" "$OUT_DIR"
info "已建 zvec-gpu0${RAG_SERVICES:+/…}、rag、remote-rag"

# ═══ 3. 模型 ══════════════════════════════════════════════════
# 远端只需要嵌入和重排两个模型（6.9G），不需要 LLM 权重。

if [ "$SKIP_MODELS" = "1" ]; then
  step "模型（已跳过）"
else
  step "检查 RAG 模型"
  # fetch_models.sh 的 --network 由环境变量透传，方便离线包场景
  bash "$SCRIPT_DIR/fetch_models.sh" --only rag --network "${HDW_DEPLOY_NET:-online}"
fi

# ═══ 4. 镜像 ══════════════════════════════════════════════════

step "准备 RAG 镜像"

image_present() { docker image inspect "$RAG_IMAGE" >/dev/null 2>&1; }

if image_present; then
  ok "镜像已存在：$RAG_IMAGE"
else
  # 没指定就按可行性自动选
  if [ -z "$IMAGE_MODE" ]; then
    if [ -n "$IMAGE_TAR" ] && [ -f "$IMAGE_TAR" ]; then
      IMAGE_MODE=load
    elif [ -n "$IMAGE_FROM" ]; then
      IMAGE_MODE=save-ssh
    elif probe_modelscope >/dev/null 2>&1; then
      IMAGE_MODE=build
    else
      die "没有可用的镜像来源。三选一：
     --image-tar   <已有的 docker save 包>
     --image-from  <主站 ssh 目标，如 user@<主站IP>>
     --image-mode build  （需要能连公网 PyPI）"
    fi
    info "自动选择镜像来源：$IMAGE_MODE"
  fi

  case "$IMAGE_MODE" in
    load)
      [ -n "$IMAGE_TAR" ] && [ -f "$IMAGE_TAR" ] || die "--image-mode load 需要 --image-tar <文件>"
      info "从 tar 载入（$IMAGE_TAR）…"
      # 兼容 .tar 和 .tar.gz
      case "$IMAGE_TAR" in
        *.gz) gunzip -c "$IMAGE_TAR" | docker load ;;
        *)    docker load -i "$IMAGE_TAR" ;;
      esac || die "docker load 失败"
      ;;

    save-ssh)
      [ -n "$IMAGE_FROM" ] || die "--image-mode save-ssh 需要 --image-from <user@host>"
      # 直接管道过去，不在本地落盘：9.4G 的中间文件很容易把盘塞满
      info "从 $IMAGE_FROM 传输镜像（约 9.4G，压缩后 ~3.5G，视网速可能几分钟）…"
      ssh -o ConnectTimeout=10 "$IMAGE_FROM" \
        "docker save '$RAG_IMAGE' | gzip -1" | gunzip | docker load \
        || die "从 $IMAGE_FROM 拉取镜像失败（检查 ssh 免密和那边是否已构建）"
      ;;

    build)
      info "在远端本地构建（需要能连公网 PyPI，torch 约 2.5G，会比较慢）…"
      ( cd "$HDW_ROOT/HDW_Inference/RAG_Service" && \
        docker build -t "$RAG_IMAGE" . ) || die "镜像构建失败"
      ;;

    *) die "--image-mode 只能是 load / save-ssh / build" ;;
  esac

  image_present || die "镜像 $RAG_IMAGE 仍不可用"
  ok "镜像就绪"
fi

# ═══ 5. 生成 compose ══════════════════════════════════════════

step "生成 remote-compose.yml"

# admin token：远端和主站必须一致，否则主站 POST /admin/reindex 会 401
if [ -z "$ADMIN_TOKEN" ]; then
  if [ -f "$HDW_ROOT/Configs/.env" ]; then
    ADMIN_TOKEN="$(env_get "$HDW_ROOT/Configs/.env" HDW_RAG_ADMIN_TOKEN || true)"
  fi
fi
[ -n "$ADMIN_TOKEN" ] || die "拿不到 HDW_RAG_ADMIN_TOKEN。
   --admin-token <token> 显式指定，或确保 Configs/.env 里有这一项"

{
  echo "name: hyperdrivewave-rag-remote"
  echo ""
  echo "services:"
  for i in $(seq 0 $((RAG_SERVICES - 1))); do
    port=$((PORT_BASE + i * 2))
    echo "  hdw-rag-gpu$i:"
    echo "    image: $RAG_IMAGE"
    echo "    container_name: hdw-rag-gpu$i"
    echo "    restart: unless-stopped"
    if [ "$USE_GPU" = "1" ]; then
      echo "    deploy:"
      echo "      resources:"
      echo "        reservations:"
      echo "          devices:"
      echo "            - driver: nvidia"
      echo "              device_ids: [\"$i\"]"
      echo "              capabilities: [gpu]"
    fi
    echo "    environment:"
    if [ "$USE_GPU" = "1" ]; then
      echo "      CUDA_VISIBLE_DEVICES: \"0\""
      echo "      HDW_RAG_DEVICE: cuda"
    else
      echo "      HDW_RAG_DEVICE: cpu"
    fi
    echo "      HDW_BGE_M3_PATH: /models/RAG_Models/bge-m3"
    echo "      HDW_RERANKER_PATH: /models/RAG_Models/bge-reranker-v2-m3"
    echo "      HDW_RAG_INDEX_PATH: /data/rag/chunks.jsonl"
    echo "      HDW_ZVEC_COLLECTION_PATH: /data/zvec/industrial_chunks"
    echo "      HDW_EMBED_MAX_LENGTH: \"1024\""
    echo "      HDW_RERANK_MAX_LENGTH: \"512\""
    echo "      HDW_RERANK_CANDIDATES: \"50\""
    echo "      HDW_RAG_ADMIN_TOKEN: \${HDW_RAG_ADMIN_TOKEN:?HDW_RAG_ADMIN_TOKEN is required}"
    echo "    volumes:"
    echo "      # 相对 compose 文件所在的 HDW_Runtime/remote-rag/，指回项目根"
    echo "      - ../../HDW_Engines:/models:ro"
    echo "      - ../rag:/data/rag"
    echo "      - ../zvec-gpu$i:/data/zvec"
    echo "    ports:"
    echo "      - \"$BIND_IP:$port:8001\""
    echo ""
  done
} > "$COMPOSE_OUT"

# token 不能世界可读
write_if_different "$ENV_OUT" "HDW_RAG_ADMIN_TOKEN=$ADMIN_TOKEN
" >/dev/null
chmod 600 "$ENV_OUT"

info "已生成 $COMPOSE_OUT（$RAG_SERVICES 个服务，GPU=$USE_GPU）"

# ═══ 6. 启动 ══════════════════════════════════════════════════
# 绝不用 --build：镜像必须是"被验证过的那一个"，远端不重新构建。
# 注意本脚本刻意不开 set -e：下面的验收要把所有项跑完再汇总。

step "启动 RAG 服务"

if [ "$DRY_RUN" = "1" ]; then
  # 只生成不启动：方便先看生成结果，也方便在非目标机上验证逻辑
  info "--dry-run：已生成配置但不启动。检查生成结果："
  log ""
  sed 's/^/    /' "$COMPOSE_OUT"
  log ""
  log "确认无误后去掉 --dry-run 重跑即可。"
  exit 0
fi

docker compose -f "$COMPOSE_OUT" --env-file "$ENV_OUT" up -d \
  || die "compose up 失败"

info "等待服务就绪…"

# ═══ 7. 验收 ══════════════════════════════════════════════════
# /health 是**写死的静态字典**（status 恒为 ok，只查路径存在性），
# 而模型是首次调用才懒加载的。所以必须真调一次 /embed。

step "验收"

PASS=0; FAIL=0
for i in $(seq 0 $((RAG_SERVICES - 1))); do
  port=$((PORT_BASE + i * 2))
  url="http://$BIND_IP:$port"

  ready=0
  for _ in $(seq 1 30); do
    curl -fsS --max-time 5 "$url/health" >/dev/null 2>&1 && { ready=1; break; }
    sleep 2
  done
  if [ "$ready" != "1" ]; then
    printf '  %s✗%s gpu%d ($url) /health 不通\n' "$_C_RED" "$_C_RESET" "$i"
    FAIL=$((FAIL + 1)); continue
  fi

  # 真调一次 /embed：冷加载 bge-m3 可能 10-30s，给足超时
  dim="$(curl -fsS --max-time 300 -H 'Content-Type: application/json' \
    -d '{"texts":["远端部署自检"]}' "$url/embed" 2>/dev/null \
    | python3 -c "
import json,sys
try: print(json.load(sys.stdin).get('dim') or 0)
except Exception: print(0)
" 2>/dev/null)"

  if [ "${dim:-0}" -gt 0 ] 2>/dev/null; then
    printf '  %s✓%s gpu%d ($url) /embed 正常，dim=%s\n' "$_C_GRN" "$_C_RESET" "$i" "$dim"
    PASS=$((PASS + 1))
  else
    printf '  %s✗%s gpu%d ($url) /health 通但 /embed 失败（CUDA 不可用或模型缺失）\n' \
      "$_C_RED" "$_C_RESET" "$i"
    FAIL=$((FAIL + 1))
  fi
done

# 挂载目录属主：Docker 会把不存在的挂载点建成 root
for d in "$RUNTIME_ROOT/rag" "$RUNTIME_ROOT"/zvec-gpu*; do
  [ -d "$d" ] || continue
  owner="$(stat -c '%U' "$d" 2>/dev/null)"
  if [ "$owner" = "root" ]; then
    warn "$d 属主是 root —— 之后 rsync/清理会需要 sudo。可执行：
     sudo chown -R $(whoami):$(whoami) $d"
  fi
done

# ═══ 8. 交接信息 ══════════════════════════════════════════════

log ""
log "════════════════ 远端 RAG 部署结果 ════════════════"
printf '  通过 %s%d%s   失败 %s%d%s\n' "$_C_GRN" "$PASS" "$_C_RESET" "$_C_RED" "$FAIL" "$_C_RESET"

if [ "$FAIL" -gt 0 ]; then
  log ""
  log "排查：docker compose -f $COMPOSE_OUT logs --tail 80"
  exit 1
fi

# 主站必须把远端地址配成**完全一致**的列表。
# 单卡时若主站还列着 8003，每次问答都有一半请求打到不存在的端口，
# 每个都要吃一次 HDW_RAG_CONNECT_TIMEOUT（默认 3s）。
URLS=""
for i in $(seq 0 $((RAG_SERVICES - 1))); do
  port=$((PORT_BASE + i * 2))
  URLS="${URLS:+$URLS,}http://$BIND_IP:$port"
done

log ""
log "在**主站**的 Configs/.env 里设置这一行（照抄，别漏端口）："
log ""
log "  HDW_RAG_REMOTE_URLS=$URLS"
log ""
log "然后回主站执行，把 chunks.jsonl 推过来并重建远端索引："
log ""
log "  cd <主站项目根>"
log "  SSHPASS='<远端密码>' bash Scripts/sync_remote_rag.sh"
log ""
log "注意：sync_remote_rag.sh 用的是主站 .env 里的 HDW_REMOTE_RAG_SSH_TARGET /"
log "HDW_REMOTE_RAG_ROOT，记得改成这台机器（当前用户 $(whoami)@$(hostname)）："
log "  HDW_REMOTE_RAG_SSH_TARGET=$(whoami)@$BIND_IP"
log "  HDW_REMOTE_RAG_ROOT=$HDW_ROOT"
log ""

ok "远端 RAG 部署完成"
