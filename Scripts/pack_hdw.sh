#!/usr/bin/env bash
# 在源机打包出可分发到别的服务器的精简包。
#
# 为什么需要它：整个文件夹 253G，其中 239G 是**当前没在用**的备份模型，
# 另有 34G 是之前下载残留的碎块。直接 rsync 整个文件夹会白搬 270G。
#
# 用法：
#   bash Scripts/pack_hdw.sh --out /media/usb/hdw          # 拷到目录（推荐，可续传）
#   bash Scripts/pack_hdw.sh --tar /tmp/hdw.tar.gz         # 打成单个压缩包
#   bash Scripts/pack_hdw.sh --app-only --out /media/usb/hdw   # 不带模型（目标机自己下）
#   bash Scripts/pack_hdw.sh --remote-rag --out /media/usb/rag # 只打远端 RAG 节点要的
#   bash Scripts/pack_hdw.sh --from-manifest need.tsv --out DIR # 按目标机的缺失清单打
#
# 默认包含：全部代码与配置 + 当前在用的 3 个模型（约 26G）。

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HDW_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export HDW_ROOT

. "$SCRIPT_DIR/lib/common.sh"
. "$SCRIPT_DIR/lib/models.sh"
. "$SCRIPT_DIR/lib/detect.sh"

OUT_DIR=""
TAR_FILE=""
INCLUDE_MODELS=1
REMOTE_RAG_ONLY=0
FROM_MANIFEST=""

while [ $# -gt 0 ]; do
  case "$1" in
    --out)           OUT_DIR="${2:-}"; shift 2 ;;
    --tar)           TAR_FILE="${2:-}"; shift 2 ;;
    --app-only)      INCLUDE_MODELS=0; shift ;;
    --remote-rag)    REMOTE_RAG_ONLY=1; shift ;;
    --from-manifest) FROM_MANIFEST="${2:-}"; shift 2 ;;
    -h|--help)       sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done

[ -n "$OUT_DIR" ] || [ -n "$TAR_FILE" ] || die "必须指定 --out <目录> 或 --tar <文件>"
[ -z "$OUT_DIR" ] || [ -z "$TAR_FILE" ] || die "--out 和 --tar 只能选一个"

# ═══ 排除清单 ═════════════════════════════════════════════════
# 每条都写清理由，避免后人"看着多余就删了"。

EXCLUDES=(
  # ── 用不上的备份模型（约 239G）──
  # 只保留 model_manifest 里那三个；其余是开发期试过的模型。
  "--exclude=/HDW_Engines/LLM_Models/Qwen3.6-35B-A3B"
  "--exclude=/HDW_Engines/LLM_Models/Qwen3.8"
  "--exclude=/HDW_Engines/LLM_Models/Qwen3.8-27B-FP8"
  "--exclude=/HDW_Engines/LLM_Models/Qwen3.8-27B-GGUF"
  "--exclude=/HDW_Engines/LLM_Models/Qwen3.8_MoE"

  # ── 数据库/缓存的裸数据目录（约 200M）──
  # 属主是容器内的 uid（host 上映射成 dnsmasq/root），普通用户拷不动；
  # 而且裸拷运行中的 PG 数据目录本就不可靠。目标机重建即可。
  "--exclude=/HDW_Runtime/postgres"
  "--exclude=/HDW_Runtime/redis"
  "--exclude=/HDW_Runtime/neo4j/data"
  "--exclude=/HDW_Runtime/neo4j/logs"

  # ── 运行态产物 ──
  "--exclude=/HDW_Runtime/backups"
  "--exclude=/HDW_Runtime/frp/*.log"
  "--exclude=/HDW_Runtime/llama-server.pid"
  "--exclude=/HDW_Runtime/.deploy"
  "--exclude=/HDW_Runtime/remote-rag"   # 本机专属生成物（含 IP、卡号）

  # ── 构建缓存 ──
  # .venv 的 console script shebang 写死绝对路径，拷过去就是坏的；
  # fetch_models.sh 会按需重建，不必带。
  "--exclude=/.venv"
  "--exclude=__pycache__"
  "--exclude=*.py[cod]"
  "--exclude=*.bak_*"
  "--exclude=*.bak-*"

  # ── 死文件 / 陈旧备份 ──
  "--exclude=/Configs/.env.before-remote-rag-*"
  "--exclude=/Configs/docker-compose.gpu.yml"      # 全项目零引用的早期方案残留
  "--exclude=/HDW_Frontend/FRP/control"            # 上游遗留脚本，路径指向本机另一个项目

  # ── 早期在源码树里编的重复产物 ──
  # 真正在用的是 llama.cpp-upstream/build*；这个是 217M 的重复品。
  "--exclude=/HDW_Inference/llama/build-cuda"

  # ── vendored 第三方 ──
  # 它们由 vendor/vendor.lock + Scripts/fetch_vendors.sh 管理，
  # 目标机自己克隆（已确认的决策）。带上就白搬 3.4G，其中 2.7G 是 git 历史。
  "--exclude=/HDW_Knowledge/MinerU"
  "--exclude=/HDW_Animation/aora-bot"
  "--exclude=/HDW_Inference/FreeToken"
  "--exclude=/HDW_Orchestrator/dify"
  "--exclude=/HDW_Orchestrator/n8n"
  "--exclude=/HDW_Orchestrator/langgraph"
  "--exclude=/HDW_Security/keycloak"
  "--exclude=/HDW_Ops/langfuse"
  "--exclude=/HDW_Frontend/open-webui"
  "--exclude=/HDW_VectorDB/zvec"

  # ── llama.cpp-upstream 里对"部署"无用的部分 ──
  # 注意**不能整个排除这个目录**：build-cuda-multi（1013M，含随包 CUDA runtime）
  # 就在它里面，那才是目标机真正要跑的推理后端。只砍掉：
  #   · .git —— 38M 的上游历史，部署用不上
  #   · build-cuda —— 225M 的单架构(sm_120a)产物，已被多架构版取代
  # 源码（172M）保留：目标机若显卡架构不在 sm_80/89/90/120 里，需要就地重编。
  "--exclude=/HDW_Inference/llama/llama.cpp-upstream/.git"
  "--exclude=/HDW_Inference/llama/llama.cpp-upstream/build-cuda"
)

# 远端 RAG 不需要的东西：主站那套编排、WebUI、数据库、llama、知识图谱，
# 以及文档解析（MinerU）和 MCP —— 那两样只在主站跑。
REMOTE_RAG_EXCLUDES=(
  "--exclude=/HDW_Orchestrator"
  "--exclude=/HDW_Frontend"
  "--exclude=/HDW_KnowledgeGraph"
  "--exclude=/HDW_Animation"
  "--exclude=/HDW_Evaluation"
  "--exclude=/HDW_Ops"
  "--exclude=/HDW_Security"
  "--exclude=/HDW_VectorDB"
  "--exclude=/HDW_Knowledge"
  "--exclude=/HDW_DataFoundation"
  "--exclude=/Configs/docker-compose.yml"
  "--exclude=/Configs/docker-compose.*.yml"
  "--exclude=/HDW_Inference/llama"
  "--exclude=/HDW_Inference/FreeToken"
  "--exclude=/HDW_Engines/LLM_API"
  "--exclude=/HDW_Runtime"
  "--exclude=/Scripts/deploy.sh"
  "--exclude=/Scripts/deploy_verify.sh"
  "--exclude=/Scripts/ingest_knowledge.sh"
  "--exclude=/Scripts/sync_remote_rag.sh"
  "--exclude=/Scripts/resource_coordinator.py"
  "--exclude=/Scripts/prepare_dirs.sh"
  "--exclude=/Scripts/start.sh"
  "--exclude=/Scripts/stop.sh"
  "--exclude=/Scripts/restart.sh"
  "--exclude=/Scripts/status.sh"
  "--exclude=/Scripts/logs.sh"
  "--exclude=/Scripts/healthcheck.sh"
  "--exclude=/Scripts/backup.sh"
  "--exclude=/Scripts/restore.sh"
  "--exclude=/Scripts/*.service"
)
# 但远端 RAG 需要：HDW_Engines（只 RAG 模型）、RAG_Service 源码、lib、本脚本

if [ -n "$FROM_MANIFEST" ]; then
  [ -f "$FROM_MANIFEST" ] || die "找不到清单文件：$FROM_MANIFEST"
  INCLUDE_MODELS=0   # 由清单决定带什么，不再整目录带模型
fi

if [ "$REMOTE_RAG_ONLY" = "1" ]; then
  EXCLUDES+=("${REMOTE_RAG_EXCLUDES[@]}")
  # 远端只要 RAG 模型
  EXCLUDES+=("--exclude=/HDW_Engines/LLM_Models")
fi

# --app-only 要连**在用**的模型一起排除，否则 rsync 照样会把 19G 搬过去，
# 后面的占位目录逻辑就白做了。
if [ "$INCLUDE_MODELS" = "0" ]; then
  EXCLUDES+=("--exclude=/HDW_Engines/LLM_Models/*")
  EXCLUDES+=("--exclude=/HDW_Engines/RAG_Models/*")
fi

# ═══ 组装 rsync 参数 ══════════════════════════════════════════

# 进度条只在终端下开。重定向到日志时 progress2 会把每一行都刷成
# `\r` 分隔的进度片段，几万行输出会把真正的报错埋掉——
# 上面"目录空了"那次排查就是被它拖慢的。
if [ -t 1 ]; then
  RSYNC_ARGS=(-aHAX --info=progress2,stats2)
else
  RSYNC_ARGS=(-aHAX --info=name0,stats2)
fi

if [ -n "$TAR_FILE" ]; then
  # tar 的 --exclude 匹配的是**归档内成员名**（这里是 `./HDW_Engines/...`），
  # 而 rsync 的 `/X` 是相对传输根锚定。语义不同，所以两种写法都下发，
  # 让同一份 EXCLUDES 在两个模式下都能命中。
  TAR_EXCLUDES=()
  for e in "${EXCLUDES[@]}"; do
    pat="${e#--exclude=}"
    TAR_EXCLUDES+=("--exclude=./${pat#/}")
    TAR_EXCLUDES+=("--exclude=${pat#/}")
  done

  info "打包到 $TAR_FILE（单流压缩，26G 大约要十几分钟）"
  tar -C "$HDW_ROOT" "${TAR_EXCLUDES[@]}" -cf - . \
    | gzip -1 > "$TAR_FILE" \
    || die "打包失败"
  ok "已生成 $TAR_FILE（$(du -h "$TAR_FILE" | cut -f1)）"
  log ""
  log "目标机上解包后跑：bash Scripts/deploy.sh"
  exit 0
fi

mkdir -p "$OUT_DIR"
[ "$(cd "$OUT_DIR" && pwd)" != "$HDW_ROOT" ] || die "--out 不能是项目根目录自己"

info "同步到 $OUT_DIR"
log ""

# ── 空间预检 ──
# rsync 写到一半空间不足时，前面的目录是好的、后面的目录是空的，
# 而且它不一定返回非零——产出一个**静默不完整**的包，
# 要到目标机上跑部署才发现。所以先算准需要多少再开工。
NEED_BYTES="$(rsync -aHAX --dry-run --stats "${EXCLUDES[@]}" "$HDW_ROOT/" "$OUT_DIR/" 2>/dev/null \
  | sed -n 's/^Total file size: \([0-9,]*\) bytes.*/\1/p' | tr -d ',')"
if [ -n "$NEED_BYTES" ]; then
  AVAIL_BYTES=$(( $(disk_free_gb "$OUT_DIR" 2>/dev/null || echo 0) * 1000000000 ))
  NEED_GB=$(( NEED_BYTES / 1000000000 ))
  AVAIL_GB=$(( AVAIL_BYTES / 1000000000 ))
  info "预计需要 ${NEED_GB}G，目标位置可用 ${AVAIL_GB}G"
  # 留 5G 余量给文件系统和元数据
  if [ "$NEED_BYTES" -gt $(( AVAIL_BYTES - 5000000000 )) ]; then
    die "空间不足：需要 ${NEED_GB}G，只有 ${AVAIL_GB}G。
     换个目标位置，或用 --app-only 只打代码（模型到目标机再下）。"
  fi
fi

# --delete 让产物是严格镜像：重跑不会留下一版的残file
rsync "${RSYNC_ARGS[@]}" --delete "${EXCLUDES[@]}" "$HDW_ROOT/" "$OUT_DIR/" \
  || die "rsync 失败"

# ═══ 模型：按清单逐个带（--from-manifest 时只带清单里的）═══════

if [ -n "$FROM_MANIFEST" ]; then
  info "按清单补齐模型…"
  MISSING_COUNT=0
  while IFS=$'\t' read -r kind repo file rel want abs; do
    [ -n "${rel:-}" ] || continue
    case "$kind" in model) ;; *) continue ;; esac
    src="$HDW_ROOT/$rel"
    dst="$OUT_DIR/$rel"
    if [ ! -e "$src" ]; then
      warn "源机上也没有：$rel —— 这一项得从别处拿"
      MISSING_COUNT=$((MISSING_COUNT + 1))
      continue
    fi
    mkdir -p "$(dirname "$dst")"
    rsync -aHAX --info=progress2 "$src" "$(dirname "$dst")/" 2>/dev/null
    info "  + $rel"
  done < "$FROM_MANIFEST"
  [ "$MISSING_COUNT" -eq 0 ] || warn "有 $MISSING_COUNT 项源机也没有，目标机仍会缺"
elif [ "$INCLUDE_MODELS" = "1" ]; then
  info "校验并带上当前在用的模型…"
  while IFS=$'\t' read -r kind repo file rel want; do
    [ -n "${kind:-}" ] || continue
    st="$(model_status "$kind" "$repo" "$file" "$rel" "$want")"
    if [ "$st" != "OK" ]; then
      warn "源机上这一项就不完整（$st）：$rel —— 先跑 bash Scripts/fetch_models.sh"
      continue
    fi
    dst="$OUT_DIR/$rel"
    mkdir -p "$(dirname "$dst")"
    if [ "$file" != "-" ]; then
      cp -a "$HDW_ROOT/$rel/$file" "$dst/" 2>/dev/null && info "  + $rel/$file"
    else
      rsync -aHAX "$HDW_ROOT/$rel/" "$dst/" 2>/dev/null && info "  + $rel/"
    fi
  done < <(model_manifest)
else
  info "按 --app-only 跳过模型（目标机需自行执行 bash Scripts/fetch_models.sh）"
  # 但目录骨架要有，否则 bind mount 会以 root 建目录
  while IFS=$'\t' read -r kind repo file rel want; do
    [ -n "${kind:-}" ] || continue
    mkdir -p "$OUT_DIR/$rel"
  done < <(model_manifest)
fi

# ═══ 收尾 ═════════════════════════════════════════════════════

# .env 含密钥，拷过去是用户的选择，但权限收紧一点
[ -f "$OUT_DIR/Configs/.env" ] && chmod 600 "$OUT_DIR/Configs/.env"

log ""
ok "打包完成：$OUT_DIR（$(du -sh "$OUT_DIR" 2>/dev/null | cut -f1)）"
log ""
log "目标机上执行："
log "  cd $OUT_DIR && bash Scripts/deploy.sh"
log ""
log "提示：目标机首次部署前先确认这几项（deploy.sh 也会检查）"
log "  · Docker Engine + compose v2"
log "  · systemd 用户管理器可用（llama 和协调器是用户级服务）"
log "  · 若用 GPU：NVIDIA 驱动 + nvidia-container-toolkit"
log "    （不必装 CUDA toolkit —— 多架构构建目录里已随包携带所需 runtime 库）"
