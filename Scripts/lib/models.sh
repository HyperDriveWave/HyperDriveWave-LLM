#!/usr/bin/env bash
# 模型/镜像清单的**唯一事实源**。被 source，不单独执行。
#
# 一张表，三个消费者：
#   1. fetch_models.sh   —— 在线/镜像模式下照它下载
#   2. offline 模式      —— 照它输出「缺什么」的 TSV 清单
#   3. deploy_verify.sh  —— 照它校验 OK / MISSING / BAD_SIZE
# 三者共用同一张表，就不可能出现在「下载清单」和「校验清单」不一致的问题。

[ -n "${HDW_MODELS_SH_LOADED:-}" ] && return 0
HDW_MODELS_SH_LOADED=1

# 当前运行时真正会用到的模型。用不上的备份模型（Qwen3.8-27B-FP8 / Qwen3.6-35B-A3B /
# Qwen3.8_MoE / Qwen3.8-27B-GGUF 共 200G+）不在清单里，不下载也不打包。
#
# 字节数是**精确校验值**，不是估算：
#   - GGUF 取自魔搭 repo 的文件列表，与本地文件 stat 一致
#   - bge 两个是快照目录的合计
# 下载后必须字节数相等，否则视为不完整。
#
# 格式：kind<TAB>repo_or_tag<TAB>file<TAB>相对目标<TAB>字节数
#   file 用 `-` 表示「整个仓库快照」，非 `-` 表示只取该文件。
#   **不能留空**：read 配 IFS=$'\t' 时连续制表符会被折叠成一个分隔符，
#   空字段会让后面所有列整体左移（踩过一次）。
model_manifest() {
  cat <<'EOF'
model	ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF	Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf	HDW_Engines/LLM_Models/Qwen3.8-27B-GSQ	12120016960
model	BAAI/bge-m3	-	HDW_Engines/RAG_Models/bge-m3	4587311333
model	BAAI/bge-reranker-v2-m3	-	HDW_Engines/RAG_Models/bge-reranker-v2-m3	2293567620
EOF
}

# 容器镜像。离线包需要它们，在线模式由 docker pull / build 得到。
# 不写字节数——用 docker image inspect 动态取，写死会随重构失效。
image_manifest() {
  cat <<'EOF'
hyperdrivewave-hdw-rag:latest
hyperdrivewave-hdw-mineru:latest
postgres:17
redis:7
neo4j:5
nginx:1.27-alpine
EOF
}

# 远端 RAG 只需要这些。用来算「远端包」的最小体积，避免把 mineru 的 16.7G 也搬过去。
image_manifest_remote_rag() {
  echo "hyperdrivewave-hdw-rag:latest"
}

# 单个模型条目的绝对路径
model_target_path() {
  local rel="$2"
  printf '%s/%s' "$HDW_ROOT" "$rel"
}

# 目录合计字节数。用 du 的 --bytes 拿精确值，不用 -h。
dir_bytes() {
  [ -d "$1" ] || { echo 0; return; }
  du -sb "$1" 2>/dev/null | cut -f1
}

# 三态校验：OK / MISSING / BAD_SIZE
# 目录型条目（bge 快照）允许少量偏差——HuggingFace 快照可能带 .cache 等元数据，
# 差 1% 以内算 OK。GGUF 是单文件，要求精确相等。
model_status() {
  local kind="$1" repo="$2" file="$3" rel="$4" want="$5"
  local target="$HDW_ROOT/$rel"
  local have=0

  if [ "$file" != "-" ]; then
    [ -f "$target/$file" ] || { echo "MISSING"; return; }
    have="$(stat -c '%s' "$target/$file" 2>/dev/null || echo 0)"
    if [ "$have" != "$want" ]; then
      echo "BAD_SIZE"
      return
    fi
    echo "OK"
    return
  fi

  [ -d "$target" ] || { echo "MISSING"; return; }
  have="$(dir_bytes "$target")"
  [ "$have" -gt 0 ] || { echo "MISSING"; return; }
  local diff=$(( have > want ? have - want : want - have ))
  if [ $(( diff * 100 )) -le "$want" ]; then
    echo "OK"
  else
    echo "BAD_SIZE"
  fi
}

# 人类可读的条目描述，供日志和「缺失清单」使用
model_describe() {
  local kind="$1" repo="$2" file="$3" rel="$4" want="$5"
  local size
  size="$(awk -v b="$want" 'BEGIN { printf "%.2fG", b / 1000000000 }')"
  if [ "$file" != "-" ]; then
    printf '%s/%s → %s（%s）' "$repo" "$file" "$rel" "$size"
  else
    printf '%s（快照）→ %s（%s）' "$repo" "$rel" "$size"
  fi
}
