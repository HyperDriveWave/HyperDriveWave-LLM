#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$ROOT/Configs/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "missing $ENV_FILE" >&2
  exit 1
fi

set -a
. "$ENV_FILE"
set +a

# 这几个都**不再给默认值**：写死一台具体机器的地址，会让换远端节点时
# 静默推错目标（而且那个地址会被带进公开仓库）。必须显式配置。
REMOTE_TARGET="${HDW_REMOTE_RAG_SSH_TARGET:-}"
REMOTE_ROOT="${HDW_REMOTE_RAG_ROOT:-}"
REMOTE_RAG_URLS="${HDW_RAG_REMOTE_URLS:-}"

if [ -z "$REMOTE_RAG_URLS" ]; then
  echo "HDW_RAG_REMOTE_URLS is empty（在 Configs/.env 里配置远端 RAG 节点地址）" >&2
  exit 1
fi

if [ -z "$REMOTE_TARGET" ]; then
  echo "HDW_REMOTE_RAG_SSH_TARGET is empty（形如 user@host，用于 rsync 推送 chunks.jsonl）" >&2
  exit 1
fi

if [ -z "$REMOTE_ROOT" ]; then
  echo "HDW_REMOTE_RAG_ROOT is empty（远端机上的项目根目录）" >&2
  exit 1
fi

SSH=(ssh -o ConnectTimeout=8)
RSYNC_SSH="ssh -o ConnectTimeout=8"
if [ -n "${SSHPASS:-}" ] && command -v sshpass >/dev/null 2>&1; then
  SSH=(sshpass -e ssh -o ConnectTimeout=8)
  RSYNC_SSH="sshpass -e ssh -o ConnectTimeout=8"
fi

"${SSH[@]}" "$REMOTE_TARGET" "mkdir -p '$REMOTE_ROOT/HDW_Runtime/rag'"
if [ -n "${SSHPASS:-}" ] && command -v sshpass >/dev/null 2>&1; then
  env SSHPASS="$SSHPASS" sshpass -e rsync -a --partial \
    -e "$RSYNC_SSH" \
    "$ROOT/HDW_Runtime/rag/chunks.jsonl" \
    "$REMOTE_TARGET:$REMOTE_ROOT/HDW_Runtime/rag/chunks.jsonl"
else
  rsync -a --partial -e "$RSYNC_SSH" \
    "$ROOT/HDW_Runtime/rag/chunks.jsonl" \
    "$REMOTE_TARGET:$REMOTE_ROOT/HDW_Runtime/rag/chunks.jsonl"
fi

job_id="sync-$(date +%s)"
declare -a pids=()
declare -a urls=()
IFS=',' read -r -a urls <<< "$REMOTE_RAG_URLS"
for url in "${urls[@]}"; do
  url="${url%/}"
  [ -n "$url" ] || continue
  curl --fail --silent --show-error --max-time 0 \
    -H 'Content-Type: application/json' \
    -d "{\"job_id\":\"$job_id\"}" \
    "$url/admin/reindex" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if [ "$failed" -ne 0 ]; then
  echo "remote RAG reindex failed" >&2
  exit 1
fi
echo "remote RAG indexes rebuilt: $job_id"
