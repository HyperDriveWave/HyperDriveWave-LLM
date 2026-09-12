#!/usr/bin/env bash
# 补齐缺失的模型权重。独立可跑——这是一次部署里唯一耗时 30 分钟的步骤，
# 单独成脚本才能"只补模型"而不用重跑整个部署。
#
# 用法：
#   bash Scripts/fetch_models.sh                    # 在线，缺什么下什么
#   bash Scripts/fetch_models.sh --network offline  # 只报缺，不下载
#   bash Scripts/fetch_models.sh --network mirror --endpoint https://ms.corp.local
#   bash Scripts/fetch_models.sh --check-only       # 只校验，不下载（等同于 offline）
#
# 退出码：0 = 全部就绪；1 = 有缺失（离线模式下的正常结果）；2 = 环境问题

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HDW_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export HDW_ROOT

# shellcheck source=lib/common.sh
. "$SCRIPT_DIR/lib/common.sh"
# shellcheck source=lib/models.sh
. "$SCRIPT_DIR/lib/models.sh"

NET_MODE="online"
ENDPOINT=""
PROXY=""
MANIFEST_OUT=""
CHECK_ONLY=0
ONLY=""

while [ $# -gt 0 ]; do
  case "$1" in
    --network)      NET_MODE="${2:-}"; shift 2 ;;
    --endpoint)     ENDPOINT="${2:-}"; shift 2 ;;
    --proxy)        PROXY="${2:-}"; shift 2 ;;
    --manifest-out) MANIFEST_OUT="${2:-}"; shift 2 ;;
    --only)         ONLY="${2:-}"; shift 2 ;;   # rag | llm
    --check-only)   CHECK_ONLY=1; shift ;;
    -h|--help)      sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done

[ "$CHECK_ONLY" = "1" ] && NET_MODE=offline

case "$NET_MODE" in
  online|mirror|offline) ;;
  *) die "--network 只能是 online / mirror / offline" ;;
esac

if [ -n "$PROXY" ]; then
  export HTTPS_PROXY="$PROXY" HTTP_PROXY="$PROXY"
  dim "  已设代理：$PROXY"
fi

# ── 1. 校验现状 ───────────────────────────────────────────────

declare -a MISSING=()
declare -a BAD=()
TOTAL_WANT=0
TOTAL_HAVE=0

while IFS=$'\t' read -r kind repo file rel want; do
  [ -n "${kind:-}" ] || continue
  # --only 过滤：rag 只要嵌入/重排，llm 只要主模型
  case "$ONLY" in
    rag) case "$rel" in *RAG_Models*) ;; *) continue ;; esac ;;
    llm) case "$rel" in *LLM_Models*) ;; *) continue ;; esac ;;
  esac

  TOTAL_WANT=$((TOTAL_WANT + 1))
  st="$(model_status "$kind" "$repo" "$file" "$rel" "$want")"
  case "$st" in
    OK)       TOTAL_HAVE=$((TOTAL_HAVE + 1)); dim "  [就绪] $rel" ;;
    MISSING)  MISSING+=("$kind|$repo|$file|$rel|$want"); info "  [缺失] $rel" ;;
    BAD_SIZE) BAD+=("$kind|$repo|$file|$rel|$want");     warn "  [不完整] $rel（大小与预期不符）" ;;
  esac
done < <(model_manifest)

if [ "${#MISSING[@]}" -eq 0 ] && [ "${#BAD[@]}" -eq 0 ]; then
  ok "模型齐备（$TOTAL_HAVE/$TOTAL_WANT）"
  exit 0
fi

warn "有 $(( ${#MISSING[@]} + ${#BAD[@]} )) 项需要处理（就绪 $TOTAL_HAVE/$TOTAL_WANT）"

# ── 2. 离线：输出清单后退出 ────────────────────────────────────

if [ "$NET_MODE" = "offline" ]; then
  log ""
  log "离线模式：不下载。以下是缺失清单。"
  log ""

  if [ -n "$MANIFEST_OUT" ]; then
    : > "$MANIFEST_OUT"
    for item in "${MISSING[@]}" "${BAD[@]}"; do
      [ -n "$item" ] || continue
      IFS='|' read -r kind repo file rel want <<< "$item"
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$kind" "$repo" "$file" "$rel" "$want" "$HDW_ROOT/$rel" >> "$MANIFEST_OUT"
    done
    info "清单已写入：$MANIFEST_OUT"
    log ""
    log "把这台机器上**已有这些文件**的目录拷过来即可（保持相对路径不变）："
    log "  rsync -aR --files-from=<(cut -f4 \"$MANIFEST_OUT\") <源机>:/ <本机>:/"
    log ""
  fi

  # 给人看的拷贝指引
  for item in "${MISSING[@]}" "${BAD[@]}"; do
    [ -n "$item" ] || continue
    IFS='|' read -r kind repo file rel want <<< "$item"
    printf '  %-52s ← %s%s\n' "$rel" "$repo" "${file:+ / $file}"
    printf '    需要 %s 字节（%s）\n' "$want" "$(awk -v b="$want" 'BEGIN { printf "%.2fG", b/1000000000 }')"
  done
  log ""
  warn "缺的就是缺的：RAG 检索需要 bge 两个模型；没有 LLM 权重则本地推理不可用，只能走在线模型。"
  exit 1
fi

# ── 3. 在线/镜像：确保 modelscope 可用 ─────────────────────────

# .venv 的 console script（bin/modelscope、bin/pip）shebang 写死了绝对路径，
# 项目移动后就失效。而 .venv/bin/python 本身是好的，所以统一用 `-m` 调用。
VENV="$HDW_ROOT/.venv"
MS_PY="$VENV/bin/python"

ms_usable() {
  [ -x "$MS_PY" ] || return 1
  "$MS_PY" -c 'import modelscope.cli.cli' >/dev/null 2>&1
}

if ! ms_usable; then
  info "当前 .venv 不可用（项目可能被移动过，console script 的 shebang 已失效），重建中…"
  need_cmd python3 ""
  rm -rf "$VENV"
  python3 -m venv "$VENV" || die "创建 venv 失败"
  # 重建后的 pip 是好的，可以用它的可执行文件（新建的 venv 路径正确）
  "$VENV/bin/pip" install --quiet --upgrade pip >/dev/null 2>&1 || true
  "$VENV/bin/pip" install --quiet -U modelscope || die "安装 modelscope 失败（检查网络或加 --proxy）"
  ok "modelscope 已就绪：$("$MS_PY" -c 'import modelscope; print(modelscope.__version__)')"
fi

MS_ARGS=()
[ -n "$ENDPOINT" ] && MS_ARGS+=(--endpoint "$ENDPOINT")

# ── 4. 下载 ───────────────────────────────────────────────────

download_one() {
  local kind="$1" repo="$2" file="$3" rel="$4"
  local dest="$HDW_ROOT/$rel"

  if [ "$kind" != "model" ]; then
    warn "  跳过不支持的条目类型：$kind"
    return 1
  fi

  mkdir -p "$dest"
  log ""
  info "下载 $repo${file:+ / $file}"
  dim "  → $dest"

  # --local-dir 是**平铺**语义：不指定文件名会下整个仓库（bge 无妨，
  # 但那个 GGUF 仓库有 100G+ 的其它量化档，必须给文件名）。
  if [ "$file" != "-" ]; then
    "$MS_PY" -m modelscope.cli.cli "${MS_ARGS[@]+"${MS_ARGS[@]}"}" download \
      "$repo" "$file" --repo-type model --local-dir "$dest"
  else
    "$MS_PY" -m modelscope.cli.cli "${MS_ARGS[@]+"${MS_ARGS[@]}"}" download \
      "$repo" --repo-type model --local-dir "$dest" --max-workers 8
  fi
}

FAILED=0
for item in "${MISSING[@]}" "${BAD[@]}"; do
  [ -n "$item" ] || continue
  IFS='|' read -r kind repo file rel want <<< "$item"
  if ! download_one "$kind" "$repo" "$file" "$rel"; then
    FAILED=$((FAILED + 1))
    warn "下载失败：$repo"
    continue
  fi
  # 下完立刻按字节数校验。modelscope 自己会续传，但不会告诉你"其实没下完"。
  st="$(model_status "$kind" "$repo" "$file" "$rel" "$want")"
  if [ "$st" = "OK" ]; then
    ok "校验通过：$rel"
  else
    FAILED=$((FAILED + 1))
    warn "下载完成但校验不通过（$st）：$rel
     可能是被中断了。重跑同一条命令会断点续传：
       bash Scripts/fetch_models.sh"
  fi
done

log ""
if [ "$FAILED" -gt 0 ]; then
  warn "$FAILED 项未就绪。重跑本脚本可续传。"
  exit 1
fi

# ── 5. 最终全量复查（含被 --only 跳过的项）────────────────────

REMAIN=0
while IFS=$'\t' read -r kind repo file rel want; do
  [ -n "${kind:-}" ] || continue
  st="$(model_status "$kind" "$repo" "$file" "$rel" "$want")"
  [ "$st" = "OK" ] || { REMAIN=$((REMAIN + 1)); warn "  [$st] $rel"; }
done < <(model_manifest)

if [ "$REMAIN" -gt 0 ]; then
  warn "仍有 $REMAIN 项不满足（可能被 --only 过滤掉了）"
  exit 1
fi

ok "全部模型就绪"
