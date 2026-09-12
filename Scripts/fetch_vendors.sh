#!/usr/bin/env bash
# 按 vendor/vendor.lock 把第三方依赖克隆回原路径。
#
# 这些仓库不进版本库，所以全新克隆的项目里它们的目录是空的。
# 本脚本负责补齐，并把项目自有的 overlay 文件（vendor/overlays/）拷回原位。
#
# 用法：
#   bash Scripts/fetch_vendors.sh                 # 只拉默认需要的（MinerU / aora-bot）
#   bash Scripts/fetch_vendors.sh --all           # 拉全部（含 6 个预留仓库，约 2.7G）
#   bash Scripts/fetch_vendors.sh --with dify,n8n # 默认的 **加上** 这两个
#   bash Scripts/fetch_vendors.sh --only MinerU   # 只要这一个
#   bash Scripts/fetch_vendors.sh --check-only    # 只报状态，不克隆
#   bash Scripts/fetch_vendors.sh --prefer gitee  # 优先走 gitee，不走 github
#
# 关于 --prefer：锁文件里规范上游（github）写在前面，便于他人复用与溯源；
# 但国内网络下按 SHA 拉 github 会**无限挂起**，虽有超时和拉黑兜底，首次仍要
# 白等 45 秒。--prefer gitee 把 gitee 的 URL 提到前面，实测冷克隆 54s → 2.5s。
# 只影响本次运行，不改锁文件。
#
# 退出码：0 = 需要的都已就绪；1 = 有缺失（离线模式下的正常结果）

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HDW_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export HDW_ROOT

. "$SCRIPT_DIR/lib/common.sh"

LOCK_FILE="$HDW_ROOT/vendor/vendor.lock"
OVERLAY_DIR="$HDW_ROOT/vendor/overlays"
# 克隆暂存目录：先克隆到这里再整体搬过去，避免"克隆失败留下半个目录"
STAGE_DIR=""

FETCH_ALL=0
WITH_LIST=""
ONLY_LIST=""
CHECK_ONLY=0
NET_MODE="online"
PREFER_HOST=""

while [ $# -gt 0 ]; do
  case "$1" in
    --all)        FETCH_ALL=1; shift ;;
    --with)       WITH_LIST="${2:-}"; shift 2 ;;
    --only)       ONLY_LIST="${2:-}"; shift 2 ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --network)    NET_MODE="${2:-}"; shift 2 ;;
    --prefer)     PREFER_HOST="${2:-}"; shift 2 ;;
    -h|--help)    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done

# ── URL 排序 ──────────────────────────────────────────────────
# 锁文件里规范上游写在前面（便于他人复用、也是溯源依据），但**这台机器**可能
# 根本连不上 github ——实测按 SHA 拉 github 会无限挂起。虽然坏 host 会被拉黑，
# 首次仍要白等一个超时。想直接优先 gitee 就跑：
#   bash Scripts/fetch_vendors.sh --prefer gitee
# 只调整本次运行的尝试顺序，不改锁文件。
order_urls() {
  local urls="$1"
  [ -z "$PREFER_HOST" ] && { printf '%s' "$urls"; return; }
  local first="" rest="" u
  for u in ${urls//,/ }; do
    [ -n "$u" ] || continue
    case "$u" in
      *"$PREFER_HOST"*) first="${first:+$first,}$u" ;;
      *)                rest="${rest:+$rest,}$u" ;;
    esac
  done
  printf '%s' "${first}${first:+,}${rest}"
}

case "$NET_MODE" in
  online|mirror) ;;
  offline) CHECK_ONLY=1 ;;   # 离线就是"只报缺不克隆"，与 --check-only 同义
  *) die "--network 只能是 online / mirror / offline" ;;
esac

# 条目失败但还要继续跑完剩下的条目，所以不能直接用 die（它会 exit）
FAIL_COUNT=0
fail() { FAIL_COUNT=$((FAIL_COUNT + 1)); printf '%s[失败]%s %s\n' "$_C_RED" "$_C_RESET" "$*" >&2; }

# ── 超时 ──────────────────────────────────────────────────────
# 必须有超时。踩过的坑：`git fetch origin <sha>` 打某些 host 不是报错而是
# **无限挂起**（既不打 "unadvertised object" 也不断开）——GitHub 对未广告的
# SHA 要做一次完整可达性遍历，大仓库上能卡几分钟。没超时的话三级回退
# 每级都干等，实测一个仓库烧掉 8 分钟。
#
# 取值偏紧：浅取正常只要几秒到几十秒，等满 45s 还没完基本就是不会完了。
# 首次运行最多在一个坏 host 上花 T1+T2 ≈ 3 分钟，之后该 host 被拉黑，其余仓库不再等。
T1_TIMEOUT="${HDW_VENDOR_T1_TIMEOUT:-45}"     # 按 SHA 浅取
T2_TIMEOUT="${HDW_VENDOR_T2_TIMEOUT:-150}"    # partial clone
T3_TIMEOUT="${HDW_VENDOR_T3_TIMEOUT:-1800}"   # 全量克隆，确实可能很慢

# timeout 不存在时退化为直接执行（BSD/macOS 上是 gtimeout）
TIMEOUT_BIN=""
if have_cmd timeout; then TIMEOUT_BIN="timeout"
elif have_cmd gtimeout; then TIMEOUT_BIN="gtimeout"
fi

run_git_limited() {
  local secs="$1"; shift
  if [ -n "$TIMEOUT_BIN" ]; then
    "$TIMEOUT_BIN" "$secs" "$@"
  else
    "$@"
  fi
}

# ── 本次运行内记住挂掉的 host ─────────────────────────────────
# 锁文件里规范上游写在前面（便于移植），但这台机器可能根本连不上 github。
# 不记的话，9 个仓库每个都要在同一个 host 上白等一遍超时。
declare -A BAD_HOST=()

[ -f "$LOCK_FILE" ] || die "找不到锁文件：$LOCK_FILE"

# ── 读锁文件 ──────────────────────────────────────────────────
# 输出的每行：path<TAB>commit<TAB>default<TAB>branch<TAB>urls
read_lock() {
  grep -vE '^\s*(#|$)' "$LOCK_FILE"
}

# 判断某条目是否本次要处理。
#
#   （无参数）    只拉默认需要的
#   --with X,Y    默认需要的 **加上** X,Y —— 是追加不是替换。
#                 踩过：先前写成替换，结果 `--with FreeToken` 把 MinerU 也跳过了，
#                 overlay 少应用一个，得靠翻日志才发现。
#   --only X,Y    只要 X,Y
#   --all         全都要
#
# X,Y 接受短名（dify）或完整路径（HDW_Orchestrator/dify）——短名好敲，
# 但清单里存的是路径，所以两种都比一遍。
matches_list() {
  local path="$1" list="$2"
  local short="${path##*/}"
  case ",$list," in
    *",$short,"*|*",$path,"*) return 0 ;;
  esac
  return 1
}

should_fetch() {
  local path="$1" is_default="$2"
  [ "$FETCH_ALL" = "1" ] && return 0
  if [ -n "$ONLY_LIST" ]; then
    matches_list "$path" "$ONLY_LIST" && return 0
    return 1
  fi
  [ "$is_default" = "yes" ] && return 0
  [ -n "$WITH_LIST" ] && matches_list "$path" "$WITH_LIST" && return 0
  return 1
}

# ── 状态判定 ──────────────────────────────────────────────────
vendor_status() {
  local path="$1" want="$2"
  local dir="$HDW_ROOT/$path"
  if [ ! -d "$dir/.git" ]; then
    # 目录存在但没有 .git —— 可能是从旧备份直接拷过来的
    if [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
      echo "NOGIT"   # 有内容但不是 git 仓库
    else
      echo "MISSING"
    fi
    return
  fi
  local head
  head="$(git -C "$dir" rev-parse HEAD 2>/dev/null)"
  if [ "$head" = "$want" ]; then
    echo "OK"
  else
    echo "WRONG_HEAD"
  fi
}

# ── 三级克隆 ──────────────────────────────────────────────────
# 第 1 级依赖服务端支持取任意 SHA（uploadpack.allowReachableSHA1InWant）。
# GitHub 支持；Gitee 未验证——这正是要有回退的原因。

try_shallow_sha() {
  local url="$1" sha="$2" dest="$3"
  rm -rf "$dest"
  mkdir -p "$dest"
  git -C "$dest" init --quiet || return 1
  git -C "$dest" remote add origin "$url" || return 1
  # 输出重定向掉：失败时 git 会打一长串 "Server does not allow request for unadvertised object"，
  # 而真正的问题（挂起）根本不打印任何东西。返回值 124 = 超时。
  run_git_limited "$T1_TIMEOUT" git -C "$dest" fetch --quiet --depth 1 origin "$sha" >/dev/null 2>&1 || return 1
  git -C "$dest" checkout --quiet FETCH_HEAD >/dev/null 2>&1 || return 1
  return 0
}

try_partial_clone() {
  local url="$1" sha="$2" branch="$3" dest="$4"
  rm -rf "$dest"
  # --filter=blob:none：只拉 commit 和 tree，blob 在 checkout 时按需取。
  # 对 langgraph 这类"500M 历史换 13M 源码"的仓库收益最大。
  run_git_limited "$T2_TIMEOUT" git clone --quiet --filter=blob:none --no-checkout --single-branch \
    --branch "$branch" "$url" "$dest" >/dev/null 2>&1 || return 1
  run_git_limited "$T2_TIMEOUT" git -C "$dest" checkout --quiet "$sha" >/dev/null 2>&1 || return 1
  return 0
}

try_full_clone() {
  local url="$1" sha="$2" branch="$3" dest="$4"
  rm -rf "$dest"
  run_git_limited "$T3_TIMEOUT" git clone --quiet --single-branch --branch "$branch" "$url" "$dest" >/dev/null 2>&1 || return 1
  git -C "$dest" checkout --quiet "$sha" >/dev/null 2>&1 || return 1
  return 0
}

# 对已有目录做增量修正（不重新克隆）
fix_existing() {
  local path="$1" sha="$2" url="$3"
  local dir="$HDW_ROOT/$path"
  info "    已有仓库但 commit 不对，增量修正…"
  git -C "$dir" fetch --quiet origin >/dev/null 2>&1
  if git -C "$dir" cat-file -e "$sha^{commit}" 2>/dev/null; then
    git -C "$dir" checkout --quiet "$sha" >/dev/null 2>&1 && return 0
  fi
  # 本地没有这个对象，从规范 URL 补取
  git -C "$dir" fetch --quiet "$url" >/dev/null 2>&1
  git -C "$dir" cat-file -e "$sha^{commit}" 2>/dev/null || return 1
  git -C "$dir" checkout --quiet "$sha" >/dev/null 2>&1
}

# ── overlay ───────────────────────────────────────────────────
# 那 3 个项目自有文件必须落在 build context 内部（Docker 要求 Dockerfile 在
# context 里，compose 的 dockerfile: 也是相对 context 解析的），
# 所以不能只存一份在外面改 compose 指向——只能在克隆后拷进去。
apply_overlays() {
  [ -d "$OVERLAY_DIR" ] || return 0
  local n=0 rel src dst
  while IFS= read -r src; do
    rel="${src#"$OVERLAY_DIR"/}"
    dst="$HDW_ROOT/$rel"
    # 只在目标仓库目录存在时才拷，避免往空目录里塞文件造成误解
    if [ -d "$(dirname "$dst")" ]; then
      if ! cmp -s "$src" "$dst" 2>/dev/null; then
        cp -a "$src" "$dst"
        dim "    overlay 已就位：$rel"
      fi
      n=$((n + 1))
    fi
  done < <(find "$OVERLAY_DIR" -type f)
  [ "$n" -gt 0 ] && info "  overlay 文件已应用（$n 个）"
  return 0
}

# ── 主流程 ────────────────────────────────────────────────────

step "检查第三方依赖"

NEED=(); OK=(); SKIP=()
declare -a FETCH_JOBS=()

while IFS=$'\t' read -r path commit is_default branch urls; do
  [ -n "${path:-}" ] || continue
  [ -n "${commit:-}" ] || { warn "锁文件条目缺 commit：$path"; continue; }

  st="$(vendor_status "$path" "$commit")"

  if ! should_fetch "$path" "$is_default"; then
    if [ "$st" = "OK" ]; then
      dim "  [就绪] $path"
    else
      SKIP+=("$path|$st|$is_default")
    fi
    continue
  fi

  case "$st" in
    OK)
      OK+=("$path")
      dim "  [就绪] $path"
      ;;
    MISSING|NOGIT|WRONG_HEAD)
      NEED+=("$path|$commit|$branch|$urls|$st")
      case "$st" in
        MISSING)    info "  [缺失] $path" ;;
        NOGIT)      info "  [无 git] $path（目录有内容但没有 .git）" ;;
        WRONG_HEAD) info "  [commit 不符] $path" ;;
      esac
      ;;
  esac
done < <(read_lock)

if [ "$CHECK_ONLY" = "1" ]; then
  log ""
  if [ "${#NEED[@]}" -eq 0 ]; then
    ok "需要的第三方依赖都已就绪"
    exit 0
  fi
  log "仅检查模式：不克隆。缺失清单："
  for item in "${NEED[@]}"; do
    IFS='|' read -r p c b u _ <<< "$item"
    printf '  %-32s ← %s @ %s\n' "$p" "$(printf '%s' "$u" | cut -d, -f1)" "${c:0:12}"
  done
  log ""
  log "有网的机器上执行：bash Scripts/fetch_vendors.sh --all"
  exit 1
fi

if [ "${#NEED[@]}" -eq 0 ]; then
  apply_overlays
  ok "第三方依赖已就绪（${#OK[@]} 个）"
  [ "${#SKIP[@]}" -gt 0 ] && dim "  另有 ${#SKIP[@]} 个预留仓库未拉取（需要时用 --all 或 --with）"
  exit 0
fi

need_cmd git "安装 git"

step "克隆缺失的依赖（${#NEED[@]} 个）"

for item in "${NEED[@]}"; do
  IFS='|' read -r path commit branch urls st <<< "$item"
  dest="$HDW_ROOT/$path"
  url_primary="$(printf '%s' "$urls" | cut -d, -f1)"

  log ""
  info "▶ $path"
  dim "    目标 commit：${commit:0:12}  分支：$branch"

  # 已有目录但 commit 不对 → 先试增量，不重新克隆
  if [ "$st" = "WRONG_HEAD" ] || [ "$st" = "NOGIT" ] && [ -d "$dest/.git" ]; then
    if fix_existing "$path" "$commit" "$url_primary"; then
      ok "  增量修正成功"
      continue
    fi
    warn "  增量修正失败，将重新克隆"
  fi

  # 三级回退，**层级为主、URL 为辅**：
  # 先把最便宜的浅取对所有镜像试一遍，都不行才升级到 partial clone，最后才全量。
  #
  # 反过来写（URL 为主）会非常慢，实测过：对 github 把三级跑完——含最贵的
  # 全量克隆——才轮到 gitee 的浅取，一个仓库白等 10 分钟。
  # 而层级为主时，github 浅取超时后立刻转 gitee 浅取，45 秒结束。
  declare -A host_tried=()
  done_ok=0
  ordered_urls="$(order_urls "$urls")"
  for tier in 1 2 3; do
    case "$tier" in
      1) tier_name="按 commit 浅取"; tier_timeout="$T1_TIMEOUT" ;;
      2) tier_name="partial clone"; tier_timeout="$T2_TIMEOUT" ;;
      3) tier_name="全量克隆";     tier_timeout="$T3_TIMEOUT" ;;
    esac

    for url in ${ordered_urls//,/ }; do
      [ -n "$url" ] || continue
      host="$(printf '%s' "$url" | sed -E 's#https?://([^/]+)/.*#\1#')"

      if [ -n "${BAD_HOST[$host]:-}" ]; then
        dim "  跳过 $host（本次运行中已确认连不通）"
        continue
      fi

      info "  [$tier/3] $tier_name（$host，超时 ${tier_timeout}s）"
      host_tried[$host]=1

      case "$tier" in
        1) try_shallow_sha    "$url" "$commit" "$dest" && { done_ok=1; } ;;
        2) try_partial_clone  "$url" "$commit" "$branch" "$dest" && { done_ok=1; } ;;
        3) try_full_clone     "$url" "$commit" "$branch" "$dest" && { done_ok=1; } ;;
      esac

      if [ "$done_ok" = "1" ]; then
        ok "  成功（$tier_name @ $host）"
        break 2
      fi
    done
  done

  if [ "$done_ok" != "1" ]; then
    # 试过的 host 全部拉黑：后续仓库不必再等同样的超时
    for h in "${!host_tried[@]}"; do BAD_HOST[$h]=1; done
    fail "$path 克隆失败（所有 URL 的三级都试过了）"
    warn "    若这台机器需要走代理，设置 HTTPS_PROXY 后重跑；
     或把可用的镜像 URL 追加到 vendor/vendor.lock 对应行的末尾。"
    continue
  fi

  # 校验：必须真的在目标 commit 上
  got="$(git -C "$dest" rev-parse HEAD 2>/dev/null)"
  if [ "$got" != "$commit" ]; then
    fail "$path 检出的 commit 不对：期望 ${commit:0:12}，实际 ${got:0:12}"
  fi
done

apply_overlays

log ""
if [ "$FAIL_COUNT" -gt 0 ]; then
  warn "$FAIL_COUNT 个依赖未就绪"
  exit 1
fi

ok "第三方依赖已就绪"

# 这几个仓库是别人的项目，许可证各自独立；这里提醒一句免得当成项目自有代码。
if [ "${#NEED[@]}" -gt 0 ]; then
  log ""
  dim "  提示：vendor/ 下是第三方代码，各自遵循其上游许可证。"
  dim "  aora-bot/emotion-ball 的许可证在商业部署前需重新核对（见 README）。"
fi
