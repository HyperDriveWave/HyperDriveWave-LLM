#!/usr/bin/env bash
# 按 vendor/vendor.lock 把第三方依赖克隆回原路径。
#
# 这些仓库不进版本库，所以全新克隆的项目里它们的目录是空的。
# 本脚本负责补齐，并把项目自有的 overlay 文件（vendor/overlays/）拷回原位。
#
# 用法：
#   bash Scripts/fetch_vendors.sh                 # 只拉默认需要的 4 个
#   bash Scripts/fetch_vendors.sh --all           # 拉全部 12 个（含 8 个预留仓库）
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
# 关于镜像 URL 的写法：镜像（如 gitee 的定时同步）**不会有**我们锁定的那个
# commit，所以 URL 后面可以直接带上它自己存在的 commit，用 # 分隔：
#   https://github.com/上游/repo#<锁定的commit>,https://gitee.com/镜像/repo#<镜像的commit>
# 不带 # 的 URL 用行首那个锁定 commit。走到镜像兜底会**明确警告版本不同**，
# 并要求重跑验收——因为拿到的确实是另一份代码。
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
    -h|--help)    # 取开头连续的注释块。**不用硬编码行号**——原先写的是
                  # `sed -n '2,20p'`，往头部加几行说明就会截断（deploy.sh 里有同样的坑）。
                  awk 'NR > 1 { if (/^#/) { sub(/^# ?/, ""); print; next } exit }' "${BASH_SOURCE[0]}"
                  exit 0 ;;
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

# ── 每个 URL 可以带自己的 commit（URL#commit）────────────────────
# **按最后一个 # 切分**。不能用 @ 当分隔符——SSH 写法 `git@host:path` 里本来
# 就有 @，切出来会把 host 当成 commit。而仓库 URL 里出现 # 的情况不存在。
# 不带 #commit 的 URL 用行首那个 commit（规范上游的锚定版本）。
#
# 为什么非要有这个：gitee 镜像**永远不会有**我们锁定的那个 commit——
# 它是别人的定时同步，天然落后几天。所以原先"多个 URL 试同一个 commit"
# 这个模型对镜像根本不成立，把镜像 URL 加进去只会每次都失败。
# 带上镜像自己的 commit，兜底才谈得上成立。
#
# 代价是**版本与锁定的不同**，所以走到兜底必须大声警告并重跑验收。
url_clean() { printf '%s' "${1%%#*}"; }

url_commit() {   # $1=url（可能带 #commit）  $2=行首 commit
  case "$1" in
    *'#'*) printf '%s' "${1##*#}" ;;
    *)     printf '%s' "$2" ;;
  esac
}

# 这一行所有可接受的 commit：行首那个 + 每个 URL 自带的，去重、逗号分隔。
# 已有仓库落在其中任何一个上都算"就绪"——否则用镜像拉过的机器每次跑都判
# WRONG_HEAD，然后反复重克隆，永远修不好。
acceptable_commits() {   # $1=行首 commit  $2=urls
  # **out 必须单独一行赋值**：`local a="$1" b="$a"` 里 $a 是在赋值前展开的，
  # 取到的是外层同名变量（未定义），锚定 commit 会被静默吞掉。
  local row="$1" urls="$2" u c
  local out="$row"
  for u in ${urls//,/ }; do
    [ -n "$u" ] || continue
    c="$(url_commit "$u" "$row")"
    [ "$c" = "$row" ] && continue
    case ",$out," in *",$c,"*) ;; *) out="$out,$c" ;; esac
  done
  printf '%s' "$out"
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
# $1=path  $2=锚定 commit  $3..=其它可接受的 commit（镜像自带的那些）
#
# 输出 OK / OK_FALLBACK / NOGIT / MISSING / WRONG_HEAD
# OK_FALLBACK 表示落在某个镜像的 commit 上：**算就绪**（不然会反复重克隆），
# 但要明确标出来，因为那是与锁定版本不同的另一份代码。
vendor_status() {
  local path="$1" want="$2"; shift 2
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
  if [ "$head" = "$want" ]; then echo "OK"; return; fi
  local c
  for c in "$@"; do
    [ -n "$c" ] && [ "$head" = "$c" ] && { echo "OK_FALLBACK"; return; }
  done
  echo "WRONG_HEAD"
}

# ── 三级克隆 ──────────────────────────────────────────────────
# 第 1 级依赖服务端支持取任意 SHA（uploadpack.allowReachableSHA1InWant）。
# GitHub 支持；**Gitee 实测也支持**（2026-09-12 对 gitee.com/mirrors/llama-cpp
# 浅取任意 SHA，9 秒成功）——所以镜像兜底走的是最快的那一级，不会退化到全量克隆。

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
# 走了镜像兜底的条目（拿到的是与锁定版本不同的代码）
declare -a FALLBACK_USED=()

while IFS=$'\t' read -r path commit is_default branch urls; do
  [ -n "${path:-}" ] || continue
  [ -n "${commit:-}" ] || { warn "锁文件条目缺 commit：$path"; continue; }

  # 允许的 commit = 锚定的 + 各镜像自带的。要把它们逐个展开成位置参数。
  read -r -a _accept <<< "$(acceptable_commits "$commit" "$urls" | tr ',' ' ')"
  st="$(vendor_status "$path" "${_accept[@]}")"

  if ! should_fetch "$path" "$is_default"; then
    if [ "$st" = "OK" ] || [ "$st" = "OK_FALLBACK" ]; then
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
    OK_FALLBACK)
      # 算就绪，不重克隆（重克隆也只会再拿回同一个镜像 commit）。但必须标出来。
      OK+=("$path")
      warn "  [就绪·镜像版本] $path"
      warn "      在 $(git -C "$HDW_ROOT/$path" rev-parse HEAD 2>/dev/null | cut -c1-12)，不是锁定的 ${commit:0:12}"
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
    printf '  %-32s ← 锁定 %s\n' "$p" "${c:0:12}"
    # 把每个候选 URL 和它**实际会检出**的 commit 都列出来——镜像那个往往
    # 不是锁定的 commit，这正是最容易看漏的一点。
    for _u in ${u//,/ }; do
      [ -n "$_u" ] || continue
      _uc="$(url_commit "$_u" "$c")"
      if [ "$_uc" = "$c" ]; then
        printf '      %s\n' "$(url_clean "$_u")"
      else
        printf '      %s\n         └ %s（镜像版本，与锁定不同）\n' \
          "$(url_clean "$_u")" "${_uc:0:12}"
      fi
    done
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
  url_primary_raw="$(printf '%s' "$urls" | cut -d, -f1)"
  url_primary="$(url_clean "$url_primary_raw")"
  primary_commit="$(url_commit "$url_primary_raw" "$commit")"

  log ""
  info "▶ $path"
  dim "    目标 commit：${commit:0:12}  分支：$branch"

  # 已有目录但 commit 不对 → 先试增量，不重新克隆
  if [ "$st" = "WRONG_HEAD" ] || [ "$st" = "NOGIT" ] && [ -d "$dest/.git" ]; then
    if fix_existing "$path" "$primary_commit" "$url_primary"; then
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

      # 每个 URL 用**它自己**的 commit（不带 #commit 的用行首那个）
      u_url="$(url_clean "$url")"
      u_commit="$(url_commit "$url" "$commit")"

      if [ "$u_commit" != "$commit" ]; then
        info "  [$tier/3] $tier_name（$host，超时 ${tier_timeout}s）"
        dim "        镜像版本 ${u_commit:0:12}（锁定的是 ${commit:0:12}）"
      else
        info "  [$tier/3] $tier_name（$host，超时 ${tier_timeout}s）"
      fi
      host_tried[$host]=1

      case "$tier" in
        1) try_shallow_sha    "$u_url" "$u_commit" "$dest" && { done_ok=1; } ;;
        2) try_partial_clone  "$u_url" "$u_commit" "$branch" "$dest" && { done_ok=1; } ;;
        3) try_full_clone     "$u_url" "$u_commit" "$branch" "$dest" && { done_ok=1; } ;;
      esac

      if [ "$done_ok" = "1" ]; then
        ok "  成功（$tier_name @ $host）"
        effective_commit="$u_commit"
        break 2
      fi
    done
  done

  if [ "$done_ok" != "1" ]; then
    # 试过的 host 全部拉黑：后续仓库不必再等同样的超时
    for h in "${!host_tried[@]}"; do BAD_HOST[$h]=1; done
    fail "$path 克隆失败（所有 URL 的三级都试过了）"
    warn "    若这台机器需要走代理，设置 HTTPS_PROXY 后重跑；
     或把可用的镜像 URL 追加到 vendor/vendor.lock 对应行的末尾。
     注意：镜像通常**没有**我们锁定的那个 commit，这时要写成
       <镜像URL>#<该镜像上存在的 commit>
     否则它每一级都会失败。"
    continue
  fi

  # 校验：必须真的在**本次实际使用**的 commit 上
  effective_commit="${effective_commit:-$commit}"
  got="$(git -C "$dest" rev-parse HEAD 2>/dev/null)"
  if [ "$got" != "$effective_commit" ]; then
    fail "$path 检出的 commit 不对：期望 ${effective_commit:0:12}，实际 ${got:0:12}"
  fi

  # 走到镜像兜底 = 拿到的是**另一份代码**，不是锁定的那个版本
  if [ "$effective_commit" != "$commit" ]; then
    FALLBACK_USED+=("$path|$commit|$effective_commit")
    warn "  ⚠ $path 用的是镜像版本 ${effective_commit:0:12}，与锁定的 ${commit:0:12} **不是同一份代码**"
  fi
done

apply_overlays

log ""
if [ "$FAIL_COUNT" -gt 0 ]; then
  warn "$FAIL_COUNT 个依赖未就绪"
  exit 1
fi

# 用了镜像兜底就得说清楚——这是**另一份代码**，不能当成本次部署没变化。
# 尤其 llama.cpp：镜像是定时同步的，可能落后若干天，缺的正是刚加的特性。
if [ "${#FALLBACK_USED[@]}" -gt 0 ]; then
  log ""
  warn "以下 ${#FALLBACK_USED[@]} 个依赖用的是**镜像版本**，与锁文件不同："
  for _f in "${FALLBACK_USED[@]}"; do
    IFS='|' read -r _p _want _got <<< "$_f"
    printf '    %-46s %s → %s\n' "$_p" "${_want:0:12}" "${_got:0:12}"
  done
  log ""
  log "  原因是这台机器取不到锁定的那个 commit（通常因为连不上 github，"
  log "  而国内镜像没有同步到那一个）。能连上 github 时重跑本脚本即可换回锁定版本："
  log "    bash Scripts/fetch_vendors.sh --all"
  log ""
  warn "  **请务必重跑验收**：bash Scripts/deploy_verify.sh"
  warn "  版本差异不会被自动测出来，但可能表现成推理异常或某个新参数不生效。"
fi

ok "第三方依赖已就绪"

# 这几个仓库是别人的项目，许可证各自独立；这里提醒一句免得当成项目自有代码。
if [ "${#NEED[@]}" -gt 0 ]; then
  log ""
  dim "  提示：vendor/ 下是第三方代码，各自遵循其上游许可证。"
  dim "  aora-bot/emotion-ball 的许可证在商业部署前需重新核对（见 README）。"
fi
