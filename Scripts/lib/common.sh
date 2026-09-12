#!/usr/bin/env bash
# 部署脚本共享设施。被 source，不单独执行。
#
# 这里只放「和项目无关」的通用能力（日志、幂等写文件、state marker）。
# 认识 HyperDriveWave 的东西放 detect.sh / models.sh。

[ -n "${HDW_COMMON_SH_LOADED:-}" ] && return 0
HDW_COMMON_SH_LOADED=1

# 颜色只在终端下开，重定向到日志文件时保持干净
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  _C_RESET=$'\033[0m'; _C_RED=$'\033[31m'; _C_GRN=$'\033[32m'
  _C_YEL=$'\033[33m'; _C_BLU=$'\033[36m'; _C_DIM=$'\033[2m'
else
  _C_RESET=; _C_RED=; _C_GRN=; _C_YEL=; _C_BLU=; _C_DIM=
fi

HDW_WARNINGS=0
HDW_ERRORS=0
_STEP_NO=0

log()  { printf '%s\n' "$*"; }
info() { printf '%s[信息]%s %s\n' "$_C_BLU" "$_C_RESET" "$*"; }
ok()   { printf '%s[完成]%s %s\n' "$_C_GRN" "$_C_RESET" "$*"; }
dim()  { printf '%s%s%s\n' "$_C_DIM" "$*" "$_C_RESET"; }

warn() {
  HDW_WARNINGS=$((HDW_WARNINGS + 1))
  printf '%s[警告]%s %s\n' "$_C_YEL" "$_C_RESET" "$*" >&2
}

# die 只用于「继续下去没有意义」的情况。可降级的问题用 warn + 返回非 0。
die() {
  HDW_ERRORS=$((HDW_ERRORS + 1))
  printf '%s[失败]%s %s\n' "$_C_RED" "$_C_RESET" "$*" >&2
  exit 1
}

step() {
  _STEP_NO=$((_STEP_NO + 1))
  printf '\n%s── 步骤 %d：%s ──%s\n' "$_C_BLU" "$_STEP_NO" "$*" "$_C_RESET"
}

# 命令存在性。缺了直接死，因为后面每一步都会用它。
need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "缺少命令：$1${2:+（$2）}"
}

have_cmd() { command -v "$1" >/dev/null 2>&1; }

# ── 幂等写文件 ────────────────────────────────────────────────
# 全项目统一走这三个函数。理由：
#   - 直接 > 覆盖会无条件更新 mtime，导致「内容没变但 systemd 重载了」，
#     进而在每次部署时重启正在服务的 llama。
#   - .env 是 set -a source 的，重复 key 后者覆盖前者，所以追加是静默事故。

# 内容不同才写。相同返回 1，写入返回 0。调用方靠返回值决定要不要 reload。
write_if_different() {
  local target="$1" content="$2"
  local tmp
  tmp="$(mktemp "${target}.tmp.XXXXXX")"
  printf '%s' "$content" > "$tmp"
  if [ -f "$target" ] && cmp -s "$tmp" "$target"; then
    rm -f "$tmp"
    return 1
  fi
  # 保留原权限（.env 是 600，不能被 mktemp 的 600 之外的默认值带偏）
  [ -f "$target" ] && chmod --reference="$target" "$tmp" 2>/dev/null || true
  mv -f "$tmp" "$target"
  return 0
}

# 原子替换整个文件（从 stdin 读）。比 write_if_different 适合大内容。
write_file_from_stdin() {
  local target="$1"
  local tmp
  tmp="$(mktemp "${target}.tmp.XXXXXX")"
  cat > "$tmp"
  [ -f "$target" ] && chmod --reference="$target" "$tmp" 2>/dev/null || true
  mv -f "$tmp" "$target"
}

# 首次改动前备份一次。重复调用不会覆盖已有备份。
backup_once() {
  local file="$1"
  [ -f "$file" ] || return 0
  local bak="${file}.bak-$(date +%Y%m%d-%H%M%S)"
  # 已经备份过就不再堆文件：同名 .bak-* 存在即跳过
  if compgen -G "${file}.bak-*" >/dev/null 2>&1; then
    return 0
  fi
  cp -a "$file" "$bak"
  dim "  已备份 $file → $(basename "$bak")"
}

# ── .env 风格文件的定点改写 ────────────────────────────────────
# 只替换目标行，保留注释、顺序、以及所有其它键（尤其是密钥）。
# 不存在的键才追加。

env_get() {
  local file="$1" key="$2"
  [ -f "$file" ] || return 1
  sed -n "s/^${key}=//p" "$file" | head -1
}

env_set() {
  local file="$1" key="$2" value="$3"
  backup_once "$file"
  local tmp
  tmp="$(mktemp "${file}.tmp.XXXXXX")"
  if grep -qE "^${key}=" "$file" 2>/dev/null; then
    # 用 awk 而不是 sed：value 里可能含 / & \ 等 sed 元字符（路径、URL、token）
    awk -v k="$key" -v v="$value" '
      BEGIN { done = 0 }
      $0 ~ "^" k "=" { if (!done) { print k "=" v; done = 1 }; next }
      { print }
    ' "$file" > "$tmp"
  else
    cat "$file" > "$tmp" 2>/dev/null || true
    printf '%s=%s\n' "$key" "$value" >> "$tmp"
  fi
  chmod --reference="$file" "$tmp" 2>/dev/null || true
  mv -f "$tmp" "$file"
}

# 只在该键为空/不存在时设值。用于「用户可能已经配过」的项。
env_set_if_empty() {
  local file="$1" key="$2" value="$3"
  local cur
  cur="$(env_get "$file" "$key" || true)"
  if [ -z "$cur" ]; then
    env_set "$file" "$key" "$value"
    return 0
  fi
  return 1
}

# ── state marker ──────────────────────────────────────────────
# 注意：marker 只用来**省时间**（跳过模型下载/镜像构建这类昂贵步骤），
# 绝不用于判断正确性。任何阶段的可重入判定都要重新探测当前状态，
# 因为 marker 会撒谎（有人删了模型、有人 docker rmi）。

deploy_state_dir() { printf '%s' "${HDW_DEPLOY_STATE_DIR:-$HDW_ROOT/HDW_Runtime/.deploy}"; }

marker_done() {
  local name="$1"
  [ -f "$(deploy_state_dir)/${name}.done" ]
}

marker_set() {
  local name="$1"; shift
  local dir; dir="$(deploy_state_dir)"
  mkdir -p "$dir"
  {
    echo "时间: $(date -Iseconds)"
    echo "主机: $(hostname)"
    echo "用户: $(whoami)"
    [ $# -gt 0 ] && printf '%s\n' "$*"
  } > "$dir/${name}.done"
}

# ── 其它 ──────────────────────────────────────────────────────

# 从 BASH_SOURCE 反推项目根。所有脚本都靠它，不依赖绝对路径。
hdw_root_from() {
  local src="$1"
  (cd -- "$(dirname -- "$src")/.." && pwd)
}

# 用户级 systemd 单元目录
user_unit_dir() { printf '%s' "${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"; }

# 把项目绝对路径转成 systemd 里最"抗移动"的写法。
# 在 $HOME 下就用 %h/<相对>——项目在家目录里挪位置不用重装单元；
# 不在 $HOME 下（如 /opt/hdw）只能写绝对路径，那就没有捷径了。
systemd_path() {
  local abs="$1"
  case "$abs" in
    "$HOME"/*)
      printf '%%h/%s' "${abs#"$HOME"/}"
      ;;
    *)
      printf '%s' "$abs"
      ;;
  esac
}

# 读秒级确认，避免 curl 挂了整个脚本
http_ok() {
  local url="$1" timeout="${2:-5}"
  curl -fsS --max-time "$timeout" -o /dev/null "$url" 2>/dev/null
}
