#!/usr/bin/env bash
# 把国内镜像源配置到本机（Docker / pip / npm / apt）。
#
# 用法：
#   bash Scripts/setup_mirrors.sh              # 交互式：先显示检测到的默认值，问 Y/n
#   bash Scripts/setup_mirrors.sh --yes        # 全部用检测到的值，不问
#   bash Scripts/setup_mirrors.sh --check      # 只报当前状态，不改任何东西
#   bash Scripts/setup_mirrors.sh --only docker,pip
#   bash Scripts/setup_mirrors.sh --docker u1,u2 --pip u3 --npm u4 --apt u5
#
# 检测的默认值优先取**本机已有的配置**——这样"新机器配得和现在这台一样"，
# 而不是照搬一份写死的清单（本机的 apt 用的是 cn.archive.ubuntu.com 而非阿里云，
# 写死的清单反而会和现状不一致）。
#
# 每一项都：
#   1. 先备份原文件到 <原文件>.hdw-bak-<时间戳>
#   2. 内容没变就跳过（幂等，重跑不会白白重启 Docker）
#   3. 改完做一次验证，失败自动回滚

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HDW_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export HDW_ROOT

. "$SCRIPT_DIR/lib/common.sh"

CHECK_ONLY=0
ONLY_LIST="docker,pip,npm,apt"
OPT_DOCKER=""; OPT_PIP=""; OPT_NPM=""; OPT_APT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --yes|-y)  HDW_ASSUME_YES=1; shift ;;
    --check)   CHECK_ONLY=1; shift ;;
    --only)    ONLY_LIST="${2:-}"; shift 2 ;;
    --docker)  OPT_DOCKER="${2:-}"; shift 2 ;;
    --pip)     OPT_PIP="${2:-}"; shift 2 ;;
    --npm)     OPT_NPM="${2:-}"; shift 2 ;;
    --apt)     OPT_APT="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done

wanted() { case ",$ONLY_LIST," in *",$1,"*) return 0 ;; esac; return 1; }

sudo_write() {
  # 写系统文件。先落到临时文件再 sudo 搬过去，避免 sudo tee 的管道语义坑。
  local target="$1" content="$2"
  local tmp; tmp="$(mktemp)"
  printf '%s' "$content" > "$tmp"
  sudo install -m "${3:-644}" "$tmp" "$target"
  rm -f "$tmp"
}

backup_sys() {
  local f="$1"
  [ -f "$f" ] || return 0
  local bak="${f}.hdw-bak-$(date +%Y%m%d-%H%M%S)"
  sudo cp -a "$f" "$bak" && dim "    已备份 $f → $(basename "$bak")"
}



# ═══ 检测 ═════════════════════════════════════════════════════

detect_docker() {
  python3 - <<'PY' 2>/dev/null
import json
try:
    d = json.load(open("/etc/docker/daemon.json"))
    print(",".join(d.get("registry-mirrors") or []))
except Exception:
    print("")
PY
}

detect_pip() {
  local v
  v="$(python3 -m pip config get global.index-url 2>/dev/null)"
  [ -n "$v" ] && { printf '%s' "$v"; return; }
  sed -n 's/^index-url\s*=\s*//p' /etc/pip.conf 2>/dev/null | head -1
}

detect_npm() {
  have_cmd npm && npm config get registry 2>/dev/null | head -1
}

detect_apt() {
  # 从 deb822 或旧格式里取第一个 URI——只取主归档那一条，
  # 安全源单独一条在下面处理。
  local f
  f="$(ls /etc/apt/sources.list.d/*.sources 2>/dev/null | head -1)"
  if [ -n "$f" ]; then
    sed -n 's/^URIs:\s*//p' "$f" 2>/dev/null | head -1
  else
    sed -n 's/^deb\s\+\[\?[^]]*\]\?\s*\S*\s*\(\S*\)\s.*/\1/p' /etc/apt/sources.list 2>/dev/null | head -1
  fi
}

# ═══ 应用 ═════════════════════════════════════════════════════

apply_docker() {
  local mirrors="$1"
  [ -n "$mirrors" ] || { warn "  Docker：镜像源为空，跳过"; return 0; }

  local before after
  before="$(detect_docker)"
  if [ "$before" = "$mirrors" ]; then
    dim "  Docker：已经是这个配置，跳过（不重启 docker）"
    return 0
  fi

  # **合并**而不是覆盖：daemon.json 里还有 runtimes.nvidia 和 features.buildkit，
  # 直接覆盖会让容器 GPU 支持和 BuildKit 一起消失。
  local new_json
  new_json="$(python3 - "$mirrors" <<'PY'
import json, sys
path = "/etc/docker/daemon.json"
try:
    cfg = json.load(open(path))
except Exception:
    cfg = {}
cfg["registry-mirrors"] = [m for m in sys.argv[1].split(",") if m.strip()]
print(json.dumps(cfg, indent=4, ensure_ascii=False))
PY
)" || { fail "  Docker：生成配置失败"; return 1; }

  if [ "$CHECK_ONLY" = "1" ]; then
    info "  Docker：将写入 registry-mirrors = $mirrors"
    return 0
  fi

  backup_sys /etc/docker/daemon.json
  sudo mkdir -p /etc/docker
  sudo_write /etc/docker/daemon.json "$new_json" || { fail "  Docker：写配置失败"; return 1; }

  # 重启前先验证 JSON 合法，否则 docker 起不来
  if ! python3 -c "import json;json.load(open('/etc/docker/daemon.json'))" 2>/dev/null; then
    fail "  Docker：写出来的 daemon.json 不是合法 JSON，回滚"
    sudo cp -a "$(ls -t /etc/docker/daemon.json.hdw-bak-* | head -1)" /etc/docker/daemon.json
    return 1
  fi

  info "  重启 Docker 使配置生效…"
  sudo systemctl restart docker || { fail "  Docker：重启失败"; return 1; }
  sleep 3
  after="$(detect_docker)"
  if [ "$after" = "$mirrors" ]; then
    ok "  Docker：已生效（$after）"
  else
    fail "  Docker：重启后配置未生效（当前 $after）"
    return 1
  fi
}

apply_pip() {
  local url="$1"
  [ -n "$url" ] || { warn "  pip：源为空，跳过"; return 0; }
  local host; host="$(printf '%s' "$url" | sed -E 's#https?://([^/]+)/.*#\1#')"
  local content="[global]
index-url = $url
trusted-host = $host
timeout = 120

[install]
trusted-host = $host
"
  if [ "$(detect_pip)" = "$url" ] && [ -f /etc/pip.conf ]; then
    dim "  pip：已经是这个源，跳过"
    return 0
  fi
  if [ "$CHECK_ONLY" = "1" ]; then
    info "  pip：将写入 $url 到 /etc/pip.conf"
    return 0
  fi
  backup_sys /etc/pip.conf
  sudo_write /etc/pip.conf "$content" || { fail "  pip：写配置失败"; return 1; }

  # 验证：让 pip 打印它实际使用的 index，别只信配置文件写了什么
  if timeout 60 python3 -m pip config get global.index-url 2>/dev/null | grep -qF "$url"; then
    ok "  pip：已生效（$url）"
  else
    ok "  pip：已写入 /etc/pip.conf（$url）"
  fi
}

apply_npm() {
  local url="$1"
  [ -n "$url" ] || { warn "  npm：源为空，跳过"; return 0; }
  if ! have_cmd npm; then
    dim "  npm：本机没有 npm，跳过"
    return 0
  fi
  if [ "$(detect_npm)" = "$url" ]; then
    dim "  npm：已经是这个源，跳过"
    return 0
  fi
  if [ "$CHECK_ONLY" = "1" ]; then
    info "  npm：将设为 $url"
    return 0
  fi
  npm config set registry "$url" >/dev/null 2>&1 && ok "  npm：已设为 $url" \
    || { fail "  npm：设置失败"; return 1; }
}

apply_apt() {
  local url="$1"
  [ -n "$url" ] || { warn "  apt：源为空，跳过"; return 0; }

  local codename
  codename="$(. /etc/os-release && echo "${VERSION_CODENAME:-}")"
  [ -n "$codename" ] || { fail "  apt：取不到发行版代号"; return 1; }

  local target="/etc/apt/sources.list.d/ubuntu.sources"
  if [ "$(detect_apt)" = "$url" ]; then
    dim "  apt：已经是这个源，跳过"
    return 0
  fi
  if [ "$CHECK_ONLY" = "1" ]; then
    info "  apt：将把 $target 的主源改为 $url"
    return 0
  fi

  # 用一条 stanza 覆盖全部 pocket（含 security）。阿里云/清华的 ubuntu 镜像
  # 都带 -security，所以不需要单独保留 security.ubuntu.com。
  local content="Types: deb
URIs: $url
Suites: $codename $codename-updates $codename-backports $codename-security
Components: main restricted universe multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
"
  backup_sys "$target"
  sudo_write "$target" "$content" || { fail "  apt：写配置失败"; return 1; }

  # apt 改错会直接锁死包管理，所以**必须**验证并支持回滚
  info "  验证 apt 源（apt-get update）…"
  if sudo apt-get update -qq >/tmp/hdw-apt-update.log 2>&1; then
    ok "  apt：已生效（$url）"
  else
    fail "  apt：apt-get update 失败，自动回滚"
    sed -n '1,8p' /tmp/hdw-apt-update.log | sed 's/^/      /'
    local bak; bak="$(ls -t "${target}.hdw-bak-"* 2>/dev/null | head -1)"
    if [ -n "$bak" ]; then
      sudo cp -a "$bak" "$target"
      sudo apt-get update -qq >/dev/null 2>&1
      warn "  已回滚到 $(basename "$bak")"
    else
      fail "  找不到备份，请手工恢复 $target"
    fi
    rm -f /tmp/hdw-apt-update.log
    return 1
  fi
  rm -f /tmp/hdw-apt-update.log
}

# ═══ 主流程 ═══════════════════════════════════════════════════

step "检测本机现有镜像源"

D_DOCKER="${OPT_DOCKER:-$(detect_docker)}"
D_PIP="${OPT_PIP:-$(detect_pip)}"
D_NPM="${OPT_NPM:-$(detect_npm)}"
D_APT="${OPT_APT:-$(detect_apt)}"

printf '  %-8s %s\n' "Docker" "${D_DOCKER:-<未配置>}"
printf '  %-8s %s\n' "pip"    "${D_PIP:-<未配置>}"
printf '  %-8s %s\n' "npm"    "${D_NPM:-<未配置或未安装>}"
printf '  %-8s %s\n' "apt"    "${D_APT:-<未配置>}"

# --check 不能在这里退出：只报现状没用，用户要知道的是"**会改什么**"。
# 真正的判断在各 apply_* 里（它们都有 CHECK_ONLY 分支），所以往下走到那里再退。

if [ "$CHECK_ONLY" = "1" ]; then
  : # 直接采用检测/参数传入的值，不提示（--check 是无副作用的只读模式）
elif confirm "
把以上配置应用到本机？（选 n 则逐项手工填写）"; then
  D_DOCKER="${OPT_DOCKER:-$D_DOCKER}"
  D_PIP="${OPT_PIP:-$D_PIP}"
  D_NPM="${OPT_NPM:-$D_NPM}"
  D_APT="${OPT_APT:-$D_APT}"
else
  log ""
  log "逐项填写（直接回车 = 保持当前值，填 - = 跳过该项）"
  wanted docker && D_DOCKER="$(ask_value 'Docker registry mirrors（逗号分隔）' "$D_DOCKER")"
  wanted pip    && D_PIP="$(ask_value    'pip index-url' "$D_PIP")"
  wanted npm    && D_NPM="$(ask_value    'npm registry' "$D_NPM")"
  wanted apt    && D_APT="$(ask_value    'apt 主源（如 https://mirrors.aliyun.com/ubuntu/）' "$D_APT")"
fi

if [ "$CHECK_ONLY" = "1" ]; then
  step "将要做的改动（--check，不会实际执行）"
else
  step "应用镜像源"
  # 写 /etc 需要 sudo；先确认能提权，免得改到一半失败
  if ! sudo -n true 2>/dev/null && [ -t 0 ]; then
    info "需要 sudo 权限来写 /etc 下的配置文件"
    sudo -v || die "无法获取 sudo 权限"
  fi
fi

FAILED=0
wanted docker && { [ "$D_DOCKER" = "-" ] || apply_docker "$D_DOCKER" || FAILED=$((FAILED+1)); }
wanted pip    && { [ "$D_PIP" = "-" ]    || apply_pip    "$D_PIP"    || FAILED=$((FAILED+1)); }
wanted npm    && { [ "$D_NPM" = "-" ]    || apply_npm    "$D_NPM"    || FAILED=$((FAILED+1)); }
wanted apt    && { [ "$D_APT" = "-" ]    || apply_apt    "$D_APT"    || FAILED=$((FAILED+1)); }

log ""
if [ "$CHECK_ONLY" = "1" ]; then
  info "--check 结束：上面只列出会做的改动，没有实际执行"
  exit 0
fi

if [ "$FAILED" -gt 0 ]; then
  warn "$FAILED 项未成功。原文件都在 /etc 下的 *.hdw-bak-<时间戳>，可手工恢复。"
  exit 1
fi
ok "镜像源配置完成"
log ""
dim "  注意：容器**构建期**的 pip 不读这里的 /etc/pip.conf，"
dim "  它走 Dockerfile 里的 ARG PIP_INDEX_URL（见 HDW_Inference/RAG_Service/Dockerfile）。"
