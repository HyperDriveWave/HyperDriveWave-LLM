#!/usr/bin/env bash
# 从源码构建 llama.cpp。
#
# 为什么需要它：`llama.cpp-upstream/` 和所有 `build*/` 都不进版本库
# （源码树 1.4G、CUDA 产物 974M），所以**从 GitHub 克隆下来的项目里没有
# llama-server 二进制**。不构建的话部署第一步就会死在
# 「找不到可用的 llama-server 二进制」。
#
# 用法：
#   bash Scripts/build_llama.sh              # 自动探测显卡并构建
#   bash Scripts/build_llama.sh --check      # 只报当前状态
#   bash Scripts/build_llama.sh --cpu        # 强制 CPU 版（无卡机器）
#   bash Scripts/build_llama.sh --vulkan     # 强制 Vulkan 版
#   bash Scripts/build_llama.sh --rebuild    # 忽略已有产物重编
#
# 耗时：CUDA 单架构约 10 分钟，Vulkan 约 5 分钟，CPU 约 3 分钟。
# 用 `-j$(nproc)` 并行编译；24 核机器实测 CUDA 单架构 8-12 分钟。

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HDW_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export HDW_ROOT

. "$SCRIPT_DIR/lib/common.sh"
. "$SCRIPT_DIR/lib/detect.sh"

LLAMA_DIR="$HDW_ROOT/HDW_Inference/llama"
SRC_DIR="$LLAMA_DIR/llama.cpp-upstream"

FORCE_BACKEND=""
CHECK_ONLY=0
REBUILD=0
JOBS="$(nproc 2>/dev/null || echo 4)"

while [ $# -gt 0 ]; do
  case "$1" in
    --check)   CHECK_ONLY=1; shift ;;
    --cpu)     FORCE_BACKEND=cpu; shift ;;
    --vulkan)  FORCE_BACKEND=vulkan; shift ;;
    --rebuild) REBUILD=1; shift ;;
    --jobs)    JOBS="${2:-4}"; shift 2 ;;
    -h|--help) sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done

# ── 后端选择 ──────────────────────────────────────────────────

choose_backend() {
  [ -n "$FORCE_BACKEND" ] && { echo "$FORCE_BACKEND"; return; }
  detect_gpu
  if [ "$HDW_GPU_COUNT" -gt 0 ]; then
    echo "cuda"
  else
    echo "cpu"
  fi
}

# CUDA 架构号：把 compute_cap 的 "12.0" 变成 "120"
cuda_arch_of_cap() {
  awk -v c="$1" 'BEGIN { split(c, p, "."); printf "%d%d", p[1], p[2] }'
}

# ── 工具链 ────────────────────────────────────────────────────

apt_install() {
  local pkgs=("$@")
  info "  apt 安装：${pkgs[*]}（需要 sudo）"
  sudo apt-get update -qq || true
  sudo apt-get install -y --no-install-recommends "${pkgs[@]}"
}

ensure_build_tools() {
  local missing=()
  have_cmd cmake || missing+=(cmake)
  have_cmd make  || missing+=(make)
  have_cmd g++   || missing+=(g++)
  [ "${#missing[@]}" -gt 0 ] || return 0
  warn "缺少编译工具：${missing[*]}"
  if confirm "  用 apt 安装？"; then
    apt_install "${missing[@]}" || return 1
  else
    return 1
  fi
}

# 找 nvcc。本机装在 /usr/local/cuda-13.1/bin，不在 PATH 里。
find_nvcc() {
  have_cmd nvcc && { command -v nvcc; return 0; }
  local c
  for c in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
    [ -x "$c" ] && { echo "$c"; return 0; }
  done
  return 1
}

# CUDA toolkit 在 Ubuntu 的 multiverse 源里，不需要加 NVIDIA 官方源。
# 包名形如 cuda-toolkit-13-1，版本随发行版不同，所以动态找。
ensure_cuda_toolkit() {
  local nvcc
  if nvcc="$(find_nvcc)"; then
    info "  已找到 nvcc：$nvcc"
    NVCC="$nvcc"
    return 0
  fi
  warn "  没找到 nvcc（CUDA 编译器）"

  local pkg
  pkg="$(apt-cache search --names-only '^cuda-toolkit-[0-9]+-[0-9]+$' 2>/dev/null \
         | awk '{print $1}' | sort -V | tail -1)"
  if [ -z "$pkg" ]; then
    warn "  当前 apt 源里没有 cuda-toolkit 包。
     它来自 Ubuntu 的 multiverse 组件。确认 /etc/apt/sources.list.d/ubuntu.sources
     里的 Components 含 multiverse，然后 apt-get update。"
    return 1
  fi
  warn "  可安装：$pkg（约 3-5G）"
  if ! confirm "  用 apt 安装 CUDA toolkit？"; then
    return 1
  fi
  apt_install "$pkg" || return 1
  if nvcc="$(find_nvcc)"; then
    NVCC="$nvcc"
    ok "  nvcc 已就绪：$nvcc"
    return 0
  fi
  warn "  装完仍找不到 nvcc"
  return 1
}

# ── 构建 ──────────────────────────────────────────────────────

BUILD_DIR=""
BACKEND=""

# 已有可用产物？架构覆盖目标卡就跳过编译。
already_built() {
  local dir="$1" arch="$2"
  [ -x "$dir/bin/llama-server" ] || return 1
  local arches
  arches="$(llama_backend_arches "$dir")"
  [ -n "$arches" ] || return 1
  local a
  for a in ${arches//;/ }; do
    [ "$a" = "$arch" ] && return 0
  done
  return 1
}

do_build() {
  local backend="$1"
  local cmake_args=(
    -DCMAKE_BUILD_TYPE=Release
    # GGML_NATIVE=OFF：默认的 -march=native 会按**编译机**的 CPU 生成指令，
    # 拷到老 CPU 上会 SIGILL。显式开一组 2013 年后都有的指令集替代。
    -DGGML_NATIVE=OFF
    -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON
    -DBUILD_SHARED_LIBS=ON
    -DLLAMA_BUILD_SERVER=ON
    -DLLAMA_CURL=OFF          # 不需要从网上下模型
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_BUILD_EXAMPLES=OFF
  )

  case "$backend" in
    cuda)
      local arch; arch="$(cuda_arch_of_cap "$HDW_GPU_CAP")"
      cmake_args+=(-DGGML_CUDA=ON -DGGML_CUDA_FA=ON "-DCMAKE_CUDA_ARCHITECTURES=$arch")
      # 只编本机架构（用户确认的策略）：约 10 分钟。要编多架构改成
      # -DCMAKE_CUDA_ARCHITECTURES="80;89;90;120"，耗时约 40 分钟。
      cmake_args+=("-DCMAKE_CUDA_COMPILER=$NVCC")
      BUILD_DIR="$SRC_DIR/build-cuda"
      info "  目标架构：sm_$arch（显卡 $HDW_GPU_NAME，cap $HDW_GPU_CAP）"
      ;;
    vulkan)
      cmake_args+=(-DGGML_VULKAN=ON)
      BUILD_DIR="$SRC_DIR/build-vulkan"
      ;;
    *)
      cmake_args+=(-DGGML_CUDA=OFF -DGGML_VULKAN=OFF)
      BUILD_DIR="$SRC_DIR/build-cpu"
      ;;
  esac

  info "  配置（$BUILD_DIR）…"
  ( cd "$SRC_DIR" && cmake -B "$(basename "$BUILD_DIR")" -S . "${cmake_args[@]}" ) \
    || { fail "  cmake 配置失败"; return 1; }

  info "  编译（-j$JOBS，这一步最慢，请耐心等）…"
  ( cd "$SRC_DIR" && cmake --build "$(basename "$BUILD_DIR")" -j "$JOBS" --target llama-server ) \
    || { fail "  编译失败"; return 1; }

  [ -x "$BUILD_DIR/bin/llama-server" ] || { fail "  编完了但没有 llama-server"; return 1; }
  return 0
}

# ── 主流程 ────────────────────────────────────────────────────

step "检查 llama.cpp 源码"

if [ ! -d "$SRC_DIR" ]; then
  warn "源码目录不存在：$SRC_DIR"
  log ""
  log "  从版本库克隆下来的项目**不含**这个目录（它 1.4G，不进库）。先跑："
  log ""
  log "    bash Scripts/fetch_vendors.sh --with llama.cpp-upstream"
  log ""
  log "  注意：llama.cpp 官方源在 github.com，没有 gitee 镜像。"
  log "  如果这台机器连不上 github，改用离线包（pack_hdw.sh 打出来的包里带着源码）。"
  exit 1
fi
[ -d "$SRC_DIR/.git" ] || warn "源码目录不是 git 仓库——可能是从离线包解出来的，继续"
ok "源码就绪：$(git -C "$SRC_DIR" rev-parse --short HEAD 2>/dev/null || echo '未知版本')"

step "确定构建目标"

BACKEND="$(choose_backend)"
detect_gpu
[ -n "$FORCE_BACKEND" ] && info "按参数强制：$BACKEND"

case "$BACKEND" in
  cuda)   info "构建 CUDA 版（显卡：${HDW_GPU_NAME:-未知}，cap ${HDW_GPU_CAP:-?}）" ;;
  cpu)    info "构建 CPU 版（未检测到 NVIDIA 显卡）" ;;
  vulkan) info "构建 Vulkan 版" ;;
esac

# 已有产物就不用重编
if [ "$REBUILD" != "1" ]; then
  case "$BACKEND" in
    cuda)
      _arch="$(cuda_arch_of_cap "$HDW_GPU_CAP")"
      if already_built "$SRC_DIR/build-cuda" "$_arch"; then
        ok "已有覆盖 sm_$_arch 的 CUDA 产物，无需重编"
        dim "  要强制重编加 --rebuild"
        exit 0
      fi
      ;;
    vulkan) [ -x "$SRC_DIR/build-vulkan/bin/llama-server" ] && { ok "已有 Vulkan 产物"; exit 0; } ;;
    cpu)    [ -x "$SRC_DIR/build-cpu/bin/llama-server" ]    && { ok "已有 CPU 产物"; exit 0; } ;;
  esac
fi

if [ "$CHECK_ONLY" = "1" ]; then
  log ""
  info "--check：以上是当前状态，未做任何改动"
  if have_cmd cmake && have_cmd g++; then
    dim "  编译工具：cmake / g++ 已就绪"
  else
    warn "  编译工具缺失，需要 apt 安装 cmake / build-essential"
  fi
  if find_nvcc >/dev/null 2>&1; then
    dim "  CUDA：$(find_nvcc)"
  else
    dim "  CUDA：未安装（需要时会尝试 apt 装 cuda-toolkit-*）"
  fi
  exit 0
fi

step "准备工具链"

ensure_build_tools || die "编译工具不可用，无法构建"

NVCC=""
if [ "$BACKEND" = "cuda" ]; then
  if ! ensure_cuda_toolkit; then
    warn "CUDA toolkit 不可用，降级为 CPU 版构建。
     没有 GPU 加速的话本地推理会非常慢（27B 模型约 1 token/s）。"
    BACKEND=cpu
  fi
fi

step "编译 llama.cpp"

info "耗时提示：CUDA 约 10 分钟，Vulkan 约 5 分钟，CPU 约 3 分钟"
log ""
_start=$(date +%s)
do_build "$BACKEND" || die "构建失败，看上面的编译输出定位"
_elapsed=$(( $(date +%s) - _start ))
ok "编译完成，用时 $(( _elapsed / 60 )) 分 $(( _elapsed % 60 )) 秒"

# ── 收拢 CUDA runtime ──
# build-cuda 的 .so 链接 /usr/local/cuda/.../libcudart.so.13 等。
# 把依赖库拷进 bin/ 并靠 LD_LIBRARY_PATH 生效（start.sh 会设），
# 目标机就**不需要装 CUDA toolkit**——只需要显卡驱动。
if [ "$BACKEND" = "cuda" ]; then
  step "收拢 CUDA runtime"
  if bundle_cuda_runtime "$BUILD_DIR/bin"; then
    ok "已随包携带 CUDA runtime（目标机无需再装 toolkit）"
    dim "  bin/ 现在 $(du -sh "$BUILD_DIR/bin" | cut -f1)"
  else
    warn "没能收拢 CUDA runtime。这台机器上能跑（有 toolkit），
     但拷到别的机器可能起不来——那边需要装 cuda-toolkit。"
  fi
fi

# ── 验证 ──
step "验证产物"

BIN="$BUILD_DIR/bin/llama-server"
export LD_LIBRARY_PATH="$BUILD_DIR/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if DEVOUT="$("$BIN" --list-devices 2>&1)"; then
  log "$DEVOUT" | sed 's/^/    /'
  case "$BACKEND" in
    cuda) echo "$DEVOUT" | grep -q "CUDA" && ok "CUDA 设备可用" || warn "没列出 CUDA 设备" ;;
  esac
else
  warn "llama-server 执行失败，产物可能不可用"
fi

log ""
ok "llama.cpp 构建完成"
log ""
log "  后端：$BACKEND"
log "  产物：$BUILD_DIR/bin/llama-server"
log ""
dim "  start.sh 会自动选中它（按可用性排序：多架构 CUDA > 单架构 CUDA > Vulkan > CPU）"
