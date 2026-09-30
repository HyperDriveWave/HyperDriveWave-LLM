#!/usr/bin/env bash
# 取「离线部署」需要的构建工具，放进仓库里的固定位置，好整包带到无外网的机器上。
#
# 现在只有一样东西：cmake。
#
# 为什么需要
# ──────────
# HDW_Inference/llama/Dockerfile.container 要在 nvidia/cuda:*-devel-* 镜像里编
# llama.cpp。那个镜像有 nvcc / gcc / make，**没有 cmake**。有外网时在镜像里
# apt 装一下就行，无外网的机器（麒麟 V10 等）装不了——所以得提前把 cmake 带过去。
#
# 为什么不用宿主现成的 cmake
# ──────────────────────────
# 宿主那份是按它自己的 glibc 编的。麒麟 V10 这类老发行版根本跑不起来它，
# 而这正是要走容器方案的原因——绕回去就自相矛盾了。
# PyPI 的 cmake wheel 是 manylinux_2_17 的，自包含、不挑 glibc，容器里直接能跑。
#
# 用法（**在有外网的机器上跑**）：
#     bash Scripts/fetch_offline_toolchain.sh
#
# 产物落在 HDW_Inference/llama/cmake-offline/，已被 .gitignore 排除（30 MB 的
# 二进制不进库）。打包带走时记得它是 gitignore 的，别指望 git 把它带上。

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DEST="$ROOT/HDW_Inference/llama/cmake-offline"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

info() { printf '  %s\n' "$*"; }
die()  { printf '错误：%s\n' "$*" >&2; exit 1; }

# pip 的版本差异：老 Python 上用 pip3，实在没有就用 python3 -m pip。
PIP=""
for cand in "python3 -m pip" "pip3" "pip"; do
  if $cand --version >/dev/null 2>&1; then PIP="$cand"; break; fi
done
[ -n "$PIP" ] || die "找不到可用的 pip。这台机器需要有外网，且装好 pip。"

info "下载 cmake wheel（manylinux，自包含）"
# --only-binary :all: 很关键：否则 pip 可能去拉源码包，那反而需要先有 cmake 才能装。
# --no-deps：cmake wheel 没有依赖，带上反而可能拉进别的东西。
$PIP download cmake --no-deps --only-binary :all: -d "$TMP" \
  || die "下载失败。确认这台机器能访问 PyPI（或先配好 PIP_INDEX_URL 指向的镜像）。"

WHEEL="$(find "$TMP" -maxdepth 1 -name 'cmake-*.whl' -print -quit)"
[ -n "$WHEEL" ] || die "没下到 cmake 的 wheel，$TMP 里是：$(ls "$TMP")"

info "解压到 $DEST"
# 保留 README.md：它既是说明，也保证这个目录在 .dockerignore 之后仍然存在
# （Dockerfile 里 `COPY cmake-offline /opt/cmake-offline` 要有个目录可拷）。
mkdir -p "$DEST"
find "$DEST" -mindepth 1 -maxdepth 1 ! -name 'README.md' -exec rm -rf {} +
python3 -m zipfile -e "$WHEEL" "$DEST" \
  || die "解压失败。需要 python3 带 zipfile 模块（标准库，正常都有）。"

CMAKE_BIN="$DEST/cmake/data/bin/cmake"
[ -x "$CMAKE_BIN" ] || chmod +x "$CMAKE_BIN" 2>/dev/null || true
[ -x "$CMAKE_BIN" ] || die "解压后没找到 $CMAKE_BIN —— wheel 的内部布局可能变了，请检查 $DEST"

# 顺手在**本机**跑一下。本机能跑不代表容器里能跑（glibc 不同），但本机都跑不起来
# 就说明 wheel 本身有问题，早点发现比在目标机上排查便宜得多。
info "本机自检：$("$CMAKE_BIN" --version | head -1)"
info "完成。$DEST 现在 $(du -sh "$DEST" | cut -f1)。"
info "提醒：这个目录是 gitignore 的，整包搬机器时要用 rsync/tar，别指望 git。"
