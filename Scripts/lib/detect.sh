#!/usr/bin/env bash
# 环境探测 → 决策变量。被 source，不单独执行。
#
# 原则：探测只读环境，不写任何文件。决策结果由调用方写进 .deploy/plan.env，
# 后续阶段只读那份快照，不重复探测（避免中途环境变化导致前后不一致）。

[ -n "${HDW_DETECT_SH_LOADED:-}" ] && return 0
HDW_DETECT_SH_LOADED=1

# detect.sh 可能被**单独** source（HDW_Inference/llama/start.sh 就是这么用的，
# 它不该为此再拖进整个 common.sh）。所以这里补上自己用到的 common.sh 函数，
# 而不是隐式依赖调用方先 source 过 common.sh。
#
# 踩过的坑：少了这个兜底，detect_gpu 里的 `have_cmd nvidia-smi` 会
# "command not found" → set -e 下静默返回 → GPU 数被当成 0 →
# 后端选择跳过所有 CUDA 分支，直接回退 Vulkan。
if ! declare -F have_cmd >/dev/null 2>&1; then
  have_cmd() { command -v "$1" >/dev/null 2>&1; }
fi

# ── GPU ───────────────────────────────────────────────────────
# 输出全局：HDW_GPU_COUNT / HDW_GPU_NAME / HDW_GPU_CAP / HDW_GPU_VRAM_MIB
#           HDW_GPU_FREE_MIB / HDW_GPU_BEST_INDEX
detect_gpu() {
  HDW_GPU_COUNT=0; HDW_GPU_NAME=""; HDW_GPU_CAP=""
  HDW_GPU_VRAM_MIB=0; HDW_GPU_FREE_MIB=0; HDW_GPU_BEST_INDEX=0

  if ! have_cmd nvidia-smi; then
    return 0
  fi
  # 只看计算卡（排除 nvidia-smi 里的其它设备类型），拿每张卡的算力/显存/空闲
  local rows
  rows="$(nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.free \
          --format=csv,noheader,nounits 2>/dev/null)" || return 0
  [ -n "$rows" ] || return 0

  HDW_GPU_COUNT="$(printf '%s\n' "$rows" | wc -l)"
  # 选**空闲显存最大**的那张，不是 index 0——多卡机上 0 号卡常被别的进程占着，
  # 而 llama/start.sh 的显存门槛只看"卡 0"，选错了会静默降级到 CPU。
  local best_free=-1 idx name cap total free
  while IFS=, read -r idx name cap total free; do
    idx="$(echo "$idx" | xargs)"; cap="$(echo "$cap" | xargs)"
    total="$(echo "$total" | xargs)"; free="$(echo "$free" | xargs)"
    if [ "${free:-0}" -gt "$best_free" ] 2>/dev/null; then
      best_free="$free"; HDW_GPU_BEST_INDEX="$idx"
      HDW_GPU_NAME="$name"; HDW_GPU_CAP="$cap"
      HDW_GPU_VRAM_MIB="$total"; HDW_GPU_FREE_MIB="$free"
    fi
  done <<< "$rows"
}

# ── CUDA 后端可用性 ───────────────────────────────────────────
# 读构建目录的 CMakeCache 拿编译时锁定的架构。二进制里读不可靠，
# CMakeCache 是 cmake 自己写的，最准且不需要 nvcc。
llama_backend_arches() {
  local dir="$1"
  local cache="$dir/CMakeCache.txt"
  [ -f "$cache" ] || { echo ""; return; }
  sed -n 's/^CMAKE_CUDA_ARCHITECTURES:[^=]*=//p' "$cache" | head -1
}

# 目标卡的 compute_cap（如 12.0）是否被某个构建目录覆盖（架构写成 120 或 80;89;90;120）
arch_covers_cap() {
  local arches="$1" cap="$2"
  [ -n "$arches" ] && [ -n "$cap" ] || return 1
  local want
  want="$(awk -v c="$cap" 'BEGIN { split(c, p, "."); printf "%d%d", p[1], p[2] }')"
  local a
  for a in ${arches//;/ }; do
    [ "$a" = "$want" ] && return 0
  done
  return 1
}

# 选后端目录，输出**相对于 llama 目录**的路径（如 llama.cpp-upstream/build-cuda-multi），
# 空表示没有可用的。
#
# 两个要点：
#   1. 构建产物分布在两个父目录下——`<llama>/build*` 是早期在源码树里编的，
#      `<llama>/llama.cpp-upstream/build*` 是现在在用的。只查一个会漏。
#   2. 判定不看"目录存在"，而看"编的架构能不能在目标卡上跑"。
#      只看存在会把只含 sm_120a 的二进制发给 4090，跑起来才炸
#      （报 no kernel image is available，而 /health 是能过的）。
select_llama_backend_dir() {
  local llama_dir="$1" cap="$2" count="$3"
  local cand root arches

  # 多架构优先，其次单架构 CUDA，最后 Vulkan
  for cand in build-cuda-multi build-cuda; do
    [ "$count" -gt 0 ] || break
    for root in "llama.cpp-upstream/$cand" "$cand"; do
      [ -x "$llama_dir/$root/bin/llama-server" ] || continue
      arches="$(llama_backend_arches "$llama_dir/$root")"
      if arch_covers_cap "$arches" "$cap"; then
        echo "$root"; return 0
      fi
    done
  done

  # 无 CUDA 可用时的兜底，按优先级：
  #   build-vulkan / build  —— Vulkan，不需要 NVIDIA 卡，但旧版 llama.cpp
  #                            不支持 --spec-type（MTP 失效），也可能不认新模型架构
  #   build-cpu             —— 纯 CPU，最慢但一定能跑
  # 选非 CUDA 后端必须让调用方给出明确警告，而不是静默降级。
  for root in llama.cpp-upstream/build-vulkan llama.cpp-upstream/build \
              build-vulkan build; do
    if [ -x "$llama_dir/$root/bin/llama-server" ]; then
      echo "$root"; return 0
    fi
  done
  for root in llama.cpp-upstream/build-cpu build-cpu; do
    if [ -x "$llama_dir/$root/bin/llama-server" ]; then
      echo "$root"; return 0
    fi
  done
  echo ""
}

# ── 网络 ──────────────────────────────────────────────────────

detect_lan_ip() {
  # 优先取默认路由出口地址；拿不到再退回第一个非回环的全局地址。
  local ip
  ip="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')"
  if [ -z "$ip" ]; then
    ip="$(ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)"
  fi
  printf '%s' "${ip:-127.0.0.1}"
}

# 探一次能否直连魔搭。用 HEAD 拿状态码，超时给短一点——探测不该让部署卡住。
probe_modelscope() {
  local domain="${MODELSCOPE_DOMAIN:-www.modelscope.cn}"
  curl -fsS --max-time 8 -o /dev/null "https://$domain" 2>/dev/null
}

probe_docker_hub() {
  docker pull --quiet nginx:1.27-alpine >/dev/null 2>&1
}

# 检测 Docker 是否配了镜像源
docker_registry_mirrors() {
  docker info --format '{{json .RegistryConfig.Mirrors}}' 2>/dev/null || echo "[]"
}

# 检测 nvidia container runtime 是否可用（compose 里 GPU 相关配置的前提）
#
# 用 here-string 而不是管道：调用方普遍开了 pipefail，而 `生产者 | grep -q`
# 会在 grep 提前退出时让生产者吃 SIGPIPE（141），整条管道被判失败。
# 后果不是报错而是**静默给出错误答案**——这里会变成"明明有 nvidia runtime 却说没有"。
docker_has_nvidia_runtime() {
  local runtimes
  runtimes="$(docker info --format '{{json .Runtimes}}' 2>/dev/null)"
  grep -q nvidia <<< "$runtimes"
}

# ── CUDA runtime 依赖 ─────────────────────────────────────────
# build-cuda* 的 .so 链接 /usr/local/cuda/.../libcudart.so.13 等。
# 目标机只有驱动没有 toolkit 时，二进制起不来——这个检查比事后报错便宜得多。
cuda_runtime_present() {
  local libdir="/usr/local/cuda/targets/x86_64-linux/lib"
  [ -e "$libdir/libcudart.so" ] || [ -e "$libdir/libcudart.so.13" ]
}

# 把 CUDA runtime 依赖收进 bin/，让二进制自包含。
# 用迭代 ldd 求闭包：第一次 ldd 出来的库里还有指向 CUDA 目录的，
# 一并收走，直到不再新增（一般 2 轮收敛）。
bundle_cuda_runtime() {
  local bindir="$1"
  local libdir="/usr/local/cuda/targets/x86_64-linux/lib"
  [ -d "$bindir" ] || return 1
  [ -d "$libdir" ] || return 1

  local elf found=0 pass=0 lib
  declare -A seen=()
  local -a queue=()
  # next 必须在循环外声明：写在循环体里的 local -a 每轮都会重置，累积会丢
  local -a next=()

  for elf in "$bindir"/llama-server "$bindir"/*.so; do
    [ -f "$elf" ] && queue+=("$elf")
  done

  while [ "${#queue[@]}" -gt 0 ] && [ "$pass" -lt 5 ]; do
    pass=$((pass + 1))
    next=()
    while IFS= read -r lib; do
      [ -n "$lib" ] || continue
      case "$lib" in
        "$libdir"/*) ;;
        *) continue ;;
      esac
      local base; base="$(basename "$lib")"
      [ -n "${seen["$base"]:-}" ] && continue
      seen["$base"]=1
      cp -Lf "$lib" "$bindir/$base" 2>/dev/null && found=$((found + 1))
      next+=("$bindir/$base")
    done < <(ldd "${queue[@]}" 2>/dev/null | awk '/=>/ {print $3}' | sort -u)
    queue=("${next[@]}")
  done

  [ "$found" -gt 0 ]
}

# ── 其它前置条件 ──────────────────────────────────────────────

# 端口是否被占用。部署前查，免得跑到 compose up 才失败。
# 同样避开管道（见 docker_has_nvidia_runtime 的说明）：这里误判成"空闲"
# 会让部署跑到一半才在 compose up 炸掉。
port_busy() {
  local port="$1" addrs
  # ss 缺失时必须当成"查不出来"，而不是"不忙"。踩过的坑：
  # 没有这个兜底时 `ss` 报 command not found（退出码 127），
  # `if port_busy` 把它当假 → 判定端口空闲 → **挑到一个被占的端口且不自知**。
  # 自动选端口依赖这个函数，所以宁可保守报"忙"。
  if ! have_cmd ss; then
    return 0
  fi
  addrs="$(ss -ltnH 2>/dev/null | awk '{print $4}')"
  [ -n "$addrs" ] || return 0      # ss 输出为空同样可疑，保守处理
  grep -qE "[:.]${port}$" <<< "$addrs"
}

# 端口是不是被**本项目自己的**东西占着。
# 部署脚本要能重复执行：重跑时 3000/8080/8001 必然被上次起的容器占着，
# 那是正常状态，不该报"端口被占用"——真正的冲突是别的进程占的。
#
# 两类都要认，缺一不可：
#   1. compose 容器（名字前缀 hyperdrivewave-）
#   2. **宿主进程**：llama-server 和 frpc。它们不在 docker ps 里，
#      不认的话 1919 永远被判成"外来冲突"，而它其实是本项目自己的。
port_held_by_hdw() {
  local port="$1" ports
  ports="$(docker ps --filter 'name=hyperdrivewave-' --format '{{.Ports}}' 2>/dev/null)"
  grep -qE "[:.]${port}->" <<< "$ports" && return 0
  port_held_by_host_service "$port"
}

# 宿主上的本项目进程（用户级 systemd 服务）是否在监听该端口。
# 用 systemd 的单元名判断，比翻 /proc/net 稳，也不依赖 ss 的输出格式。
port_held_by_host_service() {
  local port="$1"
  have_cmd systemctl || return 1
  # llama 的端口是可配的，从 .env 读，不写死 1919
  local llama_port="1919"
  if [ -f "${HDW_ROOT:-}/Configs/.env" ]; then
    local v
    v="$(sed -n 's/^HDW_LLAMA_PORT=//p' "$HDW_ROOT/Configs/.env" 2>/dev/null | tail -1)"
    [ -n "$v" ] && llama_port="$v"
  fi
  if [ "$port" = "$llama_port" ]; then
    systemctl --user is-active --quiet hyperdrivewave-llama.service 2>/dev/null && return 0
    # 服务没起但端口被占，说明是别的进程抢了这个端口，不算"我们的"
    return 1
  fi
  return 1
}

# 找一个空闲端口。从 preferred 开始往上试，最多试 span 个。
# 输出找到的端口号；找不到输出空。
port_find_free() {
  local preferred="$1" span="${2:-200}" i p
  for (( i = 0; i < span; i++ )); do
    p=$(( preferred + i ))
    [ "$p" -gt 65535 ] && break
    if ! port_busy "$p"; then
      echo "$p"
      return 0
    fi
  done
  echo ""
  return 1
}

disk_free_gb() {
  df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'
}

# Linger=no 时用户级服务只在登录会话存在时运行，重启后三个服务都不会自启。
# 这是"部署完看着好、重启就没了"的元凶。
linger_enabled() {
  local user="${1:-$(whoami)}"
  have_cmd loginctl || return 1
  [ "$(loginctl show-user "$user" -p Linger --value 2>/dev/null)" = "yes" ]
}
