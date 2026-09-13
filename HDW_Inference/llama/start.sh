#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
MODEL_DEFAULT="$ROOT/HDW_Engines/LLM_Models/Qwen3.8-27B-GSQ/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"

MODEL_CONFIG="${HDW_MODEL_CONFIG_PATH:-$ROOT/HDW_Runtime/model-config/config.json}"
case "$MODEL_CONFIG" in
  /data/model-config/*)
    MODEL_CONFIG="$ROOT/HDW_Runtime/model-config/${MODEL_CONFIG##*/}"
    ;;
esac

CONFIG_ENGINE=""
CONFIG_MODEL=""
CONFIG_CTX=""
CONFIG_MTP=""
CONFIG_MMPROJ=""
if [ -f "$MODEL_CONFIG" ] && command -v python3 >/dev/null 2>&1; then
  IFS=$'\t' read -r CONFIG_ENGINE CONFIG_MODEL CONFIG_CTX CONFIG_MTP CONFIG_MMPROJ < <(
    MODEL_CONFIG="$MODEL_CONFIG" python3 - <<'PY'
import json
import os

try:
    with open(os.environ["MODEL_CONFIG"], encoding="utf-8") as source:
        config = json.load(source)
except (OSError, ValueError):
    config = {}
local = config.get("local") if isinstance(config, dict) else {}
local = local if isinstance(local, dict) else {}
print(
    "\t".join(
        str(local.get(key) or "")
        for key in ("engine", "model", "context_window", "mtp_enabled", "mmproj")
    )
)
PY
  )
fi

ENGINE="${HDW_LLAMA_ENGINE:-${CONFIG_ENGINE:-llama.cpp}}"
MODEL="${HDW_LLAMA_MODEL:-${CONFIG_MODEL:-${HDW_LOCAL_LLM_MODEL:-$MODEL_DEFAULT}}}"
HOST="${HDW_LLAMA_HOST:-0.0.0.0}"
PORT="${HDW_LLAMA_PORT:-1919}"
DEVICE="${HDW_LLAMA_DEVICE:-Vulkan0}"
CTX="${HDW_LLAMA_CTX:-${CONFIG_CTX:-262144}}"
GPU_LAYERS="${HDW_LLAMA_GPU_LAYERS:-}"
SERVER_TIMEOUT="${HDW_LLAMA_TIMEOUT:-0}"
# 视觉投影器。来源优先级与 MODEL/CTX/MTP 同构：环境变量 > 模型配置 > 自动发现。
# 配置里可以写绝对路径，也可以只写文件名——下面的解析会去模型目录找它。
MMPROJ="${HDW_LLAMA_MMPROJ:-${CONFIG_MMPROJ:-}}"
# MTP（Multi-Token Prediction）投机解码。来源优先级：环境变量 > 模型配置的 mtp_enabled。
MTP_ENABLED="${HDW_LLAMA_MTP:-${CONFIG_MTP:-}}"
MTP_ENABLED="$(printf '%s' "$MTP_ENABLED" | tr '[:upper:]' '[:lower:]')"

if [ "${ENGINE,,}" = "freetoken" ]; then
  if ! command -v ft >/dev/null 2>&1; then
    echo "FreeToken is selected in $MODEL_CONFIG, but the ft command is not installed" >&2
    exit 1
  fi
  if [ ! -e "$MODEL" ]; then
    MODEL="$ROOT/HDW_Engines/LLM_Models/$MODEL"
  fi
  if [ ! -e "$MODEL" ]; then
    echo "missing FreeToken model: $MODEL" >&2
    exit 1
  fi
  exec ft serve \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --max-seq-len-override "$CTX"
fi

case "${ENGINE,,}" in
  llama.cpp|llama-cpp|llama)
    ;;
  *)
    echo "unsupported local inference engine: $ENGINE" >&2
    exit 1
    ;;
esac

if [ ! -f "$MODEL" ]; then
  MODEL_CANDIDATE="$ROOT/HDW_Engines/LLM_Models/$MODEL"
  if [ -f "$MODEL_CANDIDATE" ]; then
    MODEL="$MODEL_CANDIDATE"
  else
    MODEL_CANDIDATE="$(find "$ROOT/HDW_Engines/LLM_Models" -maxdepth 3 -type f -name "$(basename "$MODEL")" -print -quit)"
    if [ -n "$MODEL_CANDIDATE" ]; then
      MODEL="$MODEL_CANDIDATE"
    fi
  fi
fi

if [ -z "$GPU_LAYERS" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    FREE_VRAM_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
    if [ "${FREE_VRAM_MIB:-0}" -ge 20000 ]; then
      GPU_LAYERS=all
    else
      GPU_LAYERS=0
    fi
  else
    GPU_LAYERS=0
  fi
fi

if [ "$GPU_LAYERS" = "0" ]; then
  DEVICE=none
fi

if [ ! -f "$MODEL" ]; then
  echo "missing model: $MODEL" >&2
  exit 1
fi

if [ -z "$MMPROJ" ]; then
  while IFS= read -r candidate; do
    MMPROJ="$candidate"
    break
  done < <(find "$(dirname "$MODEL")" -maxdepth 1 -type f -name 'mmproj*.gguf' -print | sort)
fi

if [ -n "$MMPROJ" ] && [ ! -f "$MMPROJ" ]; then
  MMPROJ_CANDIDATE="$ROOT/HDW_Engines/LLM_Models/$MMPROJ"
  if [ -f "$MMPROJ_CANDIDATE" ]; then
    MMPROJ="$MMPROJ_CANDIDATE"
  else
    echo "missing multimodal projector: $MMPROJ" >&2
    exit 1
  fi
fi

LLAMA_ARGS=(
  --model "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --device "$DEVICE" \
  --ctx-size "$CTX" \
  --gpu-layers "$GPU_LAYERS" \
  --timeout "$SERVER_TIMEOUT" \
  --cache-type-k "${HDW_LLAMA_CACHE_TYPE_K:-q4_0}" \
  --cache-type-v "${HDW_LLAMA_CACHE_TYPE_V:-q4_0}" \
  --flash-attn auto
)

if [ -n "$MMPROJ" ]; then
  echo "llama.cpp multimodal projector: $MMPROJ"
  LLAMA_ARGS+=(--mmproj "$MMPROJ")
  if [ "$GPU_LAYERS" != "0" ]; then
    LLAMA_ARGS+=(--mmproj-offload)
  fi
fi

# 后端二进制**自选**，不写死绝对路径——目录一移动，绝对路径就指向不存在的文件。
# 优先级：多架构 CUDA > 单架构 CUDA > Vulkan。
#   build-cuda-multi/  sm_80;89;90;120，可移植
#   build-cuda/        仅 sm_120a，只适合同型号卡
#   build/             Vulkan，旧版 llama.cpp（无 --spec-type，MTP 会自动跳过）
# HDW_LLAMA_BINARY 仍然尊重，但**只接受确实存在的路径**：换机器后它可能指向旧目录，
# 此时退回自动选择而不是直接起不来。
BINARY=""
if [ -n "${HDW_LLAMA_BINARY:-}" ]; then
  if [ -x "$HDW_LLAMA_BINARY" ]; then
    BINARY="$HDW_LLAMA_BINARY"
  else
    echo "HDW_LLAMA_BINARY=$HDW_LLAMA_BINARY 不存在，改用自动选择（项目可能被移动过）" >&2
  fi
fi
if [ -z "$BINARY" ]; then
  # 复用 Scripts/lib/detect.sh 的架构判定，不在这里重写一份：
  # 逻辑是「编的架构能不能在目标卡上跑」，而不是「目录存不存在」。
  _detect="$ROOT/Scripts/lib/detect.sh"
  _gpu_cap=""; _gpu_count=0
  if [ -f "$_detect" ]; then
    # shellcheck source=/dev/null
    . "$_detect"
    detect_gpu
    _gpu_cap="$HDW_GPU_CAP"; _gpu_count="$HDW_GPU_COUNT"
    # shellcheck disable=SC1090
    _backend="$(select_llama_backend_dir "$SCRIPT_DIR" "$_gpu_cap" "$_gpu_count")"
    if [ -n "$_backend" ]; then
      BINARY="$SCRIPT_DIR/$_backend/bin/llama-server"
    fi
  fi
  # detect.sh 不可用时的退化路径：只按目录优先级挑，不校验架构
  if [ -z "$BINARY" ]; then
    for _cand in build-cuda-multi build-cuda build; do
      for _root in "llama.cpp-upstream/$_cand" "$_cand"; do
        if [ -x "$SCRIPT_DIR/$_root/bin/llama-server" ]; then
          BINARY="$SCRIPT_DIR/$_root/bin/llama-server"
          break 2
        fi
      done
    done
  fi
fi
if [ -z "$BINARY" ] || [ ! -x "$BINARY" ]; then
  echo "missing llama-server binary: $SCRIPT_DIR 下没找到可用的后端" >&2
  exit 1
fi

# 二进制的 RUNPATH 是**编译时的绝对路径**，项目一移动就找不到同目录的 .so
# （llama-server 本体只有 17KB，真实代码在 .so 里）。
# 用实际所在目录覆盖，顺带让随包携带的 CUDA runtime 也能被找到，不依赖目标机装 CUDA toolkit。
BINARY_DIR="$(cd -- "$(dirname -- "$BINARY")" && pwd)"
export LD_LIBRARY_PATH="$BINARY_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# --spec-type 只有新版 llama.cpp（llama.cpp-upstream）支持。旧版二进制收到该参数会启动失败，
# 所以这里探测一次，切回旧版二进制时自动跳过 MTP 而不是起不来。
if [ "$MTP_ENABLED" = "true" ] || [ "$MTP_ENABLED" = "1" ] || [ "$MTP_ENABLED" = "yes" ]; then
  if "$BINARY" --help 2>&1 | grep -q -- "--spec-type"; then
    LLAMA_ARGS+=(--spec-type draft-mtp)
    echo "MTP: enabled (--spec-type draft-mtp)"
  else
    echo "MTP: requested but $BINARY does not support --spec-type, ignoring" >&2
  fi
else
  echo "MTP: disabled"
fi

echo "llama.cpp binary: $BINARY (device $DEVICE)"

exec "$BINARY" "${LLAMA_ARGS[@]}"
