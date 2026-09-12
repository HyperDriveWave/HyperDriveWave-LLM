#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

test -f Configs/.env || cp Configs/.env.example Configs/.env
bash Scripts/prepare_dirs.sh

command -v docker >/dev/null || { echo "docker not found"; exit 1; }
docker compose version >/dev/null || { echo "docker compose v2 not found"; exit 1; }

LLAMA_MODEL="${HDW_LLAMA_MODEL:-$ROOT/HDW_Engines/LLM_Models/Qwen3.8-27B-GSQ/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf}"
test -f "$LLAMA_MODEL" || { echo "missing llama model: $LLAMA_MODEL"; exit 1; }
test -d HDW_Engines/RAG_Models/bge-m3 || { echo "missing bge-m3 model"; exit 1; }
test -d HDW_Engines/RAG_Models/bge-reranker-v2-m3 || { echo "missing bge-reranker-v2-m3 model"; exit 1; }

echo "init ok; edit Configs/.env, then run Scripts/start.sh"
