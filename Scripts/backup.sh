#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${HDW_RUNTIME_ROOT:-$ROOT/HDW_Runtime}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$RUNTIME_ROOT/backups/hdw-$STAMP.tgz"

tar -czf "$OUT" -C "$ROOT" Configs HDW_Runtime HDW_DataFoundation/RelationalDB HDW_Evaluation/RAGAS
echo "backup written: $OUT"

