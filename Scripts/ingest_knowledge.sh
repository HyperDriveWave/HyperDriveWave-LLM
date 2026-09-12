#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$ROOT/Configs/.env" ]; then
  set -a
  . "$ROOT/Configs/.env"
  set +a
fi
RUNTIME_ROOT="${HDW_RUNTIME_ROOT:-$ROOT/HDW_Runtime}"
INPUT="${1:?usage: Scripts/ingest_knowledge.sh /path/to/docs_md [output.jsonl]}"
OUTPUT="${2:-$RUNTIME_ROOT/rag/chunks.jsonl}"
PARSED="$RUNTIME_ROOT/mineru/parsed"
RAG_URL="${HDW_RAG_URL:-http://127.0.0.1:${HDW_RAG_PORT:-8001}}"
MINERU_URL="${HDW_MINERU_HOST_URL:-http://127.0.0.1:${HDW_MINERU_PORT:-8002}}"
MAINTENANCE_SOCKET="$RUNTIME_ROOT/maintenance/maintenance.sock"
export HDW_NEO4J_HTTP_URL="${HDW_NEO4J_HTTP_URL:-http://127.0.0.1:${HDW_NEO4J_HTTP_PORT:-7474}/db/neo4j/tx/commit}"

bash "$ROOT/Scripts/prepare_dirs.sh"
prepared=0
curl --fail --silent --show-error --unix-socket "$MAINTENANCE_SOCKET" \
  -X POST http://localhost/prepare >/dev/null
prepared=1
restore_resources() {
  if [ "$prepared" -eq 1 ]; then
    curl --fail --silent --show-error --unix-socket "$MAINTENANCE_SOCKET" \
      -X POST http://localhost/restore >/dev/null
    prepared=0
  fi
}
trap restore_resources EXIT

HDW_MINERU_BASE_URL="$MINERU_URL" \
python3 "$ROOT/HDW_DataFoundation/ETL_Pipelines/parse_documents.py" \
  --input "$INPUT" \
  --output "$PARSED"

python3 "$ROOT/HDW_DataFoundation/ETL_Pipelines/ingest_documents.py" \
  --input "$PARSED" \
  --output "$OUTPUT"

python3 "$ROOT/HDW_DataFoundation/ETL_Pipelines/import_graph.py" \
  --input "$OUTPUT" \
  --replace-all

echo "rag index ready: $OUTPUT"
curl -fsS -X POST "$RAG_URL/admin/reindex" >/dev/null
echo "zvec index rebuilt via $RAG_URL"
restore_resources
