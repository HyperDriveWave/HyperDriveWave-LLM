#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${HDW_RUNTIME_ROOT:-$ROOT/HDW_Runtime}"

mkdir -p \
  "$RUNTIME_ROOT/postgres" \
  "$RUNTIME_ROOT/redis" \
  "$RUNTIME_ROOT/neo4j/data" \
  "$RUNTIME_ROOT/neo4j/logs" \
  "$RUNTIME_ROOT/rag" \
  "$RUNTIME_ROOT/graph_review" \
  "$RUNTIME_ROOT/maintenance" \
  "$RUNTIME_ROOT/knowledge_sources" \
  "$RUNTIME_ROOT/chatdata" \
  "$RUNTIME_ROOT/zvec" \
  "$RUNTIME_ROOT/mineru/uploads" \
  "$RUNTIME_ROOT/mineru/parsed" \
  "$RUNTIME_ROOT/mineru/tasks" \
  "$RUNTIME_ROOT/mineru/logs" \
  "$RUNTIME_ROOT/open-webui" \
  "$RUNTIME_ROOT/langfuse/postgres" \
  "$RUNTIME_ROOT/langfuse/clickhouse" \
  "$RUNTIME_ROOT/langfuse/redis" \
  "$RUNTIME_ROOT/langfuse/minio" \
  "$RUNTIME_ROOT/keycloak" \
  "$RUNTIME_ROOT/dify" \
  "$RUNTIME_ROOT/n8n" \
  "$RUNTIME_ROOT/backups"

echo "prepared runtime dirs under $RUNTIME_ROOT"
