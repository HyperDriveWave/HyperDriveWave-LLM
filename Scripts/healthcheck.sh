#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$ROOT/Configs/.env" ]; then
  set -a
  . "$ROOT/Configs/.env"
  set +a
fi

curl -fsS "http://127.0.0.1:${HDW_QA_API_PORT:-8080}/health" >/dev/null && echo "qa-api ok" || echo "qa-api unavailable"
curl -fsS "http://127.0.0.1:${HDW_RAG_PORT:-8001}/health" >/dev/null && echo "rag ok" || echo "rag unavailable"
curl -fsS "http://127.0.0.1:${HDW_MINERU_PORT:-8002}/health" >/dev/null && echo "mineru ok" || echo "mineru unavailable"
curl -fsS "http://127.0.0.1:${HDW_NEO4J_HTTP_PORT:-7474}" >/dev/null && echo "neo4j ok" || echo "neo4j unavailable"
NEO4J_CURL_AUTH="${NEO4J_AUTH:-neo4j/change_me}"
curl -fsS -u "${NEO4J_CURL_AUTH/\//:}" -H "Content-Type: application/json" \
  -d '{"statements":[{"statement":"RETURN 1"}]}' \
  "http://127.0.0.1:${HDW_NEO4J_HTTP_PORT:-7474}/db/neo4j/tx/commit" >/dev/null \
  && echo "neo4j tx ok" || echo "neo4j tx unavailable"
WEBUI_HOST="${HDW_WEBUI_BIND:-127.0.0.1}"
[ "$WEBUI_HOST" = "0.0.0.0" ] && WEBUI_HOST="127.0.0.1"
curl -fsS "http://$WEBUI_HOST:${HDW_WEBUI_PORT:-3000}" >/dev/null && echo "webui ok" || echo "webui unavailable"
curl -fsS "http://127.0.0.1:${HDW_LLM_PORT:-8000}/v1/models" >/dev/null && echo "llm ok" || echo "llm unavailable"
