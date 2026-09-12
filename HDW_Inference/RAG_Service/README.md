# HDW RAG Service

Minimal HTTP boundary for real embedding, vector search, and reranking.

This service loads local `bge-m3` and `bge-reranker-v2-m3` models, persists
chunk vectors in zvec, and fails explicitly when the real retrieval chain is
unavailable.

Endpoints:

```text
GET  /health
POST /embed
POST /rerank
POST /query-vector
POST /search
POST /admin/reindex
POST /admin/sync
```

Run locally:

```bash
python -m pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8001
```

## Remote Dual-GPU Deployment

`remote-compose.yml` runs two independent copies of this service. Each
container has its own zvec index and is pinned to one physical GPU:

```text
GPU 0 -> http://<remote-host>:8001
GPU 1 -> http://<remote-host>:8003
```

The host running the QA API lists both URLs in `HDW_RAG_REMOTE_URLS`. Requests
are round-robin distributed across the remote nodes. If a remote node fails,
the next remote node is tried; if all remote nodes fail, QA uses the local CPU
RAG at `HDW_RAG_BASE_URL`.

After the local knowledge base changes, run:

```bash
SSHPASS='remote-password' ./Scripts/sync_remote_rag.sh
```

The script copies the current `chunks.jsonl` and rebuilds both remote zvec
indexes. It does not copy model weights again.

`POST /admin/sync` is the automated path used by the ingest API. It accepts a
`chunks.jsonl` multipart upload and requires `X-HDW-RAG-Admin-Token`. The
remote service atomically swaps the chunk file and zvec directory; a failed
rebuild restores the previous remote index.
