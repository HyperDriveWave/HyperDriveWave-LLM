# Industrial QA API

Minimal HyperDriveWave business boundary for industrial question answering.

Current scope:

```text
GET  /health
POST /qa/query
POST /v1/chat/completions
```

`/qa/query` and `/v1/chat/completions` accept `inference_mode` (`online` or
`offline`) and `conversation_id`. The selected provider is:

- `online`: the OpenAI-compatible provider configured by
  `HDW_ONLINE_LLM_*` (currently DeepSeek).
- `offline`: the local llama.cpp endpoint configured by
  `HDW_LOCAL_LLM_*` (currently Qwen3.8).

Conversation JSON files in `HDW_CHATDATA_ROOT` are used as the durable source
of history. At 75% of the configured context estimate, the API summarizes
older turns with the selected provider, retains recent turns, persists the
summary, and then sends the compacted context for the answer.

The first version calls `hdw-rag` and returns extractive answers with citations.
It refuses to invent answers when retrieval misses or services are unavailable.
