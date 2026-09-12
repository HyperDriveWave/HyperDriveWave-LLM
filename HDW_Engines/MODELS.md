# Model Inventory

## LLM

| Model | Path | Purpose |
| --- | --- | --- |
| Qwen3.8 | `HDW_Engines/LLM_Models/Qwen3.8` | Main local chat/generation model for FreeToken |

The QA API can also use the provider adapter in `HDW_Engines/LLM_API`.
`Configs/.env` currently selects the online DeepSeek provider by default; the
WebUI switch can select the local FreeToken model per request.

## RAG

| Model | Path | Purpose |
| --- | --- | --- |
| bge-m3 | `HDW_Engines/RAG_Models/bge-m3` | Embedding/query vector model |
| bge-reranker-v2-m3 | `HDW_Engines/RAG_Models/bge-reranker-v2-m3` | Candidate reranking model |

## Restore Notes

Model weights are not built into images. Mount `HDW_Engines` read-only as
`/models` and keep source/download/checksum records here as models change.
