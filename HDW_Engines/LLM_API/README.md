# HyperDriveWave LLM API

`LLM_API` is the small provider layer used by `hdw-qa-api`.

## Files

- `client.py`: OpenAI-compatible asynchronous client. It supports the local
  llama.cpp endpoint and online providers such as DeepSeek.
- `context_manager.py`: token estimation and conversation compaction. It
  keeps recent turns, summarizes older turns at the configured threshold, and
  returns the compact context for the next answer.

## Runtime modes

The QA API selects the provider per request:

- `online`: `HDW_ONLINE_LLM_BASE_URL`, `HDW_ONLINE_LLM_MODEL`,
  `HDW_ONLINE_LLM_API_KEY`
- `offline`: `HDW_LOCAL_LLM_BASE_URL`, `HDW_LOCAL_LLM_MODEL`

The current default is controlled by `HDW_LLM_MODE`. The WebUI sends the
selected mode on every question, so switching the slider does not rewrite the
local model configuration.

## Context compaction

The online and offline application context budget is `262,144` estimated
tokens. At 75%, the context manager asks the selected provider to summarize
older turns, keeps recent turns, and persists the summary under the
conversation JSON in `HDW_Runtime/chatdata`. The manager does not truncate the
current question, retrieval evidence, or graph context. The actual provider
or engine still needs to support the configured context.

These values are configurable:

```text
HDW_CONTEXT_WINDOW_TOKENS
HDW_LOCAL_CONTEXT_WINDOW_TOKENS
HDW_CONTEXT_COMPRESSION_THRESHOLD
HDW_CONTEXT_COMPRESSION_TARGET
HDW_CONTEXT_COMPRESSION_MAX_TOKENS
```

The estimate is intentionally conservative because the provider tokenizer is
not shared across local and online models. It is a trigger, not a billing or
provider-limit meter.
