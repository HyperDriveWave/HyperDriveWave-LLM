from __future__ import annotations

import os
from pathlib import Path


class Settings:
    llm_mode = os.getenv("HDW_LLM_MODE", "offline").lower()
    local_llm_base_url = os.getenv(
        "HDW_LOCAL_LLM_BASE_URL",
        os.getenv("HDW_LLM_BASE_URL", "http://localhost:8000/v1"),
    )
    local_llm_model = os.getenv("HDW_LOCAL_LLM_MODEL", "qwen3.8-27b")
    online_llm_base_url = os.getenv(
        "HDW_ONLINE_LLM_BASE_URL",
        "https://api.deepseek.com",
    )
    online_llm_model = os.getenv("HDW_ONLINE_LLM_MODEL", "deepseek-v4-flash")
    online_llm_api_key = os.getenv("HDW_ONLINE_LLM_API_KEY", "")
    llm_timeout = float(os.getenv("HDW_LLM_TIMEOUT", "120"))
    llm_max_tokens = int(os.getenv("HDW_LLM_MAX_TOKENS", "1536"))
    llm_reasoning_effort = os.getenv("HDW_LLM_REASONING_EFFORT", "low")
    online_graph_top_k = int(os.getenv("HDW_ONLINE_GRAPH_TOP_K", "40"))
    local_graph_top_k = int(os.getenv("HDW_LOCAL_GRAPH_TOP_K", "10"))
    context_window_tokens = int(os.getenv("HDW_CONTEXT_WINDOW_TOKENS", "1000000"))
    local_context_window_tokens = int(
        os.getenv("HDW_LOCAL_CONTEXT_WINDOW_TOKENS", "262144")
    )
    context_compression_threshold = float(
        os.getenv("HDW_CONTEXT_COMPRESSION_THRESHOLD", "0.75")
    )
    context_compression_target = float(
        os.getenv("HDW_CONTEXT_COMPRESSION_TARGET", "0.55")
    )
    context_compression_max_tokens = int(
        os.getenv("HDW_CONTEXT_COMPRESSION_MAX_TOKENS", "2000")
    )
    rag_base_url = os.getenv("HDW_RAG_BASE_URL", "http://localhost:8001").strip().rstrip("/")
    rag_remote_urls = tuple(
        url.strip().rstrip("/")
        for url in os.getenv("HDW_RAG_REMOTE_URLS", "").split(",")
        if url.strip()
    )
    rag_timeout = float(os.getenv("HDW_RAG_TIMEOUT", "60"))
    rag_connect_timeout = float(os.getenv("HDW_RAG_CONNECT_TIMEOUT", "3"))
    rag_health_timeout = float(os.getenv("HDW_RAG_HEALTH_TIMEOUT", "5"))
    neo4j_http_url = os.getenv(
        "HDW_NEO4J_HTTP_URL",
        "http://localhost:7474/db/neo4j/tx/commit",
    )
    neo4j_auth = os.getenv("NEO4J_AUTH", "neo4j/change_me")
    graph_timeout = float(os.getenv("HDW_GRAPH_TIMEOUT", "15"))
    mineru_base_url = os.getenv("HDW_MINERU_BASE_URL", "http://localhost:8002")
    # 图片转写用 MinerU 时的超时。**与摄取侧刻意不同**：parse_documents.py 里
    # 明写「MinerU may spend many minutes on large OCR PDFs; do not impose a
    # client deadline」——那是批处理，调用方是任务队列，没人在等。问答路径
    # **有用户在等**，且一次挂起的请求会占住一个 httpx 连接和一个 asyncio 任务。
    # 实测单页卷子图约 17 s，180 s 留了充足余量又不至于挂死。
    mineru_timeout = float(os.getenv("HDW_MINERU_TIMEOUT", "180"))
    # MinerU 是单并发（max_concurrent_requests=1）。队列里已有任务时直接放弃
    # 这个候选，而不是排在知识库入库任务后面等几分钟——失败快、可预期。
    mineru_max_queue = int(os.getenv("HDW_VISION_MINERU_MAX_QUEUE", "1"))
    mcp_base_url = os.getenv("HDW_MCP_BASE_URL", "http://hdw-mcp:8766/mcp").strip().rstrip("/")
    model_config_path = Path(
        os.getenv("HDW_MODEL_CONFIG_PATH", "/data/model-config/config.json")
    )
    maintenance_socket = Path(
        os.getenv(
            "HDW_MAINTENANCE_SOCKET",
            "/run/hdw-maintenance/maintenance.sock",
        )
    )
    maintenance_runtime_path = Path(
        os.getenv(
            "HDW_MAINTENANCE_RUNTIME_PATH",
            "/run/hdw-maintenance/llm-runtime.json",
        )
    )
    frp_public_host = os.getenv("HDW_FRP_PUBLIC_HOST", "").strip()
    internal_api_key = os.getenv("HDW_INTERNAL_API_KEY", "local-dev-key")
    enable_auth = os.getenv("HDW_ENABLE_AUTH", "false").lower() == "true"
    chatdata_root = Path(os.getenv("HDW_CHATDATA_ROOT", "/data/chatdata"))
    auth_csv_path = Path(os.getenv("HDW_AUTH_CSV_PATH", "/app/auth.csv"))
    auth_secret = os.getenv("HDW_AUTH_SECRET", "change_me")

    @property
    def llm_base_url(self) -> str:
        return self.local_llm_base_url

    @property
    def llm_model(self) -> str:
        return self.local_llm_model


settings = Settings()
