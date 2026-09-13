from __future__ import annotations

import base64
import copy
import json
import math
import os
import re
import threading
import asyncio
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from hdw_auth import AuthError, AuthStore
from LLM_API import ContextManager, OpenAICompatibleClient

from .config import settings


app = FastAPI(title="HyperDriveWave Industrial QA API", version="0.1.0")

_CONVERSATION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_USER_CODE_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
# ponytail: single-process global lock; use file locks or a database only if multiple writers become real.
_conversation_lock = threading.Lock()
# ponytail: process-local cursor; use a shared queue only when QA is replicated.
_rag_route_lock = threading.Lock()
_rag_route_cursor = 0
auth_store = AuthStore(settings.auth_csv_path, settings.auth_secret)


class ChatMessage(BaseModel):
    role: str
    content: str


class ImageAttachment(BaseModel):
    name: str = Field(default="image", max_length=240)
    mime_type: str = Field(default="image/jpeg", max_length=64)
    data_url: str = Field(min_length=32)


class QARequest(BaseModel):
    question: str = Field(default="")
    top_k: int = Field(default=40, ge=1, le=40)
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    inference_mode: Literal["online", "offline"] | None = None
    conversation_id: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    images: list[ImageAttachment] = Field(default_factory=list, max_length=4)


class ChatCompletionRequest(BaseModel):
    model: str = "hyperdrivewave-industrial-qa"
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    inference_mode: Literal["online", "offline"] | None = None
    conversation_id: str | None = None
    images: list[ImageAttachment] = Field(default_factory=list, max_length=4)


class LoginRequest(BaseModel):
    code: str = Field(min_length=1, max_length=32)


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    graph_context: list[dict[str, Any]] = Field(default_factory=list)
    emotion_id: str | None = None
    created_at: str | None = None


class ConversationPayload(BaseModel):
    title: str = Field(default="新对话", min_length=1, max_length=200)
    group: Literal["今天", "昨天", "更早"] = "今天"
    pinned: bool = False
    created_at: str | None = None
    updated_at: str | None = None
    messages: list[ConversationMessage] = Field(default_factory=list)
    context_summary: str = ""
    context_kept_from: int = Field(default=0, ge=0)
    context_token_estimate: int = Field(default=0, ge=0)
    context_compressed_at: str | None = None
    last_inference_mode: Literal["online", "offline"] | None = None


class ConversationPatch(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    group: Literal["今天", "昨天", "更早"] | None = None
    pinned: bool | None = None


class ModelConfigPatch(BaseModel):
    local: dict[str, Any] | None = None
    online: dict[str, Any] | None = None
    retrieval: dict[str, Any] | None = None
    vision_priority: list[dict[str, Any]] | None = None


class FrpPatch(BaseModel):
    enabled: bool


_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_MODEL_CONFIG_VERSION = 2


def _default_model_config() -> dict[str, Any]:
    return {
        "version": _MODEL_CONFIG_VERSION,
        "local": {
            "base_url": settings.local_llm_base_url,
            "model": settings.local_llm_model,
            "engine": "llama.cpp",
            "context_window": settings.local_context_window_tokens,
            "multimodal_enabled": True,
            "mtp_enabled": True,
            "thinking_enabled": True,
            "model_options": [
                {
                    "value": settings.local_llm_model,
                    "label": settings.local_llm_model,
                    "available": True,
                },
                {
                    "value": "Qwen3.8-27B-UD-Q5_K_M.gguf",
                    "label": "Qwen3.8-27B-UD-Q5_K_M.gguf",
                    "available": False,
                },
                {
                    "value": "Qwen3.8-27B-FP8",
                    "label": "Qwen3.8-27B-FP8",
                    "available": False,
                },
            ],
            "engine_options": ["llama.cpp", "FreeToken"],
        },
        "online": {
            "base_url": settings.online_llm_base_url,
            "model": settings.online_llm_model,
            "context_window": settings.context_window_tokens,
            "multimodal_enabled": True,
            "thinking_enabled": True,
            "enabled_for_users": True,
            "model_options": [
                {"value": settings.online_llm_model, "label": settings.online_llm_model},
                {
                    "value": "deepseek-flash",
                    "label": "deepseek-flash",
                },
            ],
        },
        "retrieval": {
            "embedding_model": "bge-m3",
            "reranker_model": "bge-reranker-v2-m3",
        },
        "vision_priority": [
            {
                "priority": 1,
                "mode": "online",
                "model": "deepseek-flash",
                "enabled": True,
            },
            {
                "priority": 2,
                "mode": "offline",
                "model": "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf",
                "enabled": True,
            },
        ],
    }


def _merge_model_config(value: dict[str, Any] | None) -> dict[str, Any]:
    merged = copy.deepcopy(_default_model_config())
    if not isinstance(value, dict):
        return merged
    for section in ("local", "online", "retrieval"):
        incoming = value.get(section)
        if isinstance(incoming, dict):
            merged[section].update(incoming)
    priorities = value.get("vision_priority")
    if isinstance(priorities, list):
        merged["vision_priority"] = priorities
    merged["version"] = _MODEL_CONFIG_VERSION
    return merged


def _read_model_config() -> dict[str, Any]:
    path = settings.model_config_path
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return _default_model_config()
    return _merge_model_config(value)


def _safe_model_config(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    # API keys are intentionally never part of this file or response.
    result.pop("api_key", None)
    for section in ("local", "online"):
        result.setdefault(section, {}).pop("api_key", None)
    return result


def _write_model_config(config: dict[str, Any]) -> dict[str, Any]:
    path = settings.model_config_path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_safe_model_config(config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return config


def _validate_model_config_patch(patch: ModelConfigPatch) -> None:
    sections = {
        "local": {
            "base_url",
            "model",
            "engine",
            "context_window",
            "multimodal_enabled",
            "mtp_enabled",
            "thinking_enabled",
        },
        "online": {
            "base_url",
            "model",
            "context_window",
            "multimodal_enabled",
            "thinking_enabled",
            "enabled_for_users",
        },
        "retrieval": {"embedding_model", "reranker_model"},
    }
    values = patch.model_dump(exclude_none=True)
    for section, allowed in sections.items():
        incoming = values.get(section)
        if incoming is None:
            continue
        unknown = set(incoming) - allowed
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"unsupported {section} config field(s): {', '.join(sorted(unknown))}",
            )
        for key, value in incoming.items():
            if key == "base_url":
                parsed = urlparse(str(value))
                if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                    raise HTTPException(status_code=422, detail=f"invalid {section}.base_url")
            elif key in {"model", "engine", "embedding_model", "reranker_model"}:
                if not str(value).strip() or len(str(value)) > 240:
                    raise HTTPException(status_code=422, detail=f"invalid {section}.{key}")
            elif key == "context_window":
                if not isinstance(value, int) or not 4096 <= value <= 2_000_000:
                    raise HTTPException(status_code=422, detail=f"invalid {section}.context_window")
            elif key in {
                "multimodal_enabled",
                "mtp_enabled",
                "thinking_enabled",
                "enabled_for_users",
            } and not isinstance(value, bool):
                raise HTTPException(status_code=422, detail=f"invalid {section}.{key}")

    priorities = values.get("vision_priority")
    if priorities is not None:
        if not 1 <= len(priorities) <= 4:
            raise HTTPException(status_code=422, detail="vision_priority must contain 1 to 4 candidates")
        for index, candidate in enumerate(priorities, 1):
            if candidate.get("mode") not in {"online", "offline"}:
                raise HTTPException(status_code=422, detail=f"vision_priority[{index}].mode is invalid")
            if not str(candidate.get("model") or "").strip():
                raise HTTPException(status_code=422, detail=f"vision_priority[{index}].model is required")
            if "enabled" in candidate and not isinstance(candidate["enabled"], bool):
                raise HTTPException(status_code=422, detail=f"vision_priority[{index}].enabled is invalid")


def _effective_profile(mode: Literal["online", "offline"]) -> dict[str, Any]:
    config = _read_model_config()
    section = config["online" if mode == "online" else "local"]
    return {
        "base_url": str(section.get("base_url") or (
            settings.online_llm_base_url if mode == "online" else settings.local_llm_base_url
        )).rstrip("/"),
        "model": str(section.get("model") or (
            settings.online_llm_model if mode == "online" else settings.local_llm_model
        )),
        "context_window": int(section.get("context_window") or _context_window_from_settings(mode)),
    }


def _context_window_from_settings(mode: Literal["online", "offline"]) -> int:
    return settings.context_window_tokens if mode == "online" else settings.local_context_window_tokens


async def _local_runtime_info(base_url: str) -> dict[str, Any]:
    endpoint = base_url.rstrip("/")
    if endpoint.endswith("/v1"):
        endpoint = endpoint[:-3].rstrip("/")
    engine = "llama.cpp"
    try:
        marker = json.loads(
            settings.maintenance_runtime_path.read_text(encoding="utf-8")
        )
        if isinstance(marker, dict) and marker.get("engine"):
            engine = str(marker["engine"])
    except (OSError, json.JSONDecodeError):
        pass
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{endpoint}/props")
        response.raise_for_status()
        payload = response.json()
        defaults = payload.get("default_generation_settings") or {}
        context_window = defaults.get("n_ctx") or payload.get("n_ctx") or 0
        loaded_model = str(
            payload.get("model_alias")
            or Path(str(payload.get("model_path") or "")).name
        ).strip()
        vision = bool((payload.get("modalities") or {}).get("vision"))
        detail = (
            f"{engine} /props reports vision={'true' if vision else 'false'}"
            f"; loaded_model={loaded_model or 'unknown'}"
            f"; context={int(context_window) if context_window else 'unknown'}"
        )
        return {
            "available": True,
            "engine": engine,
            "model": loaded_model,
            "context_window": int(context_window) if context_window else 0,
            "vision": vision,
            "detail": detail,
        }
    except Exception as props_error:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{endpoint}/v1/models")
            response.raise_for_status()
            models = response.json().get("data") or []
            item = models[0] if isinstance(models, list) and models else {}
            loaded_model = str(item.get("id") or "").strip()
            context_window = int(
                item.get("context_length")
                or item.get("max_model_len")
                or 0
            )
            engine = "FreeToken" if engine == "llama.cpp" else engine
            return {
                "available": True,
                "engine": engine,
                "model": loaded_model,
                "context_window": context_window,
                "vision": False,
                "detail": (
                    f"{engine} /v1/models reports vision=false"
                    f"; loaded_model={loaded_model or 'unknown'}"
                    f"; context={context_window or 'unknown'}"
                ),
            }
        except Exception as models_error:
            return {
                "available": False,
                "engine": "",
                "model": "",
                "context_window": 0,
                "vision": False,
                "detail": (
                    f"local runtime probe failed: /props={props_error}; "
                    f"/v1/models={models_error}"
                ),
            }


async def _maintenance_request(path: str, method: str = "POST") -> dict[str, Any]:
    if not settings.maintenance_socket.exists():
        raise RuntimeError(f"资源协调器 socket 不存在：{settings.maintenance_socket}")
    transport = httpx.AsyncHTTPTransport(uds=str(settings.maintenance_socket))
    async with httpx.AsyncClient(transport=transport, timeout=None) as client:
        response = await client.request(method, f"http://localhost{path}")
    try:
        payload = response.json()
    except ValueError:
        payload = {"detail": response.text}
    if response.status_code >= 400:
        detail = payload.get("error") or payload.get("detail") or response.text
        raise RuntimeError(f"资源协调器 HTTP {response.status_code}: {detail}")
    if not isinstance(payload, dict):
        raise RuntimeError("资源协调器返回格式错误")
    return payload


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _current_user(x_hdw_session: str | None) -> dict[str, str]:
    token = str(x_hdw_session or "").strip()
    if not token:
        raise HTTPException(status_code=401, detail="login required")
    try:
        return auth_store.verify(token)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail="invalid or expired session") from exc


def _user_chatdata_root(user_code: str) -> Path:
    if not _USER_CODE_RE.fullmatch(user_code):
        raise HTTPException(status_code=400, detail="invalid user code")
    return settings.chatdata_root / user_code


def _conversation_path(user_code: str, conversation_id: str) -> Path:
    if not _CONVERSATION_ID_RE.fullmatch(conversation_id):
        raise HTTPException(status_code=400, detail="invalid conversation id")
    return _user_chatdata_root(user_code) / f"{conversation_id}.json"


def _ensure_chatdata_root(user_code: str) -> None:
    try:
        _user_chatdata_root(user_code).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(status_code=503, detail=f"chatdata unavailable: {exc}") from exc


def _normalize_conversation(conversation_id: str, data: dict[str, Any]) -> dict[str, Any]:
    conversation = dict(data)
    conversation["id"] = conversation_id
    conversation.setdefault("title", "新对话")
    conversation.setdefault("group", "更早")
    conversation.setdefault("pinned", False)
    conversation.setdefault("created_at", None)
    conversation.setdefault("updated_at", conversation.get("created_at"))
    conversation.setdefault("messages", [])
    conversation.setdefault("context_summary", "")
    conversation.setdefault("context_kept_from", 0)
    conversation.setdefault("context_token_estimate", 0)
    conversation.setdefault("context_compressed_at", None)
    if conversation.get("last_inference_mode") not in {"online", "offline"}:
        conversation["last_inference_mode"] = None
    return conversation


def _read_conversation(user_code: str, conversation_id: str) -> dict[str, Any]:
    path = _conversation_path(user_code, conversation_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="conversation not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"conversation file is invalid: {exc}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="conversation file must contain an object")
    return _normalize_conversation(conversation_id, data)


def _write_conversation(user_code: str, conversation_id: str, conversation: dict[str, Any]) -> dict[str, Any]:
    _ensure_chatdata_root(user_code)
    path = _conversation_path(user_code, conversation_id)
    payload = _normalize_conversation(conversation_id, conversation)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError as exc:
        raise HTTPException(status_code=503, detail=f"conversation write failed: {exc}") from exc
    return payload


def _conversation_summary(conversation_id: str, data: dict[str, Any]) -> dict[str, Any]:
    conversation = _normalize_conversation(conversation_id, data)
    return {
        "id": conversation_id,
        "title": conversation["title"],
        "group": conversation["group"],
        "pinned": bool(conversation["pinned"]),
        "created_at": conversation["created_at"],
        "updated_at": conversation["updated_at"],
        "message_count": len(conversation["messages"]),
    }


def _check_auth(authorization: str | None) -> None:
    if not settings.enable_auth:
        return
    if authorization != f"Bearer {settings.internal_api_key}":
        raise HTTPException(status_code=401, detail="unauthorized")


def _last_user_message(messages: list[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    return messages[-1].content


def _resolve_inference_mode(mode: str | None) -> Literal["online", "offline"]:
    selected = (mode or settings.llm_mode or "offline").lower()
    if selected not in {"online", "offline"}:
        raise HTTPException(status_code=400, detail="inference_mode must be online or offline")
    return selected  # type: ignore[return-value]


def _user_default_inference_mode(user: dict[str, str]) -> Literal["online", "offline"]:
    selected = str(user.get("default_inference_mode") or "offline").lower()
    if selected not in {"online", "offline"}:
        return "offline"
    return selected  # type: ignore[return-value]


def _thinking_enabled(mode: Literal["online", "offline"]) -> bool:
    section = _read_model_config().get("online" if mode == "online" else "local", {})
    value = section.get("thinking_enabled", True) if isinstance(section, dict) else True
    return value if isinstance(value, bool) else True


def _online_inference_enabled_for_users() -> bool:
    online = _read_model_config().get("online", {})
    value = online.get("enabled_for_users", True) if isinstance(online, dict) else True
    return value if isinstance(value, bool) else True


def _online_inference_allowed(user: dict[str, str]) -> bool:
    return user.get("role") == "admin" or _online_inference_enabled_for_users()


def _permissions_for_user(user: dict[str, str]) -> dict[str, bool]:
    return {
        "model_management": user.get("role") == "admin",
        "frp_management": user.get("role") == "admin",
        "online_inference": _online_inference_allowed(user),
        "online_inference_for_users": _online_inference_enabled_for_users(),
    }


def _llm_client(
    mode: str | None = None,
    *,
    model_override: str | None = None,
    base_url_override: str | None = None,
) -> tuple[Literal["online", "offline"], OpenAICompatibleClient]:
    selected = _resolve_inference_mode(mode)
    profile = _effective_profile(selected)
    if selected == "online":
        if not settings.online_llm_api_key:
            raise HTTPException(status_code=503, detail="online LLM API key is not configured")
        return selected, OpenAICompatibleClient(
            base_url_override or profile["base_url"],
            model_override or profile["model"],
            api_key=settings.online_llm_api_key,
            timeout=settings.llm_timeout,
        )
    return selected, OpenAICompatibleClient(
        base_url_override or profile["base_url"],
        model_override or profile["model"],
        timeout=None,
    )


def _context_window(mode: Literal["online", "offline"]) -> int:
    return _effective_profile(mode)["context_window"]


def _validate_image_attachments(images: list[ImageAttachment]) -> list[dict[str, str]]:
    validated: list[dict[str, str]] = []
    for image in images:
        mime_type = image.mime_type.strip().lower()
        if mime_type not in _IMAGE_MIME_TYPES:
            raise HTTPException(status_code=415, detail=f"unsupported image type: {mime_type}")
        prefix = f"data:{mime_type};base64,"
        if not image.data_url.startswith(prefix):
            raise HTTPException(status_code=422, detail="image data_url must be a base64 data URL")
        encoded = image.data_url[len(prefix):].strip()
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise HTTPException(status_code=422, detail="image data_url is invalid base64") from exc
        if not decoded or len(decoded) > _MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="image is empty or larger than 10 MB")
        validated.append({
            "name": image.name,
            "mime_type": mime_type,
            "data_url": f"{prefix}{encoded}",
        })
    return validated


async def _mcp_call_tool(name: str, arguments: dict[str, Any]) -> Any:
    request_timeout = httpx.Timeout(settings.rag_connect_timeout + 15, connect=settings.rag_connect_timeout)
    async with httpx.AsyncClient(timeout=request_timeout) as client:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        request_id = 1
        response = await client.post(
            settings.mcp_base_url,
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "HyperDriveWave QA", "version": "1.0"},
                },
            },
        )
        response.raise_for_status()
        session_id = response.headers.get("mcp-session-id")
        if not session_id:
            raise RuntimeError("MCP initialize did not return a session id")
        response = await client.post(
            settings.mcp_base_url,
            headers={**headers, "Mcp-Session-Id": session_id},
            json={
                "jsonrpc": "2.0",
                "id": request_id + 1,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
        )
        response.raise_for_status()
        payload = response.json()
    if payload.get("error"):
        raise RuntimeError(payload["error"].get("message") or "MCP tool call failed")
    result = payload.get("result") or {}
    if isinstance(result.get("structuredContent"), dict):
        return result["structuredContent"]
    text = next(
        (
            item.get("text")
            for item in result.get("content", [])
            if isinstance(item, dict) and item.get("type") == "text"
        ),
        "",
    )
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"message": text}



_PLANNER_PROMPT = (
    "你是检索规划器。今天是 {today}（北京时间）。\n"
    "针对用户问题输出五行，不要解释、不要多余文字。用不到的写「无」：\n"
    "检索: yes 或 no   —— 是否需要查工业文档知识库（规程、参数、故障、操作、标准）\n"
    "测点: <关键词>[、<关键词>] 或 无   —— 是否需要读测点**当前值**\n"
    "历史: <关键词>|<起始日期>|<结束日期> 或 无   —— 是否需要测点**历史序列**\n"
    "趋势: <关键词>|<起始日期>|<结束日期>|<采样间隔秒> 或 无   —— 是否需要测点**变化趋势**\n"
    "日志: <起始日期>|<结束日期>[|重大] 或 无   —— 是否需要查**运行日志**\n"
    "\n"
    "判断依据（按语意，不要按关键词硬匹配）：\n"
    "· 「现在多少」「当前值」→ 测点；「昨天/过去三天/这段时间的变化」→ 历史；\n"
    "  「趋势」「走势」「一直涨吗」「画个曲线」→ 趋势；\n"
    "  「有什么操作」「值班日志」「重大事件」「交接班记录」→ 日志。\n"
    "· 一个问句可以同时命中多项（例如「看看汽包水位的趋势，再结合规程说说」）。\n"
    "· 只要当前值就不要填历史或趋势，它们会各自触发一次数据查询。\n"
    "· 日期一律写成 YYYY-MM-DD，相对时间按上面的今天换算；不确定结束日期就用今天。\n"
    "· 趋势的采样间隔按问题的时间跨度选：一天用 300，三天用 900，一周用 1800。\n"
    "· 日志只在对方明确问运行记录时填；问规程、原理、参数含义时填「无」。\n"
    "· 只说「重大」「重要」时在日志第三段写「重大」。\n"
    # 关键词要短，但**不能丢掉区分性限定词**。
    # 踩过的坑：早先写的是「只描述物理量本身，不要带修饰语」，模型据此把
    # 「轴封供气压力」简化成「轴封压力」——而「供气」不是修饰语，是区分
    # 「轴封供气压力」和「低压轴封压力」两个不同测点的关键。简化后检索
    # 只匹配到后者，而且是唯一命中，相关性判定也拦不住。
    "· 测点关键词控制在 2-8 个字：去掉机组号（一号机／#1）和设备全称，\n"
    "  但**必须保留区分性限定词**——「供气／供汽」「高压／低压」「A 侧／B 侧」\n"
    "  「给水／凝结水」「进口／出口」这类词决定了是哪一个测点，去掉就串号了。\n"
    "· 测点最多 3 个，其余各项最多 1 个。\n"
    "示例：\n"
    "  问：一号机凝汽器水位多少，然后结合知识库回答\n"
    "  检索: yes\n"
    "  测点: 凝汽器液位\n"
    "  历史: 无\n"
    "  趋势: 无\n"
    "  日志: 无\n"
    "  问：轴封系统的作用\n"
    "  检索: yes\n"
    "  测点: 无\n"
    "  历史: 无\n"
    "  趋势: 无\n"
    "  日志: 无\n"
    "  问：看看今天汽包水位的趋势，结合规程说说有没有问题\n"
    "  检索: yes\n"
    "  测点: 无\n"
    "  历史: 无\n"
    "  趋势: 汽包水位|{today}|{today}|300\n"
    "  日志: 无\n"
    "  问：昨天有什么重大操作吗\n"
    "  检索: no\n"
    "  测点: 无\n"
    "  历史: 无\n"
    "  趋势: 无\n"
    "  日志: {yesterday}|{yesterday}|重大\n"
)
# 规划开关：模型判错时可不重建镜像直接关掉，退回「一律检索、不读测点」
_ROUTER_ENABLED = os.getenv("HDW_RETRIEVAL_ROUTER", "true").lower() != "false"
_LIVE_POINTS_MAX = int(os.getenv("HDW_LIVE_POINTS_MAX", "3"))
# 关键词匹配偏了时，最多回退试几个候选测点
_LIVE_POINT_CANDIDATES = int(os.getenv("HDW_LIVE_POINT_CANDIDATES", "5"))
# 日志注入上限：一次抓取实测 118 条，全塞进上下文会挤占文档证据。
# 按严重程度排序后取前 N 条，重大事件优先。
_LOGS_MAX = int(os.getenv("HDW_LOGS_MAX", "40"))
# 单次查询的时间跨度上限，防止规划器给出离谱区间导致 LIEMS 端长时间抓取
_QUERY_MAX_DAYS = int(os.getenv("HDW_QUERY_MAX_DAYS", "7"))
# 历史/趋势注入上下文时的压缩点数上限。实测单次可达 11063 点，
# 全量下发会挤占文档证据；压缩时优先保留关键形状点。
_SERIES_MAX_POINTS = int(os.getenv("HDW_SERIES_MAX_POINTS", "24"))
# ── 趋势特征提取阈值，参照 SmartGasTurbine 的 Trend_Prior 配置 ──
# 斜率绝对值低于此值视为「基本平稳」（单位/分钟）
_TREND_FLAT_SLOPE = float(os.getenv("HDW_TREND_FLAT_SLOPE", "0.01"))
# 整体 r² 低于此值才值得报拐点：单条直线已能代表整段时，报拐点只是噪声
_TREND_TURNING_R2 = float(os.getenv("HDW_TREND_TURNING_R2", "0.6"))
# 导数死区：变化率低于此值视为平直（单位/秒），对应 Trend_Prior 的 0.0001
_TREND_DEADBAND = float(os.getenv("HDW_TREND_DEADBAND", "0.0001"))
# 拐点确认延迟（秒）：距序列末尾太近的「拐」多半是噪声
_TREND_TURNING_DELAY = float(os.getenv("HDW_TREND_TURNING_DELAY", "10"))
# 拐点后回归所需的最少点数，对应 Trend_Prior 的 short_min_points=6
_TREND_TURNING_MIN_POINTS = int(os.getenv("HDW_TREND_TURNING_MIN_POINTS", "6"))
# 拐点前后两段的电平差要达到全程量程的这个比例才算「真事件」。
# 实测经验：轴封供气压力从 0.018 掉到 0 这种跃迁，电平差占比接近 1；
# 而噪声抖动只有千分之几。取 0.15 能把两者分开。
_TREND_LEVEL_RATIO = float(os.getenv("HDW_TREND_LEVEL_RATIO", "0.15"))
# 最多报几个拐点。按电平变化量排序取前几个，位置先后不重要。
_TREND_TURNING_MAX = int(os.getenv("HDW_TREND_TURNING_MAX", "3"))
# 相对变化率下限（%）。**两个尺度都要达标才算突变**：
# 只看向对量程占比，会把「0.0181727→0.01829」这种 0.6% 的微动当成事件
# ——它占比高只是因为当时量程本身就极小。取 5% 能滤掉这类抖动，
# 同时保留「压力从有到无」(100%) 和 V 形转折(约 9%)。
_TREND_LEVEL_PERCENT = float(os.getenv("HDW_TREND_LEVEL_PERCENT", "5.0"))
# ── 整段显著性判据。趋势分析用，**宁可多报**：任一判据超标就展开分析 ──
# ── 整段波动的三级判定（相对读数口径）──
#   < 1%      完全隐掉：只回一句「基本无变动」
#   1% ~ 3%   只给曲线图 + 均值/极值，不给斜率、突变点、关键点
#   ≥ 3%      完整展开
# 中间档是为实测的轴封供气压力（全天波动占读数 2.8%）设的：
# 它既不该像真变化那样被逐点解读，也不该像完全静止那样一言蔽之。
_FLAT_LEVEL_MINOR = float(os.getenv("HDW_FLAT_LEVEL_MINOR", "0.01"))
_FLAT_LEVEL_SIGNIF = float(os.getenv("HDW_FLAT_LEVEL_SIGNIF", "0.03"))
# 离散跳变判据：不同取值数 / 样本数 低于此比例时，判为采集噪声（丢包或取整）。
# 实测轴封供气压力全天只在两个固定值间跳，distinct/总 = 2/88 ≈ 2%。
_DISCRETE_LEVEL_RATIO = float(os.getenv("HDW_DISCRETE_LEVEL_RATIO", "0.05"))
# ── 当前值对比。**宁可少报**：两个判据都超标才提示偏离 ──
# 当前值是随口一问，天天提示「偏离 0.5%」只会让提示失去意义。
_DRIFT_REL_PERCENT = float(os.getenv("HDW_DRIFT_REL_PERCENT", "5.0"))
# 偏离要达到几个标准差才算异常。低于 2σ 基本都在正常波动内。
_DRIFT_SIGMA = float(os.getenv("HDW_DRIFT_SIGMA", "2.0"))
# 当前值对比取多长时间的历史做基准
_DRIFT_WINDOW_MINUTES = int(os.getenv("HDW_DRIFT_WINDOW_MINUTES", "60"))
# 曲线图输出尺寸与 DPI（前端按宽度自适应，这里只定比例）
_CHART_FIGSIZE = (7.2, 2.6)
_CHART_DPI = 130
# SIS 返回的是 UTC；现场看的是本地时间，统一按 UTC+8 展示且不带时区后缀
_LIVE_TIME_OFFSET_HOURS = float(os.getenv("HDW_LIVE_TIME_OFFSET_HOURS", "8"))


def _format_live_time(value: Any) -> str:
    """ISO-8601（可带时区）→ 本地 `YYYY-MM-DD HH:MM:SS`，无时区后缀。

    解析不了就原样返回：宁可在界面上显示得难看，也不要丢掉采集时间。
    """
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    # 不带时区的按 UTC 解释——SIS 侧就是 UTC，不能当本地时间用（会多算 8 小时）
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    local = moment.astimezone(timezone(timedelta(hours=_LIVE_TIME_OFFSET_HOURS)))
    return local.strftime("%Y-%m-%d %H:%M:%S")


# 规划器输出里的「空值」写法。模型有时会写 none/null/- 而不是「无」。
_PLAN_EMPTY = ("无", "none", "null", "-", "n/a", "")


def _plan_field(line: str, label: str) -> str | None:
    """取 `标签: 值` 里的值；该行不存在返回 None，值为空返回 ""。"""
    match = re.match(rf"^{label}\s*[:：]\s*(.*)$", line)
    return match.group(1).strip() if match else None


def _plan_date(text: str) -> str:
    """把模型给的日期规整成 YYYY-MM-DD；解析不了返回空串。"""
    text = (text or "").strip()
    if not text:
        return ""
    match = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", text)
    if not match:
        return ""
    year, month, day = (int(part) for part in match.groups())
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return ""


def _clamp_range(start: str, end: str) -> tuple[str, str]:
    """把时间区间收进允许范围，并保证 start <= end。

    规划器是模型，它给出的日期不能直接信：跨度太大时抓取会长时间占用
    LIEMS 连接（该模块文档明确要求避免触发服务端异常访问判定）。
    """
    today = _local_today()
    end_date = date.fromisoformat(end) if end else today
    start_date = date.fromisoformat(start) if start else end_date
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    if end_date > today:
        end_date = today
    if (end_date - start_date).days >= _QUERY_MAX_DAYS:
        start_date = end_date - timedelta(days=_QUERY_MAX_DAYS - 1)
    return start_date.isoformat(), end_date.isoformat()


def _local_today() -> date:
    return (datetime.now(timezone.utc) + timedelta(hours=_LIVE_TIME_OFFSET_HOURS)).date()


def _render_trend_png(item: dict[str, Any], features: dict[str, Any]) -> str:
    """把趋势渲染成 PNG，返回 base64。失败返回空串——**出图失败不能影响回答**。

    样式对齐 SmartGasTurbine 前端（echarts 配置）：
      · 折线 smooth、不显示数据点标记（点数多时标记会糊成一片）
      · 只保留左/下轴线，去掉上/右框
      · 网格浅色虚线
    配色取它 `styles/main.css` 的变量：主色 #1769aa、警示 #b54708、危险 #b42318。
    """
    try:
        import base64
        import io

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 中文字体：镜像里装了 fonts-noto-cjk。缺字体时图上中文会变方框，
        # 这时退化成不写中文，也不至于出一张看不懂的图。
        from matplotlib import font_manager

        available = {f.name for f in font_manager.fontManager.ttflist}
        cjk = next(
            (n for n in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei") if n in available),
            None,
        )
        if cjk:
            plt.rcParams["font.sans-serif"] = [cjk, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        samples = item.get("samples") or []
        points = []
        for sample in samples:
            value = sample.get("value")
            if value is None:
                continue
            try:
                moment = datetime.fromisoformat(str(sample.get("time", "")).replace("Z", "+00:00"))
            except ValueError:
                continue
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            points.append(
                (moment.astimezone(timezone(timedelta(hours=_LIVE_TIME_OFFSET_HOURS))), float(value))
            )
        if len(points) < 2:
            return ""

        xs = [moment for moment, _ in points]
        ys = [value for _, value in points]
        unit = str(item.get("unit") or "")
        title = str(item.get("description") or item.get("query") or "")

        fig, ax = plt.subplots(figsize=_CHART_FIGSIZE, dpi=_CHART_DPI)
        ax.plot(xs, ys, color="#1769aa", linewidth=1.4, solid_capstyle="round")
        ax.set_facecolor("#ffffff")
        fig.patch.set_facecolor("#ffffff")

        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d7dde5")
            ax.spines[side].set_linewidth(0.8)
        ax.grid(True, color="#eef1f5", linewidth=0.7, linestyle="--")
        ax.tick_params(colors="#667085", labelsize=8, length=3, width=0.8)

        # 极值点单独标注：图上最该被一眼看到的就是「最低/最高出现在哪」
        if ys:
            for idx, color, label in (
                (ys.index(min(ys)), "#b54708", "min"),
                (ys.index(max(ys)), "#b42318", "max"),
            ):
                ax.scatter([xs[idx]], [ys[idx]], s=16, color=color, zorder=3)
                ax.annotate(
                    f"{ys[idx]:.6g}",
                    (xs[idx], ys[idx]),
                    textcoords="offset points",
                    xytext=(0, 6 if label == "max" else -12),
                    ha="center",
                    fontsize=7.5,
                    color=color,
                )

        ax.set_ylabel(f"{title} / {unit}" if unit else title, fontsize=8.5, color="#17202a")
        fig.autofmt_xdate(rotation=0, ha="center")
        fig.tight_layout(pad=0.6)

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", facecolor="#ffffff")
        plt.close(fig)
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        # 出图是锦上添花，任何异常都不该让整条问答失败
        return ""


def _percentile(values: list[float], ratio: float) -> float:
    """线性插值分位数。标准库够用，不必为这一个函数引入 numpy。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, ratio)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _series_significance(
    values: list[float], limits: dict[str, float] | None = None
) -> dict[str, Any]:
    """判断整段序列是否有「显著变化」。**趋势分析用，宁可多报。**

    两路判据，任一超标就判为「有变化」：

      判据A 相对读数 = (P95-P5) / max(|P5|, |P95|)
            无需任何配置，现在就能用。基准取分位数而不是均值：序列趋零或
            含尖峰时均值会被拉偏。

      判据B 工程量程 = (P95-P5) / (high_limit - low_limit)
            有测点上下限时最符合工程直觉——「占了量程的多少」。
            **没有量程时该路跳过**，不参与判定。

    **为什么用 P95-P5 而不是 max-min**：单个毛刺能把极差撑大好几倍，
    而分位数描述的是「大部分时间在什么范围」。实测轴封供气压力
    全天在 0.017842~0.018338 之间，占了它自身量程的 100%，
    但只占读数的 2.8%——只看前者，任何抖动都会「占量程 100%」。
    """
    usable = [float(v) for v in values if v is not None]
    if len(usable) < 3:
        # level = -1 表示「判不了」，调用方按「完整展开」的保守方向处理
        return {"comparable": False, "level": -1, "flat": False, "reason": "样本不足"}

    p5 = _percentile(usable, 0.05)
    p95 = _percentile(usable, 0.95)
    spread = p95 - p5

    base_reading = max(abs(p5), abs(p95))
    rel_reading = spread / base_reading if base_reading > 1e-12 else float("inf")

    rel_range: float | None = None
    if limits:
        high, low = limits.get("high"), limits.get("low")
        if high is not None and low is not None and abs(high - low) > 1e-12:
            rel_range = spread / abs(high - low)

    # 三级判定，而不是简单二值。实测轴封供气压力全天波动占读数 2.8%——
    # 这个量级既不该像真变化那样展开斜率与突变点，也不该像完全无变化那样一言蔽之，
    # 它需要的是「图给你，数给你，但别去解读」。
    def level_of(ratio: float | None) -> int:
        if ratio is None:
            return -1
        if ratio < _FLAT_LEVEL_MINOR:
            return 0          # 完全隐掉
        if ratio < _FLAT_LEVEL_SIGNIF:
            return 1          # 只给图与统计
        return 2              # 完整展开

    levels = [level_of(rel_reading)]
    if rel_range is not None:
        levels.append(level_of(rel_range))
    # 两路取**更高**等级（更敏感的那路说了算）——趋势场景宁可多报
    level = max(levels)

    # 离散跳变检测：取值只集中在少数几个固定值上，是采集丢包/取整的典型特征，
    # 而不是物理量的真实变化。图上看得最清楚（两个固定值之间规律跳动），
    # 但纯看统计量会被当成「有波动」。
    distinct = len({round(v, 9) for v in usable})
    discrete = len(usable) >= 10 and distinct / len(usable) < _DISCRETE_LEVEL_RATIO

    return {
        "comparable": True,
        "level": level,
        "flat": level == 0,
        "p5": p5,
        "p95": p95,
        "median": _percentile(usable, 0.5),
        "spread": spread,
        "rel_reading": rel_reading,
        "rel_range": rel_range,
        "samples": len(usable),
        "distinct_values": distinct,
        "discrete": discrete,
    }


def _series_features(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """从采样序列里提取趋势特征。**这是给模型看的核心信息，比原始点更重要。**

    做法参照 SmartGasTurbine 的 Trend_Prior：逐点有限差分求导，用
    「导数过零（极值）或落入死区（平直）」定位拐点，再对区间做最小二乘
    线性回归给出斜率与决定系数 r²。

    为什么不用等距抽稀：工业趋势大量是「长期平直 + 突然跳变」，
    等距取样会把跳变整个漏掉，而那恰恰是最该被看到的地方。特征提取不丢这个信息。

    全部用标准库实现，不引入 numpy —— qa-api 的运行环境里没有它。
    """
    points: list[tuple[float, float]] = []   # (epoch 秒, 值)
    for item in samples:
        value = item.get("value")
        if value is None:
            continue
        try:
            moment = datetime.fromisoformat(str(item.get("time", "")).replace("Z", "+00:00"))
        except ValueError:
            continue
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        points.append((moment.timestamp(), float(value)))
    if len(points) < 3:
        return {}

    features: dict[str, Any] = {"samples": len(points)}

    # ── 整体线性回归：斜率 + r² ──
    # r² 是判断「这个趋势能不能用一条直线代表」的关键：r² 低说明曲线在转折，
    # 单看斜率会得出错误结论（先降后升的曲线整体斜率可能接近 0）。
    def regress(seq: list[tuple[float, float]]) -> dict[str, float]:
        base = seq[0][0]
        xs = [moment - base for moment, _ in seq]
        ys = [value for _, value in seq]
        n = len(xs)
        mean_x, mean_y = sum(xs) / n, sum(ys) / n
        den = sum((x - mean_x) ** 2 for x in xs)
        if den <= 0:
            return {"slope_per_minute": 0.0, "r2": 0.0}
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / den
        intercept = mean_y - slope * mean_x
        ss_tot = sum((y - mean_y) ** 2 for y in ys)
        ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
        r2 = 1.0 if ss_tot <= 0 else max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
        return {"slope_per_minute": slope * 60.0, "r2": r2}

    overall = regress(points)
    features["slope_per_minute"] = overall["slope_per_minute"]
    features["r2"] = overall["r2"]
    if abs(overall["slope_per_minute"]) < _TREND_FLAT_SLOPE:
        features["direction"] = "基本平稳"
    else:
        features["direction"] = "上升" if overall["slope_per_minute"] > 0 else "下降"

    # 极值要先算出来：下面挑突变点要用它作为「全程量程」的基准
    extremes = [value for _, value in points]
    features["min"] = min(extremes)
    features["max"] = max(extremes)
    features["latest"] = extremes[-1]
    features["delta"] = extremes[-1] - extremes[0]

    # ── 逐点导数，找拐点与最大变化率 ──
    derivatives: list[float] = []
    for index in range(1, len(points)):
        seconds = max(1e-9, points[index][0] - points[index - 1][0])
        derivatives.append((points[index][1] - points[index - 1][1]) / seconds)
    if derivatives:
        peak_index = max(range(len(derivatives)), key=lambda i: abs(derivatives[i]))
        features["max_rate"] = {
            "time": _format_live_time(
                datetime.fromtimestamp(points[peak_index][0], tz=timezone.utc).isoformat()
            ),
            "per_second": derivatives[peak_index],
            "per_minute": derivatives[peak_index] * 60.0,
        }
        # 拐点按**电平变化量**挑，不按「是不是最后一个」挑。
        #
        # 踩过的坑：先前只报最后一个拐点，并用拐点后窗口的 r² 作为「可信度」给模型。
        # 但压力从有到无之后必然是一段平直，r² 因此接近 0——这个 0 说明的是
        # **之后是平的**，恰恰是发生了状态跃迁的证据，而不是拐点不可信。
        # 模型照这个数字把「压力消失」判成了噪声，属于我给错了判据。
        #
        # 真正该看的是拐点**前后两段的电平差**：差得越多，越是一次真实的状态变化。
        span = max(features["max"] - features["min"], 1e-12)
        turns = []
        for index in range(1, len(derivatives)):
            prev, curr = derivatives[index - 1], derivatives[index]
            crosses = (prev > 0 >= curr) or (prev < 0 <= curr)
            if not crosses and abs(curr) > _TREND_DEADBAND:
                continue
            moment = points[index + 1][0]
            # 延迟确认：拐点距序列末尾太近时，所谓「拐」可能只是噪声在收尾。
            # **这是时间判据，不是窗口宽度** —— 曾把它误当前后段的窗口宽度用，
            # 采样间隔 60 s 而窗口只有 10 s，前后段永远取不到点，真实的 V 形转折
            # 全被判成噪声。
            if points[-1][0] - moment < _TREND_TURNING_DELAY:
                continue

            # 切点精确定位：导数归零的那一步会把「最后一个旧状态的点」也算进后段。
            # 实测「0.018×30 后接 0×30」时，after_mean 是 0.00058 而不是 0——
            # 正是混进了一个 0.018。改为在候选点邻域内取导数绝对值最大的那一步之后切。
            cut = index + 1
            best = abs(derivatives[index])
            for probe in (index - 1, index + 1):
                if 0 <= probe < len(derivatives) and abs(derivatives[probe]) > best:
                    best = abs(derivatives[probe])
                    cut = probe + 1
            before_slice = points[max(0, cut - _TREND_TURNING_MIN_POINTS):cut]
            after_slice = points[cut:]
            if len(before_slice) < _TREND_TURNING_MIN_POINTS or len(after_slice) < _TREND_TURNING_MIN_POINTS:
                continue
            before_mean = sum(v for _, v in before_slice) / len(before_slice)
            after_mean = sum(v for _, v in after_slice) / len(after_slice)
            level_delta = after_mean - before_mean
            # 两个尺度都达标才算突变。只看占比会放过「量程极小的高占比微动」，
            # 只看百分比会放过「量程极大时的小百分比跃迁」——两者互补。
            if abs(level_delta) < span * _TREND_LEVEL_RATIO:
                continue
            if abs(before_mean) > 1e-12:
                if abs(level_delta) / abs(before_mean) * 100.0 < _TREND_LEVEL_PERCENT:
                    continue

            turns.append(
                {
                    "time": _format_live_time(
                        datetime.fromtimestamp(moment, tz=timezone.utc).isoformat()
                    ),
                    "before_mean": before_mean,
                    "after_mean": after_mean,
                    "level_delta": level_delta,
                    # 两个尺度都要给，缺一个模型就会误判：
                    # · level_ratio  相对全程量程 —— 量程大时能抓出「压力消失」这类跃迁
                    # · level_percent 相对变化率   —— 量程很小时能识破「0.6% 的抖动」
                    #   实测真实数据集里 0.0181727→0.01829 只是 0.6% 的微动，
                    #   但因为它占了当时量程的 24%，只看 level_ratio 会被当回事。
                    "level_ratio": abs(level_delta) / span,
                    "level_percent": (
                        abs(level_delta) / abs(before_mean) * 100.0
                        if abs(before_mean) > 1e-12 else float("inf")
                    ),
                    # 变化率方向，与 level_delta 同号
                    "rate_sign": "降" if level_delta < 0 else "升",
                    "after_points": len(after_slice),
                    "after_flat": regress(after_slice)["r2"] >= _TREND_TURNING_R2,
                }
            )

        if turns:
            # 按电平变化量排序取前几个：变化大的才是真事件，位置先后不重要
            turns.sort(key=lambda item: item["level_ratio"], reverse=True)
            features["turning_points"] = turns[:_TREND_TURNING_MAX]

    return features


def _compress_series(samples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """压缩采样序列，**优先保留关键形状点**，其余用等距点补齐。

    关键点 = 首尾 + 极值点 + 导数变化点。这样「平直段 + 突然跳变」的
    工业曲线在少数点下仍能保持形状；单纯等距取样会漏掉跳变。
    """
    if limit <= 0 or len(samples) <= limit:
        return samples

    keep: set[int] = {0, len(samples) - 1}

    # 极值点（局部最大/最小）
    for index in range(1, len(samples) - 1):
        prev = samples[index - 1].get("value")
        curr = samples[index].get("value")
        nxt = samples[index + 1].get("value")
        if None in (prev, curr, nxt):
            continue
        if (curr >= prev and curr >= nxt) or (curr <= prev and curr <= nxt):
            keep.add(index)

    # 导数变化点：前后斜率符号不同，说明这里有折角
    rates: list[float | None] = [None]
    for index in range(1, len(samples)):
        prev, curr = samples[index - 1].get("value"), samples[index].get("value")
        rates.append(None if None in (prev, curr) else float(curr) - float(prev))
    for index in range(1, len(rates) - 1):
        prev, curr = rates[index], rates[index + 1]
        if None in (prev, curr):
            continue
        if (prev > 0 >= curr) or (prev < 0 <= curr):
            keep.add(index + 1)

    # 关键点已超上限时按原始顺序均匀丢弃，保证首尾与分布
    ordered = sorted(keep)
    if len(ordered) > limit:
        step = (len(ordered) - 1) / (limit - 1) if limit > 1 else 0
        ordered = sorted({ordered[min(len(ordered) - 1, round(i * step))] for i in range(limit)})

    # 剩余名额用等距点补齐
    if len(ordered) < limit:
        remaining = [index for index in range(len(samples)) if index not in keep]
        need = limit - len(ordered)
        if remaining:
            step = (len(remaining) - 1) / (need - 1) if need > 1 else 0
            for i in range(need):
                ordered.append(remaining[min(len(remaining) - 1, round(i * step))])
    return [samples[index] for index in sorted(set(ordered))]


def _thin_samples(samples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """等距抽稀采样序列，始终保留首尾两点。

    模型要判断的是「趋势」而不是复现曲线，等距取样足以表达涨跌与拐点；
    随机取样会丢掉首尾这两个最能说明「现在处于什么水平」的点。
    """
    if limit <= 0 or len(samples) <= limit:
        return samples
    if limit == 1:
        return samples[-1:]
    step = (len(samples) - 1) / (limit - 1)
    picked, seen = [], set()
    for index in range(limit):
        position = min(len(samples) - 1, round(index * step))
        if position not in seen:
            seen.add(position)
            picked.append(samples[position])
    return picked


def _parse_plan(text: str) -> dict[str, Any]:
    """解析规划输出为结构化意图。

    **向后兼容**：只认出「检索/测点」两行的旧格式时，其余各项为空，
    行为与扩展前完全一致——这样即使模型漏输出某一行也不会让整条链路退化。
    任何一项解析失败都只丢该项，不影响其他项。
    """
    plan: dict[str, Any] = {
        "needs_rag": None, "points": [], "history": None, "trend": None, "logs": None,
    }
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()

        match = re.match(r"^检索\s*[:：]\s*(yes|no|是|否)", line, re.I)
        if match:
            plan["needs_rag"] = match.group(1).lower() in ("yes", "是")
            continue

        value = _plan_field(line, "测点")
        if value is not None:
            if value.lower() not in _PLAN_EMPTY:
                plan["points"] = [
                    item.strip()
                    for item in re.split(r"[、,，;；|]", value)
                    if item.strip()
                ][:_LIVE_POINTS_MAX]
            continue

        value = _plan_field(line, "历史")
        if value is not None:
            if value.lower() not in _PLAN_EMPTY:
                parts = [part.strip() for part in value.split("|")]
                if parts and parts[0]:
                    start, end = _clamp_range(
                        _plan_date(parts[1] if len(parts) > 1 else ""),
                        _plan_date(parts[2] if len(parts) > 2 else ""),
                    )
                    plan["history"] = {"keyword": parts[0], "start": start, "end": end}
            continue

        value = _plan_field(line, "趋势")
        if value is not None:
            if value.lower() not in _PLAN_EMPTY:
                parts = [part.strip() for part in value.split("|")]
                if parts and parts[0]:
                    start, end = _clamp_range(
                        _plan_date(parts[1] if len(parts) > 1 else ""),
                        _plan_date(parts[2] if len(parts) > 2 else ""),
                    )
                    interval = 0
                    if len(parts) > 3:
                        digits = re.search(r"\d+", parts[3])
                        interval = int(digits.group()) if digits else 0
                    # 间隔留 0 交给下游按跨度自动选，避免模型给出 0 或负数
                    plan["trend"] = {
                        "keyword": parts[0], "start": start, "end": end,
                        "interval_seconds": interval if interval >= 30 else 0,
                    }
            continue

        value = _plan_field(line, "日志")
        if value is not None:
            if value.lower() not in _PLAN_EMPTY:
                parts = [part.strip() for part in value.split("|")]
                start, end = _clamp_range(
                    _plan_date(parts[0] if parts else ""),
                    _plan_date(parts[1] if len(parts) > 1 else ""),
                )
                plan["logs"] = {
                    "start": start, "end": end,
                    "major_only": any("重大" in part or "重要" in part for part in parts[2:]),
                }
            continue

    return plan


def _empty_capabilities() -> dict[str, Any]:
    """保守默认：照常检索、不读任何实时数据。

    规划失败时必须退到这里——多查一次数据只是慢，少查一次却会让模型
    在缺少事实的情况下作答。
    """
    return {"needs_rag": True, "points": [], "history": None, "trend": None, "logs": None}


async def _plan_retrieval(question: str, mode: Literal["online", "offline"]) -> dict[str, Any]:
    """一次调用同时决定「要不要查文档」和「要取哪些实时数据」，判断发生在检索之前。

    极小调用：无证据、只出五行，实测约 200-400ms。
    任何异常或无法解析都退回保守默认：照常检索、不取实时数据。
    """
    if not _ROUTER_ENABLED:
        return _empty_capabilities()
    today = _local_today()
    prompt = _PLANNER_PROMPT.format(
        today=today.isoformat(),
        yesterday=(today - timedelta(days=1)).isoformat(),
    )
    try:
        _, client = _llm_client(mode)
        result = await client.complete(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ],
            # 从两行扩到五行后 32 个令牌不够，会出现「日志: 」被截断的半行。
            # 五行实测约 40-60 个令牌，留一倍余量。
            max_tokens=160,
            temperature=0.0,
            chat_template_kwargs={"enable_thinking": False} if mode == "offline" else None,
        )
    except Exception:
        return _empty_capabilities()
    plan = _parse_plan(result.content)
    if plan["needs_rag"] is None:
        plan["needs_rag"] = True
    return plan


_LIVE_POINT_PICK_PROMPT = (
    "用户要查一个实时测点，但测点表里没有精确匹配项。下面是相近候选。\n"
    "只回答最符合用户意图的那一个 KKS（第一列），不要解释、不要多余文字；"
    "都不合适就回答 无。\n\n候选：\n{candidates}"
)


async def _pick_live_point(
    keyword: str,
    candidates: list[dict[str, Any]],
    mode: Literal["online", "offline"],
) -> str | None:
    """关键词匹配不到时，让模型从候选测点里挑最符合意图的那个。

    用模型而不是同义词表（水位↔液位 之类）：测点表的命名习惯会变，
    硬编码词表很快就废，而且适配不了新机组。
    """
    lines = [
        f"{item.get('kks')}  {item.get('description')}"
        for item in candidates
        if item.get("kks")
    ]
    if not lines:
        return None
    try:
        _, client = _llm_client(mode)
        result = await client.complete(
            [
                {
                    "role": "system",
                    "content": _LIVE_POINT_PICK_PROMPT.format(candidates="\n".join(lines)),
                },
                {"role": "user", "content": keyword},
            ],
            max_tokens=24,
            temperature=0.0,
            chat_template_kwargs={"enable_thinking": False} if mode == "offline" else None,
        )
    except Exception:
        return None
    chosen = (result.content or "").strip()
    return chosen if chosen in {str(item.get("kks")) for item in candidates} else None


def _keyword_variants(keyword: str) -> list[str]:
    """逐级去掉尾字放宽搜索范围。

    测点表里叫「凝汽器液位」，用户会问「凝汽器水位」—— 整词匹配直接落空。
    这里**只放宽候选范围**，具体读哪个测点仍由模型从真实候选里判断，
    所以不是硬编码同义词表（水位→液位 那种），换机组、换命名习惯都不用改。
    """
    text = keyword.strip()
    variants = [text]
    while len(text) > 2 and len(variants) < 3:
        text = text[:-1]
        variants.append(text)
    return variants


async def _fetch_live_points(
    keywords: list[str],
    mode: Literal["online", "offline"],
) -> tuple[list[dict[str, Any]], list[str]]:
    """并行取实时测点值。单个失败只记录不抛出 —— 实时数据缺失不能拖垮文档问答。"""
    if not keywords:
        return [], []

    async def one(keyword: str) -> tuple[dict[str, Any] | None, str | None]:
        def shape(data: dict[str, Any]) -> dict[str, Any]:
            point = data.get("point") or {}
            return {
                "query": keyword,
                "kks": str(point.get("kks") or ""),
                "description": str(point.get("description") or ""),
                "value": data.get("value"),
                "unit": str(data.get("unit") or ""),
                "time": _format_live_time(data.get("time")),
                "value_source": str(data.get("value_source") or ""),
            }

        # 快路径：直接按关键词取，命中时最省一次检索。
        # **但必须校验相关性**：工具内部是按相似度取 top-1，模糊匹配可能返回
        # 完全无关的测点（实测搜「轴封供汽压力」top-1 是「高压主蒸汽压力3选1后」）。
        # 值非空就采用，等于把错数据当成事实喂给模型——比不返回更糟。
        try:
            data = await _mcp_call_tool("point_query_current_value", {"query_text": keyword})
            if isinstance(data, dict) and data.get("value") is not None:
                point = data.get("point") or {}
                if _point_matches(keyword, str(point.get("description") or "")):
                    return shape(data), None
        except Exception:
            pass

        # 回退：top-1 没值往往不是「这个测点没数据」，而是关键词匹配偏了
        # （例如「锅炉给水温度」匹配到「主蒸汽温度」）。拿候选逐个按 KKS 精确取值，
        # 命中第一个有值的就返回。命中场景耗时不变，只有失败的才多花这几百毫秒。
        candidates: list[dict[str, Any]] = []
        for variant in _keyword_variants(keyword):
            try:
                found = await _mcp_call_tool(
                    "point_query_search_points",
                    {"query_text": variant, "limit": _LIVE_POINT_CANDIDATES},
                )
            except Exception as exc:
                return None, f"{keyword}: {exc}"
            candidates = [
                item
                for item in (found.get("items") or [])
                if str(item.get("kks") or "").strip()
            ]
            if candidates:
                break

        if not candidates:
            return None, f"{keyword}: 测点表中无匹配项"
        picked = await _pick_live_point(keyword, candidates, mode)
        if not picked:
            return None, f"{keyword}: 候选中无匹配项"
        try:
            data = await _mcp_call_tool("point_query_current_value", {"kks": picked})
        except Exception as exc:
            return None, f"{keyword}: {exc}"
        if isinstance(data, dict) and data.get("value") is not None:
            return shape(data), None
        return None, f"{keyword}: 选中测点无实时值"

    results = await asyncio.gather(*(one(keyword) for keyword in keywords))
    return (
        [item for item, _ in results if item],
        [error for _, error in results if error],
    )

async def _plan_rag(
    question: str,
    mode: Literal["online", "offline"],
    base_top_k: int,
) -> tuple[dict[str, Any], str | None]:
    try:
        plan = await _mcp_call_tool(
            "rag_query_plan",
            {
                "question": question,
                "inference_mode": mode,
                "base_top_k": _graph_top_k(mode, base_top_k),
            },
        )
        rounds = max(1, min(8, int(plan.get("rounds") or 1)))
        queries = [
            str(item).strip()
            for item in plan.get("queries", [])
            if str(item).strip()
        ][:rounds]
        if not queries:
            queries = [question]
        rounds = len(queries)
        top_k = max(5, math.ceil(_graph_top_k(mode, base_top_k) / rounds))
        return {
            "rounds": rounds,
            "queries": queries,
            "top_k": top_k,
            "base_top_k": _graph_top_k(mode, base_top_k),
            "reason": str(plan.get("reason") or ""),
            "planner": "mcp:rag_query_plan",
        }, None
    except Exception as exc:
        return {
            "rounds": 1,
            "queries": [question],
            "top_k": _graph_top_k(mode, base_top_k),
            "base_top_k": _graph_top_k(mode, base_top_k),
            "planner": "fallback",
        }, str(exc)


def _point_matches(keyword: str, description: str) -> bool:
    """关键词与测点描述是否**足够**相关。

    **不能只看检索排序**。实测：搜「轴封供汽压力」返回 5 条，正确答案排第 4，
    而 top-1 是完全无关的「高压主蒸汽压力3选1后」——SIS 的检索是模糊匹配，
    取 top-1 会静默用错测点，比报「没找到」更糟。

    判据取关键词的**首二字与尾二字**都必须出现在描述里：
    首二字是设备标识（「轴封」），尾二字是物理量（「压力」）。
    两者同时命中才算相关——只命中设备会把「轴封供汽管道疏水母管气动关断阀
    开反馈」这种阀门反馈误当成压力测点。
    """
    text = (description or "").strip()
    if not text:
        return False
    probe = (keyword or "").strip()
    if len(probe) < 2:
        return probe in text
    if probe[:2] not in text:
        return False
    # 关键词太短（如「压力」）时尾部就是首部，不重复判定
    if len(probe) < 4:
        return True
    if probe[-2:] not in text:
        return False
    # 中间那段限定词至少要见一个。「轴封供气压力」的中间是「供气」——
    # 实测它被规划器简化成「轴封压力」后，唯一命中是「低压轴封压力测点1」，
    # 首尾都合、只有「供气」对不上，结果串到了另一个测点。
    # 只要求**命中一个**而不是全中：汽／气 这类异体字差异不该被拦下。
    middle = probe[2:-2]
    if not middle:
        return True
    return any(ch in text for ch in middle)


async def _resolve_point(
    keyword: str, mode: Literal["online", "offline"] = "offline"
) -> tuple[dict[str, Any] | None, str | None]:
    """中文关键词 → 具体测点。返回 (点, 错误)。

    三步：原词精确 → 放宽变体 → 模型从候选里选。
    任何一步都不接受「相关性不达标」的测点——宁可报没找到，也不能给错数据。
    """
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for variant in _keyword_variants(keyword):
        try:
            data = await _mcp_call_tool(
                "point_query_search_points",
                {"query_text": variant, "limit": _LIVE_POINT_CANDIDATES},
            )
        except Exception as exc:
            return None, f"{keyword}: {exc}"
        for item in (data or {}).get("items") or []:
            kks = str(item.get("kks") or "").strip()
            if not kks or kks in seen:
                continue
            seen.add(kks)
            candidates.append(item)
        if candidates:
            break

    if not candidates:
        return None, f"{keyword}: 测点表中无匹配项"

    # 字符级判定优先：够相关就直接用，不必再花一次模型调用
    for item in candidates:
        if _point_matches(keyword, str(item.get("description") or "")):
            return item, None

    # 字符判定全不中（测点表叫「凝汽器液位」而用户问「凝汽器水位」）：
    # 让模型从真实候选里挑。选出来的 KKS 必须确实在候选集合内。
    picked = await _pick_live_point(keyword, candidates, mode)
    for item in candidates:
        if str(item.get("kks")) == picked:
            return item, None
    return None, f"{keyword}: 候选中无相关测点"


def _shape_series(data: dict[str, Any], keyword: str, kind: str) -> dict[str, Any]:
    samples = data.get("samples") or []
    summary = data.get("summary") or {}
    return {
        "kind": kind,
        "query": keyword,
        "kks": str((data.get("point") or {}).get("kks") or ""),
        "description": (data.get("point") or {}).get("description") or keyword,
        "unit": data.get("unit") or "",
        "start": str(data.get("start_time") or ""),
        "end": str(data.get("end_time") or ""),
        "interval_seconds": data.get("interval_seconds") or 0,
        "summary": summary,
        "samples": samples,
    }


async def _fetch_series(
    spec: dict[str, Any], kind: Literal["history", "trend"], mode: str = "offline"
) -> tuple[dict[str, Any] | None, str | None]:
    """取测点历史序列或趋势。趋势即带采样间隔的历史序列，用的是同一个 MCP 工具。"""
    keyword = str(spec.get("keyword") or "").strip()
    if not keyword:
        return None, None
    point, error = await _resolve_point(keyword, mode)  # type: ignore[arg-type]
    if error or not point:
        return None, error or f"{keyword}: 未匹配到测点"
    arguments: dict[str, Any] = {
        "kks": str(point.get("kks") or ""),
        # SIS 侧按 UTC 存时间，这里把本地日期补成当天 00:00:00 ~ 23:59:59
        "start_time": f"{spec.get('start')}T00:00:00",
        "end_time": f"{spec.get('end')}T23:59:59",
    }
    if kind == "trend" and spec.get("interval_seconds"):
        arguments["interval_seconds"] = int(spec["interval_seconds"])
    try:
        data = await _mcp_call_tool("point_query_history_series", arguments)
    except Exception as exc:
        return None, f"{keyword}: {exc}"
    if not isinstance(data, dict) or not data.get("samples"):
        return None, f"{keyword}: 该时段无采样数据"
    return _shape_series(data, keyword, kind), None


async def _fetch_logs(spec: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """取运行日志。按规划给出的区间选择工具，只取重大事件时走 major_events。"""
    start, end = str(spec.get("start") or ""), str(spec.get("end") or "")
    major_only = bool(spec.get("major_only"))
    # 区间内天数决定用哪个工具：单日走 recent，跨日走 range。
    # **不给 fetch_if_missing 以外的参数**——该参数默认已是 True，
    # 日志模块不缓存，每次都是真实抓取（实测约 5.8 s）。
    try:
        today = _local_today().isoformat()
        if major_only:
            # major_events 只接受天数，不接受区间，按区间长度折算
            span = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
            data = await _mcp_call_tool(
                "log_query_major_events", {"days": max(1, min(_QUERY_MAX_DAYS, span))}
            )
        elif start == end and end == today:
            data = await _mcp_call_tool("log_query_recent", {"days": 1})
        else:
            span = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
            data = await _mcp_call_tool(
                "log_query_range", {"days": max(1, min(_QUERY_MAX_DAYS, span))}
            )
    except Exception as exc:
        return None, f"日志: {exc}"
    if not isinstance(data, dict):
        return None, "日志: 返回格式异常"
    if str(data.get("status")) == "unavailable":
        return None, f"日志: {data.get('message') or '数据源未配置'}"
    events = data.get("events") or []
    if not events:
        return None, "日志: 该时段无记录"
    return {
        "kind": "logs",
        "start": start,
        "end": end,
        "major_only": major_only,
        "summary": data.get("summary") or {},
        # 118 条全量注入会挤占文档证据，按严重程度排序后截断
        "events": sorted(
            events,
            key=lambda item: {"major": 0, "important": 1}.get(str(item.get("severity")), 2),
        )[:_LOGS_MAX],
    }, None


async def _current_drift(point: dict[str, Any], mode: str = "offline") -> dict[str, Any] | None:
    """当前值与近期均值的对比。**宁可少报**：两个判据都超标才提示偏离。

    当前值是随口一问。若只要有差异就提示「偏离 0.5%」，提示会多到没人看，
    等于没有提示。所以这里方向与趋势相反——趋势宁可多报，当前值宁可少报。

    两个判据：
      · 相对偏离 = |当前 - 均值| / |均值|      —— 物理量级上的偏离
      · σ 倍率   = |当前 - 均值| / 标准差      —— 相对近期自身波动幅度的偏离
    只用一个不够：均值很小而波动很大时，相对偏离会虚高；
    而序列本身很平稳时，σ 又会过小。
    """
    kks = str(point.get("kks") or "").strip()
    current = point.get("value")
    if not kks or current is None:
        return None
    try:
        current_value = float(current)
    except (TypeError, ValueError):
        return None

    try:
        data = await _mcp_call_tool(
            "point_query_history_series",
            {"kks": kks, "interval_seconds": max(60, _DRIFT_WINDOW_MINUTES * 60 // 60)},
        )
    except Exception:
        return None
    samples = (data or {}).get("samples") or []
    values = [float(s["value"]) for s in samples if s.get("value") is not None]
    if len(values) < 5:
        return None

    # 当前值本身也在历史里，算均值时把它排除，否则会把基准往自己身上拉
    history = values[:-1] or values
    mean = sum(history) / len(history)
    if len(history) > 1:
        variance = sum((v - mean) ** 2 for v in history) / (len(history) - 1)
        std = variance ** 0.5
    else:
        std = 0.0

    delta = current_value - mean
    rel = abs(delta) / abs(mean) * 100.0 if abs(mean) > 1e-12 else float("inf")
    sigma = abs(delta) / std if std > 1e-12 else float("inf")

    # 宁可少报：**两个都超标**才提示偏离
    drifted = rel >= _DRIFT_REL_PERCENT and sigma >= _DRIFT_SIGMA
    return {
        "mean": mean,
        "std": std,
        "delta": delta,
        "rel_percent": rel,
        "sigma": sigma,
        "drifted": drifted,
        "window_minutes": _DRIFT_WINDOW_MINUTES,
        "samples": len(history),
    }


async def _fetch_realtime(
    plan: dict[str, Any], mode: Literal["online", "offline"]
) -> tuple[dict[str, Any], list[str]]:
    """并行取齐规划要求的全部实时数据。

    三类查询互不依赖，且都远慢于本地计算（日志实测约 5.8 s），
    串行执行会让时延叠加；并发后总耗时取决于最慢的一项。

    **单项失败只记录不抛出**：实时数据缺失不能拖垮文档问答，这与
    `_fetch_live_points` 的既有约定一致。
    """
    tasks: dict[str, Any] = {}
    if plan.get("points"):
        tasks["live"] = _fetch_live_points(plan["points"], mode)
    if plan.get("history"):
        tasks["history"] = _fetch_series(plan["history"], "history", mode)
    if plan.get("trend"):
        tasks["trend"] = _fetch_series(plan["trend"], "trend", mode)
    if plan.get("logs"):
        tasks["logs"] = _fetch_logs(plan["logs"])

    result: dict[str, Any] = {"live_points": [], "series": [], "logs": None}
    errors: list[str] = []
    if not tasks:
        return result, errors

    done = await asyncio.gather(*tasks.values(), return_exceptions=True)
    for name, outcome in zip(tasks.keys(), done):
        if isinstance(outcome, BaseException):
            errors.append(f"{name}: {outcome}")
            continue
        if name == "live":
            points, point_errors = outcome
            result["live_points"] = points
            errors.extend(point_errors)
            # 给每个当前值配一段「与近期均值的对比」。并发取，不逐点串行。
            if points:
                drifts = await asyncio.gather(
                    *(_current_drift(point, mode) for point in points),
                    return_exceptions=True,
                )
                for point, drift in zip(points, drifts):
                    if isinstance(drift, dict):
                        point["drift"] = drift
        else:
            item, error = outcome
            if error:
                errors.append(error)
            elif item:
                if name == "logs":
                    result["logs"] = item
                else:
                    # 出图放在取数之后、注入之前。失败不影响任何环节——
                    # 图是锦上添花，数据才是主体。
                    _vals = [s.get("value") for s in (item.get("samples") or []) if s.get("value") is not None]
                    _sig = _series_significance(_vals, item.get("limits"))
                    # level 0（<1%）连图都不给：整段无变化时，图只会让模型去找细节
                    item["chart_png"] = "" if _sig.get("level") == 0 else _render_trend_png(item, {})
                    result["series"].append(item)
    return result, errors


async def _retrieve_planned(
    question: str,
    mode: Literal["online", "offline"],
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # 规划在检索之前：一次调用同时决定要不要查文档，以及要取哪几类实时数据
    # （测点当前值／历史序列／趋势／运行日志）。
    intent = await _plan_retrieval(question, mode)

    if not intent["needs_rag"]:
        realtime, rt_errors = await _fetch_realtime(intent, mode)
        plan = {
            "rounds": 0,
            "queries": [],
            "top_k": 0,
            "base_top_k": _graph_top_k(mode, top_k),
            "reason": "模型判定无需检索知识库",
            "planner": "model-router",
            "no_retrieval": True,
        }
        return [], {
            "plan": plan,
            "rounds": [],
            "deduplicated_count": 0,
            "no_retrieval": True,
            "live_errors": rt_errors,
            **realtime,
        }
    plan, planning_error = await _plan_rag(question, mode, top_k)
    if plan.get("no_retrieval"):
        # 不检索也不查图谱：下游拿到空 contexts 会自然产出空 citations / graph_context，
        # 前端 addEvidence() 在两者都空时不渲染依据面板。
        realtime, rt_errors = await _fetch_realtime(intent, mode)
        return [], {
            "plan": plan,
            "rounds": [],
            "deduplicated_count": 0,
            "no_retrieval": True,
            "live_errors": rt_errors,
            **realtime,
        }

    # **实时数据与检索并行**。二者互不依赖，且日志/历史抓取实测约 5.8 s，
    # 与检索轮次的 6 s 同量级——串行会让时延直接叠加，并发则相互掩盖，
    # 总耗时取决于较慢的一项而不是两者之和。
    realtime_task = asyncio.create_task(_fetch_realtime(intent, mode))

    merged: dict[str, dict[str, Any]] = {}
    round_infos: list[dict[str, Any]] = []
    for round_index, query in enumerate(plan["queries"], 1):
        contexts, info = await _retrieve(query, int(plan["top_k"]))
        round_infos.append({"round": round_index, "query": query, **info, "count": len(contexts)})
        for position, item in enumerate(contexts):
            identity = str(item.get("id") or item.get("chunk_id") or item.get("text", ""))
            if not identity:
                continue
            candidate = dict(item)
            candidate["_rag_round"] = round_index
            candidate["_rag_position"] = position
            current = merged.get(identity)
            current_score = _retrieval_score(current) if current else float("-inf")
            candidate_score = _retrieval_score(candidate)
            if current is None or candidate_score > current_score:
                merged[identity] = candidate

    realtime, rt_errors = await realtime_task
    contexts = sorted(
        merged.values(),
        key=lambda item: (
            _retrieval_score(item),
            -int(item.get("_rag_round", 1)),
            -int(item.get("_rag_position", 0)),
        ),
        reverse=True,
    )
    info = {
        "plan": plan,
        "rounds": round_infos,
        "deduplicated_count": len(contexts),
        "live_errors": rt_errors,
        **realtime,
    }
    if planning_error:
        info["planning_error"] = planning_error
    if round_infos:
        info.update({key: round_infos[0].get(key) for key in ("backend", "url") if round_infos[0].get(key)})
    return contexts, info


def _retrieval_score(item: dict[str, Any]) -> float:
    try:
        return float(item.get("score", float("-inf")))
    except (TypeError, ValueError):
        return float("-inf")


def _needs_cross_mode_compression(
    previous_mode: str | None,
    selected_mode: Literal["online", "offline"],
) -> bool:
    return previous_mode == "online" and selected_mode == "offline"


def _graph_top_k(mode: Literal["online", "offline"], requested_top_k: int) -> int:
    configured = (
        settings.online_graph_top_k
        if mode == "online"
        else settings.local_graph_top_k
    )
    return max(1, min(requested_top_k, configured, 40))


def _llm_contexts(
    mode: Literal["online", "offline"],
    contexts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return contexts[:_graph_top_k(mode, len(contexts))]


def _conversation_messages(
    user_code: str,
    conversation_id: str | None,
    requested: list[ChatMessage],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    data: dict[str, Any] = {}
    if conversation_id:
        path = _conversation_path(user_code, conversation_id)
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data = _normalize_conversation(conversation_id, loaded)
            except (OSError, json.JSONDecodeError):
                data = {}
    if requested:
        messages = [
            {"role": message.role, "content": message.content}
            for message in requested
            if message.role in {"user", "assistant"} and message.content.strip()
        ]
        return messages, data
    if not data:
        return [], {}
    messages = [
        {"role": item.get("role", ""), "content": item.get("content", "")}
        for item in data.get("messages", [])
        if isinstance(item, dict)
        and item.get("role") in {"user", "assistant"}
        and str(item.get("content", "")).strip()
    ]
    return messages, data


async def _compress_history(
    client: OpenAICompatibleClient,
    messages: list[dict[str, str]],
    *,
    mode: Literal["online", "offline"],
) -> str:
    thinking_enabled = _thinking_enabled(mode)
    return (
        await client.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "你负责压缩工业问答会话上下文。保留用户目标、已确认事实、"
                        "关键设备/参数/时间、已给出的证据结论、未解决问题和约束。"
                        "删除寒暄、重复内容和无关细节。只输出结构化中文摘要，"
                        "不要回答新问题，不要编造信息。"
                    ),
                },
                {"role": "user", "content": json.dumps(messages, ensure_ascii=False)},
            ],
            max_tokens=settings.context_compression_max_tokens,
            temperature=0.1,
            reasoning_effort="low" if thinking_enabled else None,
            chat_template_kwargs=(
                {"enable_thinking": False}
                if mode == "offline" and not thinking_enabled
                else None
            ),
            thinking_enabled=thinking_enabled if mode == "online" else None,
        )
    ).content


def _rag_route_order() -> tuple[list[str], set[str]]:
    local_url = settings.rag_base_url
    remote_urls = list(dict.fromkeys(url for url in settings.rag_remote_urls if url != local_url))
    if not remote_urls:
        return [local_url], set()

    global _rag_route_cursor
    with _rag_route_lock:
        start = _rag_route_cursor % len(remote_urls)
        _rag_route_cursor = (start + 1) % len(remote_urls)
    return remote_urls[start:] + remote_urls[:start] + [local_url], set(remote_urls)


def _rag_health_urls() -> tuple[list[str], set[str]]:
    local_url = settings.rag_base_url
    remote_urls = list(dict.fromkeys(url for url in settings.rag_remote_urls if url != local_url))
    return remote_urls + [local_url], set(remote_urls)


async def _check_rag_url(client: httpx.AsyncClient, url: str) -> dict[str, Any]:
    try:
        response = await client.get(f"{url}/health")
        response.raise_for_status()
        payload = response.json()
        return {"url": url, "status": payload.get("status", "ok"), "health": payload}
    except Exception as exc:
        return {"url": url, "status": "unavailable", "error": str(exc)}


async def _rag_health() -> dict[str, Any]:
    urls, remote_urls = _rag_health_urls()
    try:
        async with httpx.AsyncClient(timeout=settings.rag_health_timeout) as client:
            checks = await asyncio.gather(*(_check_rag_url(client, url) for url in urls))
    except Exception as exc:
        return {
            "status": "unavailable",
            "mode": "remote-with-local-fallback" if remote_urls else "local",
            "error": str(exc),
        }
    remote = [item for item in checks if item["url"] in remote_urls]
    local = next(item for item in checks if item["url"] == settings.rag_base_url)
    status = "ok" if any(item["status"] == "ok" for item in checks) else "unavailable"
    return {
        "status": status,
        "mode": "remote-with-local-fallback" if remote_urls else "local",
        "remote": remote,
        "local_fallback": local,
    }


async def _llm_health() -> dict[str, Any]:
    mode = _resolve_inference_mode(None)
    profiles = {
        name: _effective_profile(name)  # type: ignore[arg-type]
        for name in ("online", "offline")
    }
    local_runtime = await _local_runtime_info(profiles["offline"]["base_url"])
    context_windows = {
        "online": profiles["online"]["context_window"],
        "offline": local_runtime["context_window"] or profiles["offline"]["context_window"],
    }
    try:
        runtime_model = (
            local_runtime["model"]
            if mode == "offline" and local_runtime["available"] and local_runtime["model"]
            else None
        )
        _, client = _llm_client(mode, model_override=runtime_model)
        data = await client.health()
        return {
            "status": "ok",
            "mode": mode,
            "model": client.model,
            "context_windows": context_windows,
            "models": data.get("data", []),
            "runtime": {
                "offline": {
                    "available": local_runtime["available"],
                    "loaded_model": local_runtime["model"],
                    "context_window": local_runtime["context_window"],
                    "vision": local_runtime["vision"],
                    "detail": local_runtime["detail"],
                },
            },
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "mode": mode,
            "model": (
                _effective_profile("online")["model"]
                if mode == "online"
                else _effective_profile("offline")["model"]
            ),
            "context_windows": context_windows,
            "runtime": {
                "offline": {
                    "available": local_runtime["available"],
                    "loaded_model": local_runtime["model"],
                    "context_window": local_runtime["context_window"],
                    "vision": local_runtime["vision"],
                    "detail": local_runtime["detail"],
                },
            },
            "error": str(exc),
        }


def _local_vision_status(
    runtime: dict[str, Any],
    expected_model: str | None = None,
) -> tuple[bool, str]:
    if not runtime["available"]:
        return False, str(runtime["detail"])
    loaded_model = str(runtime["model"])
    if expected_model and loaded_model and expected_model not in loaded_model:
        return False, f"当前 llama.cpp 已加载 {loaded_model}，未加载候选模型 {expected_model}"
    return bool(runtime["vision"]), str(runtime["detail"])


async def _local_vision_capability(
    base_url: str,
    expected_model: str | None = None,
) -> tuple[bool, str]:
    return _local_vision_status(
        await _local_runtime_info(base_url),
        expected_model,
    )


async def _vision_candidates(
    *,
    allow_online: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    config = _read_model_config()
    candidates: list[dict[str, Any]] = []
    errors: list[str] = []
    priorities = sorted(
        (
            item
            for item in config.get("vision_priority", [])
            if isinstance(item, dict) and item.get("enabled", True)
        ),
        key=lambda item: int(item.get("priority") or 999),
    )
    for item in priorities:
        mode = str(item.get("mode") or "").lower()
        if mode not in {"online", "offline"}:
            continue
        model = str(item.get("model") or "").strip()
        if not model:
            continue
        profile = _effective_profile(mode)  # type: ignore[arg-type]
        if mode == "online":
            if not allow_online:
                errors.append(f"{model}: 当前用户未开放在线推理")
                continue
            if not settings.online_llm_api_key:
                errors.append(f"{model}: 在线 API key 未配置")
                continue
            if not bool(config.get("online", {}).get("multimodal_enabled", True)):
                errors.append(f"{model}: 在线多模态已停用")
                continue
        else:
            local_runtime = await _local_runtime_info(profile["base_url"])
            available, reason = _local_vision_status(local_runtime, model)
            if not available:
                errors.append(f"{model}: {reason}")
                continue
            if not bool(config.get("local", {}).get("multimodal_enabled", False)):
                errors.append(f"{model}: 本地多模态已停用")
                continue
            profile["context_window"] = (
                int(local_runtime["context_window"])
                or int(profile["context_window"])
            )
        candidates.append({
            "mode": mode,
            "model": model,
            "base_url": profile["base_url"],
            "context_window": int(profile["context_window"]),
        })
    return candidates, errors


async def _graph_health() -> dict[str, Any]:
    payload = {"statements": [{"statement": "RETURN 1 AS ok"}]}
    user, _, password = settings.neo4j_auth.partition("/")
    try:
        async with httpx.AsyncClient(timeout=5, auth=(user, password)) as client:
            res = await client.post(settings.neo4j_http_url, json=payload)
            res.raise_for_status()
            data = res.json()
        if data.get("errors"):
            return {"status": "unavailable", "error": data["errors"]}
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc)}


async def _retrieve(question: str, top_k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    urls, remote_urls = _rag_route_order()
    errors: list[str] = []
    timeout = httpx.Timeout(settings.rag_timeout, connect=settings.rag_connect_timeout)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for url in urls:
            try:
                response = await client.post(
                    f"{url}/search",
                    json={"query": question, "top_k": top_k},
                )
                response.raise_for_status()
                results = response.json().get("results", [])
                # Reranker logits are not calibrated probabilities and valid matches can be negative.
                filtered = [
                    item
                    for item in results
                    if isinstance(item, dict) and str(item.get("text", "")).strip()
                ]
                if url in remote_urls:
                    backend = "remote"
                elif remote_urls:
                    backend = "local-fallback"
                else:
                    backend = "local"
                return filtered, {"backend": backend, "url": url}
            except Exception as exc:
                errors.append(f"{url}: {exc}")
    raise HTTPException(
        status_code=503,
        detail=f"RAG retrieval failed; tried {len(urls)} backend(s): {'; '.join(errors)}",
    )


async def _graph_context(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunk_ids = [str(item.get("id", "")) for item in contexts if item.get("id")]
    if not chunk_ids:
        return []
    query = """
    UNWIND $chunk_ids AS chunk_id
    MATCH (c:Chunk {id: chunk_id})
    OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(c)
    OPTIONAL MATCH (s:Section)-[:HAS_CHUNK]->(c)
    OPTIONAL MATCH (prev:Chunk)-[:NEXT_CHUNK]->(c)
    OPTIONAL MATCH (c)-[:NEXT_CHUNK]->(next:Chunk)
    OPTIONAL MATCH (c)-[:MENTIONS]->(equipment:Equipment)
    WHERE coalesce(equipment.status, 'candidate') <> 'needs_review'
    OPTIONAL MATCH (c)-[:MENTIONS]->(parameter:Parameter)
    WHERE coalesce(parameter.status, 'candidate') <> 'needs_review'
    OPTIONAL MATCH (c)-[:EVIDENCE_OF]->(fault:Fault)
    WHERE coalesce(fault.status, 'candidate') <> 'needs_review'
    OPTIONAL MATCH (c)-[:MENTIONS]->(alarm:Alarm)
    WHERE coalesce(alarm.status, 'candidate') <> 'needs_review'
    OPTIONAL MATCH (c)-[:HAS_ACTION]->(action:Action)
    WHERE coalesce(action.status, 'candidate') <> 'needs_review'
    RETURN c.id AS chunk_id,
           c.section_title AS section_title,
           c.section_path AS section_path,
           d.id AS document_id,
           d.title AS document_title,
           collect(DISTINCT prev.text)[0..1] AS previous_texts,
           collect(DISTINCT next.text)[0..1] AS next_texts,
           collect(DISTINCT next.id)[0..3] AS next_chunk_ids,
           collect(DISTINCT {label: 'Equipment', name: coalesce(equipment.name, ''), value: '', unit: '', status: coalesce(equipment.status, ''), confidence: coalesce(equipment.confidence, 0.0)})[0..4] +
           collect(DISTINCT {label: 'Parameter', name: coalesce(parameter.name, ''), value: coalesce(parameter.value, ''), unit: coalesce(parameter.unit, ''), status: coalesce(parameter.status, ''), confidence: coalesce(parameter.confidence, 0.0)})[0..4] +
           collect(DISTINCT {label: 'Fault', name: coalesce(fault.name, ''), value: '', unit: '', status: coalesce(fault.status, ''), confidence: coalesce(fault.confidence, 0.0)})[0..3] +
           collect(DISTINCT {label: 'Alarm', name: coalesce(alarm.name, ''), value: '', unit: '', status: coalesce(alarm.status, ''), confidence: coalesce(alarm.confidence, 0.0)})[0..3] +
           collect(DISTINCT {label: 'Action', name: coalesce(action.name, ''), value: '', unit: '', status: coalesce(action.status, ''), confidence: coalesce(action.confidence, 0.0)})[0..5] AS entities
    """
    payload = {"statements": [{"statement": query, "parameters": {"chunk_ids": chunk_ids}}]}
    user, _, password = settings.neo4j_auth.partition("/")
    try:
        async with httpx.AsyncClient(timeout=settings.graph_timeout, auth=(user, password)) as client:
            res = await client.post(settings.neo4j_http_url, json=payload)
            res.raise_for_status()
            data = res.json()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"graph lookup failed: {exc}") from exc
    if data.get("errors"):
        raise HTTPException(status_code=503, detail=f"graph lookup failed: {data['errors']}")
    rows = []
    for result in data.get("results", []):
        columns = result.get("columns", [])
        for row in result.get("data", []):
            values = row.get("row", [])
            item = dict(zip(columns, values, strict=True))
            item["entities"] = [
                entity for entity in item.get("entities", []) if entity.get("name") or entity.get("value")
            ]
            rows.append(item)
    return rows


async def _cypher(statement: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    payload = {"statements": [{"statement": statement, "parameters": parameters or {}}]}
    user, _, password = settings.neo4j_auth.partition("/")
    try:
        async with httpx.AsyncClient(timeout=settings.graph_timeout) as client:
            response = await client.post(
                settings.neo4j_http_url,
                json=payload,
                auth=(user, password),
            )
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"graph lookup failed: {exc}") from exc
    if data.get("errors"):
        raise HTTPException(status_code=503, detail=f"graph lookup failed: {data['errors']}")
    rows: list[dict[str, Any]] = []
    for result in data.get("results", []):
        columns = result.get("columns", [])
        for item in result.get("data", []):
            values = item.get("row", [])
            rows.append(dict(zip(columns, values, strict=True)))
    return rows


@app.get("/graph")
async def knowledge_graph(
    limit: int = Query(default=60000, ge=20, le=100000),
    edge_limit: int = Query(default=120000, ge=20, le=200000),
    label: str | None = Query(default=None),
    x_hdw_session: str | None = Header(default=None),
) -> dict[str, Any]:
    _current_user(x_hdw_session)
    allowed_labels = [
        "Document",
        "Section",
        "Chunk",
        "Equipment",
        "Parameter",
        "Fault",
        "Alarm",
        "Action",
    ]
    labels = [label] if label in allowed_labels else allowed_labels
    node_rows = await _cypher(
        """
        MATCH (node)
        WHERE any(node_label IN labels(node) WHERE node_label IN $labels)
        WITH node
        ORDER BY elementId(node)
        LIMIT $limit
        RETURN {
          id: elementId(node),
          labels: labels(node),
          label: coalesce(node.name, node.title, node.section_title, node.source_file,
                          node.id, node.chunk_id, elementId(node)),
          stable_id: node.id,
          title: node.title,
          source_file: node.source_file,
          section_title: node.section_title,
          section_path: coalesce(node.section_path, node.path),
          status: node.status,
          confidence: node.confidence
        } AS node
        """,
        {"labels": labels, "limit": limit},
    )
    nodes = [
        row["node"]
        for row in node_rows
        if isinstance(row.get("node"), dict) and row["node"].get("id")
    ]
    node_ids = {node["id"] for node in nodes}
    edge_rows = await _cypher(
        """
        MATCH (source)-[relation]->(target)
        WHERE any(node_label IN labels(source) WHERE node_label IN $allowed_labels)
          AND any(node_label IN labels(target) WHERE node_label IN $allowed_labels)
        RETURN elementId(source) AS source,
               elementId(target) AS target,
               type(relation) AS type
        ORDER BY CASE type(relation)
          WHEN 'NEXT_CHUNK' THEN 0
          WHEN 'HAS_SECTION' THEN 1
          WHEN 'HAS_CHUNK' THEN 2
          WHEN 'HAS_SUBSECTION' THEN 3
          ELSE 4
        END,
        elementId(source), elementId(target)
        LIMIT $edge_limit
        """,
        {"allowed_labels": allowed_labels, "edge_limit": edge_limit},
    )
    edge_rows = [
        edge
        for edge in edge_rows
        if edge.get("source") in node_ids and edge.get("target") in node_ids
    ]
    return {
        "nodes": nodes,
        "edges": edge_rows,
        "stats": {
            "nodes": len(nodes),
            "edges": len(edge_rows),
            "label": label or "全部",
            "limit": limit,
            "edge_limit": edge_limit,
            "sampled": False,
            "strategy": "full-graph",
            "truncated": len(nodes) >= limit or len(edge_rows) >= edge_limit,
        },
    }


def _entity_summary(row: dict[str, Any]) -> str:
    items = []
    for entity in row.get("entities") or []:
        label = entity.get("label", "")
        name = entity.get("name") or entity.get("value") or ""
        unit = entity.get("unit") or ""
        if label and name:
            items.append(f"{label}:{name}{unit if unit and unit not in name else ''}")
    return "，".join(items) or "无"


def _prompt(
    question: str,
    contexts: list[dict[str, Any]],
    graph_rows: list[dict[str, Any]] | None = None,
    history_messages: list[dict[str, str]] | None = None,
    model_name: str | None = None,
    image_attachments: list[dict[str, str]] | None = None,
    skip_retrieval: bool = False,
    live_points: list[dict[str, Any]] | None = None,
    series: list[dict[str, Any]] | None = None,
    logs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if contexts:
        evidence_items = contexts
        evidence = "\n\n".join(
            (
                f"[证据 {idx}] "
                f"来源={Path(str((item.get('metadata') or {}).get('source_file', ''))).name or '未知'} "
                f"片段={(item.get('metadata') or {}).get('chunk_index', '未知')}\n"
                f"{str(item.get('text', ''))}"
            )
            for idx, item in enumerate(evidence_items, 1)
        )
    else:
        evidence = "无"

    if graph_rows:
        graph_items = graph_rows
        graph = "\n".join(
            (
                f"[图谱 {idx}] 文档={row.get('document_title') or row.get('document_id')}; "
                f"章节={'/'.join(row.get('section_path') or []) or row.get('section_title') or '无'}; "
                f"chunk={row.get('chunk_id')}; "
                f"实体={_entity_summary(row)}; "
                f"前文={((row.get('previous_texts') or [''])[0]) or '无'}; "
                f"后文={((row.get('next_texts') or [''])[0]) or '无'}"
            )
            for idx, row in enumerate(graph_items, 1)
        )
    else:
        graph = "无"

    # 实时测点单独成块：它是实时数据，不是文档证据，两者在回答里要分开对待。
    # 采集时间在后端已经换算成北京时间，这里直接给结果：标出「北京时间」是为了
    # 堵住模型自己补时区（它会写成 UTC，或者把两者都列一遍）。
    live_block = ""
    if live_points:
        rows = []
        for idx, item in enumerate(live_points, 1):
            line = (
                f"[测点 {idx}] {item.get('description') or item.get('query')}"
                f"（KKS={item.get('kks') or '未知'}） = {item.get('value')} {item.get('unit')}"
                f"  采集时间 {item.get('time')}（北京时间）"
            )
            drift = item.get("drift") or {}
            if drift:
                unit = item.get("unit") or ""
                line += (
                    f"\n    近期对比：过去 {drift.get('window_minutes')} 分钟均值 "
                    f"{drift.get('mean'):.6g} {unit}（{drift.get('samples')} 个采样点），"
                    f"当前偏离 {drift.get('delta'):+.6g} {unit}"
                    f"（{drift.get('rel_percent'):.1f}%，{drift.get('sigma'):.1f}σ）"
                )
                # 只有两个判据都超标才提示偏离；否则明确说「在正常波动内」，
                # 免得模型看到数字就自己加戏
                line += (
                    "  → **偏离显著，值得关注**"
                    if drift.get("drifted")
                    else "  → 在近期正常波动范围内"
                )
            rows.append(line)
        live_block = "\n\n实时测点数据：\n" + "\n".join(rows)

    # 历史序列与趋势单独成块。**不下发全部采样点**——单次实测 644 点，
    # 全量塞进上下文会把文档证据挤掉，而模型需要的是「变了多少、往哪变」，
    # 不是每一个采样值本身。这里给统计摘要 + 等距抽稀后的序列。
    series_block = ""
    if series:
        chunks = []
        for idx, item in enumerate(series, 1):
            label = "趋势" if item.get("kind") == "trend" else "历史序列"
            samples = item.get("samples") or []
            feat = _series_features(samples)
            # 整段显著性：判为「无显著变化」时**不喂关键点和突变点**，
            # 从源头上不给模型解读噪声的材料。出图仍照给——图上看得出全貌。
            values = [s.get("value") for s in samples if s.get("value") is not None]
            sig = _series_significance(values, item.get("limits"))
            # 只有 level 2（≥3%）才给关键点。1%~3% 那一档连点都不给——
            # 给了点，模型就会去逐点解读，那正是要避免的。
            picks = _compress_series(samples, _SERIES_MAX_POINTS) if sig.get("level") == 2 else []
            body = "、".join(
                f"{_format_live_time(sample.get('time'))} {sample.get('value')}"
                for sample in picks
                if sample.get("value") is not None
            ) or "无"

            # 特征摘要放在点序列**之前**：它是结论性信息（斜率、r²、拐点），
            # 点序列只是佐证。模型先拿到结论，再看点，比反过来更容易用对。
            lines = [
                f"[{label} {idx}] {item.get('description') or item.get('query')}"
                f"（KKS={item.get('kks') or '未知'}，单位 {item.get('unit') or '未知'}）",
                f"  区间 {_format_live_time(item.get('start'))} ~ {_format_live_time(item.get('end'))}"
                f"，共 {len(samples)} 个采样点"
                f"（采集间隔 {item.get('interval_seconds') or '原始'} 秒）",
            ]
            if sig.get("comparable"):
                scale = []
                if sig.get("rel_reading") is not None and sig["rel_reading"] < float("inf"):
                    scale.append(f"占读数 {sig['rel_reading']:.2%}")
                if sig.get("rel_range") is not None:
                    scale.append(f"占工程量程 {sig['rel_range']:.2%}")
                lines.append(
                    f"  波动幅度：P5={sig['p5']:.6g}，P95={sig['p95']:.6g}，"
                    f"中位 {sig['median']:.6g}"
                    + (f"（{'，'.join(scale)}）" if scale else "")
                )
                if sig.get("discrete"):
                    # 图上看得最清楚：取值在两个固定值之间规律跳动。
                    # 不点破的话，模型会把采集问题当成设备异常来解读。
                    lines.append(
                        f"  数据质量：**取值只集中在 {sig.get('distinct_values')} 个固定值上**"
                        f"（{sig.get('samples')} 个采样点），符合采集侧丢包或取整的特征。"
                        f"（作答要求：据此说明数据质量，说明趋势，并建议核对采集链路；"
                        f"不要把它写成设备异常、突变或工况变化。）"
                    )

            level = sig.get("level", 2)
            if level == 0:
                # 完全隐掉：不给斜率、不给突变点、不给关键点。
                # 模型没有材料可解读，自然不会把 0.6% 的抖动写成「短暂下探」。
                lines.append(
                    "  波动判定：**低于显著变化阈值**。"
                    "（作答要求：直接给「基本无变动」的结论即可，不要逐点罗列数值。）"
                )
            elif level == 1:
                # 中间档：图和统计都给，但不给可解读的「事件」。
                lines.append(
                    "  波动判定：处于**低位区间**，不构成值得单独分析的变化。"
                    "（作答要求：概述整体走势并给出统计值即可，"
                    "不要逐点罗列、不要定性为异常或突变。）"
                )
            elif feat:
                unit = item.get("unit") or ""
                lines.append(
                    f"  整体趋势：{feat.get('direction')}，"
                    f"斜率 {feat.get('slope_per_minute'):+.4f} {unit}/分钟，"
                    f"线性拟合 R²={feat.get('r2'):.3f}"
                )
                turns = feat.get("turning_points") or []
                for turn in turns:
                    # 电平差是判断「是否真发生了状态变化」的主依据；
                    # 段内 r² 只描述那一段平不平，**不是拐点的可信度**——
                    # 压力从有到无之后必然是平直段，r² 低恰恰是跃迁完成的证据。
                    lines.append(
                        f"  突变点：{turn.get('time')}，"
                        f"均值由 {turn.get('before_mean'):.6g} 变为 {turn.get('after_mean'):.6g}"
                        f"（{turn.get('rate_sign')} {abs(turn.get('level_delta')):.6g} {unit}，"
                        f"相对变化 {turn.get('level_percent'):.1f}%，"
                        f"占全程量程 {turn.get('level_ratio'):.0%}）"
                        f"；之后 {turn.get('after_points')} 点维持"
                        f"{'平直' if turn.get('after_flat') else '波动'}"
                    )
                if turns:
                    lines.append(
                        "  说明：突变点的「占全程量程」比例越高，越是一次真实的状态变化；"
                        "段内是否平直只描述该段形态，不构成对突变本身的否定。"
                    )
                rate = feat.get("max_rate") or {}
                if rate:
                    lines.append(
                        f"  最大变化率：{rate.get('per_minute'):+.4f} {unit}/分钟"
                        f"（出现在 {rate.get('time')}）"
                    )
                lines.append(
                    f"  数值范围：最小 {feat.get('min')}，最大 {feat.get('max')}，"
                    f"最新 {feat.get('latest')}，全程变化 {feat.get('delta'):+.4f}"
                )
            # flat 时 picks 为空，这一行会写成「0 点：无」——同样是噪声，跳过
            if picks:
                lines.append(f"  压缩后的关键点（{len(picks)} 点）：{body}")
            chunks.append("\n".join(lines))
        series_block = "\n\n测点序列数据：\n" + "\n".join(chunks)

    # 运行日志单独成块。只有对方明确问运行记录时才会走到这里。
    logs_block = ""
    if logs:
        summary = logs.get("summary") or {}
        events = logs.get("events") or []
        categories = "、".join(
            f"{name} {count} 条" for name, count in (summary.get("categories") or {}).items()
        ) or "无"
        lines = []
        for idx, event in enumerate(events, 1):
            severity = str(event.get("severity") or "")
            mark = "【重大】" if severity == "major" else ("【重要】" if severity == "important" else "")
            lines.append(
                f"[日志 {idx}] {mark}{event.get('time') or event.get('date') or ''} "
                f"{event.get('source') or ''}／{event.get('category') or ''}："
                f"{str(event.get('content') or '').strip()}"
            )
        logs_block = (
            f"\n\n运行日志（{logs.get('start')} ~ {logs.get('end')}"
            f"{'，仅重大事件' if logs.get('major_only') else ''}）：\n"
            f"  合计 {summary.get('total_events', len(events))} 条，"
            f"其中重大 {summary.get('major_events', 0)} 条；分类：{categories}\n"
            + "\n".join(lines)
        )

    # 未检索时不拼证据段落：一旦出现「检索证据：无」这类框架，模型会顺着去说
    # 「证据中没有」，而不是直接依据自身设定回答。
    # 实时数据块合并：测点当前值、历史/趋势序列、运行日志三者并列。
    # 它们都是「当前系统的事实」而非文档证据，拼在同一段里便于模型对照。
    realtime_block = live_block + series_block + logs_block
    has_realtime = bool(live_points or series or logs)

    if skip_retrieval:
        user = f"问题：{question}{realtime_block}"
        system_body = (
            "本轮未检索文档知识库。"
            + (
                "请基于下面给出的实时数据回答，并说明数据采集时间。"
                if has_realtime
                else "请直接依据自身设定回答，不要提及检索、证据或知识库。"
            )
            # 不加这句模型会为了「答得完整」编造上下文长度之类的配置数值，
            # 比「证据中没有」更糟。
            + "涉及本系统的具体配置数值（上下文长度、端口、模型参数等）时，"
            "不确知就直说无法确认，不要给出估计值、示例值或常见默认值。"
        )
    else:
        user = (
            f"问题：{question}{realtime_block}"
            f"\n\n检索证据：\n{evidence}\n\n知识图谱上下文：\n{graph}"
        )
        system_body = (
            "回答时先看检索证据和知识图谱上下文；有证据就基于证据和文档结构归纳，不编造来源。"
            "无证据时，可以回答模型身份、系统能力、通用操作类问题；"
            "涉及工业规程、安全边界、设备参数时必须说明缺少证据，不能臆造。"
            "这是工业场景，不要为了简短省略证据中的职责、步骤、条件、例外、参数或安全后果。"
            "必须逐条阅读并综合所有列出的证据和图谱上下文，不能只依据第一条；"
            "按主题合并重复内容，区分正常运行、启停检查和异常处理，"
            "根据资料量输出完整、可执行、层次清楚的回答。"
        )

    prompt = [
        {
            "role": "system",
            "content": (
                f"你是 {model_name or settings.llm_model}，HyperDriveWave 工业知识问答系统以你作为底座大模型。"
                f"{system_body}"
                + (
                    "实时测点的采集时间已换算为北京时间，照抄即可，"
                    "不要附加时区后缀、不要另做时区换算或补充说明。"
                    if live_points
                    else ""
                )
                + "禁止使用表情符号或 emoji。"
                # 前端已接入 Markdown 渲染（见 industrial-webui/lib/markdown.js），
                # 所以标题、列表、加粗都会正确排版，不再是字面的符号。
                # 这里的取舍是「结构跟着内容走」，不是一刀切地禁或放。
                "\n\n【作答形态】按内容决定形态，结构跟着内容走，不要一刀切。\n"
                "· 一问一答能说清的（某个值是多少、某件事有没有发生）：直接一段话讲完，"
                "不要分节也不要加标题。\n"
                "· 内容确实并列时（操作步骤、参数清单、多个独立事项、逐条对比）："
                "用有序或无序列表，让每一条独立成行，不要挤在一段里。\n"
                "· 内容跨越多个主题、篇幅较长时：用二级或三级标题分节（最多三级），"
                "让读者能跳读。\n"
                "· 关键数值、结论、风险提示用粗体标出，但不要整句加粗、更不要每行都加粗。\n"
                "· 需要强调层级时可用表格；纯叙述不要用表格。\n"
                "总的原则：**结构是为了让读者更快找到信息，不是为了显得整齐**。"
                "把一句完整的话拆成几个列表项，或者给三行内容加四个标题，都是反面例子。"
            ),
        },
    ]
    prompt.extend(history_messages or [])
    if image_attachments:
        content: list[dict[str, str]] = [{"type": "text", "text": user}]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": image["data_url"]},
            }
            for image in image_attachments
        )
        prompt.append({"role": "user", "content": content})
    else:
        prompt.append({"role": "user", "content": user})
    return prompt


async def _llm_answer(
    question: str,
    contexts: list[dict[str, Any]],
    graph_rows: list[dict[str, Any]],
    reasoning_effort: str | None = None,
    *,
    inference_mode: str | None = None,
    history_messages: list[dict[str, str]] | None = None,
    existing_summary: str = "",
    existing_kept_from: int = 0,
    previous_inference_mode: str | None = None,
    image_attachments: list[dict[str, str]] | None = None,
    allow_online: bool = True,
    skip_retrieval: bool = False,
    live_points: list[dict[str, Any]] | None = None,
    series: list[dict[str, Any]] | None = None,
    logs: dict[str, Any] | None = None,
) -> tuple[str, str, Any, str | None, dict[str, Any]]:
    if image_attachments:
        candidates, capability_errors = await _vision_candidates(
            allow_online=allow_online,
        )
        if not candidates:
            detail = "；".join(capability_errors) or "没有配置可用的视觉模型"
            raise HTTPException(status_code=503, detail=f"多模态不可用：{detail}")
    else:
        selected_mode, client = _llm_client(inference_mode)
        if selected_mode == "online" and not allow_online:
            raise HTTPException(
                status_code=403,
                detail="在线推理仅管理员可用，请联系管理员开放在线推理",
            )
        runtime_context_window = _context_window(selected_mode)
        if selected_mode == "offline":
            local_runtime = await _local_runtime_info(client.base_url)
            if local_runtime["available"] and local_runtime["model"]:
                client = _llm_client(
                    selected_mode,
                    model_override=local_runtime["model"],
                    base_url_override=client.base_url,
                )[1]
            if local_runtime["context_window"]:
                runtime_context_window = int(local_runtime["context_window"])
        candidates = [{
            "mode": selected_mode,
            "model": client.model,
            "base_url": client.base_url,
            "context_window": runtime_context_window,
        }]
        capability_errors = []

    errors: list[str] = []
    for candidate in candidates:
        selected_mode = _resolve_inference_mode(str(candidate["mode"]))
        try:
            _, client = _llm_client(
                selected_mode,
                model_override=str(candidate["model"]),
                base_url_override=str(candidate["base_url"]),
            )
            base_prompt = _prompt(
                question,
                contexts,
                graph_rows,
                model_name=client.model,
                image_attachments=image_attachments,
                skip_retrieval=skip_retrieval,
                live_points=live_points,
                series=series,
                logs=logs,
            )
            target_context_window = int(
                candidate.get("context_window") or _context_window(selected_mode)
            )
            compression_provider: str | None = None
            if _needs_cross_mode_compression(
                previous_inference_mode,
                selected_mode,
            ):
                compression_mode: Literal["online", "offline"] = "offline"
                compression_client = client
                if allow_online:
                    # The online provider handles the oversized history before
                    # the local 256K model sees it.
                    compression_mode = "online"
                    compression_client = _llm_client("online")[1]
                manager = ContextManager(
                    window_tokens=target_context_window,
                    threshold=settings.context_compression_threshold,
                    target=settings.context_compression_target,
                )

                async def summarize_cross_mode(messages: list[dict[str, str]]) -> str:
                    return await _compress_history(
                        compression_client,
                        messages,
                        mode=compression_mode,
                    )

                preparation = await manager.prepare(
                    history_messages or [],
                    existing_summary=existing_summary,
                    existing_kept_from=existing_kept_from,
                    fixed_messages=base_prompt,
                    summarize=summarize_cross_mode,
                )
                if preparation.compressed:
                    compression_provider = compression_mode
            else:
                manager = ContextManager(
                    window_tokens=target_context_window,
                    threshold=settings.context_compression_threshold,
                    target=settings.context_compression_target,
                )

                async def summarize(messages: list[dict[str, str]]) -> str:
                    return await _compress_history(
                        client,
                        messages,
                        mode=selected_mode,
                    )

                preparation = await manager.prepare(
                    history_messages or [],
                    existing_summary=existing_summary,
                    existing_kept_from=existing_kept_from,
                    fixed_messages=base_prompt,
                    summarize=summarize,
                )
                if preparation.compressed:
                    compression_provider = selected_mode
            thinking_enabled = _thinking_enabled(selected_mode)
            completion_reasoning_effort = (
                reasoning_effort or settings.llm_reasoning_effort
            ) if thinking_enabled else None
            chat_template_kwargs: dict[str, Any] | None = None
            thinking_budget_tokens: int | None = None
            if selected_mode == "offline":
                if not thinking_enabled:
                    chat_template_kwargs = {"enable_thinking": False}
                elif completion_reasoning_effort == "low":
                    # Qwen's low effort still enters a long thinking loop. Fast
                    # local replies must explicitly disable thinking in the template.
                    completion_reasoning_effort = None
                    chat_template_kwargs = {"enable_thinking": False}
                elif completion_reasoning_effort == "medium":
                    thinking_budget_tokens = 1024
                elif completion_reasoning_effort == "high":
                    completion_reasoning_effort = "xhigh"
                    thinking_budget_tokens = 4096
            prompt = [base_prompt[0], *preparation.messages, *base_prompt[1:]]
            result = await client.complete(
                prompt,
                temperature=0.2,
                reasoning_effort=completion_reasoning_effort,
                max_tokens=settings.llm_max_tokens,
                chat_template_kwargs=chat_template_kwargs,
                thinking_budget_tokens=thinking_budget_tokens,
                thinking_enabled=(
                    thinking_enabled if selected_mode == "online" else None
                ),
            )
            return (
                result.content,
                selected_mode,
                preparation,
                compression_provider,
                {
                    "mode": selected_mode,
                    "model": client.model,
                    "vision": bool(image_attachments),
                    "requested_mode": (
                        _resolve_inference_mode(inference_mode)
                        if inference_mode
                        else selected_mode
                    ),
                    "vision_fallback": bool(
                        image_attachments
                        and inference_mode
                        and selected_mode != _resolve_inference_mode(inference_mode)
                    ),
                    "attempts": len(errors) + 1,
                    "context_window": target_context_window,
                    "thinking_enabled": thinking_enabled,
                },
            )
        except Exception as exc:
            errors.append(f"{selected_mode}/{candidate['model']}: {exc}")
            if not image_attachments:
                raise HTTPException(status_code=503, detail=f"LLM generation failed: {exc}") from exc

    detail = "；".join(errors or capability_errors) or "视觉模型请求失败"
    raise HTTPException(status_code=503, detail=f"多模态不可用：{detail}")


async def _answer(
    question: str,
    contexts: list[dict[str, Any]],
    reasoning_effort: str | None = None,
    *,
    inference_mode: str | None = None,
    history_messages: list[dict[str, str]] | None = None,
    conversation_id: str | None = None,
    user_code: str | None = None,
    conversation_data: dict[str, Any] | None = None,
    image_attachments: list[dict[str, str]] | None = None,
    allow_online: bool = True,
    skip_retrieval: bool = False,
    live_points: list[dict[str, Any]] | None = None,
    series: list[dict[str, Any]] | None = None,
    logs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected_mode = _resolve_inference_mode(inference_mode)
    llm_contexts = _llm_contexts(selected_mode, contexts)
    graph_rows = await _graph_context(llm_contexts)
    data = conversation_data or {}
    answer, selected_mode, preparation, compression_provider, llm_info = await _llm_answer(
        question,
        llm_contexts,
        graph_rows,
        reasoning_effort,
        inference_mode=selected_mode,
        history_messages=history_messages,
        existing_summary=str(data.get("context_summary") or ""),
        existing_kept_from=int(data.get("context_kept_from") or 0),
        previous_inference_mode=str(data.get("last_inference_mode") or "") or None,
        image_attachments=image_attachments,
        allow_online=allow_online,
        skip_retrieval=skip_retrieval,
        live_points=live_points,
        series=series,
        logs=logs,
    )
    context_meta = {
        "compressed": preparation.compressed,
        "tokens_before": preparation.tokens_before,
        "tokens_after": preparation.tokens_after,
        "persistent_tokens_after": max(
            0,
            preparation.tokens_after - preparation.image_token_estimate,
        ),
        "image_count": preparation.image_count,
        "image_bytes": preparation.image_bytes,
        "image_token_estimate": preparation.image_token_estimate,
        "image_base64_in_text_tokens": False,
        "summary": preparation.summary,
        "kept_from": preparation.kept_from,
        "window_tokens": int(
            llm_info.get("context_window") or _context_window(selected_mode)
        ),
        "compression_provider": compression_provider,
        "previous_inference_mode": str(data.get("last_inference_mode") or "") or None,
        "last_inference_mode": selected_mode,
        "mode_switched": (
            str(data.get("last_inference_mode") or "") in {"online", "offline"}
            and str(data.get("last_inference_mode") or "") != selected_mode
        ),
        "retrieved_evidence_count": len(contexts),
        "llm_evidence_count": len(llm_contexts),
    }
    if user_code and conversation_id:
        with _conversation_lock:
            stored = _read_conversation(user_code, conversation_id)
            if preparation.compressed:
                stored["context_summary"] = preparation.summary
                stored["context_kept_from"] = preparation.kept_from
                stored["context_token_estimate"] = max(
                    0,
                    preparation.tokens_after - preparation.image_token_estimate,
                )
                stored["context_compressed_at"] = _now_iso()
            stored["last_inference_mode"] = selected_mode
            _write_conversation(user_code, conversation_id, stored)
    if contexts:
        return {
            "answer": answer,
            "question": question,
            "citations": llm_contexts,
            "graph_context": graph_rows,
            "status": "answered_by_llm_with_retrieval",
            "llm": llm_info,
            "context": context_meta,
        }

    return {
        "answer": answer,
        "question": question,
        "citations": contexts,
        "graph_context": graph_rows,
        "status": "answered_by_llm",
        "llm": llm_info,
        "context": context_meta,
    }


@app.get("/model-config")
async def get_model_config(x_hdw_session: str | None = Header(default=None)) -> dict[str, Any]:
    user = _current_user(x_hdw_session)
    config = _safe_model_config(_read_model_config())
    local_runtime = await _local_runtime_info(str(config["local"]["base_url"]))
    config["local"]["actual_multimodal"] = bool(local_runtime["vision"])
    config["local"]["capability_detail"] = str(local_runtime["detail"])
    config["local"]["runtime_available"] = bool(local_runtime["available"])
    config["local"]["runtime_model"] = str(local_runtime["model"])
    config["local"]["runtime_context_window"] = int(local_runtime["context_window"] or 0)
    config["online"]["api_key_configured"] = bool(settings.online_llm_api_key)
    return {
        "config": config,
        "user": {"code": user["code"], "role": user["role"]},
        "permissions": _permissions_for_user(user),
        "runtime": {
            "online_model_applies_immediately": True,
            "local_engine_and_model_apply": "saved and applied by restarting the local inference service",
            "local": {
                "available": bool(local_runtime["available"]),
                "engine": str(local_runtime.get("engine") or ""),
                "loaded_model": str(local_runtime["model"]),
                "context_window": int(local_runtime["context_window"] or 0),
                "detail": str(local_runtime["detail"]),
            },
            "api_key_exposed": False,
        },
    }


@app.patch("/model-config")
async def patch_model_config(
    patch: ModelConfigPatch,
    x_hdw_session: str | None = Header(default=None),
) -> dict[str, Any]:
    user = _current_user(x_hdw_session)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="admin permission required")
    _validate_model_config_patch(patch)
    previous = _read_model_config()
    current = copy.deepcopy(previous)
    incoming = patch.model_dump(exclude_none=True)
    for section in ("local", "online", "retrieval"):
        if isinstance(incoming.get(section), dict):
            current[section].update(incoming[section])
    if isinstance(incoming.get("vision_priority"), list):
        current["vision_priority"] = incoming["vision_priority"]

    local_update = incoming.get("local")
    # mtp_enabled 是 llama-server 的启动参数（--spec-type draft-mtp），改动它必须重启本地推理。
    local_runtime_fields = {"model", "engine", "context_window", "mtp_enabled"}
    needs_local_restart = (
        isinstance(local_update, dict)
        and bool(local_runtime_fields.intersection(local_update))
    )
    rollback_config = copy.deepcopy(previous)
    if needs_local_restart:
        # Keep a usable rollback even when the JSON was previously stale.
        runtime = await _local_runtime_info(
            str(current["local"].get("base_url") or settings.local_llm_base_url)
        )
        if runtime["available"] and runtime["model"]:
            rollback_config["local"]["model"] = runtime["model"]
            rollback_config["local"]["engine"] = (
                runtime.get("engine") or "llama.cpp"
            )
            if runtime["context_window"]:
                rollback_config["local"]["context_window"] = runtime["context_window"]

    _write_model_config(current)
    apply_result: dict[str, Any] | None = None
    if needs_local_restart:
        try:
            apply_result = await _maintenance_request("/switch-llm")
        except Exception as exc:
            _write_model_config(rollback_config)
            rollback_error = ""
            try:
                await _maintenance_request("/switch-llm")
            except Exception as rollback_exc:
                rollback_error = f"；旧模型恢复失败：{rollback_exc}"
            raise HTTPException(
                status_code=503,
                detail=f"本地模型切换失败，配置已回滚{rollback_error}：{exc}",
            ) from exc

    response = await get_model_config(x_hdw_session)
    if apply_result is not None:
        response["apply"] = apply_result
    return response


def _frp_public_entry() -> dict[str, str]:
    host = settings.frp_public_host
    if not host:
        return {"http": "", "https": ""}
    return {"http": f"http://{host}:8080", "https": f"https://{host}:8443"}


@app.get("/frp")
async def get_frp(x_hdw_session: str | None = Header(default=None)) -> dict[str, Any]:
    user = _current_user(x_hdw_session)
    try:
        state = await _maintenance_request("/frp", method="GET")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"资源协调器不可用：{exc}") from exc
    return {
        "state": state,
        "public": _frp_public_entry(),
        "user": {"code": user["code"], "role": user["role"]},
        "permissions": _permissions_for_user(user),
    }


@app.patch("/frp")
async def patch_frp(
    patch: FrpPatch,
    x_hdw_session: str | None = Header(default=None),
) -> dict[str, Any]:
    user = _current_user(x_hdw_session)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="admin permission required")
    try:
        await _maintenance_request("/frp-enable" if patch.enabled else "/frp-disable")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"切换外网访问失败：{exc}") from exc
    return await get_frp(x_hdw_session)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "rag": await _rag_health(),
        "graph": await _graph_health(),
        "llm": await _llm_health(),
    }


@app.post("/auth/login")
async def login(req: LoginRequest) -> dict[str, Any]:
    try:
        user, session = auth_store.login(req.code)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail="invalid login code") from exc
    return {
        "session": session,
        "user": user,
        "permissions": _permissions_for_user(user),
    }


@app.get("/auth/me")
async def auth_me(x_hdw_session: str | None = Header(default=None)) -> dict[str, Any]:
    user = _current_user(x_hdw_session)
    return {"user": user, "permissions": _permissions_for_user(user)}


@app.get("/conversations")
async def list_conversations(
    authorization: str | None = Header(default=None),
    x_hdw_session: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_auth(authorization)
    user = _current_user(x_hdw_session)
    _ensure_chatdata_root(user["code"])
    conversations: list[dict[str, Any]] = []
    try:
        with _conversation_lock:
            for path in _user_chatdata_root(user["code"]).glob("*.json"):
                conversation_id = path.stem
                if not _CONVERSATION_ID_RE.fullmatch(conversation_id):
                    continue
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(data, dict):
                    conversations.append(_conversation_summary(conversation_id, data))
    except OSError as exc:
        raise HTTPException(status_code=503, detail=f"chatdata unavailable: {exc}") from exc
    conversations.sort(
        key=lambda item: item["updated_at"] or item["created_at"] or "",
        reverse=True,
    )
    conversations.sort(key=lambda item: not item["pinned"])
    return {"conversations": conversations}


@app.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    authorization: str | None = Header(default=None),
    x_hdw_session: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_auth(authorization)
    user = _current_user(x_hdw_session)
    with _conversation_lock:
        return _read_conversation(user["code"], conversation_id)


@app.put("/conversations/{conversation_id}")
async def put_conversation(
    conversation_id: str,
    payload: ConversationPayload,
    authorization: str | None = Header(default=None),
    x_hdw_session: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_auth(authorization)
    user = _current_user(x_hdw_session)
    with _conversation_lock:
        existing: dict[str, Any] = {}
        path = _conversation_path(user["code"], conversation_id)
        if path.is_file():
            existing = _read_conversation(user["code"], conversation_id)
        conversation = payload.model_dump(mode="json")
        if conversation.get("last_inference_mode") is None:
            conversation["last_inference_mode"] = existing.get("last_inference_mode")
        now = _now_iso()
        conversation["created_at"] = conversation["created_at"] or now
        conversation["updated_at"] = now
        return _write_conversation(user["code"], conversation_id, conversation)


@app.patch("/conversations/{conversation_id}")
async def patch_conversation(
    conversation_id: str,
    payload: ConversationPatch,
    authorization: str | None = Header(default=None),
    x_hdw_session: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_auth(authorization)
    user = _current_user(x_hdw_session)
    with _conversation_lock:
        conversation = _read_conversation(user["code"], conversation_id)
        changes = payload.model_dump(exclude_unset=True)
        conversation.update(changes)
        conversation["updated_at"] = _now_iso()
        return _write_conversation(user["code"], conversation_id, conversation)


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    authorization: str | None = Header(default=None),
    x_hdw_session: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_auth(authorization)
    user = _current_user(x_hdw_session)
    path = _conversation_path(user["code"], conversation_id)
    with _conversation_lock:
        try:
            path.unlink()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="conversation not found") from exc
        except OSError as exc:
            raise HTTPException(status_code=503, detail=f"conversation delete failed: {exc}") from exc
    return {"deleted": True, "id": conversation_id}


@app.get("/v1/models")
async def list_models(
    authorization: str | None = Header(default=None),
    x_hdw_session: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_auth(authorization)
    _current_user(x_hdw_session)
    return {
        "object": "list",
        "data": [
            {
                "id": "hyperdrivewave-industrial-qa",
                "object": "model",
                "owned_by": "hyperdrivewave",
            }
        ],
    }


@app.post("/qa/query")
async def qa_query(
    req: QARequest,
    authorization: str | None = Header(default=None),
    x_hdw_session: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_auth(authorization)
    user = _current_user(x_hdw_session)
    image_attachments = _validate_image_attachments(req.images)
    question = req.question.strip() or ("请分析这张图片。" if image_attachments else "")
    if not question:
        raise HTTPException(status_code=422, detail="question is required unless an image is attached")
    selected_mode = _resolve_inference_mode(
        req.inference_mode or _user_default_inference_mode(user)
    )
    if selected_mode == "online" and not _online_inference_allowed(user):
        raise HTTPException(
            status_code=403,
            detail="在线推理仅管理员可用，请联系管理员开放在线推理",
        )
    contexts, rag_info = await _retrieve_planned(question, selected_mode, req.top_k)
    history, conversation_data = _conversation_messages(user["code"], req.conversation_id, req.messages)
    if history and history[-1]["role"] == "user" and history[-1]["content"] == question:
        history = history[:-1]
    result = await _answer(
        question,
        contexts,
        req.reasoning_effort,
        inference_mode=selected_mode,
        history_messages=history,
        conversation_id=req.conversation_id,
        user_code=user["code"],
        conversation_data=conversation_data,
        image_attachments=image_attachments,
        allow_online=_online_inference_allowed(user),
        skip_retrieval=bool(rag_info.get("no_retrieval")),
        live_points=rag_info.get("live_points") or [],
        series=rag_info.get("series") or [],
        logs=rag_info.get("logs"),
    )
    result["rag"] = rag_info
    return result


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    req: ChatCompletionRequest,
    authorization: str | None = Header(default=None),
    x_hdw_session: str | None = Header(default=None),
) -> dict[str, Any] | StreamingResponse:
    _check_auth(authorization)
    _current_user(x_hdw_session)
    question = _last_user_message(req.messages)
    result = await qa_query(
        QARequest(
            question=question,
            inference_mode=req.inference_mode,
            conversation_id=req.conversation_id,
            messages=req.messages,
            images=req.images,
        ),
        authorization,
        x_hdw_session,
    )
    content = result["answer"]
    if req.stream:
        async def chunks():
            payload = {"choices": [{"delta": {"content": content}}]}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(chunks(), media_type="text/event-stream")
    return {
        "id": f"hdw-local-{result['status']}",
        "object": "chat.completion",
        "model": req.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "context": result.get("context", {}),
        "llm": result.get("llm", {}),
        "rag": result.get("rag", {}),
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _self_check() -> None:
    assert _last_user_message([ChatMessage(role="user", content="hello")]) == "hello"
    assert _prompt("q", [])[1]["content"].startswith("问题：q")
    assert "知识图谱上下文" in _prompt("q", [])[1]["content"]
    prompt = _prompt("q", [{"text": "a"}, {"text": "b"}])
    assert "[证据 2]" in prompt[-1]["content"]
    full_prompt = _prompt("q", [{"text": "a" * 1000}])
    assert "a" * 1000 in full_prompt[-1]["content"]
    # 规划解析：检索无法判定返回 None（调用方保守按「需要检索」），测点无法判定返回空列表
    # 规划解析：返回结构化意图字典。检索无法判定为 None（调用方保守按「需要检索」），
    # 各项无法判定为 None 或空列表。
    _p = _parse_plan("检索: yes\n测点: 无")
    assert _p["needs_rag"] is True and _p["points"] == []
    _p = _parse_plan("检索: no\n测点: 主蒸汽温度")
    assert _p["needs_rag"] is False and _p["points"] == ["主蒸汽温度"]
    _p = _parse_plan("检索: yes\n测点: 闭式冷却水泵、凝结水压力")
    assert _p["points"] == ["闭式冷却水泵", "凝结水压力"]
    assert _parse_plan("检索: yes\n测点: a、b、c、d")["points"] == ["a", "b", "c"]  # 上限 3 个
    assert _parse_plan("检索：否\n测点：None")["points"] == []
    _p = _parse_plan("随便说点什么")
    assert _p["needs_rag"] is None and _p["points"] == []
    # 历史／趋势／日志三项按语意解析，日期写成 ISO，间隔与重大标记可识别
    _p = _parse_plan(
        "检索: yes\n测点: 无\n历史: 无\n"
        "趋势: 汽包水位|2026-09-12|2026-09-13|300\n日志: 2026-09-12|2026-09-12|重大"
    )
    assert _p["trend"]["keyword"] == "汽包水位"
    assert _p["trend"]["start"] == "2026-09-12" and _p["trend"]["end"] == "2026-09-13"
    assert _p["trend"]["interval_seconds"] == 300
    assert _p["logs"]["major_only"] is True
    assert _p["history"] is None
    # 旧的两行格式必须仍然可用：模型漏输出新行时不能让整条链路退化
    _p = _parse_plan("检索: yes\n测点: 给水流量")
    assert _p["points"] == ["给水流量"] and _p["history"] is None and _p["logs"] is None
    # 区间越界与起止颠倒都要被收口，不能让模型给出的日期直接进查询
    _p = _parse_plan("检索: no\n测点: 无\n历史: 无\n趋势: 无\n日志: 2020-01-01|2026-09-13")
    assert (
        date.fromisoformat(_p["logs"]["end"]) - date.fromisoformat(_p["logs"]["start"])
    ).days < _QUERY_MAX_DAYS
    _p = _parse_plan("检索: no\n测点: 无\n历史: 无\n趋势: 无\n日志: 2026-09-13|2026-09-10")
    assert _p["logs"]["start"] <= _p["logs"]["end"]
    # 序列抽稀必须保留首尾两点——它们最能说明「现在处于什么水平」
    _samples = [{"time": f"t{i}", "value": i} for i in range(644)]
    _thin = _thin_samples(_samples, _SERIES_MAX_POINTS)
    assert len(_thin) == _SERIES_MAX_POINTS
    assert _thin[0]["value"] == 0 and _thin[-1]["value"] == 643

    # ── 趋势特征提取 ──
    def _mk(values, step=60):
        base = datetime(2026, 9, 13, 0, 0, 0, tzinfo=timezone.utc)
        return [
            {"time": (base + timedelta(seconds=step * i)).isoformat(), "value": float(v)}
            for i, v in enumerate(values)
        ]

    # 单调上升：斜率应为正、r² 接近 1、不报拐点
    _f = _series_features(_mk([float(i) for i in range(40)]))
    assert _f["slope_per_minute"] > 0 and _f["r2"] > 0.99, _f
    assert _f["direction"] == "上升"
    assert "turning_point" not in _f  # 单调段报拐点只会添噪声
    assert _f["max_rate"]["per_minute"] > 0

    # 先降后升：整体斜率接近 0 但 r² 很低，必须报出突变点，
    # 否则模型会只看到「平稳」而漏掉「已经转向」
    _v = [100.0 - i for i in range(20)] + [80.0 + i for i in range(20)]
    _f = _series_features(_mk(_v))
    assert _f["r2"] < _TREND_TURNING_R2, _f
    assert _f.get("turning_points"), _f
    _t = _f["turning_points"][0]
    assert _t["rate_sign"] == "升", _t
    # V 形回升会把「前后均值差」摊薄，用不到满量程——0.3 已是明确信号
    assert _t["level_delta"] > 0 and _t["level_ratio"] > 0.3, _t

    # **压力从有到无**：这是本项目最典型的真实跃迁——均值大幅下移、
    # 之后长期平直。段内 r² 会很低，但那恰恰是跃迁完成的证据，不能据此否定它。
    _v = [0.018] * 30 + [0.0] * 30
    _f = _series_features(_mk(_v))
    assert _f.get("turning_points"), "压力消失这种真实跃迁必须被报出来"
    _t = _f["turning_points"][0]
    assert _t["rate_sign"] == "降", _t
    assert _t["level_ratio"] > 0.9, _t          # 电平差几乎等于全程量程
    assert _t["after_flat"] is True, _t         # 之后是平直段——这是特征，不是缺陷

    # 完全平直：方向为「基本平稳」，不报突变点
    _f = _series_features(_mk([5.0] * 30))
    assert _f["direction"] == "基本平稳", _f
    assert not _f.get("turning_points"), _f

    # 纯噪声抖动不该被报成突变：电平差达不到量程比例
    _v = [10.0 + (0.01 if i % 2 else -0.01) for i in range(40)]
    _f = _series_features(_mk(_v))
    assert not _f.get("turning_points"), f"噪声被误报为突变: {_f.get('turning_points')}"

    # **高占比但低百分比的微动**：真实数据集里 0.0181727→0.01829 只变了 0.6%，
    # 但因为当时量程本身就极小，它占了量程的 24%。只看占比会把它当事件，
    # 模型就会去解读一个物理上不存在的状态变化。
    _v = [0.018 + (0.0001 if 20 <= i < 40 else 0) for i in range(60)]
    _f = _series_features(_mk(_v))
    _turns = _f.get("turning_points") or []
    assert not _turns, f"0.6% 的微动被误报为突变: {_turns}"

    # 对照：同样占比、但物理量级上确有意义的跃迁，必须报出来
    _v = [0.018] * 30 + [0.0] * 30
    _f = _series_features(_mk(_v))
    assert _f.get("turning_points"), "压力消失这类跃迁不能因为百分比口径而被滤掉"
    assert _f["turning_points"][0]["level_percent"] > 99.0

    # ── 整段显著性：趋势场景宁可多报 ──
    # 分位数：单点毛刺不该把 P95-P5 撑大
    assert _percentile([1, 2, 3, 4, 5], 0.5) == 3.0
    assert _percentile([1, 2, 3, 4], 0.0) == 1.0 and _percentile([1, 2, 3, 4], 1.0) == 4.0
    assert _percentile([], 0.5) == 0.0

    # 实测那条的真实幅度：0.017842~0.018338，极差 0.000496，占读数 2.8%
    # → 落在中间档（1%~3%）：给图和统计，但不展开斜率与突变点
    _v = [0.018338 if i % 7 else 0.017842 for i in range(60)]
    _s = _series_significance(_v)
    assert _s["comparable"] and _s["level"] == 1, _s
    assert not _s["flat"], "2.8% 不该被完全隐掉"
    assert _s["rel_reading"] > _FLAT_LEVEL_MINOR, _s

    # 离散跳变：只在两个固定值之间跳，应被判为采集噪声特征
    assert _s["discrete"] and _s["distinct_values"] == 2, _s

    # 真·平坦：占读数远低于 1%，完全隐掉
    _s = _series_significance([10.0 + (0.001 if i % 2 else -0.001) for i in range(60)])
    assert _s["level"] == 0 and _s["flat"], _s

    # 尖峰毛刺：P95-P5 不受影响，仍判平坦（若用 max-min 就会被这一个点撑开）
    _v = [10.0] * 60
    _v[30] = 12.0
    assert _series_significance(_v)["level"] == 0, "单点毛刺不该改变整段判定"

    # 真实大变化：占比远超 3%，完整展开
    _s = _series_significance([0.018] * 30 + [0.0] * 30)
    assert _s["level"] == 2 and not _s["flat"], _s

    # 有工程量程时第二路参与，两路取更高等级
    _v = [50.0 + (i % 3) * 0.8 for i in range(60)]      # 占读数约 3.2%
    assert _series_significance(_v)["level"] == 2, "占读数 3.2% 应完整展开"
    _s = _series_significance(_v, {"high": 100.0, "low": 0.0})
    assert _s["rel_range"] is not None, _s

    # 样本不足时不判级别（-1），避免拿两三个点下结论
    assert _series_significance([1.0, 2.0])["level"] == -1
    assert not _series_significance([])["comparable"]

    # 极值与变化量
    _f = _series_features(_mk([1.0, 5.0, 2.0, 9.0, 3.0]))
    assert _f["min"] == 1.0 and _f["max"] == 9.0 and _f["latest"] == 3.0

    # 点数不足或时间无法解析时返回空字典，而不是抛异常
    assert _series_features([]) == {}
    assert _series_features([{"time": "x", "value": 1}]) == {}

    # ── 保形压缩：跳变点不能被抽掉 ──
    # 构造「长期平直 + 中段跳变」的曲线，压缩后跳变两侧必须还在
    _jump = _mk([10.0] * 200 + [50.0] * 200)
    _c = _compress_series(_jump, 24)
    assert len(_c) <= 24
    _vals = [p["value"] for p in _c]
    assert 10.0 in _vals and 50.0 in _vals, "跳变两侧被抽掉了"
    assert _vals[0] == 10.0 and _vals[-1] == 50.0, "首尾必须保留"
    # 点数少于上限时原样返回，不做无谓处理
    assert _compress_series(_jump[:5], 24) == _jump[:5]

    # ── 测点相关性判定：模糊检索的 top-1 不能盲信 ──
    # 实测搜「轴封供汽压力」返回 5 条，正确答案排第 4，top-1 是完全无关的
    # 「高压主蒸汽压力3选1后」。取 top-1 会把错数据当事实喂给模型。
    assert not _point_matches("轴封供汽压力", "高压主蒸汽压力3选1后")
    assert not _point_matches("轴封供汽压力", "低压主蒸汽压力3选后")
    assert _point_matches("轴封供汽压力", "#1机组轴封供气压力1")   # 汽/气 异体字仍应命中
    # 只命中设备不命中物理量：阀门反馈不是压力测点
    assert not _point_matches("轴封供汽压力", "轴封供汽管道疏水母管气动关断阀开反馈")
    # 同义词（水位↔液位）字符级判不出来，**这正是模型兜底存在的理由**：
    # 这里必须是 False，否则 _resolve_point 就不会走到 _pick_live_point
    assert not _point_matches("凝汽器水位", "凝汽器液位")
    assert not _point_matches("凝汽器水位", "主蒸汽温度")
    assert _point_matches("压力", "主蒸汽压力")                  # 短词只判首部
    assert not _point_matches("轴封", "")    # 实时块合并：三类数据都要出现在同一条 user 消息里
    _rt = _prompt(
        "q", [],
        series=[{
            "kind": "trend", "query": "汽包水位", "kks": "01K", "description": "汽包水位",
            "unit": "mm", "start": "2026-09-12T00:00:00", "end": "2026-09-13T00:00:00",
            "interval_seconds": 300,
            "summary": {"latest_value": 1, "min_value": -2, "max_value": 3},
            "samples": [{"time": "2026-09-12T01:00:00+00:00", "value": 1}],
        }],
        logs={
            "start": "2026-09-12", "end": "2026-09-12", "major_only": False,
            "summary": {"total_events": 118, "major_events": 71, "categories": {"值班记事": 88}},
            "events": [{
                "time": "2026-09-12 22:58:44", "source": "值长日志",
                "category": "值班记事", "severity": "major", "content": "某事件",
            }],
        },
    )
    assert "测点序列数据" in _rt[-1]["content"] and "运行日志" in _rt[-1]["content"]
    assert "【重大】" in _rt[-1]["content"]
    # 作答形态：必须以段落为默认，且不能残留「鼓励分级标题」的旧指令
    _sys = _prompt("q", [])[0]["content"]
    assert "作答形态" in _sys, "缺少作答形态指令"
    assert "结构跟着内容走" in _sys
    assert "列表" in _sys and "标题" in _sys, "放开结构后应明确允许列表与标题"
    assert "使用分级标题和编号列表" not in _sys, "旧的鼓励结构化指令仍在"
    # 实时测点单独成块，不与文档证据混在一起
    live_prompt = _prompt(
        "q",
        [],
        live_points=[
            {
                "description": "主蒸汽温度",
                "kks": "01LBA10BT607",
                "value": 24.2,
                "unit": "℃",
                # 这里是 shape() 之后的形态：换算在 _fetch_live_points 里做过了，
                # _prompt 只负责展示，不再二次换算（重算会多加 8 小时）。
                "time": "2026-09-12 11:37:44",
            }
        ],
    )
    assert "实时测点数据" in live_prompt[-1]["content"]
    assert "24.2" in live_prompt[-1]["content"]
    # 采集时间：统一换算成本地时间，且不带时区后缀（现场没人看 UTC）
    assert _format_live_time("2026-09-12T03:37:44+00:00") == "2026-09-12 11:37:44"
    assert _format_live_time("2026-09-12T03:37:44Z") == "2026-09-12 11:37:44"
    assert _format_live_time("2026-09-12T03:37:44") == "2026-09-12 11:37:44"  # 无时区按 UTC
    assert _format_live_time("2026-09-12T11:37:44+08:00") == "2026-09-12 11:37:44"
    assert _format_live_time("") == ""
    assert _format_live_time("不是时间") == "不是时间"  # 解析失败原样返回，不丢数据
    assert "采集时间 2026-09-12 11:37:44（北京时间）" in live_prompt[-1]["content"]
    # 系统提示词要明确不许模型自己补时区，否则它会写回 UTC 或再换算一遍
    assert "不要附加时区后缀" in live_prompt[0]["content"]
    assert "UTC" not in live_prompt[0]["content"]
    # 系统能力类问题：不拼证据段落，且系统提示词明确禁止提检索/证据
    meta = _prompt("q", [], skip_retrieval=True)
    assert meta[-1]["content"] == "问题：q"
    assert "检索证据" not in meta[-1]["content"]
    assert "未检索文档知识库" in meta[0]["content"]
    assert _graph_top_k("offline", 40) <= settings.local_graph_top_k
    assert _graph_top_k("online", 40) <= settings.online_graph_top_k
    assert len(_llm_contexts("offline", [{"id": str(i)} for i in range(40)])) == 10
    assert len(_llm_contexts("online", [{"id": str(i)} for i in range(40)])) == 40
    assert _user_default_inference_mode({}) == "offline"
    assert _user_default_inference_mode({"default_inference_mode": "online"}) == "online"
    assert _user_default_inference_mode({"default_inference_mode": "bad"}) == "offline"
    assert _needs_cross_mode_compression("online", "offline")
    assert not _needs_cross_mode_compression("offline", "online")
    assert not _needs_cross_mode_compression("online", "online")
    assert not _needs_cross_mode_compression(None, "offline")
    original_urls = settings.rag_remote_urls
    original_cursor = _rag_route_cursor
    settings.rag_remote_urls = ()
    assert _rag_route_order()[0] == [settings.rag_base_url]
    settings.rag_remote_urls = ("http://gpu-0:8001", "http://gpu-1:8001")
    first = _rag_route_order()[0]
    second = _rag_route_order()[0]
    assert first[:2] == ["http://gpu-0:8001", "http://gpu-1:8001"]
    assert second[:2] == ["http://gpu-1:8001", "http://gpu-0:8001"]
    settings.rag_remote_urls = original_urls
    globals()["_rag_route_cursor"] = original_cursor
    assert _CONVERSATION_ID_RE.fullmatch("local-123")
    assert not _CONVERSATION_ID_RE.fullmatch("../escape")


if __name__ == "__main__":
    _self_check()
    print("QA API self-check passed")
