from __future__ import annotations

import base64
import copy
import json
import math
import os
import re
import threading
import asyncio
from datetime import datetime, timedelta, timezone
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
    "你是检索规划器。针对用户问题输出两行，不要解释、不要多余文字：\n"
    "检索: yes 或 no   —— 是否需要查工业文档知识库（规程、参数、故障、操作、标准）\n"
    "测点: <定位实时测点的关键词> 或 无   —— 是否需要读实时测点数值\n"
    # 关键词必须短：测点表的命名和用户的说法常对不上（表里叫「凝汽器液位」，
    # 用户说「凝汽器水位」），而且带机组号之类的修饰语会让测点检索的排序跑偏。
    "测点关键词要短（2-6 个字），只描述物理量本身，不要带机组号、设备全称或修饰语。\n"
    "多个测点用、分隔，最多 3 个。\n"
    "示例：\n"
    "  问：一号机凝汽器水位多少，然后结合知识库回答\n"
    "  检索: yes\n"
    "  测点: 凝汽器液位\n"
    "  问：轴封系统的作用\n"
    "  检索: yes\n"
    "  测点: 无\n"
)
# 规划开关：模型判错时可不重建镜像直接关掉，退回「一律检索、不读测点」
_ROUTER_ENABLED = os.getenv("HDW_RETRIEVAL_ROUTER", "true").lower() != "false"
_LIVE_POINTS_MAX = int(os.getenv("HDW_LIVE_POINTS_MAX", "3"))
# 关键词匹配偏了时，最多回退试几个候选测点
_LIVE_POINT_CANDIDATES = int(os.getenv("HDW_LIVE_POINT_CANDIDATES", "5"))
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


def _parse_plan(text: str) -> tuple[bool | None, list[str]]:
    """解析规划输出。检索无法判定返回 None，测点无法判定返回空列表。"""
    needs_rag: bool | None = None
    points: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        match = re.match(r"^检索\s*[:：]\s*(yes|no|是|否)", line, re.I)
        if match:
            needs_rag = match.group(1).lower() in ("yes", "是")
            continue
        match = re.match(r"^测点\s*[:：]\s*(.+)$", line)
        if match:
            value = match.group(1).strip()
            if value and value.lower() not in ("无", "none", "-", "null"):
                points = [
                    item.strip()
                    for item in re.split(r"[、,，;；|]", value)
                    if item.strip()
                ]
    return needs_rag, points[:_LIVE_POINTS_MAX]


async def _plan_retrieval(
    question: str, mode: Literal["online", "offline"]
) -> tuple[bool, list[str]]:
    """一次调用同时决定「要不要查文档」和「要读哪些实时测点」，判断发生在检索之前。

    极小调用：无证据、只出两行，实测约 200-400ms。
    任何异常或无法解析都退回保守默认：照常检索、不读测点。
    """
    if not _ROUTER_ENABLED:
        return True, []
    try:
        _, client = _llm_client(mode)
        result = await client.complete(
            [
                {"role": "system", "content": _PLANNER_PROMPT},
                {"role": "user", "content": question},
            ],
            max_tokens=32,
            temperature=0.0,
            chat_template_kwargs={"enable_thinking": False} if mode == "offline" else None,
        )
    except Exception:
        return True, []
    needs_rag, points = _parse_plan(result.content)
    return (True if needs_rag is None else needs_rag), points


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

        # 快路径：直接按关键词取。内部按相似度取 top-1，命中时最省。
        try:
            data = await _mcp_call_tool("point_query_current_value", {"query_text": keyword})
            if isinstance(data, dict) and data.get("value") is not None:
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

        # 关键词整词匹配不上（测点表叫「凝汽器液位」，用户问「凝汽器水位」）时，
        # 让模型从相近候选里挑最符合意图的那个，再按 KKS 精确取值。
        # 用模型而不是同义词表：测点表的命名习惯会变，硬编码词表很快就废。
        candidates = [
            item for item in (found.get("items") or []) if str(item.get("kks") or "").strip()
        ]
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


async def _retrieve_planned(
    question: str,
    mode: Literal["online", "offline"],
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # 规划在检索之前：一次调用同时决定「要不要查文档」和「要读哪些实时测点」。
    needs_rag, point_keywords = await _plan_retrieval(question, mode)
    live_points, live_errors = await _fetch_live_points(point_keywords, mode)

    if not needs_rag:
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
            "live_points": live_points,
            "live_errors": live_errors,
        }
    plan, planning_error = await _plan_rag(question, mode, top_k)
    if plan.get("no_retrieval"):
        # 不检索也不查图谱：下游拿到空 contexts 会自然产出空 citations / graph_context，
        # 前端 addEvidence() 在两者都空时不渲染依据面板。
        return [], {
            "plan": plan,
            "rounds": [],
            "deduplicated_count": 0,
            "no_retrieval": True,
        }
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
        "live_points": live_points,
        "live_errors": live_errors,
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
        live_block = "\n\n实时测点数据：\n" + "\n".join(
            f"[测点 {idx}] {item.get('description') or item.get('query')}"
            f"（KKS={item.get('kks') or '未知'}） = {item.get('value')} {item.get('unit')}"
            f"  采集时间 {item.get('time')}（北京时间）"
            for idx, item in enumerate(live_points, 1)
        )

    # 未检索时不拼证据段落：一旦出现「检索证据：无」这类框架，模型会顺着去说
    # 「证据中没有」，而不是直接依据自身设定回答。
    if skip_retrieval:
        user = f"问题：{question}{live_block}"
        system_body = (
            "本轮未检索文档知识库。"
            + (
                "请基于下面给出的实时测点数据回答，并说明数据采集时间。"
                if live_points
                else "请直接依据自身设定回答，不要提及检索、证据或知识库。"
            )
            # 不加这句模型会为了「答得完整」编造上下文长度之类的配置数值，
            # 比「证据中没有」更糟。
            + "涉及本系统的具体配置数值（上下文长度、端口、模型参数等）时，"
            "不确知就直说无法确认，不要给出估计值、示例值或常见默认值。"
        )
    else:
        user = (
            f"问题：{question}{live_block}"
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
                "使用方便阅读的短段落、分级标题和编号列表；禁止长难句。"
                "长内容必须按主题、步骤、条件和例外分段，禁止把大量内容堆在一个长段落中。"
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
    assert _parse_plan("检索: yes\n测点: 无") == (True, [])
    assert _parse_plan("检索: no\n测点: 主蒸汽温度") == (False, ["主蒸汽温度"])
    assert _parse_plan("检索: yes\n测点: 闭式冷却水泵、凝结水压力") == (
        True,
        ["闭式冷却水泵", "凝结水压力"],
    )
    assert _parse_plan("检索: yes\n测点: a、b、c、d")[1] == ["a", "b", "c"]  # 上限 3 个
    assert _parse_plan("检索：否\n测点：None") == (False, [])
    assert _parse_plan("随便说点什么") == (None, [])
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
