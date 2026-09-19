from __future__ import annotations

import base64
import copy
import hmac
import json
import math
import os
import re
import threading
import time
import asyncio
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Literal
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from hdw_auth import AuthError, AuthStore
from LLM_API import (
    ContextManager,
    OpenAICompatibleClient,
    estimate_messages,
)

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
    # 提问方的本地时刻（ISO-8601）。**可选**：老调用方不传就回落服务端时钟。
    # 前端在用户手边，它的时钟才是用户看到的那个——容器时钟走 UTC 时
    # 「今天几点」「哪个班次」会整体偏 8 小时，而这个偏差从答案表面看不出来。
    client_time: str | None = Field(default=None, max_length=64)
    # 默认 False：老的调用方（含 /v1/chat/completions 的内部转发）行为不变。
    stream: bool = False


class InternalQARequest(BaseModel):
    """服务间问答请求（HDW_API 容器 → qa-api）。

    与 `QARequest` 的差别就是**没有 conversation_id 和 messages**：这条通道
    按约定不做上下文管理，管线只看到当次问题。不提供这两个字段，
    比提供了再在路由里强制置空更难被误用。
    """

    question: str = Field(min_length=1, max_length=8000)
    top_k: int = Field(default=40, ge=1, le=40)
    inference_mode: Literal["online", "offline"] | None = None
    # 同上：调用方若知道自己的本地时刻就带上。
    client_time: str | None = Field(default=None, max_length=64)


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
    permissions: dict[str, Any] | None = None
    vision_priority: list[dict[str, Any]] | None = None


class FrpPatch(BaseModel):
    enabled: bool


class LocalModelPatch(BaseModel):
    # True = 加载（把配置里的本地模型跑起来），False = 卸载（停进程、释放全部显存）。
    loaded: bool


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
            # 空串 = 不要本地视觉。也可以只写文件名，start.sh 会去
            # HDW_Engines/LLM_Models/ 下解析。
            "mmproj": "",
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
            # 这里原来有一份 online.model_options。删掉了：**没有任何消费者**
            # （前端在线模型名是自由文本框，不像本地那样是下拉），而把供应商的
            # 模型列表硬编码进配置必然会过期——实测线上那份列的两个模型在
            # 供应商侧早就没有了。留着只会误导，不如没有。
        },
        "permissions": {
            # 普通用户能不能在提问时上传图片。**管理员始终可以**，
            # 与 online.enabled_for_users 同一套语义。
            "image_upload_for_users": True,
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
    for section in ("local", "online", "retrieval", "permissions"):
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
            # 视觉投影器（mmproj）。空串合法，表示不要本地视觉。
            "mmproj",
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
        "permissions": {"image_upload_for_users"},
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
            elif key == "mmproj":
                # **允许空**：空串表示「不要本地视觉」，是合法且默认的状态。
                # 只在超长时拒绝，避免把垃圾塞进 llama 的启动参数。
                if len(str(value)) > 240:
                    raise HTTPException(status_code=422, detail=f"invalid {section}.{key}")
            elif key == "context_window":
                if not isinstance(value, int) or not 4096 <= value <= 2_000_000:
                    raise HTTPException(status_code=422, detail=f"invalid {section}.context_window")
            elif key in {
                "multimodal_enabled",
                "mtp_enabled",
                "thinking_enabled",
                "enabled_for_users",
                "image_upload_for_users",
            } and not isinstance(value, bool):
                raise HTTPException(status_code=422, detail=f"invalid {section}.{key}")

    priorities = values.get("vision_priority")
    if priorities is not None:
        if not 1 <= len(priorities) <= 4:
            raise HTTPException(status_code=422, detail="vision_priority must contain 1 to 4 candidates")
        for index, candidate in enumerate(priorities, 1):
            # kind 缺省为 chat：老配置里没有这个字段，必须当 chat 用，
            # 否则升级后所有视觉候选都会被判非法，视觉整体失效。
            kind = str(candidate.get("kind") or "chat").strip().lower()
            if kind not in _VISION_KINDS:
                raise HTTPException(
                    status_code=422, detail=f"vision_priority[{index}].kind is invalid"
                )
            # mode 保持只允许 online/offline。mineru 行里它不参与调用，
            # 但仍要求填一个合法值——**不引入第三态**，避免牵连
            # `_resolve_inference_mode` 那三处 Literal。
            if candidate.get("mode") not in {"online", "offline"}:
                raise HTTPException(status_code=422, detail=f"vision_priority[{index}].mode is invalid")
            # mineru 的 model 只是界面上的标签，允许留空（内部补成 "mineru"）。
            # chat 的 model 是真要拿去调用的，启用时必须非空；**停用的项不校验**——
            # 它不会被调用，填没填都不影响行为。界面固定给 4 行（与 1~4 的上限对齐），
            # 空行以 enabled=false 提交，不校验才存得下去。
            disabled = candidate.get("enabled", True) is False
            if kind == "chat" and not disabled and not str(candidate.get("model") or "").strip():
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


# 对外问答 API（HDW_API 容器）调进来的身份。**不是登录用户**：
# 它不进 auth.csv，也拿不到会话令牌，只是给管线一个「谁在问」的占位。
# 管线上真正读 user 的只有三处，都只看 role / default_inference_mode
# （_online_inference_allowed、_image_upload_allowed、_user_default_inference_mode），
# 所以这里刻意给 role="user"——服务通道不该默认拥有管理员才有的在线推理权限。
_API_SERVICE_IDENTITY = {
    "code": "api",
    "name": "对外 API",
    "role": "user",
    "default_inference_mode": "offline",
}


def _require_internal_api_key(key: str | None) -> None:
    """服务间调用的无条件鉴权。

    **刻意不复用 `_check_auth`**：那个函数在 `HDW_ENABLE_AUTH=false` 时直接
    return，而 qa-api 的端口绑在所有网卡上（compose 里是 `"${HDW_QA_API_PORT}:8080"`）。
    本机实测 `enable_auth` 就是 false——拿 `_check_auth` 守这条新入口，等于在
    局域网上开一个无鉴权的问答接口。这里不设开关，永远校验。

    密钥也没配时同样拒绝：宁可这条通道不可用，也不能静默放行。
    """
    expected = settings.api_internal_key
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="HDW_API_INTERNAL_KEY 未配置，服务间接口不可用",
        )
    if not key or not hmac.compare_digest(str(key), expected):
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


def _image_upload_enabled_for_users() -> bool:
    section = _read_model_config().get("permissions", {})
    value = section.get("image_upload_for_users", True) if isinstance(section, dict) else True
    return value if isinstance(value, bool) else True


def _image_upload_allowed(user: dict[str, str]) -> bool:
    """提问时能不能上传图片。**管理员始终可以**，与在线推理同一套语义。"""
    return user.get("role") == "admin" or _image_upload_enabled_for_users()


def _permissions_for_user(user: dict[str, str]) -> dict[str, bool]:
    return {
        "model_management": user.get("role") == "admin",
        "frp_management": user.get("role") == "admin",
        "online_inference": _online_inference_allowed(user),
        "online_inference_for_users": _online_inference_enabled_for_users(),
        "image_upload": _image_upload_allowed(user),
        "image_upload_for_users": _image_upload_enabled_for_users(),
    }


def _llm_client(
    mode: str | None = None,
    *,
    model_override: str | None = None,
    base_url_override: str | None = None,
    timeout_override: float | None = None,
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
            timeout=timeout_override or settings.llm_timeout,
        )
    return selected, OpenAICompatibleClient(
        base_url_override or profile["base_url"],
        model_override or profile["model"],
        # 本地默认**无超时**：本地推理可能被长上下文拖到几分钟，卡死一个超时值
        # 会把正常的长回答误杀。但视觉转写这类辅助调用必须有界，否则一次挂起的
        # 请求会占住一个 asyncio 任务不放——所以留 timeout_override 这个出口。
        timeout=timeout_override,
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
    # 读取超时用 settings.mcp_timeout，**不能用 `rag_connect_timeout + 15`**。
    # 后者是 18 秒，而日志工具要等 LIEMS 门户（它自己的 read timeout 就是 30 秒），
    # 结果是上游先超时、工具已经准备好的失败原因（liems_http_error 等）一个字都传不回来。
    request_timeout = httpx.Timeout(settings.mcp_timeout, connect=settings.rag_connect_timeout)
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
    "你是检索规划器。现在是 {now}（北京时间，{shift}）。\n"
    "针对用户问题输出五行，不要解释、不要多余文字。用不到的写「无」：\n"
    "检索: yes 或 no   —— 是否需要查工业文档知识库（规程、参数、故障、操作、标准）\n"
    "测点: <关键词>[、<关键词>] 或 无   —— 是否需要读测点**当前值**\n"
    "历史: <关键词>|<起始日期>|<结束日期> 或 无   —— 是否需要测点**历史序列**\n"
    "趋势: <关键词>|<起始日期>|<结束日期>|<采样间隔秒> 或 无   —— 是否需要测点**变化趋势**\n"
    "日志: <起始日期>|<结束日期>|<重大或->|<班次或-> 或 无   —— 是否需要查**运行日志**\n"
    "\n"
    "判断依据（按语意，不要按关键词硬匹配）：\n"
    "· 「现在多少」「当前值」→ 测点；「昨天/过去三天/这段时间的变化」→ 历史；\n"
    "  「趋势」「走势」「一直涨吗」「画个曲线」→ 趋势；\n"
    "  「有什么操作」「值班日志」「重大事件」「交接班记录」→ 日志。\n"
    "· 一个问句可以同时命中多项（例如「看看汽包水位的趋势，再结合规程说说」）。\n"
    "· 只要当前值就不要填历史或趋势，它们会各自触发一次数据查询。\n"
    "· 日期一律写成 YYYY-MM-DD，相对时间按上面的当前时刻换算；不确定结束日期就用今天。\n"
    "· 趋势的采样间隔按问题的时间跨度选：一天用 300，三天用 900，一周用 1800。\n"
    "· 日志只在对方明确问运行记录时填；问规程、原理、参数含义时填「无」。\n"
    "· **班次**：白班 08:30~16:30、中班 16:30~23:30、夜班 前日23:30~今日08:30。\n"
    "  对方说「今天白班」「昨晚夜班」这类**带班次**的说法时，日期照常按日期段写，\n"
    "  班次写在第四段（白班／中班／夜班）；不指定班次就写「-」。\n"
    "  「昨天」若紧接「夜班」，指的是昨天 23:30 起、延续到今天 08:30 的那一段。\n"
    "· 只说「重大」「重要」时在日志第三段写「重大」，否则写「-」。\n"
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
    "  日志: {yesterday}|{yesterday}|重大|-\n"
    "  问：今天白班有什么事\n"
    "  检索: no\n"
    "  测点: 无\n"
    "  历史: 无\n"
    "  趋势: 无\n"
    "  日志: {today}|{today}|-|白班\n"
)
# 规划开关：模型判错时可不重建镜像直接关掉，退回「一律检索、不读测点」
_ROUTER_ENABLED = os.getenv("HDW_RETRIEVAL_ROUTER", "true").lower() != "false"
_LIVE_POINTS_MAX = int(os.getenv("HDW_LIVE_POINTS_MAX", "3"))
# 关键词匹配偏了时，最多回退试几个候选测点
_LIVE_POINT_CANDIDATES = int(os.getenv("HDW_LIVE_POINT_CANDIDATES", "5"))
# 日志注入上限：一次抓取实测 118 条，全塞进上下文会挤占文档证据。
# 按严重程度排序后取前 N 条，重大事件优先。
_LOGS_MAX = int(os.getenv("HDW_LOGS_MAX", "40"))
# 日志类问题（本轮不检索文档，日志就是答案主体）的上限。**0 = 不截断**。
# 这类问题没有文档证据可挤占，截断只是让答案凭空少掉内容——实测抓 161 条
# 只给 40 条时，用户看到的就是「总结太简略」。
_LOGS_MAX_FULL = int(os.getenv("HDW_LOGS_MAX_FULL", "0"))
# 向 MCP 日志工具要多少条。**不传 limit 会落到工具自己的默认值**
# （log_query_recent 80 / log_query_range 120），再被上面的 _LOGS_MAX 截一刀——
# 所以「抓 161 条只注入 40 条」里的 161，在中途已经先被压到 120 了。
# 工具侧钳制上限是 500（recent / major_events）/ 800（range），取 500 用满。
_LOGS_FETCH_LIMIT = int(os.getenv("HDW_LOGS_FETCH_LIMIT", "500"))
# 班次刚开始的这段宽限期里，「还没有记录」是正常的，不算缺失。
# 实测 16:34 查当天：中班 16:30 才开始 4 分钟，却被报成「缺中班记录」。
_SHIFT_GRACE_MINUTES = int(os.getenv("HDW_SHIFT_GRACE_MINUTES", "30"))
# 逐日工作摘要的行数上限。明细受 _LOGS_MAX 截断（实测一次抓 161 条、只注入 40 条），
# 摘要负责体现全量规模，所以**条数统计来自全部事件**而不是截断后的那批。
# 全量注入时明细已经给全了，逐日分栏会与它重复，此时只留计数行（见 _prompt）。
_LOGS_DIGEST_LINES = int(os.getenv("HDW_LOGS_DIGEST_LINES", "120"))
_LOGS_DIGEST_ITEMS = int(os.getenv("HDW_LOGS_DIGEST_ITEMS", "6"))

# 日志作答规范。**只在有日志时注入**——没有日志还挂着，模型会去找不存在的日志。
#
# 来源：SmartGasTurbine 的 log_summary_prompt.md（14 条交接班口径）。筛掉了它的
# 语音播报条款（它要求「不要标题、不要列表、不要星号」，那是给 TTS 的；
# 我们前端渲染 Markdown，照搬会适得其反），保留按运行交接口径归纳的部分。
_LOG_ANSWER_RULES = (
    "\n\n【本次有运行日志，按下面的口径归纳】\n"
    "· 按日期分组；每个日期下分成「主要工作」和「两票工作」两部分。\n"
    "· 主要工作说明涉及的机组、系统、设备、试验、启停、补水、检修、运行交代、"
    "接地线、异常或报警——**合并同类项**，不要逐条念原文。\n"
    "· 两票（工作票、操作票、许可、批准、终结、延期）按票种或设备系统归纳，"
    "讲清任务、设备系统、负责人和风险关注点即可；票号与审批流不必写进正文。\n"
    "· **一般事项不必精确到分钟**，说明做了什么就行；只有重要事件必须带时间"
    "（并网、停机、跳闸、保护动作、重要调度命令、重大异常、重要设备启停）。\n"
    "· **某天有实际内容时，不要只写成「共有 N 条」这种统计**——统计只是佐证，"
    "要把那天做了什么说出来。\n"
    "· 事件之间要有逻辑：夜班发现的缺陷白班已处理，就写「存在缺陷，白班已处理，"
    "需严密监视」，不要只记「存在缺陷」。\n"
    "· 「合计」「分类」里的条数是**全量**统计，按天列出的只是其中一部分；"
    "引用规模时以合计为准，不要把列出的条数说成总数。\n"
)

# 完整性专项规范。缺班次时必须点破，否则读的人会以为看到的是全部。
_LOG_COMPLETENESS_RULE = (
    "· 上面给出了「日志完整性」：**缺哪些日志类型、缺哪些班次必须如实说明**，"
    "不能把不完整的日志当成已全部拿到。\n"
)


# ── 班次划分。**现场交接在半点，不是整点**（实测班次表为 08:30 / 16:30 / 23:30），
# 写成整点会让边界前后半小时的日志落到隔壁班次去。
# 元组是 (名称, 起始分钟, 结束分钟)；跨零点的夜班起始 > 结束。
_SHIFT_BOUNDS: tuple[tuple[str, int, int], ...] = (
    ("夜班", 23 * 60 + 30, 8 * 60 + 30),
    ("白班", 8 * 60 + 30, 16 * 60 + 30),
    ("中班", 16 * 60 + 30, 23 * 60 + 30),
)
_WEEKDAYS = ("一", "二", "三", "四", "五", "六", "日")


def _shift_of(moment: datetime) -> str:
    """给定时刻属于哪个班次。"""
    minute = moment.hour * 60 + moment.minute
    for name, start, end in _SHIFT_BOUNDS:
        if start > end:                       # 跨零点：落在 [start, 24h) 或 [0, end)
            if minute >= start or minute < end:
                return name
        elif start <= minute < end:
            return name
    return "夜班"                             # 只可能是结束那一分钟，归夜班


def _shift_window(shift: str, day: date) -> tuple[datetime, datetime]:
    """某天某班次的起止时刻。夜班跨零点，结束落在次日。"""
    for name, start, end in _SHIFT_BOUNDS:
        if name != shift:
            continue
        midnight = datetime.combine(day, datetime.min.time())
        begin = midnight + timedelta(minutes=start)
        finish = midnight + timedelta(minutes=end)
        if start > end:
            finish += timedelta(days=1)
        return begin, finish
    raise ValueError(f"未知班次：{shift}")


# 客户端时间戳与服务端时刻的最大容许偏差。**不能无条件信客户端**：
# 浏览器时钟错乱（机器休眠后漂移、用户改过系统时间）会让「今天」「白班」
# 整体偏掉，而答案表面看不出来——比服务端时区错更难排查。
_CLIENT_TIME_MAX_SKEW = timedelta(
    minutes=int(os.getenv("HDW_CLIENT_TIME_MAX_SKEW_MIN", "120"))
)


def _server_now() -> datetime:
    """服务端北京时间（带 +08:00）。

    容器**没有 TZ**，实测 `datetime.now()` 走 UTC（比北京时间慢 8 小时）。
    所以这里显式加偏移，与 `_local_today()`、`_format_live_time()` 同一口径。
    **不要退回裸 `datetime.now()`**——本文件此前唯一那样写的地方就是
    `_shift_context()`，它把 15:24 报成了 07:24（夜班）。
    """
    return datetime.now(timezone.utc).astimezone(
        timezone(timedelta(hours=_LIVE_TIME_OFFSET_HOURS))
    )


def _clock_now(client_time: str | None = None) -> tuple[datetime, str]:
    """本次提问的「现在」+ 这个时刻的来源。

    优先取客户端时间戳：浏览器就在提问的人手边，它的时钟才是用户看到的时间。
    偏差超过 `_CLIENT_TIME_MAX_SKEW` 就回落服务端——见该常量的注释。

    返回来源是为了**让回落这件事看得见**：静默改用服务端时钟的话，
    「客户端时钟偏了 3 小时」从答案表面完全看不出来，只会表现为
    「今天/白班偶尔答错」。诊断信息里带上它，下次一眼就能定位。
    """
    server = _server_now()
    text = str(client_time or "").strip()
    if not text:
        return server, "server"
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return server, "server(客户端时间无法解析)"
    # 不带时区的按 UTC 解释：前端发的是 `toISOString()`，本来就是 UTC 带 Z，
    # 真出现裸时间时按本地时间解释反而会多算 8 小时。
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    moment = moment.astimezone(server.tzinfo)
    if abs(moment - server) > _CLIENT_TIME_MAX_SKEW:
        return server, "server(客户端时间偏差过大)"
    return moment, "client"


def _local_now(client_time: str | None = None) -> datetime:
    """本次提问的「现在」，北京时间。只需要时刻时用它；要来源用 `_clock_now`。"""
    return _clock_now(client_time)[0]


def _now_context(now: datetime) -> str:
    """把「现在」告诉模型。**与日志无关，每轮都要给**。

    此前这段时间只在有日志时才注入，于是没日志的问答既不知道今天几号、
    也不知道现在几点，「昨天」「今天上午」只能靠猜。
    """
    return (
        f"\n\n当前时间：{now.strftime('%Y-%m-%d %H:%M')}"
        f"（星期{_WEEKDAYS[now.weekday()]}，{_shift_of(now)}）。"
        "用户说「今天」「昨天」「现在」时一律以这个时刻为准换算。"
    )


def _shift_context(now: datetime) -> str:
    """班次划分 + 「未到来的班次没有记录是正常的」。

    日志按班次逐段产生：现在是白班，当天的中班记录自然还不存在。
    不告诉模型这一点，它会把「还没发生」报成「数据缺失」——实测每次问
    「今天有什么事」，答案开头都会挂一句「缺中班记录」，而那是正常的。

    **只在有日志时注入**：它解释的是日志为什么缺段。
    """
    return (
        "\n班次划分：夜班 前日23:30~今日08:30，白班 08:30~16:30，"
        f"中班 16:30~23:30（现在是{_shift_of(now)}）。"
        "**当天的日志按班次逐段产生，尚未到来的班次没有记录是正常的**，"
        "不要当成数据缺失报出来。"
    )

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
# 前端内联 SVG 图的点数上限。原始采样可达上万点，SVG 画不完也不需要。
# 极值点与斜率变号点在降采样**之前**算出并单独保留，不受此上限影响。
_CHART_MAX_POINTS = int(os.getenv("HDW_CHART_MAX_POINTS", "320"))
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


def _plan_shift(text: str) -> str:
    """把模型给的班次写法认成标准名；认不出或不限班次返回空串。

    「-」「无」是提示词要求的「不限班次」写法，不是错误，不能当解析失败处理。
    """
    text = (text or "").strip()
    if not text or text.lower() in _PLAN_EMPTY:
        return ""
    if any(word in text for word in ("全部", "不限", "所有", "全天", "整天")):
        return ""
    for name, _, _ in _SHIFT_BOUNDS:
        if name in text:
            return name
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
    """服务端的「今天」（北京时间）。与 `_local_now()` 同源，避免两处口径分叉。"""
    return _server_now().date()


def _fit_series(points: list[tuple[float, float]]) -> list[dict[str, Any]]:
    """对采样序列做最小二乘直线拟合，返回拟合曲线的**首尾两点**。

    直线由两点即可完全确定，返回上百个共线点只会让 SVG 路径白白多出上万字符。
    拟合值**只用于画那条虚线**，不作为事实下发给模型——模型引用数值时必须用真实采样值。
    """
    if len(points) < 3:
        return []
    base = points[0][0]
    xs = [moment - base for moment, _ in points]
    ys = [value for _, value in points]
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    den = sum((x - mean_x) ** 2 for x in xs)
    if den <= 0:
        return []
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / den
    intercept = mean_y - slope * mean_x
    return [
        {
            "time": datetime.fromtimestamp(moment, tz=timezone.utc).isoformat(),
            "value": intercept + slope * (moment - base),
        }
        for moment, _ in (points[0], points[-1])
    ]


def _special_points(points: list[tuple[float, float]]) -> list[dict[str, Any]]:
    """标出值得在图上看到的点：极值、最新值、斜率变号点。

    照 SmartGasTurbine 的 `_build_trend_special_points`：
      · value 一律取**真实采样值**，不是拟合值——模型要引用的是实测数字
      · 斜率变号点按变化强度排序取前若干个，弱的抖动不值得标
      · 任一侧斜率为 0 说明是平段，**平段边界不叫「变号」**，要跳过
    """
    if len(points) < 2:
        return []
    ys = [value for _, value in points]
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    def add(index: int, point_type: str, detail: str) -> None:
        index = max(0, min(len(points) - 1, index))
        if (point_type, index) in seen:
            return
        seen.add((point_type, index))
        moment, value = points[index]
        output.append(
            {
                "time": datetime.fromtimestamp(moment, tz=timezone.utc).isoformat(),
                "value": value,
                "type": point_type,
                "detail": detail,
            }
        )

    add(min(range(len(ys)), key=lambda i: ys[i]), "global_min", "窗口内真实采样最小值")
    add(max(range(len(ys)), key=lambda i: ys[i]), "global_max", "窗口内真实采样最大值")
    add(len(points) - 1, "latest", "窗口内最新采样值")

    slopes: list[float] = []
    for index in range(1, len(points)):
        dx = points[index][0] - points[index - 1][0]
        slopes.append(0.0 if abs(dx) < 1e-12 else (points[index][1] - points[index - 1][1]) / dx)

    candidates: list[tuple[float, int, float, float]] = []
    for index in range(1, len(slopes)):
        previous, current = slopes[index - 1], slopes[index]
        if abs(previous) < 1e-12 or abs(current) < 1e-12:
            continue
        if previous * current < 0:
            candidates.append((abs(current - previous), index, previous, current))
    candidates.sort(key=lambda item: item[0], reverse=True)
    for _, index, previous, current in candidates[:6]:
        add(
            index,
            "slope_sign_change",
            f"相邻采样斜率变号，前段 {previous * 60.0:+.6g}/min，后段 {current * 60.0:+.6g}/min",
        )
    return output


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
                # 后两段是「重大」与「班次」，顺序固定但模型可能写少一段。
                # 实测最容易出的错是把班次直接顶到第三段（`today|today|白班`）——
                # 那样会被当成 major_only 的标记，班次就丢了、还白拿一批重大事件。
                # 所以第三段按**内容**认：认得出是班次就挪到班次位。
                tail = parts[2:]
                shift = ""
                for index, part in enumerate(tail):
                    name = _plan_shift(part)
                    if name:
                        shift = name
                        del tail[index]
                        break
                plan["logs"] = {
                    "start": start, "end": end,
                    "major_only": any("重大" in part or "重要" in part for part in tail),
                    "shift": shift,
                }
            continue

    return plan


def _empty_capabilities() -> dict[str, Any]:
    """保守默认：照常检索、不读任何实时数据。

    规划失败时必须退到这里——多查一次数据只是慢，少查一次却会让模型
    在缺少事实的情况下作答。
    """
    return {"needs_rag": True, "points": [], "history": None, "trend": None, "logs": None}


# 规划器调用实测约 2~12% 的概率失败（LLM 侧偶发 HTTP 400 空响应体）。
# **必须重试**，因为失败的后果不是「答得糙一点」，而是把「能答」变成「答不了」：
# 兜底计划不取任何实时数据，用户问「轴封供气压力的趋势」，模型只能回
# 「缺少实时数据」——而数据明明在。LLM 客户端内部已重试过一次，
# 那一次是紧挨着重放的；这里隔开再试，覆盖的是稍纵即逝的那类抖动。
_PLANNER_ATTEMPTS = 2

# 规划器被要求输出的行数：检索／测点／历史／趋势／日志。
_PLAN_REQUIRED_LINES = 5


def _plan_is_complete(text: str) -> bool:
    """规划器的输出是否具备应有的结构。

    缺行说明它这次没把问题读完——那种情况下它对「要取哪些数据」的判断不可信，
    值得重试一次。

    **判的是结构完整性，不是内容像不像某个意图**。内容判断是这个模型调用本身
    该干的事；再拿一张词表去猜一遍，只会把它的结论覆盖掉，而且词表对换种问法
    就失效。这里只问一个模型答不出来的问题：你按格式答了吗。
    """
    return len([line for line in str(text or "").splitlines() if line.strip()]) >= _PLAN_REQUIRED_LINES


def _plan_has_realtime(plan: dict[str, Any]) -> bool:
    return bool(plan.get("points") or plan.get("history") or plan.get("trend") or plan.get("logs"))


# ── 数字核对：答案里的数在证据里找得到吗 ──
#
# 填空类问题最容易出错的地方是数字，而「这个数在证据里有没有」**是可以直接查的**
# ——不需要模型自我评估，也不需要语义判断。
#
# 只作提示，**不删改答案**。查出无出处的数不代表它一定错（可能是单位换算、
# 可能是常识性数字），但它是最值得人工核对的地方。
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
# 单个数字（0-9）在中文行文里多半是序号或量词，查它只会制造噪声。
_NUMBER_MIN_DIGITS = 2


def _unsupported_numbers(answer: str, evidence_text: str) -> list[str]:
    """答案里出现、但检索证据里找不到的数字。

    机械校验。判据就是「这个字符串在证据里出现过没有」，含尾零的写法
    （`11.80` 与 `11.8`）按归一化后的值再比一次。
    """
    haystack = str(evidence_text or "")
    missing: list[str] = []
    for raw in _NUMBER_RE.findall(str(answer or "")):
        if len(raw.replace(".", "")) < _NUMBER_MIN_DIGITS:
            continue
        normalized = raw.rstrip("0").rstrip(".") if "." in raw else raw
        if raw in haystack or normalized in haystack:
            continue
        if raw not in missing:
            missing.append(raw)
    return missing


# ── 检索闸门：带图必须走知识库 ──
#
# 实测：用户传一张试卷照片问「做一下这张卷子」，规划器只看到这句纯文本，
# 判不出知识需求 → `检索: no` → `_retrieve_planned` 直接返回空 contexts，
# `_plan_rag` 与多轮检索根本不执行。6/6 次都这样，是确定性误判不是抖动。
# 后果不是「答得糙」，是模型手里一份证据都没有，只能用自己的常识答题——
# 于是答出满篇「（或10，视电厂具体要求）」这类猜测。电厂场景里这很危险，
# 因为它看起来像答案。
#
# 修法是**只向「要检索」单向覆写**：带图时把 False 改成 True。
# 反向不成立，所以「你好」「今天天气怎么样」这类正确跳过在结构上不可能被破坏。
#
# **刻意不按关键词判断提问类型**。曾经写过一份「试卷专有词」表，
# 但那是错的方向：本系统要的是「判断该不该检索」这个**能力**，试卷只是它的
# 一个用例。词表对换种问法就失效，而且规则散落在这里长不了。
# README 的「设计要点」里已经为路由写过同样的结论：不用关键词启发式。
_VISION_FORCE_RAG = os.getenv("HDW_VISION_FORCE_RAG", "true").lower() != "false"


async def _plan_retrieval(
    question: str,
    mode: Literal["online", "offline"],
    now: datetime | None = None,
) -> dict[str, Any]:
    """一次调用同时决定「要不要查文档」和「要取哪些实时数据」，判断发生在检索之前。

    极小调用：无证据、只出五行，实测约 200-400ms。
    失败时重试一次；仍失败才退回保守默认（照常检索、不取实时数据），
    并在返回值里带上 `degraded` 说明原因——静默退化会让下游把「取数失败」
    当成「没有数据」来讲。
    """
    if not _ROUTER_ENABLED:
        return _empty_capabilities()
    # 规划器此前只知道「今天是几号」，「今天白班」这类说法最多只能落到整天。
    # 现在把时刻与班次一并给它，日期段照旧、班次落到新加的第四段。
    moment = now or _local_now()
    today = moment.date()
    prompt = _PLANNER_PROMPT.format(
        now=moment.strftime("%Y-%m-%d %H:%M"),
        shift=_shift_of(moment),
        today=today.isoformat(),
        yesterday=(today - timedelta(days=1)).isoformat(),
    )
    failure: str | None = None
    for attempt in range(1, _PLANNER_ATTEMPTS + 1):
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
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            continue
        plan = _parse_plan(result.content)
        if plan["needs_rag"] is None:
            plan["needs_rag"] = True
        # 输出结构不完整（少了应有的行）说明这次没读全，再试一次。
        # 内容合不合预期**不判**——那是模型自己的结论，不该被外部的词表覆盖。
        if _plan_is_complete(result.content) or attempt == _PLANNER_ATTEMPTS:
            return plan
        failure = "规划输出结构不完整"
    plan = _empty_capabilities()
    plan["degraded"] = failure or "未知原因"
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


_POINT_ALIAS_PROMPT = (
    "用户想查一个电厂测点，但按他说的词在测点表里搜不到。\n"
    "测点表有自己的命名习惯，现场口语常常对不上——例如用户说「凝汽器水位」，"
    "表里写的是「凝汽器液位」；说「润滑油压」，表里是「润滑油压力」。\n"
    "给出 1~3 个更可能搜到的检索词，每行一个。"
    "只输出检索词本身，不要编号、不要解释。\n"
)


async def _point_aliases(keyword: str, mode: Literal["online", "offline"]) -> list[str]:
    """让模型给出更可能搜到的检索词。失败返回空列表，绝不抛异常。

    取代原来的「逐级去掉尾字」规则。那条规则只能把
    「凝汽器水位」放宽到「凝汽器水」「凝汽器」，**换词**是它做不到的
    （水位→液位、润滑油压→润滑油压力），而换词恰恰是不命中的主因——
    放宽到「凝汽器」能撞上纯属运气。
    """
    try:
        _, client = _llm_client(mode)
        result = await client.complete(
            [
                {"role": "system", "content": _POINT_ALIAS_PROMPT},
                {"role": "user", "content": keyword},
            ],
            max_tokens=64,
            temperature=0.0,
            chat_template_kwargs={"enable_thinking": False} if mode == "offline" else None,
        )
    except Exception:
        return []
    aliases: list[str] = []
    for raw in str(result.content or "").splitlines():
        term = _QUERY_PREFIX.sub("", raw.strip()).strip().strip("`\"'")
        if term and term != keyword and term not in aliases:
            aliases.append(term)
        if len(aliases) >= 3:
            break
    return aliases


async def _search_point_candidates(
    keyword: str, mode: Literal["online", "offline"]
) -> tuple[list[dict[str, Any]], str | None]:
    """取候选测点：原词搜一次，搜不到再让模型给别名搜。

    **只在原词搜空时才多花一次模型调用**——能直接搜到的占多数。
    注意这个分叉依据是**搜索结果本身**（有没有候选），不是关键词长什么样，
    所以它不是「按词形猜意图」那类会随命名习惯失效的规则。
    """
    seen: set[str] = set()
    attempted: list[str] = [keyword]

    async def search(term: str) -> list[dict[str, Any]]:
        data = await _mcp_call_tool(
            "point_query_search_points",
            {"query_text": term, "limit": _LIVE_POINT_CANDIDATES},
        )
        found: list[dict[str, Any]] = []
        for item in (data or {}).get("items") or []:
            kks = str(item.get("kks") or "").strip()
            if kks and kks not in seen:
                seen.add(kks)
                found.append(item)
        return found

    try:
        candidates = await search(keyword)
    except Exception as exc:
        return [], f"{keyword}: {_exc_text(exc)}"
    if candidates:
        return candidates, None

    for alias in await _point_aliases(keyword, mode):
        attempted.append(alias)
        try:
            candidates = await search(alias)
        except Exception:
            continue
        if candidates:
            return candidates, None
    return [], f"{keyword}: 测点表中无匹配项（试过：{'、'.join(attempted)}）"


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

        # **先确定是哪个测点，再按 KKS 取值**。
        # 原来是「按 query_text 直接取 top-1，再用字符规则校验相关性」——
        # 那条规则把「相关性」简化成了字面重叠，而 tool 内部本就是按相似度取
        # top-1，模糊匹配可能返回完全无关的测点（实测搜「轴封供汽压力」top-1
        # 是「高压主蒸汽压力3选1后」）。拿到 KKS 再取值，这一步就没有歧义了。
        point, error = await _resolve_point(keyword, mode)
        if error or not point:
            return None, error or f"{keyword}: 未匹配到测点"
        try:
            data = await _mcp_call_tool(
                "point_query_current_value", {"kks": str(point.get("kks") or "")}
            )
        except Exception as exc:
            return None, f"{keyword}: {_exc_text(exc)}"
        if not isinstance(data, dict) or data.get("value") is None:
            return None, f"{keyword}: 选中测点无实时值"
        return shape(data), None

    results = await asyncio.gather(*(one(keyword) for keyword in keywords))
    return (
        [item for item, _ in results if item],
        [error for _, error in results if error],
    )

def _round_robin_by_round(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按检索轮次轮转取一条，组内保持原（分数）顺序。

    为什么需要：合并后的排序键是 `(score, -round, -position)`，纯按分数。
    简答题那组往往条数多、分数高，会把填空题那组的证据整体挤出窗口——
    而 `_llm_contexts` 是**头切**，所以只要重排，头 N 条自然组组有份。
    重排而不新增截断点，是为了不动既有那两条对 `_llm_contexts` 的断言。
    """
    buckets: dict[int, list[dict[str, Any]]] = {}
    order: list[int] = []
    for item in contexts:
        index = int(item.get("_rag_round", 1) or 1)
        if index not in buckets:
            buckets[index] = []
            order.append(index)
        buckets[index].append(item)

    output: list[dict[str, Any]] = []
    position = 0
    while True:
        added = False
        for index in order:
            bucket = buckets[index]
            if position < len(bucket):
                output.append(bucket[position])
                added = True
        if not added:
            return output
        position += 1


# ── 检索规划：由模型决定发几路 ──
#
# 之前这里是两条启发式（MCP 的 `rag_query_plan` 用关键词表 + 正则切分），
# 都换掉了。原因不是它们不准，是**方向错了**：本系统要的是「判断该怎么检索」
# 这个能力，而不是「识别试卷」这个特例。规则对换种问法就失效，而且散落在
# 代码里长不了——README 的「设计要点」早已为路由写过同样的结论。
#
# 由模型决定的好处是它天然按**内容**切：一段材料里有一个问题就出一条，
# 有一份含多道题的卷子就按题出，一份并列的小问就按小问出。
# 不需要任何人预先定义「什么算多问」。
_RAG_QUERY_PLAN_PROMPT = (
    "你在为知识库检索拆查询。把用户要回答的内容拆成若干条**各自独立**的"
    "检索查询，每条都单独拿去检索。\n"
    "**先判断问题类型，再决定拆几条**——拆得多不是目的，覆盖到才是：\n"
    "· 问一个具体事实、一个参数、一个概念（「是多少」「什么是」「怎么处理」）——"
    "**1 条就够**，不要为了凑数拆开，多拆只会白花时间和上下文\n"
    "· 问「有哪些」「包括哪些」「都有什么」这类**列举**——"
    "**按该类设备的各个子项各拆一条**。一条查询只能覆盖一个子项："
    "实测问「汽轮机主保护有哪些」时，单条查询只会捞回「主蒸汽温度保护」一类"
    "（字面撞上「主…保护」），而「超速保护系统」「EH油压低保护」「差胀保护」"
    "这些逐项条目一条都进不了榜，漏掉的子项后面只能靠猜。"
    "例如这类问题就按 超速、轴承温度、润滑油压、EH油压、差胀、真空低、"
    "轴向位移、振动 等子项拆\n"
    "· 一段材料里有多个并列的问题时，**每个问题一条**\n"
    "· 每条的写法：写成**能直接检索的查询**，保留区分性的限定词"
    "（设备名、部位、参数名），去掉「请简述」「是多少」这类问句外壳\n"
    "· **不要**输出「相关定义、范围和判断依据」这类空泛查询，它们检不到东西\n"
    "· 最多 {max_queries} 条；超出就把相邻的合并成一条\n"
    "只输出查询本身，每行一条。不要编号、不要解释、不要空行。\n"
)

_RAG_MAX_QUERIES = int(os.getenv("HDW_RAG_MAX_QUERIES", "60"))
# 每一路取几条证据。总量靠 `_retrieval_evidence_budget` 兜底。
_RAG_QUERY_TOP_K = int(os.getenv("HDW_RAG_QUERY_TOP_K", "8"))
# 证据总预算的上限。一路约 400 字，80 条约 3 万字，离上下文上限还很远。
_RAG_ONLINE_EVIDENCE_CAP = int(os.getenv("HDW_RAG_ONLINE_EVIDENCE_CAP", "120"))
_RAG_LOCAL_EVIDENCE_CAP = int(os.getenv("HDW_RAG_LOCAL_EVIDENCE_CAP", "80"))
# 落到本机 CPU RAG（串行，实测 7.5 s/路）时，最多保留几路检索。
# 远端 GPU 节点可用时不生效——那边 8 路并发只要 3.7 s，没必要压。
_RAG_FALLBACK_ROUTE_CAP = int(os.getenv("HDW_RAG_FALLBACK_ROUTE_CAP", "2"))

# 模型偶尔会加行首编号，或者在行尾带解释。剥掉编号；解释留着也无害
# （检索器对长句本来就有截断），所以不做更激进的清洗。
_QUERY_PREFIX = re.compile(r"^\s*(?:[-*·]\s*|\d{1,2}\s*[.、．)）]\s*)")


def _parse_query_plan(text: str) -> list[str]:
    """把模型的输出解析成查询列表。纯函数，可进自检。"""
    queries: list[str] = []
    for raw in str(text or "").splitlines():
        line = _QUERY_PREFIX.sub("", raw.strip()).strip()
        line = line.strip("`\"' ").strip()
        if not line or line in queries:
            continue
        queries.append(line)
        if len(queries) >= _RAG_MAX_QUERIES:
            break
    return queries


async def _plan_queries(
    question: str, mode: Literal["online", "offline"]
) -> tuple[list[str], str | None]:
    """让模型决定要发几路检索。**任何失败都退回单路**，绝不抛异常。"""
    prompt = _RAG_QUERY_PLAN_PROMPT.format(max_queries=_RAG_MAX_QUERIES)
    failure: str | None = None
    for attempt in range(1, _PLANNER_ATTEMPTS + 1):
        try:
            _, client = _llm_client(mode)
            result = await client.complete(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": question},
                ],
                # 60 条查询每条约 20-40 字，4096 够用；再多说明拆得过细了。
                max_tokens=4096,
                temperature=0.0,
                chat_template_kwargs={"enable_thinking": False} if mode == "offline" else None,
            )
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            continue
        queries = _parse_query_plan(result.content)
        if queries:
            return queries, None
        failure = "模型没有给出任何查询"
    return [question], failure


def _retrieval_evidence_budget(
    mode: Literal["online", "offline"], rounds: int
) -> int:
    """证据总预算随检索路数增长。

    **必须跟着涨**：一路查询只覆盖一个问题，N 路就覆盖 N 个——预算不涨的话，
    后面几路检索到了也进不了提示词，等于白发。单路时它就等于原来的默认预算
    （10/40），所以不需要为「多轮」单独开一条路径。
    """
    base = settings.online_graph_top_k if mode == "online" else settings.local_graph_top_k
    cap = _RAG_ONLINE_EVIDENCE_CAP if mode == "online" else _RAG_LOCAL_EVIDENCE_CAP
    return max(base, min(cap, rounds * _RAG_QUERY_TOP_K))


async def _plan_rag(
    question: str,
    mode: Literal["online", "offline"],
    base_top_k: int,
) -> tuple[dict[str, Any], str | None]:
    queries, error = await _plan_queries(question, mode)
    # 单路时至少给到原来的预算：否则「不拆」反而比拆了拿到更少的证据，
    # 而单路正是最常见的路径，不该被这次改动顺带削弱。
    single_round = _graph_top_k(mode, base_top_k) if len(queries) == 1 else 0
    return {
        "rounds": len(queries),
        "queries": queries,
        "top_k": max(_RAG_QUERY_TOP_K, single_round),
        "base_top_k": _graph_top_k(mode, base_top_k),
        "reason": f"模型拆出 {len(queries)} 路检索",
        "planner": "model",
    }, error



# 测点**不存在**（而不是取数失败）时的标记。生产处与消费处共用同一个常量——
# 调用方要据此把「不检索」掰回「检索」，若各自 match 文案，改一次措辞就静默失效。
_POINT_NOT_FOUND = "测点不存在"


def _point_absent(realtime: dict[str, Any], errors: list[str]) -> bool:
    """本轮要了测点、但一个都没取到，且原因是**这个测点不存在**。

    区分「测点不存在」与「取数失败」很重要：后者重试一次可能就好，
    前者重试多少次都一样，该换条路走（去知识库里找）。
    """
    if realtime.get("live_points"):
        return False        # 取到了，问题本来就是问测点
    return any(_POINT_NOT_FOUND in str(item) for item in errors)


async def _resolve_point(
    keyword: str, mode: Literal["online", "offline"] = "offline"
) -> tuple[dict[str, Any] | None, str | None]:
    """中文关键词 → 具体测点。返回 (点, 错误)。

    **相关性完全由模型判**，不再做字符级判定。原来这里是「首二字与尾二字都必须
    出现在描述里」——它把「用户想查的是不是这个测点」简化成了字面重叠，
    既拦不住「轴封供汽压力 → 轴封供汽管道疏水母管气动关断阀开反馈」这类
    （首尾字都对，但根本不是测点），也挡不住同义换词。判断这件事模型做得更好，
    而它本来就有一次调用（`_pick_live_point`），不必再拿规则省。

    唯一保留的硬约束：**选出的 KKS 必须确实在候选集合内**——这是防幻觉，
    不是启发式。宁可报「没找到」，也不能给错测点的数据。
    """
    candidates, error = await _search_point_candidates(keyword, mode)
    if error or not candidates:
        return None, error or f"{keyword}: {_POINT_NOT_FOUND}（测点表中无匹配项）"

    picked = await _pick_live_point(keyword, candidates, mode)
    for item in candidates:
        if str(item.get("kks")) == picked:
            return item, None
    return None, f"{keyword}: {_POINT_NOT_FOUND}（候选中无相关测点）"


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
        return None, f"{keyword}: {_exc_text(exc)}"
    if not isinstance(data, dict) or not data.get("samples"):
        return None, f"{keyword}: 该时段无采样数据"
    return _shape_series(data, keyword, kind), None


def _shift_expected(shift: str, day: date, now: datetime) -> bool:
    """`day` 那天的该班次是否**应该已经有记录**了。

    上游的完整性判据是「三个班次全齐才算 complete」，**完全不看现在几点**——
    白班时段查当天，中班和夜班尚未发生，于是每个日志类型都被判为不完整。
    实测每次问「今天有什么事」，答案开头都挂着「缺中班记录」，而那段时间
    根本还没到。这里按班次开始时刻把「还没到」摘出去。

    **还要一段宽限期**：班次刚开始的几分钟里没有记录同样是正常的。
    实测 16:34 查当天，中班 16:30 才开始 4 分钟，仍旧被报成「缺中班记录」——
    摘掉「还没到」还不够，得摘掉「刚到」。
    """
    try:
        begin, _ = _shift_window(shift, day)
    except ValueError:
        return True          # 认不出的名字就当它该有，照常报缺失
    # 两边都是北京时间的墙上时钟，去掉时区直接比。
    return now.replace(tzinfo=None) >= begin + timedelta(minutes=_SHIFT_GRACE_MINUTES)


def _logs_completeness_line(completeness: dict[str, Any], now: datetime) -> str:
    """把日志完整性渲染成一行。

    结构：每天 → 各日志类型（值长／#1主值／#2主值／化学）→ 缺哪些班次。
    **只报缺失项**：全拿到的逐条念出来，这一行会比日志本身还长，
    而它唯一的作用就是提示「哪块数据不可信」。

    「尚未到来的班次」不算缺失——见 `_shift_started`。
    """
    parts: list[str] = []
    for day in completeness.get("dates") or []:
        if not isinstance(day, dict):
            continue
        date_text = str(day.get("date") or "").strip()
        try:
            day_date = date.fromisoformat(date_text)
        except ValueError:
            day_date = None          # 日期认不出就不过滤，宁可多报
        missing: list[str] = []
        suppressed = 0
        for program in day.get("byProgram") or []:
            if not isinstance(program, dict) or program.get("complete"):
                continue
            label = str(program.get("programLabel") or "日志").strip()
            gone: list[str] = []
            for item in program.get("missingShifts") or []:
                name = str(item)
                if day_date is not None and not _shift_expected(name, day_date, now):
                    suppressed += 1
                    continue
                gone.append(name)
            if gone:
                missing.append(f"{label}缺{'、'.join(gone)}")
        if missing:
            parts.append(f"{date_text} " + "、".join(missing))
        elif suppressed:
            # 该报的都到了，只是还有班次没开始——说清楚，别让读的人
            # 以为「齐全」是指一整天都齐了。
            parts.append(f"{date_text} 截至当前班次已到的各类型齐全")
        else:
            parts.append(f"{date_text} 各类型各班次齐全")
    return "；".join(parts)


def _logs_failure_reason(data: dict[str, Any]) -> str:
    """日志工具返回体若是「取数失败」，给出可读原因；正常返回空串。

    **顺序是关键**：失败判定必须早于「有没有 events」。抓取失败时 events 同样是空的，
    反过来就会把「取数失败」讲成「该时段无记录」——前者重试一次可能就好，后者意味着
    数据本来就不存在，对用户是相反的两件事。

    工具侧的约定是 `ok = not (fetch_failed and not payloads)`，所以 `ok=False` 恰好
    表示「抓取失败且本地也没有缓存」，与「确实没记录」互斥。
    """
    if str(data.get("status")) == "unavailable":
        return str(data.get("message") or "数据源未配置")
    if data.get("ok") is False:
        fetch_result = data.get("fetch_result") or {}
        # 类别名（哪个数据源、哪类错误）和具体原因**两个都要**：
        # query_status 说明是哪条链路挂了（如 liems_http_error），
        # stderr 说明挂在哪一步（连不上、超时多久）。只留一个都不够定位。
        detail = str(data.get("message") or "").strip() or " ".join(
            str(part).strip()
            for part in (
                data.get("query_status"),
                fetch_result.get("stderr") or data.get("fetch_error_type"),
            )
            if str(part or "").strip()
        )
        return f"取数失败（{' '.join((detail or '原因未知').split())[:200]}）"
    return ""


# 日志事件的班次代码 → 中文名。与 Log_Fetching 侧的 SHIFT_LABELS 一致。
_SHIFT_CODES = {"YXBB": "白班", "YXZB": "中班", "YXYB": "夜班"}

# 事件时刻的候选格式。LIEMS 返回的 `time` 是原样字符串，没有归一化：
# 可能带日期、可能只有时刻、也可能为空。
_EVENT_TIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M",
)


def _event_moment(event: dict[str, Any]) -> datetime | None:
    """事件时刻；解析不出返回 None（**不猜**）。

    `time` 可能自带日期，也可能只有 `HH:MM`——后者要用事件自带的 `date` 补。
    """
    text = str(event.get("time") or "").strip()
    if not text:
        return None
    day = str(event.get("date") or "").strip()
    candidates = [text] if re.search(r"\d{4}[-/.]\d{1,2}", text) else []
    if day:
        candidates.append(f"{day} {text}")
    for candidate in candidates:
        for fmt in _EVENT_TIME_FORMATS:
            try:
                return datetime.strptime(candidate, fmt)
            except ValueError:
                continue
    return None


def _event_shift(event: dict[str, Any]) -> str:
    """事件属于哪个班次；判不出返回空串。

    **两条腿，缺一不可**（实测）：
    1. `shiftCode` 是**日志表主行**的班次（YXBB白／YXZB中／YXYB夜），
       这正是「白班有什么事」想问的口径——一条 07:50 的事记在白班表里，
       它就属于白班。但它只出现在**值班记事**与**主表关键字段**两类事件上。
    2. **运行交代**与**接地线记录**的字典里根本没有这个键，而接地线全部标为
       major——纯按 shiftCode 过滤会把这两类整体丢掉。对它们退回按时刻判定。

    第 2 步同样不可尽信：主表关键字段的 `time` 是整张日志表的时间戳
    （BFFORM_DTM），不是逐字段时间。所以判不出就返回空串，由调用方 fail-open。
    """
    code = str(event.get("shiftCode") or "").strip()
    name = _SHIFT_CODES.get(code, "")
    if name:
        return name
    moment = _event_moment(event)
    return _shift_of(moment) if moment is not None else ""


def _filter_events_by_shift(
    events: list[dict[str, Any]], shift: str
) -> tuple[list[dict[str, Any]], int]:
    """只留指定班次的事件。返回 (保留的, 其中判不出班次的条数)。

    **判不出的一律保留**（fail-open）：静默丢弃比多给更危险——接地线记录
    全是重大事件，被过滤掉之后用户看到的是「这天没什么事」，
    而缺的那块从答案表面完全看不出来。`unknown` 回传给提示词明示。
    """
    kept: list[dict[str, Any]] = []
    unknown = 0
    for event in events:
        name = _event_shift(event)
        if not name:
            unknown += 1
            kept.append(event)
        elif name == shift:
            kept.append(event)
    return kept, unknown


async def _fetch_logs(
    spec: dict[str, Any],
    full: bool = False,
) -> tuple[dict[str, Any] | None, str | None]:
    """取运行日志。按规划给出的区间选择工具，只取重大事件时走 major_events。

    `full=True`（本轮不检索文档，日志就是答案主体）时不截断明细。
    """
    start, end = str(spec.get("start") or ""), str(spec.get("end") or "")
    major_only = bool(spec.get("major_only"))
    shift = _plan_shift(str(spec.get("shift") or ""))
    # **必须显式传 limit**。不传就落到工具自己的默认值——实测
    # log_query_recent 默认 80、log_query_range 默认 120，于是「抓 161 条」
    # 在中途已经先被压到 120，再被 _LOGS_MAX 截到 40。工具侧钳制上限是
    # 500（recent / major_events）/ 800（range），这里按 500 要。
    limit = {"limit": max(1, _LOGS_FETCH_LIMIT)}
    try:
        today = _local_today().isoformat()
        if major_only:
            # major_events 只接受天数，不接受区间，按区间长度折算
            span = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
            data = await _mcp_call_tool(
                "log_query_major_events",
                {"days": max(1, min(_QUERY_MAX_DAYS, span)), **limit},
            )
        elif start == end and end == today:
            data = await _mcp_call_tool("log_query_recent", {"days": 1, **limit})
        else:
            # **必须传 start_date / end_date，不能传 days**。
            # `log_query_range` 的签名是 (query_text, start_date, end_date, ...)，
            # 根本没有 days 这个参数——传了会被丢掉，日期留空后 `_resolve_dates`
            # 退化成 start = end = 今天，于是**只查一天**。
            # 实测：规划器算出 5 天区间（09-11~09-16），工具只返回 09-15 一天，
            # 答案里的「最近五天」其实是照一天写的。
            span = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
            if span > _QUERY_MAX_DAYS:
                # 超长区间往前收窄，终点保持用户要的那天。
                start = (
                    date.fromisoformat(end) - timedelta(days=_QUERY_MAX_DAYS - 1)
                ).isoformat()
            data = await _mcp_call_tool(
                "log_query_range", {"start_date": start, "end_date": end, **limit}
            )
    except Exception as exc:
        return None, f"日志: {_exc_text(exc)}"
    if not isinstance(data, dict):
        return None, "日志: 返回格式异常"
    failure = _logs_failure_reason(data)
    if failure:
        return None, f"日志: {failure}"
    events = data.get("events") or []
    if not events:
        return None, "日志: 该时段无记录"
    # 先在筛选**之前**记下原始规模：筛完班次之后仍要能看出「这一天一共多少条」。
    raw_total = len(events)

    # 班次筛选发生在截断**之前**：截断要作用在用户真正要看的那个班次上。
    shift_unknown = 0
    if shift:
        events, shift_unknown = _filter_events_by_shift(events, shift)
        if not events:
            return None, f"日志: {start}~{end} 无{shift}记录（该时段其余事件均属其他班次）"

    # 全量模式（日志类问题）不截断；其余维持 _LOGS_MAX，避免挤占文档证据。
    # _LOGS_MAX_FULL 留了正值就可以收紧，便于不改代码回退。
    cap = 0
    if not full:
        cap = max(0, _LOGS_MAX)
    elif _LOGS_MAX_FULL > 0:
        cap = _LOGS_MAX_FULL
    ordered = sorted(
        events,
        key=lambda item: {"major": 0, "important": 1}.get(str(item.get("severity")), 2),
    )
    if cap > 0:
        ordered = ordered[:cap]

    return {
        "kind": "logs",
        "start": start,
        "end": end,
        "major_only": major_only,
        # 本次是否全量注入。提示词按它决定措辞与是否再渲染逐日分栏。
        "full": full,
        "shift": shift,
        # 班次筛选时判不出班次、因而被保留的条数。**要明示**：
        # 不说明的话，「白班」的答案里混进别班条目也看不出来。
        "shift_unknown": shift_unknown,
        "truncated": bool(cap) and len(ordered) < len(events),
        # 本时段（**筛班次之前**）的总条数，与 summary.total_events 同源。
        # 与 injected 分开是为了让「白班筛出 61 条」这句话里的分母
        # （全厂合计 91 条）随时可查——合成一个数就说不清了。
        "total_events": raw_total,
        # 实际进入提示词的条数。截断与班次筛选之后的结果。
        "injected": len(ordered),
        "summary": data.get("summary") or {},
        # 按天分好类的全量工作摘要。**工具已经算好了，之前一直没用**——
        # 明细受 _LOGS_MAX 截断（实测一次抓取 161 条、只注入 40 条），
        # 模型看不到的 121 条只能靠这个体现：每天的各类条数都在里面，
        # 重要的逐条列出，次要的至少让模型知道「那天还有多少件事」。
        "work_digest": data.get("work_digest") or {},
        # 日志完整性：按日期 × 日志类型（值长／#1主值／#2主值／化学）× 班次
        # 记录哪些拿到了。**不说清完整性 = 把不完整的日志讲成完整的**——
        # 实测「最近五天」那次，模型根本不知道它只拿到了一天。
        "log_completeness": data.get("log_completeness") or {},
        # 118 条全量注入会挤占文档证据，按严重程度排序后截断
        "events": ordered,
    }, None


def _build_chart(item: dict[str, Any], sig: dict[str, Any]) -> dict[str, Any] | None:
    """组装前端绘图所需的结构。

    结构对齐 SmartGasTurbine 的 metadata.charts：
      · series          降采样后的真实采样点（画实线）
      · fit_series      最小二乘拟合曲线（画虚线）——**仅供视觉参考**
      · special_points  极值点与斜率变号点（画圆点/菱形）
      · summary/point   卡片上的指标

    序列降采样到 _CHART_MAX_POINTS：原始点可达上万，前端 SVG 画不完也不必画。
    极值点与变号点**在降采样前**算出并保留，它们是这张图最该被看到的东西。
    """
    samples = item.get("samples") or []
    points: list[tuple[float, float]] = []
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
        points.append((moment.timestamp(), float(value)))
    if len(points) < 2:
        return None

    special = _special_points(points)
    fit = _fit_series(points)

    step = max(1, len(points) // _CHART_MAX_POINTS)

    def to_iso(ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    # 显著性档位决定「这张图给到什么程度」（阈值见 _series_significance）：
    #   level 2（≥3%）   曲线 + 拟合线 + 关键点。关键点才是这张图值得看的地方。
    #   level 1（1~3%）  曲线 + 拟合线，**不给关键点**——低位波动里的「极值/变号」
    #                    多是采集噪声，标出来等于在图上替用户断言一件没发生的事。
    #   level 0（<1%）   只给曲线。拟合线也是一种「趋势」断言，整段没有趋势时
    #                    画一条斜线会误导人读出根本不存在的走势。
    #   level -1（判不了）按 level 2 处理：判不了时保守给全，宁可多报。
    # 注意裁剪的只是**图元**。喂给模型的材料由 _prompt 单独按 level 裁，两者
    # 口径一致但互相独立——人看到的和模型看到的本来就不必相同。
    level = sig.get("level")
    show_fit = level != 0
    show_special = level not in (0, 1)

    return {
        "kind": item.get("kind") or "history",
        "point": {
            "kks": item.get("kks") or "",
            "description": item.get("description") or item.get("query") or "",
            "unit": item.get("unit") or "",
        },
        "series": [{"time": to_iso(ms), "value": val} for ms, val in points[::step]],
        "fit_series": fit if show_fit else [],
        "special_points": special if show_special else [],
        # 指标直接从点算。**不要从 sig 取**——`_series_significance` 返回的是
        # 分位数与显著性判据，没有 min/max/delta，取了就是三个 None。
        "summary": {
            "latest_value": points[-1][1],
            "min_value": min(value for _, value in points),
            "max_value": max(value for _, value in points),
            "delta_value": points[-1][1] - points[0][1],
            "samples": len(points),
            "level": sig.get("level"),
            "discrete": sig.get("discrete"),
        },
    }


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


# 进度上报回调：(stage, label) -> None。**同步**函数——它只是往队列里塞一条，
# 不涉及 IO，所以链路深处那些不 async 的地方也能调。
ProgressFn = Callable[[str, str], None]


def _noop_progress(stage: str, label: str) -> None:
    """非流式请求（普通 JSON 返回）没有订阅者，用这个占位，省去每处判空。"""


async def _fetch_realtime(
    plan: dict[str, Any],
    mode: Literal["online", "offline"],
    progress: ProgressFn = _noop_progress,
    logs_full: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """并行取齐规划要求的全部实时数据。

    三类查询互不依赖，且都远慢于本地计算（日志实测约 5.8 s），
    串行执行会让时延叠加；并发后总耗时取决于最慢的一项。

    **单项失败只记录不抛出**：实时数据缺失不能拖垮文档问答，这与
    `_fetch_live_points` 的既有约定一致。
    """
    tasks: dict[str, Any] = {}
    labels: dict[str, str] = {}
    if plan.get("points"):
        tasks["live"] = _fetch_live_points(plan["points"], mode)
        labels["live"] = "查询实时测点：" + "、".join(plan["points"])
    if plan.get("history"):
        spec = plan["history"]
        tasks["history"] = _fetch_series(spec, "history", mode)
        labels["history"] = f"查询历史序列：{spec.get('keyword')}（{spec.get('start')} ~ {spec.get('end')}）"
    if plan.get("trend"):
        spec = plan["trend"]
        tasks["trend"] = _fetch_series(spec, "trend", mode)
        labels["trend"] = f"查询变化趋势：{spec.get('keyword')}（{spec.get('start')} ~ {spec.get('end')}）"
    if plan.get("logs"):
        spec = plan["logs"]
        span = f"{spec.get('start')} ~ {spec.get('end')}"
        tasks["logs"] = _fetch_logs(spec, full=logs_full)
        labels["logs"] = (
            f"查询运行日志：{span}"
            + ("（仅重大事件）" if spec.get("major_only") else "")
            + (f"（{spec['shift']}）" if spec.get("shift") else "")
        )
    # 先报「要查什么」，再报「查到什么」。只报后者的话，在日志那 30 秒里
    # 用户看到的是上一句话停着不动——那正是最需要他看见在做事的时候。
    for name in tasks:
        progress("mcp", labels[name])

    result: dict[str, Any] = {"live_points": [], "series": [], "logs": None, "charts": []}
    errors: list[str] = []
    if not tasks:
        return result, errors

    done = await asyncio.gather(*tasks.values(), return_exceptions=True)
    for name, outcome in zip(tasks.keys(), done):
        if isinstance(outcome, BaseException):
            errors.append(f"{name}: {_exc_text(outcome)}")
            progress("mcp", f"{labels.get(name, name)} → 失败：{_exc_text(outcome)}")
            continue
        if name == "live":
            points, point_errors = outcome
            result["live_points"] = points
            errors.extend(point_errors)
            progress(
                "mcp",
                f"{labels['live']} → {len(points)} 个测点"
                + (f"，{len(point_errors)} 个失败" if point_errors else ""),
            )
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
                # 失败也要报，而且要报出原因——「一直转圈最后什么都没有」最难受。
                # 原因文本来自 _exc_text / _fetch_logs，保证非空。
                progress("mcp", f"{labels.get(name, name)} → {error}")
            elif item:
                if name == "logs":
                    result["logs"] = item
                    events = item.get("events") or []
                    progress("mcp", f"{labels['logs']} → {len(events)} 条记录")
                else:
                    # 组装图表对象供前端内联绘制 SVG。
                    # **不出 PNG**：矢量图能自适应宽度、可交互，且不必把
                    # base64 图片塞进 JSON（一张 53KB 的图会让响应体大一个量级）。
                    # SmartGasTurbine 也是这个做法（见其 llm_inference.html 的
                    # renderTrendChartSvg）。
                    _vals = [s.get("value") for s in (item.get("samples") or []) if s.get("value") is not None]
                    _sig = _series_significance(_vals, item.get("limits"))
                    chart = _build_chart(item, _sig)
                    # 低波动也要出图——图是给人看的，人有权看到真实曲线；
                    # 「别去解读」是靠裁掉图元与提示词材料实现的，不是靠不给图。
                    if chart:
                        result["charts"].append(chart)
                    result["series"].append(item)
                    progress(
                        "mcp",
                        f"{labels.get(name, name)} → {len(item.get('samples') or [])} 个采样点",
                    )
    return result, errors


async def _retrieve_planned(
    question: str,
    mode: Literal["online", "offline"],
    top_k: int,
    *,
    has_images: bool = False,
    progress: ProgressFn = _noop_progress,
    now: datetime | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # 规划在检索之前：一次调用同时决定要不要查文档，以及要取哪几类实时数据
    # （测点当前值／历史序列／趋势／运行日志）。
    progress("plan", "分析问题、规划检索范围")
    intent = await _plan_retrieval(_truncate_for_planner(question), mode, now)
    # 把规划结论先报出来：后面几秒到几十秒的动作都是由它决定的，
    # 先说「打算做什么」，再看「做成了什么」，出问题时才知道该找哪一步。
    progress(
        "plan",
        "规划结果：" + ("需要检索知识库" if intent["needs_rag"] else "无需检索知识库")
        + (
            "；实时数据 " + "、".join(
                name
                for name, key in (("测点", "points"), ("历史", "history"), ("趋势", "trend"), ("日志", "logs"))
                if intent.get(key)
            )
            if _plan_has_realtime(intent)
            else "；不取实时数据"
        ),
    )

    # 单向覆写：只在模型判了「不检索」时把它拉回 True，永不反向。
    # 唯一依据是**本轮带了图片**——上传图片在这个系统里永远是「要处理的材料」
    # （试卷、图纸、仪表照片、铭牌），不存在「闲聊配图」这个场景。
    forced_reason: str | None = None
    if _VISION_FORCE_RAG and has_images and not intent["needs_rag"]:
        forced_reason = "本轮带图"
        intent["needs_rag"] = True

    preloaded: tuple[dict[str, Any], list[str]] | None = None
    if not intent["needs_rag"]:
        # 不检索文档，日志就是答案主体（问「今天有什么事」这类）——
        # 截断只会让答案凭空少掉内容，全量给。
        realtime, rt_errors = await _fetch_realtime(
            intent, mode, progress, logs_full=True
        )
        if _point_absent(realtime, rt_errors):
            # 规划器把它当成了「读某个测点的当前值」，而那个标识符**根本不是
            # 本厂测点**——实测「P.GAS.11是多少」：「是多少」被判成读测点，而
            # P.GAS.11 是运维手册里的参数名（SIS 映射表 607 个测点里没有这种
            # 形式）。此时直接返回等于这次问答**什么都没查**：用户拿到一句
            # 「没有取到实时值」，而手册里明明写着 >30 barg。
            #
            # 所以把「不检索」**单向掰回「检索」**，再走一遍正常链路。
            # 判据是取数的**实际结果**，不是提问的措辞——本文件另一处
            # （`_VISION_FORCE_RAG`）已写明：不用关键词启发式。
            intent["needs_rag"] = True
            forced_reason = "测点不存在，回退检索知识库"
            preloaded = (realtime, rt_errors)
        else:
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
                "live_errors": _live_errors(intent, rt_errors),
                **realtime,
            }
    plan, planning_error = await _plan_rag(question, mode, top_k)
    if forced_reason:
        # 覆写要留痕：否则「模型判了不检索、被网关拉回来」这件事在响应里看不出来，
        # 排查时无法区分「检索了」和「本该跳过却检索了」。
        plan["reason"] = f"{plan.get('reason') or ''}（本轮强制检索：{forced_reason}）".strip()
    if plan.get("no_retrieval"):
        # 不检索也不查图谱：下游拿到空 contexts 会自然产出空 citations / graph_context，
        # 前端 addEvidence() 在两者都空时不渲染依据面板。
        realtime, rt_errors = await _fetch_realtime(
            intent, mode, progress, logs_full=True
        )
        return [], {
            "plan": plan,
            "rounds": [],
            "deduplicated_count": 0,
            "no_retrieval": True,
            "live_errors": _live_errors(intent, rt_errors),
            **realtime,
        }

    # **实时数据与检索并行**。二者互不依赖，且日志/历史抓取实测约 5.8 s，
    # 与检索轮次的 6 s 同量级——串行会让时延直接叠加，并发则相互掩盖，
    # 总耗时取决于较慢的一项而不是两者之和。
    if preloaded is not None:
        # 上面已经取过一次（测点未命中的那次），**直接复用**：再跑一遍只会
        # 得到同样的错误，白等一次测点/MCP 查询。
        realtime_task: asyncio.Future = asyncio.get_running_loop().create_future()
        realtime_task.set_result(preloaded)
    else:
        realtime_task = asyncio.create_task(_fetch_realtime(intent, mode))

    # **回退路径要压路数**。`_rag_route_order` 会把多路轮询分到两个远端节点，
    # 但**每个节点内部是完全串行的**：`_search_zvec` 每次请求都要拿
    # `_MODEL_LOCK` 两次（_embed 一次、_rerank 一次），锁覆盖整个 GPU 推理。
    # 实测（2026-09-18，32 请求/档）：
    #     单节点 任意并发档位   吞吐恒 2.15 req/s，延迟随并发线性增长，加速比 1.0×
    #     两节点轮询           4.24 req/s（正好翻倍——并发能力来自节点**数量**）
    #     单请求延迟           远端 0.46 s vs 本机 CPU 7.55 s（差 16 倍）
    # 所以多路的耗时就是「路数 × 单请求延迟」，两条路径都逃不掉：
    #     10 路   生产路径 2.36 s      本机回退 75.5 s
    # 远端一挂，同一串数字涨 32 倍——正好在用户等答案的时候雪上加霜。
    # 只在路数够多时才探测（1-2 路本来就没多少可省的），探测本身有成本。
    if len(plan["queries"]) > _RAG_FALLBACK_ROUTE_CAP and not await _remote_rag_available():
        _dropped = len(plan["queries"]) - _RAG_FALLBACK_ROUTE_CAP
        plan["queries"] = plan["queries"][:_RAG_FALLBACK_ROUTE_CAP]
        plan["reason"] = (
            f"{plan.get('reason') or ''}（远端 RAG 不可用，落到本机串行路径，"
            f"路数已从 {_RAG_FALLBACK_ROUTE_CAP + _dropped} 压回 {_RAG_FALLBACK_ROUTE_CAP}）"
        ).strip()
        progress(
            "plan",
            f"远端检索不可用，本机串行执行，路数压回 {_RAG_FALLBACK_ROUTE_CAP} 路",
        )

    merged: dict[str, dict[str, Any]] = {}
    round_infos: list[dict[str, Any]] = []
    # **并发检索**。逐题检索后轮次可以到几十，串行会把这些延迟直接叠加
    # （实测 24 路串行 10.2 s、并发 5.5 s）。检索彼此不依赖，没有串行的理由。
    # 注意结果要按下标回填——`gather` 保序，但仍显式带上 round_index，
    # 免得以后有人改了写法把轮次对应关系弄错。
    outcomes = await asyncio.gather(
        *(_retrieve(query, int(plan["top_k"])) for query in plan["queries"]),
        return_exceptions=True,
    )
    for round_index, (query, outcome) in enumerate(zip(plan["queries"], outcomes), 1):
        if isinstance(outcome, BaseException):
            # 单路失败不该毁掉整轮：其它路仍有证据可答。这与原来的串行版本
            # 行为不同（那时一路抛异常会直接冒到调用方），是有意放宽的。
            round_infos.append(
                {"round": round_index, "query": query, "count": 0,
                 "backend": "", "url": "", "error": f"{type(outcome).__name__}: {outcome}"}
            )
            continue
        contexts, info = outcome
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
    if len(plan.get("queries") or []) > 1:
        # 多路检索才重排：纯按分数排序会让某一路（往往是条数多、分数高的那路）
        # 把别路的证据整体挤出窗口，而 `_llm_contexts` 是**头切**——
        # 挤出去的那几路等于白发。单路检索没有这个问题，保持原排序。
        contexts = _round_robin_by_round(contexts)
    info = {
        "plan": plan,
        "rounds": round_infos,
        "deduplicated_count": len(contexts),
        "live_errors": _live_errors(intent, rt_errors),
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


def _exc_text(exc: BaseException) -> str:
    """异常的可读描述，**保证非空**。

    httpx 的 ReadTimeout / ConnectTimeout / WriteTimeout 等 `str()` 就是空串
    （`repr(e)` 是 `ReadTimeout('')`）。直接写 `f"日志: {exc}"` 会得到一句
    「日志: 」——冒号后面什么都没有，读的人既不知道是什么错、也不知道该去查哪里。
    类型名至少能指出方向。
    """
    detail = str(exc).strip()
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


def _live_errors(intent: dict[str, Any], rt_errors: list[str]) -> list[str]:
    """把规划器降级的原因并进实时数据错误里。

    降级时「没有实时数据」和「取数失败」在下游长得一样，但应对方式完全不同：
    前者要换个问法，后者重试一次可能就好了。不点明的话，用户只会看到一句
    「缺少实时数据」——听上去像数据本来就不存在。
    """
    degraded = intent.get("degraded")
    return ([f"规划器降级（本次未取实时数据）：{degraded}"] if degraded else []) + list(rt_errors)


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
    budget: int | None = None,
) -> list[dict[str, Any]]:
    limit = budget if budget is not None else _graph_top_k(mode, len(contexts))
    return contexts[: max(1, limit)]


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


# 历史相关性选留。目的：让**与当前问题无关的轮次不进 prompt**。
# 上下文长度直接决定解码速度（短约 150 t/s，5 万 token 以上掉到约 100），
# 而历史是一轮一轮线性堆上去的。
#
# 与 _compress_history 的分工：压缩是**兜底**（选留之后仍然超长时按
# 75% 阈值出手，行为与参数都不变）；选留是**日常**手段，每轮都跑。
_HISTORY_SELECT_PROMPT = (
    "下面给出多轮对话的历史轮次和用户当前的问题。判断哪些历史轮次与当前问题相关、"
    "需要带入上下文。\n"
    "· 依据是**主题是否相关**，不是时间是否接近——紧挨着的上一轮也可能完全无关\n"
    "· 当前问题里出现「它 / 这个 / 那个 / 上面说的 / 刚才的」这类指代时，"
    "**必须保留被指代的那一轮**，哪怕它看起来和当前主题不同\n"
    "· 拿不准就保留：少带一轮只是答得差一点，丢错上下文会答错\n"
    "· 只输出轮次编号，每行一个。不要解释，不要别的文字\n"
    "· 一轮都不相关就只输出：无\n"
)

# 交给选择器的答案摘录长度。只给问题不够——「那它的定值是多少」这种指代
# 光看问题判不出来；给全文又太贵。80 字足够看出这一轮在讲什么。
_HISTORY_SELECT_ANSWER_CHARS = 80


def _group_turns(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """把扁平的 role/content 列表切成轮次。

    一轮 = 一个 user 消息 + 其后紧随的 assistant 消息。**必须按轮选**，
    按单条消息选会出现「留了答案、丢了问题」这种半截状态。
    没有配对 assistant 的 user（提问中途失败）也算一轮，答案为空串。
    """
    turns: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for message in messages:
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        if role == "user":
            if current is not None:
                turns.append(current)
            current = {"question": content, "answer": ""}
        elif role == "assistant" and current is not None:
            current["answer"] = content
    if current is not None:
        turns.append(current)
    return turns


def _parse_turn_selection(text: str, total: int) -> list[int] | None:
    """解析选择器输出，返回保序去重的 0 基下标。

    返回 `None` 表示**没看懂**，调用方必须 fail-open（保留全部）；
    返回 `[]` 表示模型明确说了「无」——那是合法结论，和故障是两回事。
    把这两者混起来，就会在解析失败时静默丢掉整段历史。
    """
    stripped = str(text or "").strip()
    if not stripped:
        return None
    if stripped.strip("。.、，,；; \n\t") == "无":
        return []
    picked: list[int] = []
    for raw in re.findall(r"\d+", stripped):
        index = int(raw) - 1  # 提示词里是 1 基编号
        if 0 <= index < total and index not in picked:
            picked.append(index)
    if not picked:
        return None
    return sorted(picked)


async def _select_relevant_turns(
    client: OpenAICompatibleClient,
    question: str,
    messages: list[dict[str, Any]],
    *,
    mode: Literal["online", "offline"],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """挑出与当前问题相关的历史轮次，返回 (选留后的消息, 统计信息)。

    **任何异常都退回「保留全部」**：选留判错只是慢一点，丢错上下文是答错，
    代价不对等，所以故障方向必须朝「全留」倒。
    """
    stats: dict[str, Any] = {
        "history_turns_total": 0,
        "history_turns_kept": 0,
        "history_dropped_turns": [],
        "history_select_ms": 0,
        "history_select_error": "",
        "history_selected": False,
        # 选留前后的历史 token 数。没选留时两者相等，一眼能看出是没跑还是白跑。
        "history_tokens_before": 0,
        "history_tokens_after": 0,
    }
    before_tokens = estimate_messages(messages)
    stats["history_tokens_before"] = before_tokens
    stats["history_tokens_after"] = before_tokens
    # 摘要那条 system 消息是压缩产物，丢了等于丢整段历史——永远保留，不参与选留。
    fixed = [m for m in messages if str(m.get("role") or "") == "system"]
    turns = _group_turns(
        [m for m in messages if str(m.get("role") or "") != "system"]
    )
    stats["history_turns_total"] = len(turns)
    if not settings.history_select_enabled:
        return messages, stats
    if len(turns) < max(1, settings.history_select_min_turns):
        return messages, stats

    # 只把最近 N 轮交给选择器，更早的一律丢弃——这是**有意的近因策略**，
    # 也让选择器自身的输入有上界。越久远的轮次越不容易和当前问题相关。
    window = turns[-max(1, settings.history_select_max_turns):]
    offset = len(turns) - len(window)
    listing = "\n".join(
        f"[{offset + i + 1}] {turn['question']}"
        + (
            "\n    → " + turn["answer"][:_HISTORY_SELECT_ANSWER_CHARS].replace("\n", " ")
            if turn["answer"]
            else ""
        )
        for i, turn in enumerate(window)
    )
    started = time.monotonic()
    try:
        thinking_enabled = _thinking_enabled(mode)
        raw = (
            await client.complete(
                [
                    {"role": "system", "content": _HISTORY_SELECT_PROMPT},
                    {
                        "role": "user",
                        "content": f"当前问题：{question}\n\n历史轮次：\n{listing}",
                    },
                ],
                max_tokens=64,
                temperature=0.0,
                chat_template_kwargs=(
                    {"enable_thinking": False}
                    if mode == "offline" and not thinking_enabled
                    else None
                ),
                thinking_enabled=thinking_enabled if mode == "online" else None,
            )
        ).content
    except Exception as exc:
        stats["history_select_error"] = f"{type(exc).__name__}: {exc}"
        return messages, stats
    finally:
        stats["history_select_ms"] = int((time.monotonic() - started) * 1000)

    picked = _parse_turn_selection(raw, len(turns))
    if picked is None:
        stats["history_select_error"] = f"无法解析选择结果：{str(raw)[:80]}"
        return messages, stats

    kept = set(picked)
    selected: list[dict[str, Any]] = []
    dropped: list[int] = []
    for index, turn in enumerate(turns):
        if index not in kept:
            dropped.append(index + 1)
            continue
        selected.append({"role": "user", "content": turn["question"]})
        if turn["answer"]:
            selected.append({"role": "assistant", "content": turn["answer"]})
    stats.update(
        history_turns_kept=len(picked),
        history_dropped_turns=dropped,
        history_selected=True,
        history_tokens_after=estimate_messages(fixed + selected),
    )
    return fixed + selected, stats


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


async def _remote_rag_available() -> bool:
    """远端 GPU 节点是否至少有一个能用。**任何异常都当"不可用"**——
    这个判断只用来决定压不压路数，判断不出来时保守按串行路径处理，
    宁可少取几路证据，也不要在用户等答案时把 8 路串行跑成 61 秒。
    """
    _, remote_urls = _rag_health_urls()
    if not remote_urls:
        return False
    try:
        async with httpx.AsyncClient(timeout=settings.rag_health_timeout) as client:
            checks = await asyncio.gather(
                *(_check_rag_url(client, url) for url in remote_urls),
                return_exceptions=True,
            )
    except Exception:  # noqa: BLE001
        return False
    return any(
        isinstance(c, dict) and c.get("status") != "unavailable" for c in checks
    )


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


# ── 图片转写：把图片变成可检索的文本 ──
#
# 为什么需要这一步：检索链（规划器、分组、RAG）只吃文本，图片一个字节都到不了。
# 用户传一张试卷照片问「做一下这张卷子」，检索拿到的 query 就是**这句指令本身**，
# 拿去查知识库什么也查不到。于是模型手里一份证据都没有，只能凭常识答——答出
# 满篇「（或10，视电厂具体要求）」这类猜测。
#
# 所以：作答前先把图片转成文字，**用题面去检索**。图片本身仍照常发给答题模型
# （它能看懂图），转写的唯一职责是解锁检索。
_VISION_TRANSCRIBE_MAX_TOKENS = int(os.getenv("HDW_VISION_TRANSCRIBE_MAX_TOKENS", "4096"))
_VISION_TRANSCRIBE_TIMEOUT = float(os.getenv("HDW_VISION_TRANSCRIBE_TIMEOUT", "180"))

# 规划器出参只有 160 tokens、目标是 200-400ms。整张卷子的题面（几千字）喂进去
# 只会拖慢它，判断质量并不会更好——它要判的是「要不要查文档」，不是读懂每道题。
_PLANNER_INPUT_MAX_CHARS = int(os.getenv("HDW_PLANNER_INPUT_MAX_CHARS", "2000"))

_TRANSCRIBE_PROMPT = (
    "你在做文字识别。逐字转写图片里的全部文字，只输出转写结果。\n"
    "· 保留题号、题型标题（如「一、填空题」）、空格、下划线、括号、选项和单位\n"
    "· **不要作答、不要解释、不要补全**——只转写你看到的字\n"
    "· 看不清的字用「?」代替，不要猜\n"
    "· 如果图片不是文字材料（仪表盘、设备铭牌、现场照片），"
    "改用一段话客观描述图中的关键信息\n"
)

# 明确在推脱的表述：它们指向「模型自己识别不了」，而不是图片里的内容。
# **不能收裸的「看不清」**——题目本身可能就在说这件事（例如安全规程里的
# 「当发现仪表看不清时，应……」），收进来会把一份正常转写整段判死，
# 于是检索退回现状，而原因看不出来。自检里有一条专门钉这个反例。
_TRANSCRIPT_REFUSALS = (
    "无法识别", "无法读取", "无法辨认", "无法看清", "图片不清晰", "图片模糊",
    "抱歉", "对不起", "我不能", "I cannot", "I'm sorry", "I am sorry",
)

# 转写来源的种类。`chat` 走对话模型（在线 DeepSeek / 本地带 mmproj 的 llama），
# `mineru` 走本地 MinerU 文档解析。二者产物形态不同但下游一视同仁。
_VISION_KINDS = ("chat", "mineru")


def _ordered_vision_candidates(
    config: dict[str, Any], kinds: set[str]
) -> list[dict[str, Any]]:
    """按 `vision_priority` 排出可用于**图片转写**的候选。纯函数，不发请求。

    与 `_vision_candidates` 的分工：那个决定「**答题**用哪个视觉模型」，
    只接受 `kind=chat`；这个决定「**转写**用哪个来源」，还接受 `kind=mineru`。
    两者读同一份配置，只是视图不同——这样界面上一个优先级列表就能同时管两件事。
    """
    output: list[dict[str, Any]] = []
    for item in config.get("vision_priority") or []:
        if not isinstance(item, dict) or not item.get("enabled", True):
            continue
        kind = str(item.get("kind") or "chat").strip().lower()
        if kind not in _VISION_KINDS or kind not in kinds:
            continue
        mode = str(item.get("mode") or "").strip().lower()
        model = str(item.get("model") or "").strip()
        if kind == "chat":
            # chat 必须有合法的 mode 与模型名，否则这一项无法调用。
            if mode not in ("online", "offline") or not model:
                continue
        else:
            # mineru 不区分在线/本地，模型名只是界面上的标签。
            mode = ""
            model = model or "mineru"
        output.append(
            {
                "kind": kind,
                "mode": mode,
                "model": model,
                "priority": int(item.get("priority") or 999),
            }
        )
    output.sort(key=lambda candidate: candidate["priority"])
    return output


def _usable_transcript(text: str) -> bool:
    """转写结果能不能拿去检索。

    **坏的转写比没有转写更糟**：它会把检索引到完全无关的文档上，而下游看不出
    区别——模型会拿着错误证据一本正经地作答，比「没证据」更难发现。
    所以宁可判不可用，换下一个候选。
    """
    stripped = str(text or "").strip()
    if len(stripped) < 8:
        return False
    # 只看**开头**：模型真拒绝时会以道歉开头；而正文中间出现「抱歉」是可能的
    # （例如题目在考服务用语）。放宽到全段会把正常转写误杀。
    head = stripped[:20]
    return not any(cue in head for cue in _TRANSCRIPT_REFUSALS)


def _compose_retrieval_question(question: str, vision: dict[str, Any] | None) -> str:
    """把「用户指令」和「图片转写」拼成用来检索的文本。

    **保留原指令**：它带着用户意图（「只给答案」「按题号给」），规划器要靠它
    判断该取哪几类数据。转写文本排在后面，是检索真正拿来匹配知识库的素材。
    """
    text = str((vision or {}).get("text") or "").strip()
    if not text:
        return question
    return f"{question}\n\n[图片内容]\n{text}"


def _truncate_for_planner(question: str) -> str:
    """把喂给规划器的文本截到有界长度。见 `_PLANNER_INPUT_MAX_CHARS` 的说明。"""
    text = str(question or "")
    if len(text) <= _PLANNER_INPUT_MAX_CHARS:
        return text
    return text[:_PLANNER_INPUT_MAX_CHARS] + "…（已截断）"


async def _transcribe_with_chat(
    candidate: dict[str, Any],
    images: list[dict[str, str]],
    *,
    allow_online: bool,
) -> tuple[str, bool]:
    """走对话模型转写图片，返回 `(文本, 是否被截断)`。

    在线与本地走的是**同一条**协议路径，只是 profile 不同——这正是把 mmproj
    归到 `offline` 而不是新开一个 mode 的理由。
    """
    mode = candidate["mode"]
    if mode == "online" and not allow_online:
        raise RuntimeError("当前用户未开放在线推理")
    if mode == "online" and not settings.online_llm_api_key:
        raise RuntimeError("在线 API key 未配置")
    if mode == "online" and not bool(_read_model_config().get("online", {}).get("multimodal_enabled", True)):
        raise RuntimeError("在线多模态已停用")
    if mode == "offline" and not bool(_read_model_config().get("local", {}).get("multimodal_enabled", False)):
        raise RuntimeError("本地多模态已停用")

    profile = _effective_profile(mode)
    if mode == "offline":
        # 本地候选的实际上下文以 llama 报的为准（与 _vision_candidates 一致）。
        runtime = await _local_runtime_info(profile["base_url"])
        if not runtime["available"]:
            raise RuntimeError(str(runtime["detail"]))
        if not bool(runtime["vision"]):
            raise RuntimeError(str(runtime["detail"]))

    _, client = _llm_client(
        mode,
        model_override=candidate["model"],
        base_url_override=profile["base_url"],
        timeout_override=_VISION_TRANSCRIBE_TIMEOUT,
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": _TRANSCRIBE_PROMPT}]
    content.extend(
        {"type": "image_url", "image_url": {"url": image["data_url"]}}
        for image in images
    )
    result = await client.complete(
        [{"role": "user", "content": content}],
        max_tokens=_VISION_TRANSCRIBE_MAX_TOKENS,
        temperature=0.0,
        chat_template_kwargs={"enable_thinking": False} if mode == "offline" else None,
    )
    # 被 max_tokens 截断：转写不完整。**不能当成失败**——前半张卷子仍有用；
    # 但也必须让调用方知道，否则「后半张卷子凭空消失」无人察觉。
    return result.content, result.finish_reason == "length"


# 上传给 MinerU 的扩展名。结果字典的键是**上传文件名的 stem**，所以文件名必须
# 可预测，否则按 stem 取结果会取空（而「取空」看起来和「解析失败」一模一样）。
_MINERU_SUFFIX = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}

# 与 ETL 侧同名同默认值（parse_documents.py:19-21），改一处即两边都改。
_MINERU_BACKEND = os.getenv("HDW_MINERU_BACKEND", "pipeline")
_MINERU_LANG = os.getenv("HDW_MINERU_LANG", "ch")


def _decode_image_bytes(image: dict[str, str]) -> bytes:
    """从 data URL 取出原始图片字节。`_validate_image_attachments` 只保留
    data_url，而 MinerU 要的是原始字节。"""
    _, _, encoded = str(image.get("data_url") or "").partition(",")
    if not encoded:
        raise ValueError("图片缺少 base64 内容")
    return base64.b64decode(encoded, validate=True)


def _mineru_upload_name(index: int, image: dict[str, str]) -> str:
    """给 MinerU 的上传文件名。

    结果字典按**上传文件名的 stem** 索引，对不上就取空——而「取空」在日志里
    看起来和「解析失败」一模一样，排查时会被引到完全错误的方向。
    所以不沿用用户的文件名：中文在 multipart 里的编码处理各实现不一致，
    而这里需要的是**可预测**。序号名天然满足，也天然不重复。
    """
    mime = str(image.get("mime_type") or "").lower()
    return f"image{index}{_MINERU_SUFFIX.get(mime, '.png')}"


async def _mineru_transcribe(images: list[dict[str, str]]) -> str:
    """用本地 MinerU 解析图片，返回 markdown 文本。

    MinerU 是**文档解析器**而不是纯 OCR：它做版式分析并输出 markdown，
    题型标题会带 `## `。这对下游没有影响：转写文本只用来给检索规划器看，
    它读得懂 markdown。

    实测单页卷子图约 17 s，比在线视觉慢一个量级，所以它更适合当兜底来源。
    """
    base = settings.mineru_base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=httpx.Timeout(settings.mineru_timeout, connect=5)) as client:
        # 先探队列：MinerU 是单并发，有入库任务在跑时排队可能等几分钟。
        # 与其让用户干等，不如快速失败、换下一个候选。
        health = await client.get(f"{base}/health")
        health.raise_for_status()
        state = health.json()
        busy = int(state.get("queued_tasks") or 0) + int(state.get("processing_tasks") or 0)
        if busy > settings.mineru_max_queue:
            raise RuntimeError(f"MinerU 忙（队列 {busy}），本次跳过")

        files = [
            (
                "files",
                (
                    _mineru_upload_name(index, image),
                    _decode_image_bytes(image),
                    str(image.get("mime_type") or "image/png"),
                ),
            )
            for index, image in enumerate(images)
        ]
        response = await client.post(
            f"{base}/file_parse",
            files=files,
            data={
                "backend": _MINERU_BACKEND,
                "lang_list": _MINERU_LANG,
                "parse_method": "auto",
                "return_md": "true",
                "return_middle_json": "false",
                "return_model_output": "false",
                "return_content_list": "false",
                "return_images": "false",
                "response_format_zip": "false",
                "return_original_file": "false",
            },
        )
        response.raise_for_status()
        payload = response.json()

    parts: list[str] = []
    for index, image in enumerate(images):
        stem = Path(_mineru_upload_name(index, image)).stem
        markdown = str(((payload.get("results") or {}).get(stem) or {}).get("md_content") or "").strip()
        if markdown:
            parts.append(markdown)
    if not parts:
        raise RuntimeError("MinerU 未返回任何 markdown")
    return "\n\n".join(parts)


async def _transcribe_images(
    images: list[dict[str, str]],
    *,
    allow_online: bool,
) -> dict[str, Any] | None:
    """把图片转成可检索的文本。**失败一律返回 None，绝不抛异常。**

    与 `_vision_candidates` 刻意不同：那里选不出候选会让 `_llm_answer` 抛 503，
    因为答题**必须要**视觉；而转写失败只是让检索退回现状（模型仍能看图，
    只是没有知识库证据），不该把「能答」变成「答不了」。
    """
    if not images:
        return None
    candidates = _ordered_vision_candidates(_read_model_config(), {"chat", "mineru"})
    attempts: list[dict[str, str]] = []
    started = time.monotonic()
    for candidate in candidates:
        label = f"{candidate['kind']}:{candidate['mode'] or '-'}:{candidate['model']}"
        try:
            if candidate["kind"] == "mineru":
                # MinerU 是同步解析，没有 finish_reason 一说——它的产物要么完整
                # 要么报错，不存在「说了一半被截断」。
                text, truncated = await _mineru_transcribe(images), False
            else:
                text, truncated = await _transcribe_with_chat(
                    candidate, images, allow_online=allow_online
                )
        except Exception as exc:
            attempts.append({"candidate": label, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if not _usable_transcript(text):
            attempts.append({"candidate": label, "error": "转写结果不可用（过短或为推脱语）"})
            continue
        return {
            "kind": candidate["kind"],
            "mode": candidate["mode"] or None,
            "model": candidate["model"],
            "source": f"{candidate['mode']}:{candidate['model']}" if candidate["mode"] else candidate["kind"],
            "text": str(text).strip(),
            "truncated": truncated,
            "chars": len(str(text).strip()),
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "attempts": attempts,
        }
    return None


def _has_chat_vision_candidate() -> bool:
    """配置里有没有启用中的 chat 类视觉候选。**不探测可用性**，只看配置。

    只用来决定「要不要把原图发给答题模型」。真正的可用性判断仍在
    `_vision_candidates`——那里失败会抛 503，这里只负责别把一个注定失败的
    请求发出去（例如用户只配了 MinerU 时）。
    """
    return bool(_ordered_vision_candidates(_read_model_config(), {"chat"}))


def _vision_diagnostics(vision: dict[str, Any] | None) -> dict[str, Any]:
    """给响应体用的转写元信息。**不带转写正文**——正文已经在提示词里，
    再回一份到 JSON 里只会让响应体大一倍。"""
    if not vision:
        return {"source": None, "used": False}
    return {
        "source": vision.get("source"),
        "kind": vision.get("kind"),
        "mode": vision.get("mode"),
        "model": vision.get("model"),
        "chars": vision.get("chars"),
        "truncated": vision.get("truncated"),
        "elapsed_ms": vision.get("elapsed_ms"),
        "attempts": vision.get("attempts") or [],
        "used": True,
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


def _evidence_block(contexts: list[dict[str, Any]]) -> str:
    """模型实际看到的证据文本（含 `[证据 N] 来源=… 片段=…` 头部）。

    **数字核对必须拿它当比对基准**：答案会引用「片段 616」这类头部里的编号，
    只比 `text` 字段会把这些引用全误报成「证据里没有的数」。
    与 `_prompt` 共用同一份，就不会各写各的而漂移。
    """
    if not contexts:
        return "无"
    return "\n\n".join(
        (
            f"[证据 {idx}] "
            f"来源={Path(str((item.get('metadata') or {}).get('source_file', ''))).name or '未知'} "
            f"片段={(item.get('metadata') or {}).get('chunk_index', '未知')}\n"
            f"{str(item.get('text', ''))}"
        )
        for idx, item in enumerate(contexts, 1)
    )


def _graph_block(graph_rows: list[dict[str, Any]]) -> str:
    """模型实际看到的图谱文本（含 `[图谱 N] … chunk=…` 头部）。理由同上。"""
    if not graph_rows:
        return "无"
    return "\n".join(
        (
            f"[图谱 {idx}] 文档={row.get('document_title') or row.get('document_id')}; "
            f"章节={'/'.join(row.get('section_path') or []) or row.get('section_title') or '无'}; "
            f"chunk={row.get('chunk_id')}; "
            f"实体={_entity_summary(row)}; "
            f"前文={((row.get('previous_texts') or [''])[0]) or '无'}; "
            f"后文={((row.get('next_texts') or [''])[0]) or '无'}"
        )
        for idx, row in enumerate(graph_rows, 1)
    )


def _prompt(
    question: str,
    contexts: list[dict[str, Any]],
    graph_rows: list[dict[str, Any]] | None = None,
    history_messages: list[dict[str, str]] | None = None,
    model_name: str | None = None,
    image_attachments: list[dict[str, str]] | None = None,
    image_text: str | None = None,
    skip_retrieval: bool = False,
    live_points: list[dict[str, Any]] | None = None,
    series: list[dict[str, Any]] | None = None,
    logs: dict[str, Any] | None = None,
    live_errors: list[str] | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    evidence = _evidence_block(contexts)
    graph = _graph_block(graph_rows)

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
        days = (logs.get("work_digest") or {}).get("dates") or []
        categories = "、".join(
            f"{name} {count} 条" for name, count in (summary.get("categories") or {}).items()
        ) or "无"
        head = (
            f"\n\n运行日志（{logs.get('start')} ~ {logs.get('end')}"
            f"{'，仅重大事件' if logs.get('major_only') else ''}"
            f"{'，' + str(logs.get('shift')) if logs.get('shift') else ''}）：\n"
            f"  合计 {summary.get('total_events', len(events))} 条，"
            f"其中重大 {summary.get('major_events', 0)} 条；分类：{categories}\n"
        )
        # 班次筛选的口径与残留。**必须明示**：判不出班次而保留的事件混在
        # 「白班」的答案里，不说明就看不出来。
        if logs.get("shift"):
            window = ""
            try:
                begin, finish = _shift_window(
                    str(logs["shift"]), date.fromisoformat(str(logs.get("start")))
                )
                window = (
                    f"（{begin.strftime('%m-%d %H:%M')}~{finish.strftime('%m-%d %H:%M')}）"
                )
            except ValueError:
                # 日期缺失或格式不对时只省掉时段，**不能让拼提示词抛异常**——
                # 那会把一次能答的问答变成 500。
                pass
            head += f"  班次筛选：已按{logs['shift']}{window}筛出 {len(events)} 条"
            unknown = int(logs.get("shift_unknown") or 0)
            if unknown:
                head += f"，其中 {unknown} 条无明确班次信息、一并给出"
            head += "。\n"
        # 完整性：哪些日志类型缺哪些班次。**不说清 = 把不完整的讲成完整的**。
        completeness = _logs_completeness_line(
            logs.get("log_completeness") or {}, now or _local_now()
        )
        if completeness:
            head += f"  日志完整性：{completeness}\n"
        # brief 与逐日分栏是两套说法，同时给会互相打架；有分栏时以分栏为准。
        brief = str(summary.get("brief") or "").strip()
        if brief and brief != "未查询到匹配日志。" and not days:
            head += f"  摘要：{brief}\n"

        # 逐日分栏。**这段承载「全量」**：下面明细受 _LOGS_MAX 截断（实测抓 161 条
        # 只注入 40 条），看不到的那部分只能靠这里的条数体现——按天给，
        # 分「主要工作／两票／重要事件」三栏，与交接班口径一致。
        #
        # 全量注入时明细已经给全了，再逐条列分栏就是同一批内容出现两次，
        # 白白占上下文。此时只留每天的计数行。
        full = bool(logs.get("full"))
        items_per_bucket = 0 if full else _LOGS_DIGEST_ITEMS
        day_lines: list[str] = []
        for day in days:
            if len(day_lines) >= _LOGS_DIGEST_LINES:
                break
            counts = day.get("counts") or {}
            day_lines.append(
                f"\n  {day.get('date')}（主要工作 {counts.get('main_work', 0)} 条、"
                f"两票 {counts.get('ticket_work', 0)} 条、"
                f"重要事件 {counts.get('important_timed_events', 0)} 条）"
            )
            for bucket, label in (
                ("important_timed_events", "重要事件"),
                ("ticket_work", "两票"),
                ("main_work", "主要工作"),
            ):
                for item in (day.get(bucket) or [])[:items_per_bucket]:
                    if len(day_lines) >= _LOGS_DIGEST_LINES:
                        break
                    text = " ".join(
                        str(item.get("content") or item.get("title") or "").split()
                    )[:140]
                    if not text:
                        continue
                    day_lines.append(
                        f"    〔{label}〕{item.get('time') or ''} "
                        f"{item.get('source') or ''}：{text}"
                    )

        detail_lines = []
        for idx, event in enumerate(events, 1):
            severity = str(event.get("severity") or "")
            mark = "【重大】" if severity == "major" else ("【重要】" if severity == "important" else "")
            detail_lines.append(
                f"[日志 {idx}] {mark}{event.get('time') or event.get('date') or ''} "
                f"{event.get('source') or ''}／{event.get('category') or ''}："
                f"{str(event.get('content') or '').strip()}"
            )
        # 明细口径按模式写准：全量时再说「受上限截断」会让模型以为还有没看到的，
        # 于是把「列出的这些」讲成「一部分」。
        detail_note = (
            "  全部明细（按严重程度排序）：\n"
            if full
            else "  明细（按严重程度排序，受上限截断；完整条数以「合计」为准）：\n"
        )
        logs_block = head + "\n".join(day_lines) + "\n\n" + detail_note + "\n".join(detail_lines)

    # 未检索时不拼证据段落：一旦出现「检索证据：无」这类框架，模型会顺着去说
    # 「证据中没有」，而不是直接依据自身设定回答。
    # 实时数据块合并：测点当前值、历史/趋势序列、运行日志三者并列。
    # 它们都是「当前系统的事实」而非文档证据，拼在同一段里便于模型对照。
    realtime_block = live_block + series_block + logs_block
    has_realtime = bool(live_points or series or logs)

    # 图片转写块。它和原图**同时**给模型：图是原始事实，转写是为了让模型能引用
    # 与检索证据对应的字面内容。**必须标明是自动识别**——OCR 会出错字漏字，
    # 而模型倾向于把白纸黑字当权威。不点破，它会拿 OCR 去跟检索证据打对台，
    # 问题就从「没检索」变成「检索了但被 OCR 带偏」。
    image_text_clean = str(image_text or "").strip()
    image_block = (
        f"\n\n图片文字（自动识别，可能有错字或漏字）：\n{image_text_clean}"
        if image_text_clean
        else ""
    )
    image_rule = (
        "图片文字是自动识别结果，只用来理解用户问的是什么，"
        "**不要把它当作证据引用**；它与检索证据冲突时以检索证据为准。"
        if image_text_clean
        else ""
    )

    # 实时数据取数失败。**必须让模型看见**，否则它会按「我没有这个能力」作答——
    # 实测：问「最近五天有什么事情」，规划正确路由到了日志查询，但日志源头不通，
    # 模型收到的是一片空白，于是回答「我没有实时感知能力，当前对话中未提供实时
    # 信息检索功能」——这句话对它当时所见而言完全准确，可用户读到的是「系统不支持」。
    live_error_list = [str(item).strip() for item in (live_errors or []) if str(item).strip()]
    live_error_block = (
        "\n\n实时数据取数失败（**是这次没取到，不是系统没有这项数据**）：\n"
        + "\n".join(f"- {item}" for item in live_error_list)
        if live_error_list
        else ""
    )
    live_error_rule = (
        # 直接针对观测到的失效写法：「取数失败」和「没有能力」在用户眼里差别很大，
        # 前者是稍后重试可能就好的问题，后者会让人以为整个功能不存在。
        "上面「实时数据取数失败」里的每一项，说明的是**这次没取到**，"
        "不是「系统没有实时数据能力」。回答时要点明是哪一项取数失败、失败原因是什么，"
        "并建议稍后重试；**绝不能把它说成「我无法获取实时数据」「本系统未提供实时信息检索」"
        "或「我没有实时感知能力」**——那会让用户以为功能不存在。"
        if live_error_list
        else ""
    )

    # 本次提问的「现在」。上游没传就自己取服务端时钟——自检与旧调用方都走这条。
    moment = now or _local_now()

    # 日志归纳规范。只在**本轮确实有日志**时注入：没日志还挂着这段，
    # 模型会去找根本不存在的日志内容（同样的道理见 image_rule）。
    log_rule = ""
    if logs:
        log_rule = _LOG_ANSWER_RULES
        # 完整性专项只在真拿到完整性数据时才加，避免无谓的规则堆积。
        if logs.get("log_completeness"):
            log_rule += _LOG_COMPLETENESS_RULE
        # 班次上下文跟着日志走：它解释的是「当天的日志为什么某几段没记录」。
        log_rule += _shift_context(moment)

    if skip_retrieval:
        user = f"问题：{question}{image_block}{realtime_block}{live_error_block}"
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
            + image_rule
        )
    else:
        user = (
            f"问题：{question}{image_block}{realtime_block}{live_error_block}"
            f"\n\n检索证据：\n{evidence}\n\n知识图谱上下文：\n{graph}"
        )
        system_body = (
            "回答时先看检索证据和知识图谱上下文；有证据就基于证据和文档结构归纳，不编造来源。"
            "无证据时，可以回答模型身份、系统能力、通用操作类问题；"
            "涉及工业规程、安全边界、设备参数时必须说明缺少证据，不能臆造。"
            # ↓ 这两条是针对**实际观测到的失效**写的，不是泛泛的「不要编造」。
            # 实测：问「凝汽器水位正常值」，证据里只有「冷态启动建议水位 800」，
            # 模型把它说成了「正常值约 800mm」——数值是真的，含义是错的。
            # 这类错误比「说没有」危险得多，因为它看起来像答案。
            "**某个具体定值在证据里没有时**，直接写「证据中未给出该定值」，"
            "不要用其它章节里数值相近的参数代替，也不要按经验估一个——"
            "替代值看起来和真值一模一样，读的人分辨不出来。"
            # 实测：参数表被解析器压平成一行一串字段（见下），模型据此把
            # `P.GAS.11 | * | PRESSURE | 燃气压力过高，报警 | >30-0.2 | barg`
            # 读成「数值栏为空」，反而答「证据中未给出该定值」——数据明明在。
            # 光有「不许挪用」的约束不够，还要告诉它**这张表该怎么对齐**。
            "**参数表在证据里是被压平的**：形如 "
            "`标识符 | * | 类型 | 描述 | 值 | 单位` 的一串字段，用 | 或换行分隔，"
            "`*` 那段有时没有、各段宽度也不固定。"
            "按**字段顺序**对齐：值就是紧跟在**描述**之后、**单位之前**的那一个。"
            "例如 `P.GAS.11 | * | PRESSURE | 燃气压力过高，报警 | >30-0.2 | barg` "
            "读作 P.GAS.11 的值为 >30-0.2 barg。"
            "若某行描述之后直接就是单位（如 `P.GAS.17 | * | PRESSURE | ESV阀上游燃气压力过低，跳机 | barg`），"
            "说明**那一行本身没有给值**，此时仍不许把相邻行的值挪过来。"
            "**证据之间数值不一致时**，把分歧原样列出来（各自的值与出处），"
            "不要自行挑一个当成唯一答案。"
            # 实测：问「滚动轴承温度最高不允许超过__℃，滑动轴承…__℃」，
            # 证据里写的是「滑动轴承不高于65℃，滚动轴承不高于80℃」，
            # 模型答成「95；80」——值都见过，但配错了空。
            # 同一段话里常并列着几个不同部位的定值，错位后照样是通顺的数字。
            "**一道题里有多个空时，逐个空与证据里对应的那一项核对**，"
            "按题干的先后顺序作答，不要把某个空的值挪到另一个空上；"
            "拿不准某个空对应哪一项就写「证据中未明确对应关系」。"
            "这是工业场景，不要为了简短省略证据中的职责、步骤、条件、例外、参数或安全后果。"
            "必须逐条阅读并综合所有列出的证据和图谱上下文，不能只依据第一条；"
            "按主题合并重复内容，区分正常运行、启停检查和异常处理，"
            "根据资料量输出完整、可执行、层次清楚的回答。"
            + image_rule
        )

    prompt = [
        {
            "role": "system",
            "content": (
                f"你是 {model_name or settings.llm_model}，HyperDriveWave 工业知识问答系统以你作为底座大模型。"
                f"{system_body}"
                # 取数失败的说明紧跟在主体之后。**放在 system_body 里面就得写两遍**
                # （skip_retrieval 分支和正常分支各一份），迟早漏一处。
                f"{live_error_rule}"
                # 日志归纳规范同理放在主体之后：skip_retrieval 分支和正常分支都要有。
                f"{log_rule}"
                # 「现在几点」与日志无关，**两个分支都要有**。此前它藏在
                # _shift_context 里、跟着 `if logs:` 走，于是没日志的问答
                # 既不知道今天几号也不知道现在几点，「昨天」只能靠猜。
                f"{_now_context(moment)}"
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
    image_text: str | None = None,
    allow_online: bool = True,
    skip_retrieval: bool = False,
    live_points: list[dict[str, Any]] | None = None,
    series: list[dict[str, Any]] | None = None,
    logs: dict[str, Any] | None = None,
    live_errors: list[str] | None = None,
    progress: ProgressFn = _noop_progress,
    now: datetime | None = None,
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
                image_text=image_text,
                skip_retrieval=skip_retrieval,
                live_points=live_points,
                series=series,
                logs=logs,
                live_errors=live_errors,
                now=now,
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
            # 相关性选留：把与当前问题无关的历史轮次挡在 prompt 之外。
            # 放在 prepare() **之后**——context_kept_from 是存盘的绝对下标，
            # 在它之前过滤会让存回去的下标指错位置。
            progress("history", f"筛选相关历史轮次（共 {len(preparation.messages)} 条消息）")
            selected_messages, select_stats = await _select_relevant_turns(
                client,
                question,
                preparation.messages,
                mode=selected_mode,
            )
            # 把选留结果报出来。这一步对用户是可感知的（同一句问题，历史长短不同
            # 速度差很多），说出来他才知道「刚才那几秒花在哪」。
            if select_stats.get("history_selected"):
                progress(
                    "history",
                    f"历史轮次：保留 {select_stats.get('history_turns_kept')} / "
                    f"{select_stats.get('history_turns_total')} 轮，"
                    f"{select_stats.get('history_tokens_before'):,} → "
                    f"{select_stats.get('history_tokens_after'):,} token",
                )
            elif select_stats.get("history_select_error"):
                progress("history", f"历史选留未生效：{select_stats['history_select_error']}")
            prompt = [base_prompt[0], *selected_messages, *base_prompt[1:]]
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
                    # 历史相关性选留的结果：选了几轮、丢了哪几轮、花了多久、
                    # 有没有失败。出问题时要能一眼分清「选错了」和「选择器挂了」。
                    "history_select": select_stats,
                    # 实际送出去的完整 prompt 大小（系统提示 + 历史 + 本轮证据）。
                    "prompt_tokens_estimate": estimate_messages(prompt),
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
    image_text: str | None = None,
    allow_online: bool = True,
    skip_retrieval: bool = False,
    live_points: list[dict[str, Any]] | None = None,
    series: list[dict[str, Any]] | None = None,
    logs: dict[str, Any] | None = None,
    live_errors: list[str] | None = None,
    evidence_budget: int | None = None,
    progress: ProgressFn = _noop_progress,
    now: datetime | None = None,
) -> dict[str, Any]:
    selected_mode = _resolve_inference_mode(inference_mode)
    llm_contexts = _llm_contexts(selected_mode, contexts, evidence_budget)
    progress("graph", f"查询知识图谱（基于 {len(llm_contexts)} 条证据）")
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
        image_text=image_text,
        allow_online=allow_online,
        skip_retrieval=skip_retrieval,
        live_points=live_points,
        series=series,
        logs=logs,
        live_errors=live_errors,
        now=now,
        progress=progress,
    )
    # 数字核对：把答案里的数与送进提示词的那批证据对一遍。
    # **只作提示**——无出处的数不一定错（单位换算、常识数字都可能），
    # 但它是这份答案里最值得人工核对的地方，比让人通篇自查有用得多。
    unsupported = (
        _unsupported_numbers(
            answer,
            # **必须与提示词里的同一份**（含证据头/图谱头），否则答案里的
            # 「片段 616」这类引用编号会被误报。问题本身也算出处——答案常复述
            # 问题里的值（「低 30kPa」）。
            _evidence_block(llm_contexts)
            + "\n"
            + _graph_block(graph_rows)
            + "\n"
            + str(question or ""),
        )
        if answer and not skip_retrieval
        else []
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
    return {
        "answer": answer,
        "question": question,
        # 始终是**真正进了提示词**的那批证据。原来这里分成两个 return 写
        # （有证据给 llm_contexts、没证据给 contexts），空列表时两者等价，
        # 分成两处只是给了「只改一处」的机会——事实上刚才就踩到了：
        # unsupported_numbers 只加在其中一个 return 上，而常见路径走的是另一个。
        "citations": llm_contexts,
        "graph_context": graph_rows,
        "status": "answered_by_llm_with_retrieval" if contexts else "answered_by_llm",
        "llm": llm_info,
        "context": context_meta,
        # 答案里在证据中找不到出处的数字。**空列表是正确的常态**——
        # 只有非空时才值得用户留意。
        "unsupported_numbers": unsupported,
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
            # 文案要点名 mmproj：它和模型/引擎一样是**启动参数**，改了不重启不生效，
            # 而用户看到「已保存」就会以为已经生效了。
            "local_engine_and_model_apply": (
                "模型、引擎、上下文、MTP 与视觉投影器（mmproj）："
                "保存后会重启本地推理服务并立即生效"
            ),
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
    for section in ("local", "online", "retrieval", "permissions"):
        if isinstance(incoming.get(section), dict):
            current[section].update(incoming[section])
    if isinstance(incoming.get("vision_priority"), list):
        current["vision_priority"] = incoming["vision_priority"]

    local_update = incoming.get("local")
    # 这几个都是 llama-server 的**启动参数**，改了不重启就不生效：
    #   mtp_enabled  → --spec-type draft-mtp
    #   mmproj       → --mmproj（视觉投影器）
    # 漏掉 mmproj 会得到一个纯静默失效：配置写了、界面报「已保存」、
    # llama 没重启、视觉永远不生效，而且没有任何报错指向这件事。
    local_runtime_fields = {"model", "engine", "context_window", "mtp_enabled", "mmproj"}
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
            # **mmproj 有意不从运行态覆写**：llama 的 /props 不报它，无从得知
            # 当前实际加载的是哪个投影器。所以回滚用的是 previous 里的值——
            # 若那一次部署本身就带着坏的 mmproj，回滚会二次失败。低概率，
            # 但比「回滚时静默清空 mmproj、把原本可用的本地视觉关掉」要好：
            # 后者会让一次失败的切换**永久**改变系统能力，且无人察觉。

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


@app.patch("/local-model")
async def patch_local_model(
    patch: LocalModelPatch,
    x_hdw_session: str | None = Header(default=None),
) -> dict[str, Any]:
    """加载 / 卸载本地推理，占回或释放显存。

    卸载只能停进程：llama.cpp 单模型模式没有任何运行期卸载接口
    （`/models/unload` 只在多模型 router 下注册），协调器的 `/unload-llm`
    实际就是 `systemctl --user stop hyperdrivewave-llama.service`。

    加载复用协调器的 `/switch-llm`：它的语义本来就是「确保配置里的本地模型
    在跑」（内部先 stop 再 start），从已卸载状态调用时 stop 是空操作。不再
    另造一个与它完全重复的端点，避免两处逻辑日后走岔。
    """
    user = _current_user(x_hdw_session)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="admin permission required")
    action = "加载" if patch.loaded else "卸载"
    try:
        apply_result = await _maintenance_request("/switch-llm" if patch.loaded else "/unload-llm")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"本地模型{action}失败：{exc}") from exc
    response = await get_model_config(x_hdw_session)
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


# 流式返回的心跳间隔。生成阶段一次要几十秒且中间没有字节流动，
# 定期发一个事件让连接保持活跃——浏览器与中间代理都会把长时间无数据的
# 连接当死连接掐掉。实测就发生过一次：服务端返回 200，前端的 fetch
# 却在 77 秒时被拒（"Load failed"）。
_SSE_HEARTBEAT = float(os.getenv("HDW_SSE_HEARTBEAT", "5"))

async def _stream_until(
    task: "asyncio.Task[Any]", queue: "asyncio.Queue[dict[str, Any]]"
) -> AsyncIterator[dict[str, Any]]:
    """等 task 完成，期间把 queue 里的进度事件与心跳吐出来。

    **必须边等边转发**：进度事件是任务内部不同时刻产生的——MCP 四类取数并发，
    各自完成时间差很远（日志要等 LIEMS 30 秒）。等任务整个跑完再一次性给，
    用户在那 30 秒里只会看到上一句话停着不动，等于没有进度。

    最后一项 yield 的是 `{"__result__": ...}`，调用方据此取回 task 的返回值。
    用 shield 保证超时不会取消任务本身。
    """
    while True:
        try:
            result = await asyncio.wait_for(asyncio.shield(task), timeout=_SSE_HEARTBEAT)
        except asyncio.TimeoutError:
            while not queue.empty():
                yield queue.get_nowait()
            yield {"heartbeat": True}
            continue
        while not queue.empty():
            yield queue.get_nowait()
        yield {"__result__": result}
        return


def _sse_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _qa_pipeline(
    *,
    question: str,
    user: dict[str, str],
    req: "QARequest",
    image_attachments: list[dict[str, str]],
    selected_mode: Literal["online", "offline"],
    now: datetime,
    clock_source: str = "server",
) -> AsyncIterator[dict[str, Any]]:
    """跑完整条链路，途中 yield 进度事件，最后 yield `{"result": ...}`。

    进度事件存在的唯一理由：**这次请求可能要几十秒**（逐题检索 + 长回答），
    中间没有任何字节流动。
    """
    if image_attachments:
        yield {"stage": "vision", "label": "识别图片文字"}
    # 先转写再检索：检索链（规划器／规划、RAG）只吃文本，图片到不了那一层。
    # 失败返回 None 而不是抛异常——检索退回现状，但答题照常。
    vision_info = (
        await _transcribe_images(
            image_attachments, allow_online=_online_inference_allowed(user)
        )
        if image_attachments
        else None
    )
    # **question 必须保持原样**：下面用 `history[-1]["content"] == question` 摘掉
    # 重复的最后一轮用户消息。把 question 改写成带转写正文的版本，这个等值判断
    # 必然失败，用户消息会被原样送进模型两遍。所以另开一条文本走检索。
    retrieval_text = _compose_retrieval_question(question, vision_info)
    # 没有 chat 类候选时不把原图发给答题模型：那样 `_vision_candidates` 会找不到
    # 候选而抛 503。题面已经在转写文本里，纯文本作答一样能答——
    # 这顺带修掉了「只配了 MinerU 的用户传图必 503」。
    answer_images = image_attachments if _has_chat_vision_candidate() else []

    # 进度队列：链路深处（_retrieve_planned / _answer）通过 progress 回调往里塞，
    # 这里边等边转发。**必须并发地等**——MCP 取数最慢的一项要等 LIEMS 30 秒，
    # 等整个任务跑完再一次性报，用户在那 30 秒里看不到任何变化。
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def progress(stage: str, label: str) -> None:
        queue.put_nowait({"stage": stage, "label": label})

    retrieval_task = asyncio.create_task(
        _retrieve_planned(
            retrieval_text,
            selected_mode,
            req.top_k,
            has_images=bool(image_attachments),
            progress=progress,
            now=now,
        )
    )
    async for event in _stream_until(retrieval_task, queue):
        if "__result__" in event:
            contexts, rag_info = event["__result__"]
        else:
            yield event
    rag_info["vision"] = _vision_diagnostics(vision_info)
    # 把「这次按哪个时刻理解相对时间」留在诊断里。答错「今天」「白班」时，
    # 第一个要看的就是它——客户端时钟偏了会静默回落服务端，不留痕就查不出来。
    rag_info["clock"] = {"now": now.strftime("%Y-%m-%d %H:%M:%S"),
                         "shift": _shift_of(now), "source": clock_source}
    history, conversation_data = _conversation_messages(user["code"], req.conversation_id, req.messages)
    if history and history[-1]["role"] == "user" and history[-1]["content"] == question:
        history = history[:-1]

    yield {"stage": "generate", "label": "生成回答", "evidence": len(contexts)}
    answer_task = asyncio.create_task(
        _answer(
            question,
            contexts,
            req.reasoning_effort,
            inference_mode=selected_mode,
            history_messages=history,
            conversation_id=req.conversation_id,
            user_code=user["code"],
            conversation_data=conversation_data,
            image_attachments=answer_images,
            image_text=(vision_info or {}).get("text"),
            allow_online=_online_inference_allowed(user),
            skip_retrieval=bool(rag_info.get("no_retrieval")),
            live_points=rag_info.get("live_points") or [],
            series=rag_info.get("series") or [],
            logs=rag_info.get("logs"),
            # 取数失败要让模型看见，否则它会按「我没有实时数据能力」作答。
            live_errors=rag_info.get("live_errors") or [],
            # 证据预算随检索路数增长。单路时等于原来的默认值，所以这里不必
            # 判断「是不是多轮」——见 _retrieval_evidence_budget。
            evidence_budget=_retrieval_evidence_budget(
                selected_mode, len((rag_info.get("plan") or {}).get("queries") or []) or 1
            ),
            progress=progress,
            now=now,
        )
    )
    # 生成阶段本身没有中间产物，但图谱查询和历史选留都在这个任务里，
    # 它们各自的进度同样要实时转发（历史选留要跑一次模型调用）。
    async for event in _stream_until(answer_task, queue):
        if "__result__" in event:
            result = event["__result__"]
        else:
            yield event
    result["rag"] = rag_info
    yield {"result": result}


@app.post("/qa/query")
async def qa_query(
    req: QARequest,
    authorization: str | None = Header(default=None),
    x_hdw_session: str | None = Header(default=None),
) -> Any:
    _check_auth(authorization)
    user = _current_user(x_hdw_session)
    image_attachments = _validate_image_attachments(req.images)
    # **服务端强制**。只在前端置灰按钮不算鉴权——接口是公开的，
    # 任何人都能直接 POST。与在线推理的 403 同一套写法。
    if image_attachments and not _image_upload_allowed(user):
        raise HTTPException(
            status_code=403,
            detail="当前未开放图片上传，请联系管理员",
        )
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
    # 校验全部留在生成器**外面**：这样 403/422 仍是真实的 HTTP 状态码，
    # 而不会变成流里的一个错误事件（前端处理前者要简单得多）。
    # 提问时刻在这里定一次，往下只传结果——「现在几点」全链路只能有一个答案。
    now, clock_source = _clock_now(req.client_time)
    pipeline = _qa_pipeline(
        question=question,
        user=user,
        req=req,
        image_attachments=image_attachments,
        selected_mode=selected_mode,
        now=now,
        clock_source=clock_source,
    )

    if not req.stream:
        async for event in pipeline:
            if "result" in event:
                return event["result"]
        raise HTTPException(status_code=500, detail="pipeline produced no result")

    async def events():
        try:
            async for event in pipeline:
                yield _sse_event(event)
        except Exception as exc:  # noqa: BLE001
            # 流已经开始就改不了状态码了，只能把错误当作一个事件发出去。
            yield _sse_event({"error": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # 明确告诉 nginx 不要缓冲。不写这条，nginx 会把整段响应攒齐再发，
            # 那就完全失去了流式的意义（而且长请求照样会被掐）。
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/internal/qa/query")
async def internal_qa_query(
    req: InternalQARequest,
    x_hdw_internal_key: str | None = Header(default=None),
) -> dict[str, Any]:
    """服务间问答入口：给 HDW_API 容器用，不做上下文管理。

    这条路的全部意义是**每次提问都独立**：不读某个用户的历史，也不写回去。
    落实方式是三个参数——`conversation_id=None`、`messages=[]`、
    `conversation_data=None`——它们各自对应管线里一道短路，缺一不可：

    · `_conversation_messages(code, None, [])` 返回 `([], {})`，不碰磁盘
    · `context_manager.prepare` 因 `not messages[kept_from:]` 提前返回，不压缩
    · `_select_relevant_turns` 因 `len(turns) < history_select_min_turns` 提前返回，
      不额外调一次模型
    · `_answer` 的写回被 `if user_code and conversation_id:` 拦住

    任何一道失效都会让「只看到当次问题」不再成立。改这里之前先看
    `_self_check` 里对应的断言。
    """
    _require_internal_api_key(x_hdw_internal_key)
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question is required")
    selected_mode = _resolve_inference_mode(req.inference_mode)
    if selected_mode == "online" and not _online_inference_allowed(_API_SERVICE_IDENTITY):
        raise HTTPException(status_code=403, detail="在线推理未开放给服务通道")
    pipeline = _qa_pipeline(
        question=question,
        user=_API_SERVICE_IDENTITY,
        req=QARequest(
            question=question,
            top_k=req.top_k,
            inference_mode=selected_mode,
            client_time=req.client_time,
        ),
        image_attachments=[],
        selected_mode=selected_mode,
        now=_local_now(req.client_time),
        clock_source="server",
    )
    async for event in pipeline:
        if "result" in event:
            return event["result"]
    raise HTTPException(status_code=500, detail="pipeline produced no result")


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
    # 实时取数失败必须进 prompt，而且要明说「是这次没取到，不是没这个能力」。
    # 实测过的失效：日志源头不通，模型收到的是一片空白，于是答「我没有实时感知能力，
    # 当前对话中未提供实时信息检索功能」——用户读到的结论是「系统不支持」。
    assert "实时数据取数失败" not in _prompt("q", [])[1]["content"], "没失败时不该出现该块"
    _le = _prompt("q", [], live_errors=["日志: ReadTimeout"])
    assert "实时数据取数失败" in _le[1]["content"], "取数失败没进 user 块"
    assert "日志: ReadTimeout" in _le[1]["content"], "失败明细没带上"
    assert "不是「系统没有实时数据能力」" in _le[0]["content"], "缺「取不到 ≠ 没能力」规则"
    # 空消息异常必须有类型名兜底（httpx 的超时异常 str() 就是空串）
    assert _exc_text(httpx.ReadTimeout("")) == "ReadTimeout"
    assert _exc_text(ValueError("bad")) == "ValueError: bad"
    # 日志工具：取数失败不能讲成「该时段无记录」。失败时 events 同样是空的，
    # 判定顺序反了就会把「重试可能就好」说成「本来就没有」——实测踩过。
    _liems_down = {
        "ok": False,
        "query_status": "liems_http_error",
        "fetch_result": {"ok": False, "stderr": "Read timed out. (read timeout=30.0)"},
        "events": [],
    }
    _r = _logs_failure_reason(_liems_down)
    assert "取数失败" in _r and "liems_http_error" in _r, f"失败原因没带出来: {_r!r}"
    assert "read timeout=30.0" in _r, "工具给的原始原因应保留"
    assert _logs_failure_reason({"ok": True, "query_status": "ok", "events": []}) == ""
    assert _logs_failure_reason({"status": "unavailable", "message": "未配置"}) == "未配置"
    # 日志摘要：**全量条数必须来自工具统计，不能是截断后的明细条数**。
    # 实测抓 161 条只注入 40 条，模型若把 40 当成全部，总结必然写少。
    _logs_fixture = {
        "kind": "logs", "start": "2026-09-11", "end": "2026-09-16", "major_only": False,
        "summary": {
            "total_events": 161, "major_events": 69,
            "categories": {"运行": 120, "检修": 41},
            "brief": "2026-09-15 运行: 循环水泵启动",
        },
        "work_digest": {"dates": [{
            "date": "2026-09-15",
            "counts": {"main_work": 12, "ticket_work": 3, "important_timed_events": 2},
            "main_work": [{"time": "08:00", "source": "运行", "content": "循环水泵启动"}],
            "ticket_work": [], "important_timed_events": [],
        }]},
        "events": [{"time": "08:00", "source": "运行", "category": "启停",
                    "severity": "major", "content": "循环水泵启动"}],
    }
    _logs_body = _prompt("q", [], logs=_logs_fixture)[-1]["content"]
    assert "合计 161 条" in _logs_body, "全量条数没进提示词"
    assert "2026-09-15（主要工作 12 条、两票 3 条、重要事件 2 条）" in _logs_body, "逐日分栏没渲染"
    assert "〔主要工作〕08:00 运行：循环水泵启动" in _logs_body, "分栏明细没渲染"
    # 日志归纳规范：有日志才注入，且完整性/班次上下文跟着走。
    _sys_with_logs = _prompt("q", [], logs=_logs_fixture)[0]["content"]
    assert "按下面的口径归纳" in _sys_with_logs, "日志归纳规范没注入"
    assert "当前时间：" in _sys_with_logs, "当前时刻没进提示词"
    assert "白班 08:30~16:30" in _sys_with_logs, "班次划分没注入"
    assert "缺哪些日志类型、缺哪些班次必须如实说明" not in _sys_with_logs, "没有完整性数据时不该加完整性专项"
    _no_logs_sys = _prompt("q", [])[0]["content"]
    assert "按下面的口径归纳" not in _no_logs_sys, "没日志时不该注入日志规范"
    # 「现在几点」与日志无关，**两个分支都要有**：没日志的问答同样要能换算
    # 「昨天」「今天上午」。此前它藏在 _shift_context 里跟着 `if logs:` 走，
    # 于是没有日志时模型既不知道今天几号也不知道现在几点。
    assert "当前时间：" in _no_logs_sys, "没日志时也该注入当前时刻"
    assert "班次划分：" not in _no_logs_sys, "没日志时不该注入班次划分"

    # 班次判定：半点交接，跨零点的夜班不能算错
    assert _shift_of(datetime(2026, 9, 15, 8, 29)) == "夜班", "08:29 应属夜班"
    assert _shift_of(datetime(2026, 9, 15, 8, 30)) == "白班", "08:30 应属白班"
    assert _shift_of(datetime(2026, 9, 15, 16, 29)) == "白班", "16:29 应属白班"
    assert _shift_of(datetime(2026, 9, 15, 16, 30)) == "中班", "16:30 应属中班"
    assert _shift_of(datetime(2026, 9, 15, 23, 29)) == "中班", "23:29 应属中班"
    assert _shift_of(datetime(2026, 9, 15, 23, 30)) == "夜班", "23:30 应属夜班"
    assert _shift_of(datetime(2026, 9, 16, 0, 30)) == "夜班", "跨零点后仍是夜班"
    assert _shift_window("夜班", date(2026, 9, 15))[1].day == 16, "夜班结束落次日"
    assert _plan_shift("白班") == "白班" and _plan_shift("-") == "" and _plan_shift("全部") == ""

    # 班次筛选：有 shiftCode 的按码判，没有的退回按时刻判，**判不出的一律保留**
    _ev = [
        {"shiftCode": "YXBB", "time": "07:50:00", "date": "2026-09-15"},   # 表属白班
        {"shiftCode": "YXZB", "time": "17:00:00", "date": "2026-09-15"},   # 表属中班
        {"time": "10:00:00", "date": "2026-09-15"},                        # 无码，按时刻→白班
        {"time": "", "date": "2026-09-15"},                                # 判不出→保留
        {"time": "03:00:00", "date": "2026-09-15"},                        # 无码，按时刻→夜班
    ]
    _kept, _unknown = _filter_events_by_shift(_ev, "白班")
    assert len(_kept) == 3 and _unknown == 1, f"班次筛选不对：留 {len(_kept)}、未知 {_unknown}"

    # 完整性行：只报缺失项，且**尚未到来的班次不算缺失**
    _completeness = {"dates": [
        {"date": "2026-09-15", "byProgram": [
            {"programLabel": "值长日志", "complete": True, "missingShifts": []},
            {"programLabel": "化学日志", "complete": False, "missingShifts": ["夜班"]},
        ]},
    ]}
    _comp = _logs_completeness_line(_completeness, datetime(2026, 9, 16, 2, 0))
    assert _comp == "2026-09-15 化学日志缺夜班", _comp
    # 10:00 是白班：中班（16:30 开始）与夜班（23:30 开始）都还没到，不该报成缺失
    _comp = _logs_completeness_line(
        {"dates": [{"date": "2026-09-15", "byProgram": [
            {"programLabel": "化学日志", "complete": False,
             "missingShifts": ["白班", "中班", "夜班"]},
        ]}]},
        datetime(2026, 9, 15, 10, 0),
    )
    assert _comp == "2026-09-15 化学日志缺白班", _comp
    # 该到齐的都到齐、只是还有班次没开始——不能写成「各类型各班次齐全」
    _comp = _logs_completeness_line(
        {"dates": [{"date": "2026-09-15", "byProgram": [
            {"programLabel": "化学日志", "complete": False, "missingShifts": ["中班", "夜班"]},
        ]}]},
        datetime(2026, 9, 15, 10, 0),
    )
    assert _comp == "2026-09-15 截至当前班次已到的各类型齐全", _comp
    assert _logs_completeness_line({}, datetime(2026, 9, 15, 10, 0)) == ""
    # 班次**刚到**也不能算缺失。实测踩到：16:34 查当天，中班 16:30 才开始
    # 4 分钟，照样被报成「缺中班记录」——只摘「还没到」不够，还要摘「刚到」。
    assert not _shift_expected("中班", date(2026, 9, 15), datetime(2026, 9, 15, 16, 34)), \
        "中班刚开始几分钟不该算缺失"
    assert _shift_expected("中班", date(2026, 9, 15), datetime(2026, 9, 15, 17, 5)), \
        "过了宽限期就该算缺失"
    assert not _shift_expected("夜班", date(2026, 9, 15), datetime(2026, 9, 15, 23, 40)), \
        "夜班刚到也不算缺失"

    # 客户端时间戳：正常用客户端，偏差过大回落服务端并**留下痕迹**。
    # 基准取「服务端现在」——写死日期的话，脚本放到几天后跑就成了偏差过大，
    # 断言会莫名其妙地红。
    _client = _server_now() + timedelta(minutes=30)
    _cnow, _csrc = _clock_now(_client.isoformat())
    assert _csrc == "client", f"客户端时刻没被采纳：{_csrc}"
    assert abs(_cnow - _client) < timedelta(seconds=1), f"采纳的时刻不对：{_cnow}"
    _, _csrc = _clock_now((_server_now() + timedelta(hours=5)).isoformat())
    assert _csrc.startswith("server"), f"偏差过大应回落服务端：{_csrc}"
    _, _csrc = _clock_now((_server_now() - timedelta(hours=5)).isoformat())
    assert _csrc.startswith("server"), f"偏差过大（偏慢）也应回落：{_csrc}"
    _, _csrc = _clock_now("不是时间")
    assert _csrc.startswith("server"), f"解析失败应回落服务端：{_csrc}"
    # 服务端时钟必须是北京时间：容器没有 TZ，裸 datetime.now() 会走 UTC。
    # 这正是本次「15:24 被报成 07:24（夜班）」的根因。
    _now_offset = (_server_now().replace(tzinfo=None)
                   - datetime.now(timezone.utc).replace(tzinfo=None))
    assert timedelta(hours=7) < _now_offset < timedelta(hours=9), \
        f"服务端时刻不在 UTC+8：偏移 {_now_offset}"
    _with_comp = dict(_logs_fixture)
    _with_comp["log_completeness"] = {"dates": [{"date": "2026-09-15", "byProgram": []}]}
    assert "日志完整性：" in _prompt("q", [], logs=_with_comp)[-1]["content"], "完整性行没渲染"
    assert "缺哪些班次必须如实说明" in _prompt("q", [], logs=_with_comp)[0]["content"], "完整性专项没注入"
    # brief 与逐日分栏是两套说法，同时给会互相打架：有分栏时以分栏为准。
    assert "摘要：" not in _logs_body, "有逐日分栏时不该再输出 brief"
    _no_digest = dict(_logs_fixture)
    _no_digest["work_digest"] = {}
    assert "摘要：2026-09-15 运行: 循环水泵启动" in _prompt("q", [], logs=_no_digest)[-1]["content"], \
        "没有分栏时 brief 应作为兜底出现"
    assert "循环水泵启动" in _logs_body
    # 日志工具的参数派发。**`log_query_range` 只认 start_date / end_date**——
    # 曾经传的是 `days`，那个参数不存在、被无声丢弃，日期留空后退化成
    # start = end = 今天，于是规划器算出的 5 天区间永远只查回一天。
    # 这类「参数名写错、不报错、只是结果变少」的 bug 只能靠断言拦住。
    _sent: list[tuple[str, dict[str, Any]]] = []

    async def _fake_mcp(tool: str, args: dict[str, Any]) -> dict[str, Any]:
        _sent.append((tool, dict(args)))
        return {"ok": True, "query_status": "ok", "events": [], "summary": {}, "work_digest": {}}

    _orig_mcp = globals()["_mcp_call_tool"]
    globals()["_mcp_call_tool"] = _fake_mcp
    try:
        asyncio.run(_fetch_logs({"start": "2026-09-11", "end": "2026-09-16", "major_only": False}))
        assert _sent[0][0] == "log_query_range", _sent
        # **limit 必须显式传**：不传就落到工具默认值（range 是 120），
        # 「抓 161 条」在中途已经先被压到 120，再被 _LOGS_MAX 截到 40。
        # 参数少传一个同样不报错、只是结果变少，只能靠断言拦。
        assert _sent[0][1] == {"start_date": "2026-09-11", "end_date": "2026-09-16",
                               "limit": _LOGS_FETCH_LIMIT}, _sent
        _sent.clear()
        asyncio.run(_fetch_logs({"start": "2026-09-11", "end": "2026-09-16", "major_only": True}))
        assert _sent[0][0] == "log_query_major_events", _sent
        assert _sent[0][1] == {"days": 6, "limit": _LOGS_FETCH_LIMIT}, _sent
    finally:
        globals()["_mcp_call_tool"] = _orig_mcp

    # 全量模式：日志类问题不截断，且提示词不再说「受上限截断」
    _many = [{"time": f"{h:02d}:00:00", "date": "2026-09-15", "source": "运行",
              "category": "启停", "severity": "normal", "content": f"第 {h} 条"}
             for h in range(_LOGS_MAX + 5)]
    _full_fixture = dict(_logs_fixture)
    _full_fixture["events"] = _many
    _full_fixture["full"] = True
    _full_fixture["work_digest"] = {}
    _full_body = _prompt("q", [], logs=_full_fixture)[-1]["content"]
    assert f"[日志 {len(_many)}]" in _full_body, "全量模式漏事件"
    assert "受上限截断" not in _full_body, "全量模式不该再说被截断"
    # 非全量：截断说明必须在，且逐日分栏照旧渲染
    _part_fixture = dict(_logs_fixture)
    _part_fixture["full"] = False
    _part_body = _prompt("q", [], logs=_part_fixture)[-1]["content"]
    assert "受上限截断" in _part_body, "非全量模式要说清被截断"
    assert "〔主要工作〕08:00 运行：循环水泵启动" in _part_body, "非全量模式分栏明细应保留"

    # 班次筛选的措辞：命中条数与「无班次信息」都要说出来
    _shift_fixture = dict(_logs_fixture)
    _shift_fixture["shift"] = "白班"
    _shift_fixture["shift_unknown"] = 2
    _shift_body = _prompt("q", [], logs=_shift_fixture)[-1]["content"]
    assert "班次筛选：已按白班" in _shift_body, "班次口径没进提示词"
    assert "2 条无明确班次信息" in _shift_body, "保留下来的未知班次条数没说明"
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

    # 规划器失败重试的判据。真实故障：LLM 侧偶发 HTTP 400 空响应体，被
    # `except Exception` 吞成默认计划——不取实时数据，用户问趋势却得到
    # 「缺少实时数据」。重试要能识别「这次结果不像话」。
    # 规划器重试的判据：只看**结构完整性**，不看内容像不像某个意图。
    # 内容判断是那次模型调用本身该干的事，外面再拿词表猜一遍只会覆盖它的结论。
    assert _plan_is_complete("检索: yes\n测点: 无\n历史: 无\n趋势: 无\n日志: 无")
    assert _plan_is_complete("检索: no\n测点: x\n历史: y\n趋势: z\n日志: w")
    assert not _plan_is_complete("检索: yes\n测点: 无"), "少了三行说明这次没读全"
    assert not _plan_is_complete("")
    assert not _plan_is_complete(None)

    # 检索闸门只向「要检索」单向生效，唯一依据是**本轮带了图片**。
    # 这里没有「什么算该检索的提问」的词表——判断提问类型是模型的事
    # （见 _plan_rag 上方）。没有词表就不会随题面内容漂移，
    # 也不会把「趋势」这类实时取数的线索误判成文档检索。

    # ── 图片转写：候选排序与可用性判据 ──
    # 转写视图接受 mineru，答题视图只接受 chat。这条界线错了的后果很具体：
    # 答题视图收进一个 mineru 项，`_vision_candidates` 就会拿它当对话模型去调用，
    # 必然失败，而用户看到的是「多模态不可用」。
    _vp = {"vision_priority": [
        {"priority": 2, "kind": "chat", "mode": "offline", "model": "m.gguf"},
        {"priority": 1, "kind": "chat", "mode": "online", "model": "deepseek-flash"},
    ]}
    assert [c["model"] for c in _ordered_vision_candidates(_vp, {"chat"})] == \
        ["deepseek-flash", "m.gguf"], "应按 priority 升序"
    _mineru_only = {"vision_priority": [
        {"priority": 1, "kind": "mineru", "mode": "offline", "model": ""},
    ]}
    _transcribe_view = _ordered_vision_candidates(_mineru_only, {"chat", "mineru"})
    assert len(_transcribe_view) == 1 and _transcribe_view[0]["kind"] == "mineru"
    assert _transcribe_view[0]["model"] == "mineru", "mineru 项缺 model 时应补成标签"
    assert _ordered_vision_candidates(_mineru_only, {"chat"}) == [], "答题视图必须排除 mineru"
    # kind 缺省为 chat：老配置里没有 kind 字段，必须当 chat 用，否则升级后视觉整体失效
    _legacy = {"vision_priority": [{"priority": 1, "mode": "online", "model": "deepseek-flash"}]}
    assert _ordered_vision_candidates(_legacy, {"chat"})[0]["kind"] == "chat"
    # enabled=false 不参与；mode 非法的 chat 项要被跳过而不是拿去调用
    assert _ordered_vision_candidates(
        {"vision_priority": [{"priority": 1, "kind": "chat", "mode": "online", "model": "x", "enabled": False}]},
        {"chat"}) == []
    assert _ordered_vision_candidates(
        {"vision_priority": [{"priority": 1, "kind": "chat", "mode": "", "model": "x"}]}, {"chat"}) == []

    # 坏转写比没转写更糟——它会把检索引到无关文档上，而下游看不出区别。
    assert _usable_transcript("一、填空题\n1. 汽轮机额定转速为 3000 r/min")
    assert not _usable_transcript("")
    assert not _usable_transcript("短")
    assert not _usable_transcript("抱歉，我无法识别这张图片里的文字")
    assert not _usable_transcript("图片不清晰，请重新上传一张")
    # 但「看不清」出现在正文里是正常的（题目本身可能就在说这个词），只在开头算推脱
    assert _usable_transcript("1. 当发现仪表看不清时，应先核对照明与镜面，再联系热工。")

    # 原指令必须保留：它带着用户意图（「只给答案」「按题号给」），规划器要靠它判类别
    assert _compose_retrieval_question("做卷子", None) == "做卷子"
    assert _compose_retrieval_question("做卷子", {"text": "   "}) == "做卷子"
    _cq = _compose_retrieval_question("做卷子", {"text": "1. 主蒸汽温度"})
    assert _cq.startswith("做卷子") and "1. 主蒸汽温度" in _cq

    assert _truncate_for_planner("短问题") == "短问题"
    _long = _truncate_for_planner("题" * (_PLANNER_INPUT_MAX_CHARS + 500))
    assert len(_long) <= _PLANNER_INPUT_MAX_CHARS + 10 and _long.endswith("（已截断）")

    # 提示词：转写块与「别把 OCR 当证据」的规则必须同时出现或同时不出现。
    # 少了规则，模型会拿 OCR 去跟检索证据打对台，比不转写还糟。
    _ip = _prompt("q", [], image_text="一、填空题 1. 额定转速 3000 r/min")
    assert "图片文字" in _ip[-1]["content"] and "额定转速" in _ip[-1]["content"]
    assert "不要把它当作证据引用" in _ip[0]["content"], "缺了这条会拿 OCR 跟证据打对台"
    _np = _prompt("q", [])
    assert "图片文字" not in _np[-1]["content"], "没有转写时不该出现空的图片块"
    assert "不要把它当作证据引用" not in _np[0]["content"], "没有转写时不该出现那条规则"

    # ── 数字核对 ──
    # 机械校验，不判语义：只问「这个数在证据里出现过没有」。
    _ev = "轴承温度>95℃，跳机值130℃，润滑油压0.25MPa"
    assert _unsupported_numbers("轴承温度95℃，跳机130℃", _ev) == [], "证据里有的不该报"
    assert _unsupported_numbers("报警值为 999℃", _ev) == ["999"], "证据里没有的要报出来"
    # 尾零写法要归一化后再比：证据写 95，答案写 95.0 不算无出处
    assert _unsupported_numbers("轴承温度 95.0℃", _ev) == []
    # 单个数字（序号、量词）不报——中文行文里全是「1」「2」，报它们只有噪声
    assert _unsupported_numbers("1. 第一条\n2. 第二条", _ev) == []
    # 重复出现只报一次
    assert _unsupported_numbers("999 和 999", _ev) == ["999"]
    assert _unsupported_numbers("", _ev) == []
    assert _unsupported_numbers(None, _ev) == []

    # ── 检索规划：模型输出 → 查询列表 ──
    # 由模型决定发几路，取代了原来的关键词表 + 正则切分。这里只锁「输出解析」
    # 这一段纯逻辑——「拆得对不对」是模型的事，测不了也不该用规则去兜。
    assert _parse_query_plan("汽轮机超速保护动作转速") == ["汽轮机超速保护动作转速"]
    assert _parse_query_plan("1. 第一题\n2. 第二题") == ["第一题", "第二题"], "行首编号要剥掉"
    assert _parse_query_plan("- 甲\n* 乙\n· 丙") == ["甲", "乙", "丙"], "列表符号要剥掉"
    assert _parse_query_plan("甲\n\n   \n乙") == ["甲", "乙"], "空行要跳过"
    assert _parse_query_plan("甲\n甲") == ["甲"], "重复的查询只留一条"
    assert _parse_query_plan("`甲`") == ["甲"], "反引号要剥掉"
    assert _parse_query_plan("") == [] and _parse_query_plan(None) == []
    assert len(_parse_query_plan("\n".join(f"查询{i}" for i in range(200)))) <= _RAG_MAX_QUERIES, \
        "查询数必须有上限，否则一份畸形输出会发出上百路检索"

    # 证据预算随路数增长：**不涨的话后面几路检索到了也进不了提示词**，等于白发。
    # 单路时必须等于原来的默认预算，否则单问一答的行为会被顺带改掉。
    assert _retrieval_evidence_budget("offline", 1) == settings.local_graph_top_k
    assert _retrieval_evidence_budget("online", 1) == settings.online_graph_top_k
    assert _retrieval_evidence_budget("offline", 20) > _retrieval_evidence_budget("offline", 1)
    assert _retrieval_evidence_budget("offline", 10_000) <= _RAG_LOCAL_EVIDENCE_CAP, "预算要有上限"
    assert _retrieval_evidence_budget("online", 10_000) <= _RAG_ONLINE_EVIDENCE_CAP
    assert _retrieval_evidence_budget("online", 20) > _retrieval_evidence_budget("offline", 20)

    # 组间轮转：高分小组不能吃光窗口。这条是「分组到底有没有用」的直接判据。
    _rr = _round_robin_by_round([
        {"id": "a1", "_rag_round": 1}, {"id": "a2", "_rag_round": 1}, {"id": "a3", "_rag_round": 1},
        {"id": "b1", "_rag_round": 2},
        {"id": "c1", "_rag_round": 3}, {"id": "c2", "_rag_round": 3},
    ])
    assert [i["id"] for i in _rr] == ["a1", "b1", "c1", "a2", "c2", "a3"], _rr
    assert sorted(i["id"] for i in _rr) == ["a1", "a2", "a3", "b1", "c1", "c2"], "轮转不能丢项"
    assert _round_robin_by_round([]) == []
    # 头 3 条必须覆盖全部 3 组——这才是配额生效的样子
    assert len({i["_rag_round"] for i in _rr[:3]}) == 3

    # MinerU 上传名必须可预测且不重复：结果字典按它的 stem 索引，对不上就取空，
    # 而「取空」在日志里和「解析失败」长得一样，会把排查引向错误方向。
    assert _mineru_upload_name(0, {"mime_type": "image/png"}) == "image0.png"
    assert _mineru_upload_name(3, {"mime_type": "image/jpeg"}) == "image3.jpg"
    assert _mineru_upload_name(1, {"mime_type": "image/webp"}) != _mineru_upload_name(2, {"mime_type": "image/webp"})
    # 未知 mime 也要给出确定的扩展名，不能抛
    assert _mineru_upload_name(0, {"mime_type": ""}).endswith(".png")



    assert _plan_has_realtime({"trend": {"keyword": "x"}})
    assert _plan_has_realtime({"points": ["x"]})
    assert not _plan_has_realtime({"points": [], "history": None, "trend": None, "logs": None})
    # 降级原因要能从 plan 传到 live_errors，否则「取数失败」会被讲成「没有数据」
    assert _live_errors({"degraded": "boom"}, []) == ["规划器降级（本次未取实时数据）：boom"]
    assert _live_errors({}, ["a"]) == ["a"]
    assert _live_errors({}, []) == []
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
    # 日志行的后两段：重大 / 班次。旧的三段式（无班次）必须照常解析。
    _p = _parse_plan("检索: no\n测点: 无\n历史: 无\n趋势: 无\n日志: 2026-09-13|2026-09-13|-|白班")
    assert _p["logs"]["shift"] == "白班" and _p["logs"]["major_only"] is False, _p["logs"]
    _p = _parse_plan("检索: no\n测点: 无\n历史: 无\n趋势: 无\n日志: 2026-09-13|2026-09-13|重大|-")
    assert _p["logs"]["shift"] == "" and _p["logs"]["major_only"] is True, _p["logs"]
    _p = _parse_plan("检索: no\n测点: 无\n历史: 无\n趋势: 无\n日志: 2026-09-13|2026-09-13|重大")
    assert _p["logs"]["shift"] == "" and _p["logs"]["major_only"] is True, _p["logs"]
    # 模型偶尔会把班次顶到第三段。**按内容认，不能按位置认**——
    # 当成长标记的话班次丢了，还白拿一批重大事件（结果变少又不报错的老毛病）。
    _p = _parse_plan("检索: no\n测点: 无\n历史: 无\n趋势: 无\n日志: 2026-09-13|2026-09-13|白班")
    assert _p["logs"]["shift"] == "白班" and _p["logs"]["major_only"] is False, _p["logs"]

    # 测点不存在 → 把「不检索」掰回「检索」。**只在「一个都没取到」且原因是
    # 「不存在」时触发**：取数失败不触发（重试可能就好）、取到了更不触发。
    assert _point_absent({}, [f"P.GAS.11: {_POINT_NOT_FOUND}（测点表中无匹配项）"])
    assert _point_absent({}, [f"x: {_POINT_NOT_FOUND}（候选中无相关测点）"])
    assert not _point_absent({"live_points": [{"kks": "A"}]},
                             [f"x: {_POINT_NOT_FOUND}"])
    assert not _point_absent({}, ["x: ReadTimeout"])
    assert not _point_absent({}, [])
    # 标记必须真的出现在 _resolve_point 的两条失败路径上——文案改了断言才会红
    _src = __file__ and open(__file__, encoding="utf-8").read()
    assert _src.count(f"{{_POINT_NOT_FOUND}}") >= 2, "测点不存在的两条路径没打标记"
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

    # 图表：指标必须从点算，且直线拟合只需两个端点
    # （踩过：读了 _series_significance 的键，那里没有 min/max/delta，结果是三个 None）
    _pts = [(1000.0 + i * 60, float(i)) for i in range(50)]
    _ch = _build_chart({"samples": [{"time": datetime.fromtimestamp(m, tz=timezone.utc).isoformat(), "value": v} for m, v in _pts],
                        "kind": "trend", "kks": "K", "description": "D", "unit": "U"}, _series_significance([v for _, v in _pts]))
    assert _ch["summary"]["min_value"] == 0.0, _ch["summary"]
    assert _ch["summary"]["max_value"] == 49.0, _ch["summary"]
    assert _ch["summary"]["delta_value"] == 49.0, _ch["summary"]
    assert _ch["summary"]["latest_value"] == 49.0, _ch["summary"]
    assert len(_ch["fit_series"]) == 2, "直线拟合只需首尾两点"
    assert _ch["special_points"], "极值点必须被标出"

    # 图元随显著性档位裁剪：低波动依然出图（人有权看到真实曲线），
    # 但裁掉一切「趋势断言」——拟合线与关键点都是断言，图还在，结论不替人下。
    def _chart_of(spread_step: float) -> dict[str, Any]:
        seq = [(1000.0 + i * 60, 100.0 + (i % 5) * spread_step) for i in range(60)]
        return _build_chart(
            {"samples": [{"time": datetime.fromtimestamp(m, tz=timezone.utc).isoformat(), "value": v} for m, v in seq],
             "kind": "trend", "kks": "K", "description": "D", "unit": "U"},
            _series_significance([v for _, v in seq]),
        )

    _ch0 = _chart_of(0.2)          # 占读数约 0.8% → level 0
    assert _series_significance([v for _, v in [(0.0, 100.0 + (i % 5) * 0.2) for i in range(60)]])["level"] == 0
    assert _ch0["series"], "最低档也要出图——不给图等于把人挡在真实数据外面"
    assert _ch0["fit_series"] == [], "最低档不给拟合线：整段没趋势时画斜线会误导人读出走势"
    assert _ch0["special_points"] == [], "最低档不给关键点"

    _ch1 = _chart_of(0.5)          # 占读数约 2.0% → level 1
    assert _series_significance([v for _, v in [(0.0, 100.0 + (i % 5) * 0.5) for i in range(60)]])["level"] == 1
    assert _ch1["fit_series"], "中档仍给拟合线"
    assert _ch1["special_points"] == [], "中档不给关键点"

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

    # 进度转发：任务内部产生的事件必须先于结果出来，否则「边等边报」就是空话——
    # 这条链路里 MCP 取数最慢的一项要 30 秒，等完再一次性给等于没有进度。
    async def _progress_probe() -> list[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        async def work() -> str:
            q.put_nowait({"stage": "mcp", "label": "查询运行日志"})
            # 让出一次控制权，模拟「事件先于完成」的真实时序
            await asyncio.sleep(0)
            return "done"

        seen: list[dict[str, Any]] = []
        async for event in _stream_until(asyncio.create_task(work()), q):
            seen.append(event)
        return seen

    _events = asyncio.run(_progress_probe())
    assert _events[-1] == {"__result__": "done"}, _events
    assert {"stage": "mcp", "label": "查询运行日志"} in _events, "进度事件没在结果之前转发出来"
    assert all("__result__" not in e for e in _events[:-1]), "结果只能出现在最后一条"
    assert _noop_progress("x", "y") is None, "空实现要能直接调用"
    # 服务间入口「不做上下文管理」的**确定性证据**。
    # 靠观察「第二轮回答有没有提到第一轮」来验证是不可靠的——模型本来就可能
    # 不提。这里直接断言管线拿到的是空历史、空会话数据，与模型行为无关。
    _hist, _cdata = _conversation_messages("api", None, [])
    assert (_hist, _cdata) == ([], {}), f"无会话时必须零磁盘读写: {_hist!r} {_cdata!r}"
    # 有 conversation_id 但文件不存在时同样不该炸，也不该造出目录
    _hist2, _cdata2 = _conversation_messages("api", "no-such-session", [])
    assert _hist2 == [], _hist2
    # 服务身份不能自带管理员权限：role=user 才走「在线推理是否对用户开放」这条判断
    assert _API_SERVICE_IDENTITY["role"] == "user", "服务身份不该是 admin"
    # 内部密钥：没配 = 拒绝（fail-closed），配了但给错 = 401，给对 = 放行。
    # 这里临时改 settings 再还原，避免依赖部署环境里到底配没配。
    _saved_key = settings.api_internal_key
    try:
        settings.api_internal_key = ""
        try:
            _require_internal_api_key("anything")
            raise AssertionError("未配置密钥时必须拒绝")
        except HTTPException as exc:
            assert exc.status_code == 503, exc.status_code
        settings.api_internal_key = "k" * 32
        for _bad in (None, "", "k" * 31, "k" * 33, "local-dev-key"):
            try:
                _require_internal_api_key(_bad)
                raise AssertionError(f"错误密钥必须 401: {_bad!r}")
            except HTTPException as exc:
                assert exc.status_code == 401, exc.status_code
        assert _require_internal_api_key("k" * 32) is None, "正确密钥要放行"
    finally:
        settings.api_internal_key = _saved_key
    # 拆查询提示词必须区分「事实题」和「列举题」。
    # 实测失效：旧提示词写着「只有一个问题时只输出一条」，于是
    # 「汽轮机主保护有哪些」只发一条查询，捞回来的全是字面撞上
    # 「主…保护」的温度保护，超速/EH油压/差胀这些逐项条目一条都进不了榜——
    # 培训教材里 28 个讲保护的片段，答案只用到 5 个。这条断言防的是
    # 有人把这个区分改回"单问题就单路"。
    assert "列举" in _RAG_QUERY_PLAN_PROMPT, "提示词丢了「列举型要拆子项」这条"
    assert "1 条就够" in _RAG_QUERY_PLAN_PROMPT, "提示词丢了「事实题不要多拆」这条"
    assert "{max_queries}" in _RAG_QUERY_PLAN_PROMPT, "占位符被改掉了"
    # 回退路径的路数上限必须是个小数字：它的唯一用途就是在串行后端上
    # 把耗时从「路数 × 7.5 s」压回来。
    assert 1 <= _RAG_FALLBACK_ROUTE_CAP <= 4, _RAG_FALLBACK_ROUTE_CAP
    # 没配远端时必须是「不可用」——这条决定了要不要压路数，判错就等于不压。
    _saved_remote = settings.rag_remote_urls
    try:
        settings.rag_remote_urls = ()
        assert asyncio.run(_remote_rag_available()) is False, "没配远端时必须判为不可用"
    finally:
        settings.rag_remote_urls = _saved_remote


if __name__ == "__main__":
    _self_check()
    print("QA API self-check passed")
