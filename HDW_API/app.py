"""HyperDriveWave 对外问答 API。

给**其它项目**用的问答接口：提一个问题，拿回回答。内网直连，暂不经 FRP。

与 WebUI 那条路的差别只有两点，都是使用方要求的：
  · 不做上下文管理——每次提问都独立，模型只看到当次问题
  · 响应只要回答，不带证据

留档落在 `chatdata/api/<session_id>.json`（问题+回答），见 store.py。

**这里不实现问答本身**。检索、图谱、MCP 取数、生成全在 qa-api 里，本服务只做
三件事：校验调用方密钥、转发给 qa-api 的服务间入口、把问答落盘。之所以不
直接把 qa-api 的管线 import 进来，是因为那要复制它的全套依赖（postgres /
neo4j / rag / mcp / model-config / auth.csv / LLM_API 卷），等于跑第二份 qa-api
并多开一套数据库连接；转发只需要一个 URL 和一个密钥。
"""

from __future__ import annotations

import hmac
import os
import re
import time
from typing import Any, Literal

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

import store

# qa-api 的服务间入口。容器间走服务名，与宿主端口映射无关。
UPSTREAM_URL = os.getenv(
    "HDW_QA_API_URL", "http://hdw-qa-api:8080"
).strip().rstrip("/") + "/internal/qa/query"
# 服务间密钥，只有本容器和 qa-api 知道。**与给调用方的密钥刻意分开**：
# 合一的话，拿到对外密钥的人可以绕过本服务直连 qa-api（那条路不留档、
# 也没有调用方维度的吊销）。
INTERNAL_KEY = os.getenv("HDW_API_INTERNAL_KEY", "").strip()
# 调用方密钥的单密钥形式。**保留作向后兼容**：只配了一个调用方的部署不用动。
API_KEY = os.getenv("HDW_API_KEY", "").strip()

# 调用方标签的字符集。它会写进留档文件的 `caller` 字段，所以限制成安全字符，
# 免得一个手滑的标签把留档写坏。
_CALLER_LABEL_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def _load_api_keys(spec: str, legacy: str) -> list[tuple[str, str]]:
    """解析 `HDW_API_KEYS`，格式 `标签:密钥,标签:密钥`。

    配了它就用它——**一把密钥对一个调用方**，吊销其中一个不影响其它。
    没配则退回单密钥 `HDW_API_KEY`（标签取 `default`）。

    **格式错误直接抛异常，不静默跳过**：一条写错的条目若被忽略，表现是
    「某个调用方突然 401」而配置看上去没问题——那种问题很难查。
    宁可启动时就炸，错误信息里直接说清期望的格式。
    """
    keys: list[tuple[str, str]] = []
    for item in str(spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        # 按**第一个**冒号切：密钥里万一有冒号也不会把标签切坏
        label, sep, key = item.partition(":")
        if not sep:
            # 没写标签就整条当密钥，标签取 default（与单密钥路径一致）
            label, key = "default", label
        label, key = label.strip(), key.strip()
        if not key:
            raise ValueError(f"HDW_API_KEYS 里有一条没有密钥：{item!r}")
        if not _CALLER_LABEL_RE.fullmatch(label):
            raise ValueError(
                f"HDW_API_KEYS 的标签 {label!r} 不合法：只允许字母、数字、"
                "下划线和连字符（1-32 字符）"
            )
        keys.append((label, key))
    if not keys and legacy:
        keys = [("default", legacy)]
    return keys


API_KEYS = _load_api_keys(os.getenv("HDW_API_KEYS", ""), API_KEY)
# 上游超时。**必须大于管线自己最慢的路径**：MCP 取数最慢的日志工具要等
# LIEMS 门户 30 秒，之后才是生成。实测单发一次提问 5 秒到 2 分钟。
#
# 600 是为并发留的：llama 有 4 个槽位，本接口与 WebUI 共享，同时跑时各自
# 的吞吐会下降，生成长回答的耗时可能翻几倍。
#
# **调这个值必须连 `HDW_LLM_TIMEOUT` 一起调**，否则没用：llama 是非流式
# （LLM_API/client.py 里 "stream": False），实测响应头要等整段生成结束才发
# （time_starttransfer == time_total），所以 HDW_LLM_TIMEOUT 是**整段生成的
# 总时限**。它比这里小的话，会先抛 ReadTimeout，调用方拿到的仍是 502，
# 外层调到多大都白搭。约定：内层（LLM）必须小于外层（本值），
# 这样超时能以内层那个更有信息量的错误暴露出来。
QA_TIMEOUT = float(os.getenv("HDW_API_QA_TIMEOUT", "600"))

app = FastAPI(title="HyperDriveWave Public QA API", version="0.1.0")


# session_id 会直接作为文件名，所以字符集受限。**报错要说清是什么字符**——
# 只说 "invalid session id"，调用方（尤其用中文当会话名时）无从下手。
_SESSION_ID_HINT = "invalid session id：只允许字母、数字、下划线和连字符（A-Za-z0-9_-），1-64 字符，不能含中文或空格"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    # 可选。不传则自动生成，并把生成的 id 回给调用方，下次带上就能续到同一份留档。
    session_id: str | None = None
    inference_mode: Literal["online", "offline"] | None = None
    top_k: int = Field(default=40, ge=1, le=40)


def _require_caller(authorization: str | None, x_hdw_api_key: str | None) -> str:
    """校验调用方，**返回匹配到的标签**（用于留档区分是谁问的）。

    **未配置密钥时拒绝服务，不静默放行**：这个端口绑在所有网卡上，没有密钥
    就等于给局域网开了个免费的大模型入口。宁可接口不可用，也不能默认敞开。
    """
    if not API_KEYS:
        raise HTTPException(
            status_code=503,
            detail="未配置任何调用方密钥（HDW_API_KEYS / HDW_API_KEY），对外接口不可用",
        )
    presented = ""
    if authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:].strip()
    elif x_hdw_api_key:
        presented = x_hdw_api_key.strip()
    if presented:
        for label, key in API_KEYS:
            # 逐个比。密钥数量是个位数，不需要建索引；每一条都是常数时间比较。
            if hmac.compare_digest(presented, key):
                return label
    # 不区分「没带」和「带错了」——对调用方都一样，也不必告诉试探者哪一步错了
    raise HTTPException(status_code=401, detail="unauthorized")


@app.get("/health")
async def health() -> dict[str, Any]:
    """给部署校验用的探活。无鉴权，但也不泄露任何配置值。

    **报数量不报标签**：这个端点不需要鉴权，标签等于告诉任何能访问的人
    「有哪些项目在调这个接口」。
    """
    return {
        "status": "ok",
        "upstream": UPSTREAM_URL,
        "key_configured": bool(API_KEYS),
        "keys_configured": len(API_KEYS),
        "sessions": len(store.list_sessions(limit=1000)),
    }


@app.post("/api/v1/ask")
async def ask(
    req: AskRequest,
    authorization: str | None = Header(default=None),
    x_hdw_api_key: str | None = Header(default=None),
) -> dict[str, Any]:
    caller = _require_caller(authorization, x_hdw_api_key)
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question is required")
    if not INTERNAL_KEY:
        raise HTTPException(
            status_code=503,
            detail="HDW_API_INTERNAL_KEY 未配置，无法访问上游问答服务",
        )
    session_id = req.session_id or store.new_session_id()
    try:
        store.validate_session_id(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=_SESSION_ID_HINT) from exc

    payload: dict[str, Any] = {"question": question, "top_k": req.top_k}
    if req.inference_mode:
        payload["inference_mode"] = req.inference_mode

    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=QA_TIMEOUT) as client:
            response = await client.post(
                UPSTREAM_URL,
                json=payload,
                headers={"X-HDW-Internal-Key": INTERNAL_KEY},
            )
    except httpx.TimeoutException as exc:
        # httpx 的超时异常 str() 是空串，必须带类型名，否则报错信息是空的
        raise HTTPException(
            status_code=504,
            detail=f"上游问答超时（>{QA_TIMEOUT:.0f}s）：{type(exc).__name__}",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502, detail=f"上游问答不可达：{type(exc).__name__}"
        ) from exc

    if response.status_code == 401:
        # 上游用的是同一个变量，出现这个说明两侧配置不一致——这是运维问题，
        # 报给调用方看没意义，所以换一个能指向原因的措辞。
        raise HTTPException(
            status_code=502,
            detail="服务间密钥不匹配：本服务与 qa-api 的 HDW_API_INTERNAL_KEY 需一致",
        )
    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"上游返回 {response.status_code}：{response.text[:200]}",
        )

    result = response.json()
    answer = str(result.get("answer") or "")
    turn = store.append_exchange(session_id, question, answer, caller=caller)
    return {
        "session_id": session_id,
        "turn": turn,
        "question": question,
        "answer": answer,
        "elapsed_ms": int((time.monotonic() - started) * 1000),
    }


@app.get("/api/v1/sessions")
async def sessions(
    limit: int = 50,
    authorization: str | None = Header(default=None),
    x_hdw_api_key: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_caller(authorization, x_hdw_api_key)
    return {"sessions": store.list_sessions(limit=max(1, min(limit, 500)))}


@app.get("/api/v1/sessions/{session_id}")
async def session_detail(
    session_id: str,
    authorization: str | None = Header(default=None),
    x_hdw_api_key: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_caller(authorization, x_hdw_api_key)
    try:
        doc = store.load(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=_SESSION_ID_HINT) from exc
    if not doc:
        raise HTTPException(status_code=404, detail="session not found")
    return doc


@app.delete("/api/v1/sessions/{session_id}")
async def session_delete(
    session_id: str,
    authorization: str | None = Header(default=None),
    x_hdw_api_key: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_caller(authorization, x_hdw_api_key)
    try:
        deleted = store.delete(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=_SESSION_ID_HINT) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="session not found")
    return {"deleted": session_id}


def _self_check() -> None:
    """配置自检。**在 import 时跑**，与 qa-api 同一套约定：
    配错了要在启动时就炸，而不是等第一个请求进来才发现。"""
    assert UPSTREAM_URL.endswith("/internal/qa/query"), UPSTREAM_URL

    # ── 多密钥的解析 ──
    # 单密钥向后兼容：只配 HDW_API_KEY 时标签取 default
    assert _load_api_keys("", "") == [], "都没配就是空"
    assert _load_api_keys("", "solo") == [("default", "solo")], "单密钥要退回 default"
    two = _load_api_keys("fault:HDW-aaa,monitor:HDW-bbb", "")
    assert two == [("fault", "HDW-aaa"), ("monitor", "HDW-bbb")], two
    # 配了列表就以列表为准，legacy 不再补进来（否则会多出一把没人知道的密钥）
    assert _load_api_keys("fault:HDW-aaa", "solo") == [("fault", "HDW-aaa")]
    # 没写标签就整条当密钥，标签 default
    assert _load_api_keys("HDW-ccc", "") == [("default", "HDW-ccc")]
    # 密钥里带冒号时按**第一个**冒号切，不会把标签切坏
    assert _load_api_keys("fault:a:b", "") == [("fault", "a:b")]
    # 空条目（尾随逗号）要跳过，不能变成一把空密钥
    assert _load_api_keys("fault:HDW-aaa,", "") == [("fault", "HDW-aaa")]
    # **格式错误必须炸**：静默跳过会让某个调用方突然 401，而配置看着没问题
    for bad_spec in ("fault:", ":", "bad label:HDW-aaa", "有中文:HDW-aaa", "a" * 33 + ":HDW-aaa"):
        try:
            _load_api_keys(bad_spec, "")
            raise AssertionError(f"非法 HDW_API_KEYS 必须抛异常：{bad_spec!r}")
        except ValueError:
            pass

    # ── 校验分支 ── 临时改模块变量再还原，不依赖部署环境配没配
    global API_KEYS
    saved = API_KEYS
    try:
        API_KEYS = []
        try:
            _require_caller(None, None)
            raise AssertionError("未配置密钥必须拒绝")
        except HTTPException as exc:
            assert exc.status_code == 503, exc.status_code

        API_KEYS = [("fault", "c" * 32), ("monitor", "d" * 32)]
        # 注意最后一个：**长度对但内容错**的 Bearer，不能因为走对了格式就放行。
        # （曾经把正确密钥写进这个列表，断言失败反而暴露的是测试写错了。）
        for bad in (None, "", "c" * 31, "c" * 33, "Bearer " + "e" * 32, "Bearer ", "Basic " + "c" * 32):
            try:
                _require_caller(bad, None)
                raise AssertionError(f"错误密钥必须 401: {bad!r}")
            except HTTPException as exc:
                assert exc.status_code == 401, exc.status_code
        # **要返回匹配到的那个标签**——留档靠它区分调用方，返回错了比不返回更糟
        assert _require_caller("Bearer " + "c" * 32, None) == "fault", "Bearer 形式"
        assert _require_caller(None, "d" * 32) == "monitor", "X-HDW-API-Key 形式"
        assert _require_caller("bearer " + "d" * 32, None) == "monitor", "Bearer 大小写不敏感"
    finally:
        API_KEYS = saved
    store.self_check()


_self_check()


if __name__ == "__main__":
    print("HDW API self-check passed")
