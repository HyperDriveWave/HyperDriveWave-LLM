"""对外问答 API 的留档：`chatdata/api/<session_id>.json`。

**只存问题与回答**。调用方要的是回答本身，证据（citations／graph_context）
留在 qa-api 侧，不进这里——这是使用方的明确要求，也让文件小得多。

文件外形沿用 WebUI 的会话文件（`id`/`title`/`messages`），这样以后要用同一套
工具读它不必再改格式。差别只在 messages 里没有 citations 字段。

写入是「读-改-写」全程持锁 + 临时文件原子替换。锁只在文件操作期间持有——
模型调用在锁**外面**（见 app.py），否则一次几十秒的生成会把所有会话串起来。
单 worker 假设，与 qa-api 的 `_conversation_lock` 同一套前提（见其注释）。
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CHATDATA_ROOT = Path(os.getenv("HDW_CHATDATA_ROOT", "/data/chatdata"))
API_DIR = CHATDATA_ROOT / "api"

# 目录名与 qa-api 的 `_CONVERSATION_ID_RE` 同源。**必须校验**：
# session_id 直接拼进文件名，不校验就是一个路径穿越漏洞
# （`../../etc/passwd` 之类）。
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_lock = threading.Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_session_id() -> str:
    """`api-<时间戳>-<随机>`。时间戳在前，列表按文件名排序时天然按时间有序。"""
    return f"api-{int(datetime.now(timezone.utc).timestamp() * 1000)}-{os.urandom(4).hex()}"


def _session_path(session_id: str) -> Path:
    if not SESSION_ID_RE.fullmatch(session_id or ""):
        raise ValueError("invalid session id")
    return API_DIR / f"{session_id}.json"


def validate_session_id(session_id: str) -> None:
    """只校验格式，不返回路径。调用方在建会话前用它挡掉非法 id，
    免得走到写文件那一步才发现。非法时抛 ValueError。"""
    _session_path(session_id)


def _read(session_id: str) -> dict[str, Any] | None:
    path = _session_path(session_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # 单个文件坏掉不该让整个列表 500。当它不存在，下次提问会重建。
        return None
    return data if isinstance(data, dict) else None


def _write(session_id: str, data: dict[str, Any]) -> None:
    API_DIR.mkdir(parents=True, exist_ok=True)
    path = _session_path(session_id)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def append_exchange(session_id: str, question: str, answer: str) -> int:
    """追加一轮问答，返回这是第几轮（user 消息的条数）。"""
    timestamp = now_iso()
    with _lock:
        data = _read(session_id) or {
            "id": session_id,
            "source": "api",
            # 标题取首问，与 WebUI 的会话列表一致，便于人读
            "title": question[:60],
            "created_at": timestamp,
            "messages": [],
        }
        data["messages"].append({"role": "user", "content": question, "created_at": timestamp})
        data["messages"].append({"role": "assistant", "content": answer, "created_at": timestamp})
        data["updated_at"] = timestamp
        _write(session_id, data)
    return sum(1 for m in data["messages"] if m.get("role") == "user")


def load(session_id: str) -> dict[str, Any] | None:
    with _lock:
        return _read(session_id)


def list_sessions(limit: int = 50) -> list[dict[str, Any]]:
    """按最近更新排序。先按 mtime 排序再读文件——会话多起来时不必全量解析。"""
    if not API_DIR.is_dir():
        return []
    entries: list[tuple[float, Path]] = []
    for path in API_DIR.glob("*.json"):
        try:
            entries.append((path.stat().st_mtime, path))
        except OSError:
            continue
    entries.sort(reverse=True)
    out: list[dict[str, Any]] = []
    for _, path in entries[: max(1, limit)]:
        data = _read(path.stem)
        if not data:
            continue
        out.append(
            {
                "session_id": data.get("id") or path.stem,
                "title": data.get("title", ""),
                "turns": sum(
                    1 for m in data.get("messages", []) if m.get("role") == "user"
                ),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
            }
        )
    return out


def delete(session_id: str) -> bool:
    with _lock:
        path = _session_path(session_id)
        if not path.is_file():
            return False
        path.unlink()
        return True


def self_check() -> None:
    # session_id 直接拼文件名，这几个必须被拒
    for bad in ("", "../etc/passwd", "a/b", "a.json", "x" * 65, "有中文"):
        try:
            _session_path(bad)
            raise AssertionError(f"非法 session id 必须拒绝: {bad!r}")
        except ValueError:
            pass
    assert SESSION_ID_RE.fullmatch(new_session_id()), "自动生成的 id 必须合法"
    assert SESSION_ID_RE.fullmatch("fault-diag-20260917-001"), "调用方自定义 id 要能用"

    # 指向临时目录跑一遍读写，不碰真实的 chatdata
    global API_DIR
    saved = API_DIR
    with tempfile.TemporaryDirectory() as tmp:
        API_DIR = Path(tmp) / "api"
        try:
            assert list_sessions() == [], "目录不存在时列表应为空而不是抛错"
            assert load("nope") is None
            assert delete("nope") is False

            assert append_exchange("s1", "第一问", "第一答") == 1
            assert append_exchange("s1", "第二问", "第二答") == 2
            doc = load("s1")
            assert doc is not None
            assert [m["role"] for m in doc["messages"]] == [
                "user",
                "assistant",
                "user",
                "assistant",
            ], doc["messages"]
            # 只存问答：消息里不该混进 citations 之类
            assert set(doc["messages"][0]) == {"role", "content", "created_at"}, doc["messages"][0]
            assert doc["messages"][1]["content"] == "第一答", doc["messages"][1]
            assert doc["title"] == "第一问"[:60]
            assert doc["source"] == "api"

            assert append_exchange("s2", "另一会话", "另一答") == 1
            assert {s["session_id"] for s in list_sessions()} == {"s1", "s2"}
            assert list_sessions(limit=1)[0]["session_id"] in {"s1", "s2"}, "limit 要生效"
            assert delete("s1") is True
            assert load("s1") is None
            assert {s["session_id"] for s in list_sessions()} == {"s2"}
        finally:
            API_DIR = saved


if __name__ == "__main__":
    self_check()
    print("HDW API store self-check passed")
