import json
import os
import re
import socket
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse, urlunparse


MCP_FEATURE = {
    "id": "rtsp_query_service",
    "mcp_name": "rtsp_query_service.py",
    "function": "RTSP流在线/离线状态查询",
    "version": "V0.1",
    "sequence": 30,
    "tools": ["rtsp_query_list_streams", "rtsp_query_stream_status"],
}


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from runtime import rtsp_streams  # noqa: E402


def _deep_copy(data: Any) -> Any:
    return json.loads(json.dumps(data, ensure_ascii=False))


def _mask_url_password(url_text: str) -> str:
    text = str(url_text or "").strip()
    if not text:
        return ""
    parsed = urlparse(text)
    if not parsed.username and not parsed.password:
        return text
    hostname = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    username = parsed.username or ""
    auth = f"{username}:******@" if username else ""
    netloc = f"{auth}{hostname}{port}"
    return urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))


def _probe_tcp(host: str, port: int, timeout: float = 0.8) -> bool:
    target_host = str(host or "").strip()
    if not target_host:
        return False
    try:
        with socket.create_connection((target_host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _normalize_stream_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in {"rgb", "thermal"} else "rgb"


def _normalize_status(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"online", "on", "true", "1", "在线"}:
        return "online"
    if text in {"offline", "off", "false", "0", "离线"}:
        return "offline"
    return ""


def _tokenize(text: str) -> List[str]:
    normalized = str(text or "").strip().lower()
    normalized = re.sub(r"[，。、“”‘’：:；;,.!?！？_\-/\\()\[\]{}]+", " ", normalized)
    tokens = re.findall(r"[A-Za-z0-9._:-]{2,}|[\u4e00-\u9fff]{2,}", normalized)
    seen = set()
    output: List[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        output.append(token)
    return output


class RTSPQueryService:
    def __init__(self, probe_timeout_seconds: float = 0.8) -> None:
        configured_timeout = os.environ.get("SMARTGASTURBINE_RTSP_MCP_PROBE_TIMEOUT_SECONDS", "")
        try:
            timeout_value = float(configured_timeout) if configured_timeout else float(probe_timeout_seconds or 0.8)
        except (TypeError, ValueError):
            timeout_value = float(probe_timeout_seconds or 0.8)
        self.probe_timeout_seconds = max(0.05, min(1.0, timeout_value))

    def looks_like_rtsp_query(self, query_text: str) -> bool:
        text = str(query_text or "").strip().lower()
        if not text:
            return False
        markers = (
            "rtsp",
            "流媒体",
            "视频流",
            "拉流",
            "摄像头",
            "相机",
            "热成像",
            "热像",
            "在线",
            "离线",
            "连接",
            "断开",
        )
        return any(marker in text for marker in markers)

    def _load_raw_streams(self) -> List[Dict[str, Any]]:
        return _deep_copy(rtsp_streams())

    def _normalize_stream(self, item: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
        source_url = str(item.get("source_url", "") or "").strip()
        target_url = str(item.get("target_url", "") or "").strip()
        mode = str(item.get("mode", "pull") or "pull").strip().lower()
        parsed = urlparse(source_url if mode == "pull" else (target_url or source_url))
        host = str(item.get("host", "") or parsed.hostname or "").strip()
        try:
            port = int(item.get("port", 0) or parsed.port or 554)
        except (TypeError, ValueError):
            port = 554
        return {
            "sequence": index + 1,
            "id": str(item.get("id", "") or f"rtsp_{index + 1}").strip(),
            "name": str(item.get("name", "") or f"RTSP流{index + 1}").strip(),
            "mode": mode if mode in {"pull", "push"} else "pull",
            "stream_type": _normalize_stream_type(item.get("stream_type", item.get("type", "rgb"))),
            "source_url": source_url,
            "target_url": target_url,
            "masked_source_url": _mask_url_password(source_url),
            "transport": str(item.get("transport", "tcp") or "tcp").strip().lower(),
            "host": host,
            "port": max(1, min(65535, port)),
            "channel": str(item.get("channel", "") or "").strip(),
            "notes": str(item.get("notes", "") or "").strip(),
        }

    def _public_stream(self, stream: Dict[str, Any]) -> Dict[str, Any]:
        online = _probe_tcp(stream.get("host", ""), int(stream.get("port", 554) or 554), self.probe_timeout_seconds)
        output = _deep_copy(stream)
        output["online"] = bool(online)
        output["status"] = "online" if online else "offline"
        output["checked_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        output.pop("source_url", None)
        return output

    def list_streams(
        self,
        stream_type: str = "",
        online_status: str = "",
        query_text: str = "",
        limit: int = 50,
    ) -> Dict[str, Any]:
        requested_type = _normalize_stream_type(stream_type) if str(stream_type or "").strip() else ""
        requested_status = _normalize_status(online_status)
        query = str(query_text or "").strip()
        query_lower = query.lower()
        tokens = _tokenize(query)
        if not requested_type:
            if any(marker in query_lower for marker in ("热成像", "热像", "thermal", "红外")):
                requested_type = "thermal"
            elif any(marker in query_lower for marker in ("rgb", "可见光", "普通")):
                requested_type = "rgb"
        if not requested_status:
            requested_status = _normalize_status(query)

        streams = [self._normalize_stream(item, index) for index, item in enumerate(self._load_raw_streams())]
        matched: List[Dict[str, Any]] = []
        for stream in streams:
            if requested_type and stream.get("stream_type") != requested_type:
                continue
            searchable = " ".join(
                str(stream.get(key, "") or "")
                for key in ("id", "name", "stream_type", "masked_source_url", "host", "channel", "notes")
            ).lower()
            if tokens and not any(token in searchable for token in tokens if token not in {"rtsp", "在线", "离线", "连接", "热成像"}):
                if not any(marker in query_lower for marker in ("在线", "离线", "热成像", "热像", "rtsp")):
                    continue
            matched.append(stream)
        try:
            actual_limit = max(1, min(100, int(limit or 50)))
        except (TypeError, ValueError):
            actual_limit = 50
        try:
            max_probes = max(
                1,
                min(100, int(os.environ.get("SMARTGASTURBINE_RTSP_MCP_MAX_PROBES", "20") or "20")),
            )
        except (TypeError, ValueError):
            max_probes = 20
        probe_budget = max(actual_limit, min(len(matched), max_probes))
        public: List[Dict[str, Any]] = []
        probed = 0
        for stream in matched:
            if requested_status and probed >= probe_budget:
                break
            if not requested_status and len(public) >= actual_limit:
                break
            item = self._public_stream(stream)
            probed += 1
            if requested_status and item.get("status") != requested_status:
                continue
            public.append(item)
            if len(public) >= actual_limit:
                break
        online_count = sum(1 for item in public if item.get("online"))
        return {
            "query_text": query,
            "count": len(public),
            "total_matched": len(matched),
            "probed_count": probed,
            "probe_limited": probed < len(matched),
            "online_count": online_count,
            "offline_count": len(public) - online_count,
            "streams": public,
        }

    def resolve_stream(self, stream_id: str = "", stream_name: str = "", query_text: str = "") -> Dict[str, Any]:
        target_id = str(stream_id or "").strip()
        target_name = str(stream_name or "").strip()
        query = str(query_text or "").strip()
        streams = [self._normalize_stream(item, index) for index, item in enumerate(self._load_raw_streams())]
        if target_id:
            for stream in streams:
                if stream.get("id") == target_id:
                    return self._public_stream(stream)
        candidates = []
        tokens = _tokenize(target_name or query)
        for stream in streams:
            score = 0
            name = str(stream.get("name", "") or "").lower()
            sid = str(stream.get("id", "") or "").lower()
            searchable = " ".join(
                str(stream.get(key, "") or "")
                for key in ("id", "name", "stream_type", "masked_source_url", "host", "channel", "notes")
            ).lower()
            if target_name and target_name.lower() == name:
                score += 500
            if target_name and target_name.lower() in name:
                score += 300
            for token in tokens:
                if token == sid:
                    score += 350
                elif token in searchable:
                    score += 120 + min(40, len(token) * 4)
            if score > 0:
                candidates.append((score, stream))
        if not candidates:
            raise ValueError("no matching RTSP stream found")
        candidates.sort(key=lambda item: (-item[0], item[1].get("id", "")))
        return self._public_stream(candidates[0][1])

    def stream_status(self, stream_id: str = "", stream_name: str = "", query_text: str = "") -> Dict[str, Any]:
        stream = self.resolve_stream(stream_id=stream_id, stream_name=stream_name, query_text=query_text)
        return {"stream": stream}

    def handle_voice_query(self, query_text: str) -> Dict[str, Any]:
        normalized = str(query_text or "").strip()
        if not normalized:
            return {"handled": False, "reason": "empty_query"}
        if not self.looks_like_rtsp_query(normalized):
            return {"handled": False, "reason": "no_rtsp_query_marker"}
        result = self.list_streams(query_text=normalized)
        streams = result.get("streams", [])
        if not streams:
            return {
                "handled": True,
                "mode": "rtsp_status",
                "reply": "当前没有匹配到RTSP流配置。",
                "tool_result": result,
                "tool_name": "rtsp_query_list_streams",
            }
        online_count = int(result.get("online_count", 0) or 0)
        offline_count = int(result.get("offline_count", 0) or 0)
        if "离线" in normalized or "断开" in normalized:
            names = "、".join(
                str(item.get("name", "") or item.get("id", ""))
                for item in streams[:5]
                if not item.get("online")
            )
            reply = f"当前离线RTSP流共{offline_count}路，{names}。"
        elif "在线" in normalized or "连接" in normalized:
            online_names = "、".join(
                str(item.get("name", "") or item.get("id", ""))
                for item in streams[:5]
                if item.get("online")
            )
            offline_names = "、".join(
                str(item.get("name", "") or item.get("id", ""))
                for item in streams[:5]
                if not item.get("online")
            )
            if online_names:
                reply = f"当前在线RTSP流共{online_count}路，{online_names}。"
            elif offline_names:
                reply = f"当前没有在线RTSP流，离线{offline_count}路，{offline_names}。"
            else:
                reply = f"当前没有在线RTSP流，离线{offline_count}路。"
        else:
            reply = f"当前匹配到{len(streams)}路RTSP流，其中在线{online_count}路，离线{offline_count}路。"
        return {
            "handled": True,
            "mode": "rtsp_status",
            "reply": reply,
            "tool_result": result,
            "tool_name": "rtsp_query_list_streams",
        }
