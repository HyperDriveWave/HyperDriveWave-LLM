from __future__ import annotations

import importlib.util
import os
import re
import sys
from difflib import SequenceMatcher
from typing import Any, Dict, List


MCP_FEATURE = {
    "id": "alarm_query_service",
    "mcp_name": "alarm_query_service.py",
    "function": "活动报警、提示、历史报警查询与报警确认",
    "version": "V0.1",
    "sequence": 40,
    "tools": ["alarm_query_active", "alarm_query_history", "alarm_acknowledge"],
}


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NATIVE_MODULE_DIR = os.path.dirname(BASE_DIR)
MODULE_ACCESS_DIR = os.path.dirname(NATIVE_MODULE_DIR)
ALARM_MANAGER_DIR = os.path.join(NATIVE_MODULE_DIR, "Alarm_Manager")
ALARM_SERVICE_PATH = os.path.join(ALARM_MANAGER_DIR, "alarm_service.py")
DEFAULT_ACK_DEDUP_MINUTES = 15
MAX_ACK_DEDUP_MINUTES = 60

if ALARM_MANAGER_DIR not in sys.path:
    sys.path.insert(0, ALARM_MANAGER_DIR)


def load_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_level(value: Any) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "提示": "warning",
        "报警": "alarm",
        "严重": "critical",
        "严重报警": "critical",
        "warning": "warning",
        "alarm": "alarm",
        "critical": "critical",
        "info": "info",
    }
    return aliases.get(text, text if text in {"info", "warning", "alarm", "critical"} else "")


def infer_status(text: str) -> str:
    query = str(text or "").strip().lower()
    if any(token in query for token in ("历史", "记录", "过去", "history")):
        return "history"
    return "active"


def infer_level(text: str) -> str:
    query = str(text or "").strip().lower()
    for token in ("严重报警", "critical", "严重"):
        if token in query:
            return "critical"
    if "报警" in query or "alarm" in query:
        return "alarm"
    if "提示" in query or "warning" in query:
        return "warning"
    return ""


def infer_source(text: str) -> str:
    query = str(text or "").strip().lower()
    if any(token in query for token in ("sis", "测点", "kks")):
        return "sis_custom_point"
    if any(token in query for token in ("热成像", "热像", "thermal")):
        return "thermal_ocr"
    return ""


def infer_ack_dedup_minutes(text: str, value: Any = None) -> int:
    query = str(text or "").strip().lower()
    if any(token in query for token in ("不再报出", "不再播报", "不要再报", "不再报警", "别再报", "停止播报", "停止报警")):
        return MAX_ACK_DEDUP_MINUTES
    if value not in (None, ""):
        try:
            return max(0, min(MAX_ACK_DEDUP_MINUTES, int(float(value))))
        except (TypeError, ValueError):
            pass
    hour_match = re.search(r"(\d+(?:\.\d+)?)\s*(小时|时|h|hour)", query)
    if hour_match:
        return max(0, min(MAX_ACK_DEDUP_MINUTES, int(float(hour_match.group(1)) * 60)))
    minute_match = re.search(r"(\d+(?:\.\d+)?)\s*(分钟|分|min|minute)", query)
    if minute_match:
        return max(0, min(MAX_ACK_DEDUP_MINUTES, int(float(minute_match.group(1)))))
    return DEFAULT_ACK_DEDUP_MINUTES


class AlarmQueryService:
    def __init__(self) -> None:
        module = load_module("smartgasturbine_mcp_alarm_service", ALARM_SERVICE_PATH)
        self.alarm_service = module.AlarmManagerService()

    def looks_like_alarm_query(self, query_text: str) -> bool:
        text = str(query_text or "").strip().lower()
        return any(token in text for token in ("报警", "告警", "提示", "alarm", "越限", "活动报警", "历史报警"))

    def active_alarms(self, query_text: str = "", level: str = "", source: str = "", limit: int = 50) -> Dict[str, Any]:
        data = self.alarm_service.list_active()
        requested_level = normalize_level(level) or infer_level(query_text)
        requested_source = str(source or "").strip() or infer_source(query_text)
        alarms = self._filter(data.get("alarms", []), requested_level, requested_source, limit)
        return {
            "query_mode": "active",
            "query_text": str(query_text or ""),
            "level": requested_level,
            "source": requested_source,
            "count": len(alarms),
            "alarms": alarms,
            "updated_at": data.get("updated_at", ""),
        }

    def history_alarms(self, query_text: str = "", level: str = "", source: str = "", limit: int = 100) -> Dict[str, Any]:
        data = self.alarm_service.list_history(limit=max(1, min(5000, int(limit or 100))))
        requested_level = normalize_level(level) or infer_level(query_text)
        requested_source = str(source or "").strip() or infer_source(query_text)
        rows = self._filter(data.get("rows", []), requested_level, requested_source, limit)
        return {
            "query_mode": "history",
            "query_text": str(query_text or ""),
            "level": requested_level,
            "source": requested_source,
            "count": len(rows),
            "rows": rows,
            "updated_at": data.get("updated_at", ""),
        }

    def acknowledge_alarm(
        self,
        query_text: str = "",
        alarm_id: str = "",
        dedup_minutes: Any = None,
        user: str = "voice_assistant",
        limit: int = 10,
    ) -> Dict[str, Any]:
        query = str(query_text or "")
        minutes = infer_ack_dedup_minutes(query, dedup_minutes)
        active = self.alarm_service.list_active()
        rows = active.get("alarms", []) if isinstance(active.get("alarms"), list) else []
        selectable_rows = [item for item in rows if isinstance(item, dict) and not bool(item.get("acknowledged"))] or rows
        selected = self._select_alarm(selectable_rows, query, str(alarm_id or ""))
        candidates = selected.get("candidates", [])
        if selected.get("status") == "not_found":
            return {
                "query_mode": "acknowledge",
                "query_text": query,
                "acknowledged": False,
                "needs_confirmation": False,
                "dedup_minutes": minutes,
                "message": "当前没有匹配到可确认的活动报警。",
                "candidates": [],
                "updated_at": active.get("updated_at", ""),
            }
        if selected.get("status") == "ambiguous":
            return {
                "query_mode": "acknowledge",
                "query_text": query,
                "acknowledged": False,
                "needs_confirmation": True,
                "dedup_minutes": minutes,
                "message": "当前匹配到多条活动报警，请说明要确认哪一条。",
                "candidates": candidates[: max(1, min(20, int(limit or 10)))],
                "updated_at": active.get("updated_at", ""),
            }
        alarm = selected.get("alarm") if isinstance(selected.get("alarm"), dict) else {}
        ack_result = self.alarm_service.acknowledge_alarm(
            {
                "alarm_id": alarm.get("alarm_id", ""),
                "user": str(user or "voice_assistant"),
                "dedup_minutes": minutes,
            }
        )
        acknowledged = bool(ack_result.get("acknowledged"))
        return {
            "query_mode": "acknowledge",
            "query_text": query,
            "acknowledged": acknowledged,
            "needs_confirmation": False,
            "dedup_minutes": minutes,
            "selected_alarm": alarm,
            "candidates": candidates[: max(1, min(20, int(limit or 10)))],
            "result": ack_result,
            "message": "报警已确认。" if acknowledged else "报警确认失败，活动报警可能已经恢复或不存在。",
            "updated_at": active.get("updated_at", ""),
        }

    def summary(self, query_text: str = "", limit: int = 20) -> Dict[str, Any]:
        mode = infer_status(query_text)
        if mode == "history":
            return self.history_alarms(query_text=query_text, limit=limit)
        return self.active_alarms(query_text=query_text, limit=limit)

    def _filter(self, rows: List[Dict[str, Any]], level: str = "", source: str = "", limit: int = 50) -> List[Dict[str, Any]]:
        requested_level = str(level or "").strip().lower()
        requested_source = str(source or "").strip().lower()
        output: List[Dict[str, Any]] = []
        for item in rows if isinstance(rows, list) else []:
            item_level = str(item.get("level", "") or "").strip().lower()
            item_source = str(item.get("source", "") or "").strip().lower()
            searchable = " ".join(str(item.get(key, "") or "") for key in ("object_name", "object_id", "message", "source_name")).lower()
            if requested_level and item_level != requested_level:
                continue
            if requested_source and requested_source not in item_source and requested_source not in searchable:
                continue
            output.append(item)
            if len(output) >= max(1, min(5000, int(limit or 50))):
                break
        return output

    def _select_alarm(self, rows: List[Dict[str, Any]], query_text: str, alarm_id: str = "") -> Dict[str, Any]:
        items = [item for item in rows if isinstance(item, dict)]
        exact_id = str(alarm_id or "").strip()
        if exact_id:
            for item in items:
                if str(item.get("alarm_id", "") or "") == exact_id:
                    return {"status": "selected", "alarm": item, "candidates": [item]}
            return {"status": "not_found", "candidates": []}
        if not items:
            return {"status": "not_found", "candidates": []}
        if len(items) == 1:
            return {"status": "selected", "alarm": items[0], "candidates": items}
        query = self._normalize_match_text(query_text)
        scored = []
        for item in items:
            haystack = self._alarm_search_text(item)
            score = self._score_alarm_match(query, haystack)
            scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        candidates = [item for score, item in scored[:10] if score > 0]
        if not candidates:
            return {"status": "ambiguous", "candidates": items[:10]}
        top_score = scored[0][0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        if top_score >= 0.78 or (top_score >= 0.45 and top_score - second_score >= 0.18):
            return {"status": "selected", "alarm": scored[0][1], "candidates": candidates}
        if top_score >= 0.62 and second_score < 0.62:
            return {"status": "selected", "alarm": scored[0][1], "candidates": candidates}
        return {"status": "ambiguous", "candidates": candidates or items[:10]}

    def _alarm_search_text(self, item: Dict[str, Any]) -> str:
        return self._normalize_match_text(
            " ".join(
                str(item.get(key, "") or "")
                for key in ("alarm_id", "object_name", "object_id", "message", "source_name", "source", "level")
            )
        )

    def _normalize_match_text(self, value: str) -> str:
        text = str(value or "").lower()
        text = re.sub(r"(确认|报警|告警|提示|去重|不再报出|不再播报|不要再报|不再报警|别再报|停止播报|停止报警)", " ", text)
        text = re.sub(r"\d+(?:\.\d+)?\s*(小时|时|h|hour|分钟|分|min|minute)", " ", text)
        return re.sub(r"\s+", "", text)

    def _score_alarm_match(self, query: str, haystack: str) -> float:
        if not query:
            return 0.0
        if query in haystack:
            return 1.0
        query_tokens = [token for token in re.split(r"[，。；;,.、\s]+", query) if token]
        token_hits = sum(1 for token in query_tokens if token and token in haystack)
        token_score = token_hits / max(1, len(query_tokens)) if query_tokens else 0.0
        seq_score = SequenceMatcher(None, query, haystack).ratio()
        partial_score = 0.0
        for size in range(min(len(query), 16), 1, -1):
            parts = [query[index : index + size] for index in range(0, max(0, len(query) - size + 1))]
            if any(part in haystack for part in parts):
                partial_score = size / max(1, len(query))
                break
        return max(token_score, seq_score, partial_score)

    def format_reply(self, result: Dict[str, Any]) -> str:
        mode = str(result.get("query_mode", "") or "")
        if mode == "acknowledge":
            if result.get("acknowledged"):
                item = result.get("selected_alarm") if isinstance(result.get("selected_alarm"), dict) else {}
                label = item.get("object_name") or item.get("object_id") or item.get("message") or "该报警"
                minutes = int(result.get("dedup_minutes", DEFAULT_ACK_DEDUP_MINUTES) or DEFAULT_ACK_DEDUP_MINUTES)
                if minutes >= 60:
                    return f"已确认报警：{label}，1小时内不再重复播报。"
                return f"已确认报警：{label}，{minutes}分钟内不再重复播报。"
            rows = result.get("candidates") if isinstance(result.get("candidates"), list) else []
            if result.get("needs_confirmation") and rows:
                labels = []
                for item in rows[:5]:
                    labels.append(f"{item.get('object_name') or item.get('object_id')}: {item.get('message', '')}")
                return "当前匹配到多条报警，请说明要确认哪一条：" + "；".join(labels)
            return str(result.get("message", "") or "当前没有匹配到可确认的活动报警。")
        rows = result.get("alarms") if mode == "active" else result.get("rows")
        rows = rows if isinstance(rows, list) else []
        if not rows:
            return "当前没有匹配的报警或提示。"
        if mode == "history":
            labels = []
            for item in rows[:5]:
                labels.append(f"{item.get('object_name') or item.get('object_id')}: {item.get('message', '')}")
            return f"已查询到{len(rows)}条历史报警或提示，最近包括：" + "；".join(labels)
        labels = []
        for item in rows[:5]:
            labels.append(f"{item.get('object_name') or item.get('object_id')}: {item.get('message', '')}")
        return f"当前有{len(rows)}条活动报警或提示，主要包括：" + "；".join(labels)
