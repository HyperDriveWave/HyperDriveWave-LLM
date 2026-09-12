from datetime import datetime, timedelta, timezone
import os
import sys
from typing import Any, Dict, List, Optional

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - Python without zoneinfo fallback.
    ZoneInfo = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NATIVE_MODULE_DIR = os.path.dirname(BASE_DIR)
WEBUI_DIR = os.path.join(NATIVE_MODULE_DIR, "WebUI")

if WEBUI_DIR not in sys.path:
    sys.path.insert(0, WEBUI_DIR)

import webui_worker_client  # noqa: E402


DISPLAY_TIMEZONE_NAME = os.environ.get("SMARTGASTURBINE_DISPLAY_TZ", "Asia/Shanghai")
if ZoneInfo is not None:
    try:
        DISPLAY_TIMEZONE = ZoneInfo(DISPLAY_TIMEZONE_NAME)
    except Exception:
        DISPLAY_TIMEZONE = timezone(timedelta(hours=8))
else:
    DISPLAY_TIMEZONE = timezone(timedelta(hours=8))


class WorkerBackedPointQueryService:
    """Use local point resolution, but execute SIS reads on the Linux worker."""

    def __init__(self, local_service: Any, point_module: Any) -> None:
        self.local_service = local_service
        self.point_module = point_module

    def __getattr__(self, name: str) -> Any:
        return getattr(self.local_service, name)

    def _worker_call(self, method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return webui_worker_client.call("LSTM_SERVICE_URL", method, payload)

    def _resolve_and_validate(self, kks: str = "", point_name: str = "", query_text: str = "") -> Dict[str, Any]:
        resolved = self.local_service._resolve_point(kks=kks, point_name=point_name, query_text=query_text)
        clarification = self.local_service._unit_clarification(query_text, resolved)
        if clarification:
            return {"clarification": clarification}
        if not str(kks or "").strip():
            clarification = self.local_service._point_name_clarification(query_text, resolved)
            if clarification:
                return {"clarification": clarification}
        point = resolved["selected"]
        self.local_service._validate_unit_consistency(query_text, point)
        return {"resolved": resolved, "point": point}

    def _safe_float(self, value: Any) -> Optional[float]:
        return self.point_module.safe_float(value)

    def _normalize_time_text(self, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return self.point_module.normalize_time_text(text)
        if parsed.tzinfo is not None:
            return parsed.isoformat(timespec="seconds")
        return self.point_module.normalize_time_text(text)

    def _display_time_text(self, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return self.point_module.normalize_time_text(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=DISPLAY_TIMEZONE)
        return parsed.astimezone(DISPLAY_TIMEZONE).isoformat(timespec="seconds")

    def _parse_datetime(self, value: str) -> datetime:
        text = str(value or "").strip()
        if not text:
            raise ValueError("datetime text is required")
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone()

    def _history_window(self, query_text: str, start_time: str, end_time: str) -> tuple[datetime, datetime]:
        if start_time and end_time:
            return self._parse_datetime(start_time), self._parse_datetime(end_time)
        return self.point_module.infer_query_window(query_text)

    def _recommended_interval_seconds(self, start_time: datetime, end_time: datetime, requested: int = 0) -> int:
        if requested and requested > 0:
            return requested
        total_seconds = max(1, int((end_time - start_time).total_seconds()))
        if total_seconds <= 15 * 60:
            return 1
        if total_seconds <= 60 * 60:
            return 2
        if total_seconds <= 2 * 60 * 60:
            return 5
        return 10

    def _last_history_value(self, point: Dict[str, Any], minutes: int = 120) -> Optional[Dict[str, Any]]:
        try:
            result = self.history_series(
                kks=str(point.get("kks", "") or ""),
                query_text=f"当前值回退 近{max(1, int(minutes or 120))}分钟趋势",
            )
        except Exception:
            return None
        samples = result.get("samples", []) if isinstance(result, dict) else []
        if not isinstance(samples, list) or not samples:
            return None
        last_sample = samples[-1]
        value = self._safe_float(last_sample.get("value"))
        if value is None:
            return None
        return {
            "value": value,
            "time": self._normalize_time_text(str(last_sample.get("time", "") or "")),
            "unit": str(result.get("unit", "") or ""),
            "value_source": "linux_worker_history_fallback",
        }

    def _live_unit(self, point: Dict[str, Any]) -> str:
        try:
            result = self._worker_call(
                "sis_live_snapshot",
                {"payload": {"tags": [point["kks"]], "force_refresh": False}},
            )
        except Exception:
            return ""
        items = result.get("items", []) if isinstance(result, dict) else []
        for item in items if isinstance(items, list) else []:
            if str(item.get("kks", "") or "").strip() == str(point["kks"]):
                return str(item.get("unit", "") or "").strip()
        if items and isinstance(items[0], dict):
            return str(items[0].get("unit", "") or "").strip()
        return ""

    def current_value(self, kks: str = "", point_name: str = "", query_text: str = "") -> Dict[str, Any]:
        resolved_payload = self._resolve_and_validate(kks=kks, point_name=point_name, query_text=query_text)
        if "clarification" in resolved_payload:
            return resolved_payload["clarification"]
        resolved = resolved_payload["resolved"]
        point = resolved_payload["point"]

        result = self._worker_call(
            "sis_live_snapshot",
            {"payload": {"tags": [point["kks"]], "force_refresh": True}},
        )
        items = result.get("items", []) if isinstance(result, dict) else []
        record = None
        for item in items if isinstance(items, list) else []:
            if str(item.get("kks", "") or "").strip() == str(point["kks"]):
                record = item
                break
        if record is None and items:
            record = items[0]

        value = self._safe_float((record or {}).get("value"))
        if value is None:
            fallback = self._last_history_value(point, minutes=120)
            if fallback is None:
                raise RuntimeError(f"SIS live response has no value for: {point['kks']}")
            return {
                "query_mode": "current",
                "point": point,
                "candidates": resolved["candidates"],
                "value": fallback.get("value"),
                "unit": fallback.get("unit", ""),
                "quality": "",
                "time": fallback.get("time", ""),
                "value_source": fallback.get("value_source", "linux_worker_history_fallback"),
                "source": "linux-5090-worker-01",
            }

        return {
            "query_mode": "current",
            "point": point,
            "candidates": resolved["candidates"],
            "value": value,
            "unit": str((record or {}).get("unit", "") or "").strip(),
            "quality": str((record or {}).get("quality", "") or "").strip(),
            "time": self._normalize_time_text(str((record or {}).get("time", "") or "")),
            "value_source": str((record or {}).get("value_source", "") or "sis_live"),
            "source": str(result.get("source", "") or "linux-5090-worker-01"),
        }

    def history_series(
        self,
        kks: str = "",
        point_name: str = "",
        query_text: str = "",
        start_time: str = "",
        end_time: str = "",
        interval_seconds: int = 0,
    ) -> Dict[str, Any]:
        resolved_payload = self._resolve_and_validate(kks=kks, point_name=point_name, query_text=query_text)
        if "clarification" in resolved_payload:
            return resolved_payload["clarification"]
        resolved = resolved_payload["resolved"]
        point = resolved_payload["point"]

        start_dt, end_dt = self._history_window(query_text, start_time, end_time)
        if end_dt <= start_dt:
            raise ValueError("end_time must be later than start_time")
        actual_interval = self._recommended_interval_seconds(start_dt, end_dt, int(interval_seconds or 0))
        payload = {
            "tags": [point["kks"]],
            "start_time": start_dt.astimezone().isoformat(timespec="seconds"),
            "end_time": end_dt.astimezone().isoformat(timespec="seconds"),
            "interval_seconds": actual_interval,
        }
        result = self._worker_call("sis_history_series", {"payload": payload})
        series_list = result.get("series", []) if isinstance(result, dict) else []
        target_series = None
        for item in series_list if isinstance(series_list, list) else []:
            if str(item.get("kks", "") or "").strip() == str(point["kks"]):
                target_series = item
                break
        if target_series is None and series_list:
            target_series = series_list[0]

        records: List[Dict[str, Any]] = []
        for sample in (target_series or {}).get("samples", []) if isinstance(target_series, dict) else []:
            value = self._safe_float(sample.get("value"))
            if value is None:
                continue
            records.append({"time": self._normalize_time_text(str(sample.get("time", "") or "")), "value": value})
        records.sort(key=lambda item: item["time"])
        if not records:
            raise RuntimeError(f"SIS history response has no values for: {point['kks']}")

        values = [item["value"] for item in records if item.get("value") is not None]
        latest_value = values[-1] if values else None
        first_value = values[0] if values else None
        delta_value = None if latest_value is None or first_value is None else latest_value - first_value
        if delta_value is None:
            trend = "unknown"
        elif abs(delta_value) < 1e-9:
            trend = "stable"
        elif delta_value > 0:
            trend = "up"
        else:
            trend = "down"

        unit = str((target_series or {}).get("unit", "") or "").strip()
        if not unit:
            unit = str(point.get("unit", "") or "").strip()
        if not unit:
            unit = self._live_unit(point)

        return {
            "query_mode": "history",
            "point": point,
            "candidates": resolved["candidates"],
            "start_time": self._display_time_text(str(result.get("start_time", "") or payload["start_time"])),
            "end_time": self._display_time_text(str(result.get("end_time", "") or payload["end_time"])),
            "interval_seconds": int(result.get("interval_seconds", actual_interval) or actual_interval),
            "payload_style": str(result.get("payload_style", "") or ""),
            "unit": unit,
            "samples": records,
            "summary": {
                "sample_count": len(records),
                "latest_value": latest_value,
                "first_value": first_value,
                "min_value": min(values) if values else None,
                "max_value": max(values) if values else None,
                "delta_value": delta_value,
                "trend": trend,
            },
            "source": str(result.get("source", "") or "linux-5090-worker-01"),
        }

    def query_resolved_point(
        self,
        point: Dict[str, Any],
        query_text: str = "",
        mode: str = "",
        start_time: str = "",
        end_time: str = "",
        interval_seconds: int = 0,
    ) -> Dict[str, Any]:
        selected_kks = str((point or {}).get("kks", "") or "").strip()
        if not selected_kks:
            raise ValueError("resolved point kks is required")
        actual_mode = str(mode or "").strip().lower() or self.point_module.detect_query_mode(query_text)
        if actual_mode == "history":
            return self.history_series(
                kks=selected_kks,
                query_text=query_text,
                start_time=start_time,
                end_time=end_time,
                interval_seconds=interval_seconds,
            )
        return self.current_value(kks=selected_kks, query_text=query_text)

    def handle_voice_query(self, query_text: str) -> Dict[str, Any]:
        normalized = str(query_text or "").strip()
        if not normalized:
            return {"handled": False, "reason": "empty_query"}
        if not self.looks_like_point_query(normalized):
            return {"handled": False, "reason": "no_point_query_marker"}
        mode = self.point_module.detect_query_mode(normalized)
        try:
            if mode == "history":
                result = self.history_series(query_text=normalized)
                point = result["point"]
                summary = result["summary"]
                trend_text = {
                    "up": "整体上升",
                    "down": "整体下降",
                    "stable": "整体基本稳定",
                    "unknown": "趋势暂时无法判断",
                }.get(str(summary.get("trend", "") or "unknown"), "趋势暂时无法判断")
                reply = (
                    f"{point['description']}，KKS {point['kks']}，在 {result['start_time']} 到 {result['end_time']} 这段时间内，"
                    f"共取到 {summary['sample_count']} 个历史点。最新值 {summary['latest_value']}，"
                    f"最低值 {summary['min_value']}，最高值 {summary['max_value']}，{trend_text}。"
                )
                return {"handled": True, "mode": "history", "reply": reply, "tool_result": result, "tool_name": "point_query_history_series"}

            result = self.current_value(query_text=normalized)
            point = result["point"]
            unit = str(result.get("unit", "") or "").strip()
            time_text = str(result.get("time", "") or "").strip()
            reply = f"{point['description']}，KKS {point['kks']}，当前值 {result.get('value')}{unit}。"
            if time_text:
                reply += f"数据时间 {time_text}。"
            return {"handled": True, "mode": "current", "reply": reply, "tool_result": result, "tool_name": "point_query_current_value"}
        except ValueError:
            candidates = self.search_points(normalized, limit=5).get("items", [])
            if not candidates:
                return {"handled": False, "reason": "no_matching_point"}
            candidate_text = "；".join(f"{item['description']}（{item['kks']}）" for item in candidates[:5])
            return {
                "handled": True,
                "mode": "search",
                "reply": f"我没有唯一匹配到测点。当前更接近的候选有：{candidate_text}。请再说得更具体一些。",
                "tool_result": {"items": candidates, "query_text": normalized},
                "tool_name": "point_query_search_points",
            }
        except RuntimeError as exc:
            return {
                "handled": True,
                "mode": mode,
                "reply": f"测点查询已命中，但当前没有拿到有效数据。{str(exc)}。",
                "tool_result": {"error": str(exc), "query_text": normalized},
                "tool_name": "point_query_current_value" if mode == "current" else "point_query_history_series",
            }
