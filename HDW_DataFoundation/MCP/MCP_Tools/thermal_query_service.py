import csv
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from difflib import SequenceMatcher
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple


MCP_FEATURE = {
    "id": "thermal_query_service",
    "mcp_name": "thermal_query_service.py",
    "function": "热成像OCR测点温度与趋势查询",
    "version": "V0.1",
    "sequence": 20,
    "tools": ["thermal_query_search_measurements", "thermal_query_current_temperature", "thermal_query_history_series"],
}


BASE_DIR = Path(__file__).resolve().parent
THERMAL_DIR = Path(os.getenv("HDW_MCP_THERMAL_ROOT", "/data/thermal")).resolve()
THERMAL_DATA_DIR = THERMAL_DIR / "DATA"
THERMAL_CONFIG_PATH = THERMAL_DIR / "Thermal_confgi.json"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from rtsp_query_service import RTSPQueryService  # type: ignore
except Exception:  # pragma: no cover
    RTSPQueryService = None  # type: ignore


def safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed


def normalize_time_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        parsed = None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y%m%d%H%M%S"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        if parsed is None:
            return text
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed.astimezone().isoformat(timespec="seconds")


def parse_time(value: Any) -> Optional[datetime]:
    normalized = normalize_time_text(value)
    if not normalized:
        return None
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def detect_query_mode(text: str) -> str:
    query = str(text or "").strip().lower()
    for keyword in ("历史", "趋势", "曲线", "变化", "波动", "回看", "过去", "最近", "近"):
        if keyword in query:
            return "history"
    return "current"


def infer_query_window(text: str) -> Tuple[datetime, datetime]:
    now = datetime.now().astimezone()
    query = str(text or "").strip().lower()
    match = re.search(r"(?:近|最近|过去)\s*(\d+)\s*(分钟|分|小时|时|天|日|d|h|m)", query)
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        if unit in {"分钟", "分", "m"}:
            return now - timedelta(minutes=max(1, value)), now
        if unit in {"小时", "时", "h"}:
            return now - timedelta(hours=max(1, value)), now
        if unit in {"天", "日", "d"}:
            return now - timedelta(days=max(1, value)), now
    if "今天" in query:
        return now.replace(hour=0, minute=0, second=0, microsecond=0), now
    if "昨天" in query or "昨日" in query:
        end = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return end - timedelta(days=1), end
    return now - timedelta(minutes=30), now


def normalize_query_text(text: str) -> str:
    query = str(text or "").strip().lower()
    query = query.replace("＃", "#")
    query = re.sub(r"(帮我|请|查询|查一下|查|看一下|看看|当前|实时|温度|历史|趋势|曲线|变化|测点|热成像|热像|红外|的)", " ", query)
    query = re.sub(r"[，。、“”‘’：:；;,.!?！？_\-/\\()\[\]{}]+", " ", query)
    return re.sub(r"\s+", " ", query).strip()


def compact_text(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", str(text or "").strip().lower())


def tokenize_query(text: str) -> List[str]:
    normalized = normalize_query_text(text)
    tokens = re.findall(r"#[0-9A-Za-z\u4e00-\u9fff]{1,}|[A-Za-z0-9._:-]{2,}|[\u4e00-\u9fff]{2,}", normalized)
    extras = []
    for keyword in ("盘车", "电机", "轴承", "驱动端", "非驱动端", "停机闭式水泵", "温度"):
        if keyword in str(text or ""):
            extras.append(keyword)
    seen = set()
    output: List[str] = []
    for token in tokens + extras:
        item = token.strip().lower()
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


class ThermalQueryService:
    def __init__(self) -> None:
        self._rtsp_service = RTSPQueryService() if RTSPQueryService is not None else None

    def looks_like_thermal_query(self, query_text: str) -> bool:
        text = str(query_text or "").strip()
        if not text:
            return False
        markers = ("热成像", "热像", "红外", "温度", "测温", "轴承", "盘车", "电机")
        if any(marker in text for marker in markers):
            return True
        normalized = normalize_query_text(text)
        for measurement in self._load_measurements():
            name = str(measurement.get("measurement_name", "") or "").lower()
            if name and (name in normalized or normalized in name):
                return True
        return False

    def get_query_mode(self, query_text: str) -> str:
        return detect_query_mode(query_text)

    def _load_config_streams(self) -> Dict[str, Dict[str, Any]]:
        cfg = _read_json(THERMAL_CONFIG_PATH)
        streams = cfg.get("streams") if isinstance(cfg.get("streams"), dict) else {}
        return {str(key): value for key, value in streams.items() if isinstance(value, dict)}

    def _stream_for_measurement(self, measurement_name: str, stream_key: str = "") -> Dict[str, Any]:
        streams = self._load_config_streams()
        if stream_key and stream_key in streams:
            return {"stream_id": stream_key, **streams[stream_key]}
        target = str(measurement_name or "").strip()
        for sid, stream in streams.items():
            for item in stream.get("measurements", []) if isinstance(stream.get("measurements"), list) else []:
                if isinstance(item, dict) and str(item.get("name", "") or "").strip() == target:
                    return {"stream_id": sid, **stream}
        return {}

    def _stream_status(self, stream_id: str, stream: Dict[str, Any]) -> Dict[str, Any]:
        source = str(stream.get("source", "") or "")
        source_type = str(stream.get("source_type", "") or "")
        if source_type == "local" or (source and not source.lower().startswith("rtsp://")):
            return {
                "stream_id": stream_id,
                "name": str(stream.get("display_name", "") or stream_id),
                "stream_type": "thermal",
                "source_type": source_type or "local",
                "online": False,
                "status": "local",
                "message": "本地视频用于试验分析，不作为实时RTSP状态。",
            }
        if self._rtsp_service is not None and stream_id:
            try:
                resolved = self._rtsp_service.stream_status(stream_id=stream_id).get("stream", {})
                if resolved:
                    return resolved
            except Exception:
                pass
        return {
            "stream_id": stream_id,
            "name": str(stream.get("display_name", "") or stream_id),
            "stream_type": "thermal",
            "source_type": source_type or "rtsp",
            "online": False,
            "status": "offline",
            "message": "RTSP流未连接或未在RTSP管理中配置。",
        }

    def _read_csv_rows(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        last_error: Optional[Exception] = None
        for encoding in ("utf-8-sig", "utf-8", "gbk"):
            try:
                with path.open("r", encoding=encoding, newline="") as handle:
                    return [dict(row) for row in csv.DictReader(handle)]
            except Exception as exc:
                last_error = exc
        if last_error:
            raise last_error
        return []

    def _value_from_row(self, row: Dict[str, Any]) -> Optional[float]:
        for key in ("value_celsius", "mean_celsius", "max_celsius", "min_celsius"):
            value = safe_float(row.get(key))
            if value is not None:
                return value
        return None

    def _load_measurements(self) -> List[Dict[str, Any]]:
        if not THERMAL_DATA_DIR.exists():
            return []
        measurements: Dict[str, Dict[str, Any]] = {}
        for path in sorted(THERMAL_DATA_DIR.glob("*.csv")):
            rows = self._read_csv_rows(path)
            latest = None
            for row in rows:
                if self._value_from_row(row) is None:
                    continue
                latest = row
            fallback_name = path.stem
            name = str((latest or {}).get("measurement_name", "") or fallback_name).strip()
            stream_key = str((latest or {}).get("stream_key", "") or "").strip()
            stream = self._stream_for_measurement(name, stream_key)
            if not stream_key:
                stream_key = str(stream.get("stream_id", "") or "")
            key = f"{stream_key}:{name}" if stream_key else name
            measurements[key] = {
                "source": "thermal_ocr",
                "measurement_name": name,
                "description": name,
                "type": str((latest or {}).get("type", "") or "point").strip() or "point",
                "stream_key": stream_key,
                "stream_name": str(stream.get("display_name", "") or stream_key),
                "csv_path": str(path),
                "sample_count": len(rows),
            }
        return list(measurements.values())

    def search_measurements(self, query_text: str, limit: int = 10) -> Dict[str, Any]:
        query = str(query_text or "").strip()
        query_norm = normalize_query_text(query)
        query_compact = compact_text(query)
        tokens = tokenize_query(query)
        scored: List[Tuple[int, Dict[str, Any]]] = []
        for measurement in self._load_measurements():
            name = str(measurement.get("measurement_name", "") or "")
            name_lower = name.lower()
            name_compact = compact_text(name)
            stream_name = str(measurement.get("stream_name", "") or "").lower()
            score = 0
            reasons: List[str] = []
            if query_compact and query_compact == name_compact:
                score += 1200
                reasons.append("measurement_compact_exact")
            elif query_compact and query_compact in name_compact:
                score += 850
                reasons.append("measurement_compact_contains")
            elif name_compact and name_compact in query_compact:
                score += 850
                reasons.append("query_compact_contains_measurement")
            if query_norm and query_norm == name_lower:
                score += 1000
                reasons.append("measurement_exact")
            elif query_norm and query_norm in name_lower:
                score += 650
                reasons.append("measurement_contains")
            elif name_lower and name_lower in query_norm:
                score += 650
                reasons.append("query_contains_measurement")
            for token in tokens:
                if token == name_lower:
                    score += 500
                    reasons.append(f"name_token:{token}")
                elif token in name_lower:
                    score += 180 + min(80, len(token) * 8)
                    reasons.append(f"name_contains:{token}")
                elif token in stream_name:
                    score += 80
                    reasons.append(f"stream_contains:{token}")
            if score <= 0 and any(marker in query for marker in ("热成像", "热像", "红外", "温度")):
                score = 20
                reasons.append("thermal_marker")
            if query_compact and name_compact:
                similarity = SequenceMatcher(None, query_compact, name_compact).ratio()
                score += int(similarity * 500)
                reasons.append(f"similarity:{similarity:.2f}")
            if score > 0:
                item = json.loads(json.dumps(measurement, ensure_ascii=False))
                item["score"] = score
                item["match_reason"] = ", ".join(reasons[:6])
                scored.append((score, item))
        scored.sort(key=lambda item: (-item[0], item[1].get("measurement_name", "")))
        actual_limit = max(1, min(50, int(limit or 10)))
        items = [item for _, item in scored[:actual_limit]]
        return {"query_text": query, "count": len(items), "items": items}

    def build_candidate_shortlist(self, query_text: str, limit: int = 8) -> List[Dict[str, Any]]:
        return self.search_measurements(query_text=query_text, limit=limit).get("items", [])

    def _resolve_measurement(self, measurement_name: str = "", query_text: str = "") -> Dict[str, Any]:
        target = str(measurement_name or "").strip()
        matches = self.search_measurements(target or query_text, limit=5).get("items", [])
        if not matches:
            raise ValueError("no matching thermal measurement found")
        if target:
            target_compact = compact_text(target)
            for item in matches:
                if compact_text(str(item.get("measurement_name", "") or "")) == target_compact:
                    return {
                        "selected": item,
                        "candidates": matches,
                        "ambiguous": len(matches) > 1 and int(matches[0].get("score", 0) or 0) == int(matches[1].get("score", 0) or 0),
                    }
        return {
            "selected": matches[0],
            "candidates": matches,
            "ambiguous": len(matches) > 1 and int(matches[0].get("score", 0) or 0) == int(matches[1].get("score", 0) or 0),
        }

    def resolve_candidate_by_name(self, measurement_name: str) -> Dict[str, Any]:
        return self._resolve_measurement(measurement_name=measurement_name)["selected"]

    def _latest_row(self, measurement: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        path = Path(str(measurement.get("csv_path", "") or ""))
        latest = None
        for row in self._read_csv_rows(path):
            if self._value_from_row(row) is None:
                continue
            latest = row
        return latest

    def current_value(self, measurement_name: str = "", query_text: str = "") -> Dict[str, Any]:
        resolved = self._resolve_measurement(measurement_name=measurement_name, query_text=query_text)
        measurement = resolved["selected"]
        stream = self._stream_for_measurement(
            str(measurement.get("measurement_name", "") or ""),
            str(measurement.get("stream_key", "") or ""),
        )
        stream_status = self._stream_status(str(measurement.get("stream_key", "") or ""), stream)
        if str(stream_status.get("status", "") or "") == "offline" or stream_status.get("online") is False:
            return {
                "query_mode": "current",
                "source": "thermal_ocr",
                "measurement": measurement,
                "candidates": resolved["candidates"],
                "value": None,
                "unit": "℃",
                "time": "",
                "value_source": "rtsp_offline",
                "stream_status": stream_status,
                "message": "因RTSP流未连接，测点暂无实时信息。",
            }
        row = self._latest_row(measurement)
        if row is None:
            raise RuntimeError(f"thermal measurement has no data: {measurement.get('measurement_name', '')}")
        return {
            "query_mode": "current",
            "source": "thermal_ocr",
            "measurement": measurement,
            "candidates": resolved["candidates"],
            "value": self._value_from_row(row),
            "unit": "℃",
            "time": normalize_time_text(row.get("timestamp")),
            "value_source": "thermal_csv_latest",
            "stream_status": stream_status,
            "raw": row,
        }

    def _load_history_rows(self, measurement: Dict[str, Any], start_dt: datetime, end_dt: datetime) -> List[Dict[str, Any]]:
        path = Path(str(measurement.get("csv_path", "") or ""))
        samples: List[Dict[str, Any]] = []
        for row in self._read_csv_rows(path):
            value = self._value_from_row(row)
            if value is None:
                continue
            sample_time = parse_time(row.get("timestamp"))
            if sample_time is None:
                continue
            if sample_time < start_dt or sample_time > end_dt:
                continue
            samples.append({"time": sample_time.isoformat(timespec="seconds"), "value": value})
        samples.sort(key=lambda item: item["time"])
        return samples

    def history_series(
        self,
        measurement_name: str = "",
        query_text: str = "",
        start_time: str = "",
        end_time: str = "",
        interval_seconds: int = 0,
    ) -> Dict[str, Any]:
        resolved = self._resolve_measurement(measurement_name=measurement_name, query_text=query_text)
        measurement = resolved["selected"]
        if start_time and end_time:
            start_dt = parse_time(start_time)
            end_dt = parse_time(end_time)
            if start_dt is None or end_dt is None:
                raise ValueError("invalid start_time or end_time")
        else:
            start_dt, end_dt = infer_query_window(query_text)
        if end_dt <= start_dt:
            raise ValueError("end_time must be later than start_time")
        samples = self._load_history_rows(measurement, start_dt, end_dt)
        if not samples:
            all_rows = self._load_history_rows(
                measurement,
                datetime.now().astimezone() - timedelta(days=3650),
                datetime.now().astimezone() + timedelta(days=1),
            )
            if all_rows:
                samples = all_rows[-120:]
                first_time = parse_time(samples[0].get("time"))
                last_time = parse_time(samples[-1].get("time"))
                if first_time is not None:
                    start_dt = first_time
                if last_time is not None:
                    end_dt = last_time
        if not samples:
            raise RuntimeError(f"thermal measurement history has no values: {measurement.get('measurement_name', '')}")
        try:
            actual_interval = max(0, int(interval_seconds or 0))
        except (TypeError, ValueError):
            actual_interval = 0
        if actual_interval > 1:
            samples = self._downsample(samples, actual_interval)
        values = [float(item["value"]) for item in samples]
        latest_value = values[-1] if values else None
        first_value = values[0] if values else None
        delta_value = None if latest_value is None or first_value is None else latest_value - first_value
        if delta_value is None:
            trend = "unknown"
        elif abs(delta_value) < 0.05:
            trend = "stable"
        elif delta_value > 0:
            trend = "up"
        else:
            trend = "down"
        duration_minutes = max(1e-9, (parse_time(samples[-1]["time"]) - parse_time(samples[0]["time"])).total_seconds() / 60.0) if len(samples) > 1 and parse_time(samples[-1]["time"]) and parse_time(samples[0]["time"]) else 0
        rate = (delta_value / duration_minutes) if delta_value is not None and duration_minutes else None
        stream = self._stream_for_measurement(
            str(measurement.get("measurement_name", "") or ""),
            str(measurement.get("stream_key", "") or ""),
        )
        return {
            "query_mode": "history",
            "source": "thermal_ocr",
            "measurement": measurement,
            "candidates": resolved["candidates"],
            "start_time": start_dt.astimezone().isoformat(timespec="seconds"),
            "end_time": end_dt.astimezone().isoformat(timespec="seconds"),
            "interval_seconds": actual_interval,
            "unit": "℃",
            "samples": samples,
            "stream_status": self._stream_status(str(measurement.get("stream_key", "") or ""), stream),
            "summary": {
                "sample_count": len(samples),
                "latest_value": latest_value,
                "first_value": first_value,
                "min_value": min(values) if values else None,
                "max_value": max(values) if values else None,
                "mean_value": mean(values) if values else None,
                "delta_value": delta_value,
                "trend": trend,
                "rate_c_per_min": rate,
            },
        }

    def _downsample(self, samples: List[Dict[str, Any]], interval_seconds: int) -> List[Dict[str, Any]]:
        if interval_seconds <= 1 or len(samples) <= 2:
            return samples
        output: List[Dict[str, Any]] = []
        last_dt: Optional[datetime] = None
        for item in samples:
            current_dt = parse_time(item.get("time"))
            if current_dt is None:
                continue
            if last_dt is None or (current_dt - last_dt).total_seconds() >= interval_seconds:
                output.append(item)
                last_dt = current_dt
        if samples[-1] not in output:
            output.append(samples[-1])
        return output

    def query_resolved_measurement(
        self,
        measurement: Dict[str, Any],
        query_text: str = "",
        mode: str = "",
        start_time: str = "",
        end_time: str = "",
        interval_seconds: int = 0,
    ) -> Dict[str, Any]:
        name = str((measurement or {}).get("measurement_name", "") or "").strip()
        if not name:
            raise ValueError("resolved thermal measurement name is required")
        actual_mode = str(mode or "").strip().lower() or detect_query_mode(query_text)
        if actual_mode == "history":
            return self.history_series(
                measurement_name=name,
                query_text=query_text,
                start_time=start_time,
                end_time=end_time,
                interval_seconds=interval_seconds,
            )
        return self.current_value(measurement_name=name, query_text=query_text)

    def handle_voice_query(self, query_text: str) -> Dict[str, Any]:
        normalized = str(query_text or "").strip()
        if not normalized:
            return {"handled": False, "reason": "empty_query"}
        if not self.looks_like_thermal_query(normalized):
            return {"handled": False, "reason": "no_thermal_query_marker"}
        mode = detect_query_mode(normalized)
        candidates = self.build_candidate_shortlist(normalized, limit=5)
        if not candidates:
            return {"handled": False, "reason": "no_matching_thermal_measurement"}
        result = self.query_resolved_measurement(candidates[0], query_text=normalized, mode=mode)
        measurement = result.get("measurement", {})
        name = str(measurement.get("measurement_name", "") or "")
        if mode == "history":
            summary = result.get("summary", {})
            trend_text = {"up": "整体上升", "down": "整体下降", "stable": "整体平稳"}.get(str(summary.get("trend", "")), "暂无法判断趋势")
            reply = (
                f"{name}历史趋势中，最新温度为{self._fmt(summary.get('latest_value'))}摄氏度，"
                f"最低{self._fmt(summary.get('min_value'))}摄氏度，最高{self._fmt(summary.get('max_value'))}摄氏度，{trend_text}。"
            )
        else:
            if result.get("value") is None and result.get("value_source") == "rtsp_offline":
                reply = f"因RTSP流未连接，{name}暂无实时信息。"
            else:
                reply = f"{name}当前温度为{self._fmt(result.get('value'))}摄氏度，数据时间为{result.get('time', '')}。"
        return {
            "handled": True,
            "mode": mode,
            "reply": reply,
            "tool_result": result,
            "tool_name": f"thermal_query_{mode}",
        }

    def _fmt(self, value: Any) -> str:
        parsed = safe_float(value)
        if parsed is None:
            return "未知"
        return f"{parsed:.2f}"
