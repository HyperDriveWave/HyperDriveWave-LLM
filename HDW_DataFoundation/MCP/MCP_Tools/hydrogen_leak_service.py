import importlib.util
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None


MCP_FEATURE = {
    "id": "hydrogen_leak_service",
    "mcp_name": "hydrogen_leak_service.py",
    "function": "发电机漏氢量计算",
    "version": "V0.1",
    "sequence": 35,
    "tools": ["hydrogen_leak_calculate"],
}


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POINT_QUERY_SERVICE_PATH = os.path.join(BASE_DIR, "point_query_service.py")
POINT_QUERY_WORKER_PROXY_PATH = os.path.join(BASE_DIR, "point_query_worker_proxy.py")
DISPLAY_TIMEZONE_NAME = os.environ.get("HDW_DISPLAY_TZ", "Asia/Shanghai")
if ZoneInfo is not None:
    try:
        DISPLAY_TIMEZONE = ZoneInfo(DISPLAY_TIMEZONE_NAME)
    except Exception:
        DISPLAY_TIMEZONE = timezone(timedelta(hours=8))
else:
    DISPLAY_TIMEZONE = timezone(timedelta(hours=8))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


POINTS = {
    "hydrogen_pressure": {
        "suffix": "MKG26CP101XQ01",
        "label": "发电机氢气压力",
        "kind": "pressure",
    },
    "hot_h2_gas_turbine_side_temp": {
        "suffix": "MKA75CT104XQ01",
        "label": "热氢温度燃机侧",
        "kind": "temperature",
    },
    "hot_h2_steam_side_temp": {
        "suffix": "MKA75CT103XQ01",
        "label": "热氢温度汽机侧",
        "kind": "temperature",
    },
    "cold_h2_steam_side_temp": {
        "suffix": "MKA75CT101XQ01",
        "label": "冷氢温度汽机侧",
        "kind": "temperature",
    },
    "cold_h2_gas_turbine_side_temp": {
        "suffix": "MKA75CT102XQ01",
        "label": "冷氢温度燃机侧",
        "kind": "temperature",
    },
    "atmospheric_pressure": {
        "suffix": "MBL10CP101XQ01",
        "label": "大气压力",
        "kind": "pressure",
    },
}

HYDROGEN_LEAK_STANDARDS = [
    {
        "band": "PN≥0.5",
        "min_inclusive": 0.5,
        "max_exclusive": None,
        "excellent_nm3_per_day": 10.875,
        "good_nm3_per_day": 14.25,
        "qualified_nm3_per_day": 17.625,
    },
    {
        "band": "0.5＞PN≥0.4",
        "min_inclusive": 0.4,
        "max_exclusive": 0.5,
        "excellent_nm3_per_day": 9.75,
        "good_nm3_per_day": 12.75,
        "qualified_nm3_per_day": 15.75,
    },
    {
        "band": "0.4＞PN≥0.3",
        "min_inclusive": 0.3,
        "max_exclusive": 0.4,
        "excellent_nm3_per_day": 8.25,
        "good_nm3_per_day": 11.25,
        "qualified_nm3_per_day": 14.25,
    },
    {
        "band": "0.3＞PN≥0.2",
        "min_inclusive": 0.2,
        "max_exclusive": 0.3,
        "excellent_nm3_per_day": 4.5,
        "good_nm3_per_day": 6.0,
        "qualified_nm3_per_day": 7.5,
    },
    {
        "band": "0.2＞PN≥0.1",
        "min_inclusive": 0.1,
        "max_exclusive": 0.2,
        "excellent_nm3_per_day": 4.125,
        "good_nm3_per_day": 4.5,
        "qualified_nm3_per_day": 4.875,
    },
    {
        "band": "0.1＞PN",
        "min_inclusive": None,
        "max_exclusive": 0.1,
        "excellent_nm3_per_day": 3.0,
        "good_nm3_per_day": 3.375,
        "qualified_nm3_per_day": 4.125,
    },
]


def load_module(module_name: str, file_path: str):
    module_dir = os.path.dirname(os.path.abspath(file_path))
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_datetime(value: str) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            parsed = datetime.fromisoformat(candidate)
            return parsed.astimezone(DISPLAY_TIMEZONE) if parsed.tzinfo is not None else parsed.replace(tzinfo=DISPLAY_TIMEZONE)
        except ValueError:
            pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=DISPLAY_TIMEZONE)
        except ValueError:
            continue
    return None


def safe_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def iso_text(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=DISPLAY_TIMEZONE)
    return value.astimezone(DISPLAY_TIMEZONE).isoformat(timespec="seconds")


class HydrogenLeakService:
    def __init__(self, point_query_service: Any = None) -> None:
        self._point_query_service = point_query_service

    def _load_point_query_service(self):
        if self._point_query_service is not None:
            return self._point_query_service
        point_module = load_module("hdw_hydrogen_leak_point_query_service", POINT_QUERY_SERVICE_PATH)
        self._point_query_service = point_module.PointQueryService()
        return self._point_query_service

    def point_kks_for_unit(self, unit: int) -> Dict[str, Dict[str, str]]:
        unit_no = 2 if int(unit or 1) == 2 else 1
        prefix = f"{unit_no:02d}"
        return {
            key: {
                "kks": prefix + item["suffix"],
                "label": f"{unit_no}号机组{item['label']}",
                "kind": item["kind"],
            }
            for key, item in POINTS.items()
        }

    def calculate(
        self,
        unit: int = 1,
        start_time: str = "",
        end_time: str = "",
        lookback_hours: int = 72,
        generator_volume_m3: float = 125.0,
        rated_pressure_mpa: float = 0.5,
        pressure_jump_kpa: float = 30.0,
        interval_seconds: int = 300,
    ) -> Dict[str, Any]:
        unit_no = 2 if int(unit or 1) == 2 else 1
        points = self.point_kks_for_unit(unit_no)
        start_dt = parse_datetime(start_time)
        end_dt = parse_datetime(end_time)
        if bool(start_dt) != bool(end_dt):
            raise ValueError("start_time and end_time must be provided together")
        if start_dt and end_dt and end_dt <= start_dt:
            raise ValueError("end_time must be later than start_time")

        if not start_dt or not end_dt:
            end_dt = datetime.now().astimezone()
            start_dt = end_dt - timedelta(hours=max(24, min(24 * 14, int(lookback_hours or 72))))

        actual_interval = max(60, min(1800, int(interval_seconds or 300)))
        pressure_history = self._history(
            points["hydrogen_pressure"]["kks"],
            start_dt,
            end_dt,
            actual_interval,
        )
        pressure_samples = self._normalize_pressure_samples(pressure_history.get("samples", []))
        if not pressure_samples:
            raise RuntimeError(f"SIS history response has no usable hydrogen pressure values for: {points['hydrogen_pressure']['kks']}")

        window_result = self._select_stable_24h_window(pressure_samples, float(pressure_jump_kpa or 30.0))
        pressure_curve = self._build_pressure_curve(pressure_samples, window_result, points["hydrogen_pressure"])
        if not window_result.get("ok"):
            return {
                "ok": False,
                "query_mode": "hydrogen_leak",
                "unit": unit_no,
                "message": window_result.get("message", "未找到满足条件的24小时无补排氢区间。"),
                "pressure_point": points["hydrogen_pressure"],
                "searched_window": {"start_time": iso_text(start_dt), "end_time": iso_text(end_dt)},
                "pressure_jump_kpa": float(pressure_jump_kpa or 30.0),
                "jump_intervals": window_result.get("jump_intervals", []),
                "stable_segments": window_result.get("stable_segments", []),
                "pressure_curve": pressure_curve,
                "reply": window_result.get("message", "未找到满足条件的24小时无补排氢区间，请指定起止时间或扩大回看范围。"),
            }

        calc_start = parse_datetime(str(window_result["start_time"])) or start_dt
        calc_end = parse_datetime(str(window_result["end_time"])) or end_dt
        history_payload = self._fetch_point_histories(points, calc_start, calc_end, actual_interval)
        missing_points = history_payload.get("missing_points", [])
        if missing_points:
            message = self._format_missing_points_message(unit_no, missing_points)
            return {
                "ok": False,
                "query_mode": "hydrogen_leak",
                "unit": unit_no,
                "unit_label": f"{unit_no}号机组",
                "message": message,
                "selected_window": {
                    "start_time": iso_text(calc_start),
                    "end_time": iso_text(calc_end),
                    "duration_hours": (calc_end - calc_start).total_seconds() / 3600.0,
                },
                "missing_points": missing_points,
                "available_points": history_payload.get("available_points", []),
                "reply": message,
            }
        point_histories = history_payload.get("histories", {})
        try:
            review_points = self._build_review_points(points, point_histories, calc_start, calc_end, actual_interval)
        except Exception as exc:
            message = f"{unit_no}号机组漏氢计算已找到稳定24小时区间，但起止复核数据不完整：{exc}"
            return {
                "ok": False,
                "query_mode": "hydrogen_leak",
                "unit": unit_no,
                "unit_label": f"{unit_no}号机组",
                "message": message,
                "selected_window": {
                    "start_time": iso_text(calc_start),
                    "end_time": iso_text(calc_end),
                    "duration_hours": (calc_end - calc_start).total_seconds() / 3600.0,
                },
                "reply": message,
            }
        computed = self._compute_leak(review_points, calc_start, calc_end, float(generator_volume_m3 or 125.0))
        assessment = self._assess_leakage(computed, float(rated_pressure_mpa or 0.5))
        operator_review_table = self._build_operator_review_table(review_points)
        model_reply = self._format_model_reply(
            unit_no,
            computed,
            assessment,
            review_points,
            operator_review_table,
            window_result,
            float(pressure_jump_kpa or 30.0),
        )
        voice_reply = self._format_voice_reply(unit_no, computed, assessment, window_result)
        return {
            "ok": True,
            "query_mode": "hydrogen_leak",
            "unit": unit_no,
            "unit_label": f"{unit_no}号机组",
            "formula_source": "发电机漏氢计算.xlsx",
            "formula_note": "按Excel主表公式：使用氢压+大气压、冷热氢平均温度、标准温度20C、标准压力0.101325MPa、发电机容积计算Nm3/d。",
            "generator_volume_m3": float(generator_volume_m3 or 125.0),
            "rated_pressure_mpa": float(rated_pressure_mpa or 0.5),
            "pressure_jump_kpa": float(pressure_jump_kpa or 30.0),
            "selected_window": {
                "start_time": iso_text(calc_start),
                "end_time": iso_text(calc_end),
                "duration_hours": computed["duration_hours"],
            },
            "stable_window": window_result,
            "pressure_curve": pressure_curve,
            "review_points": review_points,
            "operator_review_table": operator_review_table,
            "calculation": computed,
            "assessment": assessment,
            "reply": model_reply,
            "model_reply": model_reply,
            "voice_reply": voice_reply,
        }

    def _history(self, kks: str, start_dt: datetime, end_dt: datetime, interval_seconds: int) -> Dict[str, Any]:
        try:
            result = self._load_point_query_service().history_series(
                kks=kks,
                query_text=f"{kks} 漏氢计算历史趋势",
                start_time=iso_text(start_dt),
                end_time=iso_text(end_dt),
                interval_seconds=int(interval_seconds),
            )
            point = result.get("point") if isinstance(result, dict) else {}
            resolved_kks = str((point or {}).get("kks", "") or kks).strip()
            if resolved_kks and resolved_kks != kks:
                raise ValueError(f"resolved unexpected point {resolved_kks} for fixed KKS {kks}")
            return result
        except Exception as exc:
            direct = self._history_direct_worker(kks, start_dt, end_dt, interval_seconds)
            if direct:
                direct["value_source"] = "linux_worker_direct_kks"
                direct["point_resolution_fallback"] = str(exc)
                return direct
            raise

    def _history_direct_worker(
        self,
        kks: str,
        start_dt: datetime,
        end_dt: datetime,
        interval_seconds: int,
    ) -> Optional[Dict[str, Any]]:
        try:
            import webui_worker_client  # noqa: WPS433

            if not webui_worker_client.enabled("LSTM_SERVICE_URL"):
                return None
            payload = {
                "tags": [kks],
                "start_time": iso_text(start_dt),
                "end_time": iso_text(end_dt),
                "interval_seconds": int(interval_seconds),
            }
            result = webui_worker_client.call("LSTM_SERVICE_URL", "sis_history_series", {"payload": payload})
        except Exception:
            return None
        series_list = result.get("series", []) if isinstance(result, dict) else []
        target_series = None
        for item in series_list if isinstance(series_list, list) else []:
            if str((item or {}).get("kks", "") or "").strip() == kks:
                target_series = item
                break
        if target_series is None and series_list:
            target_series = series_list[0]
        samples = []
        for sample in (target_series or {}).get("samples", []) if isinstance(target_series, dict) else []:
            value = safe_float((sample or {}).get("value"))
            time_text = str((sample or {}).get("time", "") or "").strip()
            if value is None or not time_text:
                continue
            samples.append({"time": time_text, "value": value})
        samples.sort(key=lambda item: item["time"])
        if not samples:
            return None
        values = [item["value"] for item in samples]
        return {
            "query_mode": "history",
            "point": {
                "kks": kks,
                "sis_name": str((target_series or {}).get("sis_name", "") or ""),
                "description": str((target_series or {}).get("description", "") or kks),
            },
            "candidates": [],
            "start_time": str(result.get("start_time", "") or iso_text(start_dt)),
            "end_time": str(result.get("end_time", "") or iso_text(end_dt)),
            "interval_seconds": int(result.get("interval_seconds", interval_seconds) or interval_seconds),
            "payload_style": str(result.get("payload_style", "") or ""),
            "unit": str((target_series or {}).get("unit", "") or ""),
            "samples": samples,
            "summary": {
                "sample_count": len(samples),
                "latest_value": values[-1] if values else None,
                "first_value": values[0] if values else None,
                "min_value": min(values) if values else None,
                "max_value": max(values) if values else None,
                "delta_value": (values[-1] - values[0]) if len(values) >= 2 else None,
            },
            "source": str(result.get("source", "") or "linux-5090-worker-01"),
        }

    def _normalize_pressure_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for item in samples if isinstance(samples, list) else []:
            sample_time = parse_datetime(str((item or {}).get("time", "") or ""))
            raw_value = safe_float((item or {}).get("value"))
            if sample_time is None or raw_value is None:
                continue
            output.append({"time": sample_time, "time_text": iso_text(sample_time), "value_mpa": self._pressure_to_mpa(raw_value)})
        output.sort(key=lambda row: row["time"])
        return output

    def _pressure_to_mpa(self, value: float) -> float:
        # SIS pressure may be stored as MPa or kPa. Generator hydrogen pressure is normally around 0.3-0.5MPa.
        if abs(value) > 5.0:
            return value / 1000.0
        return value

    def _temperature_c(self, value: float) -> float:
        if value > 200.0:
            return value - 273.15
        return value

    def _select_stable_24h_window(self, pressure_samples: List[Dict[str, Any]], pressure_jump_kpa: float) -> Dict[str, Any]:
        threshold_mpa = max(0.001, float(pressure_jump_kpa or 30.0) / 1000.0)
        rapid_threshold_kpa = min(float(pressure_jump_kpa or 30.0), max(8.0, float(pressure_jump_kpa or 30.0) * 0.5))
        rapid_threshold_mpa = rapid_threshold_kpa / 1000.0
        rapid_window_seconds = 2 * 3600
        jump_intervals = []
        segment_start = pressure_samples[0]["time"]
        segments: List[Tuple[datetime, datetime, int]] = []
        count = 1
        idx = 0
        while idx < len(pressure_samples) - 1:
            prev = pressure_samples[idx]
            cur = pressure_samples[idx + 1]
            delta_mpa = cur["value_mpa"] - prev["value_mpa"]
            event_start_idx = idx
            event_end_idx = idx + 1
            event_delta_mpa = delta_mpa
            detection_method = ""
            if abs(delta_mpa) > threshold_mpa:
                detection_method = "adjacent_jump"
            else:
                base = prev
                for scan_idx in range(idx + 1, len(pressure_samples)):
                    candidate = pressure_samples[scan_idx]
                    elapsed = (candidate["time"] - base["time"]).total_seconds()
                    if elapsed <= 0:
                        continue
                    if elapsed > rapid_window_seconds:
                        break
                    candidate_delta_mpa = candidate["value_mpa"] - base["value_mpa"]
                    if abs(candidate_delta_mpa) >= rapid_threshold_mpa:
                        event_end_idx = scan_idx
                        event_delta_mpa = candidate_delta_mpa
                        detection_method = "rapid_cumulative_change"
                        event_slice = pressure_samples[idx : event_end_idx + 1]
                        if candidate_delta_mpa > 0:
                            event_start_idx = idx + min(range(len(event_slice)), key=lambda pos: event_slice[pos]["value_mpa"])
                        else:
                            event_start_idx = idx + max(range(len(event_slice)), key=lambda pos: event_slice[pos]["value_mpa"])
                        event_delta_mpa = pressure_samples[event_end_idx]["value_mpa"] - pressure_samples[event_start_idx]["value_mpa"]
                        break
            if detection_method:
                event_start = pressure_samples[event_start_idx]
                event_end = pressure_samples[event_end_idx]
                segment_end = event_start["time"]
                count += max(0, event_start_idx - idx)
                if segment_end > segment_start:
                    segments.append((segment_start, segment_end, count))
                jump_intervals.append(
                    {
                        "from_time": event_start["time_text"],
                        "to_time": event_end["time_text"],
                        "delta_kpa": round(event_delta_mpa * 1000.0, 3),
                        "method": detection_method,
                        "threshold_kpa": round((float(pressure_jump_kpa or 30.0) if detection_method == "adjacent_jump" else rapid_threshold_kpa), 3),
                        "duration_minutes": round((event_end["time"] - event_start["time"]).total_seconds() / 60.0, 3),
                    }
                )
                segment_start = event_end["time"]
                count = 1
                idx = event_end_idx
            else:
                count += 1
                idx += 1
        final_end = pressure_samples[-1]["time"]
        if final_end > segment_start:
            segments.append((segment_start, final_end, count))

        stable_segments = [
            {
                "start_time": iso_text(start),
                "end_time": iso_text(end),
                "duration_hours": round((end - start).total_seconds() / 3600.0, 3),
                "sample_count": sample_count,
            }
            for start, end, sample_count in segments
        ]
        eligible = [(start, end, sample_count) for start, end, sample_count in segments if (end - start).total_seconds() >= 24 * 3600]
        if not eligible:
            longest = max(stable_segments, key=lambda item: item["duration_hours"], default={})
            return {
                "ok": False,
                "message": "未找到连续24小时无明显补氢、排氢或压力阶跃的区间；可扩大lookback_hours或手动指定起止时间。",
                "jump_intervals": jump_intervals[:50],
                "stable_segments": stable_segments,
                "longest_stable_segment": longest,
            }

        chosen_start, chosen_end, _ = sorted(eligible, key=lambda item: item[1], reverse=True)[0]
        window_end = chosen_end
        window_start = window_end - timedelta(hours=24)
        selected_samples = [item for item in pressure_samples if window_start <= item["time"] <= window_end]
        fit = self._linear_fit(selected_samples)
        return {
            "ok": True,
            "start_time": iso_text(window_start),
            "end_time": iso_text(window_end),
            "selection_rule": "默认以调用时刻为结束时间向前查24小时；若24小时内存在单点压力阶跃或2小时内短时累计压力变化，则把补氢/排氢/扰动开始时刻作为新的结束时间，再向前推24小时，直到找到最近的连续24小时稳定区间。",
            "jump_intervals": jump_intervals[:50],
            "jump_count": len(jump_intervals),
            "event_detection": {
                "adjacent_jump_threshold_kpa": round(float(pressure_jump_kpa or 30.0), 3),
                "rapid_cumulative_threshold_kpa": round(rapid_threshold_kpa, 3),
                "rapid_window_minutes": round(rapid_window_seconds / 60.0, 3),
            },
            "stable_segments": stable_segments,
            "pressure_fit": fit,
            "pressure_sample_count": len(selected_samples),
        }

    def _linear_fit(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(samples) < 2:
            return {"ok": False, "message": "not enough samples"}
        t0 = samples[0]["time"]
        xs = [(item["time"] - t0).total_seconds() / 3600.0 for item in samples]
        ys = [item["value_mpa"] for item in samples]
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        denom = sum((x - mean_x) ** 2 for x in xs)
        slope = 0.0 if denom == 0 else sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
        intercept = mean_y - slope * mean_x
        ss_tot = sum((y - mean_y) ** 2 for y in ys)
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
        r2 = 1.0 if ss_tot == 0 else max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
        return {
            "ok": True,
            "slope_mpa_per_hour": slope,
            "slope_kpa_per_hour": slope * 1000.0,
            "r2": r2,
            "first_pressure_mpa": ys[0],
            "last_pressure_mpa": ys[-1],
            "delta_kpa": (ys[-1] - ys[0]) * 1000.0,
        }

    def _downsample_pressure_samples(self, samples: List[Dict[str, Any]], limit: int = 180) -> List[Dict[str, Any]]:
        if len(samples) <= limit:
            chosen = samples
        else:
            chosen = []
            for idx in range(limit):
                source_idx = int(round(idx * (len(samples) - 1) / max(1, limit - 1)))
                chosen.append(samples[source_idx])
        return [{"time": item["time_text"], "value": item["value_mpa"] * 1000.0} for item in chosen]

    def _nearest_pressure_point(self, samples: List[Dict[str, Any]], time_text: str) -> Dict[str, Any]:
        target = parse_datetime(time_text)
        if target is None or not samples:
            return {"time": time_text, "value": None}
        nearest = min(samples, key=lambda item: abs((item["time"] - target).total_seconds()))
        return {"time": nearest["time_text"], "value": nearest["value_mpa"] * 1000.0}

    def _fit_pressure_kpa_series(
        self,
        samples: List[Dict[str, Any]],
        start_time: str,
        end_time: str,
        limit: int = 80,
    ) -> List[Dict[str, Any]]:
        start_dt = parse_datetime(start_time)
        end_dt = parse_datetime(end_time)
        if start_dt is None or end_dt is None:
            return []
        selected = [item for item in samples if start_dt <= item["time"] <= end_dt]
        if len(selected) < 2:
            return []
        base = selected[0]["time"]
        xs = [(item["time"] - base).total_seconds() / 3600.0 for item in selected]
        ys = [item["value_mpa"] * 1000.0 for item in selected]
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        denom = sum((x - mean_x) ** 2 for x in xs)
        if abs(denom) < 1e-12:
            return []
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
        intercept = mean_y - slope * mean_x
        if len(selected) > limit:
            keep = []
            for idx in range(limit):
                source_idx = int(round(idx * (len(selected) - 1) / max(1, limit - 1)))
                keep.append((source_idx, selected[source_idx]))
        else:
            keep = list(enumerate(selected))
        return [
            {
                "time": item["time_text"],
                "value": slope * xs[idx] + intercept,
                "fit_type": "selected_24h_linear",
            }
            for idx, item in keep
        ]

    def _build_pressure_curve(
        self,
        pressure_samples: List[Dict[str, Any]],
        window_result: Dict[str, Any],
        pressure_point: Dict[str, str],
    ) -> Dict[str, Any]:
        if not pressure_samples:
            return {}
        values = [item["value_mpa"] * 1000.0 for item in pressure_samples]
        selected_start = str(window_result.get("start_time", "") or "")
        selected_end = str(window_result.get("end_time", "") or "")
        special_points = []
        if selected_start:
            point = self._nearest_pressure_point(pressure_samples, selected_start)
            special_points.append({**point, "type": "selected_window_start", "detail": "最终采用的漏氢计算开始时间"})
        if selected_end:
            point = self._nearest_pressure_point(pressure_samples, selected_end)
            special_points.append({**point, "type": "selected_window_end", "detail": "最终采用的漏氢计算结束时间"})
        for item in window_result.get("jump_intervals", []) if isinstance(window_result.get("jump_intervals"), list) else []:
            from_point = self._nearest_pressure_point(pressure_samples, str(item.get("from_time", "") or ""))
            to_point = self._nearest_pressure_point(pressure_samples, str(item.get("to_time", "") or ""))
            detail = f"检测到短时氢压变化 {item.get('delta_kpa')} kPa，视为可能补氢/排氢/扰动，结束时间前移避开该区间"
            special_points.append({**from_point, "type": "hydrogen_jump_before", "detail": detail})
            special_points.append({**to_point, "type": "hydrogen_jump_after", "detail": detail})
        fit_series = []
        if window_result.get("ok"):
            fit_series = self._fit_pressure_kpa_series(pressure_samples, selected_start, selected_end)
        return {
            "type": "trend_chart",
            "source": "sis",
            "title": f"{pressure_point.get('label', '发电机氢气压力')}漏氢筛选曲线",
            "unit": "kPa",
            "point": {
                "kks": pressure_point.get("kks", ""),
                "name": pressure_point.get("label", "发电机氢气压力"),
                "unit": "kPa",
            },
            "start_time": pressure_samples[0]["time_text"],
            "end_time": pressure_samples[-1]["time_text"],
            "selected_window": {
                "start_time": selected_start,
                "end_time": selected_end,
                "duration_hours": 24.0 if selected_start and selected_end else None,
                "selection_rule": window_result.get("selection_rule", ""),
            },
            "summary": {
                "sample_count": len(pressure_samples),
                "first_value": values[0],
                "latest_value": values[-1],
                "min_value": min(values),
                "max_value": max(values),
                "delta_value": values[-1] - values[0],
                "jump_count": int(window_result.get("jump_count", len(window_result.get("jump_intervals", []) or [])) or 0),
                "linear_r2": (window_result.get("pressure_fit") or {}).get("r2") if isinstance(window_result.get("pressure_fit"), dict) else None,
                "slope_kpa_per_hour": (window_result.get("pressure_fit") or {}).get("slope_kpa_per_hour") if isinstance(window_result.get("pressure_fit"), dict) else None,
            },
            "series": self._downsample_pressure_samples(pressure_samples),
            "fit_series": fit_series,
            "piecewise_fit_series": fit_series,
            "special_points": special_points[:80],
            "jump_intervals": window_result.get("jump_intervals", []),
            "stable_segments": window_result.get("stable_segments", []),
            "interpretation_hint": "蓝线为搜索窗口内发电机氢气压力真实值；橙线为最终采用24h窗口内的线性拟合；hydrogen_jump_before/after为超过阈值的补氢、排氢或扰动点；selected_window_start/end为最终取值时间。",
        }

    def _fetch_point_histories(
        self,
        points: Dict[str, Dict[str, str]],
        start_dt: datetime,
        end_dt: datetime,
        interval_seconds: int,
    ) -> Dict[str, Dict[str, Any]]:
        histories = {}
        missing_points = []
        available_points = []
        for key, point in points.items():
            try:
                history = self._history(point["kks"], start_dt, end_dt, interval_seconds)
            except Exception as exc:
                missing_points.append(
                    {
                        "key": key,
                        "kks": point["kks"],
                        "label": point["label"],
                        "error": str(exc),
                    }
                )
                continue
            sample_count = len(history.get("samples", [])) if isinstance(history, dict) else 0
            if sample_count <= 0:
                missing_points.append(
                    {
                        "key": key,
                        "kks": point["kks"],
                        "label": point["label"],
                        "error": "history has no samples",
                    }
                )
                continue
            histories[key] = history
            available_points.append({"key": key, "kks": point["kks"], "label": point["label"], "sample_count": sample_count})
        return {"histories": histories, "missing_points": missing_points, "available_points": available_points}

    def _format_missing_points_message(self, unit_no: int, missing_points: List[Dict[str, Any]]) -> str:
        details = "；".join(
            f"{item.get('label', '')} {item.get('kks', '')}: {item.get('error', '')}"
            for item in missing_points
        )
        return (
            f"{unit_no}号机组漏氢计算无法完成：必需测点历史数据不完整。"
            f"缺失测点：{details}。请先在SIS映射表或历史采集配置中补齐这些KKS后再计算。"
        )

    def _nearest_sample(
        self,
        samples: List[Dict[str, Any]],
        target: datetime,
        kind: str,
        tolerance_seconds: int,
    ) -> Dict[str, Any]:
        best = None
        best_seconds = None
        for item in samples if isinstance(samples, list) else []:
            item_time = parse_datetime(str((item or {}).get("time", "") or ""))
            value = safe_float((item or {}).get("value"))
            if item_time is None or value is None:
                continue
            seconds = abs((item_time - target).total_seconds())
            if best_seconds is None or seconds < best_seconds:
                best_seconds = seconds
                best = {"time": item_time, "raw_value": value}
        if best is None:
            raise RuntimeError(f"no sample near {iso_text(target)}")
        if best_seconds is not None and best_seconds > tolerance_seconds:
            raise RuntimeError(f"nearest sample is too far from {iso_text(target)}: {int(best_seconds)}s")
        value = self._pressure_to_mpa(best["raw_value"]) if kind == "pressure" else self._temperature_c(best["raw_value"])
        return {
            "time": iso_text(best["time"]),
            "raw_value": best["raw_value"],
            "value": value,
            "unit": "MPa" if kind == "pressure" else "C",
            "offset_seconds": int(best_seconds or 0),
        }

    def _build_review_points(
        self,
        points: Dict[str, Dict[str, str]],
        point_histories: Dict[str, Dict[str, Any]],
        start_dt: datetime,
        end_dt: datetime,
        interval_seconds: int,
    ) -> Dict[str, Any]:
        tolerance = max(900, int(interval_seconds or 300) * 4)
        review = {"start": {}, "end": {}}
        for key, point in points.items():
            history = point_histories.get(key, {})
            samples = history.get("samples", []) if isinstance(history, dict) else []
            start_sample = self._nearest_sample(samples, start_dt, point["kind"], tolerance)
            end_sample = self._nearest_sample(samples, end_dt, point["kind"], tolerance)
            for bucket, sample in (("start", start_sample), ("end", end_sample)):
                review[bucket][key] = {
                    "kks": point["kks"],
                    "label": point["label"],
                    "time": sample["time"],
                    "raw_value": sample["raw_value"],
                    "value": sample["value"],
                    "unit": sample["unit"],
                    "offset_seconds": sample["offset_seconds"],
                }
        return review

    def _compute_leak(
        self,
        review_points: Dict[str, Any],
        start_dt: datetime,
        end_dt: datetime,
        generator_volume_m3: float,
    ) -> Dict[str, Any]:
        start = review_points["start"]
        end = review_points["end"]
        duration_hours = (end_dt - start_dt).total_seconds() / 3600.0
        if duration_hours <= 0:
            raise ValueError("duration must be positive")
        start_avg_temp_c = (
            (start["hot_h2_gas_turbine_side_temp"]["value"] + start["hot_h2_steam_side_temp"]["value"]) / 2.0
            + (start["cold_h2_steam_side_temp"]["value"] + start["cold_h2_gas_turbine_side_temp"]["value"]) / 2.0
        ) / 2.0
        end_avg_temp_c = (
            (end["hot_h2_gas_turbine_side_temp"]["value"] + end["hot_h2_steam_side_temp"]["value"]) / 2.0
            + (end["cold_h2_steam_side_temp"]["value"] + end["cold_h2_gas_turbine_side_temp"]["value"]) / 2.0
        ) / 2.0
        start_abs_pressure_mpa = start["hydrogen_pressure"]["value"] + start["atmospheric_pressure"]["value"]
        end_abs_pressure_mpa = end["hydrogen_pressure"]["value"] + end["atmospheric_pressure"]["value"]
        start_temp_k = start_avg_temp_c + 273.15
        end_temp_k = end_avg_temp_c + 273.15
        leak_nm3_per_day = (
            (start_abs_pressure_mpa / start_temp_k - end_abs_pressure_mpa / end_temp_k)
            * (20.0 + 273.15)
            * generator_volume_m3
            / 0.101325
            / duration_hours
            * 24.0
        )
        leak_rate_per_day = (
            1.0 - (end_abs_pressure_mpa * start_temp_k / (start_abs_pressure_mpa * end_temp_k))
        ) * 24.0 / duration_hours
        return {
            "duration_hours": duration_hours,
            "start_average_h2_temperature_c": start_avg_temp_c,
            "end_average_h2_temperature_c": end_avg_temp_c,
            "start_absolute_pressure_mpa": start_abs_pressure_mpa,
            "end_absolute_pressure_mpa": end_abs_pressure_mpa,
            "hydrogen_pressure_delta_kpa": (end["hydrogen_pressure"]["value"] - start["hydrogen_pressure"]["value"]) * 1000.0,
            "leak_nm3_per_day": leak_nm3_per_day,
            "leak_rate_per_day": leak_rate_per_day,
            "leak_rate_percent_per_day": leak_rate_per_day * 100.0,
            "window_quality": "pressure_drop" if leak_nm3_per_day >= 0 else "pressure_rise_or_unsuitable_window",
        }

    def _standard_for_rated_pressure(self, rated_pressure_mpa: float) -> Dict[str, Any]:
        pressure = float(rated_pressure_mpa or 0.5)
        for item in HYDROGEN_LEAK_STANDARDS:
            min_value = item.get("min_inclusive")
            max_value = item.get("max_exclusive")
            if min_value is not None and pressure < float(min_value):
                continue
            if max_value is not None and pressure >= float(max_value):
                continue
            return dict(item)
        return dict(HYDROGEN_LEAK_STANDARDS[-1])

    def _assess_leakage(self, computed: Dict[str, Any], rated_pressure_mpa: float) -> Dict[str, Any]:
        standard = self._standard_for_rated_pressure(rated_pressure_mpa)
        leak_value = safe_float(computed.get("leak_nm3_per_day"))
        leak_rate_percent = safe_float(computed.get("leak_rate_percent_per_day"))
        if leak_value is None:
            return {
                "ok": False,
                "qualified": False,
                "grade": "无法评定",
                "reason": "漏氢量计算结果为空。",
                "standard_source": "发电机漏氢计算.xlsx",
                "standard": standard,
            }
        if computed.get("window_quality") != "pressure_drop" or leak_value < 0:
            return {
                "ok": False,
                "qualified": False,
                "grade": "无法评定",
                "reason": "所选窗口压力未自然下降，可能存在补氢、排氢或测点异常，不适合按自然泄漏判断合格性。",
                "leak_nm3_per_day": leak_value,
                "leak_rate_percent_per_day": leak_rate_percent,
                "standard_source": "发电机漏氢计算.xlsx",
                "standard": standard,
            }

        qualified_limit = float(standard["qualified_nm3_per_day"])
        good_limit = float(standard["good_nm3_per_day"])
        excellent_limit = float(standard["excellent_nm3_per_day"])
        if leak_value <= excellent_limit:
            grade = "优"
        elif leak_value <= good_limit:
            grade = "良"
        elif leak_value <= qualified_limit:
            grade = "合格"
        else:
            grade = "不合格"
        qualified = leak_value <= qualified_limit
        margin = qualified_limit - leak_value
        return {
            "ok": True,
            "qualified": qualified,
            "grade": grade,
            "rated_pressure_mpa": float(rated_pressure_mpa or 0.5),
            "rated_pressure_band": standard["band"],
            "leak_nm3_per_day": leak_value,
            "leak_rate_percent_per_day": leak_rate_percent,
            "qualified_limit_nm3_per_day": qualified_limit,
            "good_limit_nm3_per_day": good_limit,
            "excellent_limit_nm3_per_day": excellent_limit,
            "margin_to_qualified_limit_nm3_per_day": margin,
            "standard_source": "发电机漏氢计算.xlsx",
            "standard": standard,
            "plain_explanation": self._plain_assessment_text(leak_value, leak_rate_percent, grade, qualified, standard, margin),
        }

    def _plain_assessment_text(
        self,
        leak_value: float,
        leak_rate_percent: Optional[float],
        grade: str,
        qualified: bool,
        standard: Dict[str, Any],
        margin: float,
    ) -> str:
        rate_text = f"漏氢率约 {leak_rate_percent:.4f}%/d，" if leak_rate_percent is not None else ""
        if qualified:
            return (
                f"本次计算的漏氢量为 {leak_value:.3f} Nm3/d，{rate_text}"
                f"低于{standard['band']}档合格上限 {float(standard['qualified_nm3_per_day']):.3f} Nm3/d，"
                f"还低 {margin:.3f} Nm3/d，评定为{grade}。"
            )
        return (
            f"本次计算的漏氢量为 {leak_value:.3f} Nm3/d，{rate_text}"
            f"高于{standard['band']}档合格上限 {float(standard['qualified_nm3_per_day']):.3f} Nm3/d，"
            f"超出 {abs(margin):.3f} Nm3/d，评定为不合格。"
        )

    def _format_point_line(self, review: Dict[str, Dict[str, Any]]) -> str:
        ordered = [
            "hydrogen_pressure",
            "atmospheric_pressure",
            "hot_h2_steam_side_temp",
            "hot_h2_gas_turbine_side_temp",
            "cold_h2_steam_side_temp",
            "cold_h2_gas_turbine_side_temp",
        ]
        parts = []
        for key in ordered:
            item = review.get(key, {})
            value = item.get("value")
            unit = item.get("unit", "")
            if isinstance(value, (int, float)):
                value_text = f"{value:.5g}"
            else:
                value_text = str(value)
            parts.append(f"{item.get('label', key)} {value_text}{unit}({item.get('kks', '')}, {item.get('time', '')})")
        return "；".join(parts)

    def _review_display_value(self, item: Dict[str, Any], target_unit: str) -> Optional[float]:
        value = safe_float((item or {}).get("value"))
        if value is None:
            return None
        if target_unit == "kPa":
            return value * 1000.0
        return value

    def _build_operator_review_table(self, review_points: Dict[str, Any]) -> List[Dict[str, Any]]:
        start = review_points.get("start", {}) if isinstance(review_points.get("start"), dict) else {}
        end = review_points.get("end", {}) if isinstance(review_points.get("end"), dict) else {}
        rows = [
            ("hot_h2_steam_side_temp", "汽端热氢(℃)", "C"),
            ("hot_h2_gas_turbine_side_temp", "燃机端热氢(℃)", "C"),
            ("cold_h2_steam_side_temp", "汽端冷氢温度(℃)", "C"),
            ("cold_h2_gas_turbine_side_temp", "燃机端冷氢(℃)", "C"),
            ("hydrogen_pressure", "氢气压力(kPa)", "kPa"),
            ("atmospheric_pressure", "大气压力(kPa)", "kPa"),
        ]
        output = []
        for key, label, unit in rows:
            start_item = start.get(key, {}) if isinstance(start.get(key), dict) else {}
            end_item = end.get(key, {}) if isinstance(end.get(key), dict) else {}
            output.append(
                {
                    "field": label,
                    "start": {
                        "value": self._review_display_value(start_item, unit),
                        "unit": unit,
                        "raw_value": start_item.get("raw_value"),
                        "raw_unit": start_item.get("unit"),
                        "time": start_item.get("time", ""),
                        "kks": start_item.get("kks", ""),
                        "label": start_item.get("label", ""),
                    },
                    "end": {
                        "value": self._review_display_value(end_item, unit),
                        "unit": unit,
                        "raw_value": end_item.get("raw_value"),
                        "raw_unit": end_item.get("unit"),
                        "time": end_item.get("time", ""),
                        "kks": end_item.get("kks", ""),
                        "label": end_item.get("label", ""),
                    },
                }
            )
        output.append(
            {
                "field": "时间",
                "start": {"value": (start.get("hydrogen_pressure", {}) or {}).get("time", ""), "unit": "", "kks": ""},
                "end": {"value": (end.get("hydrogen_pressure", {}) or {}).get("time", ""), "unit": "", "kks": ""},
            }
        )
        return output

    def _format_table_value(self, value: Any, unit: str) -> str:
        number = safe_float(value)
        if number is None:
            return str(value or "")
        if unit == "kPa":
            return f"{number:.3f}"
        if unit == "C":
            return f"{number:.3f}"
        return f"{number:.5g}"

    def _format_operator_review_table_text(self, table: List[Dict[str, Any]]) -> str:
        lines = ["复核原始数据：", "项目 | 开始时数据 | 结束时数据"]
        for row in table:
            field = str(row.get("field", "") or "")
            start = row.get("start", {}) if isinstance(row.get("start"), dict) else {}
            end = row.get("end", {}) if isinstance(row.get("end"), dict) else {}
            start_unit = str(start.get("unit", "") or "")
            end_unit = str(end.get("unit", "") or "")
            start_value = self._format_table_value(start.get("value"), start_unit)
            end_value = self._format_table_value(end.get("value"), end_unit)
            if field != "时间":
                start_text = f"{start_value}{start_unit}，KKS {start.get('kks', '')}"
                end_text = f"{end_value}{end_unit}，KKS {end.get('kks', '')}"
            else:
                start_text = start_value
                end_text = end_value
            lines.append(f"{field} | {start_text} | {end_text}")
        return "\n".join(lines)

    def _format_model_reply(
        self,
        unit_no: int,
        computed: Dict[str, Any],
        assessment: Dict[str, Any],
        review_points: Dict[str, Any],
        operator_review_table: List[Dict[str, Any]],
        window_result: Dict[str, Any],
        pressure_jump_kpa: float,
    ) -> str:
        quality_note = ""
        if computed.get("window_quality") != "pressure_drop":
            quality_note = "注意：该窗口末端压力未下降，漏氢量为负或不适合按自然泄漏复核，请确认是否仍有补氢、排氢或测点异常。"
        summary = (
            f"已选取{unit_no}号机组 {window_result.get('start_time')} 至 {window_result.get('end_time')} 的24小时区间计算漏氢量。"
            f"筛选条件为相邻历史点短时压力变化不超过{pressure_jump_kpa:g}kPa，搜索窗口内识别到{int(window_result.get('jump_count', 0) or 0)}段压力阶跃。"
            f"计算结果：日漏氢量 {computed.get('leak_nm3_per_day'):.3f} Nm3/d，漏氢率 {computed.get('leak_rate_percent_per_day'):.4f}%/d，"
            f"氢压变化 {computed.get('hydrogen_pressure_delta_kpa'):.3f} kPa，平均氢温由 {computed.get('start_average_h2_temperature_c'):.2f}C 变为 {computed.get('end_average_h2_temperature_c'):.2f}C。"
            f"合格性评定：{assessment.get('grade', '无法评定')}。"
            f"标准依据：Excel表“氢气泄露量ΔVH标准”，额定氢压{assessment.get('rated_pressure_mpa', 0.5)}MPa采用{assessment.get('rated_pressure_band', '')}档，"
            f"优≤{assessment.get('excellent_limit_nm3_per_day', '')}Nm3/d，良≤{assessment.get('good_limit_nm3_per_day', '')}Nm3/d，合格≤{assessment.get('qualified_limit_nm3_per_day', '')}Nm3/d。"
            f"{assessment.get('plain_explanation', '')}"
            f"{quality_note}"
        )
        return summary + "\n\n" + self._format_operator_review_table_text(operator_review_table)

    def _format_voice_reply(
        self,
        unit_no: int,
        computed: Dict[str, Any],
        assessment: Dict[str, Any],
        window_result: Dict[str, Any],
    ) -> str:
        if not computed or not assessment:
            return f"{unit_no}号机组漏氢计算未返回有效结果。"
        return (
            f"{unit_no}号机组漏氢计算完成。"
            f"日漏氢量 {computed.get('leak_nm3_per_day'):.3f} 标方每天，漏氢率 {computed.get('leak_rate_percent_per_day'):.4f}%每天。"
            f"按额定氢压{assessment.get('rated_pressure_mpa', 0.5)}兆帕、{assessment.get('rated_pressure_band', '')}档标准，"
            f"合格上限为 {assessment.get('qualified_limit_nm3_per_day', '')} 标方每天，本次评定为{assessment.get('grade', '无法评定')}。"
        )


__all__ = ["HydrogenLeakService", "MCP_FEATURE"]
