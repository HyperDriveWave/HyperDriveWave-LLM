from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


MCP_FEATURE = {
    "id": "custom_point_rule_service",
    "mcp_name": "custom_point_rule_service.py",
    "function": "SIS与热成像自定义测点阈值和提示/报警规则配置",
    "version": "V0.1",
    "sequence": 60,
    "tools": ["custom_point_rule_set", "custom_point_rule_list"],
}


BASE_DIR = Path(__file__).resolve().parent
NATIVE_MODULE_DIR = BASE_DIR.parent
MODULE_ACCESS_DIR = NATIVE_MODULE_DIR.parent
ALARM_MANAGER_DIR = NATIVE_MODULE_DIR / "Alarm_Manager"
SIS_CUSTOM_ALARM_PATH = ALARM_MANAGER_DIR / "sis_custom_alarm_service.py"
THERMAL_DIR = MODULE_ACCESS_DIR / "Extension_Module" / "OCR_Thermal"
THERMAL_CONFIG_PATH = THERMAL_DIR / "Thermal_confgi.json"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(ALARM_MANAGER_DIR) not in sys.path:
    sys.path.insert(0, str(ALARM_MANAGER_DIR))

from point_query_service import PointQueryService  # noqa: E402
from thermal_query_service import ThermalQueryService  # noqa: E402


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed


def normalize_level(value: Any) -> str:
    text = str(value or "warning").strip().lower()
    aliases = {
        "提示": "warning",
        "提醒": "warning",
        "预警": "warning",
        "蓝色": "warning",
        "warning": "warning",
        "warn": "warning",
        "info": "warning",
        "报警": "alarm",
        "告警": "alarm",
        "红色": "alarm",
        "alarm": "alarm",
        "严重": "critical",
        "严重报警": "critical",
        "critical": "critical",
    }
    return aliases.get(text, text if text in {"warning", "alarm", "critical"} else "warning")


def normalize_source_type(value: Any, query_text: str = "") -> str:
    text = str(value or "").strip().lower()
    query = str(query_text or "").strip().lower()
    if text in {"sis", "sis_point", "point", "kks", "sis_custom_point"}:
        return "sis"
    if text in {"thermal", "thermal_ocr", "ocr", "热成像", "热像"}:
        return "thermal"
    if any(marker in query for marker in ("热成像", "热像", "红外", "温度测点", "thermal", "ocr")):
        return "thermal"
    return "sis"


def tokenize(text: str) -> List[str]:
    normalized = str(text or "").strip().lower()
    normalized = re.sub(r"[，。、；：:,.!?！？_\-/\\()\[\]{}]+", " ", normalized)
    tokens = re.findall(r"[A-Za-z0-9._:#-]{2,}|[\u4e00-\u9fff]{2,}", normalized)
    output: List[str] = []
    seen = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        output.append(token)
    return output


class CustomPointRuleService:
    def __init__(self) -> None:
        module = load_module("smartgasturbine_mcp_sis_custom_alarm_service", SIS_CUSTOM_ALARM_PATH)
        self.sis_rule_service = module.SISCustomPointAlarmService()
        self.point_query_service = PointQueryService()
        self.thermal_query_service = ThermalQueryService()

    def set_rule(
        self,
        query_text: str = "",
        source_type: str = "",
        point_name: str = "",
        kks: str = "",
        measurement_name: str = "",
        high_limit: Any = None,
        low_limit: Any = None,
        rate_limit: Any = None,
        rate_unit: str = "",
        level: str = "warning",
        enabled: bool = True,
        rate_window_seconds: int = 60,
    ) -> Dict[str, Any]:
        source = normalize_source_type(source_type, query_text=query_text)
        if source == "thermal":
            return self.set_thermal_rule(
                query_text=query_text,
                measurement_name=measurement_name or point_name,
                high_limit=high_limit,
                low_limit=low_limit,
                rate_limit=rate_limit,
                rate_unit=rate_unit,
                level=level,
                enabled=enabled,
            )
        return self.set_sis_rule(
            query_text=query_text,
            point_name=point_name,
            kks=kks,
            high_limit=high_limit,
            low_limit=low_limit,
            rate_limit=rate_limit,
            rate_unit=rate_unit,
            level=level,
            enabled=enabled,
            rate_window_seconds=rate_window_seconds,
        )

    def list_rules(self, source_type: str = "", query_text: str = "", limit: int = 50) -> Dict[str, Any]:
        source = normalize_source_type(source_type, query_text=query_text) if source_type or query_text else ""
        actual_limit = max(1, min(200, int(limit or 50)))
        rules: List[Dict[str, Any]] = []
        if source in {"", "sis"}:
            for item in self.sis_rule_service.list_rules().get("rules", []):
                rule = dict(item)
                rule["source_type"] = "sis"
                rules.append(rule)
        if source in {"", "thermal"}:
            rules.extend(self._thermal_rules())
        filtered = self._filter_rules(rules, query_text)
        return {"query_mode": "list_rules", "query_text": str(query_text or ""), "source_type": source, "count": len(filtered[:actual_limit]), "rules": filtered[:actual_limit], "updated_at": now_iso()}

    def set_sis_rule(
        self,
        query_text: str = "",
        point_name: str = "",
        kks: str = "",
        high_limit: Any = None,
        low_limit: Any = None,
        rate_limit: Any = None,
        rate_unit: str = "",
        level: str = "warning",
        enabled: bool = True,
        rate_window_seconds: int = 60,
    ) -> Dict[str, Any]:
        selected = self._resolve_sis_point(kks=kks, point_name=point_name, query_text=query_text)
        payload = {
            "rule_id": selected.get("kks", ""),
            "enabled": bool(enabled),
            "level": normalize_level(level),
            "kks": selected.get("kks", ""),
            "sis_name": selected.get("sis_name", ""),
            "description": selected.get("description", ""),
            "name": selected.get("description") or selected.get("kks", ""),
            "high_limit": safe_float(high_limit),
            "low_limit": safe_float(low_limit),
            "rate_limit": safe_float(rate_limit),
            "rate_unit": str(rate_unit or "").strip(),
            "rate_window_seconds": max(1, int(rate_window_seconds or 60)),
        }
        result = self.sis_rule_service.save_rule(payload)
        rule = result.get("rule", {})
        return {"query_mode": "set_rule", "source_type": "sis", "point": selected, "rule": rule, "message": self.format_reply({"query_mode": "set_rule", "source_type": "sis", "rule": rule, "point": selected})}

    def set_thermal_rule(
        self,
        query_text: str = "",
        measurement_name: str = "",
        high_limit: Any = None,
        low_limit: Any = None,
        rate_limit: Any = None,
        rate_unit: str = "",
        level: str = "warning",
        enabled: bool = True,
    ) -> Dict[str, Any]:
        selected = self._resolve_thermal_measurement(measurement_name=measurement_name, query_text=query_text)
        target_name = str(selected.get("measurement_name", "") or "").strip()
        stream_key = str(selected.get("stream_key", "") or "").strip()
        cfg = self._read_thermal_config()
        streams = cfg.setdefault("streams", {})
        if not isinstance(streams, dict):
            raise ValueError("thermal config streams is invalid")
        updated = None
        for sid, stream in streams.items():
            if stream_key and str(sid) != stream_key:
                continue
            measurements = stream.get("measurements", []) if isinstance(stream.get("measurements"), list) else []
            candidate_items = [item for item in measurements if isinstance(item, dict)]
            matched_items = [item for item in candidate_items if str(item.get("name", "") or "").strip() == target_name]
            if not matched_items and stream_key and len(candidate_items) == 1:
                matched_items = candidate_items
            if not matched_items:
                target_compact = self._compact(target_name)
                scored = []
                for item in candidate_items:
                    item_name = str(item.get("name", "") or "").strip()
                    item_compact = self._compact(item_name)
                    if not item_compact or not target_compact:
                        continue
                    score = 0
                    if item_compact == target_compact:
                        score += 1000
                    elif item_compact in target_compact or target_compact in item_compact:
                        score += 700
                    else:
                        common = set(item_compact) & set(target_compact)
                        score += len(common) * 10
                    if score > 0:
                        scored.append((score, item))
                scored.sort(key=lambda row: -row[0])
                if scored:
                    matched_items = [scored[0][1]]
            for item in matched_items:
                if not isinstance(item, dict):
                    continue
                alarm = item.get("alarm") if isinstance(item.get("alarm"), dict) else {}
                alarm.update(
                    {
                        "enabled": bool(enabled),
                        "level": normalize_level(level),
                        "high_celsius": safe_float(high_limit),
                        "low_celsius": safe_float(low_limit),
                        "rate_c_per_min": safe_float(rate_limit),
                    }
                )
                item["alarm"] = alarm
                updated = {"stream_id": str(sid), "measurement_name": str(item.get("name", "") or target_name), "requested_name": target_name, "alarm": dict(alarm)}
                break
            if updated:
                break
        if not updated:
            raise ValueError("thermal measurement not found in Thermal_confgi.json")
        self._write_thermal_config(cfg)
        return {"query_mode": "set_rule", "source_type": "thermal", "measurement": selected, "rule": updated, "message": self.format_reply({"query_mode": "set_rule", "source_type": "thermal", "rule": updated, "measurement": selected})}

    def format_reply(self, result: Dict[str, Any]) -> str:
        mode = str(result.get("query_mode", "") or "")
        if mode == "set_rule":
            source = str(result.get("source_type", "") or "")
            rule = result.get("rule", {}) if isinstance(result.get("rule"), dict) else {}
            target = result.get("point", {}) if source == "sis" else result.get("measurement", {})
            target_name = str(target.get("description", "") or target.get("measurement_name", "") or target.get("kks", "") or rule.get("measurement_name", "") or "")
            level_text = "报警" if normalize_level(rule.get("level") or rule.get("alarm", {}).get("level")) == "alarm" else ("严重报警" if normalize_level(rule.get("level") or rule.get("alarm", {}).get("level")) == "critical" else "提示")
            alarm = rule.get("alarm") if isinstance(rule.get("alarm"), dict) else rule
            high = alarm.get("high_limit", alarm.get("high_celsius"))
            low = alarm.get("low_limit", alarm.get("low_celsius"))
            rate = alarm.get("rate_limit", alarm.get("rate_c_per_min"))
            parts = []
            if high is not None:
                parts.append(f"高限{high}")
            if low is not None:
                parts.append(f"低限{low}")
            if rate is not None:
                parts.append(f"变化率限值{rate}每分钟")
            limit_text = "，".join(parts) if parts else "未设置上下限"
            return f"已将{target_name}设置为{level_text}规则，{limit_text}。"
        if mode == "list_rules":
            rules = result.get("rules", []) if isinstance(result.get("rules"), list) else []
            if not rules:
                return "当前没有匹配的自定义测点规则。"
            names = "、".join(str(item.get("name", "") or item.get("description", "") or item.get("measurement_name", "") or item.get("kks", "")) for item in rules[:6])
            return f"当前匹配到{len(rules)}条自定义测点规则，包括{names}。"
        return ""

    def _resolve_sis_point(self, kks: str = "", point_name: str = "", query_text: str = "") -> Dict[str, Any]:
        target_kks = str(kks or "").strip()
        if target_kks:
            return self.point_query_service.resolve_candidate_by_kks(target_kks)
        query = str(point_name or query_text or "").strip()
        candidates = self.point_query_service.search_points(query, limit=5).get("items", [])
        if not candidates:
            raise ValueError("no matching SIS point found")
        if len(candidates) > 1 and int(candidates[0].get("score", 0) or 0) == int(candidates[1].get("score", 0) or 0):
            names = "、".join(str(item.get("description", "") or item.get("kks", "")) for item in candidates[:5])
            raise ValueError("SIS point target is ambiguous: " + names)
        return candidates[0]

    def _resolve_thermal_measurement(self, measurement_name: str = "", query_text: str = "") -> Dict[str, Any]:
        query = str(measurement_name or query_text or "").strip()
        candidates = self.thermal_query_service.search_measurements(query, limit=5).get("items", [])
        if not candidates:
            raise ValueError("no matching thermal measurement found")
        if len(candidates) > 1 and int(candidates[0].get("score", 0) or 0) == int(candidates[1].get("score", 0) or 0):
            names = "、".join(str(item.get("measurement_name", "") or "") for item in candidates[:5])
            raise ValueError("thermal measurement target is ambiguous: " + names)
        return candidates[0]

    def _thermal_rules(self) -> List[Dict[str, Any]]:
        cfg = self._read_thermal_config()
        rows: List[Dict[str, Any]] = []
        streams = cfg.get("streams", {}) if isinstance(cfg.get("streams"), dict) else {}
        for stream_id, stream in streams.items():
            if not isinstance(stream, dict):
                continue
            for item in stream.get("measurements", []) if isinstance(stream.get("measurements"), list) else []:
                if not isinstance(item, dict):
                    continue
                alarm = item.get("alarm") if isinstance(item.get("alarm"), dict) else {}
                rows.append(
                    {
                        "source_type": "thermal",
                        "stream_id": str(stream_id),
                        "stream_name": str(stream.get("display_name", "") or stream_id),
                        "measurement_name": str(item.get("name", "") or ""),
                        "name": str(item.get("name", "") or ""),
                        "enabled": bool(alarm.get("enabled", False)),
                        "level": normalize_level(alarm.get("level", "warning")),
                        "high_limit": safe_float(alarm.get("high_celsius")),
                        "low_limit": safe_float(alarm.get("low_celsius")),
                        "rate_limit": safe_float(alarm.get("rate_c_per_min")),
                    }
                )
        return rows

    def _filter_rules(self, rules: List[Dict[str, Any]], query_text: str) -> List[Dict[str, Any]]:
        tokens = [token for token in tokenize(query_text) if token not in {"查询", "查看", "规则", "自定义", "测点", "热成像", "报警", "提示"}]
        if not tokens:
            return rules
        output = []
        for item in rules:
            searchable = " ".join(str(item.get(key, "") or "") for key in ("kks", "sis_name", "description", "name", "measurement_name", "stream_name")).lower()
            if any(token in searchable for token in tokens):
                output.append(item)
        return output

    def _compact(self, text: str) -> str:
        return re.sub(r"[\s\W_]+", "", str(text or "").strip().lower())

    def _read_thermal_config(self) -> Dict[str, Any]:
        if not THERMAL_CONFIG_PATH.exists():
            raise FileNotFoundError(str(THERMAL_CONFIG_PATH))
        with THERMAL_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}

    def _write_thermal_config(self, data: Dict[str, Any]) -> None:
        tmp = THERMAL_CONFIG_PATH.with_suffix(THERMAL_CONFIG_PATH.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        os.replace(tmp, THERMAL_CONFIG_PATH)
