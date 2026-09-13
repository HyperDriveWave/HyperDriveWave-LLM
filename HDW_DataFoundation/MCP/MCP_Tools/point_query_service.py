import csv
import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple


MCP_FEATURE = {
    "id": "point_query_service",
    "mcp_name": "point_query_service.py",
    "function": "SIS测点在线查询",
    "version": "V0.1",
    "sequence": 10,
    "tools": ["point_query_search_points", "point_query_current_value", "point_query_history_series"],
}


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from runtime import (  # noqa: E402
    SIS_HISTORY_CSV_FILE,
    SIS_HISTORY_DIR,
    SIS_LIVE_DIR,
    SIS_LIVE_LATEST_FILE,
    SIS_MAPPING_FILE,
    SIS_VALUE_FORMAT_PATH,
    sis_runtime_config,
)


MAPPING_DESC_ALIASES = ("测点", "description", "desc", "name")
MAPPING_SIS_ALIASES = ("SIS数据点名", "sis_name", "tagName", "tag_name", "sis")
MAPPING_KKS_ALIASES = ("对应KKS点名", "kks", "KKS", "kks_name")
KKS_PATTERN = re.compile(r"\b\d{2}[A-Z0-9-]{6,}\b", re.IGNORECASE)

_SIS_VALUE_FORMAT_MODULE = None


def get_sis_value_format_module():
    global _SIS_VALUE_FORMAT_MODULE
    if _SIS_VALUE_FORMAT_MODULE is not None:
        return _SIS_VALUE_FORMAT_MODULE
    module = load_module("hdw_mcp_sis_value_format", SIS_VALUE_FORMAT_PATH)
    _SIS_VALUE_FORMAT_MODULE = module
    return module


def apply_value_format_to_current(result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result, dict) or result.get("value") is None:
        return result
    point = result.get("point") if isinstance(result.get("point"), dict) else {}
    kks = str(result.get("kks") or point.get("kks") or "").strip()
    if not kks:
        return result
    try:
        decorated = get_sis_value_format_module().decorate_value(kks, result.get("value"))
    except Exception:
        return result
    formatted = dict(result)
    formatted["raw_value"] = decorated.get("raw_value")
    formatted["value"] = decorated.get("value")
    formatted["value_format"] = decorated.get("value_format", {})
    return formatted


def apply_value_format_to_history(result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return result
    point = result.get("point") if isinstance(result.get("point"), dict) else {}
    kks = str(point.get("kks") or result.get("kks") or "").strip()
    if not kks:
        return result
    formatter = None
    try:
        formatter = get_sis_value_format_module()
    except Exception:
        formatter = None
    formatted = dict(result)
    samples = []
    for sample in result.get("samples", []) or []:
        if not isinstance(sample, dict):
            continue
        sample_copy = dict(sample)
        if formatter is not None and sample_copy.get("value") is not None:
            try:
                decorated = formatter.decorate_value(kks, sample_copy.get("value"))
                sample_copy["raw_value"] = decorated.get("raw_value")
                sample_copy["value"] = decorated.get("value")
                sample_copy["value_format"] = decorated.get("value_format", {})
            except Exception:
                pass
        samples.append(sample_copy)
    formatted["samples"] = samples
    values = [safe_float(item.get("value")) for item in samples if safe_float(item.get("value")) is not None]
    if isinstance(formatted.get("summary"), dict) and values:
        first_value = values[0]
        latest_value = values[-1]
        delta_value = latest_value - first_value
        summary = dict(formatted["summary"])
        summary.update(
            {
                "latest_value": latest_value,
                "first_value": first_value,
                "min_value": min(values),
                "max_value": max(values),
                "delta_value": delta_value,
                "trend": "stable" if abs(delta_value) < 1e-9 else ("up" if delta_value > 0 else "down"),
            }
        )
        formatted["summary"] = summary
    return formatted

STOPWORDS = {
    "帮我",
    "请",
    "查",
    "查询",
    "查一下",
    "查询一下",
    "看一下",
    "看一眼",
    "给我查",
    "测点",
    "当前",
    "当前值",
    "实时",
    "现在",
    "历史",
    "趋势",
    "曲线",
    "变化",
    "波动",
    "值",
    "的",
    "和",
    "一个",
    "看看",
    "帮忙",
}

STOPWORD_PHRASES = (
    "帮我",
    "请帮我",
    "请",
    "查一下",
    "查询一下",
    "看一下",
    "看一眼",
    "给我查",
    "当前值",
    "实时值",
    "历史趋势",
    "历史曲线",
    "历史",
    "趋势",
    "曲线",
    "变化",
    "波动",
)

DOMAIN_KEYWORDS = (
    "燃机",
    "汽机",
    "锅炉",
    "机组",
    "负荷",
    "功率",
    "压力",
    "温度",
    "液位",
    "流量",
    "阀位",
    "电流",
    "电压",
    "频率",
    "真空",
    "入口",
    "出口",
    "排气",
    "振动",
)

POINT_QUERY_NORMALIZATION_RULES = (
    ("燃气符合", "燃机负荷"),
    ("燃机符合", "燃机负荷"),
    ("燃气负荷", "燃机负荷"),
    ("气包", "汽包"),
    ("气机", "汽机"),
    ("符合", "负荷"),
)

POINT_QUERY_TOKEN_EXPANSIONS = (
    ("燃气", "燃机"),
    ("符合", "负荷"),
    ("气包", "汽包"),
    ("气机", "汽机"),
)

UNIT_HINT_PATTERNS = (
    (re.compile(r"(?:^|[^0-9])#?1号(?:机组|燃机|汽机)?"), "01"),
    (re.compile(r"(?:^|[^0-9])#?2号(?:机组|燃机|汽机)?"), "02"),
    (re.compile(r"(?:^|[^0-9])一号(?:机组|燃机|汽机)?"), "01"),
    (re.compile(r"(?:^|[^0-9])二号(?:机组|燃机|汽机)?"), "02"),
    (re.compile(r"(?:^|[^0-9])#?1机"), "01"),
    (re.compile(r"(?:^|[^0-9])#?2机"), "02"),
    (re.compile(r"(?:^|[^0-9])一机"), "01"),
    (re.compile(r"(?:^|[^0-9])二机"), "02"),
)
KNOWN_UNIT_PREFIXES = ("01", "02")
QUERY_TIME_PATTERNS = (
    re.compile(r"(?:最近|近|过去|这)?\s*\d+\s*(?:天|日|小时|时|分钟|分|d|h|m)\s*(?:内|以内|之内)?", re.IGNORECASE),
    re.compile(r"(?:今天|昨日|昨天|前天|本周|本月|最近|近期|当前|实时|现在)"),
)


def deep_copy_json(data: Any) -> Any:
    return json.loads(json.dumps(data, ensure_ascii=False))


def first_row_value(row: Dict[str, str], aliases: Iterable[str], default: str = "") -> str:
    for key in aliases:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return default


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


def normalize_time_text(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        parsed = None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M"):
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


def safe_float(value: Any):
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed


def parse_query_duration(text: str) -> Optional[int]:
    query = str(text or "").strip().lower()
    if not query:
        return None
    match = re.search(r"近\s*(\d+)\s*(分钟|分|小时|时|天|d|h|m)", query)
    if not match:
        return None
    value = int(match.group(1))
    unit = match.group(2)
    if unit in {"分钟", "分", "m"}:
        return max(1, value)
    if unit in {"小时", "时", "h"}:
        return max(1, value * 60)
    if unit in {"天", "d"}:
        return max(1, value * 24 * 60)
    return None


def infer_query_window(text: str) -> Tuple[datetime, datetime]:
    now = datetime.now().astimezone()
    query = str(text or "").strip().lower()
    minutes = parse_query_duration(query)
    if minutes is not None:
        return now - timedelta(minutes=minutes), now
    if "今天" in query:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, now
    if "昨天" in query or "昨日" in query:
        end = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=1)
        return start, end
    return now - timedelta(minutes=30), now


def detect_query_mode(text: str) -> str:
    query = str(text or "").strip()
    for keyword in ("历史", "趋势", "曲线", "变化", "波动", "回看", "回溯"):
        if keyword in query:
            return "history"
    return "current"


def extract_kks_candidate(text: str) -> str:
    match = KKS_PATTERN.search(str(text or ""))
    if not match:
        return ""
    return match.group(0).upper()


def extract_unit_prefix(text: str) -> str:
    query = str(text or "").strip()
    for pattern, prefix in UNIT_HINT_PATTERNS:
        if pattern.search(query):
            return prefix
    return ""


def extract_kks_unit_prefix(kks: str) -> str:
    text = str(kks or "").strip().upper()
    match = re.match(r"^(0[1-2])", text)
    return match.group(1) if match else ""


def is_unit_sensitive_query(text: str) -> bool:
    query = normalize_query_text(text)
    if not query:
        return False
    return any(keyword in query for keyword in ("机组", "燃机", "汽机", "锅炉", "负荷", "功率"))


def strip_trailing_number(text: str) -> str:
    return re.sub(r"\s*[0-9一二三四五六七八九十]+$", "", str(text or "").strip())


def normalize_query_text(text: str) -> str:
    query = str(text or "").strip().lower()
    for source, target in POINT_QUERY_NORMALIZATION_RULES:
        query = query.replace(source.lower(), target.lower())
    query = query.replace("机组的", "机组 ").replace("燃机的", "燃机 ").replace("汽机的", "汽机 ")
    for pattern in QUERY_TIME_PATTERNS:
        query = pattern.sub(" ", query)
    for phrase in STOPWORD_PHRASES:
        query = query.replace(phrase, " ")
    query = re.sub(r"[，。、“”‘’：:；;,.!?？!_\-/\\()\[\]{}]+", " ", query)
    query = re.sub(r"\s+", " ", query).strip()
    return query


def tokenize_query(text: str) -> List[str]:
    query = normalize_query_text(text)
    tokens: List[str] = []
    raw_tokens = re.findall(r"[A-Za-z0-9._:-]{2,}|[\u4e00-\u9fff]{2,}", query)
    for token in raw_tokens:
        normalized = token.strip().lower()
        if not normalized or normalized in STOPWORDS:
            continue
        tokens.append(normalized)
    for keyword in DOMAIN_KEYWORDS:
        if keyword in query and keyword not in tokens:
            tokens.append(keyword)
    for source, target in POINT_QUERY_TOKEN_EXPANSIONS:
        if source in query and target not in tokens:
            tokens.append(target)
    deduped: List[str] = []
    seen = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


@dataclass
class PointRecord:
    kks: str
    sis_name: str
    description: str


class PointQueryService:
    def __init__(self) -> None:
        self._live_module = None
        self._history_module = None
        self._live_collector = None
        self._history_collector = None

    def _build_sis_runtime_config(self) -> Dict[str, Any]:
        return sis_runtime_config()

    def _load_live_module(self):
        if self._live_module is None:
            self._live_module = load_module(
                "hyperdrivewave_mcp_sis_live",
                os.path.join(SIS_LIVE_DIR, "sis_collector.py"),
            )
        return self._live_module

    def _load_history_module(self):
        if self._history_module is None:
            self._history_module = load_module(
                "hyperdrivewave_mcp_sis_history",
                os.path.join(SIS_HISTORY_DIR, "sis_history_collector.py"),
            )
        return self._history_module

    def _get_live_collector(self):
        if self._live_collector is None:
            module = self._load_live_module()
            self._live_collector = module.SISKKSCollector(self._build_sis_runtime_config())
        return self._live_collector

    def _get_history_collector(self):
        if self._history_collector is None:
            module = self._load_history_module()
            self._history_collector = module.SISHistoryCollector(self._build_sis_runtime_config())
        return self._history_collector

    def _ensure_live_login(self) -> None:
        collector = self._get_live_collector()
        if not collector.login(allow_login_page=False, allow_configured_login=True, allow_unauthenticated_fallback=True):
            raise RuntimeError("SIS live login failed")

    def _ensure_history_login(self) -> None:
        collector = self._get_history_collector()
        if not collector.login(allow_login_page=False, allow_configured_login=True, allow_unauthenticated_fallback=True):
            raise RuntimeError("SIS history login failed")

    def load_point_records(self) -> List[PointRecord]:
        if not os.path.exists(SIS_MAPPING_FILE):
            return []
        output: List[PointRecord] = []
        seen = set()
        with open(SIS_MAPPING_FILE, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                kks = first_row_value(row, MAPPING_KKS_ALIASES).strip()
                sis_name = first_row_value(row, MAPPING_SIS_ALIASES).strip()
                description = first_row_value(row, MAPPING_DESC_ALIASES).strip()
                if not kks or not sis_name or kks in seen:
                    continue
                seen.add(kks)
                output.append(PointRecord(kks=kks, sis_name=sis_name, description=description or kks))
        return output

    def looks_like_point_query(self, query_text: str) -> bool:
        normalized = str(query_text or "").strip()
        if not normalized:
            return False
        markers = (
            "测点",
            "kks",
            "当前值",
            "实时值",
            "历史",
            "趋势",
            "曲线",
            "波动",
            "查询",
            "查一下",
            "查",
            "负荷",
            "功率",
            "压力",
            "温度",
            "液位",
            "流量",
        )
        return any(marker.lower() in normalized.lower() for marker in markers) or bool(
            re.search(r"\b\d{2}[A-Z0-9-]{6,}\b", normalized, flags=re.IGNORECASE)
        )

    def get_query_mode(self, query_text: str) -> str:
        return detect_query_mode(query_text)

    def _description_substring_matches(self, query_text: str, points: List[PointRecord], limit: int = 10) -> List[Dict[str, Any]]:
        query_lower = normalize_query_text(query_text)
        if len(query_lower) < 2:
            return []
        matches: List[Tuple[int, Dict[str, Any]]] = []
        for point in points:
            description_lower = point.description.lower()
            if not description_lower:
                continue
            score = 0
            reason = ""
            if query_lower == description_lower:
                score = 2000 + len(description_lower)
                reason = "description_exact_after_noise_cleanup"
            elif query_lower in description_lower:
                score = 1800 + len(query_lower) * 6
                reason = "description_contains_query_after_noise_cleanup"
            elif description_lower in query_lower:
                score = 1700 + len(description_lower) * 6
                reason = "query_contains_description_after_noise_cleanup"
            if score <= 0:
                continue
            matches.append(
                (
                    score,
                    {
                        "kks": point.kks,
                        "sis_name": point.sis_name,
                        "description": point.description,
                        "score": score,
                        "match_reason": reason,
                    },
                )
            )
        matches.sort(key=lambda item: (-item[0], item[1]["kks"]))
        return [item[1] for item in matches[: max(1, limit)]]

    def _search_candidates(self, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        exact_kks = extract_kks_candidate(query_text)
        unit_prefix = extract_unit_prefix(query_text)
        points = self.load_point_records()

        if exact_kks:
            matches = [
                {
                    "kks": point.kks,
                    "sis_name": point.sis_name,
                    "description": point.description,
                    "score": 1000,
                    "match_reason": "kks_exact",
                }
                for point in points
                if point.kks.upper() == exact_kks.upper()
            ]
            if matches:
                return matches[:limit]

        direct_matches = self._description_substring_matches(query_text, points, limit=limit)
        if direct_matches:
            if unit_prefix:
                direct_matches = [
                    item
                    for item in direct_matches
                    if str(item.get("kks", "")).startswith(unit_prefix)
                ]
            if direct_matches:
                return direct_matches[:limit]

        tokens = tokenize_query(query_text)
        query_lower = normalize_query_text(query_text)
        scored: List[Tuple[int, Dict[str, Any]]] = []
        for point in points:
            description_lower = point.description.lower()
            sis_lower = point.sis_name.lower()
            kks_lower = point.kks.lower()
            score = 0
            reasons: List[str] = []

            if unit_prefix:
                if point.kks.startswith(unit_prefix):
                    score += 260
                    reasons.append(f"unit_prefix:{unit_prefix}")
                else:
                    score -= 180

            if query_lower:
                if query_lower == kks_lower:
                    score += 900
                    reasons.append("kks_exact")
                elif query_lower in kks_lower:
                    score += 500
                    reasons.append("kks_contains")
                if query_lower == description_lower:
                    score += 800
                    reasons.append("description_exact")
                elif query_lower in description_lower:
                    score += 450
                    reasons.append("description_contains")

            for token in tokens:
                if token == kks_lower:
                    score += 700
                    reasons.append(f"kks_token:{token}")
                elif token in kks_lower:
                    score += 240
                    reasons.append(f"kks_contains:{token}")

                if token == description_lower:
                    score += 650
                    reasons.append(f"description_token:{token}")
                elif token in description_lower:
                    score += 220 + min(40, len(token) * 4)
                    reasons.append(f"description_contains:{token}")

                if token == sis_lower:
                    score += 400
                    reasons.append(f"sis_token:{token}")
                elif token in sis_lower:
                    score += 160
                    reasons.append(f"sis_contains:{token}")

            if score <= 0:
                continue
            scored.append(
                (
                    score,
                    {
                        "kks": point.kks,
                        "sis_name": point.sis_name,
                        "description": point.description,
                        "score": score,
                        "match_reason": ", ".join(reasons[:6]),
                    },
                )
            )

        scored.sort(key=lambda item: (-item[0], item[1]["kks"]))
        ranked = [item[1] for item in scored]
        if unit_prefix:
            ranked = [
                item
                for item in ranked
                if str(item.get("kks", "")).startswith(unit_prefix)
            ]
        return ranked[: max(1, limit)]

    def search_points(self, query_text: str, limit: int = 10) -> Dict[str, Any]:
        items = self._search_candidates(query_text, limit=limit)
        return {"query_text": str(query_text or "").strip(), "count": len(items), "items": items}

    def build_candidate_shortlist(self, query_text: str, limit: int = 8) -> List[Dict[str, Any]]:
        return self.search_points(query_text=query_text, limit=limit).get("items", [])

    def _resolve_point(self, kks: str = "", point_name: str = "", query_text: str = "") -> Dict[str, Any]:
        target_kks = str(kks or "").strip()
        target_name = str(point_name or "").strip()
        if target_kks:
            matches = self._search_candidates(target_kks, limit=5)
        elif target_name:
            matches = self._search_candidates(target_name, limit=5)
        else:
            matches = self._search_candidates(query_text, limit=5)
        unit_prefix = extract_unit_prefix(query_text)
        if unit_prefix:
            matches = [item for item in matches if str(item.get("kks", "") or "").startswith(unit_prefix)]
        if not matches:
            raise ValueError("no matching point found")
        return {
            "selected": matches[0],
            "candidates": matches,
            "ambiguous": len(matches) > 1 and matches[0]["score"] == matches[1]["score"],
        }

    def resolve_candidate_by_kks(self, kks: str) -> Dict[str, Any]:
        resolved = self._resolve_point(kks=kks)
        return resolved["selected"]

    def _unit_clarification(self, query_text: str, resolved: Dict[str, Any]) -> Dict[str, Any]:
        query = str(query_text or "").strip()
        selected = resolved.get("selected") if isinstance(resolved, dict) else {}
        candidates = resolved.get("candidates") if isinstance(resolved, dict) else []
        if not query or extract_kks_candidate(query) or extract_unit_prefix(query):
            return {}
        if not is_unit_sensitive_query(query):
            return {}
        selected_prefix = extract_kks_unit_prefix(str((selected or {}).get("kks", "") or ""))
        if not selected_prefix:
            return {}

        options = []
        seen = set()
        for item in candidates if isinstance(candidates, list) else []:
            prefix = extract_kks_unit_prefix(str((item or {}).get("kks", "") or ""))
            if not prefix:
                continue
            key = str((item or {}).get("kks", "") or "")
            if key in seen:
                continue
            seen.add(key)
            options.append(
                {
                    "unit_prefix": prefix,
                    "unit_label": f"#{int(prefix)}机组",
                    "kks": key,
                    "description": str((item or {}).get("description", "") or ""),
                    "score": (item or {}).get("score", 0),
                }
            )
        if not options:
            options.append(
                {
                    "unit_prefix": selected_prefix,
                    "unit_label": f"#{int(selected_prefix)}机组",
                    "kks": str((selected or {}).get("kks", "") or ""),
                    "description": str((selected or {}).get("description", "") or ""),
                    "score": (selected or {}).get("score", 0),
                }
            )
        known_text = "、".join(f"#{int(prefix)}机组" for prefix in KNOWN_UNIT_PREFIXES)
        option_text = "、".join(
            f"{item['unit_label']}（{item.get('description') or '候选测点'}，{item.get('kks')}）"
            for item in options[:6]
        )
        return {
            "query_mode": "clarification",
            "requires_clarification": True,
            "clarification_type": "unit_prefix",
            "message": f"请先确认要查询 {known_text} 中的哪一台。当前匹配到的候选：{option_text}。例如可以说：查一下二号燃机负荷。",
            "point": selected,
            "candidates": candidates,
            "options": options[:6],
        }

    def _point_name_clarification(self, query_text: str, resolved: Dict[str, Any]) -> Dict[str, Any]:
        query = normalize_query_text(query_text)
        selected = resolved.get("selected") if isinstance(resolved, dict) else {}
        candidates = resolved.get("candidates") if isinstance(resolved, dict) else []
        if not query or not isinstance(candidates, list) or len(candidates) < 2:
            return {}
        query_base = strip_trailing_number(query)
        options = []
        for item in candidates:
            description = str((item or {}).get("description", "") or "")
            desc_norm = normalize_query_text(description)
            desc_base = strip_trailing_number(desc_norm)
            if desc_base != query_base:
                continue
            options.append(
                {
                    "kks": str((item or {}).get("kks", "") or ""),
                    "description": description,
                    "score": (item or {}).get("score", 0),
                }
            )
        if len(options) < 2:
            return {}
        option_text = "、".join(f"{item['description']}（{item['kks']}）" for item in options[:8])
        return {
            "query_mode": "clarification",
            "requires_clarification": True,
            "clarification_type": "point_name_suffix",
            "message": f"已匹配到多个相近测点，请确认要查询哪一个：{option_text}。例如可以说：查一下{options[0]['description']}趋势。",
            "point": selected,
            "candidates": candidates,
            "options": options[:8],
        }

    def _validate_unit_consistency(self, query_text: str, point: Dict[str, Any]) -> None:
        unit_prefix = extract_unit_prefix(query_text)
        if not unit_prefix:
            return
        point_prefix = extract_kks_unit_prefix(str((point or {}).get("kks", "") or ""))
        if point_prefix and point_prefix != unit_prefix:
            raise ValueError(
                f"query unit prefix {unit_prefix} does not match selected point {point.get('kks', '')}"
            )

    def _load_live_fallback(self, point: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not os.path.exists(SIS_LIVE_LATEST_FILE):
            return None
        target_kks = str(point.get("kks", "") or "").strip()
        with open(SIS_LIVE_LATEST_FILE, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if str(row.get("kks", "") or "").strip() != target_kks:
                    continue
                return {
                    "kks": target_kks,
                    "sis_name": str(row.get("sis_name", "") or "").strip() or str(point.get("sis_name", "") or ""),
                    "description": str(row.get("description", "") or "").strip() or str(point.get("description", "") or ""),
                    "time": normalize_time_text(str(row.get("time", "") or "")),
                    "value": safe_float(row.get("value")),
                    "unit": str(row.get("unit", "") or "").strip(),
                    "quality": str(row.get("quality", "") or "").strip(),
                    "value_source": "latest_csv_fallback",
                }
        return None

    def refresh_live_cache(self) -> Dict[str, Any]:
        collector = self._get_live_collector()
        self._ensure_live_login()
        result = collector.fetch_mapped_kks_data()
        if not result.ok or not result.records:
            return {
                "ok": False,
                "record_count": 0,
                "message": str(result.message or "SIS returned no mapped KKS records"),
            }
        storage_module = load_module("hdw_mcp_sis_live_storage", os.path.join(SIS_LIVE_DIR, "storage.py"))
        runtime_config = self._build_sis_runtime_config()
        storage_cfg = runtime_config.get("storage", {}) if isinstance(runtime_config, dict) else {}
        output_dir = runtime_config.get("storage", {}).get("output_dir", os.path.join(SIS_LIVE_DIR, "data"))
        storage = storage_module.CsvStorage(output_dir, storage_cfg.get("encoding", "utf-8-sig"))
        latest_name = str(storage_cfg.get("latest_file", "latest_kks_data.csv") or "latest_kks_data.csv")
        history_name = str(storage_cfg.get("history_file", "history_kks_data.csv") or "history_kks_data.csv")
        latest_path = storage.write_latest(latest_name, result.records)
        history_path = storage.append_history(history_name, result.records)
        return {
            "ok": True,
            "record_count": len(result.records),
            "message": str(result.message or ""),
            "latest_path": latest_path,
            "history_path": history_path,
        }

    def _load_last_history_value_fallback(self, point: Dict[str, Any], lookback_hours: int = 24) -> Optional[Dict[str, Any]]:
        if not os.path.exists(SIS_HISTORY_CSV_FILE):
            return None
        target_kks = str(point.get("kks", "") or "").strip()
        latest_record: Optional[Dict[str, Any]] = None
        deadline = datetime.now().astimezone() - timedelta(hours=max(1, int(lookback_hours or 24)))
        with open(SIS_HISTORY_CSV_FILE, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if str(row.get("kks", "") or "").strip() != target_kks:
                    continue
                normalized_time = normalize_time_text(str(row.get("time", "") or ""))
                if not normalized_time:
                    continue
                try:
                    sample_time = datetime.fromisoformat(normalized_time)
                except ValueError:
                    continue
                if sample_time < deadline:
                    continue
                value = safe_float(row.get("value"))
                if value is None:
                    continue
                candidate = {
                    "kks": target_kks,
                    "sis_name": str(row.get("sis_name", "") or "").strip() or str(point.get("sis_name", "") or ""),
                    "description": str(row.get("description", "") or "").strip() or str(point.get("description", "") or ""),
                    "time": normalized_time,
                    "value": value,
                    "unit": str(row.get("unit", "") or "").strip(),
                    "quality": str(row.get("quality", "") or "").strip(),
                    "value_source": "history_csv_latest_fallback",
                }
                if latest_record is None or normalized_time > str(latest_record.get("time", "")):
                    latest_record = candidate
        return latest_record

    def _query_recent_history_value(self, point: Dict[str, Any], minutes: int = 60) -> Optional[Dict[str, Any]]:
        try:
            result = self.history_series(
                kks=str(point.get("kks", "") or "").strip(),
                query_text=f"当前值回退 近{max(1, int(minutes or 60))}分钟趋势",
            )
        except Exception:
            return None
        samples = result.get("samples", []) if isinstance(result, dict) else []
        if not isinstance(samples, list) or not samples:
            return None
        last_sample = samples[-1]
        value = safe_float(last_sample.get("value"))
        if value is None:
            return None
        return {
            "kks": str(point.get("kks", "") or "").strip(),
            "sis_name": str(point.get("sis_name", "") or "").strip(),
            "description": str(point.get("description", "") or "").strip(),
            "time": normalize_time_text(str(last_sample.get("time", "") or "")),
            "value": value,
            "unit": str(result.get("unit", "") or ""),
            "quality": "",
            "value_source": "history_recent_query_fallback",
        }

    def current_value(self, kks: str = "", point_name: str = "", query_text: str = "") -> Dict[str, Any]:
        resolved = self._resolve_point(kks=kks, point_name=point_name, query_text=query_text)
        clarification = self._unit_clarification(query_text, resolved)
        if clarification:
            return clarification
        if not str(kks or "").strip():
            clarification = self._point_name_clarification(query_text, resolved)
            if clarification:
                return clarification
        point = resolved["selected"]
        self._validate_unit_consistency(query_text, point)
        collector = self._get_live_collector()
        records: List[Dict[str, Any]] = []

        for _ in range(2):
            try:
                self._ensure_live_login()
                endpoint = collector.sis_config["by_name_endpoint"]
                items = collector._fetch_tag_chunk(endpoint, [point["sis_name"]])
                mapping_points = {
                    item.sis_name: item
                    for item in collector.load_mapping_points()
                    if item.kks == point["kks"] or item.sis_name == point["sis_name"]
                }
                records = collector.normalize_items(items, mapping_points)
            except Exception:
                records = []
            if records:
                break

        if records:
            record = records[0]
            return apply_value_format_to_current({
                "query_mode": "current",
                "point": point,
                "candidates": resolved["candidates"],
                "value": safe_float(record.get("value")),
                "unit": str(record.get("unit", "") or "").strip(),
                "quality": str(record.get("quality", "") or "").strip(),
                "time": normalize_time_text(str(record.get("time", "") or "")),
                "value_source": "sis_live",
            })

        fallback = self._load_live_fallback(point)
        if fallback is not None:
            return apply_value_format_to_current({
                "query_mode": "current",
                "point": point,
                "candidates": resolved["candidates"],
                "value": fallback.get("value"),
                "unit": fallback.get("unit", ""),
                "quality": fallback.get("quality", ""),
                "time": fallback.get("time", ""),
                "value_source": str(fallback.get("value_source", "") or "latest_csv_fallback"),
            })

        recent_history_fallback = self._query_recent_history_value(point, minutes=120)
        if recent_history_fallback is not None:
            return apply_value_format_to_current({
                "query_mode": "current",
                "point": point,
                "candidates": resolved["candidates"],
                "value": recent_history_fallback.get("value"),
                "unit": recent_history_fallback.get("unit", ""),
                "quality": recent_history_fallback.get("quality", ""),
                "time": recent_history_fallback.get("time", ""),
                "value_source": str(recent_history_fallback.get("value_source", "") or "history_recent_query_fallback"),
            })

        history_fallback = self._load_last_history_value_fallback(point)
        if history_fallback is not None:
            return apply_value_format_to_current({
                "query_mode": "current",
                "point": point,
                "candidates": resolved["candidates"],
                "value": history_fallback.get("value"),
                "unit": history_fallback.get("unit", ""),
                "quality": history_fallback.get("quality", ""),
                "time": history_fallback.get("time", ""),
                "value_source": str(history_fallback.get("value_source", "") or "history_csv_latest_fallback"),
            })

        raise RuntimeError(f"SIS live response has no value for: {point['kks']}")

    def _load_history_fallback(self, point: Dict[str, Any], start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        if not os.path.exists(SIS_HISTORY_CSV_FILE):
            return []
        target_kks = str(point.get("kks", "") or "").strip()
        samples: List[Dict[str, Any]] = []
        with open(SIS_HISTORY_CSV_FILE, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if str(row.get("kks", "") or "").strip() != target_kks:
                    continue
                normalized_time = normalize_time_text(str(row.get("time", "") or ""))
                if not normalized_time:
                    continue
                try:
                    sample_time = datetime.fromisoformat(normalized_time)
                except ValueError:
                    continue
                if sample_time < start_time or sample_time > end_time:
                    continue
                value = safe_float(row.get("value"))
                if value is None:
                    continue
                samples.append({"time": normalized_time, "value": value})
        samples.sort(key=lambda item: item["time"])
        return samples

    def history_series(
        self,
        kks: str = "",
        point_name: str = "",
        query_text: str = "",
        start_time: str = "",
        end_time: str = "",
        interval_seconds: int = 0,
    ) -> Dict[str, Any]:
        resolved = self._resolve_point(kks=kks, point_name=point_name, query_text=query_text)
        clarification = self._unit_clarification(query_text, resolved)
        if clarification:
            return clarification
        if not str(kks or "").strip():
            clarification = self._point_name_clarification(query_text, resolved)
            if clarification:
                return clarification
        point = resolved["selected"]
        self._validate_unit_consistency(query_text, point)
        history_module = self._load_history_module()
        if start_time and end_time:
            start_dt = history_module.parse_datetime(start_time)
            end_dt = history_module.parse_datetime(end_time)
        else:
            start_dt, end_dt = infer_query_window(query_text)
        if end_dt <= start_dt:
            raise ValueError("end_time must be later than start_time")

        actual_interval = max(1, int(interval_seconds or 0))
        if not interval_seconds:
            total_seconds = max(1, int((end_dt - start_dt).total_seconds()))
            if total_seconds <= 15 * 60:
                actual_interval = 1
            elif total_seconds <= 60 * 60:
                actual_interval = 2
            elif total_seconds <= 2 * 60 * 60:
                actual_interval = 5
            else:
                actual_interval = 10

        records: List[Dict[str, Any]] = []
        payload_style = "fallback_csv"
        try:
            self._ensure_history_login()
            collector = self._get_history_collector()
            result = collector.fetch_history(
                start_time=start_dt,
                end_time=end_dt,
                interval_seconds=actual_interval,
                payload_style="auto",
                extra_payload=None,
                selected_tags=[point["kks"]],
            )
            payload_style = str(result.payload_style or "auto")
            if result.ok:
                for item in result.records:
                    if str(item.get("kks", "") or "").strip() != point["kks"]:
                        continue
                    value = safe_float(item.get("value"))
                    if value is None:
                        continue
                    records.append(
                        {
                            "time": normalize_time_text(str(item.get("time", "") or "")),
                            "value": value,
                        }
                    )
        except Exception:
            records = []

        if not records:
            records = self._load_history_fallback(point, start_dt, end_dt)
            payload_style = "fallback_csv"
        if not records:
            raise RuntimeError(f"SIS history response has no values for: {point['kks']}")

        records.sort(key=lambda item: item["time"])
        unit = ""
        for item in result.records if 'result' in locals() and getattr(result, 'records', None) else []:
            if str(item.get("kks", "") or "").strip() != point["kks"]:
                continue
            unit = str(item.get("unit", "") or "").strip()
            if unit:
                break
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

        return apply_value_format_to_history({
            "query_mode": "history",
            "point": point,
            "candidates": resolved["candidates"],
            "start_time": start_dt.astimezone().isoformat(timespec="seconds"),
            "end_time": end_dt.astimezone().isoformat(timespec="seconds"),
            "interval_seconds": actual_interval,
            "payload_style": payload_style,
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
        })

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
        actual_mode = str(mode or "").strip().lower() or detect_query_mode(query_text)
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

        mode = detect_query_mode(normalized)
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
                return {
                    "handled": True,
                    "mode": "history",
                    "reply": reply,
                    "tool_result": result,
                    "tool_name": "point_query_history_series",
                }

            result = self.current_value(query_text=normalized)
            point = result["point"]
            unit = str(result.get("unit", "") or "").strip()
            time_text = str(result.get("time", "") or "").strip()
            reply = f"{point['description']}，KKS {point['kks']}，当前值 {result.get('value')}{unit}。"
            if time_text:
                reply += f"数据时间 {time_text}。"
            return {
                "handled": True,
                "mode": "current",
                "reply": reply,
                "tool_result": result,
                "tool_name": "point_query_current_value",
            }
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
