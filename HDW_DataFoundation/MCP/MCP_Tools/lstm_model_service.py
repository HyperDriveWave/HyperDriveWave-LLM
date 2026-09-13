import csv
import importlib.util
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple


MCP_FEATURE = {
    "id": "lstm_model_service",
    "mcp_name": "lstm_model_service.py",
    "function": "LSTM控制模型查找与目标值调用",
    "version": "V0.1",
    "sequence": 70,
    "tools": ["lstm_model_list", "lstm_model_search", "lstm_model_predict_control"],
}


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LSTM_DIR = os.getenv("HDW_MCP_LSTM_ROOT", "/data/lstm")
LSTM_APP_PATH = os.path.join(LSTM_DIR, "app.py")
LSTM_METADATA_DIR = os.path.join(LSTM_DIR, "data", "model_metadata")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from runtime import SIS_MAPPING_FILE  # noqa: E402

MAPPING_DESC_ALIASES = ("测点", "description", "desc", "name")
MAPPING_SIS_ALIASES = ("SIS数据点名", "sis_name", "tagName", "tag_name", "sis")
MAPPING_KKS_ALIASES = ("对应KKS点名", "kks", "KKS", "kks_name")

KKS_PATTERN = re.compile(r"\b\d{2}[A-Z0-9-]{6,}\b", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?")
TARGET_HINT_PATTERN = re.compile(
    r"(?:目标|设定|给定|控制到|调到|调整到|维持到|保持到|达到|到)\s*([-+]?\d+(?:\.\d+)?)"
)

STOPWORDS = {
    "帮我",
    "请",
    "查",
    "查询",
    "查找",
    "调用",
    "模型",
    "控制",
    "控制模型",
    "一下",
    "当前",
    "实时",
    "现在",
    "建议",
    "输出",
    "执行量",
    "阀门",
    "开度",
    "目标",
    "设定",
    "给定",
    "到",
    "的",
    "和",
    "把",
    "将",
    "需要",
}

NORMALIZATION_RULES = (
    ("气包", "汽包"),
    ("气机", "汽机"),
    ("符合", "负荷"),
    ("燃气", "燃机"),
    ("夜位", "液位"),
    ("水为", "水位"),
)

STAGE_ALIASES = {
    "stable": ("stable", "平稳", "稳定", "稳态", "正常"),
    "startup": ("startup", "start", "启动", "启机", "开机"),
}

UNIT_HINT_PATTERNS = (
    (re.compile(r"(?:^|[^0-9])#?1号(?:机组|燃机|汽机|炉)?"), "01"),
    (re.compile(r"(?:^|[^0-9])#?2号(?:机组|燃机|汽机|炉)?"), "02"),
    (re.compile(r"(?:^|[^0-9])#?3号(?:机组|燃机|汽机|炉)?"), "03"),
    (re.compile(r"(?:^|[^0-9])#?4号(?:机组|燃机|汽机|炉)?"), "04"),
    (re.compile(r"(?:^|[^0-9])一号(?:机组|燃机|汽机|炉)?"), "01"),
    (re.compile(r"(?:^|[^0-9])二号(?:机组|燃机|汽机|炉)?"), "02"),
    (re.compile(r"(?:^|[^0-9])三号(?:机组|燃机|汽机|炉)?"), "03"),
    (re.compile(r"(?:^|[^0-9])四号(?:机组|燃机|汽机|炉)?"), "04"),
    (re.compile(r"(?:^|[^0-9])#?1机"), "01"),
    (re.compile(r"(?:^|[^0-9])#?2机"), "02"),
    (re.compile(r"(?:^|[^0-9])#?3机"), "03"),
    (re.compile(r"(?:^|[^0-9])#?4机"), "04"),
    (re.compile(r"(?:^|[^0-9])一机"), "01"),
    (re.compile(r"(?:^|[^0-9])二机"), "02"),
    (re.compile(r"(?:^|[^0-9])三机"), "03"),
    (re.compile(r"(?:^|[^0-9])四机"), "04"),
)


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


def normalize_text(value: str) -> str:
    text = str(value or "").strip()
    for source, target in NORMALIZATION_RULES:
        text = text.replace(source, target)
    return text


def infer_unit_hint(text: str) -> str:
    normalized = normalize_text(text)
    for pattern, prefix in UNIT_HINT_PATTERNS:
        if pattern.search(normalized):
            return prefix
    return ""


def infer_stage(text: str) -> str:
    normalized = normalize_text(text).lower()
    for stage, aliases in STAGE_ALIASES.items():
        if any(alias.lower() in normalized for alias in aliases):
            return stage
    return ""


def split_query_terms(text: str) -> List[str]:
    normalized = normalize_text(text)
    for source, target in (("#", " "), ("，", " "), ("。", " "), ("、", " "), (":", " "), ("：", " ")):
        normalized = normalized.replace(source, target)
    terms: List[str] = []
    for raw in re.split(r"[\s,/;|]+", normalized):
        token = raw.strip()
        if not token or token in STOPWORDS:
            continue
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", token):
            continue
        terms.append(token)
    compact = re.sub(r"\s+", "", normalized)
    for word in ("水位", "液位", "汽包", "高压", "中压", "低压", "压力", "温度", "负荷", "开度", "阀位", "流量"):
        if word in compact and word not in terms:
            terms.append(word)
    return terms[:16]


def parse_target_value(query_text: str, explicit_value: Any = None) -> Tuple[Optional[float], str]:
    if explicit_value not in (None, ""):
        try:
            return float(explicit_value), "explicit"
        except (TypeError, ValueError):
            return None, "invalid_explicit"
    text = normalize_text(query_text)
    match = TARGET_HINT_PATTERN.search(text)
    if match:
        try:
            return float(match.group(1)), "target_hint"
        except (TypeError, ValueError):
            pass
    numbers = [m.group(0) for m in NUMBER_PATTERN.finditer(text)]
    if len(numbers) == 1:
        try:
            return float(numbers[0]), "single_number"
        except (TypeError, ValueError):
            pass
    return None, "missing"


def parse_time_text(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    return f"{parsed.month}月{parsed.day}日{parsed.hour}点{parsed.minute:02d}分"


def safe_float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def format_number(value: Any) -> str:
    number = safe_float(value)
    if number is None:
        return ""
    return f"{number:.2f}"


class LSTMModelMCPService:
    """MCP adapter for HyperDriveWave LSTM setpoint tracking controller models."""

    def __init__(self) -> None:
        self._lstm_app = None
        self._mapping_cache: Optional[Dict[str, Dict[str, str]]] = None

    def _load_lstm_app(self):
        if self._lstm_app is None:
            self._lstm_app = load_module("hdw_mcp_lstm_app", LSTM_APP_PATH)
        return self._lstm_app

    def _load_mapping(self) -> Dict[str, Dict[str, str]]:
        if self._mapping_cache is not None:
            return self._mapping_cache
        mapping: Dict[str, Dict[str, str]] = {}
        if os.path.exists(SIS_MAPPING_FILE):
            with open(SIS_MAPPING_FILE, "r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    kks = first_row_value(row, MAPPING_KKS_ALIASES).strip()
                    if not kks:
                        continue
                    mapping[kks.upper()] = {
                        "kks": kks,
                        "description": first_row_value(row, MAPPING_DESC_ALIASES).strip(),
                        "sis_name": first_row_value(row, MAPPING_SIS_ALIASES).strip(),
                    }
        self._mapping_cache = mapping
        return mapping

    def _metadata_from_files(self) -> List[Dict[str, Any]]:
        models: List[Dict[str, Any]] = []
        if not os.path.isdir(LSTM_METADATA_DIR):
            return models
        for filename in sorted(os.listdir(LSTM_METADATA_DIR)):
            if not filename.lower().endswith(".json"):
                continue
            path = os.path.join(LSTM_METADATA_DIR, filename)
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
            except Exception:
                continue
            if str(metadata.get("model_type", "") or "") != "setpoint_tracking_controller":
                continue
            metadata["metadata_file"] = metadata.get("metadata_file") or path
            models.append(metadata)
        return models

    def _list_raw_models(self) -> Tuple[List[Dict[str, Any]], str]:
        try:
            app = self._load_lstm_app()
            models = app.list_controller_models()
            return [dict(item) for item in models if isinstance(item, dict)], "lstm_app"
        except Exception:
            return self._metadata_from_files(), "metadata_files"

    def _point_info(self, kks: str) -> Dict[str, str]:
        key = str(kks or "").strip().upper()
        return self._load_mapping().get(key, {"kks": str(kks or ""), "description": "", "sis_name": ""})

    def _stage_from_model(self, model: Dict[str, Any]) -> str:
        values = [
            model.get("stage"),
            model.get("model_name"),
            model.get("model_id"),
            model.get("training_stage"),
        ]
        text = " ".join(str(item or "") for item in values).lower()
        if "startup" in text or "启动" in text:
            return "startup"
        if "stable" in text or "平稳" in text or "稳定" in text:
            return "stable"
        return str(model.get("model_name", "") or "").strip()

    def _enrich_model(self, model: Dict[str, Any], source: str = "") -> Dict[str, Any]:
        process_kks = str(model.get("process_kks") or model.get("control_kks") or "").strip()
        output_kks = str(model.get("output_kks") or model.get("target_kks") or "").strip()
        process_info = self._point_info(process_kks)
        output_info = self._point_info(output_kks)
        enriched = dict(model)
        enriched.update(
            {
                "model_id": str(model.get("model_id", "") or "").strip(),
                "model_name": str(model.get("model_name", "") or "").strip(),
                "stage": self._stage_from_model(model),
                "process_kks": process_kks,
                "control_kks": process_kks,
                "process_name": process_info.get("description", ""),
                "process_sis_name": process_info.get("sis_name", ""),
                "output_kks": output_kks,
                "target_kks": output_kks,
                "output_name": output_info.get("description", ""),
                "output_sis_name": output_info.get("sis_name", ""),
                "metadata_file": str(model.get("metadata_file", "") or ""),
                "onnx_file": str(model.get("onnx_file", "") or ""),
                "source": source,
            }
        )
        return enriched

    def list_models(self, query_text: str = "", limit: int = 50) -> Dict[str, Any]:
        models, source = self._list_raw_models()
        enriched = [self._public_model_fields(self._enrich_model(item, source)) for item in models]
        if query_text:
            return self.search_models(query_text=query_text, limit=limit)
        return {
            "ok": True,
            "query_mode": "lstm_model_list",
            "count": len(enriched[: max(1, int(limit or 50))]),
            "items": enriched[: max(1, int(limit or 50))],
            "source": source,
        }

    def _public_model_fields(self, model: Dict[str, Any]) -> Dict[str, Any]:
        keys = (
            "model_id",
            "model_name",
            "stage",
            "process_kks",
            "process_name",
            "process_sis_name",
            "output_kks",
            "output_name",
            "output_sis_name",
            "manual_lag_seconds",
            "saved_at",
            "metadata_file",
            "onnx_file",
            "score",
            "match_reason",
            "source",
        )
        return {key: model.get(key, "") for key in keys if key in model}

    def _score_model(self, model: Dict[str, Any], query_text: str, stage: str = "") -> Tuple[int, str]:
        query = normalize_text(query_text)
        query_lower = query.lower()
        terms = split_query_terms(query)
        unit_hint = infer_unit_hint(query)
        requested_stage = stage or infer_stage(query)
        searchable = " ".join(
            str(model.get(key, "") or "")
            for key in (
                "model_id",
                "model_name",
                "stage",
                "process_kks",
                "process_name",
                "process_sis_name",
                "output_kks",
                "output_name",
                "output_sis_name",
            )
        )
        searchable_lower = searchable.lower()
        score = 0
        reasons: List[str] = []
        for kks in KKS_PATTERN.findall(query):
            if kks.upper() in searchable.upper():
                score += 80
                reasons.append(f"KKS匹配{kks.upper()}")
        if unit_hint:
            if str(model.get("process_kks", "") or "").startswith(unit_hint) or str(model.get("output_kks", "") or "").startswith(unit_hint):
                score += 35
                reasons.append(f"机组前缀匹配{unit_hint}")
            else:
                score -= 10
        if requested_stage:
            if requested_stage == str(model.get("stage", "") or "").lower():
                score += 30
                reasons.append(f"阶段匹配{requested_stage}")
            else:
                score -= 8
        for term in terms:
            term_lower = term.lower()
            if term_lower and term_lower in searchable_lower:
                score += 14 if len(term) >= 2 else 4
                reasons.append(f"关键词匹配{term}")
        if "水位" in query and "水位" in str(model.get("process_name", "")):
            score += 25
            reasons.append("被控值水位匹配")
        if "压力" in query and "压力" in str(model.get("process_name", "")):
            score += 25
            reasons.append("被控值压力匹配")
        if "温度" in query and "温度" in str(model.get("process_name", "")):
            score += 25
            reasons.append("被控值温度匹配")
        if "控制" in query and model.get("process_kks") and model.get("output_kks"):
            score += 8
            reasons.append("控制模型")
        if not query_lower:
            score = 1
            reasons.append("列出全部模型")
        return score, "；".join(dict.fromkeys(reasons)) or "低置信度匹配"

    def search_models(self, query_text: str = "", control_text: str = "", target_text: str = "", stage: str = "", limit: int = 10) -> Dict[str, Any]:
        query = " ".join(str(item or "") for item in (query_text, control_text, target_text)).strip()
        models, source = self._list_raw_models()
        scored: List[Dict[str, Any]] = []
        for raw in models:
            model = self._enrich_model(raw, source)
            score, reason = self._score_model(model, query, stage=stage)
            model["score"] = score
            model["match_reason"] = reason
            if score > 0 or not query:
                scored.append(model)
        scored.sort(key=lambda item: (int(item.get("score", 0) or 0), str(item.get("saved_at", "") or "")), reverse=True)
        cap = max(1, min(50, int(limit or 10)))
        return {
            "ok": True,
            "query_mode": "lstm_model_search",
            "query_text": query,
            "count": len(scored[:cap]),
            "items": [self._public_model_fields(item) for item in scored[:cap]],
            "source": source,
        }

    def _resolve_model_for_predict(self, model_id: str, query_text: str) -> Dict[str, Any]:
        requested = str(model_id or "").strip()
        models, source = self._list_raw_models()
        if requested:
            for raw in models:
                enriched = self._enrich_model(raw, source)
                if str(enriched.get("model_id", "") or "") == requested:
                    return {"ok": True, "model": enriched, "source": "model_id"}
            return {"ok": False, "error": f"未找到LSTM控制模型：{requested}"}
        search = self.search_models(query_text=query_text, limit=5)
        items = search.get("items", []) if isinstance(search.get("items"), list) else []
        if not items:
            return {"ok": False, "error": "未匹配到可用LSTM控制模型", "candidates": []}
        best = items[0]
        second_score = int(items[1].get("score", 0) or 0) if len(items) > 1 else -1
        best_score = int(best.get("score", 0) or 0)
        if len(items) > 1 and best_score - second_score < 12 and best_score < 70:
            return {"ok": False, "error": "匹配到多个相近LSTM控制模型，需要进一步说明对象或机组", "candidates": items}
        return {"ok": True, "model": best, "source": "semantic_search", "candidates": items}

    def predict_control(
        self,
        model_id: str = "",
        query_text: str = "",
        control_target_value: Any = None,
        target_value: Any = None,
        output_deadband: Any = None,
    ) -> Dict[str, Any]:
        target, target_source = parse_target_value(query_text, control_target_value if control_target_value not in (None, "") else target_value)
        if target is None:
            candidates = self.search_models(query_text=query_text, limit=5).get("items", [])
            return {
                "ok": False,
                "query_mode": "lstm_model_predict_control",
                "needs_target_value": True,
                "error": "调用LSTM控制模型需要目标控制值，例如：把二号高压汽包水位控制到-80。",
                "candidates": candidates,
                "target_value_source": target_source,
            }
        resolved = self._resolve_model_for_predict(model_id, query_text)
        if not resolved.get("ok"):
            result = dict(resolved)
            result.update({"query_mode": "lstm_model_predict_control", "needs_model": True, "control_target_value": target})
            return result
        model = resolved.get("model", {}) if isinstance(resolved.get("model"), dict) else {}
        payload: Dict[str, Any] = {
            "model_id": str(model.get("model_id", "") or model_id),
            "control_target_value": target,
            "use_control_target": True,
        }
        if output_deadband not in (None, ""):
            payload["output_deadband"] = output_deadband
        try:
            prediction = self._load_lstm_app().handle_predict_live(payload)
        except Exception as exc:
            return {
                "ok": False,
                "query_mode": "lstm_model_predict_control",
                "error": str(exc),
                "model": self._public_model_fields(model),
                "control_target_value": target,
                "target_value_source": target_source,
                "candidates": resolved.get("candidates", []),
            }
        return {
            "ok": True,
            "query_mode": "lstm_model_predict_control",
            "model": self._public_model_fields(model),
            "control_target_value": target,
            "target_value_source": target_source,
            "prediction": prediction,
            "candidates": resolved.get("candidates", []),
        }

    def format_reply(self, result: Dict[str, Any]) -> str:
        if not isinstance(result, dict):
            return ""
        mode = str(result.get("query_mode", "") or "")
        if mode == "lstm_model_list":
            items = result.get("items", []) if isinstance(result.get("items"), list) else []
            if not items:
                return "当前没有可用的LSTM控制模型。"
            labels = []
            for item in items[:5]:
                name = str(item.get("process_name", "") or item.get("process_kks", "") or item.get("model_name", "") or item.get("model_id", ""))
                output_name = str(item.get("output_name", "") or item.get("output_kks", ""))
                stage = str(item.get("stage", "") or "")
                detail = name
                if output_name:
                    detail += f"，执行量为{output_name}"
                if stage:
                    detail += f"，阶段为{stage}"
                labels.append(detail)
            return f"当前共有{len(items)}个可用的LSTM控制模型，包括" + "；".join(labels) + "。"
        if mode == "lstm_model_search":
            items = result.get("items", []) if isinstance(result.get("items"), list) else []
            if not items:
                return "当前没有匹配到可用的LSTM控制模型。"
            labels = []
            for item in items[:5]:
                name = str(item.get("process_name", "") or item.get("process_kks", "") or item.get("model_id", ""))
                output_name = str(item.get("output_name", "") or item.get("output_kks", ""))
                stage = str(item.get("stage", "") or "")
                labels.append(f"{name}，执行量为{output_name}，阶段为{stage}")
            return "我找到的LSTM控制模型包括：" + "；".join(labels) + "。"
        if mode != "lstm_model_predict_control":
            return ""
        if not result.get("ok"):
            if result.get("needs_target_value"):
                return "调用LSTM控制模型还需要目标控制值，请说明要把被控值控制到多少。"
            return f"LSTM控制模型调用失败：{result.get('error', '未知错误')}"
        model = result.get("model", {}) if isinstance(result.get("model"), dict) else {}
        prediction = result.get("prediction", {}) if isinstance(result.get("prediction"), dict) else {}
        process_name = str(model.get("process_name", "") or prediction.get("process_kks", "") or "被控值")
        output_name = str(model.get("output_name", "") or prediction.get("output_kks", "") or "执行量")
        current_control = format_number(prediction.get("process_live_value"))
        target_control = format_number(prediction.get("control_target_value"))
        current_output = format_number(prediction.get("current_output_value"))
        target_output = format_number(prediction.get("target_value"))
        direction = str(prediction.get("control_direction", "") or "")
        direction_text = {"increase": "增大", "decrease": "减小", "hold": "保持"}.get(direction, direction or "调整")
        time_text = ""
        live_values = prediction.get("live_values", {}) if isinstance(prediction.get("live_values"), dict) else {}
        process_kks = str(prediction.get("process_kks", "") or "")
        if isinstance(live_values.get(process_kks), dict):
            time_text = parse_time_text(str(live_values.get(process_kks, {}).get("time", "") or ""))
        parts = [
            f"已匹配{process_name}控制模型",
            f"当前{process_name}为{current_control}" if current_control else "",
            f"目标为{target_control}" if target_control else "",
            f"当前{output_name}为{current_output}" if current_output else "",
            f"建议{output_name}{direction_text}到{target_output}" if target_output else "",
            f"数据时间为{time_text}" if time_text else "",
        ]
        return "，".join(item for item in parts if item) + "。"
