from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


MAPPING_ROOT = Path(os.getenv("HDW_MCP_MAPPING_ROOT", "/data/mapping")).resolve()
SIS_MAPPING_FILE = MAPPING_ROOT / "mapping.csv"
SIS_LIVE_DIR = MAPPING_ROOT / "SIS_live"
SIS_HISTORY_DIR = MAPPING_ROOT / "SIS_history"
SIS_LIVE_LATEST_FILE = SIS_LIVE_DIR / "data" / "latest_kks_data.csv"
SIS_HISTORY_CSV_FILE = SIS_HISTORY_DIR / "data" / "history_series_kks_data.csv"
SIS_VALUE_FORMAT_PATH = MAPPING_ROOT / "sis_value_format.py"
SIS_CONFIG_PATH = Path(os.getenv("HDW_MCP_SIS_CONFIG_PATH", "/data/mapping/.sis-runtime.json"))


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def load_json_file(path: Path, default: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def sis_runtime_config() -> dict[str, Any]:
    return {
        "sis": {
            "base_url": _env("HDW_SIS_BASE_URL"),
            "login_url": _env("HDW_SIS_LOGIN_URL", f"{_env('HDW_SIS_BASE_URL')}/login.html"),
            "username": _env("HDW_SIS_USERNAME"),
            "password": _env("HDW_SIS_PASSWORD"),
            "language": _env("HDW_SIS_LANGUAGE", "zh-Hans"),
            "connect_timeout_seconds": int(_env("HDW_SIS_CONNECT_TIMEOUT", "10")),
            "read_timeout_seconds": int(_env("HDW_SIS_READ_TIMEOUT", "30")),
            "mapping_file": str(SIS_MAPPING_FILE),
            "request_chunk_size": int(_env("HDW_SIS_REQUEST_CHUNK_SIZE", "500")),
            "history_request_chunk_size": int(_env("HDW_SIS_HISTORY_REQUEST_CHUNK_SIZE", "200")),
            "by_name_endpoint": _env(
                "HDW_SIS_BY_NAME_ENDPOINT",
                "/luculent-liems-sis/api/services/realdb/TagInfo/GetTagInfosWithValueByNameList",
            ),
            "history_endpoint": _env(
                "HDW_SIS_HISTORY_ENDPOINT",
                "/luculent-liems-sis/api/services/realdb/TagHisData/GetSeriesValuesByNameList",
            ),
            "login_endpoints": [
                "/api/Account/Login",
                "/api/auth/login",
                "/api/user/login",
                "/TokenAuth/Authenticate",
                "/api/authenticate",
                "/login",
            ],
        },
        "storage": {
            "output_dir": str(SIS_LIVE_DIR / "data"),
            "latest_file": "latest_kks_data.csv",
            "history_file": "history_kks_data.csv",
            "encoding": "utf-8-sig",
        },
        "history": {
            "endpoint": _env(
                "HDW_SIS_HISTORY_ENDPOINT",
                "/luculent-liems-sis/api/services/realdb/TagHisData/GetSeriesValuesByNameList",
            ),
            "request_chunk_size": int(_env("HDW_SIS_HISTORY_REQUEST_CHUNK_SIZE", "200")),
            "output_file": str(SIS_HISTORY_CSV_FILE),
            "default_interval_seconds": int(_env("HDW_SIS_HISTORY_INTERVAL", "6")),
        },
        "_config_path": str(SIS_CONFIG_PATH),
    }


def rtsp_streams() -> list[dict[str, Any]]:
    raw = _env("HDW_RTSP_STREAMS_JSON")
    if not raw:
        return []
    value = load_json_file(Path(raw), []) if raw.startswith("/") else None
    if value is None:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if isinstance(value, dict):
        value = value.get("streams", [])
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def capability_status(feature: str, reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "feature": feature,
        "message": reason,
    }
