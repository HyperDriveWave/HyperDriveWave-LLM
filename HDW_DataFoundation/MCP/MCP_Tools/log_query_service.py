from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Dict


MCP_FEATURE = {
    "id": "log_query_service",
    "mcp_name": "log_query_service.py",
    "function": "LIEMS日志在线抓取、主值/化学/值长日志查询与重大事件汇总",
    "version": "V0.1",
    "sequence": 80,
    "tools": ["log_query_recent", "log_query_range", "log_query_major_events", "log_query_fetch_range"],
}


MCP_DIR = Path(__file__).resolve().parent
LOG_FETCHING_DIR = Path(os.getenv("HDW_MCP_LOG_ROOT", "/data/mapping/Log_Fetching")).resolve()
LOG_FETCHING_SERVICE_PATH = LOG_FETCHING_DIR / "service.py"


def _load_log_fetching_service_class():
    if not LOG_FETCHING_SERVICE_PATH.exists():
        raise FileNotFoundError(f"Log fetching service not found: {LOG_FETCHING_SERVICE_PATH}")
    spec = importlib.util.spec_from_file_location("hdw_mcp_log_fetching_service", LOG_FETCHING_SERVICE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load Log_Fetching service module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LogFetchingService


class LogQueryService:
    def __init__(self):
        service_cls = _load_log_fetching_service_class()
        self.service = service_cls(LOG_FETCHING_DIR)

    def recent_logs(
        self,
        query_text: str = "",
        days: int = 1,
        major_only: bool = False,
        limit: int = 0,
        # **默认必须是 True**。下层 LogFetchingService.query() 的同名参数默认就是
        # True，这里写成 False 会把它覆盖掉：本地缓存没有当天的 JSON 时，
        # 工具既不抓取、也不报错，直接返回 total_events=0 的"成功"结果。
        # 对模型调用方来说这等于"看着成功、实际什么都没给"，是最难排查的一类失败。
        fetch_if_missing: bool = True,
        force_fetch: bool = False,
    ) -> Dict[str, Any]:
        return self.service.query(
            query_text=query_text,
            last_days=max(1, min(31, int(days or 1))),
            major_only=bool(major_only),
            fetch_if_missing=bool(fetch_if_missing),
            force_fetch=bool(force_fetch),
            limit=max(0, int(limit or 0)),
        )

    def range_logs(
        self,
        query_text: str = "",
        start_date: str = "",
        end_date: str = "",
        major_only: bool = False,
        limit: int = 0,
        fetch_if_missing: bool = True,
        force_fetch: bool = False,
    ) -> Dict[str, Any]:
        return self.service.query(
            query_text=query_text,
            start_date=start_date,
            end_date=end_date,
            major_only=bool(major_only),
            fetch_if_missing=bool(fetch_if_missing),
            force_fetch=bool(force_fetch),
            limit=max(0, int(limit or 0)),
        )

    def major_events(
        self,
        days: int = 7,
        query_text: str = "",
        limit: int = 0,
        fetch_if_missing: bool = True,
        force_fetch: bool = False,
    ) -> Dict[str, Any]:
        return self.service.query(
            query_text=query_text,
            last_days=max(1, min(31, int(days or 7))),
            major_only=True,
            fetch_if_missing=bool(fetch_if_missing),
            force_fetch=bool(force_fetch),
            limit=max(0, int(limit or 0)),
        )

    def fetch_range(self, start_date: str = "", end_date: str = "", last_days: int = 0) -> Dict[str, Any]:
        if last_days:
            return self.service.fetch_range(last_days=max(1, int(last_days)), end_date=end_date)
        return self.service.fetch_range(start_date=start_date, end_date=end_date)
