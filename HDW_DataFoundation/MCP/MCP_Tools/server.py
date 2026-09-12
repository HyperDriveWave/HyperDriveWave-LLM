import argparse
import math
import os
import re
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_SDK_SRC = os.path.join(BASE_DIR, "python-sdk", "src")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if PYTHON_SDK_SRC not in sys.path:
    sys.path.insert(0, PYTHON_SDK_SRC)

from mcp.server import MCPServer  # noqa: E402

from point_query_service import PointQueryService  # noqa: E402
from thermal_query_service import ThermalQueryService  # noqa: E402
from rtsp_query_service import RTSPQueryService  # noqa: E402
from alarm_query_service import AlarmQueryService  # noqa: E402
from edge_device_service import EdgeDeviceMCPService  # noqa: E402
from custom_point_rule_service import CustomPointRuleService  # noqa: E402
from lstm_model_service import LSTMModelMCPService  # noqa: E402
from log_query_service import LogQueryService  # noqa: E402
from hydrogen_leak_service import HydrogenLeakService  # noqa: E402
from unavailable import UnavailableService  # noqa: E402


def _construct(service_cls, feature: str):
    try:
        return service_cls()
    except Exception as exc:
        return UnavailableService(feature, f"依赖未配置：{type(exc).__name__}: {exc}")


SERVICE = _construct(PointQueryService, "SIS测点查询")
THERMAL_SERVICE = _construct(ThermalQueryService, "热成像查询")
RTSP_SERVICE = _construct(RTSPQueryService, "RTSP查询")
ALARM_SERVICE = _construct(AlarmQueryService, "报警查询")
EDGE_DEVICE_SERVICE = _construct(EdgeDeviceMCPService, "边缘设备")
CUSTOM_POINT_RULE_SERVICE = _construct(CustomPointRuleService, "自定义测点规则")
LSTM_MODEL_SERVICE = _construct(LSTMModelMCPService, "LSTM控制模型")
LOG_QUERY_SERVICE = _construct(LogQueryService, "LIEMS日志")
HYDROGEN_LEAK_SERVICE = _construct(HydrogenLeakService, "发电机漏氢计算")
mcp = MCPServer(
    name="HyperDriveWave Industrial MCP",
    instructions=(
        "This server exposes HyperDriveWave SIS point query tools, thermal OCR query tools, RTSP stream status tools, alarm query tools, edge device tools, custom point rule tools, LSTM control model tools, LIEMS log query tools, and generator hydrogen leakage calculation tools. "
        "Use search first when the target is ambiguous, then use current or history tools. "
        "Thermal query tools should be used for OCR-derived custom temperature points. "
        "RTSP tools should be used to check whether a stream is online or offline."
        "Alarm tools should be used to query active alarms, warnings, prompts, and alarm history."
        "Edge device tools should be used to list edge devices, check online devices and capabilities, and perform explicit single-device shutdown or reboot commands."
        "Custom point rule tools should be used to set SIS or thermal measurement high/low/rate thresholds and warning/alarm levels."
        "LSTM model tools should be used to search setpoint tracking controller models and call a matched model with a target control value."
        "Log query tools should be used to fetch and summarize LIEMS duty notes, important handovers, grounding records, and major events. "
        "Hydrogen leak tools should be used to calculate #1/#2 generator hydrogen leakage from SIS pressure and temperature history. "
        "rag_query_plan should be used before RAG retrieval to split complex multi-question requests into focused rounds. "
        "Use log_query_fetch_range only when fresh LIEMS fetching is explicitly needed; normal log query tools first read local exported logs."
    ),
)


@mcp.tool()
def mcp_service_status() -> dict:
    """Return the runtime availability of migrated MCP feature groups."""
    services = {
        "sis_point": SERVICE,
        "thermal": THERMAL_SERVICE,
        "rtsp": RTSP_SERVICE,
        "alarm": ALARM_SERVICE,
        "edge_device": EDGE_DEVICE_SERVICE,
        "custom_rule": CUSTOM_POINT_RULE_SERVICE,
        "lstm": LSTM_MODEL_SERVICE,
        "liems_log": LOG_QUERY_SERVICE,
        "hydrogen_leak": HYDROGEN_LEAK_SERVICE,
        "rag_planner": True,
    }
    return {
        "server": "HyperDriveWave Industrial MCP",
        "transport": "stdio_or_streamable_http",
        "services": {
            name: {
                "available": not isinstance(service, UnavailableService),
                "message": service.reason if isinstance(service, UnavailableService) else "ready",
            }
            for name, service in services.items()
        },
    }


@mcp.tool()
def rag_query_plan(
    question: str,
    inference_mode: str = "offline",
    base_top_k: int = 10,
) -> dict:
    """Plan one or more focused RAG rounds for a complex industrial question."""
    text = " ".join(str(question or "").split())
    if not text:
        return {
            "rounds": 1,
            "queries": [""],
            "top_k": max(5, int(base_top_k or 10)),
            "base_top_k": max(5, int(base_top_k or 10)),
            "reason": "empty question",
        }

    numbered = [
        item.strip(" \t\r\n;；")
        for item in re.split(
            r"(?:^|[\n;；])\s*(?=(?:\d{1,2}|[一二三四五六七八九十]+)[、.)．）])",
            str(question),
        )
        if item.strip(" \t\r\n;；")
    ]
    complex_markers = (
        "试卷",
        "题目",
        "逐题",
        "每题",
        "分别",
        "全部",
        "多道",
        "多项",
        "清单",
        "逐条",
        "完整分析",
        "详细说明",
    )
    if len(numbered) >= 2:
        queries = numbered[:8]
        reason = "检测到编号或分号分隔的多个子问题"
    elif any(marker in text for marker in complex_markers):
        queries = [
            text,
            f"{text} 相关定义、范围和判断依据",
            f"{text} 操作步骤、参数和适用条件",
            f"{text} 异常处理、限制和安全要求",
        ]
        reason = "检测到多问题或完整分析意图，拆分为主题检索"
    else:
        queries = [text]
        reason = "单一问题，使用一次检索"

    rounds = max(1, min(8, len(queries)))
    queries = queries[:rounds]
    configured = max(1, min(40, int(base_top_k or 10)))
    per_round = max(5, math.ceil(configured / rounds))
    return {
        "rounds": rounds,
        "queries": queries,
        "top_k": per_round,
        "base_top_k": configured,
        "inference_mode": inference_mode if inference_mode in {"online", "offline"} else "offline",
        "reason": reason,
        "minimum_top_k": 5,
    }
@mcp.tool()
def point_query_search_points(query_text: str, limit: int = 10) -> dict:
    """Search KKS points by KKS code, SIS tag name, or Chinese point description."""
    return SERVICE.search_points(query_text=query_text, limit=max(1, min(20, int(limit or 10))))


@mcp.tool()
def point_query_current_value(kks: str = "", point_name: str = "", query_text: str = "") -> dict:
    """Query the current value for one SIS point."""
    return SERVICE.current_value(kks=kks, point_name=point_name, query_text=query_text)


@mcp.tool()
def point_query_history_series(
    kks: str = "",
    point_name: str = "",
    query_text: str = "",
    start_time: str = "",
    end_time: str = "",
    interval_seconds: int = 0,
) -> dict:
    """Query one SIS point history series and return structured samples plus summary."""
    return SERVICE.history_series(
        kks=kks,
        point_name=point_name,
        query_text=query_text,
        start_time=start_time,
        end_time=end_time,
        interval_seconds=interval_seconds,
    )


@mcp.tool()
def thermal_query_search_measurements(query_text: str, limit: int = 10) -> dict:
    """Search thermal OCR measurement points by name or query text."""
    return THERMAL_SERVICE.search_measurements(query_text=query_text, limit=max(1, min(20, int(limit or 10))))


@mcp.tool()
def thermal_query_current_temperature(measurement_name: str = "", query_text: str = "") -> dict:
    """Query the latest value for a thermal OCR measurement point."""
    return THERMAL_SERVICE.current_value(measurement_name=measurement_name, query_text=query_text)


@mcp.tool()
def thermal_query_history_series(
    measurement_name: str = "",
    query_text: str = "",
    start_time: str = "",
    end_time: str = "",
    interval_seconds: int = 0,
) -> dict:
    """Query a thermal OCR measurement point history series and return samples plus summary."""
    return THERMAL_SERVICE.history_series(
        measurement_name=measurement_name,
        query_text=query_text,
        start_time=start_time,
        end_time=end_time,
        interval_seconds=interval_seconds,
    )


@mcp.tool()
def rtsp_query_list_streams(query_text: str = "", stream_type: str = "", online_status: str = "", limit: int = 50) -> dict:
    """List RTSP streams and filter them by type or online status."""
    return RTSP_SERVICE.list_streams(
        query_text=query_text,
        stream_type=stream_type,
        online_status=online_status,
        limit=max(1, min(100, int(limit or 50))),
    )


@mcp.tool()
def rtsp_query_stream_status(stream_id: str = "", stream_name: str = "", query_text: str = "") -> dict:
    """Resolve one RTSP stream and return its online/offline status."""
    return RTSP_SERVICE.stream_status(stream_id=stream_id, stream_name=stream_name, query_text=query_text)


@mcp.tool()
def alarm_query_active(query_text: str = "", level: str = "", source: str = "", limit: int = 50) -> dict:
    """Query current active alarms or warning prompts."""
    return ALARM_SERVICE.active_alarms(
        query_text=query_text,
        level=level,
        source=source,
        limit=max(1, min(500, int(limit or 50))),
    )


@mcp.tool()
def alarm_query_history(query_text: str = "", level: str = "", source: str = "", limit: int = 100) -> dict:
    """Query historical alarms or warning prompts."""
    return ALARM_SERVICE.history_alarms(
        query_text=query_text,
        level=level,
        source=source,
        limit=max(1, min(1000, int(limit or 100))),
    )


@mcp.tool()
def alarm_acknowledge(query_text: str = "", alarm_id: str = "", dedup_minutes: int = 15, user: str = "mcp") -> dict:
    """Acknowledge one active alarm. Default dedup is 15 minutes and max suppression is 1 hour."""
    return ALARM_SERVICE.acknowledge_alarm(
        query_text=query_text,
        alarm_id=alarm_id,
        dedup_minutes=dedup_minutes,
        user=user,
    )


@mcp.tool()
def edge_device_list(query_text: str = "", group: str = "", online_status: str = "", limit: int = 50) -> dict:
    """List edge devices and optionally filter by group or online/offline status."""
    return EDGE_DEVICE_SERVICE.list_devices(
        query_text=query_text,
        group=group,
        online_status=online_status,
        limit=max(1, min(200, int(limit or 50))),
    )


@mcp.tool()
def edge_device_status(device_id: str = "", device_name: str = "", query_text: str = "") -> dict:
    """Resolve one edge device and return its online/offline status."""
    return EDGE_DEVICE_SERVICE.device_status(device_id=device_id, device_name=device_name, query_text=query_text)


@mcp.tool()
def edge_device_capabilities(device_id: str = "", device_name: str = "", query_text: str = "") -> dict:
    """Resolve one edge device and return its supported functions."""
    return EDGE_DEVICE_SERVICE.device_capabilities(device_id=device_id, device_name=device_name, query_text=query_text)


@mcp.tool()
def edge_device_power_action(device_id: str = "", device_name: str = "", query_text: str = "", action: str = "") -> dict:
    """Shutdown or reboot exactly one resolved edge device. Action must be shutdown or reboot."""
    return EDGE_DEVICE_SERVICE.power_action(device_id=device_id, device_name=device_name, query_text=query_text, action=action)


@mcp.tool()
def custom_point_rule_set(
    query_text: str = "",
    source_type: str = "",
    point_name: str = "",
    kks: str = "",
    measurement_name: str = "",
    high_limit: float | None = None,
    low_limit: float | None = None,
    rate_limit: float | None = None,
    rate_unit: str = "",
    level: str = "warning",
    enabled: bool = True,
    rate_window_seconds: int = 60,
) -> dict:
    """Set high/low/rate custom rule for a SIS point or thermal OCR measurement."""
    return CUSTOM_POINT_RULE_SERVICE.set_rule(
        query_text=query_text,
        source_type=source_type,
        point_name=point_name,
        kks=kks,
        measurement_name=measurement_name,
        high_limit=high_limit,
        low_limit=low_limit,
        rate_limit=rate_limit,
        rate_unit=rate_unit,
        level=level,
        enabled=enabled,
        rate_window_seconds=rate_window_seconds,
    )


@mcp.tool()
def custom_point_rule_list(source_type: str = "", query_text: str = "", limit: int = 50) -> dict:
    """List custom SIS point and thermal OCR measurement rules."""
    return CUSTOM_POINT_RULE_SERVICE.list_rules(source_type=source_type, query_text=query_text, limit=max(1, min(200, int(limit or 50))))


@mcp.tool()
def lstm_model_list(query_text: str = "", limit: int = 50) -> dict:
    """List available LSTM setpoint tracking controller models."""
    return LSTM_MODEL_SERVICE.list_models(query_text=query_text, limit=max(1, min(50, int(limit or 50))))


@mcp.tool()
def lstm_model_search(query_text: str = "", control_text: str = "", target_text: str = "", stage: str = "", limit: int = 10) -> dict:
    """Search LSTM control models by natural language, KKS, process object, unit number, or stage."""
    return LSTM_MODEL_SERVICE.search_models(
        query_text=query_text,
        control_text=control_text,
        target_text=target_text,
        stage=stage,
        limit=max(1, min(50, int(limit or 10))),
    )


@mcp.tool()
def lstm_model_predict_control(
    model_id: str = "",
    query_text: str = "",
    control_target_value: float | None = None,
    target_value: float | None = None,
    output_deadband: float | None = None,
) -> dict:
    """Call an LSTM setpoint tracking controller model with a target control value."""
    return LSTM_MODEL_SERVICE.predict_control(
        model_id=model_id,
        query_text=query_text,
        control_target_value=control_target_value,
        target_value=target_value,
        output_deadband=output_deadband,
    )


@mcp.tool()
def log_query_recent(query_text: str = "", days: int = 1, major_only: bool = False, limit: int = 80, fetch_if_missing: bool = False) -> dict:
    """Query recent LIEMS logs and summarize what happened."""
    return LOG_QUERY_SERVICE.recent_logs(
        query_text=query_text,
        days=max(1, min(30, int(days or 1))),
        major_only=major_only,
        limit=max(1, min(500, int(limit or 80))),
        fetch_if_missing=fetch_if_missing,
    )


@mcp.tool()
def log_query_range(
    query_text: str = "",
    start_date: str = "",
    end_date: str = "",
    major_only: bool = False,
    limit: int = 120,
    fetch_if_missing: bool = False,
) -> dict:
    """Query LIEMS logs in a date range and return structured events plus summary."""
    return LOG_QUERY_SERVICE.range_logs(
        query_text=query_text,
        start_date=start_date,
        end_date=end_date,
        major_only=major_only,
        limit=max(1, min(800, int(limit or 120))),
        fetch_if_missing=fetch_if_missing,
    )


@mcp.tool()
def log_query_major_events(days: int = 7, query_text: str = "", limit: int = 120, fetch_if_missing: bool = False) -> dict:
    """Query major LIEMS events in the recent N days."""
    return LOG_QUERY_SERVICE.major_events(
        days=max(1, min(60, int(days or 7))),
        query_text=query_text,
        limit=max(1, min(500, int(limit or 120))),
        fetch_if_missing=fetch_if_missing,
    )


@mcp.tool()
def log_query_fetch_range(start_date: str = "", end_date: str = "", last_days: int = 0) -> dict:
    """Fetch LIEMS logs from the logged-in CDP browser and store them under Log_Fetching/DATA."""
    return LOG_QUERY_SERVICE.fetch_range(start_date=start_date, end_date=end_date, last_days=max(0, int(last_days or 0)))


@mcp.tool()
def hydrogen_leak_calculate(
    unit: int = 1,
    start_time: str = "",
    end_time: str = "",
    lookback_hours: int = 72,
    generator_volume_m3: float = 125.0,
    rated_pressure_mpa: float = 0.5,
    pressure_jump_kpa: float = 30.0,
    interval_seconds: int = 300,
) -> dict:
    """Calculate #1/#2 generator hydrogen leakage from SIS pressure/temperature history and return review points."""
    return HYDROGEN_LEAK_SERVICE.calculate(
        unit=unit,
        start_time=start_time,
        end_time=end_time,
        lookback_hours=lookback_hours,
        generator_volume_m3=generator_volume_m3,
        rated_pressure_mpa=rated_pressure_mpa,
        pressure_jump_kpa=pressure_jump_kpa,
        interval_seconds=interval_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SmartGasTurbine MCP server.")
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="MCP transport.",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("HDW_MCP_BIND", "127.0.0.1"),
        help="Host for streamable-http transport.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("HDW_MCP_PORT", "8766")),
        help="Port for streamable-http transport.",
    )
    parser.add_argument("--json-response", action="store_true", help="Enable JSON HTTP response mode.")
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
        return
    mcp.run(
        transport="streamable-http",
        host=args.host,
        port=args.port,
        json_response=bool(args.json_response),
    )


if __name__ == "__main__":
    main()
