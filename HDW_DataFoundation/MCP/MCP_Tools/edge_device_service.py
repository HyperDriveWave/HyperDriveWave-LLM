from __future__ import annotations

import importlib.util
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple


MCP_FEATURE = {
    "id": "edge_device_service",
    "mcp_name": "edge_device_service.py",
    "function": "边缘设备查询、功能查询、关机与重启",
    "version": "V0.1",
    "sequence": 50,
    "tools": ["edge_device_list", "edge_device_status", "edge_device_capabilities", "edge_device_power_action"],
}


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NATIVE_MODULE_DIR = os.path.dirname(BASE_DIR)
EDGE_DEVICE_DIR = os.path.join(NATIVE_MODULE_DIR, "Edge_Device")
EDGE_DEVICE_SERVICE_PATH = os.path.join(EDGE_DEVICE_DIR, "edge_device_service.py")

if EDGE_DEVICE_DIR not in sys.path:
    sys.path.insert(0, EDGE_DEVICE_DIR)


def load_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_online_status(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"offline", "off", "false", "0", "离线"} or any(marker in text for marker in ("offline", "离线")):
        return "offline"
    if text in {"online", "on", "true", "1", "在线"} or any(marker in text for marker in ("online", "在线")):
        return "online"
    return ""


def normalize_group(value: Any) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "orin": "orin_nx",
        "orin nx": "orin_nx",
        "orin_nx": "orin_nx",
        "jetson orin nx": "orin_nx",
        "orin nano": "orin_nano",
        "orin_nano": "orin_nano",
        "jetson orin nano": "orin_nano",
        "orange pi": "orange_pi",
        "orange_pi": "orange_pi",
        "香橙派": "orange_pi",
    }
    return aliases.get(text, text if text in {"orin_nx", "orin_nano", "orange_pi"} else "")


def tokenize(text: str) -> List[str]:
    normalized = str(text or "").strip().lower()
    normalized = re.sub(r"[，。、；：:,.!?！？_\-/\\()\[\]{}]+", " ", normalized)
    tokens = re.findall(r"[A-Za-z0-9._:-]{2,}|[\u4e00-\u9fff]{2,}", normalized)
    output: List[str] = []
    seen = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        output.append(token)
    return output


def infer_group(query_text: str) -> str:
    text = str(query_text or "").strip().lower()
    if any(marker in text for marker in ("orin nano", "orin_nano")):
        return "orin_nano"
    if any(marker in text for marker in ("orin nx", "orin_nx", "jetson")):
        return "orin_nx"
    if any(marker in text for marker in ("orange pi", "orange_pi", "香橙派")):
        return "orange_pi"
    return ""


def infer_power_action(query_text: str) -> str:
    text = str(query_text or "").strip().lower()
    if any(marker in text for marker in ("重启", "重新启动", "reboot", "restart")):
        return "reboot"
    if any(marker in text for marker in ("关机", "关闭设备", "shutdown", "power off", "poweroff")):
        return "shutdown"
    return ""


class EdgeDeviceMCPService:
    def __init__(self) -> None:
        self.edge_module = load_module("smartgasturbine_mcp_edge_device_service", EDGE_DEVICE_SERVICE_PATH)

    def looks_like_edge_device_query(self, query_text: str) -> bool:
        text = str(query_text or "").strip().lower()
        return any(
            marker in text
            for marker in (
                "边缘设备",
                "边缘端",
                "设备列表",
                "在线设备",
                "离线设备",
                "jetson",
                "orin",
                "orange pi",
                "香橙派",
                "重启",
                "关机",
                "reboot",
                "shutdown",
            )
        )

    def list_devices(
        self,
        query_text: str = "",
        group: str = "",
        online_status: str = "",
        limit: int = 50,
    ) -> Dict[str, Any]:
        requested_group = normalize_group(group) or infer_group(query_text)
        requested_status = normalize_online_status(online_status) or normalize_online_status(query_text)
        tokens = [token for token in tokenize(query_text) if token not in {"边缘设备", "设备列表", "在线设备", "离线设备", "查询", "查看"}]
        data = self.edge_module.list_devices(requested_group or None)
        devices = data.get("devices", []) if isinstance(data, dict) else []
        matched: List[Dict[str, Any]] = []
        for device in devices if isinstance(devices, list) else []:
            online = bool(device.get("online", False))
            status = "online" if online else "offline"
            if requested_status and status != requested_status:
                continue
            searchable = " ".join(str(device.get(key, "") or "") for key in ("id", "name", "group", "group_name", "ip", "system", "operation_method")).lower()
            if tokens and not any(token in searchable for token in tokens):
                if not any(marker in str(query_text or "").lower() for marker in ("边缘设备", "在线设备", "离线设备", "设备列表")):
                    continue
            item = dict(device)
            item["status"] = status
            item["capabilities"] = self._capabilities(item)
            matched.append(item)
        actual_limit = max(1, min(200, int(limit or 50)))
        online_count = sum(1 for item in matched if item.get("online"))
        return {
            "query_mode": "list",
            "query_text": str(query_text or ""),
            "group": requested_group,
            "online_status": requested_status,
            "count": len(matched[:actual_limit]),
            "total_matched": len(matched),
            "online_count": online_count,
            "offline_count": len(matched) - online_count,
            "devices": matched[:actual_limit],
            "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }

    def resolve_device(self, device_id: str = "", device_name: str = "", query_text: str = "") -> Dict[str, Any]:
        target_id = str(device_id or "").strip()
        target_name = str(device_name or "").strip()
        data = self.edge_module.list_devices(None)
        devices = data.get("devices", []) if isinstance(data, dict) else []
        if target_id:
            for device in devices:
                if str(device.get("id", "") or "") == target_id:
                    item = dict(device)
                    item["status"] = "online" if item.get("online") else "offline"
                    item["capabilities"] = self._capabilities(item)
                    return item
        scored: List[Tuple[int, Dict[str, Any]]] = []
        tokens = tokenize(target_name or query_text)
        for device in devices if isinstance(devices, list) else []:
            score = 0
            did = str(device.get("id", "") or "").lower()
            name = str(device.get("name", "") or "").lower()
            searchable = " ".join(str(device.get(key, "") or "") for key in ("id", "name", "group", "group_name", "ip", "system", "operation_method")).lower()
            if target_name and target_name.lower() == name:
                score += 600
            if target_name and target_name.lower() in name:
                score += 350
            for token in tokens:
                if token == did:
                    score += 500
                elif token in name:
                    score += 180 + min(40, len(token) * 5)
                elif token in searchable:
                    score += 80 + min(30, len(token) * 4)
            if score > 0:
                scored.append((score, device))
        if not scored:
            if len(devices) == 1 and self.looks_like_edge_device_query(query_text or target_name):
                item = dict(devices[0])
                item["status"] = "online" if item.get("online") else "offline"
                item["capabilities"] = self._capabilities(item)
                return item
            raise ValueError("no matching edge device found")
        scored.sort(key=lambda item: (-item[0], str(item[1].get("id", ""))))
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            names = [str(item[1].get("name", "") or item[1].get("id", "")) for item in scored[:5]]
            raise ValueError("edge device target is ambiguous: " + "、".join(names))
        item = dict(scored[0][1])
        item["status"] = "online" if item.get("online") else "offline"
        item["capabilities"] = self._capabilities(item)
        return item

    def device_status(self, device_id: str = "", device_name: str = "", query_text: str = "") -> Dict[str, Any]:
        device = self.resolve_device(device_id=device_id, device_name=device_name, query_text=query_text)
        return {"query_mode": "status", "device": device, "checked_at": datetime.now().astimezone().isoformat(timespec="seconds")}

    def device_capabilities(self, device_id: str = "", device_name: str = "", query_text: str = "") -> Dict[str, Any]:
        device = self.resolve_device(device_id=device_id, device_name=device_name, query_text=query_text)
        return {"query_mode": "capabilities", "device": device, "capabilities": device.get("capabilities", [])}

    def power_action(self, device_id: str = "", device_name: str = "", query_text: str = "", action: str = "") -> Dict[str, Any]:
        action_key = str(action or "").strip().lower() or infer_power_action(query_text)
        if action_key not in {"shutdown", "reboot"}:
            raise ValueError("action must be shutdown or reboot")
        device = self.resolve_device(device_id=device_id, device_name=device_name, query_text=query_text)
        if not device.get("online"):
            return {
                "query_mode": "power_action",
                "accepted": False,
                "reason": "device_offline",
                "message": f"{device.get('name') or device.get('id')} 当前离线，未执行{self._action_label(action_key)}。",
                "device": device,
                "action": action_key,
            }
        result = self.edge_module.power_action(str(device.get("id", "") or ""), action_key)
        return {"query_mode": "power_action", "accepted": bool(result.get("accepted", False)), "device": device, "action": action_key, "result": result}

    def format_reply(self, result: Dict[str, Any]) -> str:
        mode = str(result.get("query_mode", "") or "")
        if mode == "list":
            devices = result.get("devices", []) if isinstance(result.get("devices"), list) else []
            if not devices:
                return "当前没有匹配到边缘设备。"
            names = "、".join(f"{item.get('name') or item.get('id')}（{'在线' if item.get('online') else '离线'}）" for item in devices[:6])
            return f"当前匹配到{len(devices)}台边缘设备，其中在线{result.get('online_count', 0)}台，离线{result.get('offline_count', 0)}台，包含{names}。"
        if mode in {"status", "capabilities"}:
            device = result.get("device", {}) if isinstance(result.get("device"), dict) else {}
            status = "在线" if device.get("online") else "离线"
            caps = "、".join(str(item) for item in device.get("capabilities", [])[:5])
            return f"{device.get('name') or device.get('id')}当前{status}，IP为{device.get('ip', '')}，支持功能包括{caps}。"
        if mode == "power_action":
            action_label = self._action_label(str(result.get("action", "") or ""))
            if not result.get("accepted"):
                return str(result.get("message", "") or f"边缘设备{action_label}命令未执行。")
            device = result.get("device", {}) if isinstance(result.get("device"), dict) else {}
            return f"{device.get('name') or device.get('id')}的{action_label}命令已发送。"
        return ""

    def _capabilities(self, device: Dict[str, Any]) -> List[str]:
        method = str(device.get("operation_method", "") or "").strip().upper()
        capabilities = ["状态查询", "功能查询"]
        if method == "SSH":
            capabilities.extend(["SSH终端", "重启", "关机"])
        return capabilities

    def _action_label(self, action: str) -> str:
        return "重启" if str(action or "").lower() == "reboot" else "关机"
