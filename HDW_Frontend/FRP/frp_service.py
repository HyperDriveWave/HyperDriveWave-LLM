import json
import os
import platform
import re
import signal
import shutil
import ssl
import subprocess
import sys
import tarfile
import time
from typing import Any, Dict, List, Tuple
from urllib.request import Request, urlopen

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))
COMMON_DIR = os.path.join(PROJECT_ROOT, "Module_Access", "Common")
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

from config_store import get_config_value, set_config_value

FRP_ROOT = BASE_DIR
FRP_CONF_DIR = os.environ.get("SMARTGASTURBINE_FRP_CONF_DIR") or os.path.join(FRP_ROOT, "conf")
FRP_LEGACY_CONF_DIR = os.path.join(FRP_ROOT, "conf")
FRP_BIN_DIR = os.path.join(FRP_ROOT, "bin")


def _current_system_key() -> str:
    return "windows" if platform.system().strip().lower() == "windows" else "linux"

DEFAULT_SETTINGS = {
    "active_config": "frpc_8765_tcp.toml",
    "active_role": "frpc",
    "active_system": _current_system_key(),
    "notes": "主服务器可选 Linux 部署 frps，备用服务器或本机可选 Windows/Linux 部署 frpc。",
}

RUNTIME_DIR = os.environ.get("SMARTGASTURBINE_FRP_RUNTIME_DIR") or os.path.join(FRP_ROOT, "runtime")
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PUBLIC_PROBE_TIMEOUT_SECONDS = float(os.environ.get("SMARTGASTURBINE_FRP_PUBLIC_PROBE_TIMEOUT", "3") or "3")


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _rel(path: str) -> str:
    abs_path = os.path.abspath(path)
    project_root = os.path.abspath(PROJECT_ROOT)
    try:
        common = os.path.commonpath([abs_path, project_root])
    except ValueError:
        common = ""
    if common == project_root:
        return os.path.relpath(abs_path, project_root).replace("\\", "/")
    return abs_path.replace("\\", "/")


def _command_path(path: str, system_name: str) -> str:
    abs_path = os.path.abspath(path)
    project_root = os.path.abspath(PROJECT_ROOT)
    try:
        common = os.path.commonpath([abs_path, project_root])
    except ValueError:
        common = ""
    if common == project_root:
        rendered = f"./{os.path.relpath(abs_path, project_root).replace(os.sep, '/')}"
    else:
        rendered = abs_path.replace("\\", "/")
    if str(system_name or "").strip().lower() == "windows":
        return rendered.replace("/", "\\")
    return rendered


def _deep_copy(data: Any) -> Any:
    return json.loads(json.dumps(data, ensure_ascii=False))


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text or "")


def _ensure_config_dir() -> None:
    os.makedirs(FRP_CONF_DIR, exist_ok=True)
    try:
        has_toml = any(name.lower().endswith(".toml") for name in os.listdir(FRP_CONF_DIR))
    except OSError:
        has_toml = False
    if has_toml or os.path.abspath(FRP_CONF_DIR) == os.path.abspath(FRP_LEGACY_CONF_DIR):
        return
    if not os.path.isdir(FRP_LEGACY_CONF_DIR):
        return
    for name in sorted(os.listdir(FRP_LEGACY_CONF_DIR)):
        if not name.lower().endswith(".toml"):
            continue
        source = os.path.join(FRP_LEGACY_CONF_DIR, name)
        target = os.path.join(FRP_CONF_DIR, name)
        if os.path.isfile(source) and not os.path.exists(target):
            shutil.copy2(source, target)


def _load_settings() -> Dict[str, Any]:
    settings = get_config_value(["frp"], None)
    if not isinstance(settings, dict):
        set_config_value(["frp"], DEFAULT_SETTINGS)
        return _deep_copy(DEFAULT_SETTINGS)
    merged = _deep_copy(DEFAULT_SETTINGS)
    merged.update({k: v for k, v in settings.items() if v is not None})
    current_system = _current_system_key()
    configured_system = str(merged.get("active_system", "") or "").strip().lower()
    if configured_system not in {"linux", "windows"}:
        merged["active_system"] = current_system
    elif configured_system != current_system:
        pid_file = _pid_path(configured_system, str(merged.get("active_role", "frpc") or "frpc").strip().lower())
        if not _process_running(_read_pid(pid_file)):
            merged["active_system"] = current_system
    return merged


def _save_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    settings = _load_settings()
    for key in DEFAULT_SETTINGS:
        if key in payload and payload[key] is not None:
            settings[key] = str(payload[key])
    set_config_value(["frp"], settings)
    return settings


def _ensure_runtime_dir() -> None:
    os.makedirs(RUNTIME_DIR, exist_ok=True)


def _runtime_path(name: str) -> str:
    _ensure_runtime_dir()
    return os.path.join(RUNTIME_DIR, name)


def _pid_path(system_name: str, role: str) -> str:
    return _runtime_path(f"{system_name}_{role}.pid")


def _log_path(system_name: str, role: str) -> str:
    return _runtime_path(f"{system_name}_{role}.log")


def _read_pid(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return int(handle.read().strip() or "0")
    except (OSError, ValueError):
        return 0


def _write_pid(path: str, pid: int) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(str(int(pid)))


def _remove_pid(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _terminate_pid(pid: int) -> None:
    if pid <= 0:
        return
    try:
        if platform.system().strip().lower() == "windows":
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, text=True, timeout=10, check=False)
        else:
            os.kill(pid, signal.SIGTERM)
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if not _process_running(pid):
                    return
                time.sleep(0.1)
            os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def _active_binary_path(system_name: str, role: str) -> str:
    binaries = _scan_binaries()
    system_key = "linux" if str(system_name or "").strip().lower() == "linux" else "windows"
    role_key = "frps" if str(role or "").strip().lower() == "frps" else "frpc"
    configured = binaries.get(system_key, {}).get(f"{role_key}_path", "")
    if configured:
        return os.path.join(PROJECT_ROOT, configured.replace("/", os.sep))
    if system_key == "windows":
      return os.path.join(FRP_BIN_DIR, "frp_0.62.1_windows_amd64", "frp_0.62.1_windows_amd64", f"{role_key}.exe")
    return os.path.join(FRP_BIN_DIR, "frp_0.62.1_linux_amd64", role_key)


def _ensure_linux_binaries() -> None:
    linux_dir = os.path.join(FRP_BIN_DIR, "frp_0.62.1_linux_amd64")
    linux_frpc = os.path.join(linux_dir, "frpc")
    linux_frps = os.path.join(linux_dir, "frps")
    archive = os.path.join(FRP_BIN_DIR, "frp_0.62.1_linux_amd64.tar.gz")
    if (not os.path.exists(linux_frpc) or not os.path.exists(linux_frps)) and os.path.exists(archive):
        with tarfile.open(archive, "r:gz") as handle:
            for member in handle.getmembers():
                target = os.path.abspath(os.path.join(FRP_BIN_DIR, member.name))
                if not target.startswith(os.path.abspath(FRP_BIN_DIR) + os.sep):
                    raise ValueError(f"unsafe FRP archive member: {member.name}")
            handle.extractall(FRP_BIN_DIR)
    for binary in (linux_frpc, linux_frps):
        if os.path.exists(binary):
            _ensure_executable(binary)


def _ensure_executable(path: str) -> None:
    if not os.path.exists(path):
        return
    if os.access(path, os.X_OK):
        return
    try:
        os.chmod(path, os.stat(path).st_mode | 0o755)
    except PermissionError:
        if not os.access(path, os.X_OK):
            raise


def _prepare_binary(system_name: str, role: str) -> str:
    system_key = "linux" if str(system_name or "").strip().lower() == "linux" else "windows"
    if system_key != _current_system_key():
        raise RuntimeError(f"FRP {system_key} binary cannot be started from {_current_system_key()} runtime")
    if system_key == "linux":
        _ensure_linux_binaries()
    binary_path = _active_binary_path(system_key, role)
    if not os.path.exists(binary_path):
        raise ValueError(f"FRP binary not found: {binary_path}")
    if system_key == "linux":
        _ensure_executable(binary_path)
    return binary_path


def _spawn_process(binary_path: str, config_path: str, log_path: str) -> int:
    _ensure_parent(log_path)
    with open(log_path, "a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            [binary_path, "-c", config_path],
            cwd=PROJECT_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system().strip().lower() == "windows" else 0,
        )
    return int(process.pid)


def _public_probe_candidates(settings: Dict[str, Any]) -> List[Dict[str, str]]:
    config_name = str(settings.get("active_config") or "").strip()
    if not config_name:
        return []
    try:
        path = _config_path(config_name)
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            root, proxies = _parse_simple_toml(handle.read())
    except Exception:
        return []
    server_addr = str(root.get("serverAddr") or root.get("bindAddr") or "").strip()
    if not server_addr:
        return []
    candidates: List[Dict[str, str]] = []
    seen: set[str] = set()
    for proxy in proxies:
        remote_port = str(proxy.get("remotePort") or "").strip()
        if not remote_port:
            continue
        local_port = str(proxy.get("localPort") or "").strip()
        name = str(proxy.get("name") or "").strip()
        scheme = "https" if local_port == "443" or remote_port in {"443", "8443"} or "https" in name.lower() else "http"
        suffix = "/app/" if scheme == "https" or remote_port in {"8443", "8080"} else "/"
        url = f"{scheme}://{server_addr}:{remote_port}{suffix}"
        if url in seen:
            continue
        seen.add(url)
        candidates.append({"url": url, "proxy": name, "remote_port": remote_port, "scheme": scheme})
    return candidates


def _probe_public_access(settings: Dict[str, Any]) -> Dict[str, Any]:
    candidates = _public_probe_candidates(settings)
    errors: List[str] = []
    context = ssl._create_unverified_context()
    for item in candidates[:5]:
        url = item.get("url", "")
        if not url:
            continue
        try:
            request = Request(url, headers={"User-Agent": "SmartGasTurbine-FRP-Probe/1.0"})
            with urlopen(request, timeout=PUBLIC_PROBE_TIMEOUT_SECONDS, context=context if url.startswith("https://") else None) as response:
                status_code = int(getattr(response, "status", 0) or response.getcode() or 0)
                reachable = 100 <= status_code < 500
                return {
                    "reachable": reachable,
                    "url": url,
                    "status_code": status_code,
                    "proxy": item.get("proxy", ""),
                    "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "message": "外网入口可达" if reachable else f"HTTP {status_code}",
                    "errors": errors,
                }
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    return {
        "reachable": False,
        "url": candidates[0].get("url", "") if candidates else "",
        "status_code": 0,
        "proxy": candidates[0].get("proxy", "") if candidates else "",
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "message": errors[-1] if errors else "未找到可探测的公网入口",
        "errors": errors,
    }


def process_status() -> Dict[str, Any]:
    settings = _load_settings()
    role = str(settings.get("active_role", "frpc") or "frpc").strip().lower()
    system_name = str(settings.get("active_system", _current_system_key()) or _current_system_key()).strip().lower()
    pid_file = _pid_path(system_name, role)
    pid = _read_pid(pid_file)
    running = _process_running(pid)
    if pid and not running:
        _remove_pid(pid_file)
    public_probe = _probe_public_access(settings)
    effective_running = bool(running or public_probe.get("reachable"))
    return {
        "running": running,
        "effective_running": effective_running,
        "external_running": bool(public_probe.get("reachable") and not running),
        "public_probe": public_probe,
        "pid": pid if running else 0,
        "pid_file": _rel(pid_file),
        "log_file": _rel(_log_path(system_name, role)),
        "role": role,
        "system": system_name,
    }


def start_process(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = payload or {}
    settings = _load_settings()
    role = str(payload.get("role") or settings.get("active_role") or "frpc").strip().lower()
    system_name = str(payload.get("system") or settings.get("active_system") or _current_system_key()).strip().lower()
    config_name = str(payload.get("config_name") or settings.get("active_config") or "").strip()
    if not config_name:
        raise ValueError("active config is required")
    pid_file = _pid_path(system_name, role)
    current_pid = _read_pid(pid_file)
    if current_pid and _process_running(current_pid):
        return {
            "started": False,
            "already_running": True,
            **process_status(),
        }
    binary_path = _prepare_binary(system_name, role)
    config_path = _config_path(config_name)
    log_path = _log_path(system_name, role)
    pid = _spawn_process(binary_path, config_path, log_path)
    _write_pid(pid_file, pid)
    settings_update = {"active_role": role, "active_system": system_name, "active_config": config_name}
    _save_settings(settings_update)
    time.sleep(0.8)
    return {
        "started": True,
        "already_running": False,
        **process_status(),
    }


def stop_process(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = payload or {}
    settings = _load_settings()
    role = str(payload.get("role") or settings.get("active_role") or "frpc").strip().lower()
    system_name = str(payload.get("system") or settings.get("active_system") or _current_system_key()).strip().lower()
    pid_file = _pid_path(system_name, role)
    pid = _read_pid(pid_file)
    if pid:
        _terminate_pid(pid)
        time.sleep(0.4)
    _remove_pid(pid_file)
    return {"stopped": True, **process_status()}


def read_runtime_log(lines: int = 120) -> Dict[str, Any]:
    status = process_status()
    log_path = os.path.join(PROJECT_ROOT, status.get("log_file", "").replace("/", os.sep)) if status.get("log_file") else ""
    content = ""
    if log_path and os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
            text = _strip_ansi(handle.read())
        parts = text.splitlines()
        content = "\n".join(parts[-max(1, int(lines or 120)):])
    return {"status": status, "content": content}


def _scan_configs() -> List[Dict[str, Any]]:
    _ensure_config_dir()
    if not os.path.isdir(FRP_CONF_DIR):
        return []
    items: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(FRP_CONF_DIR)):
        path = os.path.join(FRP_CONF_DIR, name)
        if not os.path.isfile(path) or not name.lower().endswith(".toml"):
            continue
        role = "frps" if name.lower().startswith("frps") else "frpc"
        items.append(
            {
                "name": name,
                "role": role,
                "path": _rel(path),
                "size_bytes": os.path.getsize(path),
            }
        )
    return items


def _scan_binaries() -> Dict[str, Dict[str, Any]]:
    if _current_system_key() == "linux":
        try:
            _ensure_linux_binaries()
        except Exception:
            pass
    windows_dir = os.path.join(FRP_BIN_DIR, "frp_0.62.1_windows_amd64", "frp_0.62.1_windows_amd64")
    windows_frpc = os.path.join(windows_dir, "frpc.exe")
    windows_frps = os.path.join(windows_dir, "frps.exe")
    linux_archive = os.path.join(FRP_BIN_DIR, "frp_0.62.1_linux_amd64.tar.gz")
    linux_dir = os.path.join(FRP_BIN_DIR, "frp_0.62.1_linux_amd64")
    linux_frpc = os.path.join(linux_dir, "frpc")
    linux_frps = os.path.join(linux_dir, "frps")
    return {
        "windows": {
            "label": "Windows",
            "frpc_path": _rel(windows_frpc) if os.path.exists(windows_frpc) else "",
            "frps_path": _rel(windows_frps) if os.path.exists(windows_frps) else "",
            "available": os.path.exists(windows_frpc) or os.path.exists(windows_frps),
        },
        "linux": {
            "label": "Linux",
            "frpc_path": _rel(linux_frpc) if os.path.exists(linux_frpc) else "",
            "frps_path": _rel(linux_frps) if os.path.exists(linux_frps) else "",
            "archive_path": _rel(linux_archive) if os.path.exists(linux_archive) else "",
            "available": os.path.exists(linux_frpc) or os.path.exists(linux_frps) or os.path.exists(linux_archive),
        },
    }


def _parse_simple_toml(text: str) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    root: Dict[str, str] = {}
    proxies: List[Dict[str, str]] = []
    current_proxy: Dict[str, str] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "[[proxies]]":
            current_proxy = {}
            proxies.append(current_proxy)
            continue
        if "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        if current_proxy is not None:
            current_proxy[key] = value
        else:
            root[key] = value
    return root, proxies


def _render_toml(root: Dict[str, str], proxies: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for key, value in root.items():
        if value.isdigit():
            lines.append(f"{key} = {value}")
        else:
            lines.append(f'{key} = "{value}"')
    if proxies:
        lines.append("")
    for index, proxy in enumerate(proxies):
        lines.append("[[proxies]]")
        for key, value in proxy.items():
            if value.isdigit():
                lines.append(f"{key} = {value}")
            else:
                lines.append(f'{key} = "{value}"')
        if index != len(proxies) - 1:
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def _config_path(config_name: str) -> str:
    _ensure_config_dir()
    safe_name = os.path.basename(str(config_name or "").strip())
    if not safe_name:
        raise ValueError("config name is required")
    path = os.path.abspath(os.path.join(FRP_CONF_DIR, safe_name))
    if not path.startswith(os.path.abspath(FRP_CONF_DIR)):
        raise ValueError("invalid config path")
    return path


def _build_command(system_name: str, role: str, config_name: str) -> Dict[str, str]:
    system_key = "linux" if str(system_name or "").strip().lower() == "linux" else "windows"
    role_key = "frps" if str(role or "").strip().lower() == "frps" else "frpc"
    config_path = _config_path(config_name)
    binaries = _scan_binaries()
    if system_key == "windows":
        binary_rel = binaries["windows"][f"{role_key}_path"] or f"Module_Access/Extension_Module/FRP/bin/frp_0.62.1_windows_amd64/frp_0.62.1_windows_amd64/{role_key}.exe"
        binary_win = binary_rel.replace("/", "\\")
        config_win = _command_path(config_path, "windows")
        command = f".\\{binary_win} -c {config_win}"
        shell = "PowerShell / CMD"
    else:
        binary_rel = binaries["linux"][f"{role_key}_path"] or f"Module_Access/Extension_Module/FRP/bin/frp_0.62.1_linux_amd64/{role_key}"
        extract = ""
        archive_rel = binaries["linux"].get("archive_path", "")
        if archive_rel and not binaries["linux"][f"{role_key}_path"]:
            extract = f"tar -xzf ./{archive_rel} -C ./Module_Access/Extension_Module/FRP/bin"
        config_arg = _command_path(config_path, "linux")
        command = f"./{binary_rel} -c {config_arg}"
        shell = "Bash / SSH"
        if extract:
            command = f"{extract}\n{command}"
    return {"system": system_key, "role": role_key, "shell": shell, "command": command}


def get_overview() -> Dict[str, Any]:
    settings = _load_settings()
    configs = _scan_configs()
    binaries = _scan_binaries()
    commands = {
        "windows_frpc": _build_command("windows", "frpc", settings["active_config"]),
        "windows_frps": _build_command("windows", "frps", settings["active_config"]),
        "linux_frpc": _build_command("linux", "frpc", settings["active_config"]),
        "linux_frps": _build_command("linux", "frps", settings["active_config"]),
    }
    return {
        "settings": settings,
        "runtime_system": _current_system_key(),
        "configs": configs,
        "binaries": binaries,
        "commands": commands,
        "paths": {
            "frp_root": _rel(FRP_ROOT),
            "conf_dir": _rel(FRP_CONF_DIR),
            "bin_dir": _rel(FRP_BIN_DIR),
        },
    }


def get_config(config_name: str) -> Dict[str, Any]:
    path = _config_path(config_name)
    if not os.path.exists(path):
        raise ValueError("config not found")
    with open(path, "r", encoding="utf-8") as handle:
        content = handle.read()
    root, proxies = _parse_simple_toml(content)
    role = "frps" if os.path.basename(path).lower().startswith("frps") else "frpc"
    return {
        "name": os.path.basename(path),
        "role": role,
        "path": _rel(path),
        "content": content,
        "root": root,
        "proxies": proxies,
    }


def save_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    config_name = str(payload.get("name", "") or "").strip()
    path = _config_path(config_name)
    raw_content = payload.get("content")
    if isinstance(raw_content, str) and raw_content.strip():
        content = raw_content.replace("\r\n", "\n")
        normalized_root, normalized_proxies = _parse_simple_toml(content)
        if not content.endswith("\n"):
            content += "\n"
    else:
        root = payload.get("root") if isinstance(payload.get("root"), dict) else {}
        proxies = payload.get("proxies") if isinstance(payload.get("proxies"), list) else []
        normalized_root = {str(k).strip(): str(v).strip() for k, v in root.items() if str(k).strip()}
        normalized_proxies = []
        for item in proxies:
            if not isinstance(item, dict):
                continue
            proxy = {str(k).strip(): str(v).strip() for k, v in item.items() if str(k).strip() and str(v).strip()}
            if proxy:
                normalized_proxies.append(proxy)
        content = _render_toml(normalized_root, normalized_proxies)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    settings_update = {}
    if payload.get("set_active"):
        settings_update["active_config"] = config_name
        settings_update["active_role"] = "frps" if config_name.lower().startswith("frps") else "frpc"
        if payload.get("active_system"):
            settings_update["active_system"] = str(payload.get("active_system"))
        _save_settings(settings_update)
    return get_config(config_name)


def save_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _save_settings(payload)
