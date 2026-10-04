#!/usr/bin/env python3
"""Coordinate the single RTX 5090 between llama.cpp and knowledge rebuilds."""

from __future__ import annotations

import json
import os
import shutil
import socket
import socketserver
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import RLock
from urllib.request import urlopen


ROOT = Path(os.getenv("HDW_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
ENV_FILE = ROOT / "Configs/.env"
BASE_COMPOSE = ROOT / "Configs/docker-compose.yml"
GPU_COMPOSE = ROOT / "Configs/docker-compose.rebuild-gpu.yml"
RUNTIME_ROOT = Path(os.getenv("HDW_RUNTIME_ROOT", ROOT / "HDW_Runtime"))
SOCKET_PATH = Path(
    os.getenv(
        "HDW_MAINTENANCE_SOCKET",
        RUNTIME_ROOT / "maintenance/maintenance.sock",
    )
)
LLAMA_UNIT = os.getenv("HDW_LLAMA_SYSTEMD_UNIT", "hyperdrivewave-llama.service")

# ── 两种部署形态，协调器要区别对待 ───────────────────────────────
#
# 主站那套的前提是「**GPU 是稀缺资源**」：平时给 llama，知识库重建期间临时
# 借给 RAG/MinerU，重建完再还回去。prepare/restore 就是干这个的。
#
# 但还有另一种部署：**RAG 常驻 GPU**（多卡机器上单独给 RAG/MinerU 一张卡，
# `.env` 里 `HDW_RAG_DEVICE=cuda`）。这时根本不需要腾挪，照搬主站那套会出事：
#
#   · `restore` 里的 `_compose(gpu=False)` 会把 RAG 的 GPU **摘掉** ——
#     等于每次入库结束都把 RAG 悄悄降级成 CPU，而界面上一切正常；
#   · `_start_llama()` 会去 `systemctl --user enable --now` 宿主 llama 单元，
#     而**容器化 llama 的部署里那个单元是 disabled 的**，且预编译二进制在
#     麒麟这类老发行版上根本起不来、端口还被负载均衡器占着 ——
#     于是入库明明已经干完，却报
#         「任务结束后恢复大模型失败: [Errno 111] Connection refused」
#     把一次成功的入库显示成失败。
#
# 判据用 `.env` 的 `HDW_RAG_DEVICE`：它是「RAG 跑在哪个设备上」的唯一事实源。
# 主站是 `cpu`（需要腾挪），常驻 GPU 的部署是 `cuda`（不需要）。
RAG_ON_PERMANENT_GPU = os.getenv("HDW_RAG_DEVICE", "cpu").strip().lower() == "cuda"
# 宿主有没有 llama 单元可管。容器化 llama 的部署里没有（HDW_SKIP_LOCAL_LLM=true）。
HOST_LLAMA_MANAGED = os.getenv("HDW_SKIP_LOCAL_LLM", "false").strip().lower() != "true"
FRPC_UNIT = os.getenv("HDW_FRP_SYSTEMD_UNIT", "hyperdrivewave-frpc.service")

# 端口由 HDW_LLAMA_PORT 推导，不写死 1919。
# 原来这里是 `os.getenv("HDW_LLAMA_HEALTH_URL", "http://127.0.0.1:1919/health")`——
# 出口是有的，但**全项目没有任何地方设置过它**（.env/.env.example/两个 systemd
# 单元里都没有），所以事实上等于写死。改了 HDW_LLAMA_PORT 之后，
# 协调器的 /switch-llm、GPU 抢占、健康探测会全部指向旧端口。
LLAMA_PORT = os.getenv("HDW_LLAMA_PORT", "1919")
LLAMA_HEALTH_URL = os.getenv(
    "HDW_LLAMA_HEALTH_URL",
    f"http://127.0.0.1:{LLAMA_PORT}/health",
)
LLAMA_PROPS_URL = os.getenv(
    "HDW_LLAMA_PROPS_URL",
    f"http://127.0.0.1:{LLAMA_PORT}/props",
)
MODEL_CONFIG_PATH = Path(
    os.getenv(
        "HDW_MODEL_CONFIG_PATH",
        RUNTIME_ROOT / "model-config/config.json",
    )
)
if not MODEL_CONFIG_PATH.exists() and str(MODEL_CONFIG_PATH).startswith("/data/model-config/"):
    MODEL_CONFIG_PATH = RUNTIME_ROOT / "model-config" / MODEL_CONFIG_PATH.name
RUNTIME_MARKER_PATH = RUNTIME_ROOT / "maintenance/llm-runtime.json"
RAG_HEALTH_URL = os.getenv(
    "HDW_RAG_HEALTH_URL",
    f"http://127.0.0.1:{os.getenv('HDW_RAG_PORT', '8001')}/health",
)
MINERU_HEALTH_URL = os.getenv(
    "HDW_MINERU_HEALTH_URL",
    f"http://127.0.0.1:{os.getenv('HDW_MINERU_PORT', '8002')}/health",
)
OPERATION_TIMEOUT = float(os.getenv("HDW_MAINTENANCE_TIMEOUT", "0"))

state_lock = RLock()
operation_lock = RLock()
state = {
    "mode": "idle",
    "updated_at": time.time(),
    "error": "",
}


def _run(command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout or "command failed").strip()
        raise RuntimeError(f"{' '.join(command[:3])}: {detail[-2000:]}")
    return result.stdout.strip()


def _set_state(mode: str, error: str = "") -> None:
    with state_lock:
        state.update(mode=mode, error=error, updated_at=time.time())


def _snapshot() -> dict[str, object]:
    with state_lock:
        return {**state}


def _unit_active(unit: str) -> bool:
    return subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", unit],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def _wait_until(predicate, description: str) -> None:
    started = time.monotonic()
    while True:
        if predicate():
            return
        if OPERATION_TIMEOUT > 0 and time.monotonic() - started >= OPERATION_TIMEOUT:
            raise RuntimeError(f"等待超时：{description}")
        time.sleep(1)


def _wait_http(url: str, expected_device: str | None = None) -> dict[str, object]:
    def ready() -> bool:
        try:
            with urlopen(url, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return (
                payload.get("status") not in {"loading", "starting"}
                and (expected_device is None or payload.get("device") == expected_device)
            )
        except Exception:
            return False

    _wait_until(ready, f"{url} {expected_device or 'healthy'}")
    with urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _read_model_config() -> dict[str, object]:
    try:
        value = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise RuntimeError(f"无法读取模型配置：{MODEL_CONFIG_PATH}") from None
    if not isinstance(value, dict):
        raise RuntimeError(f"模型配置不是 JSON 对象：{MODEL_CONFIG_PATH}")
    return value


def _resolve_model_path(model: str) -> Path:
    candidate = Path(model).expanduser()
    if candidate.exists():
        return candidate
    candidate = ROOT / "HDW_Engines/LLM_Models" / model
    if candidate.exists():
        return candidate
    matches = list(
        (ROOT / "HDW_Engines/LLM_Models").glob(f"*/{Path(model).name}")
    )
    if matches:
        return matches[0]
    raise RuntimeError(f"模型不存在：{model}")


def _resolve_mmproj_path(mmproj: str) -> Path:
    """按 **start.sh 的解析顺序**校验视觉投影器路径。

    刻意不复用 `_resolve_model_path` 的 glob 兜底：start.sh 对 mmproj 只认
    两种形态——原样路径，或相对 `HDW_Engines/LLM_Models` 的路径。这里要是比它
    宽松，就会「放行 → 停掉 llama → start.sh exit 1」，以「本地推理挂了」收场，
    比在这里拦住难查得多。
    """
    candidate = Path(mmproj).expanduser()
    if candidate.is_file():
        return candidate
    candidate = ROOT / "HDW_Engines/LLM_Models" / mmproj
    if candidate.is_file():
        return candidate
    raise RuntimeError(
        f"视觉投影器不存在：{mmproj}"
        "（只认原样路径，或相对 HDW_Engines/LLM_Models 的路径）"
    )


def _configured_llm_target() -> dict[str, object]:
    local = _read_model_config().get("local")
    if not isinstance(local, dict):
        raise RuntimeError("模型配置缺少 local 段")
    engine = str(local.get("engine") or "llama.cpp").strip()
    model = str(local.get("model") or "").strip()
    if not model:
        raise RuntimeError("模型配置缺少 local.model")
    try:
        context_window = int(local.get("context_window") or 262144)
    except (TypeError, ValueError):
        raise RuntimeError("local.context_window 不是整数") from None
    if context_window < 4096:
        raise RuntimeError("local.context_window 不能小于 4096")
    normalized = engine.lower()
    if normalized not in {"llama.cpp", "llama-cpp", "llama", "freetoken"}:
        raise RuntimeError(f"不支持的本地推理引擎：{engine}")
    model_path = _resolve_model_path(model)
    if normalized == "freetoken" and shutil.which("ft") is None:
        raise RuntimeError(
            "FreeToken 未安装为宿主机 ft 命令，当前不能切换；"
            "请先安装 FreeToken CLI，或选择 llama.cpp"
        )
    mmproj = str(local.get("mmproj") or "").strip()
    mmproj_path = ""
    if normalized in {"llama.cpp", "llama-cpp", "llama"}:
        binary = ROOT / "HDW_Inference/llama/build/bin/llama-server"
        if not binary.is_file():
            raise RuntimeError(f"llama-server 不存在：{binary}")
        if not model_path.is_file():
            raise RuntimeError(
                f"llama.cpp 需要 GGUF 文件，当前模型不是文件：{model_path}"
            )
        # 提前校验投影器：start.sh 也会校验并在缺失时 exit 1，但那是在**停掉
        # llama 之后**——那时服务已经中断了。在这里拦住，配置回滚就不会以
        # 「本地推理挂了」收场。留空是合法的（表示不要视觉）。
        if mmproj:
            mmproj_path = str(_resolve_mmproj_path(mmproj))
    return {
        "engine": engine,
        "model": model,
        "model_path": str(model_path),
        "context_window": context_window,
        "mmproj": mmproj,
        "mmproj_path": mmproj_path,
    }


def _llama_props() -> dict[str, object]:
    try:
        with urlopen(LLAMA_PROPS_URL, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"读取本地推理实际状态失败：{exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("本地推理 /props 返回格式错误")
    return payload


def _runtime_model(props: dict[str, object]) -> str:
    return str(
        props.get("model_alias")
        or Path(str(props.get("model_path") or "")).name
        or ""
    ).strip()


def _model_matches(expected: str, actual: str) -> bool:
    expected_name = Path(expected).name
    actual_name = Path(actual).name
    return bool(
        expected_name
        and actual_name
        and (
            expected_name == actual_name
            or expected_name in actual
            or actual_name in expected_name
        )
    )


def _write_runtime_marker(
    target: dict[str, object],
    loaded_model: str,
    context_window: int,
    vision: bool,
) -> None:
    RUNTIME_MARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "engine": target["engine"],
        "configured_model": target["model"],
        "loaded_model": loaded_model,
        "context_window": context_window,
        "vision": vision,
        "updated_at": time.time(),
    }
    temporary = RUNTIME_MARKER_PATH.with_name(f".{RUNTIME_MARKER_PATH.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(RUNTIME_MARKER_PATH)


def _verify_llm_target(target: dict[str, object]) -> dict[str, object]:
    if str(target["engine"]).lower() == "freetoken":
        try:
            with urlopen(f"{LLAMA_HEALTH_URL.rsplit('/', 1)[0]}/v1/models", timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"读取 FreeToken /v1/models 失败：{exc}") from exc
        models = payload.get("data") if isinstance(payload, dict) else None
        item = models[0] if isinstance(models, list) and models else {}
        loaded_model = str(item.get("id") or "").strip()
        actual_context = int(
            item.get("context_length")
            or item.get("max_model_len")
            or 0
        )
        if not _model_matches(str(target["model"]), loaded_model):
            raise RuntimeError(
                f"FreeToken 实际加载的是 {loaded_model or '未知模型'}，"
                f"不是配置的 {target['model']}"
            )
        expected_context = int(target["context_window"])
        if actual_context and actual_context != expected_context:
            raise RuntimeError(
                f"FreeToken 实际上下文为 {actual_context}，不是配置的 {expected_context}"
            )
        _write_runtime_marker(
            target,
            loaded_model,
            actual_context or expected_context,
            False,
        )
        return {
            "engine": target["engine"],
            "configured_model": target["model"],
            "loaded_model": loaded_model,
            "context_window": actual_context or expected_context,
            "vision": False,
        }

    props = _llama_props()
    loaded_model = _runtime_model(props)
    if not _model_matches(str(target["model"]), loaded_model):
        raise RuntimeError(
            f"本地推理实际加载的是 {loaded_model or '未知模型'}，"
            f"不是配置的 {target['model']}"
        )
    actual_context = int(
        (props.get("default_generation_settings") or {}).get("n_ctx")
        or props.get("n_ctx")
        or 0
    )
    expected_context = int(target["context_window"])
    if actual_context and actual_context != expected_context:
        raise RuntimeError(
            f"本地推理实际上下文为 {actual_context}，不是配置的 {expected_context}"
        )
    _write_runtime_marker(
        target,
        loaded_model,
        actual_context or expected_context,
        bool((props.get("modalities") or {}).get("vision")),
    )
    return {
        "engine": target["engine"],
        "configured_model": target["model"],
        "loaded_model": loaded_model,
        "context_window": actual_context or expected_context,
        "vision": bool((props.get("modalities") or {}).get("vision")),
    }


def _compose_command(gpu: bool, action: str, service: str) -> list[str]:
    command = [
        "docker",
        "compose",
        "--env-file",
        str(ENV_FILE),
        "-f",
        str(BASE_COMPOSE),
    ]
    if gpu:
        command.extend(["-f", str(GPU_COMPOSE)])
    command.extend(["--profile", "base", "--profile", "knowledge", action])
    if action == "up":
        command.extend(["-d", "--force-recreate", "--no-deps"])
    if action == "ps":
        command.append("-q")
    command.append(service)
    return command


def _compose(gpu: bool) -> None:
    _run(_compose_command(gpu, "up", "hdw-rag"))
    _run(_compose_command(gpu, "up", "hdw-mineru"))


def _container_id(service: str) -> str:
    return _run(_compose_command(True, "ps", service)).strip()


def _verify_gpu_container(service: str, code: str) -> None:
    container = _container_id(service)
    if not container:
        raise RuntimeError(f"未找到容器：{service}")
    output = _run(["docker", "exec", container, "python", "-c", code])
    if not output:
        raise RuntimeError(f"{service} GPU 验证没有输出")


def _stop_llama() -> None:
    if not _unit_active(LLAMA_UNIT):
        return
    _run(["systemctl", "--user", "stop", LLAMA_UNIT])
    _wait_until(lambda: not _unit_active(LLAMA_UNIT), f"{LLAMA_UNIT} 停止")


def _start_llama() -> None:
    if not HOST_LLAMA_MANAGED:
        # 容器化 llama 的部署：宿主没有这个单元。硬去 enable --now 会：
        #   1) 拉起的二进制在老发行版上根本跑不起来；2) 就算起来了，
        #   端口被负载均衡器占着。失败会把整个 restore 拖垮，
        #   于是**一次成功的入库被显示成失败**。
        # llama 要不要跑是容器的事（hdw-llama / hdw-llama-lb），不归协调器管。
        return
    _run(["systemctl", "--user", "enable", "--now", LLAMA_UNIT])
    _wait_http(LLAMA_HEALTH_URL)


def _restore_locked() -> dict[str, object]:
    # ⚠ 这里**不要**无条件 `_compose(gpu=False)`：在 RAG 常驻 GPU 的部署上，
    # 那等于把 RAG 悄悄降级成 CPU，而且没有任何报错——下次提问才发现变慢。
    if RAG_ON_PERMANENT_GPU:
        rag = _wait_http(RAG_HEALTH_URL, expected_device="cuda")
    else:
        _compose(gpu=False)
        rag = _wait_http(RAG_HEALTH_URL, expected_device="cpu")
    mineru = _wait_http(MINERU_HEALTH_URL)
    _start_llama()
    _set_state("restored")
    return {"mode": "restored", "rag": rag, "mineru": mineru}


def prepare() -> dict[str, object]:
    with operation_lock:
        if _snapshot()["mode"] == "prepared":
            return _snapshot()
        _set_state("preparing")
        try:
            _stop_llama()
            # 常驻 GPU 时不用重建容器——但**下面的 GPU 校验照做**：
            # 那一步能在"RAG 不知何时掉了 GPU"时立刻抓住，而不是等用户发现变慢。
            if not RAG_ON_PERMANENT_GPU:
                _compose(gpu=True)
            rag = _wait_http(RAG_HEALTH_URL, expected_device="cuda")
            mineru = _wait_http(MINERU_HEALTH_URL)
            _verify_gpu_container(
                "hdw-rag",
                "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))",
            )
            _verify_gpu_container(
                "hdw-mineru",
                "import torch; from mineru.utils.config_reader import get_device; assert torch.cuda.is_available(); assert get_device() == 'cuda'; print(torch.cuda.get_device_name(0))",
            )
            _set_state("prepared")
            return {"mode": "prepared", "rag": rag, "mineru": mineru}
        except Exception as exc:
            error = str(exc)
            _set_state("restoring", error)
            try:
                _restore_locked()
            except Exception as restore_error:
                error = f"{error}; 自动恢复失败：{restore_error}"
                _set_state("error", error)
            raise RuntimeError(error) from exc


def restore() -> dict[str, object]:
    with operation_lock:
        if _snapshot()["mode"] == "restored":
            return _snapshot()
        _set_state("restoring")
        try:
            return _restore_locked()
        except Exception as exc:
            _set_state("error", str(exc))
            raise


def switch_llm() -> dict[str, object]:
    with operation_lock:
        current_mode = str(_snapshot()["mode"])
        if current_mode in {"preparing", "prepared", "restoring", "switching_llm"}:
            raise RuntimeError(f"当前资源状态为 {current_mode}，暂不能切换本地模型")
        if not HOST_LLAMA_MANAGED:
            # **不要让它"成功"。** 容器化 llama 的部署里 _stop/_start_llama 都是
            # 空操作，但下面 `_verify_llm_target` 探的是负载均衡器 —— 它一直
            # 活着、一直返回 200，于是这里会**报成功而什么都没重启**：
            # 界面显示「已切换」，模型 / MTP / mmproj 一个都没生效。
            # 假成功比报错难查得多，所以直接拦下来。
            raise RuntimeError(
                "本地推理跑在容器里（hdw-llama），协调器管不到它——"
                "systemctl 的宿主单元在本部署里是停用的。"
                "改完模型 / MTP / mmproj 要重建容器才生效："
                "bash Scripts/compose.sh up -d --force-recreate hdw-llama hdw-llama-2"
            )
        target = _configured_llm_target()
        _set_state("switching_llm")
        try:
            _stop_llama()
            _start_llama()
            runtime = _verify_llm_target(target)
            _set_state("llm_ready")
            return {
                "status": "ok",
                "mode": "llm_ready",
                "target": target,
                "runtime": runtime,
            }
        except Exception as exc:
            _set_state("error", str(exc))
            raise


def unload_llm() -> dict[str, object]:
    """停掉本地推理，把显存整个还回去。

    为什么只能停进程：llama.cpp 单模型模式**没有任何运行期卸载接口**。
    `/models/unload` 确实存在，但只在多模型 router 模式下注册
    （tools/server/server.cpp:239-243 的 `if (is_router_server)` 里），
    本项目是单模型启动，那条路由根本没挂上。

    为什么必须走 `systemctl stop` 而不能 kill 进程：单元是 `Restart=always`
    + `RestartSec=3`，杀掉会在 3 秒后自己回来。显式 stop 属于主动停止，
    systemd 不会重启它——`_stop_llama()` 在 prepare 流程里依赖的正是这一点。
    """
    with operation_lock:
        current_mode = str(_snapshot()["mode"])
        if current_mode in {
            "preparing",
            "prepared",
            "restoring",
            "switching_llm",
            "unloading_llm",
            "loading_llm",
        }:
            raise RuntimeError(f"当前资源状态为 {current_mode}，暂不能卸载本地模型")
        if not HOST_LLAMA_MANAGED:
            # 同上：_stop_llama 在这类部署里是空操作，不拦的话这里会返回
            # `llm_unloaded`，而 llama 容器照跑、显存一点没还。界面显示"已卸载"，
            # 用户以为卡腾出来了。直接说清楚该停什么。
            raise RuntimeError(
                "本地推理跑在容器里（hdw-llama），卸载要停容器而不是 systemctl：\n"
                "  停：bash Scripts/compose.sh stop hdw-llama hdw-llama-2\n"
                "  恢复：bash Scripts/compose.sh start hdw-llama hdw-llama-2"
            )
        _set_state("unloading_llm")
        try:
            _stop_llama()
            _set_state("llm_unloaded")
            return {"status": "ok", "mode": "llm_unloaded"}
        except Exception as exc:
            _set_state("error", str(exc))
            raise


def _frp_status() -> dict[str, object]:
    enabled = subprocess.run(
        ["systemctl", "--user", "is-enabled", "--quiet", FRPC_UNIT],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    return {"unit": FRPC_UNIT, "enabled": enabled, "active": _unit_active(FRPC_UNIT)}


def frp_enable() -> dict[str, object]:
    with operation_lock:
        _run(["systemctl", "--user", "enable", "--now", FRPC_UNIT])
        return _frp_status()


def frp_disable() -> dict[str, object]:
    with operation_lock:
        _run(["systemctl", "--user", "disable", "--now", FRPC_UNIT])
        return _frp_status()


def _unlink_quiet(path: Path) -> None:
    """删一个文件，不存在就算了。

    **不要改回 `path.unlink(missing_ok=True)`。** 那个参数是 **Python 3.8**
    才加的，而这个脚本跑在**宿主**的 python3 上——麒麟 V10 / RHEL8 这类系统
    自带的是 3.7，会直接抛

        TypeError: unlink() got an unexpected keyword argument 'missing_ok'

    后果不只是这一个函数失败：它在 `server_bind()` 里，异常会让协调器**启动即崩**，
    systemd 于是每 3 秒重启一次（`Active: activating (auto-restart)`）。
    socket 文件还留在原地，看起来"服务装着"，但后面没有进程在听——
    入库脚本连上去就是 `[Errno 111] Connection refused`，而报错信息把它
    伪装成"网络问题"。

    为什么会漏掉：这个脚本用了 `from __future__ import annotations`，
    `str | None` / `dict[str, object]` 这类 **3.10+ 的标注**全都延迟求值、
    在 3.7 上不报错，于是"语法检查通过、模块级执行也通过"。
    但 `missing_ok` 不是标注，是**运行时 API**，只有真的把 server 建起来
    才会碰到。**「能编译、能导入」不等于「能跑」**——验证这类脚本必须真的
    启动一次、确认 `systemctl --user is-active` 返回 `active`，
    而不是看到 socket 文件存在就当成功。
    """
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:  # 权限之类：删不掉也别让协调器起不来
        print(f"警告：无法删除 {path}（{exc}）", file=sys.stderr)


class UnixHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    address_family = socket.AF_UNIX
    daemon_threads = True
    allow_reuse_address = True

    def server_bind(self) -> None:
        SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)
        _unlink_quiet(SOCKET_PATH)
        super().server_bind()
        os.chmod(SOCKET_PATH, 0o660)

    def server_close(self) -> None:
        super().server_close()
        _unlink_quiet(SOCKET_PATH)


class Handler(BaseHTTPRequestHandler):
    server_version = "HyperDriveWaveMaintenance/1.0"

    def _reply(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in {"/health", "/status"}:
            self._reply(200, {"status": "ok", **_snapshot()})
            return
        if path == "/frp":
            self._reply(200, {"status": "ok", **_frp_status()})
            return
        self._reply(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length:
            self.rfile.read(length)
        try:
            if path == "/prepare":
                self._reply(200, prepare())
            elif path == "/restore":
                self._reply(200, restore())
            elif path == "/switch-llm":
                self._reply(200, switch_llm())
            elif path == "/unload-llm":
                self._reply(200, unload_llm())
            elif path == "/frp-enable":
                self._reply(200, {"status": "ok", **frp_enable()})
            elif path == "/frp-disable":
                self._reply(200, {"status": "ok", **frp_disable()})
            else:
                self._reply(404, {"error": "not found"})
        except Exception as exc:
            self._reply(503, {"status": "error", **_snapshot(), "error": str(exc)})

    def log_message(self, format: str, *args: object) -> None:
        print(format % args, file=sys.stderr, flush=True)


def main() -> None:
    SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    server = UnixHTTPServer(str(SOCKET_PATH), Handler)
    print(f"HyperDriveWave resource coordinator: {SOCKET_PATH}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
