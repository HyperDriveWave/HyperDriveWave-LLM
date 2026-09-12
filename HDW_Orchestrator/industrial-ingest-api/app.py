from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import shutil
import socket
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from http import client as http_client
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from hdw_auth import AuthError, AuthStore


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".gif",
    ".tiff",
    ".docx",
    ".pptx",
    ".xlsx",
    ".md",
    ".markdown",
}
SOURCE_ROOT = Path(os.getenv("HDW_KNOWLEDGE_SOURCE_ROOT", "/data/knowledge_sources"))
PARSED_ROOT = Path(os.getenv("HDW_MINERU_PARSED_ROOT", "/data/mineru/parsed"))
CHUNKS_PATH = Path(os.getenv("HDW_RAG_INDEX_PATH", "/data/rag/chunks.jsonl"))
ZVEC_PATH = Path(os.getenv("HDW_ZVEC_COLLECTION_PATH", "/data/zvec/industrial_chunks"))
STATE_PATH = Path(os.getenv("HDW_INGEST_STATE_PATH", "/data/ingest/state.json"))
INGEST_ROOT = STATE_PATH.parent
JOB_ROOT = INGEST_ROOT / "jobs"
PIPELINES_ROOT = Path(os.getenv("HDW_PIPELINES_ROOT", "/pipelines"))
MINERU_URL = os.getenv("HDW_MINERU_BASE_URL", "http://hdw-mineru:8002")
RAG_URL = os.getenv("HDW_RAG_BASE_URL", "http://hdw-rag:8001")
RAG_REMOTE_URLS = tuple(
    url.strip().rstrip("/")
    for url in os.getenv("HDW_RAG_REMOTE_URLS", "").split(",")
    if url.strip()
)
RAG_ADMIN_TOKEN = os.getenv("HDW_RAG_ADMIN_TOKEN", "").strip()
REMOTE_RAG_SYNC_TIMEOUT = float(os.getenv("HDW_REMOTE_RAG_SYNC_TIMEOUT", "0"))
MAX_UPLOAD_BYTES = int(os.getenv("HDW_MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))
RAG_REQUEST_TIMEOUT = int(os.getenv("HDW_INGEST_RAG_TIMEOUT", "0"))
AUTH_CSV_PATH = Path(os.getenv("HDW_AUTH_CSV_PATH", "/app/auth.csv"))
AUTH_SECRET = os.getenv("HDW_AUTH_SECRET", "change_me")
MAINTENANCE_SOCKET = Path(
    os.getenv("HDW_MAINTENANCE_SOCKET", "/run/hdw-maintenance/maintenance.sock")
)
MAINTENANCE_TIMEOUT = float(os.getenv("HDW_MAINTENANCE_TIMEOUT", "0"))

PROGRESS_STAGES = {
    "llm_unloading": (0, 5),
    "clearing": (5, 35),
    "parsing": (5, 20),
    "chunking": (25, 15),
    "graph": (40, 20),
    "embedding": (60, 30),
    "local_rebuild_completed": (90, 2),
    "llm_starting": (92, 3),
    "sync": (95, 5),
    "sync_completed": (100, 0),
    "completed": (100, 0),
}

app = FastAPI(title="HyperDriveWave Knowledge Ingest API", version="0.1.0")
state_lock = threading.RLock()
job_lock = threading.Lock()
auth_store = AuthStore(AUTH_CSV_PATH, AUTH_SECRET)


class IngestRequest(BaseModel):
    document_ids: list[str] = Field(default_factory=list)
    full_rebuild: bool = False


def require_user(x_hdw_session: str | None) -> dict[str, str]:
    token = str(x_hdw_session or "").strip()
    if not token:
        raise HTTPException(status_code=401, detail="login required")
    try:
        return auth_store.verify(token)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail="invalid or expired session") from exc


def require_admin(x_hdw_session: str | None) -> dict[str, str]:
    user = require_user(x_hdw_session)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="admin permission required")
    return user


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"documents": {}, "jobs": {}}
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"documents": {}, "jobs": {}}
    data.setdefault("documents", {})
    data.setdefault("jobs", {})
    return data


def _write_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(STATE_PATH)


def _document_id(filename: str) -> str:
    return hashlib.sha1(f"source:{filename}".encode("utf-8")).hexdigest()[:16]


def _safe_filename(filename: str | None) -> str:
    name = Path(filename or "").name.strip()
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="filename is required")
    if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"unsupported file type: {Path(name).suffix}")
    return name[:240]


def _scan_files() -> dict[str, Path]:
    files: dict[str, Path] = {}
    if SOURCE_ROOT.exists():
        for path in SOURCE_ROOT.iterdir():
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS | {".mineru"}:
                files[path.name] = path
    source_stems = {path.stem for path in files.values()}
    if PARSED_ROOT.exists():
        for path in PARSED_ROOT.iterdir():
            if not path.is_file() or path.suffix.lower() not in {".md", ".markdown"}:
                continue
            if path.stem not in source_stems:
                files[path.name] = path
    return files


def _chunk_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    if not CHUNKS_PATH.exists():
        return counts
    try:
        with CHUNKS_PATH.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                source = Path(str(row.get("source_file", ""))).name
                counts[source] = counts.get(source, 0) + 1
    except (OSError, json.JSONDecodeError):
        return {}
    return counts


def _merge_documents(state: dict[str, Any]) -> None:
    counts = _chunk_counts()
    scanned = _scan_files()
    source_stems = {path.stem for path in SOURCE_ROOT.iterdir()} if SOURCE_ROOT.exists() else set()
    for doc_id, item in list(state["documents"].items()):
        if not item.get("uploaded") and Path(str(item.get("filename", ""))).stem in source_stems:
            state["documents"].pop(doc_id, None)
    for filename, path in scanned.items():
        doc_id = _document_id(filename)
        item = state["documents"].setdefault(
            doc_id,
            {
                "id": doc_id,
                "filename": filename,
                "source_format": Path(filename).suffix.lower().lstrip("."),
                "uploaded": path.parent == SOURCE_ROOT,
                "uploaded_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                "size": path.stat().st_size,
                "parse_status": "unknown",
                "chunk_status": "unknown",
                "embedding_status": "unknown",
                "graph_status": "unknown",
                "rerank_status": "query_time",
                "remote_sync_status": "not_configured",
                "total_status": "not_ingested",
                "chunk_count": 0,
                "error": "",
            },
        )
        item["filename"] = filename
        item["size"] = path.stat().st_size
        item["uploaded"] = path.parent == SOURCE_ROOT or bool(item.get("uploaded"))
        item.setdefault("remote_sync_status", "not_configured")
        if item.get("rerank_status") in {None, "", "unknown", "unverified"}:
            item["rerank_status"] = "query_time"
        item["chunk_count"] = counts.get(
            filename,
            counts.get(f"{Path(filename).stem}.md", item.get("chunk_count", 0)),
        )
        if item["chunk_count"] and CHUNKS_PATH.exists():
            if item.get("parse_status") in {None, "", "unknown"}:
                item["parse_status"] = "completed"
            if item.get("chunk_status") in {None, "", "unknown"}:
                item["chunk_status"] = "completed"
            if item.get("embedding_status") in {None, "", "unknown"}:
                item["embedding_status"] = "available" if ZVEC_PATH.exists() else "unknown"
            if item.get("graph_status") in {None, "", "unknown"}:
                item["graph_status"] = "available"
            if item.get("total_status") in {None, "", "unknown", "not_ingested"}:
                item["total_status"] = (
                    "ingested" if item["embedding_status"] == "available" else "not_ingested"
                )


def _snapshot() -> dict[str, Any]:
    with state_lock:
        state = _read_state()
        _merge_documents(state)
        _write_state(state)
        return state


def _update_job(job_id: str, **changes: Any) -> None:
    with state_lock:
        state = _read_state()
        job = state["jobs"].setdefault(job_id, {"id": job_id})
        job.update(changes, updated_at=_now())
        _write_state(state)


def _update_documents(document_ids: list[str], **changes: Any) -> None:
    with state_lock:
        state = _read_state()
        for doc_id in document_ids:
            if doc_id in state["documents"]:
                state["documents"][doc_id].update(changes)
        _write_state(state)


def _progress(
    stage: str,
    completed: int,
    total: int,
    *,
    unit: str = "documents",
    current_document_id: str = "",
    current_document: str = "",
    message: str = "",
) -> dict[str, Any]:
    completed = max(0, completed)
    total = max(0, total)
    stage_percent = 100.0 if total and completed >= total else (
        round(completed / total * 100, 2) if total else 0.0
    )
    base, span = PROGRESS_STAGES.get(stage, (0, 0))
    percent = round(base + span * stage_percent / 100, 2)
    return {
        "stage": stage,
        "completed": completed,
        "total": total,
        "unit": unit,
        "stage_percent": stage_percent,
        "percent": percent,
        "current_document_id": current_document_id,
        "current_document": current_document,
        "message": message,
    }


def _normalize_live_progress(value: dict[str, Any]) -> dict[str, Any]:
    progress = dict(value)
    stage = str(progress.get("stage") or "")
    base, span = PROGRESS_STAGES.get(stage, (0, 0))
    if "stage_percent" not in progress:
        completed = int(progress.get("completed") or 0)
        total = int(progress.get("total") or 0)
        progress["stage_percent"] = round(completed / total * 100, 2) if total else 0.0
    progress["percent"] = round(
        base + span * float(progress.get("stage_percent") or 0) / 100,
        2,
    )
    return progress


def _read_progress(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _maintenance_request(action: str) -> dict[str, Any]:
    if action not in {"prepare", "restore"}:
        raise ValueError(f"unsupported maintenance action: {action}")
    if not MAINTENANCE_SOCKET.exists():
        raise RuntimeError(f"GPU resource coordinator socket not found: {MAINTENANCE_SOCKET}")

    request = (
        f"POST /{action} HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Connection: close\r\n"
        "Content-Length: 0\r\n\r\n"
    ).encode("ascii")
    timeout = MAINTENANCE_TIMEOUT if MAINTENANCE_TIMEOUT > 0 else None
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(str(MAINTENANCE_SOCKET))
        connection.sendall(request)
        response = bytearray()
        while chunk := connection.recv(64 * 1024):
            response.extend(chunk)

    header, separator, body = bytes(response).partition(b"\r\n\r\n")
    if not separator:
        raise RuntimeError("GPU resource coordinator returned an invalid response")
    status_line = header.splitlines()[0].decode("ascii", errors="replace")
    try:
        status_code = int(status_line.split()[1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"GPU resource coordinator returned: {status_line}") from exc
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("GPU resource coordinator returned invalid JSON") from exc
    if status_code >= 400:
        raise RuntimeError(str(payload.get("error") or f"maintenance HTTP {status_code}"))
    return payload


def _live_job(job: dict[str, Any]) -> dict[str, Any]:
    result = dict(job)
    job_id = str(result.get("id") or "")
    if str(result.get("stage") or "") not in {"sync", "completed", "failed"}:
        local_progress = _read_progress(JOB_ROOT / job_id / "progress.json")
        shared_progress = _read_progress(ZVEC_PATH.parent / "reindex_progress.json")
        for progress in (local_progress, shared_progress):
            if progress and str(progress.get("job_id") or job_id) == job_id:
                result["progress"] = _normalize_live_progress(progress)
    return result


def _run(command: list[str], env: dict[str, str]) -> str:
    result = subprocess.run(
        command,
        cwd=str(PIPELINES_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout or "command failed").strip()
        raise RuntimeError(f"{Path(command[1]).name if len(command) > 1 else command[0]}: {detail[-2000:]}")
    return result.stdout.strip()


def _reindex(job_id: str = "") -> dict[str, Any]:
    request = Request(
        f"{RAG_URL.rstrip('/')}/admin/reindex",
        data=json.dumps({"job_id": job_id}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if RAG_REQUEST_TIMEOUT > 0:
        response_context = urlopen(request, timeout=RAG_REQUEST_TIMEOUT)
    else:
        response_context = urlopen(request)
    with response_context as response:
        return json.loads(response.read().decode("utf-8"))


def _remote_rag_urls() -> list[str]:
    return list(dict.fromkeys(url for url in RAG_REMOTE_URLS if url != RAG_URL))


def _sync_remote_url(url: str, chunks_path: Path, job_id: str) -> dict[str, Any]:
    parsed = urlsplit(f"{url.rstrip('/')}/admin/sync")
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RuntimeError(f"invalid remote RAG URL: {url}")
    boundary = f"----HyperDriveWave{uuid.uuid4().hex}"
    prefix = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="chunks.jsonl"\r\n'
        "Content-Type: application/x-ndjson\r\n\r\n"
    ).encode("ascii")
    suffix = f"\r\n--{boundary}--\r\n".encode("ascii")
    request_target = parsed.path or "/"
    query = parsed.query
    if job_id:
        query = f"{query}&{urlencode({'job_id': job_id})}" if query else urlencode(
            {"job_id": job_id}
        )
    if query:
        request_target = f"{request_target}?{query}"
    content_length = len(prefix) + chunks_path.stat().st_size + len(suffix)
    connection_type = (
        http_client.HTTPSConnection
        if parsed.scheme == "https"
        else http_client.HTTPConnection
    )
    timeout = REMOTE_RAG_SYNC_TIMEOUT if REMOTE_RAG_SYNC_TIMEOUT > 0 else None
    connection = connection_type(parsed.hostname, parsed.port, timeout=timeout)
    try:
        connection.putrequest("POST", request_target)
        connection.putheader(
            "Content-Type",
            f"multipart/form-data; boundary={boundary}",
        )
        connection.putheader("Content-Length", str(content_length))
        connection.putheader("X-HDW-RAG-Admin-Token", RAG_ADMIN_TOKEN)
        connection.putheader("Connection", "close")
        connection.endheaders()
        connection.send(prefix)
        with chunks_path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                connection.send(chunk)
        connection.send(suffix)
        response = connection.getresponse()
        body = response.read().decode("utf-8", errors="replace")
        if response.status >= 400:
            raise RuntimeError(f"HTTP {response.status}: {body[:500]}")
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid remote RAG response: {body[:500]}") from exc
        return {"url": url, "status": "completed", "response": payload}
    finally:
        connection.close()


def _sync_remote_rag(job_id: str) -> dict[str, Any]:
    urls = _remote_rag_urls()
    if not urls:
        return {"status": "skipped", "reason": "no remote RAG URL configured", "nodes": []}
    if not RAG_ADMIN_TOKEN:
        raise RuntimeError("HDW_RAG_ADMIN_TOKEN is not configured")
    if not CHUNKS_PATH.is_file():
        raise RuntimeError(f"RAG chunks file not found: {CHUNKS_PATH}")

    results: list[dict[str, Any]] = []
    errors: list[str] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=len(urls)) as executor:
        futures = {
            executor.submit(_sync_remote_url, url, CHUNKS_PATH, job_id): url
            for url in urls
        }
        for future in as_completed(futures):
            url = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append(f"{url}: {exc}")
            completed += 1
            _update_job(
                job_id,
                progress=_progress(
                    "sync",
                    completed,
                    len(urls),
                    unit="remote_rag_nodes",
                    current_document=url,
                    message=f"已完成 {completed} / {len(urls)} 个远端 RAG 节点",
                ),
            )
    if errors:
        raise RuntimeError("远端 RAG 同步失败：" + "；".join(errors))
    return {
        "status": "completed",
        "nodes": sorted(results, key=lambda item: item["url"]),
    }


def _chunk_counts_for_documents() -> dict[str, int]:
    counts = _chunk_counts()
    result: dict[str, int] = {}
    with state_lock:
        state = _read_state()
        for doc_id, item in state["documents"].items():
            result[doc_id] = counts.get(
                item["filename"],
                counts.get(f"{Path(item['filename']).stem}.md", 0),
            )
    return result


def _prepare_pipeline_input(
    input_root: Path,
    excluded_filenames: set[str] | None = None,
) -> None:
    excluded_filenames = excluded_filenames or set()
    excluded_stems = {Path(name).stem for name in excluded_filenames}
    if input_root.exists():
        shutil.rmtree(input_root)
    input_root.mkdir(parents=True, exist_ok=True)
    source_stems: set[str] = set()
    if SOURCE_ROOT.exists():
        for path in SOURCE_ROOT.iterdir():
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                if path.name in excluded_filenames or path.stem in excluded_stems:
                    continue
                shutil.copy2(path, input_root / path.name)
                source_stems.add(path.stem)
    if PARSED_ROOT.exists():
        for path in PARSED_ROOT.iterdir():
            if (
                path.is_file()
                and path.suffix.lower() in {".md", ".markdown"}
                and path.stem not in source_stems
                and path.stem not in excluded_stems
            ):
                shutil.copy2(path, input_root / path.name)


def _replace_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)


def _stabilize_chunks(path: Path) -> None:
    source_names = {
        source.stem: source.name
        for source in SOURCE_ROOT.iterdir()
        if source.is_file() and source.suffix.lower() in SUPPORTED_EXTENSIONS
    } if SOURCE_ROOT.exists() else {}
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            parsed_name = Path(str(row.get("source_file", ""))).name
            filename = source_names.get(Path(parsed_name).stem, parsed_name)
            document_id = _document_id(filename)
            chunk_index = int(row.get("chunk_index", 0))
            row["document_id"] = document_id
            row["chunk_id"] = f"{document_id}-{chunk_index:04d}"
            row["source_file"] = str(PARSED_ROOT / parsed_name)
            row["source_format"] = Path(filename).suffix.lower().lstrip(".")
            rows.append(row)
    temporary = path.with_suffix(path.suffix + ".stable.tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _document_artifacts(filename: str) -> list[Path]:
    stem = Path(filename).stem
    return [
        SOURCE_ROOT / filename,
        PARSED_ROOT / f"{stem}.md",
        PARSED_ROOT / f"{stem}.markdown",
        PARSED_ROOT / f"{stem}.mineru.json",
    ]


def _backup_chunks(job_dir: Path) -> Path | None:
    if not CHUNKS_PATH.exists():
        return None
    job_dir.mkdir(parents=True, exist_ok=True)
    backup = job_dir / "previous_chunks.jsonl"
    shutil.copy2(CHUNKS_PATH, backup)
    return backup


def _restore_graph_and_index(
    job_id: str,
    backup_chunks: Path | None,
    graph_script: str,
    python: str,
    env: dict[str, str],
) -> None:
    if not backup_chunks or not backup_chunks.exists():
        return
    _replace_file(backup_chunks, CHUNKS_PATH)
    _run([python, graph_script, "--input", str(backup_chunks), "--replace-all"], env)
    _reindex(job_id)


def _commit_parsed(parsed_root: Path) -> None:
    temporary = PARSED_ROOT.with_name(f"{PARSED_ROOT.name}.commit")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(parsed_root, temporary)
    if PARSED_ROOT.exists():
        shutil.rmtree(PARSED_ROOT)
    temporary.replace(PARSED_ROOT)


def _remove_document_artifacts(filename: str) -> None:
    for path in _document_artifacts(filename):
        path.unlink(missing_ok=True)


def _clear_directory(path: Path) -> list[str]:
    removed: list[str] = []
    if not path.exists():
        return removed
    for child in path.iterdir():
        removed.append(str(child))
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)
    return removed


def _clear_knowledge_runtime() -> list[str]:
    removed: list[str] = []
    for path in (SOURCE_ROOT, PARSED_ROOT, INGEST_ROOT / "pipeline_input", JOB_ROOT):
        removed.extend(_clear_directory(path))
        path.mkdir(parents=True, exist_ok=True)

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CHUNKS_PATH.exists():
        removed.append(str(CHUNKS_PATH))
    CHUNKS_PATH.write_text("", encoding="utf-8")
    return removed


def _prune_state_after_rebuild(
    *,
    deleted_document_id: str | None = None,
    document_ids: list[str],
    full_rebuild: bool = False,
) -> None:
    with state_lock:
        state = _read_state()
        if deleted_document_id:
            state["documents"].pop(deleted_document_id, None)
        if full_rebuild:
            valid_ids = {_document_id(filename) for filename in _scan_files()}
            state["documents"] = {
                doc_id: item
                for doc_id, item in state["documents"].items()
                if doc_id in valid_ids
            }
        for doc_id in document_ids:
            if doc_id in state["documents"]:
                state["documents"][doc_id].update(
                    parse_status="completed",
                    chunk_status="completed",
                    embedding_status="completed",
                    graph_status="completed",
                    rerank_status="query_time",
                    remote_sync_status="pending",
                    total_status="ingested",
                    error="",
                )
        counts = _chunk_counts_for_documents()
        for doc_id, count in counts.items():
            if doc_id in state["documents"]:
                state["documents"][doc_id]["chunk_count"] = count
        _write_state(state)


def _run_ingest(
    job_id: str,
    document_ids: list[str],
    *,
    full_rebuild: bool = False,
    reuse_mineru_cache: bool = False,
    excluded_filenames: set[str] | None = None,
    deleted_document_id: str | None = None,
) -> None:
    env = os.environ.copy()
    env.update(
        {
            "HDW_MINERU_BASE_URL": MINERU_URL,
            "HDW_NEO4J_HTTP_URL": os.getenv(
                "HDW_NEO4J_HTTP_URL",
                "http://hdw-neo4j:7474/db/neo4j/tx/commit",
            ),
            "NEO4J_AUTH": os.getenv("NEO4J_AUTH", "neo4j/change_me"),
            "HDW_ENTITY_ALIASES_PATH": os.getenv(
                "HDW_ENTITY_ALIASES_PATH", "/app/entity_aliases.json"
            ),
            "HDW_GRAPH_REVIEW_PATH": os.getenv(
                "HDW_GRAPH_REVIEW_PATH", "/data/ingest/entities_review.jsonl"
            ),
            "HDW_MINERU_REUSE_CACHE": "1" if reuse_mineru_cache else "0",
            "HDW_MINERU_API_OUTPUT_ROOT": os.getenv(
                "HDW_MINERU_API_OUTPUT_ROOT", "/data/mineru/api_output"
            ),
        }
    )
    python = os.getenv("PYTHON", "python")
    parse_script = str(PIPELINES_ROOT / "parse_documents.py")
    chunks_script = str(PIPELINES_ROOT / "ingest_documents.py")
    graph_script = str(PIPELINES_ROOT / "import_graph.py")
    job_dir = JOB_ROOT / job_id
    staged_input = job_dir / "input"
    staged_parsed = job_dir / "parsed"
    staged_chunks = job_dir / "chunks.jsonl"
    progress_path = job_dir / "progress.json"
    graph_touched = False
    local_rebuild_committed = False
    previous_chunks: Path | None = None
    current_stage = "llm_unloading"
    maintenance_requested = False
    maintenance_restored = False
    remote_sync: dict[str, Any] = {
        "status": "skipped",
        "reason": "no remote RAG URL configured",
        "nodes": [],
    }
    try:
        JOB_ROOT.mkdir(parents=True, exist_ok=True)
        _update_job(
            job_id,
            status="running",
            stage="llm_unloading",
            message="大模型卸载",
            progress=_progress(
                "llm_unloading",
                0,
                1,
                unit="job",
                message="正在释放本机 5090 给知识库重建",
            ),
        )
        maintenance_requested = True
        _maintenance_request("prepare")

        current_stage = "parsing"
        job_message = "删除文档并全量一致性重建" if deleted_document_id else (
            "全量一致性重建中" if full_rebuild else "MinerU 文档解析中"
        )
        _update_job(
            job_id,
            status="running",
            stage="parsing",
            message=job_message,
            progress=_progress(
                "parsing",
                0,
                len(document_ids),
                message="等待 MinerU 解析",
            ),
        )
        _update_documents(
            document_ids,
            parse_status="running",
            chunk_status="pending",
            embedding_status="pending",
            graph_status="pending",
            total_status="deleting" if deleted_document_id else "processing",
            error="",
        )
        previous_chunks = _backup_chunks(job_dir)
        _prepare_pipeline_input(staged_input, excluded_filenames)
        _run(
            [
                python,
                parse_script,
                "--input",
                str(staged_input),
                "--output",
                str(staged_parsed),
                "--progress-file",
                str(progress_path),
            ],
            env,
        )

        current_stage = "chunking"
        _update_job(
            job_id,
            stage="chunking",
            message="文档切分与结构化中",
            progress=_progress("chunking", 0, len(document_ids), message="等待文档切分"),
        )
        _update_documents(document_ids, parse_status="completed", chunk_status="running")
        _run(
            [
                python,
                chunks_script,
                "--input",
                str(staged_parsed),
                "--output",
                str(staged_chunks),
                "--progress-file",
                str(progress_path),
            ],
            env,
        )
        _stabilize_chunks(staged_chunks)

        current_stage = "graph"
        _update_job(
            job_id,
            stage="graph",
            message="Neo4j 知识图谱全量替换中" if full_rebuild or deleted_document_id else "Neo4j 知识图谱构建中",
            progress=_progress("graph", 0, len(document_ids), message="等待知识图谱构建"),
        )
        _update_documents(document_ids, chunk_status="completed", graph_status="running")
        graph_command = [
            python,
            graph_script,
            "--input",
            str(staged_chunks),
            "--progress-file",
            str(progress_path),
        ]
        if full_rebuild or deleted_document_id:
            graph_command.append("--replace-all")
        graph_touched = True
        _run(graph_command, env)

        current_stage = "embedding"
        with staged_chunks.open(encoding="utf-8") as chunks_file:
            embedding_total = sum(1 for line in chunks_file if line.strip())
        _update_job(
            job_id,
            stage="embedding",
            message="BGE-M3 嵌入与 Zvec 索引重建中",
            progress=_progress(
                "embedding",
                0,
                embedding_total,
                unit="chunks",
                message="等待 BGE-M3 嵌入",
            ),
        )
        _update_documents(document_ids, graph_status="completed", embedding_status="running")
        _replace_file(staged_chunks, CHUNKS_PATH)
        rag_result = _reindex(job_id)

        _commit_parsed(staged_parsed)
        if deleted_document_id and excluded_filenames:
            for filename in excluded_filenames:
                _remove_document_artifacts(filename)
        _prune_state_after_rebuild(
            deleted_document_id=deleted_document_id,
            document_ids=document_ids,
            full_rebuild=full_rebuild or bool(deleted_document_id),
        )
        local_rebuild_committed = True

        current_stage = "local_rebuild_completed"
        _update_job(
            job_id,
            stage="local_rebuild_completed",
            message="重建完成",
            progress=_progress(
                "local_rebuild_completed",
                1,
                1,
                unit="job",
                message="本机知识库重建完成",
            ),
        )

        current_stage = "llm_starting"
        _update_job(
            job_id,
            stage="llm_starting",
            message="拉起大模型",
            progress=_progress(
                "llm_starting",
                0,
                1,
                unit="job",
                message="正在恢复本地大模型和 CPU RAG 后备",
            ),
        )
        _maintenance_request("restore")
        maintenance_restored = True

        remote_urls = _remote_rag_urls()
        if remote_urls:
            current_stage = "sync"
            _update_documents(document_ids, remote_sync_status="running")
            _update_job(
                job_id,
                stage="sync",
                message="同步远端RAG中",
                progress=_progress(
                    "sync",
                    0,
                    len(remote_urls),
                    unit="remote_rag_nodes",
                    message="等待远端 RAG 节点",
                ),
            )
            remote_sync = _sync_remote_rag(job_id)
            _update_documents(document_ids, remote_sync_status="completed")
            current_stage = "sync_completed"
            _update_job(
                job_id,
                stage="sync_completed",
                message="同步完成",
                progress=_progress(
                    "sync_completed",
                    1,
                    1,
                    unit="remote_rag_nodes",
                    message="远端 RAG 同步完成",
                ),
            )
        else:
            _update_documents(document_ids, remote_sync_status="not_configured")
        completion_message = "最终完成："
        completion_message += "文档删除并全量重建完成" if deleted_document_id else (
            "知识库全量重建完成" if full_rebuild else "知识库入库完成"
        )
        completion_message += "，大模型已拉起"
        if remote_sync["status"] == "completed":
            completion_message += "，远端 RAG 同步完成"
        elif remote_sync["status"] == "skipped":
            completion_message += "，远端 RAG 未配置"
        _update_job(
            job_id,
            status="completed",
            stage="completed",
            message=completion_message,
            finished_at=_now(),
            rag=rag_result,
            remote_rag=remote_sync,
            progress=_progress("completed", 1, 1, unit="job", message="全部处理完成"),
        )
    except Exception as exc:
        try:
            if graph_touched and not local_rebuild_committed and current_stage != "sync":
                _restore_graph_and_index(job_id, previous_chunks, graph_script, python, env)
        except Exception as restore_error:
            exc = RuntimeError(f"{exc}; consistency restore failed: {restore_error}")
        if current_stage == "sync":
            _update_documents(
                document_ids,
                remote_sync_status="failed",
                error=str(exc),
            )
        else:
            failure_status = {
                "parsing": {"parse_status": "failed"},
                "chunking": {"chunk_status": "failed"},
                "graph": {"graph_status": "failed"},
                "embedding": {"embedding_status": "failed"},
                "llm_unloading": {"embedding_status": "failed"},
                "local_rebuild_completed": {"embedding_status": "failed"},
                "llm_starting": {"embedding_status": "failed"},
            }.get(current_stage, {})
            _update_documents(
                document_ids,
                total_status="failed",
                error=str(exc),
                **failure_status,
            )
        _update_job(
            job_id,
            status="failed",
            stage="failed",
            message=str(exc),
            error=str(exc),
            finished_at=_now(),
            progress={
                **_progress("failed", 0, 1, unit="job", message="任务失败"),
                "error": str(exc),
            },
        )
    finally:
        if maintenance_requested and not maintenance_restored:
            try:
                _maintenance_request("restore")
            except Exception as restore_error:
                with state_lock:
                    current_error = str(
                        _read_state().get("jobs", {}).get(job_id, {}).get("error") or ""
                    )
                _update_job(
                    job_id,
                    message=f"任务结束后恢复大模型失败：{restore_error}",
                    error="; ".join(filter(None, (current_error, str(restore_error)))),
                )
        shutil.rmtree(job_dir, ignore_errors=True)


def _run_reset(job_id: str) -> None:
    env = os.environ.copy()
    env.update(
        {
            "HDW_NEO4J_HTTP_URL": os.getenv(
                "HDW_NEO4J_HTTP_URL",
                "http://hdw-neo4j:7474/db/neo4j/tx/commit",
            ),
            "NEO4J_AUTH": os.getenv("NEO4J_AUTH", "neo4j/change_me"),
            "HDW_ENTITY_ALIASES_PATH": os.getenv(
                "HDW_ENTITY_ALIASES_PATH", "/app/entity_aliases.json"
            ),
            "HDW_GRAPH_REVIEW_PATH": os.getenv(
                "HDW_GRAPH_REVIEW_PATH", "/data/ingest/entities_review.jsonl"
            ),
        }
    )
    python = os.getenv("PYTHON", "python")
    graph_script = str(PIPELINES_ROOT / "import_graph.py")
    current_stage = "llm_unloading"
    maintenance_requested = False
    maintenance_restored = False
    remote_sync: dict[str, Any] = {
        "status": "skipped",
        "reason": "no remote RAG URL configured",
        "nodes": [],
    }
    try:
        _update_job(
            job_id,
            status="running",
            stage="llm_unloading",
            message="大模型卸载",
            progress=_progress(
                "llm_unloading",
                0,
                1,
                unit="job",
                message="正在释放本机 5090 给知识库重建",
            ),
        )
        maintenance_requested = True
        _maintenance_request("prepare")

        current_stage = "clearing"
        _update_job(
            job_id,
            stage="clearing",
            message="清理知识库文件与入库状态",
            progress=_progress("clearing", 0, 1, unit="job", message="清理旧数据"),
        )
        removed_files = _clear_knowledge_runtime()
        _update_job(
            job_id,
            stage="graph",
            message="清空 Neo4j 知识图谱中",
            removed_files_count=len(removed_files),
        )
        graph_result = _run(
            [python, graph_script, "--input", str(CHUNKS_PATH), "--replace-all"],
            env,
        )
        _update_job(
            job_id,
            stage="embedding",
            message="清空 Zvec 向量索引中",
            progress=_progress("embedding", 0, 0, unit="chunks", message="清空向量索引"),
        )
        rag_result = _reindex(job_id)
        with state_lock:
            state = _read_state()
            state["documents"] = {}
            _write_state(state)

        current_stage = "local_rebuild_completed"
        _update_job(
            job_id,
            stage="local_rebuild_completed",
            message="重建完成",
            progress=_progress(
                "local_rebuild_completed",
                1,
                1,
                unit="job",
                message="纯净底库重建完成",
            ),
        )

        current_stage = "llm_starting"
        _update_job(
            job_id,
            stage="llm_starting",
            message="拉起大模型",
            progress=_progress(
                "llm_starting",
                0,
                1,
                unit="job",
                message="正在恢复本地大模型和 CPU RAG 后备",
            ),
        )
        _maintenance_request("restore")
        maintenance_restored = True

        remote_urls = _remote_rag_urls()
        if remote_urls:
            current_stage = "sync"
            _update_job(
                job_id,
                stage="sync",
                message="同步远端RAG中",
                progress=_progress(
                    "sync",
                    0,
                    len(remote_urls),
                    unit="remote_rag_nodes",
                    message="等待远端 RAG 节点",
                ),
            )
            remote_sync = _sync_remote_rag(job_id)
            current_stage = "sync_completed"
            _update_job(
                job_id,
                stage="sync_completed",
                message="同步完成",
                progress=_progress(
                    "sync_completed",
                    1,
                    1,
                    unit="remote_rag_nodes",
                    message="远端 RAG 同步完成",
                ),
            )
        current_stage = "completed"
        completion_message = "最终完成：纯净知识库底库已准备完成，大模型已拉起"
        if remote_sync["status"] == "completed":
            completion_message += "，远端 RAG 同步完成"
        elif remote_sync["status"] == "skipped":
            completion_message += "，远端 RAG 未配置"
        _update_job(
            job_id,
            status="completed",
            stage="completed",
            message=completion_message,
            finished_at=_now(),
            removed_files_count=len(removed_files),
            graph=graph_result,
            rag=rag_result,
            remote_rag=remote_sync,
            progress=_progress("completed", 1, 1, unit="job", message="全部处理完成"),
        )
    except Exception as exc:
        if current_stage == "sync":
            _update_job(job_id, message="远端 RAG 同步失败")
        _update_job(
            job_id,
            status="failed",
            stage="failed",
            message=str(exc),
            error=str(exc),
            finished_at=_now(),
            progress={
                **_progress("failed", 0, 1, unit="job", message="任务失败"),
                "error": str(exc),
            },
        )
    finally:
        if maintenance_requested and not maintenance_restored:
            try:
                _maintenance_request("restore")
            except Exception as restore_error:
                with state_lock:
                    current_error = str(
                        _read_state().get("jobs", {}).get(job_id, {}).get("error") or ""
                    )
                _update_job(
                    job_id,
                    message=f"任务结束后恢复大模型失败：{restore_error}",
                    error="; ".join(filter(None, (current_error, str(restore_error)))),
                )


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "source_root": str(SOURCE_ROOT),
        "pipeline_root": str(PIPELINES_ROOT),
        "documents": len(_snapshot()["documents"]),
    }


@app.get("/documents")
def documents(x_hdw_session: str | None = Header(default=None)) -> dict[str, Any]:
    require_user(x_hdw_session)
    state = _snapshot()
    items = sorted(
        state["documents"].values(),
        key=lambda item: (item.get("uploaded_at") or "", item.get("filename") or ""),
        reverse=True,
    )
    jobs = list(state.get("jobs", {}).values())
    latest_job = max(
        jobs,
        key=lambda item: item.get("updated_at") or item.get("created_at") or "",
        default=None,
    )
    active_job = max(
        (
            item
            for item in jobs
            if item.get("status") in {"queued", "running"}
        ),
        key=lambda item: item.get("updated_at") or item.get("created_at") or "",
        default=None,
    )
    active_view = _live_job(active_job) if active_job else None
    live_documents = (active_view or {}).get("progress", {}).get("documents", {})
    live_stage = (active_view or {}).get("progress", {}).get("stage", "")
    if live_documents:
        for item in items:
            progress = (
                live_documents.get(item.get("id"))
                or live_documents.get(item.get("filename"))
                or live_documents.get(f"{Path(str(item.get('filename', ''))).stem}.md")
                or live_documents.get(Path(str(item.get("filename", ""))).stem)
            )
            if progress:
                item["progress"] = {**progress, "stage": live_stage}
    latest_failed_job = max(
        (
            item
            for item in jobs
            if item.get("status") == "failed"
        ),
        key=lambda item: item.get("updated_at") or item.get("created_at") or "",
        default=None,
    )
    return {
        "documents": items,
        "active_job": active_view,
        "latest_job": _live_job(latest_job) if latest_job else None,
        "latest_failed_job": _live_job(latest_failed_job) if latest_failed_job else None,
    }


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    x_hdw_session: str | None = Header(default=None),
) -> dict[str, Any]:
    require_admin(x_hdw_session)
    filename = _safe_filename(file.filename)
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    target = SOURCE_ROOT / filename
    total = 0
    digest = hashlib.sha256()
    temporary = SOURCE_ROOT / f".{filename}.{uuid.uuid4().hex}.uploading"
    try:
        with temporary.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="file is too large")
                digest.update(chunk)
                output.write(chunk)
    except HTTPException:
        temporary.unlink(missing_ok=True)
        raise
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        await file.close()

    doc_id = _document_id(filename)
    try:
        with job_lock:
            state = _snapshot()
            active = next(
                (
                    job
                    for job in state["jobs"].values()
                    if job.get("status") in {"queued", "running"}
                ),
                None,
            )
            if active:
                raise HTTPException(status_code=409, detail=f"ingest already running: {active['id']}")
            temporary.replace(target)
            with state_lock:
                state = _read_state()
                state["documents"][doc_id] = {
                    "id": doc_id,
                    "filename": filename,
                    "source_format": target.suffix.lower().lstrip("."),
                    "size": total,
                    "sha256": digest.hexdigest(),
                    "uploaded": True,
                    "uploaded_at": _now(),
                    "parse_status": "pending",
                    "chunk_status": "pending",
                    "embedding_status": "pending",
                    "graph_status": "pending",
                    "rerank_status": "query_time",
                    "remote_sync_status": "pending",
                    "total_status": "uploaded",
                    "chunk_count": 0,
                    "error": "",
                }
                _write_state(state)
                return {"document": state["documents"][doc_id]}
    except HTTPException:
        temporary.unlink(missing_ok=True)
        raise
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ingest")
def ingest(
    req: IngestRequest,
    x_hdw_session: str | None = Header(default=None),
) -> dict[str, Any]:
    require_admin(x_hdw_session)
    with job_lock:
        state = _snapshot()
        active = next(
            (
                job
                for job in state["jobs"].values()
                if job.get("status") in {"queued", "running"}
            ),
            None,
        )
        if active:
            raise HTTPException(status_code=409, detail=f"ingest already running: {active['id']}")
        if req.full_rebuild:
            document_ids = [_document_id(filename) for filename in _scan_files()]
        else:
            document_ids = req.document_ids or list(state["documents"])
        document_ids = [doc_id for doc_id in document_ids if doc_id in state["documents"]]
        if not document_ids and not req.full_rebuild:
            raise HTTPException(status_code=400, detail="no knowledge documents")
        job_id = uuid.uuid4().hex
        state["jobs"][job_id] = {
            "id": job_id,
            "status": "queued",
            "stage": "queued",
            "message": "等待全量一致性重建" if req.full_rebuild else "等待入库任务",
            "document_ids": document_ids,
            "full_rebuild": req.full_rebuild,
            "created_at": _now(),
        }
        _write_state(state)
        thread = threading.Thread(
            target=_run_ingest,
            args=(job_id, document_ids),
            kwargs={"full_rebuild": req.full_rebuild},
            daemon=True,
        )
        thread.start()
    return {"job": state["jobs"][job_id]}


@app.post("/retry")
def retry_failed_ingest(x_hdw_session: str | None = Header(default=None)) -> dict[str, Any]:
    require_admin(x_hdw_session)
    with job_lock:
        state = _snapshot()
        active = next(
            (
                job
                for job in state["jobs"].values()
                if job.get("status") in {"queued", "running"}
            ),
            None,
        )
        if active:
            raise HTTPException(status_code=409, detail=f"ingest already running: {active['id']}")

        failed_jobs = [
            job for job in state.get("jobs", {}).values() if job.get("status") == "failed"
        ]
        previous = max(
            failed_jobs,
            key=lambda item: item.get("updated_at") or item.get("created_at") or "",
            default=None,
        )
        if not previous:
            raise HTTPException(status_code=404, detail="no failed ingest job to retry")

        full_rebuild = bool(previous.get("full_rebuild"))
        if full_rebuild:
            document_ids = [_document_id(filename) for filename in _scan_files()]
        else:
            document_ids = [
                document_id
                for document_id in previous.get("document_ids", [])
                if document_id in state["documents"]
            ]
        if not document_ids:
            raise HTTPException(status_code=400, detail="no knowledge documents to retry")

        job_id = uuid.uuid4().hex
        retry_job = {
            "id": job_id,
            "status": "queued",
            "stage": "queued",
            "message": "等待失败任务重试（复用已完成的 MinerU 解析）",
            "document_ids": document_ids,
            "full_rebuild": full_rebuild,
            "retry_of": previous["id"],
            "created_at": _now(),
        }
        state["jobs"][job_id] = retry_job
        _write_state(state)
        thread = threading.Thread(
            target=_run_ingest,
            args=(job_id, document_ids),
            kwargs={"full_rebuild": full_rebuild, "reuse_mineru_cache": True},
            daemon=True,
        )
        thread.start()
    return {"job": retry_job}


@app.post("/documents/reset")
def reset_documents(x_hdw_session: str | None = Header(default=None)) -> dict[str, Any]:
    require_admin(x_hdw_session)
    with job_lock:
        state = _snapshot()
        active = next(
            (
                job
                for job in state["jobs"].values()
                if job.get("status") in {"queued", "running"}
            ),
            None,
        )
        if active:
            raise HTTPException(status_code=409, detail=f"ingest already running: {active['id']}")
        job_id = uuid.uuid4().hex
        reset_job = {
            "id": job_id,
            "status": "queued",
            "stage": "queued",
            "message": "等待清空知识库",
            "document_ids": [],
            "full_reset": True,
            "created_at": _now(),
        }
        state["documents"] = {}
        state["jobs"] = {job_id: reset_job}
        _write_state(state)
        thread = threading.Thread(target=_run_reset, args=(job_id,), daemon=True)
        thread.start()
    return {"job": reset_job}


@app.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
    x_hdw_session: str | None = Header(default=None),
) -> dict[str, Any]:
    require_admin(x_hdw_session)
    with job_lock:
        state = _snapshot()
        active = next(
            (
                job
                for job in state["jobs"].values()
                if job.get("status") in {"queued", "running"}
            ),
            None,
        )
        if active:
            raise HTTPException(status_code=409, detail=f"ingest already running: {active['id']}")
        item = state["documents"].get(document_id)
        if not item:
            raise HTTPException(status_code=404, detail="document not found")
        filename = str(item.get("filename") or "")
        if not filename:
            raise HTTPException(status_code=400, detail="document has no source filename")
        job_id = uuid.uuid4().hex
        state["documents"][document_id].update(
            total_status="deleting",
            delete_status="pending",
            error="",
        )
        state["jobs"][job_id] = {
            "id": job_id,
            "status": "queued",
            "stage": "queued",
            "message": f"等待删除并重建：{filename}",
            "document_ids": [document_id],
            "full_rebuild": True,
            "delete_document_id": document_id,
            "created_at": _now(),
        }
        _write_state(state)
        thread = threading.Thread(
            target=_run_ingest,
            args=(job_id, [document_id]),
            kwargs={
                "full_rebuild": True,
                "excluded_filenames": {filename},
                "deleted_document_id": document_id,
            },
            daemon=True,
        )
        thread.start()
    return {"job": state["jobs"][job_id]}


@app.delete("/documents/{document_id}/staged")
def discard_staged_document(
    document_id: str,
    x_hdw_session: str | None = Header(default=None),
) -> dict[str, Any]:
    """Remove an uploaded file that has not entered an ingest job."""
    require_admin(x_hdw_session)
    with job_lock:
        state = _snapshot()
        active = next(
            (
                job
                for job in state["jobs"].values()
                if job.get("status") in {"queued", "running"}
            ),
            None,
        )
        if active:
            raise HTTPException(status_code=409, detail=f"ingest already running: {active['id']}")
        item = state["documents"].get(document_id)
        if not item:
            raise HTTPException(status_code=404, detail="document not found")
        if (
            not item.get("uploaded")
            or item.get("total_status") != "uploaded"
            or any(
                item.get(field) not in {None, "", "pending", "unknown"}
                for field in ("parse_status", "chunk_status", "embedding_status", "graph_status")
            )
        ):
            raise HTTPException(
                status_code=409,
                detail="only uploaded documents that have not entered ingest can be discarded",
            )
        filename = str(item.get("filename") or "")
        if not filename:
            raise HTTPException(status_code=400, detail="document has no source filename")
        _remove_document_artifacts(filename)
        with state_lock:
            state = _read_state()
            state["documents"].pop(document_id, None)
            _write_state(state)
    return {"deleted": True, "document_id": document_id, "filename": filename}


@app.get("/jobs/{job_id}")
def job(
    job_id: str,
    x_hdw_session: str | None = Header(default=None),
) -> dict[str, Any]:
    require_user(x_hdw_session)
    state = _snapshot()
    item = state["jobs"].get(job_id)
    if not item:
        raise HTTPException(status_code=404, detail="job not found")
    return {"job": _live_job(item)}


if __name__ == "__main__":
    print("knowledge ingest api")
