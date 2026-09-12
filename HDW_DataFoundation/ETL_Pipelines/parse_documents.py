#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


TEXT_EXTENSIONS = {".md", ".markdown"}
MINERU_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".docx", ".pptx", ".xlsx"}
DEFAULT_BACKEND = os.getenv("HDW_MINERU_BACKEND", "pipeline")
DEFAULT_MINERU_BASE_URL = os.getenv("HDW_MINERU_BASE_URL", "http://127.0.0.1:8002")
DEFAULT_LANG = os.getenv("HDW_MINERU_LANG", "ch")
REUSE_MINERU_CACHE = os.getenv("HDW_MINERU_REUSE_CACHE", "").lower() in {"1", "true", "yes"}
MINERU_API_OUTPUT_ROOT = Path(
    os.getenv("HDW_MINERU_API_OUTPUT_ROOT", "/data/mineru/api_output")
)
_FILE_HASHES: dict[Path, str] = {}


def _write_progress(
    path: Path | None,
    *,
    completed: int,
    total: int,
    current_document: str = "",
    documents: dict[str, dict[str, object]] | None = None,
    message: str = "",
) -> None:
    if not path:
        return
    payload = {
        "stage": "parsing",
        "completed": completed,
        "total": total,
        "stage_percent": round(completed / total * 100, 2) if total else 100.0,
        "current_document": current_document,
        "documents": documents or {},
        "message": message,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _truncate_utf8(value: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    data = value.encode("utf-8")
    if len(data) <= max_bytes:
        return value
    data = data[:max_bytes]
    while data:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            data = data[:exc.start]
    return ""


def _normalize_upload_name(name: str) -> str:
    path = Path(name).name
    stem = _truncate_utf8(Path(path).stem, 200)
    return f"{stem}{Path(path).suffix}"


def _is_supported(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS | MINERU_EXTENSIONS


def _iter_sources(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.rglob("*") if p.is_file() and _is_supported(p))


def _relative_target(source: Path, input_root: Path, output_root: Path) -> Path:
    relative = source.name if input_root.is_file() else source.relative_to(input_root)
    relative_path = Path(relative)
    if relative_path.suffix.lower() in {".md", ".markdown"}:
        relative_path = relative_path.with_suffix(".md")
    else:
        relative_path = relative_path.with_suffix(".md")
    return output_root / relative_path


def _multipart_body(fields: dict[str, str], file_field: str, filename: str, content_type: str, payload: bytes) -> tuple[str, bytes]:
    boundary = f"----HDWMinerU{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for key, value in fields.items():
        parts.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    parts.extend(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'.encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            payload,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return f"multipart/form-data; boundary={boundary}", b"".join(parts)


def _post_mineru(source: Path) -> dict[str, object]:
    normalized_name = _normalize_upload_name(source.name)
    result_key = Path(normalized_name).stem
    content_type = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
    fields = {
        "backend": DEFAULT_BACKEND,
        "lang_list": DEFAULT_LANG,
        "parse_method": "auto",
        "return_md": "true",
        "return_middle_json": "false",
        "return_model_output": "false",
        "return_content_list": "false",
        "return_images": "false",
        "response_format_zip": "false",
        "return_original_file": "false",
    }
    body_type, body = _multipart_body(fields, "files", normalized_name, content_type, source.read_bytes())
    request = urllib.request.Request(
        f"{DEFAULT_MINERU_BASE_URL.rstrip('/')}/file_parse",
        data=body,
        headers={"Content-Type": body_type},
        method="POST",
    )
    try:
        # MinerU may spend many minutes on large OCR PDFs; do not impose a client deadline.
        with urllib.request.urlopen(request) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"MinerU parse failed for {source}: HTTP {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"MinerU parse unavailable at {DEFAULT_MINERU_BASE_URL}: {exc}") from exc

    result = json.loads(payload)
    results = result.get("results") or {}
    parsed = results.get(result_key) or results.get(Path(source.name).stem) or {}
    md_content = parsed.get("md_content")
    if not isinstance(md_content, str) or not md_content.strip():
        raise RuntimeError(f"MinerU returned no markdown for {source}")
    return result


def _sha256(path: Path) -> str:
    cached = _FILE_HASHES.get(path)
    if cached:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    _FILE_HASHES[path] = value
    return value


def _reused_mineru_markdown(source: Path) -> Path | None:
    if not REUSE_MINERU_CACHE or not MINERU_API_OUTPUT_ROOT.is_dir():
        return None

    normalized_name = _normalize_upload_name(source.name)
    source_size = source.stat().st_size
    source_hash = _sha256(source)
    for upload in MINERU_API_OUTPUT_ROOT.glob("*/uploads/*"):
        if (
            not upload.is_file()
            or upload.name != normalized_name
            or upload.stat().st_size != source_size
            or _sha256(upload) != source_hash
        ):
            continue
        request_root = upload.parent.parent
        candidates = [
            path
            for path in request_root.rglob("*.md")
            if path.is_file() and path.name == f"{Path(normalized_name).stem}.md" and path.stat().st_size > 0
        ]
        candidates.sort(key=lambda path: ("/auto/" not in f"/{path.relative_to(request_root)}/", str(path)))
        if candidates:
            return candidates[0]
    return None


def _prepare_output(output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def _write_text(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def parse_tree(input_root: Path, output_root: Path, progress_path: Path | None = None) -> list[Path]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    if input_root == output_root:
        raise ValueError("input and output directories must be different")
    _prepare_output(output_root)
    sources = _iter_sources(input_root)
    progress_documents = {
        source.name: {"status": "pending", "percent": 0}
        for source in sources
    }
    _write_progress(
        progress_path,
        completed=0,
        total=len(sources),
        documents=progress_documents,
        message="等待 MinerU 解析",
    )
    written: list[Path] = []
    for index, source in enumerate(sources, start=1):
        progress_documents[source.name] = {"status": "running", "percent": 0}
        _write_progress(
            progress_path,
            completed=index - 1,
            total=len(sources),
            current_document=source.name,
            documents=progress_documents,
            message=f"正在解析：{source.name}",
        )
        target = _relative_target(source, input_root, output_root)
        if source.suffix.lower() in TEXT_EXTENSIONS:
            shutil.copy2(source, target)
            written.append(target)
        else:
            reused = _reused_mineru_markdown(source)
            if reused:
                shutil.copy2(reused, target)
                written.append(target)
                print(f"reused MinerU output: {source.name}")
            else:
                result = _post_mineru(source)
                normalized_name = _normalize_upload_name(source.name)
                parsed = result.get("results", {}).get(Path(normalized_name).stem) or result.get("results", {}).get(source.stem) or {}
                md_content = parsed["md_content"]
                _write_text(target, md_content)
                _write_text(target.with_suffix(".mineru.json"), json.dumps(result, ensure_ascii=False, indent=2))
                written.append(target)
        progress_documents[source.name] = {"status": "completed", "percent": 100}
        _write_progress(
            progress_path,
            completed=index,
            total=len(sources),
            current_document=source.name,
            documents=progress_documents,
            message=f"解析完成：{source.name}",
        )
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--progress-file")
    args = parser.parse_args()

    written = parse_tree(
        Path(args.input),
        Path(args.output),
        Path(args.progress_file) if args.progress_file else None,
    )
    print(f"parsed {len(written)} documents to {args.output}")


def _self_check() -> None:
    assert _normalize_upload_name("abc.pdf") == "abc.pdf"
    assert Path(_normalize_upload_name("abc.pdf")).stem == "abc"
    assert _relative_target(Path("/in/a/b.pdf"), Path("/in"), Path("/out")) == Path("/out/a/b.md")
    body_type, body = _multipart_body({"a": "1"}, "files", "x.pdf", "application/pdf", b"abc")
    assert body_type.startswith("multipart/form-data; boundary=")
    assert b'filename="x.pdf"' in body


if __name__ == "__main__":
    _self_check()
    main()
