from __future__ import annotations

import argparse
import hashlib
import html
import json
import zipfile
import re
import xml.etree.ElementTree as ET
from pathlib import Path


MAX_CHARS = 1200
OVERLAP = 120
DOCX_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+?)(?:\s+\d+)?$")


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
        "stage": "chunking",
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


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"</(p|td|tr|table|h[1-6])>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def chunk_text(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def _numbered_heading(line: str) -> tuple[int, str] | None:
    if "|" in line:
        return None
    match = NUMBERED_HEADING_RE.match(line.replace("\t", " "))
    if not match:
        return None
    number, title = match.groups()
    title = re.sub(r"\s+\d+$", "", title).strip()
    if not title or len(title) > 80:
        return None
    return number.count(".") + 1, title


def _heading_paths(lines: list[str]) -> dict[str, list[str]]:
    paths: dict[str, list[str]] = {}
    stack: list[str] = []
    for line in lines:
        heading = _numbered_heading(line)
        if not heading:
            continue
        level, title = heading
        stack[:] = stack[: max(0, level - 1)]
        stack.append(title)
        paths[title] = stack.copy()
    return paths


def structured_blocks(text: str) -> list[dict[str, object]]:
    lines = clean_text(text).splitlines()
    blocks: list[dict[str, object]] = []
    heading_stack: list[str] = []
    current_lines: list[str] = []
    heading_paths = _heading_paths(lines)

    def flush() -> None:
        if not current_lines:
            return
        blocks.append(
            {
                "section_path": heading_stack.copy(),
                "text": "\n".join(current_lines).strip(),
            }
        )

    for line in lines:
        heading = HEADING_RE.match(line)
        if heading:
            flush()
            current_lines.clear()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            heading_stack[:] = heading_stack[: max(0, level - 1)]
            heading_stack.append(title)
            continue
        numbered = _numbered_heading(line)
        if numbered:
            flush()
            current_lines.clear()
            level, title = numbered
            heading_stack[:] = heading_stack[: max(0, level - 1)]
            heading_stack.append(title)
            continue
        # ponytail: this uses the document's table of contents; replace with MinerU layout headings if OCR style metadata is needed.
        if line in heading_paths:
            flush()
            current_lines.clear()
            heading_stack[:] = heading_paths[line]
            continue
        if line.strip():
            current_lines.append(line.strip())
        elif current_lines:
            current_lines.append("")

    flush()
    return blocks


def document_id(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]


def iter_sources(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(
        p for p in input_path.rglob("*") if p.suffix.lower() in {".md", ".markdown", ".docx"}
    )


def _docx_paragraph_text(paragraph: ET.Element) -> str:
    parts: list[str] = []
    for node in paragraph.iter():
        tag = node.tag.rsplit("}", 1)[-1]
        if tag == "t" and node.text:
            parts.append(node.text)
        elif tag == "tab":
            parts.append("\t")
        elif tag == "br":
            parts.append("\n")
    return "".join(parts).strip()


def _docx_style(paragraph: ET.Element) -> str:
    style = paragraph.find("./w:pPr/w:pStyle", DOCX_NS)
    return str(style.attrib.get(f"{{{DOCX_NS['w']}}}val", "")) if style is not None else ""


def extract_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))
    body = root.find("w:body", DOCX_NS)
    if body is None:
        return ""

    blocks: list[str] = []
    for child in list(body):
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            text = _docx_paragraph_text(child)
            if not text:
                continue
            style = _docx_style(child).lower()
            if style.startswith("heading"):
                level = re.search(r"\d+", style)
                prefix = "#" * max(1, int(level.group()) if level else 1)
                blocks.append(f"{prefix} {text}")
            else:
                blocks.append(text)
        elif tag == "tbl":
            rows: list[str] = []
            for row in child.findall(".//w:tr", DOCX_NS):
                cells = [_docx_paragraph_text(cell) for cell in row.findall("./w:tc", DOCX_NS)]
                row_text = " | ".join(cell for cell in cells if cell)
                if row_text:
                    rows.append(row_text)
            if rows:
                blocks.append("\n".join(rows))
    return "\n".join(blocks)


def build_records(
    input_path: Path,
    progress_path: Path | None = None,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    sources = iter_sources(input_path)
    progress_documents = {
        path.name: {"status": "pending", "percent": 0}
        for path in sources
    }
    _write_progress(
        progress_path,
        completed=0,
        total=len(sources),
        documents=progress_documents,
        message="等待文档切分",
    )
    for index, path in enumerate(sources, start=1):
        progress_documents[path.name] = {"status": "running", "percent": 0}
        _write_progress(
            progress_path,
            completed=index - 1,
            total=len(sources),
            current_document=path.name,
            documents=progress_documents,
            message=f"正在切分：{path.name}",
        )
        record_start = len(records)
        doc_id = document_id(path)
        if path.suffix.lower() == ".docx":
            text = extract_docx_text(path)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
        blocks = structured_blocks(text)
        document_title = (
            blocks[0]["section_path"][0]
            if blocks and blocks[0]["section_path"]
            else path.stem
        )
        chunk_index = 0
        for block_index, block in enumerate(blocks or [{"section_path": [], "text": text}]):
            block_text = str(block["text"]).strip()
            if not block_text:
                continue
            section_path = [str(item) for item in block["section_path"]]
            for chunk in chunk_text(block_text):
                records.append(
                    {
                        "document_id": doc_id,
                        "document_title": document_title,
                        "chunk_id": f"{doc_id}-{chunk_index:04d}",
                        "chunk_index": chunk_index,
                        "section_index": block_index,
                        "source_file": str(path),
                        "source_format": path.suffix.lower().lstrip("."),
                        "text": chunk,
                        "security_level": "internal",
                        "section_path": section_path,
                        "section_title": section_path[-1] if section_path else document_title,
                    }
                )
                chunk_index += 1
        progress_documents[path.name] = {
            "status": "completed",
            "percent": 100,
            "chunk_count": len(records) - record_start,
        }
        _write_progress(
            progress_path,
            completed=index,
            total=len(sources),
            current_document=path.name,
            documents=progress_documents,
            message=f"切分完成：{path.name}",
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--progress-file")
    args = parser.parse_args()

    records = build_records(
        Path(args.input),
        Path(args.progress_file) if args.progress_file else None,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} chunks to {output}")


def _self_check() -> None:
    chunks = chunk_text("a" * 1300, max_chars=1000, overlap=100)
    assert len(chunks) == 2
    assert chunks[1].startswith("a")
    assert clean_text("<p>调压器</p><td>3.8</td>") == "调压器\n3.8"
    blocks = structured_blocks("# A\nhello\n## B\nworld")
    assert blocks[0]["section_path"] == ["A"]
    assert blocks[1]["section_path"] == ["A", "B"]
    blocks = structured_blocks("1\t范围\t1\n1.1\t设备\t2\n范围\n正文\n设备\n参数")
    assert blocks[0]["section_path"] == ["范围"]
    assert blocks[1]["section_path"] == ["范围", "设备"]


if __name__ == "__main__":
    _self_check()
    main()
