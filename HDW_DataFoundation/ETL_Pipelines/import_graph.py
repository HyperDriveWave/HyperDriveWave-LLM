from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any


try:
    ROOT = Path(__file__).resolve().parents[2]
except IndexError:
    # Mounted ETL scripts may live at /pipelines instead of the project tree.
    ROOT = Path.cwd()
NEO4J_HTTP_URL = os.getenv(
    "HDW_NEO4J_HTTP_URL",
    "http://127.0.0.1:7474/db/neo4j/tx/commit",
)
NEO4J_AUTH = os.getenv("NEO4J_AUTH", "neo4j/change_me")
ENTITY_ALIASES_PATH = Path(
    os.getenv("HDW_ENTITY_ALIASES_PATH", str(ROOT / "HDW_KnowledgeGraph/entity_aliases.json"))
)
REVIEW_PATH = Path(
    os.getenv("HDW_GRAPH_REVIEW_PATH", str(ROOT / "HDW_Runtime/graph_review/entities_review.jsonl"))
)
EQUIPMENT_RE = re.compile(
    r"[\u4e00-\u9fa5A-Za-z0-9#、（）()/-]{1,24}"
    r"(?:系统|调压站|调压器|燃气轮机|汽轮机|余热锅炉|发电机|阀组|阀|泵|风机|轴承|汽包|凝汽器|压气机|燃烧室|透平|盘车装置|人孔门|母线|变压器)"
)
PARAMETER_RE = re.compile(r"(?<![#A-Za-z0-9])[-+]?\d+(?:\.\d+)?\s*(?:MPa|kPa|MW|kW|V|A|℃|mm|r/min|%|小时|min|ppmvd|mg/Nm3)")
FAULT_WORDS = ("故障", "报警", "跳闸", "泄漏", "超时", "水击", "卡涩", "突跳", "不合格", "不能", "禁止")
ACTION_WORDS = ("应", "检查", "监视", "启动", "停止", "打开", "关闭", "投入", "退出", "切换", "调整", "控制", "确认", "处理")
LEADING_NOISE_RE = re.compile(r"^(?:特别是|则|采取|方式|尽快|恢复|防止|以防|应|需|要|注意|加强|及时|立即|缓慢|检查|监视|启动|停止|打开|关闭|投入|退出|切换|调整|控制|确认|若|如|当)+")
EQUIPMENT_SPLIT_RE = re.compile(r"[。；;！？?\n，,、]|(?:以及|并且|同时|如果|由于|导致|引起|发生|出现|通过|采取|方式|尽快|恢复|防止|以防|进行|期间|过程中|应|需|要|注意|检查|监视|加强|控制|切换|处理|进入|使用|向|和|与|或|及)")
EQUIPMENT_NOISE = ("应", "不能", "无", "没有", "已", "未", "允许", "延时", "报警", "故障", "信号", "时间", "低于", "后续", "有关", "情况", "变化", "达到", "完成", "所有", "方式", "失常", "产生")
EQUIPMENT_BAD_PREFIX_RE = re.compile(r"^(?:[/]|(?:19|20)\d{2}|\d+(?:s|S|秒|倍|台|取|r/min))")


def _write_progress(
    path: Path | None,
    *,
    completed: int,
    total: int,
    current_document_id: str = "",
    documents: dict[str, dict[str, object]] | None = None,
    stage_percent: float | None = None,
    message: str = "",
) -> None:
    if not path:
        return
    payload = {
        "stage": "graph",
        "completed": completed,
        "total": total,
        "stage_percent": (
            stage_percent
            if stage_percent is not None
            else round(completed / total * 100, 2) if total else 100.0
        ),
        "current_document_id": current_document_id,
        "documents": documents or {},
        "message": message,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _auth() -> tuple[str, str]:
    user, _, password = NEO4J_AUTH.partition("/")
    return user, password


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _run_cypher(statement: str, parameters: dict[str, Any] | None = None) -> None:
    payload = {"statements": [{"statement": statement, "parameters": parameters or {}}]}
    user, password = _auth()
    token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
    request = urllib.request.Request(
        NEO4J_HTTP_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Basic {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read().decode("utf-8")) from exc
    if data.get("errors"):
        raise RuntimeError(data["errors"])


def _section_id(document_id: str, section_path: list[str]) -> str:
    key = "\n".join([document_id, *section_path])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _node_id(kind: str, name: str) -> str:
    return hashlib.sha1(f"{kind}:{name}".encode("utf-8")).hexdigest()[:16]


def _entity_key(name: str) -> str:
    return re.sub(r"[\s（）()、,，:：/-]+", "", name)


@lru_cache(maxsize=1)
def _aliases() -> dict[str, dict[str, str]]:
    if not ENTITY_ALIASES_PATH.exists():
        return {}
    raw = json.loads(ENTITY_ALIASES_PATH.read_text(encoding="utf-8"))
    aliases: dict[str, dict[str, str]] = {}
    for label, items in raw.items():
        label_aliases = {}
        for canonical, names in items.items():
            for name in [canonical, *names]:
                label_aliases[_entity_key(str(name))] = str(canonical)
        aliases[label] = label_aliases
    return aliases


def _normalize(label: str, name: str, base_confidence: float) -> dict[str, Any]:
    key = _entity_key(name)
    alias_map = _aliases().get(label, {})
    canonical = alias_map.get(key)
    if not canonical:
        matches = [alias for alias in alias_map if len(alias) >= 3 and alias in key]
        canonical = alias_map[max(matches, key=len)] if matches else name
    confidence = 0.95 if canonical != name or _entity_key(canonical) in alias_map else base_confidence
    status = "verified" if confidence >= 0.9 else "candidate" if confidence >= 0.7 else "needs_review"
    return {
        "id": _node_id(label, canonical),
        "name": canonical,
        "observed_name": name,
        "confidence": confidence,
        "status": status,
    }


def _sentences(text: str) -> list[str]:
    parts = re.split(r"[。；;！？?\n]+", text)
    return [re.sub(r"\s+", " ", part).strip(" -—:：，,、") for part in parts if part.strip()]


def _equipment_names(text: str, section_path: list[str]) -> list[str]:
    names = set()
    for title in section_path:
        if any(word in title for word in ("系统", "设备", "燃气轮机", "汽轮机", "余热锅炉", "发电机", "调压站")):
            names.add(title)
    for clause in EQUIPMENT_SPLIT_RE.split(text):
        for match in EQUIPMENT_RE.findall(clause):
            name = re.sub(r"^[（(]?\d+[）)、.]*", "", match)
            name = name.rsplit("且", 1)[-1].rsplit("延时", 1)[-1].rsplit("（", 1)[-1].rsplit("(", 1)[-1]
            name = LEADING_NOISE_RE.sub("", name).strip(" -—:：，,、）)")
            if (
                2 <= len(name) <= 24
                and not any(word in name for word in EQUIPMENT_NOISE)
                and not EQUIPMENT_BAD_PREFIX_RE.match(name)
            ):
                names.add(name)
    return sorted(names)[:12]


def _parameters(text: str) -> list[dict[str, str]]:
    values = []
    for raw in PARAMETER_RE.findall(text):
        value = re.sub(r"\s+", "", raw)
        match = re.match(r"([-+]?\d+(?:\.\d+)?)(.+)", value)
        if match:
            values.append({"name": value, "value": match.group(1), "unit": match.group(2)})
    return values[:12]


def _faults(sentences: list[str]) -> list[str]:
    return [sentence[:120] for sentence in sentences if any(word in sentence for word in FAULT_WORDS)][:8]


def _actions(sentences: list[str]) -> list[str]:
    return [sentence[:160] for sentence in sentences if any(word in sentence for word in ACTION_WORDS)][:8]


def _entities(row: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    text = str(row.get("text", ""))
    section_path = [str(item) for item in row.get("section_path", [])]
    sentences = _sentences(text)
    equipment = [
        {**_normalize("Equipment", name, 0.72 if len(name) <= 12 else 0.62), "chunk_id": row["chunk_id"]}
        for name in _equipment_names(text, section_path)
    ]
    parameters = [
        {
            **_normalize("Parameter", item["name"], 0.9),
            "chunk_id": row["chunk_id"],
            "value": item["value"],
            "unit": item["unit"],
        }
        for item in _parameters(text)
    ]
    faults = [{**_normalize("Fault", name, 0.75), "chunk_id": row["chunk_id"]} for name in _faults(sentences)]
    alarms = [
        {**_normalize("Alarm", name, 0.85), "chunk_id": row["chunk_id"]}
        for name in _faults(sentences)
        if "报警" in name or "保护" in name or "跳闸" in name
    ][:8]
    actions = [{**_normalize("Action", name, 0.8), "chunk_id": row["chunk_id"]} for name in _actions(sentences)]
    return {
        "equipment": equipment,
        "parameters": parameters,
        "faults": faults,
        "alarms": alarms,
        "actions": actions,
    }


def _build_graph(
    rows: list[dict[str, Any]],
    progress_path: Path | None = None,
) -> dict[str, Any]:
    documents: dict[str, dict[str, Any]] = {}
    sections: dict[str, dict[str, Any]] = {}
    chunk_rows: list[dict[str, Any]] = []
    chunk_chain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    equipment: dict[str, dict[str, str]] = {}
    parameters: dict[str, dict[str, str]] = {}
    faults: dict[str, dict[str, str]] = {}
    alarms: dict[str, dict[str, str]] = {}
    actions: dict[str, dict[str, str]] = {}
    entity_edges: dict[str, list[dict[str, str]]] = defaultdict(list)
    review_rows: list[dict[str, Any]] = []
    document_totals: dict[str, int] = defaultdict(int)
    for row in rows:
        document_totals[str(row["document_id"])] += 1
    progress_documents = {
        document_id: {"status": "pending", "percent": 0}
        for document_id in document_totals
    }
    processed: dict[str, int] = defaultdict(int)
    completed_documents: set[str] = set()
    _write_progress(
        progress_path,
        completed=0,
        total=len(document_totals),
        documents=progress_documents,
        message="分析知识图谱结构",
    )

    for row in rows:
        document_id = str(row["document_id"])
        document_title = str(row.get("document_title") or row.get("section_title") or document_id)
        source_file = str(row.get("source_file", ""))
        source_format = str(row.get("source_format", ""))
        security_level = str(row.get("security_level", "internal"))
        section_path = [str(item) for item in row.get("section_path", [])]
        section_title = str(row.get("section_title") or document_title)
        section_key = _section_id(document_id, section_path) if section_path else ""
        chunk_id = str(row["chunk_id"])
        chunk_index = int(row.get("chunk_index", 0))
        processed[document_id] += 1
        progress_documents[document_id] = {
            "status": "running",
            "percent": round(processed[document_id] / document_totals[document_id] * 100, 2),
        }
        if processed[document_id] >= document_totals[document_id]:
            progress_documents[document_id] = {"status": "ready", "percent": 100}
            completed_documents.add(document_id)
        if processed[document_id] % 50 == 0 or processed[document_id] >= document_totals[document_id]:
            _write_progress(
                progress_path,
                completed=len(completed_documents),
                total=len(document_totals),
                current_document_id=document_id,
                documents=progress_documents,
                message=f"分析图谱：{document_title}",
            )

        documents[document_id] = {
            "id": document_id,
            "title": document_title,
            "source_file": source_file,
            "source_format": source_format,
            "security_level": security_level,
        }

        if section_path:
            for depth in range(1, len(section_path) + 1):
                prefix = section_path[:depth]
                sid = _section_id(document_id, prefix)
                sections[sid] = {
                    "id": sid,
                    "document_id": document_id,
                    "title": prefix[-1],
                    "path": prefix,
                    "parent_id": _section_id(document_id, prefix[:-1]) if depth > 1 else "",
                    "depth": depth,
                }

        chunk_rows.append(
            {
                "id": chunk_id,
                "document_id": document_id,
                "section_id": section_key,
                "section_path": section_path,
                "section_title": section_title,
                "chunk_index": chunk_index,
                "source_file": source_file,
                "security_level": security_level,
                "text": str(row.get("text", "")),
            }
        )
        chunk_chain[document_id].append({"id": chunk_id, "chunk_index": chunk_index})
        extracted = _entities({**row, "chunk_id": chunk_id, "section_path": section_path})
        for key, bucket in (
            ("equipment", equipment),
            ("faults", faults),
            ("alarms", alarms),
            ("actions", actions),
        ):
            for item in extracted[key]:
                bucket[item["id"]] = {
                    "id": item["id"],
                    "name": item["name"],
                    "observed_name": item["observed_name"],
                    "confidence": item["confidence"],
                    "status": item["status"],
                }
                entity_edges[key].append({"chunk_id": chunk_id, "entity_id": item["id"]})
                if item["status"] == "needs_review":
                    review_rows.append(
                        {
                            "label": key,
                            "name": item["name"],
                            "observed_name": item["observed_name"],
                            "chunk_id": chunk_id,
                            "section_path": section_path,
                        }
                    )
        for item in extracted["parameters"]:
            parameters[item["id"]] = {
                "id": item["id"],
                "name": item["name"],
                "value": item["value"],
                "unit": item["unit"],
                "observed_name": item["observed_name"],
                "confidence": item["confidence"],
                "status": item["status"],
            }
            entity_edges["parameters"].append({"chunk_id": chunk_id, "entity_id": item["id"]})

    next_edges = []
    for document_id, items in chunk_chain.items():
        ordered = sorted(items, key=lambda item: item["chunk_index"])
        for left, right in zip(ordered, ordered[1:]):
            next_edges.append({"from": left["id"], "to": right["id"]})

    section_edges = [
        {"parent_id": section["parent_id"], "child_id": section["id"]}
        for section in sections.values()
        if section["parent_id"]
    ]

    return {
        "documents": list(documents.values()),
        "sections": list(sections.values()),
        "chunks": chunk_rows,
        "section_edges": section_edges,
        "next_edges": next_edges,
        "equipment": list(equipment.values()),
        "parameters": list(parameters.values()),
        "faults": list(faults.values()),
        "alarms": list(alarms.values()),
        "actions": list(actions.values()),
        "entity_edges": entity_edges,
        "review_rows": review_rows,
    }


def import_graph(
    rows: list[dict[str, Any]],
    replace_all: bool = False,
    progress_path: Path | None = None,
) -> dict[str, int]:
    graph = _build_graph(rows, progress_path)
    document_ids = [row["id"] for row in graph["documents"]]
    progress_documents = {
        document_id: {"status": "writing", "percent": 100}
        for document_id in document_ids
    }
    _write_progress(
        progress_path,
        completed=len(document_ids),
        total=len(document_ids),
        documents=progress_documents,
        stage_percent=90,
        message="正在写入 Neo4j 图谱",
    )

    _run_cypher(
        "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE"
    )
    _run_cypher(
        "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (n:Section) REQUIRE n.id IS UNIQUE"
    )
    _run_cypher("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.id IS UNIQUE")
    _run_cypher(
        "CREATE CONSTRAINT equipment_id IF NOT EXISTS FOR (n:Equipment) REQUIRE n.id IS UNIQUE"
    )
    _run_cypher("CREATE CONSTRAINT fault_id IF NOT EXISTS FOR (n:Fault) REQUIRE n.id IS UNIQUE")
    _run_cypher("CREATE CONSTRAINT alarm_id IF NOT EXISTS FOR (n:Alarm) REQUIRE n.id IS UNIQUE")
    _run_cypher("CREATE CONSTRAINT action_id IF NOT EXISTS FOR (n:Action) REQUIRE n.id IS UNIQUE")
    _run_cypher(
        "CREATE CONSTRAINT parameter_id IF NOT EXISTS FOR (n:Parameter) REQUIRE n.id IS UNIQUE"
    )

    if replace_all:
        _run_cypher(
            """
            MATCH (n)
            WHERE n:Document OR n:Section OR n:Chunk OR n:Equipment OR n:Fault OR n:Alarm OR n:Action OR n:Parameter
            DETACH DELETE n
            """
        )
    elif document_ids:
        _run_cypher(
            """
            MATCH (d:Document)
            WHERE d.id IN $document_ids
            OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
            DETACH DELETE c
            """,
            {"document_ids": document_ids},
        )
        _run_cypher(
            """
            MATCH (d:Document)
            WHERE d.id IN $document_ids
            OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
            DETACH DELETE s
            """,
            {"document_ids": document_ids},
        )
        _run_cypher(
            """
            MATCH (d:Document)
            WHERE d.id IN $document_ids
            DETACH DELETE d
            """,
            {"document_ids": document_ids},
        )

    if graph["documents"]:
        _run_cypher(
            """
            UNWIND $rows AS row
            MERGE (d:Document {id: row.id})
            SET d.title = row.title,
                d.source_file = row.source_file,
                d.source_format = row.source_format,
                d.security_level = row.security_level
            """,
            {"rows": graph["documents"]},
        )

    if graph["sections"]:
        _run_cypher(
            """
            UNWIND $rows AS row
            MATCH (d:Document {id: row.document_id})
            MERGE (s:Section {id: row.id})
            SET s.title = row.title,
                s.path = row.path,
                s.depth = row.depth
            MERGE (d)-[:HAS_SECTION]->(s)
            """,
            {"rows": graph["sections"]},
        )

    if graph["section_edges"]:
        _run_cypher(
            """
            UNWIND $rows AS row
            MATCH (p:Section {id: row.parent_id})
            MATCH (c:Section {id: row.child_id})
            MERGE (p)-[:HAS_SUBSECTION]->(c)
            """,
            {"rows": graph["section_edges"]},
        )

    if graph["chunks"]:
        _run_cypher(
            """
            UNWIND $rows AS row
            MATCH (d:Document {id: row.document_id})
            MERGE (c:Chunk {id: row.id})
            SET c.text = row.text,
                c.index = row.chunk_index,
                c.source_file = row.source_file,
                c.section_title = row.section_title,
                c.section_path = row.section_path,
                c.security_level = row.security_level
            MERGE (d)-[:HAS_CHUNK]->(c)
            """,
            {"rows": graph["chunks"]},
        )

    section_chunks = [row for row in graph["chunks"] if row["section_id"]]
    if section_chunks:
        _run_cypher(
            """
            UNWIND $rows AS row
            MATCH (s:Section {id: row.section_id})
            MATCH (c:Chunk {id: row.id})
            MERGE (s)-[:HAS_CHUNK]->(c)
            """,
            {"rows": section_chunks},
        )

    if graph["next_edges"]:
        _run_cypher(
            """
            UNWIND $rows AS row
            MATCH (a:Chunk {id: row.from})
            MATCH (b:Chunk {id: row.to})
            MERGE (a)-[:NEXT_CHUNK]->(b)
            """,
            {"rows": graph["next_edges"]},
        )

    if graph["equipment"]:
        _run_cypher(
            """
            UNWIND $rows AS row
            MERGE (e:Equipment {id: row.id})
            SET e.name = row.name,
                e.observed_name = row.observed_name,
                e.confidence = row.confidence,
                e.status = row.status
            """,
            {"rows": graph["equipment"]},
        )
    if graph["parameters"]:
        _run_cypher(
            """
            UNWIND $rows AS row
            MERGE (p:Parameter {id: row.id})
            SET p.name = row.name,
                p.value = row.value,
                p.unit = row.unit,
                p.observed_name = row.observed_name,
                p.confidence = row.confidence,
                p.status = row.status
            """,
            {"rows": graph["parameters"]},
        )
    if graph["faults"]:
        _run_cypher(
            """
            UNWIND $rows AS row
            MERGE (f:Fault {id: row.id})
            SET f.name = row.name,
                f.observed_name = row.observed_name,
                f.confidence = row.confidence,
                f.status = row.status
            """,
            {"rows": graph["faults"]},
        )
    if graph["alarms"]:
        _run_cypher(
            """
            UNWIND $rows AS row
            MERGE (a:Alarm {id: row.id})
            SET a.name = row.name,
                a.observed_name = row.observed_name,
                a.confidence = row.confidence,
                a.status = row.status
            """,
            {"rows": graph["alarms"]},
        )
    if graph["actions"]:
        _run_cypher(
            """
            UNWIND $rows AS row
            MERGE (a:Action {id: row.id})
            SET a.name = row.name,
                a.observed_name = row.observed_name,
                a.confidence = row.confidence,
                a.status = row.status
            """,
            {"rows": graph["actions"]},
        )

    edge_queries = {
        "equipment": ("Equipment", "MENTIONS"),
        "parameters": ("Parameter", "MENTIONS"),
        "faults": ("Fault", "EVIDENCE_OF"),
        "alarms": ("Alarm", "MENTIONS"),
        "actions": ("Action", "HAS_ACTION"),
    }
    for key, (label, relation) in edge_queries.items():
        rows = graph["entity_edges"][key]
        if rows:
            _run_cypher(
                f"""
                UNWIND $rows AS row
                MATCH (c:Chunk {{id: row.chunk_id}})
                MATCH (e:{label} {{id: row.entity_id}})
                MERGE (c)-[:{relation}]->(e)
                """,
                {"rows": rows},
            )

    REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REVIEW_PATH.open("w", encoding="utf-8") as fh:
        for row in graph["review_rows"]:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_progress(
        progress_path,
        completed=len(document_ids),
        total=len(document_ids),
        documents={
            document_id: {"status": "completed", "percent": 100}
            for document_id in document_ids
        },
        stage_percent=100,
        message="Neo4j 图谱构建完成",
    )

    return {
        "documents": len(graph["documents"]),
        "sections": len(graph["sections"]),
        "chunks": len(graph["chunks"]),
        "equipment": len(graph["equipment"]),
        "parameters": len(graph["parameters"]),
        "faults": len(graph["faults"]),
        "alarms": len(graph["alarms"]),
        "actions": len(graph["actions"]),
        "review": len(graph["review_rows"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--replace-all", action="store_true")
    parser.add_argument("--progress-file")
    args = parser.parse_args()
    stats = import_graph(
        _rows(Path(args.input)),
        replace_all=args.replace_all,
        progress_path=Path(args.progress_file) if args.progress_file else None,
    )
    print(json.dumps(stats, ensure_ascii=False))


def _self_check() -> None:
    sample = [
        {
            "document_id": "doc1",
            "document_title": "Doc",
            "chunk_id": "doc1-0000",
            "chunk_index": 0,
            "source_file": "a.md",
            "source_format": "md",
            "security_level": "internal",
            "section_path": ["A", "B"],
            "section_title": "B",
            "text": "燃气轮机点火后，应检查天然气系统压力3.5MPa，无泄漏报警。",
        },
        {
            "document_id": "doc1",
            "document_title": "Doc",
            "chunk_id": "doc1-0001",
            "chunk_index": 1,
            "source_file": "a.md",
            "source_format": "md",
            "security_level": "internal",
            "section_path": ["A", "B"],
            "section_title": "B",
            "text": "若盘车装置不能自动脱开时，应立即紧急停机。",
        },
    ]
    graph = _build_graph(sample)
    assert len(graph["documents"]) == 1
    assert len(graph["sections"]) == 2
    assert len(graph["chunks"]) == 2
    assert len(graph["next_edges"]) == 1
    assert graph["equipment"]
    assert graph["parameters"]
    assert graph["faults"]
    assert graph["actions"]
    assert any(item["name"] == "燃气系统" and item["status"] == "verified" for item in graph["equipment"])


if __name__ == "__main__":
    if os.getenv("HDW_SELF_CHECK") == "1":
        _self_check()
        print("graph import self-check passed")
    else:
        main()
