from __future__ import annotations

import json
import hmac
import os
import shutil
import threading
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import zvec
from fastapi import File, FastAPI, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


BGE_M3_PATH = os.getenv("HDW_BGE_M3_PATH", "/models/RAG_Models/bge-m3")
RERANKER_PATH = os.getenv("HDW_RERANKER_PATH", "/models/RAG_Models/bge-reranker-v2-m3")
INDEX_PATH = Path(os.getenv("HDW_RAG_INDEX_PATH", "/data/rag/chunks.jsonl"))
ZVEC_PATH = Path(os.getenv("HDW_ZVEC_COLLECTION_PATH", "/data/zvec/industrial_chunks"))
REINDEX_PROGRESS_PATH = ZVEC_PATH.parent / "reindex_progress.json"
DEVICE = torch.device(os.getenv("HDW_RAG_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
EMBED_DIM = 1024
EMBED_MAX_LENGTH = int(os.getenv("HDW_EMBED_MAX_LENGTH", "1024"))
RERANK_MAX_LENGTH = int(os.getenv("HDW_RERANK_MAX_LENGTH", "512"))
RERANK_CANDIDATES = int(os.getenv("HDW_RERANK_CANDIDATES", "50"))
# ── 混合检索（稠密向量 + 全文）────────────────────────────────
# 只走稠密向量时有两类查询召不回来：
#   1. **任意标识符**——`P.GAS.11`、`02MBR20CP101XQ01` 这种，稠密向量对它
#      没有"语义"可言。实测「运维手册中 P.GAS.11 对应的值是多少」：
#      含答案的分块余弦 0.7302，而召回的块 0.8162~0.8585——差 0.09 就永远
#      进不了 top-40，重排连看都看不到它。
#   2. **与文档标题近乎逐字重合的问题**——实测「轴封系统投运的注意事项」，
#      纯向量 top-1 得分 4.89，而标题就叫《轴封系统启机前投运注意事项》的
#      两块压根没进候选；补上全文通道后这两块升到 6.19 / 6.16 并置顶。
#
# `content` 字段上**早就建好了 FTS 索引**（见 `_create_zvec_collection`），
# 只是从来没查过。这里补上全文通道，两路排名用 RRF 融合后仍交给交叉编码器定序。
# 关掉它就退回原来的纯向量行为（改 .env 即可，不用回滚镜像）。
_LEXICAL_ENABLED = os.getenv("HDW_RAG_LEXICAL", "true").lower() != "false"
# 融合后进入重排的候选数。默认与重排候选数一致——**不额外抬高**：
# 交叉编码器是整条检索链的瓶颈（实测每篇约 10 ms），抬一倍就多花一倍时间，
# 而实测融合后目标块的排名已经很靠前（见上）。
FUSED_CANDIDATES = int(os.getenv("HDW_RAG_FUSED_CANDIDATES", str(RERANK_CANDIDATES)))
# 全文通道的运行状态，供 /health 暴露。
# **回落必须是可见的**：集合建得早、没有 FTS 索引时，混合检索会静默退回
# 纯向量——那就等于这次改动没生效，而 /health 上一切正常。这个项目已经
# 栽过好几次「看着成功、实际没生效」的坑，所以状态要报出来。
_LEXICAL_STATE = "unknown" if _LEXICAL_ENABLED else "disabled"
_LEXICAL_FALLBACK_WARNED = False
ZVEC_EMBED_BATCH_SIZE = max(1, int(os.getenv("HDW_ZVEC_EMBED_BATCH_SIZE", "8")))
RAG_ADMIN_TOKEN = os.getenv("HDW_RAG_ADMIN_TOKEN", "").strip()
_ZVEC_INIT_DONE = False
# ponytail: process-wide locks are enough for P0 single-worker CPU service; use blue/green indexes before scaling writers.
_MODEL_LOCK = threading.Lock()
_INDEX_LOCK = threading.Lock()

app = FastAPI(title="HyperDriveWave RAG Service", version="0.2.0")


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1)
    normalize: bool = True


class EmbedResponse(BaseModel):
    vectors: list[list[float]]
    dim: int
    backend: str


class RerankDocument(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RerankRequest(BaseModel):
    query: str
    documents: list[RerankDocument]
    top_k: int = Field(default=40, ge=1, le=40)


class RerankResult(BaseModel):
    id: str
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=40, ge=1, le=40)


class ReindexRequest(BaseModel):
    job_id: str = ""


def _require_admin_token(token: str | None) -> None:
    if not RAG_ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="RAG admin token is not configured")
    if not token or not hmac.compare_digest(token, RAG_ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="invalid RAG admin token")


@lru_cache(maxsize=1)
def _embedder():
    tokenizer = AutoTokenizer.from_pretrained(BGE_M3_PATH, local_files_only=True)
    model = AutoModel.from_pretrained(BGE_M3_PATH, local_files_only=True).to(DEVICE)
    model.eval()
    return tokenizer, model


@lru_cache(maxsize=1)
def _reranker():
    tokenizer = AutoTokenizer.from_pretrained(RERANKER_PATH, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_PATH,
        local_files_only=True,
    ).to(DEVICE)
    model.eval()
    return tokenizer, model


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def _embed(texts: list[str], normalize: bool = True) -> list[list[float]]:
    with _MODEL_LOCK:
        tokenizer, model = _embedder()
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=EMBED_MAX_LENGTH,
            return_tensors="pt",
        ).to(DEVICE)
        with torch.inference_mode():
            pooled = _mean_pool(model(**encoded).last_hidden_state, encoded["attention_mask"])
            if normalize:
                pooled = F.normalize(pooled, p=2, dim=1)
        return pooled.cpu().float().tolist()


def _rerank(query: str, docs: list[RerankDocument], top_k: int) -> list[RerankResult]:
    if not docs:
        return []
    with _MODEL_LOCK:
        tokenizer, model = _reranker()
        pairs = [[query, doc.text] for doc in docs]
        encoded = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=RERANK_MAX_LENGTH,
            return_tensors="pt",
        ).to(DEVICE)
        with torch.inference_mode():
            scores = model(**encoded, return_dict=True).logits.view(-1).float().cpu().tolist()
    ranked = sorted(
        (
            RerankResult(id=doc.id, score=float(score), text=doc.text, metadata=doc.metadata)
            for doc, score in zip(docs, scores, strict=True)
        ),
        key=lambda item: item.score,
        reverse=True,
    )
    return ranked[:top_k]


def _chunk_rows() -> list[dict[str, Any]]:
    with INDEX_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _validate_index_file(path: Path) -> None:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid chunks.jsonl at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"invalid chunks.jsonl at line {line_number}: object required")


def _write_reindex_progress(
    job_id: str,
    *,
    stage: str,
    completed: int,
    total: int,
    current_document_id: str = "",
    current_document: str = "",
    documents: dict[str, dict[str, Any]] | None = None,
    stage_percent: float | None = None,
    message: str = "",
    error: str = "",
) -> None:
    if stage_percent is None:
        stage_percent = round(completed / total * 100, 2) if total else 100.0
    payload = {
        "job_id": job_id,
        "stage": stage,
        "completed": completed,
        "total": total,
        "unit": "chunks",
        "stage_percent": stage_percent,
        "current_document_id": current_document_id,
        "current_document": current_document,
        "documents": documents or {},
        "message": message,
        "error": error,
    }
    REINDEX_PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = REINDEX_PROGRESS_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(REINDEX_PROGRESS_PATH)


def _init_zvec() -> None:
    global _ZVEC_INIT_DONE
    if _ZVEC_INIT_DONE:
        return
    ZVEC_PATH.parent.mkdir(parents=True, exist_ok=True)
    zvec.init(log_dir=str(ZVEC_PATH.parent / "logs"))
    _ZVEC_INIT_DONE = True


def _create_zvec_collection():
    schema = zvec.CollectionSchema(
        name="industrial_chunks",
        fields=[
            zvec.FieldSchema("document_id", zvec.DataType.STRING, nullable=False),
            zvec.FieldSchema("source_file", zvec.DataType.STRING, nullable=False),
            zvec.FieldSchema("security_level", zvec.DataType.STRING, nullable=False),
            zvec.FieldSchema("chunk_index", zvec.DataType.INT64, nullable=False),
            zvec.FieldSchema(
                "content",
                zvec.DataType.STRING,
                nullable=False,
                index_param=zvec.FtsIndexParam(tokenizer_name="standard", filters=["lowercase"]),
            ),
        ],
        vectors=[
            zvec.VectorSchema(
                "embedding",
                zvec.DataType.VECTOR_FP32,
                dimension=EMBED_DIM,
                index_param=zvec.HnswIndexParam(),
            )
        ],
    )
    return zvec.create_and_open(str(ZVEC_PATH), schema=schema)


def _open_zvec_collection(rebuild: bool = False):
    _init_zvec()
    if rebuild and ZVEC_PATH.exists():
        shutil.rmtree(ZVEC_PATH)
    if ZVEC_PATH.exists():
        return zvec.open(str(ZVEC_PATH))
    return _create_zvec_collection()


def _reindex_zvec(job_id: str = "") -> dict[str, Any]:
    rows = _chunk_rows()
    coll = _open_zvec_collection(rebuild=True)
    if not rows:
        _write_reindex_progress(
            job_id,
            stage="completed",
            completed=0,
            total=0,
            stage_percent=100,
            message="Zvec 索引为空",
        )
        coll.flush()
        return {"backend": "zvec+bge-m3", "indexed": 0, "collection_path": str(ZVEC_PATH)}
    totals: dict[str, int] = {}
    names: dict[str, str] = {}
    for row in rows:
        document_id = str(row.get("document_id", ""))
        totals[document_id] = totals.get(document_id, 0) + 1
        names.setdefault(document_id, Path(str(row.get("source_file", ""))).stem)
    completed_by_document = {document_id: 0 for document_id in totals}
    progress_documents = {
        document_id: {
            "status": "pending",
            "percent": 0,
            "completed": 0,
            "total": total,
            "filename": names[document_id],
        }
        for document_id, total in totals.items()
    }
    _write_reindex_progress(
        job_id,
        stage="embedding",
        completed=0,
        total=len(rows),
        documents=progress_documents,
        message="开始 BGE-M3 嵌入",
    )
    indexed = 0
    for start in range(0, len(rows), ZVEC_EMBED_BATCH_SIZE):
        batch = rows[start : start + ZVEC_EMBED_BATCH_SIZE]
        texts = [str(row.get("text", "")) for row in batch]
        vectors = _embed(texts)
        docs = [
            zvec.Doc(
                id=str(row.get("chunk_id") or row.get("id")),
                fields={
                    "document_id": str(row.get("document_id", "")),
                    "source_file": str(row.get("source_file", "")),
                    "security_level": str(row.get("security_level", "internal")),
                    "chunk_index": int(row.get("chunk_index", 0)),
                    "content": text,
                },
                vectors={"embedding": vector},
            )
            for row, text, vector in zip(batch, texts, vectors, strict=True)
        ]
        if docs:
            coll.insert(docs)
            indexed += len(docs)
            for row in batch:
                document_id = str(row.get("document_id", ""))
                completed_by_document[document_id] += 1
                document_total = totals[document_id]
                progress_documents[document_id] = {
                    **progress_documents[document_id],
                    "status": (
                        "completed"
                        if completed_by_document[document_id] >= document_total
                        else "running"
                    ),
                    "percent": round(
                        completed_by_document[document_id] / document_total * 100,
                        2,
                    ),
                    "completed": completed_by_document[document_id],
                }
            current_document_id = str(batch[-1].get("document_id", ""))
            _write_reindex_progress(
                job_id,
                stage="embedding",
                completed=indexed,
                total=len(rows),
                current_document_id=current_document_id,
                current_document=names.get(current_document_id, ""),
                documents=progress_documents,
                message=f"已完成 {indexed} / {len(rows)} 个片段",
            )
        del docs, vectors
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    # ponytail: one flush at the end avoids RocksDB file churn; batch insert keeps GPU/CPU memory flat.
    _write_reindex_progress(
        job_id,
        stage="embedding",
        completed=indexed,
        total=len(rows),
        documents=progress_documents,
        stage_percent=99,
        message="Zvec 索引提交中",
    )
    coll.flush()
    _write_reindex_progress(
        job_id,
        stage="completed",
        completed=1,
        total=1,
        documents={
            document_id: {
                **value,
                "status": "completed",
                "percent": 100,
                "completed": value["total"],
            }
            for document_id, value in progress_documents.items()
        },
        stage_percent=100,
        message="Zvec 索引提交完成",
    )
    return {"backend": "zvec+bge-m3", "indexed": indexed, "collection_path": str(ZVEC_PATH)}


def _mark_lexical_ok() -> None:
    global _LEXICAL_STATE
    if _LEXICAL_STATE != "ok":
        _LEXICAL_STATE = "ok"


def _warn_lexical_fallback(exc: BaseException) -> None:
    """全文通道不可用时提示一次（不刷屏），并把状态留给 /health。"""
    global _LEXICAL_FALLBACK_WARNED, _LEXICAL_STATE
    _LEXICAL_STATE = f"unavailable: {type(exc).__name__}: {exc}"
    if _LEXICAL_FALLBACK_WARNED:
        return
    _LEXICAL_FALLBACK_WARNED = True
    print(
        f"[warn] 全文通道不可用，已回落到纯向量检索：{type(exc).__name__}: {exc}",
        flush=True,
    )


def _search_zvec(query: str, top_k: int) -> list[RerankResult]:
    if not ZVEC_PATH.exists():
        raise RuntimeError("zvec index not found; run POST /admin/reindex")
    coll = _open_zvec_collection()
    limit = max(top_k, RERANK_CANDIDATES, FUSED_CANDIDATES)
    # 用 `Query` 而不是原来的 `VectorQuery`：后者已被 zvec 标记为 deprecated
    # （运行时会打 DeprecationWarning），且它是 `Query` 的子类，行为一致。
    dense = zvec.Query("embedding", vector=_embed([query])[0])

    docs = None
    if _LEXICAL_ENABLED:
        # 整句直接当字面查询丢进去即可，**不需要先拆词**：实测
        # 「P.GAS.11对应的值是多少」与只喂「P.GAS.11」，目标块都排全文通道第 1。
        lexical = zvec.Query("content", fts=zvec.Fts(match_string=query))
        try:
            docs = coll.query(
                [dense, lexical], topk=limit, reranker=zvec.RrfReRanker()
            )
            _mark_lexical_ok()
        except Exception as exc:  # noqa: BLE001
            # 集合建得早、schema 里还没带上 FTS 索引时会走到这里。
            # **必须回落到纯向量**：混合检索是为了多召回，不能因为它
            # 把原本能用的检索整个弄挂——那比召不全严重得多。
            _warn_lexical_fallback(exc)
    if docs is None:
        docs = coll.query(dense, topk=limit)

    candidates = [
        RerankDocument(
            id=doc.id,
            text=str(doc.field("content") or ""),
            metadata={
                "document_id": doc.field("document_id"),
                "chunk_index": doc.field("chunk_index"),
                "source_file": doc.field("source_file"),
                "security_level": doc.field("security_level"),
                # 召回的原始分。**混合检索下这是 RRF 融合分**（约 1/60 量级），
                # 不再是向量距离——两者量纲不同，别拿它跟历史值比大小。
                # 下游定序用的是外层 `score`（交叉编码器分），不是这个。
                "retrieval_score": float(doc.score or 0.0),
            },
        )
        for doc in docs
    ]
    return _rerank(query, candidates, top_k)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "backend": "zvec+bge-m3+bge-reranker-v2-m3",
        "device": str(DEVICE),
        "bge_m3_path_exists": os.path.isdir(BGE_M3_PATH),
        "reranker_path_exists": os.path.isdir(RERANKER_PATH),
        "index_path": str(INDEX_PATH),
        "index_exists": INDEX_PATH.exists(),
        "zvec_collection_path": str(ZVEC_PATH),
        "zvec_index_exists": ZVEC_PATH.exists(),
        "dim": EMBED_DIM,
        # 全文通道是否真的在用。`unknown` = 还没查过（服务刚起），
        # `ok` = 混合检索生效，`unavailable: ...` = 已静默回落到纯向量——
        # 看到这个就说明本次改动没生效，要重建集合（POST /admin/reindex）。
        "lexical": _LEXICAL_STATE,
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    return EmbedResponse(vectors=_embed(req.texts, req.normalize), dim=EMBED_DIM, backend="bge-m3")


@app.post("/query-vector", response_model=EmbedResponse)
def query_vector(req: EmbedRequest) -> EmbedResponse:
    return embed(req)


@app.post("/rerank")
def rerank(req: RerankRequest) -> dict[str, Any]:
    return {"results": _rerank(req.query, req.documents, req.top_k), "backend": "bge-reranker-v2-m3"}


@app.post("/search")
def search(req: SearchRequest) -> dict[str, Any]:
    try:
        with _INDEX_LOCK:
            results = _search_zvec(req.query, req.top_k)
        return {"results": results, "backend": "zvec+bge-m3+bge-reranker-v2-m3"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/admin/reindex")
def admin_reindex(req: ReindexRequest | None = None) -> dict[str, Any]:
    job_id = req.job_id if req else ""
    try:
        with _INDEX_LOCK:
            return _reindex_zvec(job_id)
    except Exception as exc:
        _write_reindex_progress(
            job_id,
            stage="failed",
            completed=0,
            total=0,
            message="Zvec 索引重建失败",
            error=str(exc),
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/admin/sync")
def admin_sync(
    file: UploadFile = File(...),
    job_id: str = "",
    x_hdw_rag_admin_token: str | None = Header(
        default=None,
        alias="X-HDW-RAG-Admin-Token",
    ),
) -> dict[str, Any]:
    _require_admin_token(x_hdw_rag_admin_token)
    temporary = INDEX_PATH.with_name(f".{INDEX_PATH.name}.{uuid.uuid4().hex}.uploading")
    previous_index = INDEX_PATH.with_name(f".{INDEX_PATH.name}.{uuid.uuid4().hex}.previous")
    previous_zvec = ZVEC_PATH.with_name(f".{ZVEC_PATH.name}.{uuid.uuid4().hex}.previous")
    try:
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as target:
            while chunk := file.file.read(1024 * 1024):
                target.write(chunk)
        _validate_index_file(temporary)
        with _INDEX_LOCK:
            if INDEX_PATH.exists():
                INDEX_PATH.replace(previous_index)
            if ZVEC_PATH.exists():
                ZVEC_PATH.replace(previous_zvec)
            try:
                temporary.replace(INDEX_PATH)
                result = _reindex_zvec(job_id)
            except Exception as exc:
                try:
                    _write_reindex_progress(
                        job_id,
                        stage="failed",
                        completed=0,
                        total=0,
                        message="远端 RAG 同步失败",
                        error=str(exc),
                    )
                finally:
                    if ZVEC_PATH.exists():
                        shutil.rmtree(ZVEC_PATH)
                    if previous_zvec.exists():
                        previous_zvec.replace(ZVEC_PATH)
                    if INDEX_PATH.exists():
                        INDEX_PATH.unlink()
                    if previous_index.exists():
                        previous_index.replace(INDEX_PATH)
                raise
            else:
                previous_index.unlink(missing_ok=True)
                if previous_zvec.exists():
                    shutil.rmtree(previous_zvec)
        return {
            "status": "synced",
            "job_id": job_id,
            "index": result,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        temporary.unlink(missing_ok=True)
        previous_index.unlink(missing_ok=True)
        if previous_zvec.exists():
            shutil.rmtree(previous_zvec)
        file.file.close()


def _self_check() -> None:
    assert os.path.isdir(BGE_M3_PATH)
    assert os.path.isdir(RERANKER_PATH)


if __name__ == "__main__":
    _self_check()
    print("RAG service self-check passed")
