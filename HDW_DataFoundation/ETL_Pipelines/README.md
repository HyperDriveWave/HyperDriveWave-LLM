# ETL Pipelines

Minimal ingestion utilities for parsed MinerU markdown.

Current scripts:

```bash
python parse_documents.py --input /path/to/raw_docs --output /tmp/hdw_mineru_parsed
python ingest_documents.py --input /tmp/hdw_mineru_parsed --output /tmp/hdw_chunks.jsonl
```

Recommended P0 chain:

```bash
bash Scripts/ingest_knowledge.sh /path/to/raw_docs
```

The first script uses MinerU for `PDF` / `DOCX` / `PPTX` / `XLSX` / images and
copies plain Markdown directly. The second script writes
`HDW_Runtime/rag/chunks.jsonl`, which `hdw-rag` reads through `POST /search`.
