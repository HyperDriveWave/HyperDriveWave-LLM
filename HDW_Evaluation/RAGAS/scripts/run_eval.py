from __future__ import annotations

import json
import sys
from pathlib import Path


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    dataset = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("datasets/golden_qa.jsonl")
    rows = load_jsonl(dataset)
    print(json.dumps({"cases": len(rows), "status": "dataset-loaded"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

