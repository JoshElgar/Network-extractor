#!/usr/bin/env python3
"""Run one of six parallel chunks of the forced ISBN fallback."""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
from pathlib import Path

from enrich import load_books
from enrich_missing_fast import assign

ROOT = Path(__file__).resolve().parent
CHUNK_INDEX = int(os.environ.get("CHUNK_INDEX", "0"))
CHUNK_COUNT = int(os.environ.get("CHUNK_COUNT", "6"))
IDS_PATH = ROOT / "data" / "missing_ids_520.json"
OUT_DIR = ROOT / "results_missing_chunks"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    missing_ids = json.loads(IDS_PATH.read_text(encoding="utf-8"))
    all_books = {str(book["id"]): book for book in load_books()}
    selected_ids = missing_ids[CHUNK_INDEX::CHUNK_COUNT]
    books = [all_books[book_id] for book_id in selected_ids]
    print(f"Chunk {CHUNK_INDEX}: {len(books)} books", flush=True)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {executor.submit(assign, book): book for book in books}
        for done, future in enumerate(concurrent.futures.as_completed(future_map), 1):
            book = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"Failed {book['id']}: {exc}", flush=True)
                results.append(assign(book))
            if done % 20 == 0 or done == len(books):
                print(f"Chunk {CHUNK_INDEX}: assigned {done}/{len(books)}", flush=True)

    results.sort(key=lambda row: int(re.sub(r"\D", "", str(row["id"])) or "0"))
    output = OUT_DIR / f"results_chunk_{CHUNK_INDEX}.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "chunk": CHUNK_INDEX,
        "total": len(results),
        "assigned": sum(1 for row in results if row.get("isbn13") or row.get("isbn10")),
        "synthetic": sum(1 for row in results if row.get("synthetic")),
    }
    (OUT_DIR / f"summary_chunk_{CHUNK_INDEX}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
