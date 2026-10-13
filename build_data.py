"""
build_data.py
-------------
Build the vector database at deploy time (Render's build step).

A free Render instance has no persistent disk, so the database is created
during the build and shipped with the service. Set MSMARCO_PASSAGES to choose
how many MS MARCO passages to embed (default 5000, about a minute to build).

    python build_data.py
"""

from __future__ import annotations

import os
import sys

import msmarco
from sample_data import SAMPLE_DOCS
from search_engine import SearchEngine

DB_PATH = os.environ.get("DB_PATH", "data/vectors.db")
PASSAGES = int(os.environ.get("MSMARCO_PASSAGES", "5000"))
SPLIT = os.environ.get("MSMARCO_SPLIT", "validation")


def main() -> int:
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    engine = SearchEngine(db_path=DB_PATH)
    print(f"Embedding backend: {engine.backend}", flush=True)
    if engine.backend == "hashing-fallback":
        print("ERROR: no embedding model available — install fastembed or sentence-transformers.")
        return 1

    if engine.count_category(msmarco.CATEGORY) >= PASSAGES:
        print(f"Database already has {engine.count():,} documents; nothing to build.")
        return 0

    engine.add_many(SAMPLE_DOCS)
    report = msmarco.import_msmarco(engine, split=SPLIT, limit=PASSAGES,
                                    on_progress=lambda p: print(f"  {p.message}", flush=True)
                                    if p.done % 2560 == 0 else None)
    if os.environ.get("PREFETCH_RERANKER", "true").lower() in ("1", "true", "yes"):
        # Download the cross-encoder now so the running service never has to.
        print("Prefetching the MS MARCO re-ranker…", flush=True)
        print("  re-ranker ready" if engine.reranker.load() else "  re-ranker unavailable", flush=True)

    print(report.message, flush=True)
    print(f"Database ready: {engine.count():,} documents at {DB_PATH}")
    return 0 if report.state == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
