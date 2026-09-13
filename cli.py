"""
cli.py
------
Command-line interface for the vector search database — handy for quick tests
or demoing without the web UI.

Examples
    python cli.py load-samples
    python cli.py add "Deep learning powers modern AI" --category technology
    python cli.py search "I enjoy sports" --top-k 3
    python cli.py stats
    python cli.py clear
"""

from __future__ import annotations

import argparse

from sample_data import SAMPLE_DOCS
from search_engine import SearchEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Vector Search Database CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Add a document")
    p_add.add_argument("text")
    p_add.add_argument("--category", default="")

    p_search = sub.add_parser("search", help="Semantic search")
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--min-score", type=float, default=0.0)
    p_search.add_argument("--category", default=None)

    p_index = sub.add_parser("index", help="Index text files in a folder (default: cwd)")
    p_index.add_argument("folder", nargs="?", default=".")

    sub.add_parser("load-samples", help="Load the sample corpus")
    sub.add_parser("stats", help="Show database stats")
    sub.add_parser("clear", help="Delete all documents")

    args = parser.parse_args()
    engine = SearchEngine()

    if args.command == "add":
        meta = {"category": args.category} if args.category else {}
        doc_id = engine.add(args.text, metadata=meta)
        print(f"Added document {doc_id}")

    elif args.command == "search":
        out = engine.search(
            args.query, top_k=args.top_k, min_score=args.min_score, category=args.category
        )
        print(f"\nQuery: {out['query']!r}")
        print(
            f"Searched {out['num_docs_searched']} docs "
            f"(encode {out['encode_ms']} ms, search {out['search_ms']} ms)\n"
        )
        if not out["results"]:
            print("  (no results)")
        for rank, r in enumerate(out["results"], start=1):
            m = r.metadata
            if "path" in m:
                print(f"  {rank}. [{r.score:.3f}] {m['path']}:{m['start_line']}-{m['end_line']}")
                snippet = " ".join(r.text.split())
                print(f"       {snippet[:160]}{'...' if len(snippet) > 160 else ''}")
            else:
                print(f"  {rank}. [{r.score:.3f}] ({m.get('category', '-')}) {r.text}")
        print()

    elif args.command == "index":
        rep = engine.index_folder(args.folder)
        print(
            f"Indexed {rep.indexed_files} files ({rep.chunks_added} chunks), "
            f"{rep.skipped_unchanged} unchanged, {rep.removed_files} removed. "
            f"Total docs: {engine.count()}"
        )
        for err in rep.errors:
            print(f"  ! {err}")

    elif args.command == "load-samples":
        engine.add_many(SAMPLE_DOCS)
        print(f"Loaded {len(SAMPLE_DOCS)} sample documents. Total: {engine.count()}")

    elif args.command == "stats":
        print(f"Backend : {engine.backend}")
        print(f"Dim     : {engine.embedder.dim}")
        print(f"Docs    : {engine.count()}")

    elif args.command == "clear":
        engine.clear()
        print("Database cleared.")


if __name__ == "__main__":
    main()
