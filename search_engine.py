"""
search_engine.py
----------------
High-level facade that combines the EmbeddingService and the SQLite VectorDB
into one convenient object. This is what the UI and the CLI talk to.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from embedding import get_embedding_service
from vector_db import SearchResult, VectorDB


class SearchEngine:
    def __init__(self, db_path: str = "data/vectors.db"):
        self.embedder = get_embedding_service()
        self.db = VectorDB(db_path=db_path, dim=self.embedder.dim)

    @property
    def backend(self) -> str:
        return self.embedder.backend

    # ---- writes ---------------------------------------------------------- #
    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        vec = self.embedder.encode_one(text)
        return self.db.add_document(text=text, embedding=vec, metadata=metadata)

    def add_many(self, records: List[Dict[str, Any]]) -> List[str]:
        """records: list of {text, metadata?}."""
        texts = [r["text"] for r in records]
        vecs = self.embedder.encode(texts)
        items = [
            {"text": r["text"], "embedding": vecs[i], "metadata": r.get("metadata")}
            for i, r in enumerate(records)
        ]
        return self.db.add_documents(items)

    def index_folder(self, root: str = "."):
        """Index every text file under root (incremental). Returns an IndexReport."""
        from folder_indexer import index_folder

        return index_folder(self, root)

    def delete(self, doc_id: str) -> bool:
        return self.db.delete_document(doc_id)

    def clear(self) -> None:
        self.db.clear()

    # ---- reads ----------------------------------------------------------- #
    def count(self) -> int:
        return self.db.count()

    def list_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self.db.list_documents(limit=limit)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a semantic search. Returns results plus timing info."""
        t0 = time.perf_counter()
        q_vec = self.embedder.encode_one(query)
        encode_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        results: List[SearchResult] = self.db.search(
            q_vec, top_k=top_k, min_score=min_score, category=category
        )
        search_ms = (time.perf_counter() - t1) * 1000

        return {
            "query": query,
            "results": results,
            "encode_ms": round(encode_ms, 2),
            "search_ms": round(search_ms, 2),
            "num_docs_searched": self.db.count(),
        }
