"""
search_engine.py
----------------
High-level facade that combines the EmbeddingService and the SQLite VectorDB
into one convenient object. This is what the web API and the CLI talk to.

All database access goes through a lock so the web server's worker threads can
share one SQLite connection safely.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

from embedding import EmbeddingService, get_embedding_service
from vector_db import SearchResult, VectorDB


class SearchEngine:
    def __init__(self, db_path: str = "data/vectors.db",
                 embedder: Optional[EmbeddingService] = None):
        self.embedder = embedder or get_embedding_service()
        self.db = VectorDB(db_path=db_path, dim=self.embedder.dim)
        self.lock = threading.RLock()
        self.reembedded = self._sync_embedding_backend()

    @property
    def backend(self) -> str:
        return self.embedder.backend

    def _sync_embedding_backend(self) -> int:
        """Vectors from different models are not comparable. If the stored
        vectors came from another embedder, re-embed every document from its
        stored text. Returns the number of documents re-embedded."""
        stored = self.db.get_meta("embedding_backend")
        count = 0
        if stored is not None and stored != self.backend:
            docs = self.db.all_texts()
            for start in range(0, len(docs), 256):
                batch = docs[start:start + 256]
                vecs = self.embedder.encode([text for _, text in batch])
                self.db.update_embeddings([(doc_id, vecs[i]) for i, (doc_id, _) in enumerate(batch)])
            count = len(docs)
        if stored != self.backend:
            self.db.set_meta("embedding_backend", self.backend)
        return count

    # ---- writes ---------------------------------------------------------- #
    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        vec = self.embedder.encode_one(text)
        with self.lock:
            return self.db.add_document(text=text, embedding=vec, metadata=metadata)

    def add_many(self, records: List[Dict[str, Any]]) -> List[str]:
        """records: list of {text, metadata?}."""
        texts = [r["text"] for r in records]
        vecs = self.embedder.encode(texts)
        items = [
            {"text": r["text"], "embedding": vecs[i], "metadata": r.get("metadata")}
            for i, r in enumerate(records)
        ]
        with self.lock:
            return self.db.add_documents(items)

    def index_folder(self, root: str = "."):
        """Index every text file under root (incremental). Returns an IndexReport."""
        from folder_indexer import index_folder

        with self.lock:
            return index_folder(self, root)

    def delete(self, doc_id: str) -> bool:
        with self.lock:
            return self.db.delete_document(doc_id)

    def clear(self) -> None:
        with self.lock:
            self.db.clear()

    # ---- reads ----------------------------------------------------------- #
    def count(self) -> int:
        with self.lock:
            return self.db.count()

    def categories(self) -> List[str]:
        with self.lock:
            return self.db.categories()

    def list_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self.lock:
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
        with self.lock:
            results: List[SearchResult] = self.db.search(
                q_vec, top_k=top_k, min_score=min_score, category=category
            )
            total = self.db.count()
        search_ms = (time.perf_counter() - t1) * 1000

        return {
            "query": query,
            "results": results,
            "encode_ms": round(encode_ms, 2),
            "search_ms": round(search_ms, 2),
            "num_docs_searched": total,
        }
