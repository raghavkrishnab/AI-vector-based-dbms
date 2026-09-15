"""
search_engine.py
----------------
High-level facade that combines the EmbeddingService and the SQLite VectorDB
into one convenient object. This is what the web API and the CLI talk to.

All database access goes through a lock so the web server's worker threads can
share one SQLite connection safely.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any, Dict, List, Optional

from embedding import EmbeddingService, get_embedding_service
from reranker import Reranker
from vector_db import SearchResult, VectorDB

# With re-ranking on, vector search fetches this many candidates (at least) for
# the cross-encoder to re-score.
RERANK_CANDIDATES = 30


def _norm_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def drop_near_duplicates(results: List[SearchResult]) -> List[SearchResult]:
    """Remove hits that repeat a higher-ranked hit — MS MARCO often has short and
    long versions of a passage, or copies that differ only after the first sentence."""
    kept: List[SearchResult] = []
    kept_norm: List[str] = []
    for r in results:
        t = _norm_text(r.text)
        if any(t in k or (len(k) >= 40 and k in t) or (len(t) >= 80 and t[:80] == k[:80])
               for k in kept_norm):
            continue
        kept.append(r)
        kept_norm.append(t)
    return kept


class SearchEngine:
    def __init__(self, db_path: str = "data/vectors.db",
                 embedder: Optional[EmbeddingService] = None,
                 reranker: Optional[Reranker] = None):
        self.embedder = embedder or get_embedding_service()
        self.reranker = reranker or Reranker()
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
        rerank: bool = False,
    ) -> Dict[str, Any]:
        """Run a semantic search, optionally re-ranked by the MS MARCO cross-encoder.
        Returns results plus timing info."""
        t0 = time.perf_counter()
        q_vec = self.embedder.encode_one(query)
        encode_ms = (time.perf_counter() - t0) * 1000

        # Fetch extra candidates: some are dropped as near-duplicates, and the
        # re-ranker needs a wider pool to choose from.
        fetch = max(top_k * 4, RERANK_CANDIDATES) if rerank else top_k * 2 + 4
        t1 = time.perf_counter()
        with self.lock:
            results: List[SearchResult] = self.db.search(
                q_vec, top_k=fetch, min_score=min_score, category=category
            )
            total = self.db.count()
        results = drop_near_duplicates(results)
        search_ms = (time.perf_counter() - t1) * 1000

        rerank_ms = None
        reranked = False
        if rerank and results:
            t2 = time.perf_counter()
            results = self.reranker.rerank(query, results, top_k)
            reranked = results[0].rerank_score is not None
            rerank_ms = round((time.perf_counter() - t2) * 1000, 2)

        return {
            "query": query,
            "results": results[:top_k],
            "encode_ms": round(encode_ms, 2),
            "search_ms": round(search_ms, 2),
            "rerank_ms": rerank_ms,
            "reranked": reranked,
            "num_docs_searched": total,
        }

    def count_category(self, category: str) -> int:
        with self.lock:
            return self.db.count_category(category)
