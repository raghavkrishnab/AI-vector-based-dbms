"""
vector_db.py
------------
A tiny vector database backed by SQLite (a real relational DBMS built into
Python's standard library).

What "DBMS-powered" means here:
  * Documents live in an actual SQL table with a schema, primary keys, indexes,
    timestamps and metadata.
  * You add / delete / list / count rows with SQL statements.
  * The 384-dimensional embedding for each document is stored in a BLOB column
    (the raw float32 bytes of a NumPy array).

Similarity search:
  * On search we load the stored vectors, compute cosine similarity against the
    query vector, and return the top-k highest-scoring rows.
  * This is exact (brute-force) nearest-neighbour search. It is simple and
    perfectly fast for demos of up to tens of thousands of documents. For very
    large corpora you would swap in FAISS or Postgres + pgvector (see README).
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SearchResult:
    doc_id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class VectorDB:
    """SQLite-backed store of documents and their embedding vectors."""

    def __init__(self, db_path: str = "data/vectors.db", dim: int = 384):
        self.db_path = db_path
        self.dim = dim
        # check_same_thread=False lets the web server's threads share the
        # connection; SearchEngine serializes access with a lock.
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #
    def _create_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id      TEXT PRIMARY KEY,
                text        TEXT NOT NULL,
                embedding   BLOB NOT NULL,
                dim         INTEGER NOT NULL,
                metadata    TEXT DEFAULT '{}',
                created_at  REAL NOT NULL
            )
            """
        )
        # Index on created_at so "most recent" queries stay fast.
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at)"
        )
        # Key/value settings, e.g. which embedding model produced the vectors.
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Serialization helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _vec_to_blob(vec: np.ndarray) -> bytes:
        return np.asarray(vec, dtype=np.float32).tobytes()

    @staticmethod
    def _blob_to_vec(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Write operations
    # ------------------------------------------------------------------ #
    def add_document(
        self,
        text: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """Insert one document. Returns the generated (or supplied) doc_id."""
        doc_id = doc_id or uuid.uuid4().hex[:12]
        metadata = metadata or {}
        self.conn.execute(
            """
            INSERT OR REPLACE INTO documents
                (doc_id, text, embedding, dim, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                text,
                self._vec_to_blob(embedding),
                int(len(embedding)),
                json.dumps(metadata),
                time.time(),
            ),
        )
        self.conn.commit()
        return doc_id

    def add_documents(self, items: List[Dict[str, Any]]) -> List[str]:
        """Batch insert in a single transaction. Each item: {text, embedding, metadata?, doc_id?}."""
        ids, rows, now = [], [], time.time()
        for item in items:
            doc_id = item.get("doc_id") or uuid.uuid4().hex[:12]
            ids.append(doc_id)
            rows.append((
                doc_id,
                item["text"],
                self._vec_to_blob(item["embedding"]),
                int(len(item["embedding"])),
                json.dumps(item.get("metadata") or {}),
                now,
            ))
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO documents
                (doc_id, text, embedding, dim, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()
        return ids

    def delete_document(self, doc_id: str) -> bool:
        cur = self.conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def delete_by_source(self, source: str) -> int:
        """Delete every chunk that came from one indexed file."""
        cur = self.conn.execute(
            "DELETE FROM documents WHERE json_extract(metadata, '$.source') = ?", (source,)
        )
        self.conn.commit()
        return cur.rowcount

    def indexed_sources(self, root: str) -> Dict[str, int]:
        """Return {source_path: mtime} for files indexed from the given root folder."""
        rows = self.conn.execute(
            "SELECT DISTINCT json_extract(metadata, '$.source') AS source, "
            "json_extract(metadata, '$.mtime') AS mtime FROM documents "
            "WHERE json_extract(metadata, '$.root') = ?",
            (root,),
        ).fetchall()
        return {r["source"]: r["mtime"] for r in rows}

    def all_texts(self) -> List[tuple]:
        """Return [(doc_id, text), ...] for every stored document."""
        rows = self.conn.execute("SELECT doc_id, text FROM documents").fetchall()
        return [(r["doc_id"], r["text"]) for r in rows]

    def update_embeddings(self, pairs: List[tuple]) -> None:
        """Replace vectors in bulk. pairs: [(doc_id, embedding), ...]."""
        self.conn.executemany(
            "UPDATE documents SET embedding = ?, dim = ? WHERE doc_id = ?",
            [(self._vec_to_blob(v), int(len(v)), doc_id) for doc_id, v in pairs],
        )
        self.conn.commit()

    def categories(self) -> List[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT json_extract(metadata, '$.category') AS c FROM documents "
            "WHERE c IS NOT NULL AND c != '' ORDER BY c"
        ).fetchall()
        return [r["c"] for r in rows]

    def clear(self) -> None:
        self.conn.execute("DELETE FROM documents")
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Read operations
    # ------------------------------------------------------------------ #
    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
        return int(row["n"])

    def list_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT doc_id, text, metadata, created_at FROM documents "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "doc_id": r["doc_id"],
                "text": r["text"],
                "metadata": json.loads(r["metadata"] or "{}"),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def _load_all_vectors(self):
        """Return (ids, texts, metadata_list, matrix[N, dim])."""
        rows = self.conn.execute(
            "SELECT doc_id, text, embedding, metadata FROM documents"
        ).fetchall()
        ids, texts, metas, vectors = [], [], [], []
        for r in rows:
            ids.append(r["doc_id"])
            texts.append(r["text"])
            metas.append(json.loads(r["metadata"] or "{}"))
            vectors.append(self._blob_to_vec(r["embedding"]))
        matrix = np.vstack(vectors) if vectors else np.zeros((0, self.dim), dtype=np.float32)
        return ids, texts, metas, matrix

    # ------------------------------------------------------------------ #
    # Similarity search
    # ------------------------------------------------------------------ #
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_score: float = 0.0,
        category: Optional[str] = None,
    ) -> List[SearchResult]:
        """Return the top_k most similar documents by cosine similarity.

        category: optional metadata filter (only rows whose metadata['category']
                  matches are considered) — demonstrates DBMS-style filtering.
        """
        ids, texts, metas, matrix = self._load_all_vectors()
        if matrix.shape[0] == 0:
            return []

        # Optional metadata filter.
        if category:
            keep = [i for i, m in enumerate(metas) if m.get("category") == category]
            if not keep:
                return []
            ids = [ids[i] for i in keep]
            texts = [texts[i] for i in keep]
            metas = [metas[i] for i in keep]
            matrix = matrix[keep]

        scores = self._cosine_similarity(query_embedding, matrix)

        order = np.argsort(-scores)  # highest score first
        results: List[SearchResult] = []
        for idx in order[:top_k]:
            score = float(scores[idx])
            if score < min_score:
                continue
            results.append(
                SearchResult(
                    doc_id=ids[idx],
                    text=texts[idx],
                    score=score,
                    metadata=metas[idx],
                )
            )
        return results

    @staticmethod
    def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Cosine similarity between a query vector and every row of a matrix."""
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        q_norm = np.linalg.norm(query)
        if q_norm == 0:
            return np.zeros(matrix.shape[0], dtype=np.float32)

        m_norms = np.linalg.norm(matrix, axis=1)
        # Avoid division by zero for any zero-length stored vectors.
        m_norms[m_norms == 0] = 1e-12
        return (matrix @ query) / (m_norms * q_norm)

    def close(self) -> None:
        self.conn.close()
