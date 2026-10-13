"""
reranker.py
-----------
Second-stage re-ranking with a cross-encoder trained on MS MARCO.

Vector search (a bi-encoder) is fast but compares query and document vectors
that were computed separately. A cross-encoder reads the query and each
candidate passage *together*, which is far more accurate but too slow to run
over the whole database — so we run it only on the top candidates.

Model: cross-encoder/ms-marco-MiniLM-L6-v2 (~90 MB, trained on the MS MARCO
passage-ranking dataset). Downloads once on first use; if it can't load, search
simply skips re-ranking.
"""

from __future__ import annotations

import math
import os
import threading
from typing import List, Optional

from vector_db import SearchResult

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
ONNX_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"  # same model, ONNX build
ONNX_THREADS = int(os.environ.get("ONNX_THREADS", "1"))
BATCH_SIZE = int(os.environ.get("RERANK_BATCH", "4"))


class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        self.model_name = model_name
        self._model = None
        self._kind = ""
        self._error: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def status(self) -> str:
        if self._model is not None:
            return "ready"
        return "unavailable" if self._error else "not loaded"

    def load(self) -> bool:
        """Load the model (once). Returns True if it is usable."""
        with self._lock:
            if self._model is None and self._error is None:
                # RERANK_BACKEND=fastembed forces the ONNX model (used in deployment).
                if os.environ.get("RERANK_BACKEND") == "fastembed":
                    self._try_fastembed()
                    return self._model is not None
                try:
                    from sentence_transformers import CrossEncoder  # type: ignore

                    # 256 tokens covers typical passages and roughly halves CPU time vs 512.
                    self._model = CrossEncoder(self.model_name, max_length=256)
                    self._kind = "sentence-transformers"
                except Exception as exc:  # missing package, no network, ...
                    self._error = str(exc)
                    self._try_fastembed()
        return self._model is not None

    def _try_fastembed(self) -> None:
        """ONNX version of the same MS MARCO cross-encoder — no PyTorch, so it
        fits small cloud instances."""
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder  # type: ignore

            # One thread and small batches keep peak memory near 200 MB instead
            # of ~800 MB, which is what lets this run on a 512 MB instance.
            self._model = TextCrossEncoder(ONNX_RERANK_MODEL, threads=ONNX_THREADS)
            self._kind = "fastembed"
            self.model_name = ONNX_RERANK_MODEL
            self._error = None
        except Exception:
            pass  # keep the original error; re-ranking stays off

    def rerank(self, query: str, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """Re-score results against the query and return the best top_k."""
        if not results or not self.load():
            return results[:top_k]
        if self._kind == "fastembed":
            logits = list(self._model.rerank(query, [r.text for r in results], batch_size=BATCH_SIZE))
        else:
            logits = self._model.predict([(query, r.text) for r in results], batch_size=64)
        for r, logit in zip(results, logits):
            # Sigmoid maps the raw relevance logit to an intuitive 0-1 score.
            r.rerank_score = 1.0 / (1.0 + math.exp(-float(logit)))
        return sorted(results, key=lambda r: r.rerank_score, reverse=True)[:top_k]
