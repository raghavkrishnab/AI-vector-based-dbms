"""
embedding.py
------------
Embedding service for the Vector Search Database.

Turns text into a fixed-length numeric vector (an "embedding") so that pieces of
text with similar *meaning* end up close together in vector space.

Primary model:  sentence-transformers  ->  all-MiniLM-L6-v2  (384 dimensions)
Fallback model: a lightweight, dependency-free hashing embedder that runs fully
                offline. It is used automatically if sentence-transformers (or
                its model download) is unavailable, so the demo ALWAYS runs.

The fallback is not as smart as the real transformer, but it keeps the app
functional on machines with no internet access or without the heavy ML deps.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List

import numpy as np

# Dimension of the primary model (all-MiniLM-L6-v2). The fallback matches it so
# the rest of the system does not care which embedder is active.
EMBEDDING_DIM = 384


class EmbeddingService:
    """Encodes text into normalized 384-dimensional float32 vectors."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", prefer_transformer: bool = True):
        self.model_name = model_name
        self.dim = EMBEDDING_DIM
        self._model = None
        self.backend = "hashing-fallback"

        if prefer_transformer:
            self._try_load_transformer()

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #
    def _try_load_transformer(self) -> None:
        """Attempt to load the real sentence-transformer model.

        Any failure (package missing, no internet to download weights, etc.)
        silently falls back to the offline hashing embedder.
        """
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self.model_name)
            # Trust the model's real dimensionality if it differs.
            get_dim = getattr(self._model, "get_embedding_dimension", None) \
                or self._model.get_sentence_embedding_dimension
            self.dim = get_dim()
            self.backend = f"sentence-transformers:{self.model_name}"
        except Exception:
            self._model = None
            self.backend = "hashing-fallback"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of strings into an (N, dim) float32 array of unit vectors."""
        if isinstance(texts, str):
            texts = [texts]

        if self._model is not None:
            vecs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return vecs.astype(np.float32)

        vecs = np.vstack([self._hash_embed(t) for t in texts])
        return vecs.astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        """Convenience helper: encode a single string into a 1-D vector."""
        return self.encode([text])[0]

    # ------------------------------------------------------------------ #
    # Offline fallback embedder
    # ------------------------------------------------------------------ #
    def _hash_embed(self, text: str) -> np.ndarray:
        """A deterministic bag-of-words hashing embedder (no network needed).

        Each token is hashed into a bucket and its weight added. The result is
        L2-normalized so cosine similarity behaves sensibly. This captures word
        overlap (and a little sub-word signal), which is enough to demo the
        pipeline offline.
        """
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = self._tokenize(text)
        if not tokens:
            return vec

        for token in tokens:
            # Hash the whole token and a few character trigrams for robustness.
            for feature in [token] + self._char_ngrams(token, n=3):
                bucket = int(hashlib.md5(feature.encode("utf-8")).hexdigest(), 16) % self.dim
                sign = 1.0 if (bucket % 2 == 0) else -1.0
                vec[bucket] += sign

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    @staticmethod
    def _char_ngrams(token: str, n: int = 3) -> List[str]:
        if len(token) <= n:
            return []
        return [token[i:i + n] for i in range(len(token) - n + 1)]


# Simple module-level singleton so the whole app shares one loaded model.
_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    global _service
    if _service is None:
        _service = EmbeddingService()
    return _service


if __name__ == "__main__":
    svc = get_embedding_service()
    print(f"Backend: {svc.backend}  (dim={svc.dim})")
    v = svc.encode_one("Machine learning is powerful")
    print("Vector shape:", v.shape, "| norm:", round(float(np.linalg.norm(v)), 4))
