"""
Unit tests for the vector search database.

Run with:
    pytest -q
or:
    python -m pytest tests/ -q

The tests use the offline hashing embedder (prefer_transformer=False) and a
temporary SQLite file, so they run fast and need no internet.
"""

import os
import sys
import tempfile

import numpy as np
import pytest

# Make the project root importable when running from the tests/ folder.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from embedding import EmbeddingService  # noqa: E402
from vector_db import VectorDB  # noqa: E402


@pytest.fixture
def embedder():
    # Force the deterministic offline embedder for reproducible tests.
    return EmbeddingService(prefer_transformer=False)


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = VectorDB(db_path=path, dim=384)
    yield database
    database.close()
    os.remove(path)


def test_embedding_shape_and_norm(embedder):
    vec = embedder.encode_one("hello world")
    assert vec.shape == (384,)
    # Non-empty text should produce a unit-norm vector.
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_embedding_is_deterministic(embedder):
    a = embedder.encode_one("machine learning")
    b = embedder.encode_one("machine learning")
    assert np.allclose(a, b)


def test_add_and_count(db, embedder):
    assert db.count() == 0
    db.add_document("first doc", embedder.encode_one("first doc"))
    db.add_document("second doc", embedder.encode_one("second doc"))
    assert db.count() == 2


def test_add_returns_id_and_persists(db, embedder):
    doc_id = db.add_document("persist me", embedder.encode_one("persist me"))
    assert isinstance(doc_id, str) and len(doc_id) > 0
    docs = db.list_documents()
    assert any(d["doc_id"] == doc_id for d in docs)


def test_delete(db, embedder):
    doc_id = db.add_document("delete me", embedder.encode_one("delete me"))
    assert db.delete_document(doc_id) is True
    assert db.count() == 0
    # Deleting a non-existent id returns False.
    assert db.delete_document("nope") is False


def test_search_ranks_relevant_first(db, embedder):
    corpus = [
        "I love playing basketball and football",
        "The weather is sunny and warm today",
        "Machine learning and neural networks are exciting",
    ]
    for text in corpus:
        db.add_document(text, embedder.encode_one(text))

    q = embedder.encode_one("sports like basketball")
    results = db.search(q, top_k=3)

    assert len(results) == 3
    # The basketball/football sentence should rank first.
    assert "basketball" in results[0].text.lower()
    # Scores must be sorted in descending order.
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_empty_db_returns_empty(db, embedder):
    results = db.search(embedder.encode_one("anything"), top_k=5)
    assert results == []


def test_category_filter(db, embedder):
    db.add_document("football match", embedder.encode_one("football match"),
                    metadata={"category": "sports"})
    db.add_document("neural network", embedder.encode_one("neural network"),
                    metadata={"category": "tech"})

    results = db.search(embedder.encode_one("game"), top_k=5, category="sports")
    assert len(results) == 1
    assert results[0].metadata["category"] == "sports"


def test_cosine_similarity_bounds(db, embedder):
    db.add_document("identical text", embedder.encode_one("identical text"))
    results = db.search(embedder.encode_one("identical text"), top_k=1)
    # A vector compared with itself should score ~1.0.
    assert results[0].score == pytest.approx(1.0, abs=1e-4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
