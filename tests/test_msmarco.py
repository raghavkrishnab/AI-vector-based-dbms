"""Tests for MS MARCO import, re-ranking and the vector cache (offline, synthetic data)."""

import os
import sys
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import msmarco  # noqa: E402
from app import create_app  # noqa: E402
from embedding import EmbeddingService  # noqa: E402
from search_engine import SearchEngine  # noqa: E402
from test_app import KeywordReranker  # noqa: E402


@pytest.fixture
def parquet(tmp_path):
    """A tiny file with the same shape as the MS MARCO v1.1 parquet."""
    rows = [
        {"query": "what is the capital of france", "query_id": 1, "query_type": "location",
         "passages": {"passage_text": ["Paris is the capital and largest city of France.",
                                       "France is a country in Western Europe with many regions."],
                      "url": ["https://en.wikipedia.org/wiki/Paris", "https://example.com/france"],
                      "is_selected": [1, 0]}},
        {"query": "how do vaccines work", "query_id": 2, "query_type": "description",
         "passages": {"passage_text": ["Vaccines train the immune system to recognize a pathogen.",
                                       "Paris is the capital and largest city of France.",  # duplicate
                                       "too short"],
                      "url": ["https://example.org/vaccines", "https://dup.example", "https://x"],
                      "is_selected": [1, 0, 0]}},
    ]
    path = tmp_path / "msmarco.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return str(path)


@pytest.fixture
def engine(tmp_path):
    eng = SearchEngine(db_path=str(tmp_path / "v.db"), embedder=EmbeddingService(prefer_transformer=False),
                       reranker=KeywordReranker())
    yield eng
    eng.db.close()


def test_iter_passages_dedupes_and_keeps_metadata(parquet):
    passages = list(msmarco.iter_passages(parquet, limit=100))
    assert len(passages) == 3  # duplicate and too-short passages dropped
    paris = passages[0]
    assert paris["doc_id"].startswith("msmarco-")
    assert paris["metadata"]["query"] == "what is the capital of france"
    assert paris["metadata"]["selected"] is True
    assert list(msmarco.iter_passages(parquet, limit=2)).__len__() == 2


def test_import_is_idempotent_and_searchable(engine, parquet, monkeypatch):
    monkeypatch.setattr(msmarco, "download_split", lambda split: parquet)
    prog = msmarco.import_msmarco(engine, limit=100, batch_size=2)
    assert prog.state == "done" and prog.done == 3
    msmarco.import_msmarco(engine, limit=100)
    assert engine.count_category("msmarco") == 3  # stable ids: no duplicates

    top = engine.search("capital of france paris", top_k=1)["results"][0]
    assert "Paris" in top.text and top.metadata["url"].startswith("https://")


def test_rerank_reorders_and_scores(engine):
    engine.add_many([{"text": "Paris is the capital of France"},
                     {"text": "capital gains tax rules explained"},
                     {"text": "the weather in paris today"}])
    out = engine.search("capital of france", top_k=2, rerank=True)
    assert out["reranked"] and out["rerank_ms"] is not None
    assert len(out["results"]) == 2
    assert out["results"][0].text == "Paris is the capital of France"
    assert all(r.rerank_score is not None for r in out["results"])


def test_near_duplicate_passages_collapsed(engine):
    short = "Normal body temperature is about 98.6 F (37 C) for most adults."
    engine.add_many([{"text": short},
                     {"text": short + " It varies slightly during the day and between people."},
                     {"text": "Fever is a body temperature above the normal range."}])
    texts = [r.text for r in engine.search("normal body temperature", top_k=5)["results"]]
    assert len(texts) == 2
    assert sum(short in t for t in texts) == 1


def test_vector_cache_invalidated_on_write(engine):
    engine.add("first document about apples")
    engine.search("apples")  # fills the cache
    assert engine.db._cache is not None
    doc_id = engine.add("second document about apples")
    assert engine.db._cache is None
    assert len(engine.search("apples", top_k=5)["results"]) == 2
    engine.delete(doc_id)
    assert len(engine.search("apples", top_k=5)["results"]) == 1


def test_import_api_runs_in_background(engine, parquet, monkeypatch):
    monkeypatch.setattr(msmarco, "download_split", lambda split: parquet)
    client = TestClient(create_app(engine))
    assert client.post("/api/msmarco/import", json={"limit": 100}).status_code == 200
    for _ in range(50):
        status = client.get("/api/msmarco/status").json()
        if status["state"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert status["state"] == "done" and status["passages"] == 3
    stats = client.get("/api/stats").json()
    assert stats["msmarco"]["passages"] == 3 and "msmarco" in stats["categories"]

    res = client.post("/api/search", json={"query": "how do vaccines work", "top_k": 1}).json()
    assert res["reranked"] and res["results"][0]["rerank_score"] is not None
