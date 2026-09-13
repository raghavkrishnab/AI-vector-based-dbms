"""API tests for the FastAPI web app (offline embedder, temp database, no network)."""

import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import rag  # noqa: E402
from app import create_app  # noqa: E402
from embedding import EmbeddingService  # noqa: E402
from search_engine import SearchEngine  # noqa: E402


@pytest.fixture
def engine(tmp_path):
    eng = SearchEngine(db_path=str(tmp_path / "v.db"), embedder=EmbeddingService(prefer_transformer=False))
    yield eng
    eng.db.close()


@pytest.fixture
def client(engine):
    return TestClient(create_app(engine))


def test_ui_is_served(client):
    res = client.get("/")
    assert res.status_code == 200 and "Vector Search" in res.text


def test_samples_search_and_stats(client):
    assert client.post("/api/samples").json()["added"] > 0
    stats = client.get("/api/stats").json()
    assert stats["documents"] > 0 and "sports" in stats["categories"]

    out = client.post("/api/search", json={"query": "basketball weekends", "top_k": 3}).json()
    assert len(out["results"]) == 3
    assert "basketball" in out["results"][0]["text"].lower()


def test_add_delete_clear(client):
    doc_id = client.post("/api/documents", json={"text": "hello vector world"}).json()["doc_id"]
    assert any(d["doc_id"] == doc_id for d in client.get("/api/documents").json())
    assert client.delete(f"/api/documents/{doc_id}").status_code == 200
    assert client.delete(f"/api/documents/{doc_id}").status_code == 404
    client.post("/api/samples")
    client.delete("/api/documents")
    assert client.get("/api/stats").json()["documents"] == 0


def test_index_folder_endpoint(client, tmp_path):
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "a.md").write_text("Photosynthesis converts sunlight into chemical energy.")
    rep = client.post("/api/index", json={"folder": str(folder)}).json()
    assert rep["indexed_files"] == 1
    assert client.post("/api/index", json={"folder": str(folder / "missing")}).status_code == 400


def test_ask_streams_sources_then_answer(client, monkeypatch):
    async def fake_stream(question, results, embedder):
        yield {"type": "delta", "text": "Basketball [1]"}
        yield {"type": "done", "model": "test", "stop_reason": "end_turn"}

    monkeypatch.setattr(rag, "stream_answer", fake_stream)
    client.post("/api/samples")
    with client.stream("POST", "/api/ask", json={"query": "sports"}) as res:
        events = [json.loads(line[6:]) for line in res.iter_lines() if line.startswith("data: ")]
    assert [e["type"] for e in events] == ["sources", "delta", "done"]
    assert events[0]["results"]


def test_ask_with_empty_db_returns_error(client):
    with client.stream("POST", "/api/ask", json={"query": "anything"}) as res:
        events = [json.loads(line[6:]) for line in res.iter_lines() if line.startswith("data: ")]
    assert events[-1]["type"] == "error"


def _events(client, query):
    with client.stream("POST", "/api/ask", json={"query": query}) as res:
        return [json.loads(line[6:]) for line in res.iter_lines() if line.startswith("data: ")]


def test_extractive_answer_without_ollama(client, monkeypatch):
    monkeypatch.setattr(rag, "ollama_status", lambda: {"running": False, "model_ready": False, "models": []})
    client.post("/api/samples")
    assert client.get("/api/stats").json()["ai"]["backend"] == "extractive"
    events = _events(client, "which sport is played on weekends with friends")
    assert events[-1]["type"] == "done" and events[-1]["model"].startswith("extractive")
    answer = "".join(e["text"] for e in events if e["type"] == "delta")
    assert "basketball" in answer.lower() and "[" in answer


def test_ollama_answer_is_streamed(client, monkeypatch):
    monkeypatch.setattr(rag, "ollama_status", lambda: {"running": True, "model_ready": True, "models": ["llama3.2:latest"]})

    async def fake_ollama(question, results):
        yield {"type": "delta", "text": "Basketball [1]"}
        yield {"type": "done", "model": "ollama · llama3.2"}

    monkeypatch.setattr(rag, "_stream_ollama", fake_ollama)
    client.post("/api/samples")
    events = _events(client, "sports")
    assert [e["type"] for e in events] == ["sources", "delta", "done"]
    assert events[-1]["model"].startswith("ollama")


def test_falls_back_when_ollama_unreachable(client, monkeypatch):
    monkeypatch.setattr(rag, "ollama_status", lambda: {"running": True, "model_ready": True, "models": []})
    monkeypatch.setattr(rag, "OLLAMA_URL", "http://127.0.0.1:9")  # nothing listens here
    client.post("/api/samples")
    events = _events(client, "sports")
    types = [e["type"] for e in events]
    assert "notice" in types and types[-1] == "done"


def test_prompt_includes_numbered_sources(engine):
    engine.add("The monarch occupies the royal seat.", metadata={"category": "royalty"})
    results = engine.search("king throne")["results"]
    msg = rag.build_user_message("Who sits on the throne?", results)
    assert '<source index="1"' in msg and "Question: Who sits on the throne?" in msg


def test_backend_change_reembeds(tmp_path):
    path = str(tmp_path / "v.db")
    eng = SearchEngine(db_path=path, embedder=EmbeddingService(prefer_transformer=False))
    eng.add("some stored text")
    eng.db.set_meta("embedding_backend", "some-other-model")
    eng.db.close()
    eng2 = SearchEngine(db_path=path, embedder=EmbeddingService(prefer_transformer=False))
    assert eng2.reembedded == 1
    eng2.db.close()
