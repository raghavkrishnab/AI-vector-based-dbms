"""Tests for indexing a folder of files into the vector database."""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from embedding import EmbeddingService  # noqa: E402
from folder_indexer import chunk_lines, index_folder  # noqa: E402
from vector_db import VectorDB  # noqa: E402


class _Engine:
    """Minimal stand-in for SearchEngine using the offline embedder."""

    def __init__(self, db_path):
        self.embedder = EmbeddingService(prefer_transformer=False)
        self.db = VectorDB(db_path=db_path, dim=self.embedder.dim)


@pytest.fixture
def engine(tmp_path):
    eng = _Engine(str(tmp_path / "vectors.db"))
    yield eng
    eng.db.close()


@pytest.fixture
def folder(tmp_path):
    root = tmp_path / "corpus"
    (root / "sub").mkdir(parents=True)
    (root / "sports.md").write_text("Basketball and football are popular sports.\n")
    (root / "sub" / "ml.py").write_text("# neural network training loop\ndef train(): pass\n")
    (root / "image.png").write_bytes(b"\x89PNG\x00\x00binary")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "junk.js").write_text("ignored")
    return root


def test_chunk_lines_covers_all_lines():
    text = "\n".join(f"line {i} " + "x" * 50 for i in range(100))
    chunks = chunk_lines(text, chunk_chars=300, overlap=2)
    assert chunks[0][0] == 1
    assert chunks[-1][1] == 100
    assert all(len(body) <= 400 for _, _, body in chunks)


def test_index_and_search_folder(engine, folder):
    rep = index_folder(engine, str(folder))
    assert rep.indexed_files == 2  # png and node_modules skipped

    q = engine.embedder.encode_one("basketball sports")
    top = engine.db.search(q, top_k=1)[0]
    assert top.metadata["path"] == "sports.md"
    assert top.metadata["start_line"] == 1


def test_reindex_is_incremental(engine, folder):
    index_folder(engine, str(folder))
    rep = index_folder(engine, str(folder))
    assert rep.indexed_files == 0 and rep.skipped_unchanged == 2

    target = folder / "sports.md"
    target.write_text("Tennis is played at Wimbledon.\n")
    os.utime(target, (time.time() + 5, time.time() + 5))
    (folder / "sub" / "ml.py").unlink()

    rep = index_folder(engine, str(folder))
    assert rep.indexed_files == 1 and rep.removed_files == 1
    texts = [d["text"] for d in engine.db.list_documents()]
    assert texts == ["Tennis is played at Wimbledon."]
