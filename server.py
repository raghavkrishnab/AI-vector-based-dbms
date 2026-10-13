"""
server.py
---------
ASGI entrypoint for deployment (Render runs: uvicorn server:app --host 0.0.0.0 --port $PORT).

Differences from `python app.py`:
  * the database path comes from DB_PATH (default data/vectors.db);
  * the re-ranker is only pre-loaded when RERANK_DEFAULT is true, because a
    512 MB instance has little room to spare.
"""

from __future__ import annotations

import os
import threading

import rag
from app import create_app
from search_engine import SearchEngine

DB_PATH = os.environ.get("DB_PATH", "data/vectors.db")

os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
engine = SearchEngine(db_path=DB_PATH)
app = create_app(engine)

print(f"Embedding backend: {engine.backend} ({engine.count():,} documents)")
print(f"Answer backend:    {rag.backend_info(engine.embedder)['label']}")
if os.environ.get("RERANK_DEFAULT", "true").lower() in ("1", "true", "yes"):
    threading.Thread(target=engine.reranker.load, daemon=True).start()
