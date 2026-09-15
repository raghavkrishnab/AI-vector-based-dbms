"""
app.py
------
FastAPI web server: a JSON API over the SearchEngine plus the single-page UI
in static/.

Run with:
    python app.py                      # http://127.0.0.1:8000
    python app.py --folder C:\\notes    # index a folder on startup
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import msmarco
import rag
from sample_data import EXAMPLE_QUERIES, SAMPLE_DOCS
from search_engine import SearchEngine

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=6, ge=1, le=20)
    min_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    category: Optional[str] = None
    rerank: bool = True


class MsMarcoRequest(BaseModel):
    split: Literal["validation", "train"] = "validation"
    limit: int = Field(default=20000, ge=100, le=700000)


class IndexRequest(BaseModel):
    folder: str


class AddRequest(BaseModel):
    text: str = Field(min_length=1)
    category: str = ""


def serialize(r) -> dict:
    return {"doc_id": r.doc_id, "text": r.text, "score": round(r.score, 4), "metadata": r.metadata,
            "rerank_score": None if r.rerank_score is None else round(r.rerank_score, 4)}


def create_app(engine: SearchEngine) -> FastAPI:
    app = FastAPI(title="Vector Search DB")
    import_progress = msmarco.ImportProgress()
    import_lock = threading.Lock()

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    @app.get("/api/ai")
    def ai_status():
        """Lightweight answer-backend status, polled by the UI."""
        return rag.backend_info(engine.embedder)

    @app.get("/api/stats")
    def stats():
        with engine.lock:
            stored = engine.db.get_meta("msmarco_examples")
        msmarco_count = engine.count_category(msmarco.CATEGORY)
        examples = EXAMPLE_QUERIES[:2] + (json.loads(stored)[:3] if stored and msmarco_count else [])
        return {
            "documents": engine.count(),
            "categories": engine.categories(),
            "backend": engine.backend,
            "dim": engine.embedder.dim,
            "semantic": engine.backend != "hashing-fallback",
            "reranker": {"model": engine.reranker.model_name, "status": engine.reranker.status},
            "msmarco": {"passages": msmarco_count, "import": import_progress.__dict__},
            "ai": rag.backend_info(engine.embedder),
            "cwd": os.getcwd(),
            "examples": examples if msmarco_count else EXAMPLE_QUERIES,
        }

    @app.post("/api/search")
    def search(req: SearchRequest):
        out = engine.search(req.query, top_k=req.top_k, min_score=req.min_score,
                            category=req.category or None, rerank=req.rerank)
        out["results"] = [serialize(r) for r in out["results"]]
        return out

    @app.post("/api/msmarco/import")
    def start_msmarco_import(req: MsMarcoRequest):
        if not import_lock.acquire(blocking=False):
            raise HTTPException(409, "An MS MARCO import is already running")

        def worker():
            try:
                msmarco.import_msmarco(engine, split=req.split, limit=req.limit, progress=import_progress)
            finally:
                import_lock.release()

        import_progress.__dict__.update(state="downloading", done=0, total=req.limit, message="Starting…")
        threading.Thread(target=worker, daemon=True, name="msmarco-import").start()
        return import_progress.__dict__

    @app.get("/api/msmarco/status")
    def msmarco_status():
        return {**import_progress.__dict__, "passages": engine.count_category(msmarco.CATEGORY)}

    @app.post("/api/ask")
    async def ask(req: SearchRequest):
        out = await run_in_threadpool(engine.search, req.query, req.top_k,
                                      req.min_score, req.category or None, req.rerank)
        results = out.pop("results")

        async def events():
            payload = {"type": "sources", **out, "results": [serialize(r) for r in results]}
            yield f"data: {json.dumps(payload)}\n\n"
            if not results:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No documents matched, so there is nothing to answer from. Index a folder or add documents first.'})}\n\n"
                return
            async for event in rag.stream_answer(req.query, results, engine.embedder):
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    @app.post("/api/index")
    def index_folder(req: IndexRequest):
        folder = os.path.expanduser(req.folder.strip().strip('"'))
        if not os.path.isdir(folder):
            raise HTTPException(400, f"Folder not found: {folder}")
        rep = engine.index_folder(folder)
        return {"folder": os.path.abspath(folder), **rep.__dict__, "documents": engine.count()}

    @app.get("/api/documents")
    def documents(limit: int = 500):
        return engine.list_documents(limit=limit)

    @app.post("/api/documents")
    def add_document(req: AddRequest):
        meta = {"category": req.category.strip()} if req.category.strip() else {}
        return {"doc_id": engine.add(req.text.strip(), metadata=meta)}

    @app.delete("/api/documents/{doc_id}")
    def delete_document(doc_id: str):
        if not engine.delete(doc_id):
            raise HTTPException(404, "No document with that id")
        return {"deleted": doc_id}

    @app.delete("/api/documents")
    def clear_documents():
        engine.clear()
        return {"cleared": True}

    @app.post("/api/samples")
    def load_samples():
        engine.add_many(SAMPLE_DOCS)
        return {"added": len(SAMPLE_DOCS), "documents": engine.count()}

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Vector Search DB web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--db", default="data/vectors.db")
    parser.add_argument("--folder", help="Index this folder before starting")
    parser.add_argument("--msmarco", type=int, metavar="N",
                        help="Import N MS MARCO passages (validation split) before starting")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.db) or ".", exist_ok=True)
    engine = SearchEngine(db_path=args.db)
    # Load the MS MARCO re-ranker in the background so the first search isn't slow.
    threading.Thread(target=engine.reranker.load, daemon=True).start()
    print(f"Embedding backend: {engine.backend}")
    ai = rag.backend_info(engine.embedder)
    print(f"Answer backend:    {ai['label']}" + (f"  ({ai['note']})" if ai.get("note") else ""))
    if engine.reembedded:
        print(f"Re-embedded {engine.reembedded} documents for the current model.")
    if args.folder:
        rep = engine.index_folder(args.folder)
        print(f"Indexed {rep.indexed_files} files ({rep.chunks_added} chunks).")
    if args.msmarco:
        prog = msmarco.import_msmarco(engine, limit=args.msmarco,
                                      on_progress=lambda p: print(f"  {p.message}", end="\r"))
        print(f"\n{prog.message}")
    print(f"Open http://{args.host}:{args.port}")
    uvicorn.run(create_app(engine), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
