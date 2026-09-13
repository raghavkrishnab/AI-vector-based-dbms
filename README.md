# ✦ Vector Search — AI-Powered Semantic Search Database

Search your documents and code **by meaning, not keywords**, and get **AI answers
with citations**. Text is embedded with a transformer model, stored in a real
**SQLite** database, retrieved by **cosine similarity**, and answered by
**Claude** using retrieval-augmented generation (RAG).

> Engineering Project — VIT Chennai

---

## Features

- **Semantic search** — `all-MiniLM-L6-v2` sentence embeddings (384-d). A query like
  *"king on throne"* finds *"The monarch occupies the royal seat"* with zero shared words.
- **Ask AI (RAG)** — the top matches are sent to Claude (`claude-opus-5`), which streams
  an answer that cites its sources as clickable `[1]`, `[2]` chips.
- **Index any folder** — point it at a directory of notes, docs or code. Files are split into
  line-numbered chunks; re-indexing is incremental (only changed files are re-embedded).
- **Modern web UI** — light/dark themes, streaming answers, score meters, highlighted
  snippets, category filters, a data drawer to index folders, add text and browse/delete documents.
- **Real DBMS underneath** — documents, vectors (BLOBs), metadata (JSON) and timestamps live
  in SQL tables with indexes; filtering uses SQLite's JSON functions.
- **Works offline** — if the transformer model can't load, a built-in hashing embedder keeps
  search running; switching models later automatically re-embeds stored documents.

---

## Quick start

```bash
# 1. (optional) virtual environment
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. (optional) enable AI answers
set ANTHROPIC_API_KEY=sk-ant-...  # macOS/Linux: export ANTHROPIC_API_KEY=sk-ant-...

# 4. run the web app
python app.py
```

Open **http://127.0.0.1:8000**, click **☰ Data** → **Index folder** (or *Load sample
documents*), then search or ask a question.

Useful flags: `python app.py --folder C:\path\to\notes` indexes a folder at startup;
`--port 9000`, `--db data/other.db`. Set `CLAUDE_MODEL` to use a different Claude model.

### Command line

```bash
python cli.py index                                   # index the current folder
python cli.py search "how are similarity scores calculated" --top-k 3
python cli.py load-samples
python cli.py add "Deep learning powers modern AI" --category technology
python cli.py stats
```

---

## Architecture

```
                 ┌──────────────────────────┐
  Browser  ────► │ app.py  (FastAPI)        │  /api/search  /api/ask (SSE)  /api/index ...
  static/        └────────────┬─────────────┘
  index.html                  │
                 ┌────────────▼─────────────┐      ┌──────────────────────────┐
                 │ search_engine.py         │ ───► │ embedding.py             │
                 │ (facade + thread lock)   │      │ sentence-transformers    │
                 └──────┬─────────────┬─────┘      │ (or offline fallback)    │
                        │             │            └──────────────────────────┘
        ┌───────────────▼───┐   ┌─────▼──────────────┐   ┌─────────────────────┐
        │ vector_db.py      │   │ folder_indexer.py  │   │ rag.py              │
        │ SQLite + cosine   │   │ walk, chunk, embed │   │ Claude, streaming,  │
        │ similarity        │   │ (incremental)      │   │ cited answers       │
        └───────────────────┘   └────────────────────┘   └─────────────────────┘
```

### How "Ask AI" works

1. The question is embedded with the same model as the documents.
2. SQLite returns the stored vectors; the top-k by cosine similarity become the **sources**.
3. The sources are sent to the UI immediately, then to Claude inside `<source>` tags.
4. Claude's answer streams back over Server-Sent Events, citing sources by number.

```
similarity = (a · b) / (‖a‖ · ‖b‖)      # 1.0 = identical meaning, 0 = unrelated
```

### Database schema

```sql
CREATE TABLE documents (
    doc_id      TEXT PRIMARY KEY,
    text        TEXT NOT NULL,
    embedding   BLOB NOT NULL,     -- raw float32 bytes of the vector
    dim         INTEGER NOT NULL,
    metadata    TEXT DEFAULT '{}', -- JSON: category, source path, line range, mtime
    created_at  REAL NOT NULL
);
CREATE INDEX idx_documents_created_at ON documents(created_at);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);  -- e.g. embedding model
```

---

## Project structure

| File | Purpose |
|------|---------|
| `app.py` | FastAPI server: JSON API + serves the UI |
| `static/index.html` | Single-page web UI (no build step) |
| `rag.py` | Retrieval-augmented answers with Claude (streaming) |
| `search_engine.py` | Facade combining embeddings, database and indexing |
| `embedding.py` | Transformer embeddings + offline fallback |
| `vector_db.py` | SQLite vector store + cosine similarity search |
| `folder_indexer.py` | Walks a folder, chunks & embeds files incrementally |
| `cli.py` | Command-line interface |
| `sample_data.py` | Demo corpus and example queries |
| `tests/` | Unit and API tests |

---

## Tests

```bash
python -m pytest tests -q
```

Tests use the deterministic offline embedder, a temporary database and a mocked Claude
stream, so they need no internet or API key.

---

## Scaling beyond the demo

Exact search is ideal up to tens of thousands of chunks. For larger corpora:

- **FAISS / HNSW** — approximate nearest-neighbour search for millions of vectors.
- **PostgreSQL + `pgvector`** — vector similarity as a native SQL operator.
- **Hybrid search** — combine BM25 keyword scores with vector scores.
- **Multilingual model** — swap in `multilingual-e5` for cross-language search.
