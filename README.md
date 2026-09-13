# 🔎 Vector Search Database — Semantic Similarity Search

A simple, **DBMS-powered AI application** that finds documents by *meaning*
instead of exact keywords. Text is turned into 384-dimensional embedding
vectors, stored in a real **SQLite** database, and retrieved with **cosine
similarity** search.

> Engineering Project — VIT Chennai

---

## What it does

Traditional keyword search only matches exact words. This project uses machine
learning **embeddings** so that a query like *"I enjoy sports"* correctly
retrieves *"I love playing basketball"* — even though they share no words.

| | Traditional Search | Vector / Semantic Search |
|---|---|---|
| Matches | exact keywords | meaning / intent |
| `"king on throne"` finds | only literal matches | *"the monarch occupies the royal seat"* (0.96) |
| Handles synonyms | ❌ | ✅ |

---

## Why it's "DBMS-powered"

Every document is a **row in a SQL table** with a schema, primary key, index,
metadata and timestamp. The app uses SQL to insert, delete, list and count
documents, and stores each embedding as a `BLOB` column.

```sql
CREATE TABLE documents (
    doc_id      TEXT PRIMARY KEY,
    text        TEXT NOT NULL,
    embedding   BLOB NOT NULL,     -- raw float32 bytes of the vector
    dim         INTEGER NOT NULL,
    metadata    TEXT DEFAULT '{}', -- JSON (e.g. category)
    created_at  REAL NOT NULL
);
CREATE INDEX idx_documents_created_at ON documents(created_at);
```

Search loads the stored vectors and ranks them by cosine similarity to the
query vector (exact brute-force nearest-neighbour — simple and fast for demos).

---

## Architecture

```
        ┌──────────────┐   text    ┌───────────────┐  384-d vector  ┌──────────────┐
 User → │  demo.py     │ ────────► │ embedding.py  │ ─────────────► │ vector_db.py │
        │ (Streamlit)  │           │ (transformer  │                │  (SQLite)    │
        │              │ ◄──────── │  or fallback) │ ◄───────────── │              │
        └──────────────┘  results  └───────────────┘   cosine sim   └──────────────┘
                 ▲                          search_engine.py ties these together
```

**Tech stack:** Python · SQLite · NumPy · sentence-transformers
(`all-MiniLM-L6-v2`) · Streamlit.

The app has a built-in **offline fallback embedder**, so it runs even without
internet access or the heavy ML dependency — great for a live demo where WiFi
might fail.

---

## Quick start

```bash
# 1. (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run the web app
streamlit run demo.py
```

Then, in the app: click **"Load sample documents"** in the sidebar and try a
search like `I enjoy sports` or `king on throne`.

### Command-line usage (no browser needed)

```bash
python cli.py load-samples
python cli.py search "artificial intelligence and data" --top-k 3
python cli.py add "Deep learning powers modern AI" --category technology
python cli.py stats
```

### Search the files in a folder

Index every text/code file in a folder (defaults to the current directory),
then search their contents by meaning. Files are split into ~800-char chunks
and results show `path:start-end` lines. Re-running `index` is incremental —
unchanged files are skipped, edited files re-embedded, deleted files removed.

```bash
python cli.py index                       # current folder
python cli.py index C:\path\to\folder     # any other folder
python cli.py search "how are similarity scores calculated"
```

In the web app, use **📁 Index a Folder** in the sidebar.

---

## Running the tests

```bash
pip install pytest
python -m pytest tests/ -q
```

The tests use the deterministic offline embedder and a temporary database, so
they need no internet and run in a second.

---

## Project structure

| File | Purpose |
|------|---------|
| `demo.py` | Streamlit web UI (search, add, browse) |
| `embedding.py` | Embedding service (transformer + offline fallback) |
| `vector_db.py` | SQLite vector database + cosine similarity search |
| `search_engine.py` | High-level facade combining the two |
| `folder_indexer.py` | Walks a folder, chunks & embeds files (incremental) |
| `sample_data.py` | Demo corpus and example queries |
| `cli.py` | Command-line interface |
| `tests/test_demo.py` | Unit tests |
| `requirements.txt` | Python dependencies |
| `data/vectors.db` | SQLite database (auto-created on first run) |

---

## How it works, step by step

1. **Encoding** — `embedding.py` converts each piece of text into a
   384-dimensional unit vector. Similar meanings → nearby vectors.
2. **Storage** — `vector_db.py` writes the text, its vector (as a BLOB) and
   metadata into the `documents` table in SQLite.
3. **Search** — a query is encoded the same way, then compared against every
   stored vector with **cosine similarity**; the top-k highest scores are
   returned.

Cosine similarity between vectors **a** and **b**:

```
similarity = (a · b) / (‖a‖ · ‖b‖)      # 1.0 = identical meaning, 0 = unrelated
```

---

## Scaling beyond the demo

This project uses exact search, which is ideal up to tens of thousands of
documents. For production scale:

- **FAISS / Annoy** — approximate nearest-neighbour search for millions of vectors.
- **PostgreSQL + `pgvector`** — vector similarity as a native SQL operator.
- **Redis / caching** — cache frequent query embeddings.
- **Multilingual model** — swap in `multilingual-e5` for cross-language search.

---

## Real-world use cases

Semantic search powers product recommendations, document & FAQ retrieval,
Q&A systems, plagiarism detection, and customer-support ticket matching — and
is the retrieval foundation for modern **RAG** (retrieval-augmented generation)
AI systems.
