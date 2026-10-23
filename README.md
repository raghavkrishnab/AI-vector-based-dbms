# ✦ Vector Search — AI-Powered Semantic Search Database

Search your documents and code **by meaning, not keywords**, and get **AI answers
with citations**. Text is embedded with a transformer model, stored in a real
**SQLite** database, retrieved by **cosine similarity**, and answered by a
**free, local AI model** (Ollama or sentence-transformers) using retrieval-augmented
generation (RAG). No paid API keys, and your data never leaves your machine.

> Engineering Project — VIT Chennai

---

## Features

- **Semantic search** — `all-MiniLM-L6-v2` sentence embeddings (384-d). A query like
  *"king on throne"* finds *"The monarch occupies the royal seat"* with zero shared words.
- **Ask AI (RAG), 100% free** — answers cite their sources as clickable `[1]`, `[2]` chips:
  - **Ollama** (e.g. `llama3.2`) writes a natural-language answer, streamed token by token.
  - Without Ollama, an **extractive** answer uses the same sentence-transformers model to pick
    the source sentences that best answer the question — zero setup.
- **MS MARCO knowledge base** — import real web passages from Microsoft's
  [MS MARCO](https://microsoft.github.io/msmarco/) dataset (real Bing questions + the passages
  retrieved for them), so the database can answer general-knowledge questions out of the box.
- **MS MARCO re-ranking** — the top 40 vector hits are re-scored by
  `cross-encoder/ms-marco-MiniLM-L6-v2`, a model trained on MS MARCO that reads the query and
  passage together — much more precise than vector similarity alone.
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

# 3. (optional) generated answers with a free local LLM
#    install Ollama from https://ollama.com, then:
ollama pull llama3.2

# 4. run the web app
python app.py
```

Open **http://127.0.0.1:8000**, click **☰ Data** → **Index folder** (or *Load sample
documents*), then search or ask a question. The pill in the top bar shows which answer
backend is active; the app detects Ollama automatically (no restart needed).

To add MS MARCO data, open **☰ Data → MS MARCO** and click **Import**, or run
`python cli.py msmarco --limit 20000` (the 21 MB validation split is downloaded once and cached).

Useful flags: `python app.py --folder C:\path\to\notes` indexes a folder at startup;
`--msmarco 20000` imports MS MARCO passages at startup;
`--port 9000`, `--db data/other.db`.
Environment variables: `OLLAMA_MODEL` (default `llama3.2`, e.g. `qwen2.5:3b`, `mistral`,
`phi3`) and `OLLAMA_URL` (default `http://127.0.0.1:11434`).

### Command line

```bash
python cli.py index                                   # index the current folder
python cli.py search "how are similarity scores calculated" --top-k 3
python cli.py msmarco --limit 20000                   # import MS MARCO web passages
python cli.py search "what causes rainbows" --rerank  # re-rank with the MS MARCO cross-encoder
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
        │ SQLite + cosine   │   │ walk, chunk, embed │   │ Ollama or extractive│
        │ similarity        │   │ (incremental)      │   │ cited answers       │
        └───────────────────┘   └────────────────────┘   └─────────────────────┘
```

### About MS MARCO

[MS MARCO](https://microsoft.github.io/msmarco/) (Microsoft MAchine Reading COmprehension) is
a large-scale dataset of ~1M anonymized Bing questions, each paired with the web passages a
search engine retrieved for it and human judgements of which passage answers the question.
It is the standard benchmark for training and evaluating semantic search models. This project
uses it in two ways:

| Use | What | Why |
|-----|------|-----|
| **Data** | v1.1 passages from [`microsoft/ms_marco`](https://huggingface.co/datasets/microsoft/ms_marco) (validation: 82k passages, train: 676k) | Gives the vector DB real-world content to search and answer from. Each passage keeps its URL, the Bing question and whether it was judged relevant. |
| **Model** | `cross-encoder/ms-marco-MiniLM-L6-v2` re-ranker (≈90 MB) | A model already *trained on* MS MARCO. Training one from scratch needs a GPU and days of compute; using the published checkpoint gives the same benefit instantly. |

> MS MARCO is released for non-commercial research purposes. The data is downloaded on demand
> and is never committed to this repository.

### How search works: retrieve, then re-rank

1. **Retrieve** — the query is embedded (`all-MiniLM-L6-v2`) and compared to every stored vector
   by cosine similarity; the top 40 candidates are kept. Vectors are cached in memory, so this
   takes milliseconds even with tens of thousands of passages.
2. **Re-rank** — the MS MARCO cross-encoder reads the query together with each candidate and
   assigns a relevance probability; the best top-k are returned.

### How "Ask AI" works

1. The question is embedded with the same model as the documents.
2. Retrieval and re-ranking (above) select the **sources**.
3. The sources are sent to the UI immediately.
4. **If Ollama is running** with the model pulled, the sources go to the local LLM inside
   `<source>` tags and its answer streams back over Server-Sent Events, citing sources by number.
5. **Otherwise**, the sources are split into sentences, each is embedded, and the sentences
   most similar to the question become the answer — each tagged with its source number.

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

## Deploying to Render (free tier)

The repo ships a [`render.yaml`](render.yaml) blueprint. In the Render dashboard choose
**New → Blueprint**, pick this repository, and Render will:

1. install [`requirements-deploy.txt`](requirements-deploy.txt) — no PyTorch;
2. run [`build_data.py`](build_data.py), which embeds `MSMARCO_PASSAGES` passages (5,000 by
   default) into `data/vectors.db` and pre-downloads the re-ranker;
3. start `uvicorn server:app`.

### Why the server runs different models

A free instance has 512 MB RAM and 0.1 CPU, and PyTorch alone does not fit. The server
therefore loads the *same two models* through ONNX runtime (`fastembed`) instead:

| | Local | Render |
|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | same model, ONNX — **identical vectors** |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L6-v2` | `Xenova/ms-marco-MiniLM-L-6-v2` (same model, ONNX) |
| Answers | Ollama (local LLM) | hosted OpenAI-compatible API (`LLM_API_KEY`) |
| Memory | ~800 MB | ~210 MB idle, ~350 MB with re-ranking |

Because the vectors are identical, a database built by either backend works with the other.

### AI answers on the server

Ollama needs several GB of RAM, so it cannot run on a small instance. Set `LLM_API_KEY` in the
Render dashboard to a free key from [Groq](https://console.groq.com) (default) or any
OpenAI-compatible provider via `LLM_BASE_URL` / `LLM_MODEL`. Without a key the deployed app
still works and answers by quoting the best-matching sentences from the sources.

### Free-tier limits to expect

- The service **sleeps after 15 minutes** of inactivity; the next request takes ~1 minute.
- There is **no persistent disk**: documents added through the UI are lost on restart, and the
  database is rebuilt from `build_data.py` on each deploy.
- Re-ranking works but is slow on 0.1 CPU, so `RERANK_DEFAULT` is `false` — the UI toggle
  still turns it on per search.
- Want the full 20,000 passages, always-on and fast re-ranking? Raise `MSMARCO_PASSAGES` and
  move to a paid instance (Starter 512 MB / Standard 2 GB) with a persistent disk.

---

## Project structure

| File | Purpose |
|------|---------|
| `app.py` | FastAPI server: JSON API + serves the UI |
| `static/index.html` | Single-page web UI (no build step) |
| `rag.py` | Free RAG answers: Ollama (generative) or extractive fallback |
| `search_engine.py` | Facade combining embeddings, database and indexing |
| `embedding.py` | Transformer embeddings + offline fallback |
| `vector_db.py` | SQLite vector store + cosine similarity search |
| `server.py` | ASGI entrypoint used in deployment |
| `build_data.py` | Builds the database at deploy time |
| `render.yaml` | Render blueprint (free tier) |
| `msmarco.py` | Downloads MS MARCO and imports passages (background job with progress) |
| `reranker.py` | MS MARCO cross-encoder re-ranking |
| `folder_indexer.py` | Walks a folder, chunks & embeds files incrementally |
| `cli.py` | Command-line interface |
| `sample_data.py` | Demo corpus and example queries |
| `tests/` | Unit and API tests |

---

## Tests

```bash
python -m pytest tests -q
```

Tests use the deterministic offline embedder, a temporary database and a mocked Ollama
stream, so they need no internet, GPU or Ollama install.

---

## Scaling beyond the demo

Exact search is ideal up to tens of thousands of chunks. For larger corpora:

- **FAISS / HNSW** — approximate nearest-neighbour search for millions of vectors.
- **PostgreSQL + `pgvector`** — vector similarity as a native SQL operator.
- **Hybrid search** — combine BM25 keyword scores with vector scores.
- **Multilingual model** — swap in `multilingual-e5` for cross-language search.

## Project deployment Link - onrender
- **Link** - https://vector-search-db-f2md.onrender.com
