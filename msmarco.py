"""
msmarco.py
----------
Import passages from Microsoft's MS MARCO dataset into the vector database.

MS MARCO (https://microsoft.github.io/msmarco/) contains real, anonymized Bing
search questions, each paired with ~8 web passages that a search engine
retrieved for it. It is the standard benchmark for semantic search — and the
data the embedding and re-ranking models in this project were trained on.

Source: the Hugging Face copy at microsoft/ms_marco (v1.1).
    validation split: ~21 MB, 10,047 questions, ~82k passages
    train split:      ~175 MB, 82,326 questions, ~676k passages

MS MARCO is licensed for non-commercial research use; the data is downloaded
on demand and is never committed to this repository.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional

DATASET_REPO = "microsoft/ms_marco"
SPLITS = {"validation": "v1.1/validation-00000-of-00001.parquet",
          "train": "v1.1/train-00000-of-00001.parquet"}
CATEGORY = "msmarco"


@dataclass
class ImportProgress:
    state: str = "idle"          # idle | downloading | embedding | done | error
    done: int = 0
    total: int = 0
    message: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    errors: List[str] = field(default_factory=list)


def download_split(split: str) -> str:
    """Download (or reuse the cached copy of) a split; returns the local parquet path."""
    from huggingface_hub import hf_hub_download

    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; choose one of {sorted(SPLITS)}")
    return hf_hub_download(DATASET_REPO, SPLITS[split], repo_type="dataset")


def iter_passages(parquet_path: str, limit: int) -> Iterator[Dict]:
    """Yield up to `limit` unique passages as {doc_id, text, metadata}."""
    import pandas as pd

    df = pd.read_parquet(parquet_path, columns=["query", "query_id", "query_type", "passages"])
    seen = set()
    for row in df.itertuples(index=False):
        p = row.passages
        for text, url, selected in zip(p["passage_text"], p["url"], p["is_selected"]):
            text = str(text).strip()
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
            if len(text) < 30 or digest in seen:
                continue
            seen.add(digest)
            yield {
                "doc_id": f"msmarco-{digest}",   # stable id: re-importing never duplicates
                "text": text,
                "metadata": {
                    "category": CATEGORY,
                    "url": str(url),
                    "query": str(row.query),        # the Bing question this passage was retrieved for
                    "query_type": str(row.query_type),
                    "selected": bool(selected),     # human-judged as answering that question
                },
            }
            if len(seen) >= limit:
                return


def sample_queries(parquet_path: str, n: int = 4) -> List[str]:
    """A few short, readable questions from the split, for the UI's example chips."""
    import pandas as pd

    df = pd.read_parquet(parquet_path, columns=["query", "query_type"])
    picks = df[df["query_type"].isin(["description", "entity"])]["query"]
    picks = picks[picks.str.len().between(18, 45)]
    return picks.sample(n=min(n, len(picks)), random_state=7).str.capitalize().tolist()


def import_msmarco(engine, split: str = "validation", limit: int = 20000, batch_size: int = 256,
                   progress: Optional[ImportProgress] = None,
                   on_progress: Optional[Callable[[ImportProgress], None]] = None) -> ImportProgress:
    """Download a split and embed up to `limit` passages into the engine's database."""
    prog = progress or ImportProgress()
    prog.__dict__.update(state="downloading", done=0, total=limit, errors=[],
                         message=f"Downloading MS MARCO {split} split…",
                         started_at=time.time(), finished_at=0.0)
    notify = on_progress or (lambda _p: None)
    notify(prog)
    try:
        path = download_split(split)
        prog.state, prog.message = "embedding", "Embedding passages…"
        notify(prog)

        batch: List[Dict] = []

        def flush():
            vecs = engine.embedder.encode([b["text"] for b in batch])
            with engine.lock:
                engine.db.add_documents([{**b, "embedding": vecs[i]} for i, b in enumerate(batch)])
            prog.done += len(batch)
            prog.message = f"Embedded {prog.done:,} of {prog.total:,} passages"
            batch.clear()
            notify(prog)

        for passage in iter_passages(path, limit):
            batch.append(passage)
            if len(batch) >= batch_size:
                flush()
        if batch:
            flush()
        prog.total = prog.done

        with engine.lock:
            engine.db.set_meta("msmarco_examples", json.dumps(sample_queries(path)))
        prog.state = "done"
        prog.message = f"Imported {prog.done:,} MS MARCO passages"
    except Exception as exc:
        prog.state, prog.message = "error", f"Import failed: {exc}"
        prog.errors.append(str(exc))
    prog.finished_at = time.time()
    notify(prog)
    return prog
