"""
rag.py
------
Retrieval-augmented answers using free, local models only.

The vector search finds the most relevant chunks, then one of two backends
answers from them, citing sources as [1], [2], ...

1. Ollama (preferred) — a local LLM server (https://ollama.com). Writes a
   natural-language answer grounded in the sources.
       ollama pull llama3.2
   Configure with OLLAMA_URL (default http://127.0.0.1:11434) and
   OLLAMA_MODEL (default llama3.2).

2. Extractive (fallback, no setup) — uses the sentence-transformers embedding
   model already loaded for search to pick the source sentences that best
   answer the question. Always available, fully offline.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import AsyncIterator, Dict, List

import httpx
import numpy as np
from fastapi.concurrency import run_in_threadpool

from vector_db import SearchResult

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

SYSTEM_PROMPT = """You answer questions using only the numbered sources retrieved \
from the user's vector database. Cite the sources you rely on inline as [1], [2], etc. \
If the sources don't contain the answer, say so plainly instead of guessing. \
Keep answers concise; use Markdown bullet lists or `code` where it helps."""

_status_cache: Dict = {"at": 0.0, "value": None}


def ollama_status() -> Dict:
    """Is Ollama reachable, and is the configured model pulled?
    A healthy result is cached for 10 s; a failure only for 2 s, so Ollama
    starting up (or recovering) is picked up almost immediately."""
    cached = _status_cache["value"]
    if cached is not None and time.time() - _status_cache["at"] < (10 if cached["model_ready"] else 2):
        return cached
    status = {"running": False, "model_ready": False, "models": []}
    try:
        res = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3.0)
        res.raise_for_status()
        names = [m["name"] for m in res.json().get("models", [])]
        status = {
            "running": True,
            "models": names,
            # "llama3.2" matches "llama3.2:latest"; an explicit tag must match exactly.
            "model_ready": any(n == OLLAMA_MODEL or n.split(":")[0] == OLLAMA_MODEL for n in names),
        }
    except (httpx.HTTPError, ValueError):
        pass
    _status_cache.update(at=time.time(), value=status)
    return status


def backend_info(embedder) -> Dict:
    st = ollama_status()
    if st["model_ready"]:
        return {"backend": "ollama", "label": f"Ollama · {OLLAMA_MODEL}", "generative": True}
    if st["running"]:
        return {"backend": "extractive", "label": f"Model {OLLAMA_MODEL} missing", "generative": False,
                "note": f"Ollama is running but '{OLLAMA_MODEL}' is not downloaded. Run: ollama pull {OLLAMA_MODEL} "
                        f"(installed: {', '.join(st['models']) or 'none'}). Until then answers quote your sources."}
    return {"backend": "extractive", "label": "Ollama offline", "generative": False,
            "note": f"Can't reach Ollama at {OLLAMA_URL}. Start the Ollama app (or run `ollama serve`). "
                    "Until then answers quote the most relevant sentences from your sources instead of being written by AI."}


def source_label(r: SearchResult) -> str:
    m = r.metadata
    if "path" in m:
        return f"{m['path']} (lines {m['start_line']}-{m['end_line']})"
    if m.get("category") == "msmarco":
        return f"MS MARCO web passage from {m.get('url', 'unknown source')}"
    return f"document {r.doc_id}" + (f" [{m['category']}]" if m.get("category") else "")


def build_user_message(question: str, results: List[SearchResult]) -> str:
    blocks = [
        f'<source index="{i}" label="{source_label(r)}">\n{r.text}\n</source>'
        for i, r in enumerate(results, start=1)
    ]
    # Small local models follow formatting best when the rule sits right next to the question.
    return (
        "<sources>\n" + "\n".join(blocks) + "\n</sources>\n\n"
        f"Question: {question}\n\n"
        "Answer in a few sentences using only the sources above. End every sentence that uses "
        "a source with its number in square brackets, for example: "
        "\"Unchanged files are skipped [2].\""
    )


# --------------------------------------------------------------------------- #
# Backend 1: Ollama
# --------------------------------------------------------------------------- #
async def _stream_ollama(question: str, results: List[SearchResult]) -> AsyncIterator[Dict]:
    payload = {
        "model": OLLAMA_MODEL,
        "stream": True,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(question, results)},
        ],
    }
    timeout = httpx.Timeout(connect=3.0, read=300.0, write=30.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as res:
            if res.status_code != 200:
                body = (await res.aread()).decode(errors="replace")
                raise RuntimeError(f"Ollama returned {res.status_code}: {body[:200]}")
            async for line in res.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise RuntimeError(f"Ollama error: {chunk['error']}")
                text = chunk.get("message", {}).get("content", "")
                if text:
                    yield {"type": "delta", "text": text}
                if chunk.get("done"):
                    break
    yield {"type": "done", "model": f"ollama · {OLLAMA_MODEL}"}


# --------------------------------------------------------------------------- #
# Backend 2: extractive answer with the local embedding model
# --------------------------------------------------------------------------- #
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n\s*\n|\n(?=\s*(?:[-*#>]|\d+[.)])\s)")


def _sentences(text: str) -> List[str]:
    """Split prose into sentences, re-joining lines that were hard-wrapped."""
    out = []
    for part in _SENTENCE_SPLIT.split(text):
        part = re.sub(r"^.*\n\s*[-=]{3,}\s*$", "", part, flags=re.M)  # drop underlined headings
        sent = re.sub(r"\s*\n\s*", " ", part).replace("**", "").strip(" \t-*#>`\"'")
        if len(sent) < 25:
            continue
        words = re.findall(r"[A-Za-z]{2,}", sent)
        symbols = len(re.findall(r"[(){}\[\]=;_<>]", sent))
        # Keep real prose: enough words, mostly letters, few code symbols.
        if len(words) >= 5 and sum(map(len, words)) / len(sent) > 0.55 and symbols / len(sent) < 0.06:
            out.append(sent if len(sent) <= 400 else sent[:400].rsplit(" ", 1)[0] + "…")
    return out


def extract_answer(question: str, results: List[SearchResult], embedder, max_sentences: int = 4) -> str:
    """Pick the source sentences most similar to the question, cited by source number."""
    candidates = [(i, s) for i, r in enumerate(results, start=1) for s in _sentences(r.text)]
    if not candidates:
        return "I couldn't find a clear answer in the retrieved sources — see the matches below."

    q = embedder.encode_one(question)
    vecs = embedder.encode([s for _, s in candidates])
    scores = vecs @ q / (np.linalg.norm(vecs, axis=1) * (np.linalg.norm(q) or 1) + 1e-12)

    picked, seen = [], set()
    for idx in np.argsort(-scores):
        src, sent = candidates[idx]
        key = sent.lower()
        if key in seen or scores[idx] < 0.2:
            continue
        seen.add(key)
        picked.append((idx, src, sent))
        if len(picked) == max_sentences:
            break
    if not picked:
        return "The retrieved sources don't seem to answer this directly — see the closest matches below."

    picked.sort(key=lambda p: (p[1], p[0]))  # keep source order for readability
    lines = [f"- {sent} [{src}]" for _, src, sent in picked]
    return "**Most relevant passages from your data:**\n\n" + "\n".join(lines)


async def _stream_extractive(question: str, results: List[SearchResult], embedder) -> AsyncIterator[Dict]:
    text = await run_in_threadpool(extract_answer, question, results, embedder)
    for piece in re.findall(r"\S+\s*", text):  # stream word by word for a consistent UI
        yield {"type": "delta", "text": piece}
    yield {"type": "done", "model": f"extractive · {embedder.backend.split(':')[-1]}"}


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
async def stream_answer(question: str, results: List[SearchResult], embedder) -> AsyncIterator[Dict]:
    """Yield {"type": "delta", "text"} events, then "done" (or "error")."""
    if (await run_in_threadpool(ollama_status))["model_ready"]:
        sent_any = False
        try:
            async for event in _stream_ollama(question, results):
                sent_any = sent_any or event["type"] == "delta"
                yield event
            return
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            _status_cache["value"] = None  # re-check next time
            if sent_any:
                yield {"type": "error", "message": f"Ollama stopped mid-answer: {exc}"}
                return
            yield {"type": "notice", "message": f"Ollama failed ({exc}), so this answer quotes your sources instead."}
    else:
        yield {"type": "notice", "message": backend_info(embedder).get("note", "")}

    async for event in _stream_extractive(question, results, embedder):
        yield event
