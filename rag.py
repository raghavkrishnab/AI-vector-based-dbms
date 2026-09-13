"""
rag.py
------
Retrieval-augmented generation: the vector search finds the most relevant
chunks, then Claude writes an answer grounded in them, citing sources as [1],
[2], ...

Credentials are resolved by the Anthropic SDK (ANTHROPIC_API_KEY, or an
`ant auth login` profile). Search keeps working without them.
"""

from __future__ import annotations

import os
from typing import AsyncIterator, Dict, List

from vector_db import SearchResult

MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5")

SYSTEM_PROMPT = """You answer questions using only the numbered sources retrieved \
from the user's vector database. Cite the sources you rely on inline as [1], [2], \
etc. If the sources don't contain the answer, say so plainly instead of guessing, \
and mention what related information the sources do contain. Keep answers concise \
and well structured; use Markdown (short paragraphs, bullet lists, `code`) where it helps. \
Latency-sensitive; begin your visible answer promptly."""


def ai_configured() -> bool:
    """Cheap hint for the UI: is an obvious credential present?"""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or os.environ.get("ANTHROPIC_PROFILE"))


def source_label(r: SearchResult) -> str:
    m = r.metadata
    if "path" in m:
        return f"{m['path']} (lines {m['start_line']}-{m['end_line']})"
    return f"document {r.doc_id}" + (f" [{m['category']}]" if m.get("category") else "")


def build_user_message(question: str, results: List[SearchResult]) -> str:
    blocks = [
        f'<source index="{i}" label="{source_label(r)}" similarity="{r.score:.2f}">\n{r.text}\n</source>'
        for i, r in enumerate(results, start=1)
    ]
    return "<sources>\n" + "\n".join(blocks) + "\n</sources>\n\nQuestion: " + question


async def stream_answer(question: str, results: List[SearchResult]) -> AsyncIterator[Dict]:
    """Yield {"type": "delta", "text": ...} events, then a final "done" or "error"."""
    try:
        import anthropic
    except ImportError:
        yield {"type": "error", "message": "The `anthropic` package is not installed (pip install anthropic)."}
        return

    try:
        client = anthropic.AsyncAnthropic()
        async with client.beta.messages.stream(
            model=MODEL,
            max_tokens=64000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_message(question, results)}],
            # If Claude Opus 5 declines, the API re-runs on Anthropic's recommended fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            async for text in stream.text_stream:
                yield {"type": "delta", "text": text}
            final = await stream.get_final_message()

        if final.stop_reason == "refusal":
            yield {"type": "error", "message": "Claude declined to answer this question."}
        else:
            yield {"type": "done", "model": final.model, "stop_reason": final.stop_reason}
    except anthropic.AuthenticationError:
        yield {"type": "error", "message": "Invalid Anthropic API key."}
    except anthropic.RateLimitError:
        yield {"type": "error", "message": "Rate limited by the Anthropic API. Try again shortly."}
    except anthropic.APIStatusError as e:
        yield {"type": "error", "message": f"Anthropic API error ({e.status_code}): {e.message}"}
    except anthropic.APIConnectionError:
        yield {"type": "error", "message": "Could not reach the Anthropic API. Check your connection."}
    except TypeError as e:
        # Raised by the SDK when no credentials can be resolved.
        if "api_key" in str(e) or "auth" in str(e).lower():
            yield {"type": "error", "message": "No Anthropic credentials found. Set ANTHROPIC_API_KEY and restart the server."}
        else:
            raise
