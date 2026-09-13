"""
folder_indexer.py
-----------------
Walks a folder, splits every text file into overlapping chunks, embeds each
chunk and stores it in the vector database — so you can semantically search
the contents of a directory (by default, the current working directory).

Each chunk is one row in the `documents` table with metadata:
    {category: <file extension>, source: <absolute path>, path: <relative path>,
     start_line, end_line, mtime}

Re-indexing is incremental: unchanged files (same mtime) are skipped, changed
files have their old chunks replaced, and deleted files are removed.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Iterator, List, Tuple

# Directories that are never worth indexing.
SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", "node_modules", "venv", ".venv", "env",
    ".mypy_cache", ".pytest_cache", ".idea", ".vscode", "dist", "build", "data",
}

# File types treated as searchable text.
TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c",
    ".h", ".cpp", ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".html", ".css",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".sql", ".sh", ".ps1",
    ".csv", ".xml", ".ipynb",
}

MAX_FILE_BYTES = 1_000_000   # skip files larger than ~1 MB
CHUNK_CHARS = 800            # target characters per chunk
OVERLAP_LINES = 3            # lines repeated between consecutive chunks


@dataclass
class IndexReport:
    indexed_files: int = 0
    skipped_unchanged: int = 0
    removed_files: int = 0
    chunks_added: int = 0
    errors: List[str] = field(default_factory=list)


def iter_text_files(root: str) -> Iterator[str]:
    """Yield absolute paths of indexable text files under root."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            if os.path.splitext(name)[1].lower() not in TEXT_EXTENSIONS:
                continue
            path = os.path.join(dirpath, name)
            try:
                if os.path.getsize(path) <= MAX_FILE_BYTES:
                    yield os.path.abspath(path)
            except OSError:
                continue


def read_text(path: str) -> str | None:
    """Read a file as UTF-8 text; return None for binary/undecodable files."""
    with open(path, "rb") as f:
        raw = f.read()
    if b"\x00" in raw[:4096]:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def chunk_lines(text: str, chunk_chars: int = CHUNK_CHARS,
                overlap: int = OVERLAP_LINES) -> List[Tuple[int, int, str]]:
    """Split text into (start_line, end_line, chunk_text) with 1-based lines."""
    lines = text.splitlines()
    chunks: List[Tuple[int, int, str]] = []
    start = 0
    while start < len(lines):
        end, size = start, 0
        while end < len(lines) and (size == 0 or size + len(lines[end]) <= chunk_chars):
            size += len(lines[end]) + 1
            end += 1
        body = "\n".join(lines[start:end]).strip()
        if body:
            chunks.append((start + 1, end, body))
        if end >= len(lines):
            break
        start = max(end - overlap, start + 1)
    return chunks


def index_folder(engine, root: str = ".") -> IndexReport:
    """Index (or incrementally re-index) every text file under root."""
    root = os.path.abspath(root)
    report = IndexReport()
    db = engine.db
    known = db.indexed_sources(root)          # {source: mtime}
    seen = set()

    for path in iter_text_files(root):
        seen.add(path)
        mtime = os.path.getmtime(path)
        if known.get(path) == mtime:
            report.skipped_unchanged += 1
            continue
        try:
            text = read_text(path)
        except OSError as exc:
            report.errors.append(f"{path}: {exc}")
            continue

        db.delete_by_source(path)
        if not text:
            continue
        chunks = chunk_lines(text)
        if not chunks:
            continue

        rel = os.path.relpath(path, root)
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        vecs = engine.embedder.encode([f"{rel}\n{body}" for _, _, body in chunks])
        items = []
        for i, (s, e, body) in enumerate(chunks):
            items.append({
                "doc_id": hashlib.sha1(f"{path}:{i}".encode()).hexdigest()[:16],
                "text": body,
                "embedding": vecs[i],
                "metadata": {
                    "category": ext, "source": path, "path": rel, "root": root,
                    "start_line": s, "end_line": e, "mtime": mtime,
                },
            })
        db.add_documents(items)
        report.indexed_files += 1
        report.chunks_added += len(items)

    for stale in set(known) - seen:
        db.delete_by_source(stale)
        report.removed_files += 1

    return report
