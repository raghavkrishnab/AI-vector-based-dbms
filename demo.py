"""
demo.py
-------
Streamlit UI for the SQLite-powered Vector Search Database.

Run with:
    streamlit run demo.py

Features
    * Add your own documents (with an optional category).
    * Load the built-in sample corpus with one click.
    * Semantic search with a similarity score for every hit.
    * Optional category filter (DBMS-style metadata filtering).
    * Live database stats and a browsable table of stored documents.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from sample_data import EXAMPLE_QUERIES, SAMPLE_DOCS
from search_engine import SearchEngine

st.set_page_config(page_title="Vector Search DB", page_icon="🔎", layout="wide")


# --------------------------------------------------------------------------- #
# Engine (cached so the model + DB connection load once per session)
# --------------------------------------------------------------------------- #
@st.cache_resource
def get_engine() -> SearchEngine:
    return SearchEngine(db_path="data/vectors.db")


engine = get_engine()


def score_color(score: float) -> str:
    if score >= 0.75:
        return "#16a34a"   # green  – strong match
    if score >= 0.45:
        return "#d97706"   # amber  – partial match
    return "#dc2626"       # red    – weak match


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🔎 Vector Search Database")
st.caption(
    "Semantic similarity search powered by embeddings + SQLite. "
    "Finds documents by *meaning*, not just keywords."
)

# --------------------------------------------------------------------------- #
# Sidebar: status + data management
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Status")
    st.metric("Documents in DB", engine.count())
    st.write(f"**Embedding backend:**\n\n`{engine.backend}`")
    st.write(f"**Vector dimension:** {engine.embedder.dim}")
    if engine.embedder.backend == "hashing-fallback":
        st.info(
            "Running the offline fallback embedder. Install `sentence-transformers` "
            "and reconnect to the internet for full semantic quality.",
            icon="ℹ️",
        )

    st.divider()
    st.header("📁 Index a Folder")
    folder = st.text_input("Folder path", value=os.getcwd())
    if st.button("Index folder", use_container_width=True):
        if os.path.isdir(folder):
            with st.spinner("Embedding files..."):
                rep = engine.index_folder(folder)
            st.success(
                f"Indexed {rep.indexed_files} files ({rep.chunks_added} chunks), "
                f"{rep.skipped_unchanged} unchanged, {rep.removed_files} removed."
            )
        else:
            st.error("That folder does not exist.")

    st.divider()
    st.header("📦 Sample Data")
    if st.button("Load sample documents", use_container_width=True):
        engine.add_many(SAMPLE_DOCS)
        st.success(f"Loaded {len(SAMPLE_DOCS)} sample documents.")
        st.rerun()

    if st.button("Clear database", type="secondary", use_container_width=True):
        engine.clear()
        st.warning("Database cleared.")
        st.rerun()

# --------------------------------------------------------------------------- #
# Main tabs
# --------------------------------------------------------------------------- #
search_tab, add_tab, browse_tab = st.tabs(["🔍 Search", "➕ Add Document", "🗂️ Browse"])

# ---- Search tab ----------------------------------------------------------- #
with search_tab:
    if engine.count() == 0:
        st.info("Your database is empty. Load the sample documents from the sidebar to try it out.")

    st.write("**Try an example:**")
    ex_cols = st.columns(len(EXAMPLE_QUERIES))
    for i, ex in enumerate(EXAMPLE_QUERIES):
        if ex_cols[i].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["query"] = ex

    query = st.text_input(
        "Search query",
        key="query",
        placeholder="e.g. I enjoy sports",
    )

    c1, c2, c3 = st.columns([1, 1, 2])
    top_k = c1.slider("Top K results", 1, 10, 5)
    min_score = c2.slider("Min similarity", 0.0, 1.0, 0.0, 0.05)

    categories = sorted({d["metadata"].get("category", "") for d in engine.list_documents(1000)})
    categories = [c for c in categories if c]
    category = c3.selectbox("Filter by category", ["(all)"] + categories)
    category = None if category == "(all)" else category

    if query:
        out = engine.search(query, top_k=top_k, min_score=min_score, category=category)
        st.caption(
            f"Searched {out['num_docs_searched']} documents · "
            f"encode {out['encode_ms']} ms · search {out['search_ms']} ms"
        )

        if not out["results"]:
            st.warning("No results above the similarity threshold.")
        for rank, r in enumerate(out["results"], start=1):
            color = score_color(r.score)
            with st.container(border=True):
                left, right = st.columns([5, 1])
                cat = r.metadata.get("category", "—")
                if "path" in r.metadata:
                    m = r.metadata
                    left.markdown(f"**{rank}. `{m['path']}` · lines {m['start_line']}–{m['end_line']}**")
                    left.code(r.text, language=None)
                    left.caption(f"type: `{cat}` · id: `{r.doc_id}`")
                else:
                    left.markdown(f"**{rank}. {r.text}**")
                    left.caption(f"category: `{cat}` · id: `{r.doc_id}`")
                right.markdown(
                    f"<div style='text-align:right;font-size:1.5rem;font-weight:700;"
                    f"color:{color}'>{r.score:.2f}</div>"
                    f"<div style='text-align:right;color:gray;font-size:0.75rem'>cosine</div>",
                    unsafe_allow_html=True,
                )

# ---- Add tab -------------------------------------------------------------- #
with add_tab:
    st.write("Add a new document to the vector database.")
    with st.form("add_form", clear_on_submit=True):
        new_text = st.text_area("Document text", height=120)
        new_cat = st.text_input("Category (optional)", placeholder="e.g. technology")
        submitted = st.form_submit_button("Add document")
        if submitted:
            if new_text.strip():
                meta = {"category": new_cat.strip()} if new_cat.strip() else {}
                doc_id = engine.add(new_text.strip(), metadata=meta)
                st.success(f"Added document `{doc_id}`.")
            else:
                st.error("Document text cannot be empty.")

# ---- Browse tab ----------------------------------------------------------- #
with browse_tab:
    docs = engine.list_documents(limit=1000)
    st.write(f"**{len(docs)} documents stored** (most recent first)")
    if docs:
        df = pd.DataFrame(
            [
                {
                    "doc_id": d["doc_id"],
                    "path": d["metadata"].get("path", ""),
                    "text": d["text"],
                    "category": d["metadata"].get("category", ""),
                }
                for d in docs
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

        del_id = st.text_input("Delete a document by id")
        if st.button("Delete") and del_id.strip():
            ok = engine.delete(del_id.strip())
            st.success("Deleted.") if ok else st.error("No document with that id.")
            if ok:
                st.rerun()
    else:
        st.info("No documents yet.")
