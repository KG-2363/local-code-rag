"""Streamlit UI. Start with `lcr ui` or `streamlit run src/local_code_rag/app.py -- --config config.toml`."""

from __future__ import annotations

import argparse
import sys

import streamlit as st

from local_code_rag.config import load_config
from local_code_rag.embeddings import make_embedder
from local_code_rag.llm import OllamaChat, build_messages
from local_code_rag.search import Searcher, build_context
from local_code_rag.store.symbols import SymbolStore
from local_code_rag.store.vectors import VectorStore

KIND_COLOUR = {"definition": "blue", "caller": "orange", "table_read": "green", "table_write": "red",
               "config": "violet", "semantic": "gray"}


def _config_path() -> str:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.toml")
    return ap.parse_known_args(sys.argv[1:])[0].config


@st.cache_resource
def resources(config_path: str):
    cfg = load_config(config_path)
    if not cfg.db_path.exists():
        st.error(f"No index found at {cfg.index_dir}. Run `lcr index` first.")
        st.stop()
    vectors = None
    if cfg.vector_path.exists():
        try:
            vectors = VectorStore(cfg.vector_path, make_embedder(cfg.embed_model, cfg.ollama_url), cfg.repo_name)
            vectors = vectors if vectors.count() else None
        except Exception:
            vectors = None
    return cfg, Searcher(SymbolStore(cfg.db_path), vectors), OllamaChat(cfg.chat_model, cfg.ollama_url)


def render(results) -> None:
    if not results:
        st.info("No results.")
    for r in results:
        st.markdown(f":{KIND_COLOUR.get(r.kind, 'gray')}[{r.kind}] **{r.title}** `{r.location}`")
        lang = "sql" if r.file_path.endswith(".sql") else "yaml" if r.file_path.endswith((".yml", ".yaml")) \
            else "markdown" if r.file_path.endswith(".md") else "python"
        st.code(r.snippet[:800], language=lang)


def main() -> None:
    st.set_page_config(page_title="Local Code RAG", layout="wide")
    cfg, searcher, chat = resources(_config_path())

    with st.sidebar:
        st.title("Local Code RAG")
        st.caption(f"Repo: `{cfg.repo_name}`")
        mode = st.radio("Search mode", ["auto", "exact", "semantic"], horizontal=True)
        n = st.slider("Results", 3, 25, cfg.max_results)
        models = chat.available_models()
        model = st.selectbox("Chat model", sorted(set(models + [cfg.chat_model])),
                             index=sorted(set(models + [cfg.chat_model])).index(cfg.chat_model))
        if not models:
            st.warning("Ollama isn't reachable: search works, chat doesn't.")
        st.divider()
        for k, v in searcher.symbols.stats().items():
            st.metric(k.replace("_", " "), v)
        st.metric("vectors", searcher.vectors.count() if searcher.vectors else 0)

    search_tab, chat_tab = st.tabs(["Search", "Chat"])
    with search_tab:
        q = st.text_input("Query", placeholder='"load_orders", "who calls dedupe_latest", "shop.silver.orders", '
                                                '"how are late events handled"')
        if q:
            render(searcher.search(q, mode=mode, n=n))

    with chat_tab:
        history = st.session_state.setdefault("history", [])
        for m in history:
            st.chat_message(m["role"]).markdown(m["content"])
        if question := st.chat_input(f"Ask about {cfg.repo_name}"):
            st.chat_message("user").markdown(question)
            results = searcher.search(question, n=8)
            messages = build_messages(question, build_context(results), cfg.repo_name, cfg.project_description,
                                      history)
            chat.model = model
            with st.chat_message("assistant"):
                try:
                    answer = st.write_stream(chat.stream(messages))
                except Exception as e:
                    answer = f"Chat failed: {e}. Is Ollama running?"
                    st.error(answer)
                with st.expander(f"Sources ({len(results)})"):
                    render(results)
            history += [{"role": "user", "content": question}, {"role": "assistant", "content": str(answer)}]
        if history and st.button("Clear chat"):
            st.session_state.history = []
            st.rerun()


main()
