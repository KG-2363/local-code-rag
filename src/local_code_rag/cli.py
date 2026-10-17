"""Command line: `lcr index`, `lcr search`, `lcr ask`, `lcr stats`, `lcr ui`."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .config import Config, load_config


def _open(cfg: Config, need_vectors: bool):
    from .embeddings import make_embedder
    from .search import Searcher
    from .store.symbols import SymbolStore

    if not cfg.db_path.exists():
        sys.exit(f"No index at {cfg.index_dir}. Run: lcr index")
    vectors = None
    if need_vectors and cfg.vector_path.exists():
        from .store.vectors import VectorStore

        vectors = VectorStore(cfg.vector_path, make_embedder(cfg.embed_model, cfg.ollama_url), cfg.repo_name)
        if vectors.count() == 0:
            vectors = None
    return Searcher(SymbolStore(cfg.db_path), vectors)


def cmd_index(cfg: Config, args) -> None:
    from .build import build_index

    report = build_index(cfg, skip_vectors=args.skip_vectors)
    print(f"Done in {report.seconds}s: {report.files} files, {report.chunks} chunks. "
          f"Index at {cfg.index_dir}")


def cmd_search(cfg: Config, args) -> None:
    searcher = _open(cfg, need_vectors=args.mode != "exact")
    results = searcher.search(args.query, mode=args.mode, n=args.n)
    if not results:
        print("No results.")
    for r in results:
        score = f" ({r.score:.2f})" if r.kind == "semantic" else ""
        print(f"[{r.kind}] {r.title}{score}\n  {r.location}")
        for line in r.snippet.splitlines()[:6]:
            print(f"    {line}")
        print()


def cmd_ask(cfg: Config, args) -> None:
    from .llm import OllamaChat, build_messages
    from .search import build_context

    searcher = _open(cfg, need_vectors=True)
    results = searcher.search(args.question, n=8)
    messages = build_messages(args.question, build_context(results), cfg.repo_name, cfg.project_description)
    chat = OllamaChat(cfg.chat_model, cfg.ollama_url)
    try:
        for token in chat.stream(messages):
            print(token, end="", flush=True)
        print()
    except Exception as e:  # Ollama not running or model not pulled
        sys.exit(f"\nChat failed ({e}). Is Ollama running and is '{cfg.chat_model}' pulled?")


def cmd_stats(cfg: Config, args) -> None:
    searcher = _open(cfg, need_vectors=True)
    for k, v in searcher.symbols.stats().items():
        print(f"{k:12s} {v}")
    print(f"{'vectors':12s} {searcher.vectors.count() if searcher.vectors else 0}")


def cmd_ui(cfg: Config, args) -> None:
    app = Path(__file__).with_name("app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app), "--", "--config", args.config], check=False)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="lcr", description="Local code search and chat for one repository.")
    p.add_argument("--config", default="config.toml", help="path to config.toml (default: ./config.toml)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("index", help="build or rebuild the index")
    s.add_argument("--skip-vectors", action="store_true", help="exact index only; no embeddings")
    s.set_defaults(func=cmd_index)

    s = sub.add_parser("search", help="search the index")
    s.add_argument("query")
    s.add_argument("--mode", choices=["auto", "exact", "semantic"], default="auto")
    s.add_argument("-n", type=int, default=10)
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("ask", help="ask a question; the answer is grounded in retrieved code")
    s.add_argument("question")
    s.set_defaults(func=cmd_ask)

    sub.add_parser("stats", help="show index sizes").set_defaults(func=cmd_stats)
    sub.add_parser("ui", help="open the Streamlit app").set_defaults(func=cmd_ui)

    args = p.parse_args(argv)
    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))
    args.func(cfg, args)


if __name__ == "__main__":
    main()
