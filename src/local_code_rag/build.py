"""Build (or rebuild) both indexes for a repository."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from .config import Config
from .embeddings import make_embedder
from .files import iter_files
from .indexing.chunking import Chunk, chunk_file
from .indexing.config_index import index_configs
from .indexing.python_ast import CodeIndexer
from .store.symbols import SymbolStore
from .store.vectors import VectorStore


@dataclass
class BuildReport:
    files: int
    symbols: int
    calls: int
    table_refs: int
    config_keys: int
    chunks: int
    vectors: int
    seconds: float


def collect_chunks(cfg: Config) -> tuple[int, list[Chunk]]:
    files, chunks = 0, []
    for path, rel in iter_files(cfg.repo_path, cfg.extensions, cfg.exclude_dirs):
        files += 1
        chunks.extend(chunk_file(path.read_text(encoding="utf-8", errors="replace"), rel))
    return files, chunks


def build_index(cfg: Config, skip_vectors: bool = False,
                log: Callable[[str], None] = print,
                progress: Callable[[int, int], None] | None = None) -> BuildReport:
    if not cfg.repo_path.is_dir():
        raise FileNotFoundError(f"repo path not found: {cfg.repo_path}")
    t0 = time.time()

    log(f"Indexing {cfg.repo_path}")
    symbols, calls, refs = CodeIndexer(cfg.repo_path, cfg.exclude_dirs, cfg.extra_read_patterns,
                                       cfg.extra_write_patterns).index()
    configs = index_configs(cfg.repo_path, cfg.exclude_dirs)
    store = SymbolStore(cfg.db_path)
    store.replace_all(symbols, calls, refs, configs)
    store.close()
    log(f"Exact index: {len(symbols)} symbols, {len(calls)} calls, {len(refs)} table refs, "
        f"{len(configs)} config keys")

    files, chunks = collect_chunks(cfg)
    n_vectors = 0
    if not skip_vectors:
        embedder = make_embedder(cfg.embed_model, cfg.ollama_url)
        vectors = VectorStore(cfg.vector_path, embedder, collection=cfg.repo_name)
        vectors.reset()
        done = 0

        def tick(k: int) -> None:
            nonlocal done
            done += k
            if progress:
                progress(done, len(chunks))

        vectors.add(chunks, progress=tick)
        n_vectors = vectors.count()
        log(f"Semantic index: {n_vectors} chunks embedded with {embedder.name}")

    return BuildReport(files, len(symbols), len(calls), len(refs), len(configs), len(chunks), n_vectors,
                       round(time.time() - t0, 1))
