"""Chroma vector store for semantic search over code chunks."""

from __future__ import annotations

import contextlib
import re
from pathlib import Path
from typing import Any

from ..embeddings import Embedder
from ..indexing.chunking import Chunk


class VectorStore:
    def __init__(self, path: str | Path, embedder: Embedder, collection: str):
        import chromadb
        from chromadb.config import Settings

        p = Path(path).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(p), settings=Settings(anonymized_telemetry=False))
        self.embedder = embedder
        # Chroma names: 3-63 chars of [a-zA-Z0-9._-]; include the embedder so vectors never mix models.
        name = re.sub(r"[^a-zA-Z0-9._-]", "-", f"{collection}-{embedder.name}")[:63].strip("-._")
        self._name = name if len(name) >= 3 else f"lcr-{name}"
        self._col = self._client.get_or_create_collection(self._name, metadata={"hnsw:space": "cosine"})

    def reset(self) -> None:
        with contextlib.suppress(Exception):  # collection may not exist yet
            self._client.delete_collection(self._name)
        self._col = self._client.get_or_create_collection(self._name, metadata={"hnsw:space": "cosine"})

    def add(self, chunks: list[Chunk], batch_size: int = 32, progress=None) -> None:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            self._col.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.content for c in batch],
                embeddings=self.embedder.embed([_embed_text(c) for c in batch]),
                metadatas=[{"file_path": c.file_path, "line_start": c.line_start, "line_end": c.line_end,
                            "symbol": c.symbol or "", "kind": c.kind} for c in batch],
            )
            if progress:
                progress(len(batch))

    def query(self, text: str, n: int = 8) -> list[dict[str, Any]]:
        if self._col.count() == 0:
            return []
        res = self._col.query(query_embeddings=self.embedder.embed([text]), n_results=min(n, self._col.count()),
                              include=["documents", "metadatas", "distances"])
        return [
            {"content": doc, **meta, "score": round(1 - dist, 4)}
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True)
        ]

    def count(self) -> int:
        return self._col.count()


def _embed_text(c: Chunk) -> str:
    # Prefix the path and symbol: questions often name the file or function, not just the logic.
    header = f"{c.file_path}" + (f" {c.symbol}" if c.symbol else "")
    return f"{header}\n{c.content}"
