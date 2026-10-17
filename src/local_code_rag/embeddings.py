"""Embedding backends.

`OllamaEmbedder` gives real semantic search with a local model. `HashingEmbedder`
needs nothing installed: it hashes identifier-aware tokens into a fixed-size vector,
which is effectively lexical search. It keeps the tool usable (and testable) when
Ollama isn't running.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        import ollama

        self.name = f"ollama:{model}"
        self.model = model
        self._client = ollama.Client(host=base_url)

    def embed(self, texts: list[str]) -> list[list[float]]:
        # /api/embed accepts a list, so one request per batch rather than per chunk.
        cleaned = [t if t.strip() else " " for t in texts]
        return [list(v) for v in self._client.embed(model=self.model, input=cleaned).embeddings]


_TOKEN = re.compile(r"[A-Za-z][a-z]+|[A-Z]+(?![a-z])|\d+")


class HashingEmbedder:
    """Bag of identifier sub-words, hashed into `dim` buckets and L2-normalised."""

    def __init__(self, dim: int = 512):
        self.name = f"hashing:{dim}"
        self.dim = dim

    @staticmethod
    def tokens(text: str) -> list[str]:
        # "load_orders_silver" and "loadOrdersSilver" both become ["load", "orders", "silver"]
        return [t.lower() for t in _TOKEN.findall(text.replace("_", " "))]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in self.tokens(text):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def make_embedder(model: str, base_url: str) -> Embedder:
    if model.startswith("hashing"):
        dim = int(model.split(":", 1)[1]) if ":" in model else 512
        return HashingEmbedder(dim)
    return OllamaEmbedder(model, base_url)
