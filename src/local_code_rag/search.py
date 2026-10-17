"""Route a question to the right index: exact structure first, semantic similarity second."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .store.symbols import SymbolStore
from .store.vectors import VectorStore


@dataclass
class Result:
    kind: str            # "definition" | "caller" | "table_read" | "table_write" | "config" | "semantic"
    title: str
    file_path: str
    line_number: int
    snippet: str
    score: float = 1.0
    extra: dict = field(default_factory=dict)

    @property
    def location(self) -> str:
        return f"{self.file_path}:{self.line_number}"


_IDENT = r"[`'\"]?([A-Za-z_][\w.]*)[`'\"]?"
CALLERS_RE = re.compile(rf"(?:who calls|callers? of|what calls|call sites? (?:for|of)|usages? of)\s+{_IDENT}", re.I)
DEFINITION_RE = re.compile(
    r"(?:where is|where's|definition of|defined|find|show me)\s+(?:the\s+)?"
    rf"(?:function\s+|class\s+|method\s+)?{_IDENT}",
    re.I,
)
TABLE_RE = re.compile(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*){1,2})\b")
CONFIG_RE = re.compile(r"\b(config|setting|yaml|parameter|schedule|configured|set to)\b", re.I)
SYMBOL_RE = re.compile(r"^[A-Za-z_][\w.]*$")
_STOP = {"defined", "the", "a", "an", "is", "it", "this", "that", "where", "how", "what"}


class Searcher:
    def __init__(self, symbols: SymbolStore, vectors: VectorStore | None = None):
        self.symbols = symbols
        self.vectors = vectors

    def search(self, query: str, mode: str = "auto", n: int = 10) -> list[Result]:
        q = query.strip()
        if not q:
            return []
        if mode == "semantic":
            return self.semantic(q, n)
        if mode == "exact":
            return self.exact(q, n)
        return self.auto(q, n)

    def auto(self, q: str, n: int) -> list[Result]:
        if m := CALLERS_RE.search(q):
            found = self.callers(m.group(1), n)
            if found:
                return found
        if (m := DEFINITION_RE.search(q)) and m.group(1).lower() not in _STOP:
            found = self.definitions(m.group(1), n)
            if found:
                return found
        for m in TABLE_RE.finditer(q):
            found = self.tables(m.group(1), n)
            if found:
                return found
        if SYMBOL_RE.match(q):
            found = self.definitions(q, n) or self.config(q, n)
            if found:
                return found
        if CONFIG_RE.search(q):
            hits = [r for word in _keywords(q) for r in self.config(word, n)]
            if hits:
                return _unique(hits)[:n]
        return self.semantic(q, n) if self.vectors else self.exact(q, n)

    def exact(self, q: str, n: int) -> list[Result]:
        results: list[Result] = []
        for word in [q] + _keywords(q):
            results += self.definitions(word, n) + self.tables(word, n) + self.config(word, n)
            if len(results) >= n:
                break
        return _unique(results)[:n]

    # ---- strategies --------------------------------------------------------
    def definitions(self, name: str, n: int) -> list[Result]:
        rows = self.symbols.find_symbol(name) or self.symbols.find_symbol_fuzzy(name, n)
        return [Result("definition", f"{r['kind']} {r['qualname']}", r["file_path"], r["line_number"],
                       r["snippet"] or "", extra={"end_line": r["end_line"], "docstring": r["docstring"]})
                for r in rows[:n]]

    def callers(self, name: str, n: int) -> list[Result]:
        rows = self.symbols.callers_of(name)
        return [Result("caller", f"{r['caller']} calls {r['callee']}", r["caller_file"], r["line_number"],
                       f"{r['caller']}() → {r['callee']}()") for r in rows[:n]]

    def tables(self, table: str, n: int) -> list[Result]:
        rows = self.symbols.table_refs(table)
        return [Result(f"table_{r['ref_type']}", f"{r['ref_type']} {r['table_name']}", r["file_path"],
                       r["line_number"], r["context"] or "") for r in rows[:n]]

    def config(self, text: str, n: int) -> list[Result]:
        rows = self.symbols.config_keys(text)
        return [Result("config", r["key"], r["file_path"], r["line_number"], f"{r['key']}: {r['value']}")
                for r in rows[:n]]

    def semantic(self, q: str, n: int) -> list[Result]:
        if not self.vectors:
            return []
        return [Result("semantic", h["symbol"] or h["file_path"].rsplit("/", 1)[-1], h["file_path"],
                       h["line_start"], h["content"], score=h["score"],
                       extra={"line_end": h["line_end"], "chunk_kind": h["kind"]})
                for h in self.vectors.query(q, n)]


def build_context(results: list[Result], max_chars: int = 8000) -> str:
    """Format results for the LLM, each with a file:line the model can cite."""
    blocks, total = [], 0
    for r in results:
        block = f"[{r.kind}] {r.title} — {r.location}\n```\n{r.snippet}\n```"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def _keywords(q: str) -> list[str]:
    words = re.findall(r"[A-Za-z_][\w.]{2,}", q)
    return [w for w in words if w.lower() not in _STOP and not CONFIG_RE.fullmatch(w)]


def _unique(results: list[Result]) -> list[Result]:
    seen, out = set(), []
    for r in results:
        key = (r.kind, r.file_path, r.line_number)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out
