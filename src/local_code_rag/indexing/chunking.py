"""Split source files into chunks for embedding.

Every non-blank line of a file lands in exactly one chunk, and each chunk records
the real line range it came from, so answers can cite file:line accurately.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str        # "path::start-end"
    file_path: str
    line_start: int
    line_end: int
    content: str
    symbol: str | None   # qualname when the chunk is (part of) a function or class
    kind: str            # "function" | "class" | "module" | "sql" | "config" | "doc"


MAX_LINES = 60


def chunk_file(source: str, rel: str) -> list[Chunk]:
    """Pick a chunker by file extension."""
    lower = rel.lower()
    if lower.endswith(".py"):
        return chunk_python(source, rel)
    if lower.endswith(".sql"):
        return chunk_sql(source, rel)
    if lower.endswith((".md", ".markdown")):
        return chunk_markdown(source, rel)
    return chunk_lines(source.splitlines(), rel, 1, kind="config")


def chunk_python(source: str, rel: str, max_lines: int = MAX_LINES) -> list[Chunk]:
    lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return chunk_lines(lines, rel, 1, kind="module")

    chunks: list[Chunk] = []
    covered: set[int] = set()

    def emit_def(node: ast.AST, prefix: str = "") -> None:
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        end = getattr(node, "end_lineno", node.lineno)
        qual = f"{prefix}{node.name}"
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        if isinstance(node, ast.ClassDef) and end - start + 1 > max_lines:
            # Large class: one chunk for the header/attributes, one per method.
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            for m in methods:
                emit_def(m, prefix=f"{qual}.")
            return  # header and class-level lines are picked up as leftovers
        for a, b in _windows(start, end, max_lines):
            chunks.append(_make(lines, rel, a, b, qual, kind))
        covered.update(range(start, end + 1))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            emit_def(node)

    # Group every remaining line (imports, constants, script code, class headers) into module chunks.
    run_start = None
    for i in range(1, len(lines) + 2):
        inside = i <= len(lines) and i not in covered
        if inside and run_start is None:
            run_start = i
        elif not inside and run_start is not None:
            chunks.extend(chunk_lines(lines[run_start - 1: i - 1], rel, run_start, kind="module"))
            run_start = None

    return sorted(chunks, key=lambda c: c.line_start)


_STATEMENT_END = re.compile(r";\s*(?:--[^\n]*)?$")


def chunk_sql(source: str, rel: str) -> list[Chunk]:
    """One chunk per statement, with exact line numbers (statements end at a line ending in ';')."""
    lines = source.splitlines()
    chunks: list[Chunk] = []
    start = None
    for i, line in enumerate(lines, 1):
        if start is None and line.strip():
            start = i
        if start is not None and _STATEMENT_END.search(line.rstrip()):
            chunks.extend(chunk_lines(lines[start - 1: i], rel, start, kind="sql"))
            start = None
    if start is not None:
        chunks.extend(chunk_lines(lines[start - 1:], rel, start, kind="sql"))
    return chunks


def chunk_markdown(source: str, rel: str) -> list[Chunk]:
    """One chunk per heading section."""
    lines = source.splitlines()
    starts = [i for i, line in enumerate(lines, 1) if line.startswith("#")] or [1]
    if starts[0] != 1:
        starts.insert(0, 1)
    chunks: list[Chunk] = []
    for a, b in zip(starts, starts[1:] + [len(lines) + 1], strict=True):
        heading = lines[a - 1].lstrip("#").strip() if lines[a - 1].startswith("#") else None
        chunks.extend(chunk_lines(lines[a - 1: b - 1], rel, a, kind="doc", symbol=heading))
    return chunks


def chunk_lines(block: list[str], rel: str, first_line: int, kind: str, size: int = MAX_LINES,
                symbol: str | None = None) -> list[Chunk]:
    """Fixed-size windows over a block, trimming blank edges and skipping blank-only windows."""
    chunks = []
    for offset in range(0, len(block), size):
        window = block[offset: offset + size]
        idx = [j for j, line in enumerate(window) if line.strip()]
        if not idx:
            continue
        a = first_line + offset + idx[0]
        b = first_line + offset + idx[-1]
        content = "\n".join(window[idx[0]: idx[-1] + 1])
        chunks.append(Chunk(f"{rel}::{a}-{b}", rel, a, b, content, symbol, kind))
    return chunks


def _windows(start: int, end: int, size: int) -> list[tuple[int, int]]:
    return [(a, min(a + size - 1, end)) for a in range(start, end + 1, size)]


def _make(lines: list[str], rel: str, a: int, b: int, symbol: str, kind: str) -> Chunk:
    return Chunk(f"{rel}::{a}-{b}", rel, a, b, "\n".join(lines[a - 1: b]), symbol, kind)
