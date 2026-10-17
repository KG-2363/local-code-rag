"""Extract symbols, call edges and table references from Python and SQL files."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from ..files import iter_files


@dataclass
class Symbol:
    name: str
    qualname: str        # e.g. "OrderLoader.load" for a method
    kind: str            # "function" | "method" | "class"
    file_path: str       # relative to repo root, POSIX separators
    line_number: int
    end_line: int
    docstring: str | None
    snippet: str         # signature plus the first few lines


@dataclass
class Call:
    caller: str          # qualname of the enclosing function
    caller_file: str
    line_number: int
    callee: str          # bare name of the called function or method


@dataclass
class TableRef:
    table_name: str      # lower-cased, e.g. "shop.silver.orders"
    ref_type: str        # "read" | "write"
    file_path: str
    line_number: int
    context: str         # the source line


_IDENT = r"[A-Za-z_][\w$]*"
_QUOTED_TABLE = rf"""["']({_IDENT}(?:\.{_IDENT}){{0,2}})["']"""
_SQL_TABLE = rf"`?({_IDENT}(?:`?\.`?{_IDENT}){{0,2}})`?"

# Generic Spark / Delta patterns. Project-specific helpers can be added in config ([tables]).
READ_PATTERNS = [
    rf"spark\.table\(\s*{_QUOTED_TABLE}",
    rf"\.read\.table\(\s*{_QUOTED_TABLE}",
    rf"\.readStream\.table\(\s*{_QUOTED_TABLE}",
    rf"DeltaTable\.forName\(\s*\w+\s*,\s*{_QUOTED_TABLE}",
]
WRITE_PATTERNS = [
    rf"\.saveAsTable\(\s*{_QUOTED_TABLE}",
    rf"\.toTable\(\s*{_QUOTED_TABLE}",
    rf"\.insertInto\(\s*{_QUOTED_TABLE}",
]
SQL_READ = [rf"\b(?:FROM|JOIN|USING)\s+{_SQL_TABLE}"]
SQL_WRITE = [
    rf"\bINSERT\s+(?:INTO|OVERWRITE)\s+(?:TABLE\s+)?{_SQL_TABLE}",
    rf"\bMERGE\s+INTO\s+{_SQL_TABLE}",
    rf"\bCREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP(?:ORARY)?\s+)?(?:TABLE|VIEW)\s+(?:IF\s+NOT\s+EXISTS\s+)?{_SQL_TABLE}",
    rf"\bUPDATE\s+{_SQL_TABLE}\s+SET\b",
    rf"\bDELETE\s+FROM\s+{_SQL_TABLE}",
]
_SQL_NOT_TABLES = {"select", "values", "lateral", "unnest", "table", "the", "a", "an", "dual"}


class CodeIndexer:
    """Walk a repo and index Python symbols, call edges and table references."""

    def __init__(self, repo_root: Path, exclude_dirs: list[str], extra_read: list[str] | None = None,
                 extra_write: list[str] | None = None):
        self.repo_root = repo_root
        self.exclude_dirs = exclude_dirs
        self._read = [re.compile(p) for p in READ_PATTERNS + list(extra_read or [])]
        self._write = [re.compile(p) for p in WRITE_PATTERNS + list(extra_write or [])]
        self._sql_read = [re.compile(p, re.IGNORECASE) for p in SQL_READ]
        self._sql_write = [re.compile(p, re.IGNORECASE) for p in SQL_WRITE]

    def index(self) -> tuple[list[Symbol], list[Call], list[TableRef]]:
        symbols: list[Symbol] = []
        calls: list[Call] = []
        refs: list[TableRef] = []
        for path, rel in iter_files(self.repo_root, [".py", ".sql"], self.exclude_dirs):
            source = path.read_text(encoding="utf-8", errors="replace")
            if path.suffix == ".py":
                s, c = parse_python(source, rel)
                symbols.extend(s)
                calls.extend(c)
                refs.extend(self.python_table_refs(source, rel))
            else:
                refs.extend(self.sql_table_refs(source, rel))
        return symbols, calls, refs

    def python_table_refs(self, source: str, rel: str) -> list[TableRef]:
        refs: list[TableRef] = []
        for i, line in enumerate(source.splitlines(), 1):
            refs += _match(self._read, line, "read", rel, i)
            refs += _match(self._write, line, "write", rel, i)
            # SQL embedded in strings, e.g. spark.sql("SELECT ... FROM db.tbl"): dotted names only
            if re.search(r"\b(SELECT|FROM|INSERT|MERGE|CREATE)\b", line, re.IGNORECASE):
                refs += [r for r in _match(self._sql_read, line, "read", rel, i) if "." in r.table_name]
                refs += [r for r in _match(self._sql_write, line, "write", rel, i) if "." in r.table_name]
        return _dedupe(refs)

    def sql_table_refs(self, source: str, rel: str) -> list[TableRef]:
        refs: list[TableRef] = []
        for i, line in enumerate(source.splitlines(), 1):
            code = line.split("--", 1)[0]
            refs += _match(self._sql_write, code, "write", rel, i)
            refs += _match(self._sql_read, code, "read", rel, i)
        return _dedupe(refs)


def parse_python(source: str, rel: str) -> tuple[list[Symbol], list[Call]]:
    """Parse one file. Calls are attributed to the innermost enclosing function only."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], []
    visitor = _Visitor(rel, source.splitlines())
    visitor.visit(tree)
    return visitor.symbols, visitor.calls


class _Visitor(ast.NodeVisitor):
    def __init__(self, rel: str, lines: list[str]):
        self.rel = rel
        self.lines = lines
        self.scope: list[tuple[str, str]] = []   # (kind, name)
        self.symbols: list[Symbol] = []
        self.calls: list[Call] = []

    def _qual(self, name: str) -> str:
        return ".".join([n for _, n in self.scope] + [name])

    def _add(self, node: ast.AST, kind: str) -> None:
        end = getattr(node, "end_lineno", node.lineno) or node.lineno
        start = node.lineno
        # include decorators in the snippet start
        if getattr(node, "decorator_list", None):
            start = min(d.lineno for d in node.decorator_list)
        snippet = "\n".join(self.lines[start - 1: min(end, node.lineno + 5)])
        self.symbols.append(Symbol(
            name=node.name, qualname=self._qual(node.name), kind=kind, file_path=self.rel,
            line_number=node.lineno, end_line=end, docstring=ast.get_docstring(node), snippet=snippet,
        ))

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add(node, "class")
        self.scope.append(("class", node.name))
        self.generic_visit(node)
        self.scope.pop()

    def _visit_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        kind = "method" if self.scope and self.scope[-1][0] == "class" else "function"
        self._add(node, kind)
        self.scope.append(("function", node.name))
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = _visit_func
    visit_AsyncFunctionDef = _visit_func

    def visit_Call(self, node: ast.Call) -> None:
        func_scope = [n for k, n in self.scope if k == "function"]
        if func_scope:
            callee = _call_name(node)
            if callee:
                self.calls.append(Call(
                    caller=".".join(n for _, n in self.scope), caller_file=self.rel,
                    line_number=node.lineno, callee=callee,
                ))
        self.generic_visit(node)


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _match(patterns: list[re.Pattern], line: str, ref_type: str, rel: str, lineno: int) -> list[TableRef]:
    out = []
    for pat in patterns:
        for m in pat.finditer(line):
            name = m.group(1).replace("`", "").lower()
            if name in _SQL_NOT_TABLES or name.startswith("("):
                continue
            out.append(TableRef(name, ref_type, rel, lineno, line.strip()))
    return out


def _dedupe(refs: list[TableRef]) -> list[TableRef]:
    """A line like `INSERT INTO t SELECT ... FROM t` should report write and read once each."""
    seen: set[tuple[str, str, int]] = set()
    out = []
    for r in refs:
        key = (r.table_name, r.ref_type, r.line_number)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out
