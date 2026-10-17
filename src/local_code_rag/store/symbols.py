"""SQLite index for exact lookups: symbols, call edges, table references, config keys."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ..indexing.config_index import ConfigKey
from ..indexing.python_ast import Call, Symbol, TableRef

SCHEMA = """
CREATE TABLE IF NOT EXISTS symbols (
    name TEXT NOT NULL, qualname TEXT NOT NULL, kind TEXT NOT NULL, file_path TEXT NOT NULL,
    line_number INTEGER, end_line INTEGER, docstring TEXT, snippet TEXT
);
CREATE TABLE IF NOT EXISTS calls (
    caller TEXT NOT NULL, caller_file TEXT NOT NULL, line_number INTEGER, callee TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS table_refs (
    table_name TEXT NOT NULL, ref_type TEXT NOT NULL, file_path TEXT NOT NULL,
    line_number INTEGER, context TEXT
);
CREATE TABLE IF NOT EXISTS config_keys (
    key TEXT NOT NULL, value TEXT, file_path TEXT NOT NULL, line_number INTEGER
);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_symbols_qualname ON symbols(qualname COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_calls_callee ON calls(callee);
CREATE INDEX IF NOT EXISTS idx_calls_caller ON calls(caller);
CREATE INDEX IF NOT EXISTS idx_table_refs_name ON table_refs(table_name);
CREATE INDEX IF NOT EXISTS idx_config_key ON config_keys(key);
"""
TABLES = ("symbols", "calls", "table_refs", "config_keys")


class SymbolStore:
    def __init__(self, db_path: str | Path):
        path = Path(db_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)

    # ---- write -------------------------------------------------------------
    def replace_all(self, symbols: list[Symbol], calls: list[Call], refs: list[TableRef],
                    configs: list[ConfigKey]) -> None:
        """Swap the whole index in one transaction, so a failed rebuild leaves the old one intact."""
        with self._conn:
            for t in TABLES:
                self._conn.execute(f"DELETE FROM {t}")
            self._conn.executemany(
                "INSERT INTO symbols VALUES (?,?,?,?,?,?,?,?)",
                [(s.name, s.qualname, s.kind, s.file_path, s.line_number, s.end_line, s.docstring, s.snippet)
                 for s in symbols])
            self._conn.executemany(
                "INSERT INTO calls VALUES (?,?,?,?)",
                [(c.caller, c.caller_file, c.line_number, c.callee) for c in calls])
            self._conn.executemany(
                "INSERT INTO table_refs VALUES (?,?,?,?,?)",
                [(r.table_name, r.ref_type, r.file_path, r.line_number, r.context) for r in refs])
            self._conn.executemany(
                "INSERT INTO config_keys VALUES (?,?,?,?)",
                [(k.key, k.value, k.file_path, k.line_number) for k in configs])

    # ---- read --------------------------------------------------------------
    def find_symbol(self, name: str) -> list[dict]:
        """Exact match on name or qualname (case-insensitive)."""
        return self._rows(
            "SELECT * FROM symbols WHERE name = ? COLLATE NOCASE OR qualname = ? COLLATE NOCASE "
            "ORDER BY file_path, line_number", (name, name))

    def find_symbol_fuzzy(self, partial: str, limit: int = 50) -> list[dict]:
        return self._rows(
            "SELECT * FROM symbols WHERE qualname LIKE ? ORDER BY length(qualname), file_path LIMIT ?",
            (f"%{partial}%", limit))

    def callers_of(self, name: str) -> list[dict]:
        return self._rows(
            "SELECT * FROM calls WHERE callee = ? ORDER BY caller_file, line_number", (name.split(".")[-1],))

    def calls_made_by(self, qualname: str) -> list[dict]:
        return self._rows(
            "SELECT * FROM calls WHERE caller = ? OR caller LIKE ? ORDER BY line_number",
            (qualname, f"%.{qualname}"))

    def table_refs(self, table: str) -> list[dict]:
        return self._rows(
            "SELECT * FROM table_refs WHERE table_name LIKE ? ORDER BY ref_type DESC, file_path, line_number "
            "LIMIT 200", (f"%{table.lower()}%",))

    def config_keys(self, text: str) -> list[dict]:
        like = f"%{text}%"
        return self._rows(
            "SELECT * FROM config_keys WHERE key LIKE ? OR value LIKE ? ORDER BY file_path, line_number "
            "LIMIT 100", (like, like))

    def stats(self) -> dict[str, int]:
        return {t: self._conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in TABLES}

    def close(self) -> None:
        self._conn.close()

    def _rows(self, sql: str, params: tuple) -> list[dict]:
        return [dict(r) for r in self._conn.execute(sql, params).fetchall()]
