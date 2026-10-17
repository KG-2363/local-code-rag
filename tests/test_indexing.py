from local_code_rag.indexing.chunking import chunk_file, chunk_lines, chunk_python, chunk_sql
from local_code_rag.indexing.config_index import index_configs
from local_code_rag.indexing.python_ast import CodeIndexer, parse_python

NESTED = '''
class Outer:
    """Doc."""

    def method(self):
        helper()

        def inner():
            deep()

        return inner


def top():
    Outer().method()
'''


def test_symbols_have_qualnames_and_kinds():
    symbols, _ = parse_python(NESTED, "m.py")
    by_qual = {s.qualname: s for s in symbols}
    assert by_qual["Outer"].kind == "class"
    assert by_qual["Outer"].docstring == "Doc."
    assert by_qual["Outer.method"].kind == "method"
    assert by_qual["Outer.method.inner"].kind == "function"
    assert by_qual["top"].kind == "function"
    assert len(symbols) == len(by_qual), "no duplicate symbols"


def test_calls_attributed_to_innermost_function():
    _, calls = parse_python(NESTED, "m.py")
    edges = {(c.caller, c.callee) for c in calls}
    assert ("Outer.method", "helper") in edges
    assert ("Outer.method.inner", "deep") in edges
    assert ("Outer.method", "deep") not in edges
    assert len(calls) == len(set((c.caller, c.callee, c.line_number) for c in calls))


def test_syntax_error_is_skipped():
    assert parse_python("def broken(:\n", "bad.py") == ([], [])


def test_table_refs_from_python_and_sql(sample_repo):
    _, _, refs = CodeIndexer(sample_repo, [".git"]).index()
    got = {(r.table_name, r.ref_type, r.file_path) for r in refs}
    assert ("shop.bronze.orders_raw", "read", "pipelines/orders.py") in got
    assert ("shop.silver.orders", "read", "pipelines/orders.py") in got
    assert ("shop.gold.daily_revenue", "write", "pipelines/orders.py") in got
    assert ("shop.quarantine.late_orders", "write", "pipelines/orders.py") in got
    assert ("shop.gold.customer_metrics", "write", "sql/customer_metrics.sql") in got
    assert ("shop.silver.customers", "read", "sql/customer_metrics.sql") in got
    assert ("shop.gold.revenue_report", "write", "sql/customer_metrics.sql") in got
    assert ("shop.gold.daily_revenue", "read", "sql/customer_metrics.sql") in got


def test_sql_ref_line_numbers(sample_repo):
    _, _, refs = CodeIndexer(sample_repo, [".git"]).index()
    lines = {(r.table_name, r.ref_type): r.line_number for r in refs if r.file_path.endswith(".sql")}
    assert lines[("shop.silver.customers", "read")] == 9
    assert lines[("shop.gold.revenue_report", "write")] == 15


def test_extra_patterns(tmp_path):
    (tmp_path / "x.py").write_text('df = load_bronze("raw.events")\n')
    _, _, refs = CodeIndexer(tmp_path, [], extra_read=[r'load_bronze\(\s*["\']([\w.]+)["\']']).index()
    assert [(r.table_name, r.ref_type) for r in refs] == [("raw.events", "read")]


def _covered(chunks):
    return {i for c in chunks for i in range(c.line_start, c.line_end + 1)}


def test_python_chunks_cover_every_nonblank_line(sample_repo):
    source = (sample_repo / "pipelines" / "orders.py").read_text()
    chunks = chunk_python(source, "pipelines/orders.py")
    lines = source.splitlines()
    nonblank = {i for i, line in enumerate(lines, 1) if line.strip()}
    assert nonblank <= _covered(chunks)
    # the trailing `if __name__ == "__main__":` block must be indexed too
    assert any("getOrCreate" in c.content for c in chunks)
    for c in chunks:
        assert c.content == "\n".join(lines[c.line_start - 1:c.line_end])


def test_long_function_split_into_windows():
    body = "\n".join(f"    x{i} = {i}" for i in range(150))
    chunks = chunk_python(f"def big():\n{body}\n", "big.py", max_lines=60)
    assert len(chunks) >= 3
    assert all(c.symbol == "big" for c in chunks)
    assert _covered(chunks) == set(range(1, 152))


def test_sql_chunks_have_exact_lines(sample_repo):
    source = (sample_repo / "sql" / "customer_metrics.sql").read_text()
    chunks = chunk_sql(source, "sql/customer_metrics.sql")
    lines = source.splitlines()
    assert len(chunks) == 2
    for c in chunks:
        assert c.content == "\n".join(lines[c.line_start - 1:c.line_end])
    assert "MERGE INTO" in chunks[1].content and chunks[1].line_end == 19


def test_markdown_chunked_per_heading(sample_repo):
    source = (sample_repo / "README.md").read_text()
    chunks = chunk_file(source, "README.md")
    assert [c.symbol for c in chunks][-2:] == ["Layers", "Late events"]


def test_chunk_lines_trims_blank_edges():
    chunks = chunk_lines(["", "", "a", "b", ""], "f.txt", 1, kind="text")
    assert [(c.line_start, c.line_end) for c in chunks] == [(3, 4)]


def test_config_keys_with_lines(sample_repo):
    keys = {k.key: k for k in index_configs(sample_repo, [".git"])}
    assert keys["tables.orders.retention_days"].value == "30"
    assert keys["tables.orders.retention_days"].line_number == 6
    assert keys["jobs[1].schedule"].value == "30 1 * * *"
    assert keys["jobs[1].schedule"].line_number == 14
    assert keys["late_events.watermark_hours"].file_path == "config/pipeline.yaml"
