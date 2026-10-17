import pytest

from local_code_rag.build import build_index
from local_code_rag.embeddings import HashingEmbedder, make_embedder
from local_code_rag.llm import build_messages
from local_code_rag.search import Searcher, build_context
from local_code_rag.store.symbols import SymbolStore
from local_code_rag.store.vectors import VectorStore


@pytest.fixture
def built(cfg):
    report = build_index(cfg, log=lambda _: None)
    symbols = SymbolStore(cfg.db_path)
    vectors = VectorStore(cfg.vector_path, make_embedder(cfg.embed_model, cfg.ollama_url), collection=cfg.repo_name)
    yield report, Searcher(symbols, vectors), symbols
    symbols.close()


def test_build_report(built):
    report, _, symbols = built
    assert report.files == 5
    assert report.symbols >= 9
    assert report.vectors == report.chunks > 0
    assert symbols.stats()["config_keys"] == report.config_keys


def test_rebuild_replaces_rather_than_appends(cfg, built):
    report, _, _ = built
    again = build_index(cfg, log=lambda _: None)
    assert (again.symbols, again.vectors) == (report.symbols, report.vectors)


def test_callers_route(built):
    _, s, _ = built
    results = s.search("who calls dedupe_latest")
    assert results and all(r.kind == "caller" for r in results)
    assert results[0].title.startswith("load_orders_silver")
    assert results[0].file_path == "pipelines/orders.py"


def test_definition_route(built):
    _, s, _ = built
    results = s.search("where is LateEventHandler defined")
    assert results[0].kind == "definition"
    assert results[0].location == "pipelines/orders.py:32"


def test_method_lookup_by_qualname(built):
    _, s, _ = built
    results = s.search("LateEventHandler.quarantine")
    assert results[0].kind == "definition" and "quarantine" in results[0].title


def test_table_route(built):
    _, s, _ = built
    results = s.search("what writes to shop.gold.daily_revenue?")
    kinds = {r.kind for r in results}
    assert {"table_read", "table_write"} <= kinds
    assert {r.file_path for r in results} == {"pipelines/orders.py", "sql/customer_metrics.sql"}


def test_config_route(built):
    _, s, _ = built
    results = s.search("what schedule is configured for jobs")
    assert results and all(r.kind == "config" for r in results)
    assert any("schedule" in r.title for r in results)


def test_semantic_fallback_finds_relevant_chunk(built):
    _, s, _ = built
    results = s.search("how are late arriving events handled after the watermark", n=3)
    assert results and results[0].kind == "semantic"
    assert any("quarantine" in r.snippet or "watermark" in r.snippet.lower() for r in results)


def test_exact_mode_without_vectors(cfg, built):
    _, _, symbols = built
    results = Searcher(symbols).search("write_table", mode="exact")
    assert results[0].title == "function write_table"


def test_empty_query(built):
    assert built[1].search("   ") == []


def test_build_context_respects_budget(built):
    _, s, _ = built
    results = s.search("orders", mode="semantic", n=10)
    ctx = build_context(results, max_chars=500)
    assert len(ctx) <= 600
    assert ctx.startswith("[semantic] ") and " — " in ctx


def test_build_messages():
    history = [{"role": "user", "content": str(i)} for i in range(10)]
    msgs = build_messages("Q?", "CTX", project="sample-shop", description="A shop", history=history)
    assert msgs[0]["role"] == "system"
    assert "sample-shop" in msgs[0]["content"] and "A shop" in msgs[0]["content"]
    assert len(msgs) == 1 + 6 + 1
    assert "CTX" in msgs[-1]["content"] and "Q?" in msgs[-1]["content"]


def test_hashing_embedder_is_normalised_and_deterministic():
    e = HashingEmbedder(dim=64)
    a, b = e.embed(["load_orders_silver", "loadOrdersSilver"])
    assert abs(sum(x * x for x in a) - 1) < 1e-6
    assert a == e.embed(["load_orders_silver"])[0]
    assert sum(x * y for x, y in zip(a, b, strict=True)) > 0.99  # same sub-tokens
    assert make_embedder("hashing:128", "").embed(["x"])[0].__len__() == 128
