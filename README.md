# local-code-rag

Ask questions about a codebase and get answers with `file:line` citations, running entirely on
your own machine. Nothing leaves your laptop: the index is SQLite + Chroma on disk, and the
models are served by [Ollama](https://ollama.com).

Built for data-engineering repos (PySpark, SQL, YAML job configs), but it works on any Python
project.

```text
$ lcr search "who calls dedupe_latest"
[caller] load_orders_silver calls dedupe_latest
  pipelines/orders.py:17
    load_orders_silver() → dedupe_latest()

$ lcr search "what writes to shop.gold.daily_revenue"
[table_write] write shop.gold.daily_revenue
  pipelines/orders.py:28
    daily.write.mode("overwrite").saveAsTable("shop.gold.daily_revenue")

[table_read] read shop.gold.daily_revenue
  sql/customer_metrics.sql:16
    USING shop.gold.daily_revenue s

$ lcr ask "How are late events handled?"      # illustrative; wording depends on the model
Events older than the watermark (6 hours by default, config/pipeline.yaml:17) are split off by
LateEventHandler.split (pipelines/orders.py:39) and appended to shop.quarantine.late_orders by
LateEventHandler.quarantine (pipelines/orders.py:45) ...
```

## Why two indexes

Pure vector search is bad at the questions engineers actually ask about code: *who calls this*,
*where is this defined*, *what writes to this table*. Those have exact answers. So questions are
routed first:

| Question looks like | Answered from | Example |
| --- | --- | --- |
| who calls / usages of `X` | call graph (Python AST) | `who calls write_table` |
| where is `X` defined | symbol table | `where is LateEventHandler defined` |
| mentions `schema.table` | table read/write refs (PySpark + SQL) | `what reads shop.silver.orders` |
| config / schedule / setting | flattened YAML keys with line numbers | `what schedule is configured` |
| a bare identifier | symbols, then config keys | `LateEventHandler.quarantine` |
| anything else | semantic search over code chunks | `how is the latest order version kept` |

If the exact route finds nothing, the query falls through to semantic search. `lcr ask` puts the
top results into a grounded prompt that tells the model to answer only from the context and cite
`file:line`.

See [docs/architecture.md](docs/architecture.md) for the details.

## Quick start (no Ollama needed)

```bash
git clone https://github.com/KG-2363/local-code-rag && cd local-code-rag
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

lcr --config examples/config.sample.toml index
lcr --config examples/config.sample.toml search "who calls dedupe_latest"
```

The sample config uses the built-in **hashing embedder** (`embed_model = "hashing"`), which splits
identifiers into sub-tokens (`load_orders_silver` → `load`, `orders`, `silver`) and hashes them
into a vector. It needs no model download and is good enough for identifier-heavy code; use a real
embedding model for prose-style questions.

## Using it on your own repo

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5-coder:7b        # or qwen2.5-coder:3b on 8 GB RAM
pip install -e ".[ui]"

cp config.example.toml config.toml  # set [repo] path
lcr index                           # re-run after big changes; it rebuilds from scratch
lcr search "where is build_daily_revenue defined"
lcr ask "what happens to cancelled orders?"
lcr ui                              # Streamlit app with Search and Chat tabs
lcr stats
```

Relative paths in the config are resolved against the config file's folder. The index lives in
`~/.local-code-rag/<name>/` unless you set `[index] dir`.

### Custom table helpers

Teams usually wrap reads and writes in their own helpers. Teach the indexer about them with
regexes whose first group is the table name:

```toml
[tables]
read_patterns  = ['read_layer\(\s*["\']([\w.]+)["\']']
write_patterns = ['publish_table\(\s*\w+\s*,\s*["\']([\w.]+)["\']']
```

## What gets indexed

- **Python**: functions, classes and methods with qualified names (`Class.method`,
  `outer.inner`), docstrings, line ranges, and call edges attributed to the innermost enclosing
  function. Table refs from `spark.table`, `spark.read.table`, `readStream.table`,
  `DeltaTable.forName`, `saveAsTable`, `toTable`, `insertInto`, and dotted names inside embedded
  SQL strings.
- **SQL**: reads from `FROM` / `JOIN` / `USING`; writes from `INSERT INTO|OVERWRITE`,
  `MERGE INTO`, `CREATE [OR REPLACE] TABLE|VIEW`, `UPDATE`, `DELETE FROM`.
- **YAML**: every leaf flattened to a dotted key (`jobs[1].schedule`) with its line number.
- **Chunks for semantic search**: one per top-level definition (large classes split per method,
  long functions windowed), module-level code in between, one per SQL statement, one per
  Markdown section. Every non-blank line lands in some chunk.

## Limits

- Call resolution is by name, not type: `obj.save()` is recorded as a call to `save`, so callers
  of a common method name can include false positives.
- Tables built from f-strings or variables (`spark.table(f"{db}.orders")`) are not resolved.
- Re-indexing rebuilds everything; there is no incremental update yet.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## Licence

MIT
