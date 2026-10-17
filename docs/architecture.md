# Architecture

```text
           lcr index                                   lcr search / ask / ui
               │                                               │
   ┌───────────┴────────────┐                        ┌─────────┴─────────┐
   │ files.iter_files       │                        │ search.Searcher   │
   │  (extensions, excludes)│                        │  intent routing   │
   └───┬──────────┬─────────┘                        └──┬─────────────┬──┘
       │          │                                     │             │
 python_ast    chunking ──► embeddings ──► Chroma ◄─────┘ semantic    │ exact
 config_index                (Ollama or hashing)                      │
       │                                                              │
       └────────────────────► SQLite (symbols, calls, ────────────────┘
                               table_refs, config_keys)
                                                     │
                                          llm.build_messages ──► Ollama chat (streamed)
```

## Modules

| Module | Responsibility |
| --- | --- |
| `config.py` | `Config` dataclass and TOML loader; only `[repo] path` is required |
| `files.py` | Deterministic, sorted walk of the repo honouring extensions and excluded folders |
| `indexing/python_ast.py` | Symbols, scoped call edges, and PySpark/SQL table references |
| `indexing/config_index.py` | YAML flattened to dotted keys, keeping each key's line number |
| `indexing/chunking.py` | Language-aware chunks with exact line ranges |
| `store/symbols.py` | SQLite exact index, rebuilt in one transaction |
| `store/vectors.py` | Chroma collection per repo *and* embedder, so switching models never mixes vectors |
| `embeddings.py` | `OllamaEmbedder` (batched `/api/embed`) and offline `HashingEmbedder` |
| `search.py` | Routes a question to the right index; `build_context` packs results under a size budget |
| `llm.py` | Grounded system prompt, short chat history, streaming Ollama client |
| `build.py` | Orchestrates a full rebuild and returns a `BuildReport` |
| `cli.py` / `app.py` | `lcr` command and Streamlit UI |

## Design choices

**Exact before semantic.** Structural questions have exact answers, and an embedding model will
happily return a chunk that *mentions* a function instead of the place that *calls* it. Routing
is a handful of regexes; when a route returns nothing, the next one is tried and semantic search
is the final fallback.

**Line numbers are the contract.** Every result carries `file:line`, and the prompt requires the
model to cite them. Chunks are cut on real line boundaries (SQL statements end on the line that
holds the `;`), and a test checks that each chunk's content matches the file at its stated lines.

**Rebuild, don't patch.** A full rebuild of a few thousand files takes seconds for the exact
index. It avoids stale rows from deleted or renamed files, which is the most common way a code
index starts lying.

**Offline by default in tests.** The hashing embedder makes the whole pipeline, Chroma included,
testable in CI without a model server.

## Hashing embedder

Text is tokenised into identifier sub-tokens (`snake_case` and `camelCase` both split), each
token is hashed with BLAKE2b into one of `dim` buckets with a hash-derived sign, and the vector is
L2-normalised. Similar identifiers share buckets, so `loadOrdersSilver` and `load_orders_silver`
are near-identical. It has no notion of synonyms, which is the trade-off for zero setup.
