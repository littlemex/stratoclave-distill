# stratoclave-distill: Design Notes

**Last updated**: 2026-05-22
**Status**: Stage A baseline

This document is distill's slice of the series-wide design. The full
series design lives in
[stratoclave-loom/docs/DESIGN.md](https://github.com/littlemex/stratoclave-loom/blob/main/docs/DESIGN.md)
section 6; this file mirrors the parts a contributor needs without leaving
this repo.

## 1. Position in the series

```
stratoclave           : auth / credits / audit (Bedrock proxy)
stratoclave-loom      : single-agent execution abstraction (Pure Python lib)
stratoclave-distill   : session distillation + hybrid search (this repo)
stratoclave-atelier   : web UI, version DB, cross-session orchestration (planned)
```

distill is a Pure Python library plus a CLI. It does not run as a
long-lived service in v0.1. It does not depend on loom or atelier; it
takes raw JSONL and a session identifier, persists derived artefacts to
Postgres, and answers retrieval queries.

## 2. Public API surface (Stage A)

```python
from stratoclave_distill import (
    DistillerConfig,
    SessionPurpose,
    SessionDigest,
    Learning,
    GroupLearning,
    EmbeddingRecord,
    ContextPack,
    ContextPackItem,
    NormalizedTurn,
    DistillError,
    ConfigError,
    SchemaError,
    IngestError,
    LLMError,
    EmbeddingError,
    NotFoundError,
)
from stratoclave_distill.providers import (
    LLMProvider,
    EmbeddingProvider,
    StubLLM,
    StubEmbedding,
    AnthropicLLM,
    OpenAILLM,
    VoyageEmbedding,
    OpenAIEmbedding,
    build_llm_provider,
    build_embedding_provider,
)
```

The `Distiller`, `Curator`, `Aggregator`, `Retriever`, and `ContextPacker`
classes will appear in Stage B and Stage C.

## 3. Persistence model

### 3.1 Postgres schema (revision `0001_initial_schema`)

| Table              | Purpose                                              |
|--------------------|------------------------------------------------------|
| `session_purposes` | One row per session: role, tags, pollution flag      |
| `session_digests`  | Compact session summary, indexed by HNSW + tsvector  |
| `learnings`        | Individual rules; supports merge / supersede         |
| `distill_watermarks` | Per-session `published_up_to` for incremental ingest |
| `group_learnings`  | Group-level rollups produced by Aggregator           |

Embedding columns use `vector(N)` with `N` set from
`DISTILL_EMBEDDING_DIM` at migration time. The default 1024 matches
Voyage `voyage-3`; a separate database deployment can use 1536 for
OpenAI `text-embedding-3-small` by re-running the migration with a
different env value.

### 3.2 Indexes

- **HNSW on every embedding column** (`m=16`, `ef_construction=64`).
- **GIN on every `bm25_tsv` column** (generated tsvector in `simple`
  configuration).
- **B-tree on filter columns** (`session_id`, `group_id`, `polluted`,
  `scope` + `archived_at`).

## 4. Pipeline stages

The pipeline is split into three stages so each can be tested and
deployed independently:

1. **Distiller (Stage 1)** — read incremental turns, prompt the LLM with a
   structured extraction template, write `session_purposes` /
   `session_digests` / `learnings` rows. Watermark advances on success.
2. **Curator (Stage 2)** — for each new learning, hybrid-search the top
   K candidates and decide whether to MERGE / SUPERSEDE / INSERT. Uses
   `tau_merge` and `tau_conflict` from `DistillerConfig`.
3. **Aggregator (Stage 3)** — periodically (or on demand) rolls up
   per-session learnings into `group_learnings`.

Stages 1 and 2 are mandatory for v0.1; Stage 3 is targeted at v0.2.

## 5. Retrieval

### 5.1 Hybrid fusion

The Retriever runs vector search and BM25 in parallel, then fuses the
results with Reciprocal Rank Fusion (RRF). The RRF constant comes from
`DistillerConfig.rrf_k` (default 60). RRF is chosen because it requires
no per-query hyperparameter tuning and behaves well across very
different score scales (cosine in [0, 1] vs tsvector in arbitrary
ranges).

### 5.2 Context Packer

`ContextPacker` consumes the fused results and packs them into a
Markdown blob within a token budget:

1. `group_learnings` first (highest signal-to-token ratio).
2. `learnings` (active only) ranked by `confidence * recency`.
3. `session_digests` ranked by query similarity.

The token count is measured with the same tokenizer used downstream
(Anthropic for Claude, tiktoken for OpenAI), so it is safe to add to
other budgeted text segments.

## 6. Provider abstraction

```python
class LLMProvider(Protocol):
    @property
    def model(self) -> str: ...

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str: ...


class EmbeddingProvider(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
```

Provider classes lazy-import the SDK on first call so `pip install
stratoclave-distill` (no extras) is enough to use the stubs.

## 7. Configuration

`DistillerConfig` is the single source of truth for every knob. It is
constructed from environment variables via
`DistillerConfig.from_env(...)` with explicit overrides accepted as
kwargs. Hardcoding any of these values inside `src/` is forbidden by
[`PROJECT_RULES.md`](./PROJECT_RULES.md).

| Variable                     | Default                       | Purpose |
|------------------------------|-------------------------------|---------|
| `DATABASE_URL`               | (required)                    | Postgres connection |
| `DISTILL_LLM_PROVIDER`       | `anthropic`                   | LLM transport choice |
| `DISTILL_LLM_MODEL`          | `claude-haiku-4-5-20251001`   | Model id |
| `DISTILL_LLM_BASE_URL`       | (empty)                       | Optional proxy |
| `DISTILL_LLM_API_KEY`        | (required for non-stub)       | API key |
| `DISTILL_EMBEDDING_PROVIDER` | `voyage`                      | Embedding choice |
| `DISTILL_EMBEDDING_MODEL`    | `voyage-3`                    | Model id |
| `DISTILL_EMBEDDING_DIM`      | `1024`                        | Schema dimension |
| `DISTILL_EMBEDDING_API_KEY`  | (required for non-stub)       | API key |
| `DISTILL_AUTO_TURNS`         | `20`                          | Auto-extract trigger |
| `DISTILL_WORKERS`            | `2`                           | Concurrent workers |
| `DISTILL_HNSW_M`             | `16`                          | HNSW build |
| `DISTILL_HNSW_EFC`           | `64`                          | HNSW build |
| `DISTILL_HNSW_EF`            | `64`                          | HNSW search |
| `DISTILL_TAU_MERGE`          | `0.95`                        | Cosine merge threshold |
| `DISTILL_TAU_CONFLICT`       | `0.80`                        | Cosine conflict threshold |
| `DISTILL_RRF_K`              | `60`                          | RRF constant |
| `DISTILL_CONTEXT_BUDGET_DEFAULT` | `2000`                    | Default Context Pack budget |

## 8. Open questions (revisit during Stage B / C)

- How should `polluted` propagate when a session has been edited mid-run?
  We need a clear policy on whether the digest is invalidated wholesale
  or only past the pollution boundary.
- Embedding cache: should it be a database table (`distill_embedding_cache`)
  or an external KV (Redis)? The schema column is reserved either way.
- MCP server packaging: same wheel under an extra, or a separate
  `stratoclave-distill-mcp` package? Decision deferred until we have a
  concrete consumer.

## 9. References

- Series-wide design:
  [stratoclave-loom/docs/DESIGN.md](https://github.com/littlemex/stratoclave-loom/blob/main/docs/DESIGN.md)
- pgvector documentation: <https://github.com/pgvector/pgvector>
- alembic documentation: <https://alembic.sqlalchemy.org/>
- Voyage AI embeddings: <https://docs.voyageai.com>
- Anthropic Messages API: <https://docs.anthropic.com>
