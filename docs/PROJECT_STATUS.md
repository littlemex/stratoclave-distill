# stratoclave-distill: Implementation Status

**Last updated**: 2026-05-22
**Project started**: 2026-05-22
**Current stage**: Stage A (bootstrap)

## Overall progress

### Component status

| Component                      | Tests | Status     | Notes |
|--------------------------------|-------|------------|-------|
| `core.types` (frozen dataclasses) | unit  | done    | Round-tripped through pytest, frozen + slots locked in |
| `core.errors`                  | unit  | done       | Hierarchy verified |
| `config.DistillerConfig`       | unit  | done       | Env loader + invariants enforced |
| Provider abstractions (Stub / Anthropic / OpenAI / Voyage) | unit | done | Lazy SDK imports, dispatch tested |
| CLI (`version`, `check-config`) | unit | done       | Stage B/C subcommands not yet wired |
| Postgres + pgvector schema     | unit (static), integration (opt-in) | done | alembic migration `0001_initial_schema` |
| `Distiller` (Stage 1 LLM extract) | -  | planned    | Stage B |
| `Curator` (dedup / merge / supersede) | - | planned   | Stage B |
| `Aggregator` (group rollup)    | -     | planned    | Stage C |
| `Retriever` (RRF hybrid)       | -     | planned    | Stage C |
| `ContextPacker` (token budget) | -     | planned    | Stage C |
| Watermark store                | -     | planned    | Stage B |

### Integration status

| Integration             | Status      | Notes |
|-------------------------|-------------|-------|
| docker-compose Postgres | done        | `pgvector/pgvector:pg16` |
| alembic migrations      | done        | `DISTILL_EMBEDDING_DIM` env-driven |
| Anthropic LLM           | scaffolded  | Real call exercised in Stage B e2e |
| Voyage embedding        | scaffolded  | Real call exercised in Stage B e2e |
| OpenAI LLM / embedding  | scaffolded  | Optional fallback |
| GitHub Actions CI       | done        | matrix on Python 3.11 / 3.12, ruff + mypy + pytest |

## Completed work

### Stage A — Bootstrap

- Repository skeleton: `LICENSE`, `README.md`, `CONTRIBUTING.md`,
  `SECURITY.md`, `CODE_OF_CONDUCT.md`, `.gitignore`.
- `pyproject.toml` with hatchling, ruff, mypy strict, pytest markers
  (`integration`, `e2e`, `slow`).
- Public dataclasses in `core.types`: `NormalizedTurn`, `SessionPurpose`,
  `SessionDigest`, `Learning`, `GroupLearning`, `EmbeddingRecord`,
  `ContextPackItem`, `ContextPack`. All frozen and `__slots__`-enabled.
- Error hierarchy in `core.errors`: `DistillError` and six subclasses.
- `DistillerConfig` (`config.py`) with env loader, invariants, and
  override priority documented and tested.
- LLM provider abstractions (`providers.llm`): `LLMProvider` Protocol +
  `StubLLM` + `AnthropicLLM` + `OpenAILLM` (lazy SDK imports).
- Embedding provider abstractions (`providers.embedding`): Protocol +
  `StubEmbedding` (deterministic, unit-norm) + `VoyageEmbedding` +
  `OpenAIEmbedding`.
- CLI scaffolding (`cli.py`): `version` and `check-config` subcommands.
- alembic migration `0001_initial_schema`: 5 tables + HNSW + tsvector
  GIN indexes. Embedding dimension is env-driven so providers can swap
  without forking the migration.
- `docker-compose.yml` for local Postgres + pgvector.
- Documentation: this file plus `GETTING_STARTED.md`, `PROJECT_RULES.md`,
  and `DESIGN.md` (which points at the series-wide design in loom).
- Test suite: 119 unit tests covering every Stage A module — type
  invariants, error hierarchy, config edge cases, CLI exit codes,
  schema-migration metadata, and **wire-level coverage** of the real
  LLM / embedding adapters via injected fake SDK clients (so the
  Anthropic / OpenAI / Voyage code paths are exercised without the SDKs
  installed). Plus an opt-in integration test that runs `alembic
  upgrade head` against a live Postgres + pgvector and asserts the
  `vector` / `pg_trgm` extensions, every required HNSW / GIN index, and
  that `alembic downgrade base` cleans the schema up. Total line
  coverage on `src/stratoclave_distill`: 99% (only Protocol fall-through
  branches missed).

## Technical highlights

- **No hard-coded paths, URLs, or credentials.** Every provider knob is
  routed through `DistillerConfig`. See `PROJECT_RULES.md`.
- **Tests reflect real requirements.** Every assertion exists because the
  pipeline depends on the property: dataclass freeze, vector unit-norm,
  threshold ordering, env override priority, etc.
- **Provider stubs are first-class.** Stage B can be developed entirely
  offline because `StubLLM` and `StubEmbedding` mirror the real
  contracts.
- **Schema is dimension-agnostic.** A single migration adapts to Voyage
  (1024) or OpenAI text-embedding-3-small (1536) via
  `DISTILL_EMBEDDING_DIM`.

## Future work

| Priority | Item | Stage |
|----------|------|-------|
| P0 | Distiller LLM extraction with watermarks | B |
| P0 | Curator dedup / merge / supersede | B |
| P0 | Persistent stores (asyncpg-backed `WatermarkStore`, `LearningStore`, `DigestStore`) | B |
| P1 | Retriever RRF hybrid search                | C |
| P1 | ContextPacker token-budget rendering       | C |
| P1 | CLI `ingest` / `query` / `export` / `gc`   | B + C |
| P2 | Aggregator group rollup                    | C |
| P2 | Optional MCP server                        | v0.x |
| P2 | Optional FastAPI HTTP server               | v0.x |

## Team / ownership

| Role             | Owner        | Status      | Current task |
|------------------|--------------|-------------|--------------|
| Maintainer       | littlemex    | active      | Stage A bootstrap |

## Next steps

1. Land Stage A on `main` (slug: `initial`).
2. Begin Stage B: `Distiller` end-to-end against one captured JSONL,
   then add `Curator`.
3. Stage C: `Retriever` + `ContextPacker`, with `claude-capture` JSONL
   used as the e2e fixture.
