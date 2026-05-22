<div align="center">

# stratoclave-distill

**Session distillation, learning aggregation, and hybrid search for the stratoclave family.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#project-status)

*"Boil sessions down to what matters; surface it to the next agent."*

</div>

---

## What is stratoclave-distill?

stratoclave-distill turns raw agent JSONL transcripts into a
queryable knowledge base. Instead of replaying entire conversations into the
next session's prompt, it persists three derived artefacts:

- **Session purposes** — a one-line role and a health flag per session.
- **Session digests** — RAG-friendly summaries used for retrieval.
- **Learnings** — durable rules with `why` / `triggers` so they can be
  injected into future sessions.

The library then offers hybrid (vector + BM25) search and a token-budgeted
Context Pack so callers can ask *"what did we learn about X?"* and get back
a Markdown blob ready to drop into a prompt.

```text
        Raw JSONL transcripts
                |
                v
   +-------------------------+
   |   stratoclave-distill   |  <-- you are here
   |   Distiller / Curator   |
   |   Aggregator / Retriever|
   +-----------+-------------+
               | Postgres + pgvector
               v
   +----------------------------------+
   | session_purposes / digests       |
   | learnings / group_learnings      |
   +----------------------------------+
```

## Project status

**Alpha — v0.1 in active development.** Stage A (this commit) ships the
package skeleton, configuration, schema, and provider abstractions. Stage B
adds the Distiller / Curator pipeline, Stage C adds the Retriever and
Context Packer.

| Component                       | Status     | Notes |
|---------------------------------|------------|-------|
| Core types and config           | v0.1       | `DistillerConfig`, frozen dataclasses, env loader |
| Postgres + pgvector schema      | v0.1       | alembic migration `0001_initial_schema` |
| LLM / Embedding providers       | v0.1       | Anthropic, OpenAI, Voyage, plus deterministic stubs |
| Distiller (Stage 1 extract)     | v0.2 (planned) | Turn-batched LLM extraction with watermarks |
| Curator (dedup / merge / supersede) | v0.2 (planned) | Cosine + LLM judgement |
| Aggregator (group rollup)       | v0.3 (planned) | |
| Retriever (RRF hybrid search)   | v0.3 (planned) | pgvector HNSW + tsvector GIN |
| Context Packer (token budget)   | v0.3 (planned) | |
| CLI                             | v0.1 partial | `version`, `check-config`; rest in Stage B/C |

## Installation

```bash
pip install stratoclave-distill
```

For local development with a live Postgres:

```bash
git clone https://github.com/littlemex/stratoclave-distill.git
cd stratoclave-distill
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,anthropic,voyage,openai]"

docker compose up -d
DATABASE_URL=postgresql+psycopg://distill:distill@localhost:5432/distill \
  alembic upgrade head
```

## Quick start

```python
from stratoclave_distill import DistillerConfig
from stratoclave_distill.providers import StubEmbedding, StubLLM

cfg = DistillerConfig.from_env()
llm = StubLLM(responses=["{...}"])
embedding = StubEmbedding(dimension=cfg.embedding_dim)
```

The Distiller / Retriever entrypoints land in Stage B/C.

## Configuration

stratoclave-distill never hard-codes paths, URLs, or API keys.
Configuration flows through `DistillerConfig` and these environment
variables:

| Variable                     | Purpose                                    |
|------------------------------|--------------------------------------------|
| `DATABASE_URL`               | Postgres connection (required)             |
| `DISTILL_LLM_PROVIDER`       | `anthropic` / `openai` / `stub`            |
| `DISTILL_LLM_MODEL`          | LLM model identifier                       |
| `DISTILL_LLM_API_KEY`        | LLM API key                                |
| `DISTILL_LLM_BASE_URL`       | Optional proxy (e.g. stratoclave)          |
| `DISTILL_EMBEDDING_PROVIDER` | `voyage` / `openai` / `stub`               |
| `DISTILL_EMBEDDING_MODEL`    | Embedding model identifier                 |
| `DISTILL_EMBEDDING_DIM`      | Embedding dimension (must match schema)    |
| `DISTILL_AUTO_TURNS`         | Auto-extract every N turns                 |
| `DISTILL_HNSW_M` / `EFC` / `EF` | pgvector HNSW knobs                     |
| `DISTILL_TAU_MERGE` / `TAU_CONFLICT` | Cosine thresholds for the Curator |

See `docs/PROJECT_RULES.md` for the full no-hardcode policy.

## Series

stratoclave-distill is part of the stratoclave family of OSS projects:

| Project                     | Role                                                       |
|-----------------------------|------------------------------------------------------------|
| [stratoclave](https://github.com/littlemex/stratoclave) | Tenant-aware Bedrock proxy (auth, credit, audit). |
| [stratoclave-loom](https://github.com/littlemex/stratoclave-loom) | Single-agent execution abstraction.       |
| **stratoclave-distill**     | **Session distillation and hybrid search. (this repo)**     |
| stratoclave-atelier         | Web UI, version DB, cross-session orchestration (planned).  |

stratoclave-distill is independent: it has no compile-time or runtime
dependency on the other projects. Use it standalone, or compose it.

## Documentation

- [`docs/DESIGN.md`](./docs/DESIGN.md) — distill's slice of the series design.
- [`docs/GETTING_STARTED.md`](./docs/GETTING_STARTED.md) — install and run.
- [`docs/PROJECT_STATUS.md`](./docs/PROJECT_STATUS.md) — current state.
- [`docs/PROJECT_RULES.md`](./docs/PROJECT_RULES.md) — project-specific rules.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md). Security issues belong in
[SECURITY.md](./SECURITY.md), not in public issues.

## License

Apache 2.0 — see [LICENSE](./LICENSE).
