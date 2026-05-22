# stratoclave-distill: Getting Started Guide

**Last updated**: 2026-05-22
**Audience**: First-time contributors and operators

## Introduction

stratoclave-distill is a Pure Python library that turns raw agent JSONL
transcripts into purposes, digests, and learnings, then offers a hybrid
(vector + BM25) retriever on top of them. This guide walks through the
local setup that lets you run the unit tests and exercise the configuration
loader end-to-end.

## Prerequisites

- Python 3.11 or newer (3.12 recommended)
- Docker (or finch / podman) for the local Postgres + pgvector container
- Familiarity with `venv` and `pip install`
- Basic asyncio knowledge

Reference links:

- Python documentation: <https://docs.python.org/3.12/>
- pgvector: <https://github.com/pgvector/pgvector>
- alembic: <https://alembic.sqlalchemy.org/>

## Environment setup

### 1. Clone and install

```bash
git clone https://github.com/littlemex/stratoclave-distill.git
cd stratoclave-distill

python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The last line should print:

```
Successfully installed stratoclave-distill-0.1.0.dev0 ...
```

For real provider calls you also need provider extras:

```bash
pip install -e ".[anthropic,voyage,openai]"
```

### 2. Verify the install

```bash
stratoclave-distill version
```

Expected output:

```
0.1.0.dev0
```

### 3. Validate your environment

```bash
export DATABASE_URL=postgresql+asyncpg://distill:distill@localhost:5432/distill
export DISTILL_LLM_PROVIDER=stub
export DISTILL_EMBEDDING_PROVIDER=stub
export DISTILL_EMBEDDING_DIM=8

stratoclave-distill check-config
```

Expected output:

```json
{
  "database_url": "postgresql+asyncpg://distill:distill@localhost:5432/distill",
  "embedding_dim": 8,
  "embedding_model": "voyage-3",
  "embedding_provider": "stub",
  "llm_model": "claude-haiku-4-5-20251001",
  "llm_provider": "stub"
}
```

### 4. Bring up Postgres + pgvector

```bash
docker compose up -d
```

Wait until the healthcheck reports ``healthy``:

```bash
docker compose ps
```

Then run the migrations:

```bash
DATABASE_URL=postgresql+psycopg://distill:distill@localhost:5432/distill \
  alembic upgrade head
```

You should see five new tables (`session_purposes`, `session_digests`,
`learnings`, `distill_watermarks`, `group_learnings`).

## Quick start (Python API)

The Distiller / Retriever land in Stage B and C. For Stage A you can
already exercise the providers and configuration loader directly:

```python
import asyncio
from stratoclave_distill import DistillerConfig
from stratoclave_distill.providers import StubEmbedding, StubLLM, LLMMessage


async def main() -> None:
    cfg = DistillerConfig.from_env()
    llm = StubLLM(responses=['{"purpose": "demo", "polluted": false}'])
    embedding = StubEmbedding(dimension=cfg.embedding_dim)

    completion = await llm.complete([LLMMessage(role="user", content="hi")])
    [vec] = await embedding.embed(["hello"])
    print(completion, len(vec))


asyncio.run(main())
```

## How the pieces fit together

| Module                                  | Role                                |
|-----------------------------------------|-------------------------------------|
| `stratoclave_distill.config`            | Frozen `DistillerConfig`, env loader |
| `stratoclave_distill.core.types`        | Dataclasses on the public surface   |
| `stratoclave_distill.core.errors`       | Error hierarchy                     |
| `stratoclave_distill.providers`         | LLM and embedding adapters          |
| `stratoclave_distill.cli`               | `stratoclave-distill` entrypoint    |
| `migrations/versions/0001_initial_schema.py` | First Postgres migration       |

## Running the tests

The unit suite has zero external dependencies and runs in seconds:

```bash
pytest
```

You should see all unit tests pass. The integration suite is opt-in: it
needs a live Postgres + pgvector and is skipped by default.

```bash
docker compose up -d
DATABASE_URL=postgresql+psycopg://distill:distill@localhost:5432/distill \
  alembic upgrade head

DISTILL_TEST_DATABASE_URL=postgresql+psycopg://distill:distill@localhost:5432/distill \
  pytest -m integration
```

E2E tests against real provider APIs are also opt-in (Stage B). They require
real API keys and deliberately consume credit, so CI runs them only on
release branches.

```bash
pytest -m e2e
```

## Troubleshooting

### `ModuleNotFoundError: stratoclave_distill`

Confirm your venv is active:

```bash
which python   # should point inside .venv
pip list | grep stratoclave-distill
```

### `ConfigError: DATABASE_URL is required`

`stratoclave-distill check-config` reads from the environment — export
`DATABASE_URL` before running it (any non-empty string works for
``check-config``; the actual connection only matters for migrations).

### `extension "vector" is not available`

The plain ``postgres`` Docker image does not bundle pgvector. Use the
``pgvector/pgvector:pg16`` image as in the included ``docker-compose.yml``.

### `column "embedding" does not exist` after switching providers

Embedding dimension is wired into the schema at migration time. If you
move from Voyage (1024) to OpenAI text-embedding-3-small (1536) you must
re-run the migrations against a fresh database (``alembic downgrade base``
then ``alembic upgrade head`` after exporting the new
``DISTILL_EMBEDDING_DIM``).

## Next steps

- [PROJECT_STATUS.md](./PROJECT_STATUS.md) — current implementation state
- [PROJECT_RULES.md](./PROJECT_RULES.md) — project-specific rules
- [DESIGN.md](./DESIGN.md) — distill's slice of the series design
