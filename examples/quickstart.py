"""Stage A quickstart for stratoclave-distill.

Demonstrates the configuration loader and stub providers without needing a
running Postgres or any provider API key. Stage B adds an end-to-end
ingestion example, and Stage C adds a retrieval example, both of which
will land beside this file.

Run with::

    DATABASE_URL=postgresql+asyncpg://distill:distill@localhost:5432/distill \
        DISTILL_LLM_PROVIDER=stub DISTILL_EMBEDDING_PROVIDER=stub \
        DISTILL_EMBEDDING_DIM=8 python examples/quickstart.py
"""

from __future__ import annotations

import asyncio

from stratoclave_distill import DistillerConfig
from stratoclave_distill.providers import LLMMessage, StubEmbedding, StubLLM


async def _main() -> None:
    cfg = DistillerConfig.from_env()
    print(f"loaded config: provider={cfg.llm_provider} dim={cfg.embedding_dim}")

    llm = StubLLM(responses=['{"purpose": "demo", "polluted": false, "tags": ["demo"]}'])
    completion = await llm.complete([LLMMessage(role="user", content="ping")])
    print(f"llm reply: {completion}")

    embedding = StubEmbedding(dimension=cfg.embedding_dim)
    [vec] = await embedding.embed(["sample"])
    print(f"embedding length: {len(vec)} unit-norm? {abs(sum(v * v for v in vec) - 1.0) < 1e-6}")


if __name__ == "__main__":
    asyncio.run(_main())
