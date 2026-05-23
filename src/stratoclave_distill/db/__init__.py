"""Postgres + pgvector access layer.

Stores are exposed as :class:`typing.Protocol` so the pipeline depends on
a narrow shape rather than on asyncpg or a particular SQL dialect. The
in-memory implementations (:mod:`stratoclave_distill.db.memory`) cover
unit tests; the asyncpg-backed implementation
(:mod:`stratoclave_distill.db.asyncpg`) is what production deployments
use and is exercised by the integration tests.
"""

from stratoclave_distill.db.memory import (
    InMemoryConflictStore,
    InMemoryDigestStore,
    InMemoryGapStore,
    InMemoryLearningStore,
    InMemoryPurposeStore,
    InMemoryWatermarkStore,
)
from stratoclave_distill.db.stores import (
    ConflictStore,
    DigestStore,
    GapStore,
    LearningSearchHit,
    LearningStore,
    PurposeStore,
    RetrievalLane,
    WatermarkStore,
)

__all__ = [
    "ConflictStore",
    "DigestStore",
    "GapStore",
    "InMemoryConflictStore",
    "InMemoryDigestStore",
    "InMemoryGapStore",
    "InMemoryLearningStore",
    "InMemoryPurposeStore",
    "InMemoryWatermarkStore",
    "LearningSearchHit",
    "LearningStore",
    "PurposeStore",
    "RetrievalLane",
    "WatermarkStore",
]
