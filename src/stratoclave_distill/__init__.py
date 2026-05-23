"""stratoclave-distill: session distillation, learning aggregation, hybrid search.

Public surface area kept intentionally small. Internal modules
(:mod:`stratoclave_distill.db`, :mod:`stratoclave_distill.pipeline`,
:mod:`stratoclave_distill.retrieval`) are importable but their stability is
not guaranteed before v0.1 ships.
"""

from stratoclave_distill.config import (
    DistillerConfig,
    EmbeddingProviderName,
    LLMProviderName,
)
from stratoclave_distill.core import (
    BranchKind,
    BranchState,
    ClaimType,
    ConfigError,
    ConflictResolution,
    ContextPack,
    ContextPackItem,
    DistillError,
    EmbeddingError,
    EmbeddingRecord,
    GroupLearning,
    IngestError,
    Learning,
    LearningConflict,
    LearningScope,
    LLMError,
    NormalizedTurn,
    NotFoundError,
    SchemaError,
    SessionDigest,
    SessionGap,
    SessionPurpose,
)

__all__ = [
    "BranchKind",
    "BranchState",
    "ClaimType",
    "ConfigError",
    "ConflictResolution",
    "ContextPack",
    "ContextPackItem",
    "DistillError",
    "DistillerConfig",
    "EmbeddingError",
    "EmbeddingProviderName",
    "EmbeddingRecord",
    "GroupLearning",
    "IngestError",
    "LLMError",
    "LLMProviderName",
    "Learning",
    "LearningConflict",
    "LearningScope",
    "NormalizedTurn",
    "NotFoundError",
    "SchemaError",
    "SessionDigest",
    "SessionGap",
    "SessionPurpose",
]

__version__ = "0.1.0.dev0"
