"""Core types and errors for stratoclave-distill."""

from stratoclave_distill.core.errors import (
    ConfigError,
    DistillError,
    EmbeddingError,
    IngestError,
    LLMError,
    NotFoundError,
    SchemaError,
)
from stratoclave_distill.core.types import (
    ContextPack,
    ContextPackItem,
    EmbeddingRecord,
    GroupLearning,
    Learning,
    LearningScope,
    NormalizedTurn,
    SessionDigest,
    SessionPurpose,
)

__all__ = [
    "ConfigError",
    "ContextPack",
    "ContextPackItem",
    "DistillError",
    "EmbeddingError",
    "EmbeddingRecord",
    "GroupLearning",
    "IngestError",
    "LLMError",
    "Learning",
    "LearningScope",
    "NormalizedTurn",
    "NotFoundError",
    "SchemaError",
    "SessionDigest",
    "SessionPurpose",
]
