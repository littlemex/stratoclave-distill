"""Error hierarchy for stratoclave-distill.

The library raises subclasses of :class:`DistillError` so that callers can
catch a single base type if they only want broad-strokes handling, but each
subclass also carries enough information to drive targeted recovery (e.g.
configuration validation versus a transient embedding-provider failure).
"""

from __future__ import annotations


class DistillError(Exception):
    """Base class for all stratoclave-distill errors."""


class ConfigError(DistillError):
    """Raised when the runtime configuration is invalid or incomplete.

    Examples: missing ``DATABASE_URL``, an unknown ``embedding_provider``,
    an embedding dimension that does not match the schema column.
    """


class SchemaError(DistillError):
    """Raised when the database schema is missing or out of sync.

    Triggered when alembic has not been run, when a required extension
    (``vector`` / ``pg_trgm``) is unavailable, or when the deployed dimension
    of the ``embedding`` column disagrees with ``DistillerConfig``.
    """


class IngestError(DistillError):
    """Raised when ingestion of a JSONL transcript fails.

    Wraps backend errors (LLM / embedding) so that callers can choose to
    requeue without inspecting provider-specific exceptions.
    """


class LLMError(DistillError):
    """Raised when the LLM provider returns an error or malformed output."""


class EmbeddingError(DistillError):
    """Raised when the embedding provider returns an error."""


class NotFoundError(DistillError):
    """Raised when a requested entity (session / digest / learning) is absent."""
