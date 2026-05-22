"""Core dataclasses exchanged across the stratoclave-distill API surface.

All types are :func:`~dataclasses.dataclass(frozen=True, slots=True)` so that
they can be safely shared across coroutines and used as cache keys.

The shapes mirror the persistence layer (see ``migrations/``) but are
deliberately narrower: timestamps are encoded as ISO-8601 strings instead of
``datetime`` to keep the public API JSON-friendly, and only the fields that
external callers need to read are exposed. Internal pipeline stages may carry
richer state via private attributes on their own dataclasses.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

LearningScope = Literal["session", "project", "group", "shared"]
"""Where a learning applies. Aligned with DESIGN section 6.4."""

DigestKind = Literal["digest", "learning", "group_learning"]
"""Which table a ContextPack item came from. Drives Markdown formatting."""


@dataclass(frozen=True, slots=True)
class NormalizedTurn:
    """A single normalized turn extracted from a backend's JSONL output.

    distill receives turns from arbitrary loom-compatible adapters. The
    ``raw_line`` is preserved so re-normalization is possible if the schema
    evolves, and ``seq`` is the monotonic ordering that ``distill_watermarks``
    persists.
    """

    turn_id: str
    session_id: str
    seq: int
    role: str
    text_content: str
    tool_name: str | None
    tool_input: Mapping[str, object] | None
    occurred_at: str
    raw_line: str


@dataclass(frozen=True, slots=True)
class SessionPurpose:
    """The role and health of a single session."""

    session_id: str
    purpose: str
    domain_tags: tuple[str, ...] = ()
    success_score: float | None = None
    polluted: bool = False
    pollution_reason: str | None = None
    derived_from_version: str = ""
    derived_at: str = ""
    last_updated_at: str = ""


@dataclass(frozen=True, slots=True)
class SessionDigest:
    """A compact summary of a session, used for retrieval."""

    digest_id: str
    session_id: str
    version_id: str
    summary_md: str
    bm25_text: str
    extracted_at: str = ""


@dataclass(frozen=True, slots=True)
class Learning:
    """An individual lesson distilled from one or more sessions.

    Mirrors ``learnings`` in the schema. ``superseded_by`` and ``archived_at``
    let the Curator express conflict resolution without deleting history.
    """

    learning_id: str
    scope: LearningScope
    rule: str
    why: str
    triggers: Mapping[str, object] = field(default_factory=dict)
    project_key: str | None = None
    group_id: str | None = None
    source_session: str | None = None
    source_version: str | None = None
    evidence_count: int = 1
    confidence: float = 0.5
    archived_at: str | None = None
    superseded_by: str | None = None
    bm25_text: str = ""
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True, slots=True)
class GroupLearning:
    """A group-level rollup produced by the Aggregator stage."""

    group_learning_id: str
    group_id: str
    summary_md: str
    contributing_learnings: tuple[str, ...]
    bm25_text: str = ""
    created_at: str = ""


@dataclass(frozen=True, slots=True)
class EmbeddingRecord:
    """A precomputed embedding tied to its source text and model identity.

    The ``text_hash`` and ``model`` together act as a content-addressed key
    so callers can deduplicate embedding-provider calls.
    """

    text_hash: str
    model: str
    dimension: int
    vector: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ContextPackItem:
    """One element of a :class:`ContextPack`.

    The ``source_id`` is the primary key of the originating row (digest_id /
    learning_id / group_learning_id) so that UI layers can deep-link back to
    the raw evidence.
    """

    kind: DigestKind
    source_id: str
    text: str
    score: float
    tokens: int


@dataclass(frozen=True, slots=True)
class ContextPack:
    """A budgeted bundle of distilled context, ready to inject into a prompt.

    ``markdown`` is the rendered prompt fragment. ``items`` is the ordered
    list of contributing rows (highest priority first). ``total_tokens`` is
    measured with the same tokenizer used by the budget calculation, so it
    is safe to add to other budgeted text segments.
    """

    markdown: str
    items: Sequence[ContextPackItem]
    total_tokens: int

    def to_markdown(self) -> str:
        """Return the rendered Markdown fragment.

        Provided as a method (rather than relying on attribute access alone)
        so ContextPack matches the call shape documented in ``docs/DESIGN.md``.
        """

        return self.markdown
