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

LearningScope = Literal["session", "project", "group", "shared", "experiment"]
"""Where a learning applies. Aligned with DESIGN section 6.4.

Stage B+ widens the vocabulary with ``experiment`` so an experiment branch can
record learnings that are deliberately quarantined from the canonical lane
until the branch is promoted.
"""

DigestKind = Literal["digest", "learning", "group_learning"]
"""Which table a ContextPack item came from. Drives Markdown formatting."""

BranchKind = Literal["main", "experiment"]
"""Whether a session is on the canonical timeline or an experimental branch.

Stage B+ topology lets an ``experiment`` branch sit alongside ``main`` without
duplicating the parent's learnings; promotion merges the experiment back in.
"""

BranchState = Literal["open", "closed", "promoted"]
"""Lifecycle state for a branched session.

- ``open``: actively accepting new turns / learnings.
- ``closed``: no further work, retained for history.
- ``promoted``: an ``experiment`` branch whose learnings have been merged
  into the canonical lane. Downgrade of migration 0002 refuses to run while
  any session is in this state, so operators do not silently lose semantics.
"""

ClaimType = Literal["observation", "interpretation", "signal", "norm"]
"""Epistemic kind of a learning.

- ``observation``: directly witnessed fact (e.g. \"the API returned 503\").
- ``interpretation``: a model of *why* something happened.
- ``signal``: a heuristic worth noticing, not yet a stable rule.
- ``norm``: a recommended practice. The retriever's canonical lane prefers
  ``norm`` and ``observation`` over ``signal`` and ``interpretation``.

When omitted, callers default to ``signal`` semantics (see Distiller).
"""


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
    """The role and health of a single session.

    Stage B+ adds the branching topology fields. Existing rows pick up
    ``branch_kind = 'main'`` and ``branch_state = 'open'`` from the migration
    defaults, so callers that ignore branching see the same behavior as before.
    """

    session_id: str
    purpose: str
    domain_tags: tuple[str, ...] = ()
    success_score: float | None = None
    polluted: bool = False
    pollution_reason: str | None = None
    derived_from_version: str = ""
    derived_at: str = ""
    last_updated_at: str = ""
    parent_session_id: str | None = None
    branched_at_seq: int | None = None
    branch_kind: BranchKind = "main"
    branch_state: BranchState = "open"
    closed_at: str | None = None


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
    claim_type: ClaimType | None = None


ConflictResolution = Literal["open", "merged", "superseded", "coexist"]
"""How a logged contradiction between two learnings was eventually resolved.

- ``open``: the conflict is unresolved and the retriever should flag both
  sides as contested.
- ``merged``: the curator combined them into one learning.
- ``superseded``: one explicitly replaced the other (the loser is archived).
- ``coexist``: both are intentionally kept (they are scoped differently
  enough that they do not actually contradict).
"""


@dataclass(frozen=True, slots=True)
class LearningConflict:
    """A logged contradiction between two learnings.

    Stage B+ surfaces conflicts as first-class rows so the retriever can flag
    contested rules cheaply (via the partial index on ``resolution = 'open'``).
    """

    conflict_id: str
    from_id: str
    to_id: str
    reason: str
    cosine_at_detection: float
    detected_at: str = ""
    resolution: ConflictResolution = "open"


@dataclass(frozen=True, slots=True)
class SessionGap:
    """An unresolved question a session noted but did not answer.

    These are searchable via BM25 (``bm25_text``) and are linked back to the
    learning that eventually resolved them via ``resolved_by_learning``.
    """

    gap_id: str
    session_id: str
    topic: str
    why_unknown: str
    bm25_text: str = ""
    detected_at: str = ""
    resolved_at: str | None = None
    resolved_by_learning: str | None = None


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
