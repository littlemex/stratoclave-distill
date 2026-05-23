"""Stage C ContextPacker — turn a :class:`RetrievalResult` into a prompt-ready bundle.

Stage B+ delivered the :class:`Retriever` that splits hits into canonical /
emerging lanes. Stage C composes on top of that with a small, deterministic
formatter that:

1. groups learnings by lane and ``claim_type`` (norm > observation >
   interpretation > signal);
2. renders a Markdown fragment with stable section headings so callers can
   rely on the surface for prompt templating;
3. enforces a ``token_budget`` so the fragment can be added to a turn-level
   prompt without blowing the model's context window.

The token counter is intentionally **approximate** (``len(text) /
chars_per_token``) so the package has no extra runtime dependency on
``tiktoken`` / ``transformers``. Callers that need byte-accurate counting
can wrap :class:`ContextPacker` and pass their own ``token_counter``.

Conflicts and gaps from the :class:`RetrievalResult` are surfaced as their
own sections at the bottom, where the prompt can flag contested rules and
unanswered questions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from stratoclave_distill.core.types import (
    ClaimType,
    ContextPack,
    ContextPackItem,
    Learning,
    LearningConflict,
    SessionGap,
)
from stratoclave_distill.db.stores import LearningSearchHit
from stratoclave_distill.retrieval.retriever import RetrievalResult

# Order in which claim types are rendered within a lane. ``norm`` first
# because rules of conduct are the most directly actionable; ``signal``
# last because unclassified / weak hits land here.
_CLAIM_TYPE_ORDER: tuple[ClaimType, ...] = (
    "norm",
    "observation",
    "interpretation",
    "signal",
)

_CLAIM_TYPE_HEADINGS: dict[ClaimType, str] = {
    "norm": "Norms",
    "observation": "Observations",
    "interpretation": "Interpretations",
    "signal": "Signals",
}

# Default characters-per-token approximation. Empirically close enough to
# ``cl100k_base`` for English / Markdown to keep budgets honest without
# pulling tokenizer code into the runtime.
DEFAULT_CHARS_PER_TOKEN = 4

TokenCounter = Callable[[str], int]
"""A function that maps a string to a token count.

Defaults to :func:`approximate_token_count`. Callers wanting exact counts
can pass a tiktoken-backed wrapper.
"""


def approximate_token_count(text: str, *, chars_per_token: int = DEFAULT_CHARS_PER_TOKEN) -> int:
    """Approximate token count via ``ceil(len(text) / chars_per_token)``.

    Returns ``0`` for empty input. The 4-chars-per-token rule of thumb
    holds well for English Markdown; non-Latin scripts pack tighter so
    callers concerned about JP / CN should override ``chars_per_token``
    or supply their own counter.
    """

    if not text:
        return 0
    if chars_per_token < 1:
        raise ValueError(f"chars_per_token must be >= 1, got {chars_per_token}")
    n = len(text)
    return (n + chars_per_token - 1) // chars_per_token


@dataclass(frozen=True, slots=True)
class _LaneRender:
    """Internal helper bundling a lane's title + per-claim-type buckets."""

    title: str
    buckets: dict[ClaimType, list[LearningSearchHit]]


def _bucket_hits(hits: Sequence[LearningSearchHit]) -> dict[ClaimType, list[LearningSearchHit]]:
    """Group ``hits`` by ``claim_type``; ``None`` falls back to ``signal``.

    Insertion order within a bucket is the lane's own RRF order, which the
    Retriever already sorted, so the rendered output respects relevance
    inside each claim type.
    """

    buckets: dict[ClaimType, list[LearningSearchHit]] = {ct: [] for ct in _CLAIM_TYPE_ORDER}
    for hit in hits:
        ct: ClaimType = hit.learning.claim_type or "signal"
        buckets[ct].append(hit)
    return buckets


def _format_learning_line(learning: Learning) -> str:
    """One bullet per learning. Compact but carries the actionable parts.

    Format: ``- (scope) rule. why=...`` with a trailing ``[id]`` so the
    LLM can cite the source if asked. ``why`` is omitted when empty so
    short-form learnings stay readable.
    """

    pieces = [f"- ({learning.scope}) {learning.rule.strip()}"]
    if learning.why and learning.why.strip():
        pieces.append(f" why={learning.why.strip()}")
    pieces.append(f" [{learning.learning_id}]")
    return "".join(pieces)


def _format_conflict_line(conflict: LearningConflict) -> str:
    return (
        f"- conflict between {conflict.from_id} and {conflict.to_id}"
        f" — {conflict.reason.strip()}"
        f" (cosine={conflict.cosine_at_detection:.2f})"
    )


def _format_gap_line(gap: SessionGap) -> str:
    return f"- ({gap.session_id}) {gap.topic.strip()} — {gap.why_unknown.strip()}"


class ContextPacker:
    """Compose a budgeted Markdown bundle from a :class:`RetrievalResult`.

    Parameters
    ----------
    token_budget:
        Hard upper bound on :attr:`ContextPack.total_tokens`. The packer
        admits learnings in lane / claim-type / RRF order until adding the
        next one would exceed the budget.
    token_counter:
        Function that maps text to a token count. Defaults to the
        approximate counter to avoid pulling in a tokenizer dependency.
    include_conflicts / include_gaps:
        Toggles for the sidecar sections. Default ``True``: when the
        :class:`RetrievalResult` carries no conflicts or gaps the sections
        are simply omitted, which is the same as turning these off.
    title:
        Optional H1 heading. ``None`` produces no title and starts with
        the canonical lane heading.
    """

    __slots__ = (
        "_include_conflicts",
        "_include_gaps",
        "_title",
        "_token_budget",
        "_token_counter",
    )

    def __init__(
        self,
        *,
        token_budget: int,
        token_counter: TokenCounter | None = None,
        include_conflicts: bool = True,
        include_gaps: bool = True,
        title: str | None = "Distilled context",
    ) -> None:
        if token_budget < 1:
            raise ValueError(f"token_budget must be >= 1, got {token_budget}")
        self._token_budget = token_budget
        self._token_counter = token_counter or approximate_token_count
        self._include_conflicts = include_conflicts
        self._include_gaps = include_gaps
        self._title = title

    @property
    def token_budget(self) -> int:
        return self._token_budget

    def pack(self, result: RetrievalResult) -> ContextPack:
        """Render ``result`` as a budgeted :class:`ContextPack`.

        The greedy admission strategy is intentionally simple: lanes are
        traversed in canonical → emerging order, claim types in
        ``_CLAIM_TYPE_ORDER``, and within each bucket hits keep the
        retriever's RRF ordering. The first hit that does not fit causes
        the packer to skip to the *next claim type* — this lets a single
        oversized observation not starve the rest of the bundle.
        """

        canonical = _LaneRender("Canonical", _bucket_hits(result.canonical))
        emerging = _LaneRender("Emerging", _bucket_hits(result.emerging))

        items: list[ContextPackItem] = []
        lines: list[str] = []
        used_tokens = 0

        if self._title:
            heading = f"# {self._title}"
            tokens = self._token_counter(heading)
            if tokens <= self._token_budget:
                lines.append(heading)
                lines.append("")  # blank line after H1
                used_tokens += tokens
        if result.query_text:
            qline = f"_query_: {result.query_text}"
            tokens = self._token_counter(qline)
            if used_tokens + tokens <= self._token_budget:
                lines.append(qline)
                lines.append("")
                used_tokens += tokens

        for lane in (canonical, emerging):
            used_tokens, lane_added = self._emit_lane(
                lane=lane,
                used_tokens=used_tokens,
                lines=lines,
                items=items,
            )
            if not lane_added:
                # No content for this lane fitted at all — skip blank trailer.
                continue

        if self._include_conflicts and result.conflicts:
            used_tokens, _ = self._emit_section(
                title="## Open conflicts",
                lines_to_add=[_format_conflict_line(c) for c in result.conflicts],
                used_tokens=used_tokens,
                lines=lines,
            )
        if self._include_gaps and result.gaps:
            used_tokens, _ = self._emit_section(
                title="## Open gaps",
                lines_to_add=[_format_gap_line(g) for g in result.gaps],
                used_tokens=used_tokens,
                lines=lines,
            )

        markdown = "\n".join(lines).rstrip() + ("\n" if lines else "")
        return ContextPack(
            markdown=markdown,
            items=tuple(items),
            total_tokens=used_tokens,
        )

    def _emit_lane(
        self,
        *,
        lane: _LaneRender,
        used_tokens: int,
        lines: list[str],
        items: list[ContextPackItem],
    ) -> tuple[int, bool]:
        """Render one lane. Returns ``(new_used_tokens, anything_added)``."""

        any_hits = any(lane.buckets[ct] for ct in _CLAIM_TYPE_ORDER)
        if not any_hits:
            return used_tokens, False

        # Tentatively add the lane heading; if even that doesn't fit, bail.
        lane_heading = f"## {lane.title}"
        heading_tokens = self._token_counter(lane_heading)
        if used_tokens + heading_tokens > self._token_budget:
            return used_tokens, False
        lines.append(lane_heading)
        used_tokens += heading_tokens

        added_anything = False
        for ct in _CLAIM_TYPE_ORDER:
            bucket = lane.buckets[ct]
            if not bucket:
                continue
            sub_heading = f"### {_CLAIM_TYPE_HEADINGS[ct]}"
            sub_tokens = self._token_counter(sub_heading)
            if used_tokens + sub_tokens > self._token_budget:
                continue
            tentative_lines = [sub_heading]
            tentative_tokens = sub_tokens
            tentative_items: list[ContextPackItem] = []
            for hit in bucket:
                line = _format_learning_line(hit.learning)
                line_tokens = self._token_counter(line)
                if used_tokens + tentative_tokens + line_tokens > self._token_budget:
                    # Skip this oversized hit but keep trying the rest of
                    # the bucket — a long observation should not starve a
                    # short signal that follows it.
                    continue
                tentative_lines.append(line)
                tentative_tokens += line_tokens
                tentative_items.append(
                    ContextPackItem(
                        kind="learning",
                        source_id=hit.learning.learning_id,
                        text=line,
                        score=hit.rrf_score,
                        tokens=line_tokens,
                    )
                )
            if len(tentative_lines) <= 1:
                # Heading fit but no learning bullets did; drop the heading.
                continue
            lines.extend(tentative_lines)
            used_tokens += tentative_tokens
            items.extend(tentative_items)
            added_anything = True

        if not added_anything:
            # Drop the bare lane heading we tentatively appended.
            lines.pop()
            used_tokens -= heading_tokens
            return used_tokens, False
        # Trailing blank line for readability.
        lines.append("")
        return used_tokens, True

    def _emit_section(
        self,
        *,
        title: str,
        lines_to_add: Iterable[str],
        used_tokens: int,
        lines: list[str],
    ) -> tuple[int, bool]:
        """Append a flat section (conflicts / gaps). Returns updated state."""

        title_tokens = self._token_counter(title)
        if used_tokens + title_tokens > self._token_budget:
            return used_tokens, False
        tentative_lines = [title]
        tentative_tokens = title_tokens
        for line in lines_to_add:
            line_tokens = self._token_counter(line)
            if used_tokens + tentative_tokens + line_tokens > self._token_budget:
                continue
            tentative_lines.append(line)
            tentative_tokens += line_tokens
        if len(tentative_lines) <= 1:
            return used_tokens, False
        lines.extend(tentative_lines)
        lines.append("")
        return used_tokens + tentative_tokens, True


__all__ = [
    "DEFAULT_CHARS_PER_TOKEN",
    "ContextPacker",
    "TokenCounter",
    "approximate_token_count",
]
