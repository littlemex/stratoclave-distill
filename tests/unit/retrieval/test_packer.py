"""Unit tests for :class:`ContextPacker`.

The packer's contract:

- groups hits by lane (canonical first) and ``claim_type``
  (norm > observation > interpretation > signal);
- enforces ``token_budget`` so the rendered Markdown plus per-line items
  never exceed the cap (with the configured ``token_counter``);
- skips an oversized hit but keeps trying the rest of the bucket so a
  long norm does not starve a short signal that follows it;
- omits empty lanes / claim types; renders sidecar conflict / gap
  sections only when present and toggles are on;
- ``approximate_token_count`` is a pure ``ceil(len/cpt)`` helper that
  returns 0 for empty input and rejects non-positive ``chars_per_token``.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from stratoclave_distill.core.types import (
    ClaimType,
    GroupLearning,
    Learning,
    LearningConflict,
    SessionGap,
)
from stratoclave_distill.db.stores import GroupLearningSearchHit, LearningSearchHit
from stratoclave_distill.retrieval import (
    ContextPacker,
    RetrievalResult,
    approximate_token_count,
)


def _hit(
    learning_id: str,
    *,
    rule: str,
    claim_type: ClaimType | None = None,
    scope: str = "session",
    rrf_score: float = 0.5,
) -> LearningSearchHit:
    learning = Learning(
        learning_id=learning_id,
        scope=scope,  # type: ignore[arg-type]
        rule=rule,
        why="because tests",
        bm25_text=rule,
        evidence_count=1,
        created_at="2026-05-22T00:00:00Z",
        updated_at="2026-05-22T00:00:00Z",
        claim_type=claim_type,
    )
    return LearningSearchHit(
        learning=learning,
        cosine=0.9,
        vector_rank=1,
        bm25_rank=1,
        rrf_score=rrf_score,
    )


def _group_hit(
    group_learning_id: str,
    *,
    group_id: str,
    summary_md: str,
    rrf_score: float = 0.7,
) -> GroupLearningSearchHit:
    g = GroupLearning(
        group_learning_id=group_learning_id,
        group_id=group_id,
        summary_md=summary_md,
        contributing_learnings=(),
        bm25_text=summary_md,
        created_at="2026-05-25T00:00:00Z",
    )
    return GroupLearningSearchHit(
        group=g,
        cosine=0.95,
        vector_rank=1,
        bm25_rank=1,
        rrf_score=rrf_score,
    )


def _result(
    *,
    canonical: Sequence[LearningSearchHit] = (),
    emerging: Sequence[LearningSearchHit] = (),
    conflicts: Sequence[LearningConflict] = (),
    gaps: Sequence[SessionGap] = (),
    groups: Sequence[GroupLearningSearchHit] = (),
    query_text: str = "what should we do about flaky tests?",
) -> RetrievalResult:
    return RetrievalResult(
        query_text=query_text,
        canonical=tuple(canonical),
        emerging=tuple(emerging),
        conflicts=tuple(conflicts),
        gaps=tuple(gaps),
        groups=tuple(groups),
    )


# --------------------------------------------------------------------------
# approximate_token_count
# --------------------------------------------------------------------------


def test_approximate_token_count_empty_string_is_zero() -> None:
    assert approximate_token_count("") == 0


def test_approximate_token_count_ceils_short_strings() -> None:
    # 5 characters at 4 cpt -> ceil(5/4) = 2
    assert approximate_token_count("hello") == 2


def test_approximate_token_count_rejects_non_positive_chars_per_token() -> None:
    with pytest.raises(ValueError, match="chars_per_token must be"):
        approximate_token_count("hi", chars_per_token=0)


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def test_packer_rejects_non_positive_budget() -> None:
    with pytest.raises(ValueError, match="token_budget must be"):
        ContextPacker(token_budget=0)


def test_packer_exposes_budget_via_property() -> None:
    p = ContextPacker(token_budget=512)
    assert p.token_budget == 512


# --------------------------------------------------------------------------
# Rendering: lanes, claim_type ordering, fallback to signal
# --------------------------------------------------------------------------


def test_pack_renders_canonical_before_emerging_with_claim_type_groups() -> None:
    packer = ContextPacker(token_budget=10_000, title="Distilled context")
    result = _result(
        canonical=[
            _hit("L-1", rule="commit before deploy", claim_type="norm", rrf_score=0.9),
            _hit("L-2", rule="env mismatch causes 5xx", claim_type="observation", rrf_score=0.8),
        ],
        emerging=[
            _hit("L-3", rule="vector search may help", claim_type="signal", rrf_score=0.5),
        ],
    )
    pack = packer.pack(result)
    md = pack.markdown
    assert md.startswith("# Distilled context\n")
    assert "_query_: what should we do about flaky tests?" in md
    # Canonical block precedes Emerging block.
    can_idx = md.index("## Canonical")
    em_idx = md.index("## Emerging")
    assert can_idx < em_idx
    # Within canonical, Norms precede Observations.
    norm_idx = md.index("### Norms")
    obs_idx = md.index("### Observations")
    assert can_idx < norm_idx < obs_idx < em_idx
    # Each rule shows up with id citation.
    assert "[L-1]" in md
    assert "[L-2]" in md
    assert "[L-3]" in md
    # Items reflect what landed in the pack, in the same order.
    assert [item.source_id for item in pack.items] == ["L-1", "L-2", "L-3"]


def test_pack_treats_missing_claim_type_as_signal() -> None:
    packer = ContextPacker(token_budget=10_000)
    result = _result(canonical=[_hit("L-9", rule="unclassified", claim_type=None)])
    pack = packer.pack(result)
    assert "### Signals" in pack.markdown
    assert "### Norms" not in pack.markdown


def test_pack_omits_empty_lanes_and_claim_buckets() -> None:
    packer = ContextPacker(token_budget=10_000, title=None)
    result = _result(
        canonical=[_hit("L-1", rule="x", claim_type="norm")],
        # emerging is empty
    )
    pack = packer.pack(result)
    assert "## Canonical" in pack.markdown
    assert "## Emerging" not in pack.markdown
    # No Observations bucket because that claim_type had no hits.
    assert "### Observations" not in pack.markdown


# --------------------------------------------------------------------------
# Token budget enforcement
# --------------------------------------------------------------------------


def test_pack_respects_token_budget_dropping_oversized_hits() -> None:
    # Budget tight enough that the long observation does not fit but a
    # following short signal still does. Verifies that the packer keeps
    # iterating past an oversized hit instead of bailing on the bucket.
    long_text = "x" * 400  # ~100 tokens at default 4 cpt
    short_text = "tip"  # ~1 token
    result = _result(
        canonical=[
            _hit("L-long", rule=long_text, claim_type="norm", rrf_score=0.9),
            _hit("L-short", rule=short_text, claim_type="norm", rrf_score=0.8),
        ],
    )
    packer = ContextPacker(token_budget=40, title=None)
    pack = packer.pack(result)
    ids = [item.source_id for item in pack.items]
    assert "L-long" not in ids
    assert "L-short" in ids
    assert pack.total_tokens <= 40


def test_pack_total_tokens_never_exceeds_budget() -> None:
    packer = ContextPacker(token_budget=80, title="Distilled")
    hits = [
        _hit(f"L-{i}", rule=f"rule {i} body content", claim_type="norm", rrf_score=1.0 - i * 0.01)
        for i in range(20)
    ]
    pack = packer.pack(_result(canonical=hits))
    assert pack.total_tokens <= 80


# --------------------------------------------------------------------------
# Sidecar sections: conflicts and gaps
# --------------------------------------------------------------------------


def test_pack_renders_open_conflicts_section_when_present() -> None:
    conflict = LearningConflict(
        conflict_id="C-1",
        from_id="L-1",
        to_id="L-2",
        reason="contradicts the new norm",
        cosine_at_detection=0.91,
    )
    packer = ContextPacker(token_budget=10_000, title=None)
    pack = packer.pack(_result(conflicts=[conflict]))
    assert "## Open conflicts" in pack.markdown
    assert "L-1" in pack.markdown
    assert "L-2" in pack.markdown
    assert "0.91" in pack.markdown


def test_pack_renders_gaps_section_when_present() -> None:
    gap = SessionGap(
        gap_id="G-1",
        session_id="s-42",
        topic="why does retry fail twice in a row",
        why_unknown="logs are missing the second failure",
        bm25_text="retry double failure",
    )
    packer = ContextPacker(token_budget=10_000, title=None)
    pack = packer.pack(_result(gaps=[gap]))
    assert "## Open gaps" in pack.markdown
    assert "s-42" in pack.markdown
    assert "retry fail twice" in pack.markdown


def test_pack_can_disable_sidecar_sections() -> None:
    conflict = LearningConflict(
        conflict_id="C-1",
        from_id="L-1",
        to_id="L-2",
        reason="r",
        cosine_at_detection=0.9,
    )
    packer = ContextPacker(
        token_budget=10_000,
        title=None,
        include_conflicts=False,
        include_gaps=False,
    )
    pack = packer.pack(_result(conflicts=[conflict]))
    assert "Open conflicts" not in pack.markdown


# --------------------------------------------------------------------------
# Empty result
# --------------------------------------------------------------------------


def test_pack_empty_result_returns_at_most_a_title() -> None:
    packer = ContextPacker(token_budget=200, title="Empty")
    pack = packer.pack(_result(query_text=""))
    assert "Canonical" not in pack.markdown
    assert "Emerging" not in pack.markdown
    # Title still rendered.
    assert pack.markdown.startswith("# Empty\n")
    assert pack.items == ()


# --------------------------------------------------------------------------
# Custom token counter
# --------------------------------------------------------------------------


def test_pack_uses_custom_token_counter() -> None:
    calls: list[str] = []

    def counter(text: str) -> int:
        calls.append(text)
        return 1  # every fragment is 1 token

    packer = ContextPacker(token_budget=10, token_counter=counter, title="T")
    pack = packer.pack(
        _result(
            canonical=[
                _hit("L-1", rule="a", claim_type="norm"),
                _hit("L-2", rule="b", claim_type="norm"),
            ]
        )
    )
    # Title (1) + query line (1) + lane heading (1) + claim heading (1)
    # + two learnings (1+1) = 6 tokens.
    assert pack.total_tokens == 6
    assert calls  # custom counter was actually used


# --------------------------------------------------------------------------
# Group rollups (Stage D)
# --------------------------------------------------------------------------


def test_pack_renders_group_rollups_before_canonical() -> None:
    packer = ContextPacker(token_budget=10_000, title="Distilled")
    result = _result(
        groups=[
            _group_hit("gl-1", group_id="g-A", summary_md="Group A rollup body."),
        ],
        canonical=[
            _hit("L-1", rule="commit before deploy", claim_type="norm"),
        ],
    )
    pack = packer.pack(result)
    md = pack.markdown
    assert "## Group rollups" in md
    assert "### Group rollup: g-A [gl-1]" in md
    assert "Group A rollup body." in md
    # Section ordering: rollups precede the canonical lane.
    assert md.index("## Group rollups") < md.index("## Canonical")
    # Items contain the rollup with kind=group_learning.
    kinds = [item.kind for item in pack.items]
    assert kinds[0] == "group_learning"
    assert pack.items[0].source_id == "gl-1"


def test_pack_skips_groups_section_when_empty() -> None:
    packer = ContextPacker(token_budget=10_000, title=None)
    result = _result(
        canonical=[_hit("L-1", rule="x", claim_type="norm")],
    )
    pack = packer.pack(result)
    assert "## Group rollups" not in pack.markdown


def test_pack_drops_oversized_group_rollup_but_keeps_others() -> None:
    long_body = "x" * 800  # ~200 tokens at default 4 cpt
    short_body = "tip"
    result = _result(
        groups=[
            _group_hit("gl-long", group_id="g-A", summary_md=long_body),
            _group_hit("gl-short", group_id="g-B", summary_md=short_body),
        ],
    )
    packer = ContextPacker(token_budget=60, title=None)
    pack = packer.pack(result)
    ids = [item.source_id for item in pack.items if item.kind == "group_learning"]
    assert "gl-long" not in ids
    assert "gl-short" in ids
    assert pack.total_tokens <= 60


def test_pack_drops_groups_section_when_no_rollup_fits() -> None:
    long_body = "x" * 4000  # way bigger than budget
    result = _result(
        groups=[_group_hit("gl-long", group_id="g-A", summary_md=long_body)],
        canonical=[_hit("L-1", rule="rule", claim_type="norm")],
    )
    packer = ContextPacker(token_budget=80, title=None)
    pack = packer.pack(result)
    assert "## Group rollups" not in pack.markdown
    # The canonical hit still gets rendered.
    assert "L-1" in pack.markdown
