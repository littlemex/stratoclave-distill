"""Unit tests for :class:`Aggregator`.

The Aggregator is intentionally tiny: prompt build, JSON parse,
dataclass mint. The tests pin down the contract the CLI relies on:

- the prompt has exactly one system message and one user message,
  with each contributing learning rendered in input order;
- a well-formed JSON response yields an :class:`AggregationResult`
  whose ``group_learning`` carries the LLM's ``summary_md`` /
  ``bm25_text``, the input ``group_id``, the input ``learning_id``s
  as ``contributing_learnings``, and the injected timestamp;
- malformed JSON / wrong field types / empty fields raise
  :class:`LLMError` — the parser is the single point of trust;
- mixing learnings from multiple ``group_id``s raises before any LLM
  / embedder calls fire (caller bug, not a quietly merged rollup);
- empty learnings sequence and empty group_id raise too;
- the embedder is called once with ``[bm25_text]`` and the returned
  vector is asserted to match :attr:`EmbeddingProvider.dimension`.
"""

from __future__ import annotations

import json

import pytest

from stratoclave_distill.core.errors import LLMError
from stratoclave_distill.core.types import Learning
from stratoclave_distill.pipeline import (
    AggregationResult,
    Aggregator,
    build_aggregate_prompt,
)
from stratoclave_distill.providers.embedding import StubEmbedding
from stratoclave_distill.providers.llm import StubLLM


def _learning(
    learning_id: str,
    *,
    rule: str,
    why: str = "because tests",
    group_id: str = "g-A",
    scope: str = "group",
    claim_type: str | None = "norm",
    evidence_count: int = 3,
) -> Learning:
    return Learning(
        learning_id=learning_id,
        scope=scope,  # type: ignore[arg-type]
        rule=rule,
        why=why,
        group_id=group_id,
        bm25_text=rule,
        claim_type=claim_type,  # type: ignore[arg-type]
        evidence_count=evidence_count,
        created_at="2026-05-25T00:00:00Z",
        updated_at="2026-05-25T00:00:00Z",
    )


def _ok_response(
    *,
    summary_md: str = "Group rollup.\n\n- always commit before deploy",
    bm25_text: str = "Group rollup. always commit before deploy",
) -> str:
    return json.dumps({"summary_md": summary_md, "bm25_text": bm25_text})


def _make(llm: StubLLM, *, dimension: int = 8) -> tuple[Aggregator, StubEmbedding]:
    embedder = StubEmbedding(dimension=dimension)
    aggregator = Aggregator(
        llm,
        embedder,
        clock=lambda: "2026-05-25T12:00:00Z",
    )
    return aggregator, embedder


# --------------------------------------------------------------------------
# build_aggregate_prompt
# --------------------------------------------------------------------------


def test_build_aggregate_prompt_emits_exactly_two_messages() -> None:
    learnings = [_learning("L-1", rule="commit early")]
    messages = build_aggregate_prompt(learnings, group_id="g-A")
    assert len(messages) == 2
    assert messages[0].role == "system"
    assert messages[1].role == "user"


def test_build_aggregate_prompt_renders_each_learning_in_order() -> None:
    learnings = [
        _learning("L-1", rule="commit early"),
        _learning("L-2", rule="never deploy on Friday"),
    ]
    [_, user] = build_aggregate_prompt(learnings, group_id="g-A")
    assert "commit early" in user.content
    assert "never deploy on Friday" in user.content
    # Order is preserved (L-1 appears before L-2 in the rendered transcript).
    assert user.content.index("commit early") < user.content.index("never deploy on Friday")
    assert "g-A" in user.content


def test_build_aggregate_prompt_handles_empty_input_gracefully() -> None:
    [_, user] = build_aggregate_prompt([], group_id="g-A")
    assert "(no learnings)" in user.content


# --------------------------------------------------------------------------
# Aggregator.run — happy path
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_aggregation_result_with_dataclass() -> None:
    llm = StubLLM(responses=[_ok_response()])
    aggregator, _embedder = _make(llm)
    learnings = [
        _learning("L-1", rule="commit early"),
        _learning("L-2", rule="never deploy on Friday"),
    ]

    result = await aggregator.run(learnings, group_id="g-A")

    assert isinstance(result, AggregationResult)
    g = result.group_learning
    assert g.group_id == "g-A"
    assert g.summary_md.startswith("Group rollup.")
    assert "always commit before deploy" in g.bm25_text
    assert g.contributing_learnings == ("L-1", "L-2")
    assert g.created_at == "2026-05-25T12:00:00Z"
    # group_learning_id is freshly minted; only assert it's a non-empty str.
    assert isinstance(g.group_learning_id, str) and g.group_learning_id


@pytest.mark.asyncio
async def test_run_calls_embedder_once_with_bm25_text() -> None:
    llm = StubLLM(responses=[_ok_response(bm25_text="rollup body for embedding")])
    aggregator, embedder = _make(llm)
    learnings = [_learning("L-1", rule="commit early")]

    result = await aggregator.run(learnings, group_id="g-A")

    assert embedder.calls == (("rollup body for embedding",),)
    assert len(result.embedding) == embedder.dimension


# --------------------------------------------------------------------------
# Aggregator.run — error paths (parser + caller)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_rejects_empty_group_id() -> None:
    aggregator, _embedder = _make(StubLLM(responses=[]))
    with pytest.raises(LLMError):
        await aggregator.run([_learning("L-1", rule="x")], group_id="")


@pytest.mark.asyncio
async def test_run_rejects_empty_learnings() -> None:
    aggregator, _embedder = _make(StubLLM(responses=[]))
    with pytest.raises(LLMError):
        await aggregator.run([], group_id="g-A")


@pytest.mark.asyncio
async def test_run_rejects_mixed_group_ids() -> None:
    aggregator, _embedder = _make(StubLLM(responses=[]))
    learnings = [
        _learning("L-1", rule="x", group_id="g-A"),
        _learning("L-2", rule="y", group_id="g-B"),
    ]
    with pytest.raises(LLMError):
        await aggregator.run(learnings, group_id="g-A")


@pytest.mark.asyncio
async def test_run_rejects_malformed_json() -> None:
    llm = StubLLM(responses=["not json"])
    aggregator, _embedder = _make(llm)
    with pytest.raises(LLMError):
        await aggregator.run([_learning("L-1", rule="x")], group_id="g-A")


@pytest.mark.asyncio
async def test_run_rejects_wrong_field_type() -> None:
    llm = StubLLM(responses=[json.dumps({"summary_md": 42, "bm25_text": "ok"})])
    aggregator, _embedder = _make(llm)
    with pytest.raises(LLMError):
        await aggregator.run([_learning("L-1", rule="x")], group_id="g-A")


@pytest.mark.asyncio
async def test_run_rejects_empty_summary_md() -> None:
    llm = StubLLM(responses=[json.dumps({"summary_md": "   ", "bm25_text": "ok"})])
    aggregator, _embedder = _make(llm)
    with pytest.raises(LLMError):
        await aggregator.run([_learning("L-1", rule="x")], group_id="g-A")


@pytest.mark.asyncio
async def test_run_rejects_non_object_response() -> None:
    llm = StubLLM(responses=[json.dumps(["array", "not", "object"])])
    aggregator, _embedder = _make(llm)
    with pytest.raises(LLMError):
        await aggregator.run([_learning("L-1", rule="x")], group_id="g-A")
