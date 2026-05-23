"""Unit tests for :class:`Distiller`.

The tests pin down the contract Stage B's CLI relies on:

- the prompt has exactly one system message and one user message,
  with the transcript rendered in seq order and tool calls preserved;
- a well-formed JSON response yields a :class:`DistillationResult`
  whose ``purpose`` / ``digest`` / ``candidate_learnings`` carry the
  expected values, with fresh UUIDs and the injected timestamp;
- malformed JSON raises :class:`LLMError`, and so does any field with
  the wrong type — the parser is the single point of trust between
  the LLM and the persistence layer;
- the embedder is called once with ``[digest, *learnings]`` and the
  returned vectors are zipped onto the right rows;
- ``last_seq`` is the max of the input turns (or 0 for empty input);
- the LLM call surfaces ``max_tokens`` / ``temperature`` from the
  Distiller's constructor (StubLLM ignores them; we assert via the
  recorded calls that the prompt was actually issued).
"""

from __future__ import annotations

import json

import pytest

from stratoclave_distill.core.errors import LLMError
from stratoclave_distill.core.types import NormalizedTurn, SessionPurpose
from stratoclave_distill.pipeline import (
    CandidateLearning,
    DistillationResult,
    Distiller,
    build_distill_prompt,
)
from stratoclave_distill.providers.embedding import StubEmbedding
from stratoclave_distill.providers.llm import LLMMessage, StubLLM

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _turn(
    seq: int,
    *,
    role: str = "user",
    text: str = "hello",
    session_id: str = "s-1",
    tool_name: str | None = None,
    tool_input: dict[str, object] | None = None,
) -> NormalizedTurn:
    return NormalizedTurn(
        turn_id=f"t-{seq}",
        session_id=session_id,
        seq=seq,
        role=role,
        text_content=text,
        tool_name=tool_name,
        tool_input=tool_input,
        occurred_at="2026-05-22T00:00:00Z",
        raw_line="{}",
    )


def _payload(
    *,
    purpose: str = "investigate flaky test",
    domain_tags: list[str] | None = None,
    polluted: bool = False,
    pollution_reason: str | None = None,
    success_score: float | None = 0.9,
    summary_md: str = "# Summary\nThe user fixed a flaky test.",
    bm25_digest: str = "flaky test fix",
    learnings: list[dict[str, object]] | None = None,
) -> str:
    return json.dumps(
        {
            "purpose": {
                "purpose": purpose,
                "domain_tags": domain_tags or ["testing"],
                "success_score": success_score,
                "polluted": polluted,
                "pollution_reason": pollution_reason,
            },
            "digest": {"summary_md": summary_md, "bm25_text": bm25_digest},
            "learnings": learnings if learnings is not None else _default_learnings(),
        }
    )


def _default_learnings() -> list[dict[str, object]]:
    return [
        {
            "scope": "session",
            "rule": "rerun pytest with -p no:cacheprovider to confirm flakes",
            "why": "the cache hid the bug",
            "triggers": {"pytest": True},
            "evidence_count": 1,
            "confidence": 0.7,
            "bm25_text": "pytest cacheprovider flake",
        }
    ]


def _make(
    response: str,
    *,
    embedding_dim: int = 4,
    version_id: str = "v-test-2026-05-22",
    clock_value: str = "2026-05-22T01:00:00Z",
) -> tuple[Distiller, StubLLM, StubEmbedding]:
    llm = StubLLM(responses=[response])
    embedder = StubEmbedding(dimension=embedding_dim)
    distiller = Distiller(
        llm,
        embedder,
        version_id=version_id,
        clock=lambda: clock_value,
    )
    return distiller, llm, embedder


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------


def test_build_distill_prompt_emits_system_and_user_messages() -> None:
    msgs = build_distill_prompt(
        [_turn(0, text="hello"), _turn(1, role="assistant", text="hi")],
        session_id="s-1",
        prior_purpose=None,
    )
    assert len(msgs) == 2
    assert msgs[0].role == "system"
    assert "JSON" in msgs[0].content
    assert msgs[1].role == "user"
    assert "Session id: s-1" in msgs[1].content
    assert "[0000 user] hello" in msgs[1].content
    assert "[0001 assistant] hi" in msgs[1].content


def test_build_distill_prompt_renders_tool_calls() -> None:
    msgs = build_distill_prompt(
        [_turn(0, role="tool_use", text="", tool_name="shell.run", tool_input={"cmd": "ls"})],
        session_id="s-1",
        prior_purpose=None,
    )
    user = msgs[1].content
    assert "tool=shell.run(cmd)" in user


def test_build_distill_prompt_handles_empty_session() -> None:
    msgs = build_distill_prompt([], session_id="s-1", prior_purpose=None)
    assert "(empty session)" in msgs[1].content
    assert "Transcript (0 turns)" in msgs[1].content


def test_build_distill_prompt_includes_prior_purpose() -> None:
    prior = SessionPurpose(session_id="s-1", purpose="prior goal")
    msgs = build_distill_prompt([_turn(0)], session_id="s-1", prior_purpose=prior)
    assert "prior goal" in msgs[1].content


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_distill_happy_path_returns_populated_result() -> None:
    distiller, llm, embedder = _make(_payload())
    result = await distiller.distill([_turn(0), _turn(1)], session_id="s-1")

    assert isinstance(result, DistillationResult)
    assert result.purpose.session_id == "s-1"
    assert result.purpose.purpose == "investigate flaky test"
    assert result.purpose.domain_tags == ("testing",)
    assert result.purpose.success_score == pytest.approx(0.9)
    assert result.purpose.polluted is False
    assert result.purpose.derived_from_version == "v-test-2026-05-22"
    assert result.purpose.derived_at == "2026-05-22T01:00:00Z"

    assert result.digest.session_id == "s-1"
    assert result.digest.version_id == "v-test-2026-05-22"
    assert result.digest.summary_md.startswith("# Summary")
    # digest_id is a fresh UUID
    assert len(result.digest.digest_id) == 36

    assert len(result.candidate_learnings) == 1
    cand = result.candidate_learnings[0]
    assert isinstance(cand, CandidateLearning)
    assert cand.learning.rule.startswith("rerun pytest")
    assert cand.learning.scope == "session"
    assert cand.learning.source_session == "s-1"
    assert cand.learning.source_version == "v-test-2026-05-22"
    assert cand.learning.evidence_count == 1
    assert cand.learning.confidence == pytest.approx(0.7)
    assert cand.learning.created_at == "2026-05-22T01:00:00Z"

    # last_seq is the max seq of the input
    assert result.last_seq == 1

    # one LLM call, one embedding batch sized 1 (digest) + 1 (learning) = 2
    assert len(llm.calls) == 1
    assert len(embedder.calls) == 1
    assert len(embedder.calls[0]) == 2
    assert len(result.digest_embedding) == 4
    assert len(cand.embedding) == 4


@pytest.mark.asyncio
async def test_distill_empty_learnings_yields_zero_candidates() -> None:
    distiller, _, embedder = _make(_payload(learnings=[]))
    result = await distiller.distill([_turn(0)], session_id="s-1")
    assert result.candidate_learnings == ()
    # still embeds the digest
    assert len(embedder.calls[0]) == 1


@pytest.mark.asyncio
async def test_distill_empty_turns_yields_last_seq_zero() -> None:
    distiller, _, _ = _make(_payload(learnings=[]))
    result = await distiller.distill([], session_id="s-1")
    assert result.last_seq == 0


@pytest.mark.asyncio
async def test_distill_polluted_session_round_trips_reason() -> None:
    distiller, _, _ = _make(_payload(polluted=True, pollution_reason="user typed lorem ipsum"))
    result = await distiller.distill([_turn(0)], session_id="s-1")
    assert result.purpose.polluted is True
    assert result.purpose.pollution_reason == "user typed lorem ipsum"


@pytest.mark.asyncio
async def test_distill_uses_prior_purpose_in_prompt() -> None:
    distiller, llm, _ = _make(_payload())
    prior = SessionPurpose(session_id="s-1", purpose="we already knew this goal")
    await distiller.distill([_turn(0)], session_id="s-1", prior_purpose=prior)
    user_msg = llm.calls[0][1]
    assert isinstance(user_msg, LLMMessage)
    assert "we already knew this goal" in user_msg.content


# --------------------------------------------------------------------------
# Error cases
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_distill_rejects_invalid_json() -> None:
    distiller, _, _ = _make("not json at all")
    with pytest.raises(LLMError, match=r"distill JSON could not be parsed"):
        await distiller.distill([_turn(0)], session_id="s-1")


@pytest.mark.asyncio
async def test_distill_rejects_non_object_top_level() -> None:
    distiller, _, _ = _make("[]")
    with pytest.raises(LLMError, match=r"top-level value must be an object"):
        await distiller.distill([_turn(0)], session_id="s-1")


@pytest.mark.asyncio
async def test_distill_rejects_missing_purpose_section() -> None:
    distiller, _, _ = _make(json.dumps({"digest": {}, "learnings": []}))
    with pytest.raises(LLMError, match=r"'purpose' must be an object"):
        await distiller.distill([_turn(0)], session_id="s-1")


@pytest.mark.asyncio
async def test_distill_rejects_unknown_scope() -> None:
    bad = _default_learnings()
    bad[0]["scope"] = "team"  # not in the allowlist
    distiller, _, _ = _make(_payload(learnings=bad))
    with pytest.raises(LLMError, match=r"learning scope must be one of"):
        await distiller.distill([_turn(0)], session_id="s-1")


@pytest.mark.asyncio
async def test_distill_rejects_non_int_evidence_count() -> None:
    bad = _default_learnings()
    bad[0]["evidence_count"] = "1"
    distiller, _, _ = _make(_payload(learnings=bad))
    with pytest.raises(LLMError, match=r"evidence_count' must be an integer"):
        await distiller.distill([_turn(0)], session_id="s-1")


@pytest.mark.asyncio
async def test_distill_rejects_bool_evidence_count() -> None:
    """``True`` is technically an int subclass; the parser must reject it."""

    bad = _default_learnings()
    bad[0]["evidence_count"] = True
    distiller, _, _ = _make(_payload(learnings=bad))
    with pytest.raises(LLMError, match=r"evidence_count' must be an integer"):
        await distiller.distill([_turn(0)], session_id="s-1")


@pytest.mark.asyncio
async def test_distill_rejects_non_object_triggers() -> None:
    bad = _default_learnings()
    bad[0]["triggers"] = ["pytest"]
    distiller, _, _ = _make(_payload(learnings=bad))
    with pytest.raises(LLMError, match=r"'triggers' must be an object"):
        await distiller.distill([_turn(0)], session_id="s-1")


@pytest.mark.asyncio
async def test_distill_rejects_non_string_purpose() -> None:
    payload = json.loads(_payload())
    payload["purpose"]["purpose"] = 42
    distiller, _, _ = _make(json.dumps(payload))
    with pytest.raises(LLMError, match=r"'purpose' must be a string"):
        await distiller.distill([_turn(0)], session_id="s-1")


@pytest.mark.asyncio
async def test_distill_rejects_non_bool_polluted() -> None:
    payload = json.loads(_payload())
    payload["purpose"]["polluted"] = "yes"
    distiller, _, _ = _make(json.dumps(payload))
    with pytest.raises(LLMError, match=r"'polluted' must be a boolean"):
        await distiller.distill([_turn(0)], session_id="s-1")


@pytest.mark.asyncio
async def test_distill_rejects_non_string_domain_tag() -> None:
    payload = json.loads(_payload())
    payload["purpose"]["domain_tags"] = ["ok", 123]
    distiller, _, _ = _make(json.dumps(payload))
    with pytest.raises(LLMError, match=r"'domain_tags' elements must be strings"):
        await distiller.distill([_turn(0)], session_id="s-1")


@pytest.mark.asyncio
async def test_distill_rejects_session_mismatch() -> None:
    distiller, _, _ = _make(_payload())
    other = _turn(0, session_id="other")
    with pytest.raises(LLMError, match=r"expected 's-1'"):
        await distiller.distill([other], session_id="s-1")


def test_distiller_requires_version_id() -> None:
    with pytest.raises(LLMError, match=r"non-empty version_id"):
        Distiller(StubLLM(responses=[""]), StubEmbedding(dimension=2), version_id="")


@pytest.mark.asyncio
async def test_distill_requires_session_id() -> None:
    distiller, _, _ = _make(_payload())
    with pytest.raises(LLMError, match=r"non-empty session_id"):
        await distiller.distill([_turn(0)], session_id="")


@pytest.mark.asyncio
async def test_distill_rejects_non_string_pollution_reason() -> None:
    payload = json.loads(_payload())
    payload["purpose"]["pollution_reason"] = 7
    distiller, _, _ = _make(json.dumps(payload))
    with pytest.raises(LLMError, match=r"pollution_reason' must be a string or null"):
        await distiller.distill([_turn(0)], session_id="s-1")


@pytest.mark.asyncio
async def test_distill_collapses_empty_pollution_reason_to_none() -> None:
    payload = json.loads(_payload())
    payload["purpose"]["pollution_reason"] = ""
    distiller, _, _ = _make(json.dumps(payload))
    result = await distiller.distill([_turn(0)], session_id="s-1")
    assert result.purpose.pollution_reason is None


@pytest.mark.asyncio
async def test_distill_rejects_non_numeric_success_score() -> None:
    payload = json.loads(_payload())
    payload["purpose"]["success_score"] = "high"
    distiller, _, _ = _make(json.dumps(payload))
    with pytest.raises(LLMError, match=r"'success_score' must be a number or null"):
        await distiller.distill([_turn(0)], session_id="s-1")


@pytest.mark.asyncio
async def test_distill_rejects_non_array_domain_tags() -> None:
    payload = json.loads(_payload())
    payload["purpose"]["domain_tags"] = "testing"
    distiller, _, _ = _make(json.dumps(payload))
    with pytest.raises(LLMError, match=r"'domain_tags' must be an array of strings"):
        await distiller.distill([_turn(0)], session_id="s-1")


@pytest.mark.asyncio
async def test_distill_rejects_non_array_learnings() -> None:
    payload = json.loads(_payload())
    payload["learnings"] = {"not": "a list"}
    distiller, _, _ = _make(json.dumps(payload))
    with pytest.raises(LLMError, match=r"'learnings' must be an array"):
        await distiller.distill([_turn(0)], session_id="s-1")


@pytest.mark.asyncio
async def test_distill_rejects_non_object_learning_item() -> None:
    payload = json.loads(_payload())
    payload["learnings"] = ["not an object"]
    distiller, _, _ = _make(json.dumps(payload))
    with pytest.raises(LLMError, match=r"learnings\[0\] must be an object"):
        await distiller.distill([_turn(0)], session_id="s-1")


@pytest.mark.asyncio
async def test_distill_rejects_missing_digest_section() -> None:
    payload = json.loads(_payload())
    payload["digest"] = "not an object"
    distiller, _, _ = _make(json.dumps(payload))
    with pytest.raises(LLMError, match=r"'digest' must be an object"):
        await distiller.distill([_turn(0)], session_id="s-1")


@pytest.mark.asyncio
async def test_distill_rejects_bool_confidence() -> None:
    bad = _default_learnings()
    bad[0]["confidence"] = True
    distiller, _, _ = _make(_payload(learnings=bad))
    with pytest.raises(LLMError, match=r"'confidence' must be a number"):
        await distiller.distill([_turn(0)], session_id="s-1")


@pytest.mark.asyncio
async def test_distill_rejects_string_confidence() -> None:
    bad = _default_learnings()
    bad[0]["confidence"] = "high"
    distiller, _, _ = _make(_payload(learnings=bad))
    with pytest.raises(LLMError, match=r"'confidence' must be a number"):
        await distiller.distill([_turn(0)], session_id="s-1")


@pytest.mark.asyncio
async def test_distill_rejects_dimension_mismatch_from_embedder() -> None:
    """A misconfigured embedder (wrong dim per row) is caught immediately."""

    class WideEmbedding(StubEmbedding):
        async def embed(self, texts: list[str] | tuple[str, ...]) -> list[list[float]]:  # type: ignore[override]
            return [[0.0] * 99 for _ in texts]

    llm = StubLLM(responses=[_payload()])
    embedder = WideEmbedding(dimension=4)  # claims 4, returns 99
    distiller = Distiller(llm, embedder, version_id="v")
    with pytest.raises(LLMError, match=r"vector of length 99"):
        await distiller.distill([_turn(0)], session_id="s-1")


@pytest.mark.asyncio
async def test_distill_falls_back_to_placeholder_for_empty_digest() -> None:
    """When the model produces no summary text, we still embed something."""

    payload = json.loads(_payload(learnings=[]))
    payload["digest"]["summary_md"] = ""
    payload["digest"]["bm25_text"] = ""
    distiller, _, embedder = _make(json.dumps(payload))
    await distiller.distill([_turn(0)], session_id="s-1")
    assert embedder.calls[0] == ("(empty)",)


# --------------------------------------------------------------------------
# Stage B+ claim_type and experiment scope
# --------------------------------------------------------------------------


def test_distill_system_prompt_documents_claim_type_vocabulary() -> None:
    msgs = build_distill_prompt([_turn(0)], session_id="s-1", prior_purpose=None)
    system = msgs[0].content
    assert "claim_type" in system
    for fragment in ("observation", "interpretation", "signal", "norm"):
        assert fragment in system, f"system prompt missing claim_type value {fragment!r}"
    assert "experiment" in system, "system prompt must list 'experiment' scope"


@pytest.mark.asyncio
async def test_distill_extracts_claim_type_when_present() -> None:
    bad = _default_learnings()
    bad[0]["claim_type"] = "norm"
    distiller, _, _ = _make(_payload(learnings=bad))
    result = await distiller.distill([_turn(0)], session_id="s-1")
    assert result.candidate_learnings[0].learning.claim_type == "norm"


@pytest.mark.asyncio
async def test_distill_defaults_claim_type_to_signal_when_omitted() -> None:
    """Existing prompts that pre-date claim_type still parse cleanly.

    When the LLM does not emit ``claim_type`` we silently default to
    ``signal``. The retriever falls back to signal semantics for legacy
    rows so this matches the behaviour the data already implies.
    """

    distiller, _, _ = _make(_payload())  # _default_learnings has no claim_type
    result = await distiller.distill([_turn(0)], session_id="s-1")
    assert result.candidate_learnings[0].learning.claim_type == "signal"


@pytest.mark.asyncio
async def test_distill_rejects_unknown_claim_type() -> None:
    bad = _default_learnings()
    bad[0]["claim_type"] = "rumor"
    distiller, _, _ = _make(_payload(learnings=bad))
    with pytest.raises(LLMError, match=r"'claim_type' must be one of"):
        await distiller.distill([_turn(0)], session_id="s-1")


@pytest.mark.asyncio
async def test_distill_rejects_non_string_claim_type() -> None:
    bad = _default_learnings()
    bad[0]["claim_type"] = 42
    distiller, _, _ = _make(_payload(learnings=bad))
    with pytest.raises(LLMError, match=r"'claim_type' must be a string or null"):
        await distiller.distill([_turn(0)], session_id="s-1")


@pytest.mark.asyncio
async def test_distill_accepts_experiment_scope() -> None:
    """Stage B+ widens the scope vocabulary to include ``experiment``."""

    bad = _default_learnings()
    bad[0]["scope"] = "experiment"
    distiller, _, _ = _make(_payload(learnings=bad))
    result = await distiller.distill([_turn(0)], session_id="s-1")
    assert result.candidate_learnings[0].learning.scope == "experiment"
