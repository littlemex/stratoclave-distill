"""Unit tests for :class:`IngestRunner`.

The contract Stage B's CLI relies on:

- a session whose ``seq`` are all <= the watermark is reported
  ``distilled=False`` and the watermark does *not* move;
- a session whose ``seq`` partially exceed the watermark is distilled
  on the fresh subset only, and the watermark advances to ``last_seq``;
- the Distiller receives the prior :class:`SessionPurpose` if one is
  on file, so the prompt can carry the running narrative;
- on success, the runner upserts purpose, upserts the digest with its
  embedding, and hands every candidate to the Curator;
- a :class:`DistillError` aborts only that one session: the watermark
  stays put, the report carries the message, and other sessions in the
  same batch still ingest;
- ``strict=True`` re-raises the first error instead of recording it;
- multiple sessions in one file are grouped and processed independently;
- :meth:`run_path` reads the JSONL file and forwards
  ``reader.skipped`` to :class:`IngestReport`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import pytest

from stratoclave_distill.core.errors import DistillError
from stratoclave_distill.core.types import NormalizedTurn
from stratoclave_distill.db.memory import (
    InMemoryDigestStore,
    InMemoryLearningStore,
    InMemoryPurposeStore,
    InMemoryWatermarkStore,
)
from stratoclave_distill.pipeline import (
    Curator,
    Distiller,
    IngestReport,
    IngestRunner,
    SessionIngestResult,
)
from stratoclave_distill.providers.embedding import StubEmbedding
from stratoclave_distill.providers.llm import LLMMessage, StubLLM

# --------------------------------------------------------------------------
# Fixture helpers
# --------------------------------------------------------------------------


def _turn(
    seq: int,
    *,
    session_id: str = "s-1",
    role: str = "user",
    text: str = "hello",
) -> NormalizedTurn:
    return NormalizedTurn(
        turn_id=f"{session_id}-t-{seq}",
        session_id=session_id,
        seq=seq,
        role=role,
        text_content=text,
        tool_name=None,
        tool_input=None,
        occurred_at="2026-05-22T00:00:00Z",
        raw_line="{}",
    )


def _ok_payload(
    *,
    purpose: str = "investigate flaky test",
    learnings: list[dict[str, object]] | None = None,
    polluted: bool = False,
) -> str:
    return json.dumps(
        {
            "purpose": {
                "purpose": purpose,
                "domain_tags": ["testing"],
                "success_score": 0.9,
                "polluted": polluted,
                "pollution_reason": None,
            },
            "digest": {
                "summary_md": "# Summary\nThe user fixed a flaky test.",
                "bm25_text": "flaky test fix",
            },
            "learnings": (
                learnings
                if learnings is not None
                else [
                    {
                        "scope": "session",
                        "rule": "rerun pytest with -p no:cacheprovider",
                        "why": "the cache hid the bug",
                        "triggers": {"pytest": True},
                        "evidence_count": 1,
                        "confidence": 0.7,
                        "bm25_text": "pytest cacheprovider flake",
                    }
                ]
            ),
        }
    )


def _make_runner(
    *,
    llm_responses: Sequence[str] | None = None,
    llm_responder=None,
    strict: bool = False,
    embedding_dim: int = 4,
    clock_value: str = "2026-05-22T01:00:00Z",
) -> tuple[
    IngestRunner,
    InMemoryWatermarkStore,
    InMemoryPurposeStore,
    InMemoryDigestStore,
    InMemoryLearningStore,
    StubLLM,
]:
    if llm_responder is not None:
        llm = StubLLM(responder=llm_responder)
    else:
        llm = StubLLM(responses=llm_responses or [_ok_payload()])
    embedder = StubEmbedding(dimension=embedding_dim)
    distiller = Distiller(
        llm,
        embedder,
        version_id="v-test-2026-05-22",
        clock=lambda: clock_value,
    )
    learnings = InMemoryLearningStore()
    curator = Curator(learnings, clock=lambda: clock_value)
    watermarks = InMemoryWatermarkStore()
    purposes = InMemoryPurposeStore()
    digests = InMemoryDigestStore()
    runner = IngestRunner(
        distiller=distiller,
        curator=curator,
        watermarks=watermarks,
        purposes=purposes,
        digests=digests,
        strict=strict,
    )
    return runner, watermarks, purposes, digests, learnings, llm


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_turns_empty_yields_empty_report() -> None:
    runner, *_ = _make_runner()
    report = await runner.run_turns([])
    assert report.session_count == 0
    assert report.distilled_count == 0
    assert report.error_count == 0
    assert report.sessions == ()
    assert report.skipped_lines == ()


# --------------------------------------------------------------------------
# Single-session happy path
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_turns_distills_fresh_session() -> None:
    runner, watermarks, purposes, digests, learnings, llm = _make_runner()
    turns = [_turn(1), _turn(2), _turn(3)]

    report = await runner.run_turns(turns)

    [session] = report.sessions
    assert session.session_id == "s-1"
    assert session.distilled is True
    assert session.error is None
    assert session.prior_seq == 0
    assert session.new_seq == 3
    assert session.candidate_count == 1
    assert session.curation is not None
    assert tuple(d.action for d in session.curation.decisions) == ("INSERT",)

    # Stores were populated
    assert (await purposes.get("s-1")) is not None
    assert (await digests.get("s-1")) is not None
    assert await digests.get_embedding("s-1") is not None
    assert (await watermarks.get("s-1")) == 3
    # The single learning made it into the LearningStore via the Curator
    assert len(await learnings.list_active()) == 1
    # The LLM was called exactly once with system + user
    assert len(llm.calls) == 1
    roles = [m.role for m in llm.calls[0]]
    assert roles == ["system", "user"]


# --------------------------------------------------------------------------
# Watermark filtering
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_turns_skips_already_distilled_turns() -> None:
    runner, watermarks, purposes, digests, _learnings, llm = _make_runner()
    await watermarks.advance("s-1", to_seq=5, last_run_at="t0")

    report = await runner.run_turns([_turn(1), _turn(3), _turn(5)])

    [session] = report.sessions
    assert session.distilled is False
    assert session.prior_seq == 5
    assert session.new_seq == 5
    assert session.candidate_count == 0
    assert session.curation is None
    assert (await purposes.get("s-1")) is None  # nothing was distilled
    assert (await digests.get("s-1")) is None
    assert (await watermarks.get("s-1")) == 5  # unchanged
    assert llm.calls == ()


@pytest.mark.asyncio
async def test_run_turns_distills_only_seq_above_watermark() -> None:
    runner, watermarks, _, _, _, llm = _make_runner()
    await watermarks.advance("s-1", to_seq=2, last_run_at="t0")

    await runner.run_turns([_turn(1), _turn(2), _turn(3), _turn(4)])

    # Only seq 3 and 4 should have appeared in the prompt
    [(_system, user)] = llm.calls
    assert "0003" in user.content
    assert "0004" in user.content
    assert "0001" not in user.content
    assert "0002" not in user.content
    assert (await watermarks.get("s-1")) == 4


# --------------------------------------------------------------------------
# Multi-session batches
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_turns_groups_sessions_independently() -> None:
    runner, watermarks, *_ = _make_runner(
        llm_responses=[_ok_payload(purpose="A"), _ok_payload(purpose="B")]
    )
    turns = [
        _turn(1, session_id="s-A"),
        _turn(1, session_id="s-B"),
        _turn(2, session_id="s-A"),
    ]

    report = await runner.run_turns(turns)
    by_id = {s.session_id: s for s in report.sessions}
    assert set(by_id) == {"s-A", "s-B"}
    assert by_id["s-A"].new_seq == 2
    assert by_id["s-B"].new_seq == 1
    assert (await watermarks.get("s-A")) == 2
    assert (await watermarks.get("s-B")) == 1
    assert report.distilled_count == 2


# --------------------------------------------------------------------------
# Prior purpose forwarded to Distiller
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_turns_forwards_prior_purpose_into_prompt() -> None:
    captured: list[Sequence[LLMMessage]] = []

    def responder(messages: Sequence[LLMMessage]) -> str:
        captured.append(tuple(messages))
        return _ok_payload(purpose="updated purpose")

    runner, watermarks, purposes, *_ = _make_runner(llm_responder=responder)

    # First ingest establishes the purpose.
    await runner.run_turns([_turn(1)])
    first = await purposes.get("s-1")
    assert first is not None

    # Second ingest with a fresh seq must include the prior purpose in the user prompt.
    await runner.run_turns([_turn(2)])
    assert len(captured) == 2
    second_user = captured[1][1].content
    assert "Prior purpose for this session" in second_user
    assert first.purpose in second_user
    assert (await watermarks.get("s-1")) == 2


# --------------------------------------------------------------------------
# DistillError handling
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_turns_records_distill_error_and_does_not_advance_watermark() -> None:
    runner, watermarks, purposes, digests, learnings, _ = _make_runner(
        llm_responses=["this is not json at all"]
    )

    report = await runner.run_turns([_turn(1), _turn(2)])

    [session] = report.sessions
    assert session.distilled is False
    assert session.error is not None
    assert "JSON" in session.error or "json" in session.error
    assert session.prior_seq == 0
    assert session.new_seq == 0
    assert session.curation is None
    # No partial state was committed.
    assert (await purposes.get("s-1")) is None
    assert (await digests.get("s-1")) is None
    assert (await watermarks.get("s-1")) == 0
    assert len(await learnings.list_active()) == 0
    assert report.error_count == 1
    assert report.distilled_count == 0


@pytest.mark.asyncio
async def test_run_turns_continues_after_error_in_one_session() -> None:
    runner, watermarks, *_ = _make_runner(llm_responses=["broken json", _ok_payload(purpose="ok")])

    report = await runner.run_turns([_turn(1, session_id="s-bad"), _turn(1, session_id="s-good")])
    by_id = {s.session_id: s for s in report.sessions}
    assert by_id["s-bad"].error is not None
    assert by_id["s-bad"].distilled is False
    assert by_id["s-good"].distilled is True
    assert (await watermarks.get("s-bad")) == 0
    assert (await watermarks.get("s-good")) == 1


@pytest.mark.asyncio
async def test_strict_mode_raises_on_first_error() -> None:
    runner, *_ = _make_runner(llm_responses=["broken json", _ok_payload()], strict=True)

    with pytest.raises(DistillError):
        await runner.run_turns([_turn(1, session_id="s-bad")])


# --------------------------------------------------------------------------
# run_path
# --------------------------------------------------------------------------


def _write_jsonl(path: Path, lines: Iterable[object]) -> None:
    text = "\n".join(json.dumps(line) if not isinstance(line, str) else line for line in lines)
    path.write_text(text + "\n", encoding="utf-8")


@pytest.mark.asyncio
async def test_run_path_reads_jsonl_and_reports_skipped(tmp_path: Path) -> None:
    runner, _, _, _, _, llm = _make_runner()
    path = tmp_path / "session.jsonl"
    _write_jsonl(
        path,
        [
            {
                "turn_id": "t-1",
                "session_id": "s-1",
                "seq": 1,
                "role": "user",
                "text_content": "hello",
                "occurred_at": "2026-05-22T00:00:00Z",
            },
            "not even json",  # raw string -> JSON decode error
            {
                "turn_id": "t-2",
                "session_id": "s-1",
                "seq": 2,
                "role": "assistant",
                "text_content": "hi",
                "occurred_at": "2026-05-22T00:00:01Z",
            },
        ],
    )

    report = await runner.run_path(str(path))

    assert isinstance(report, IngestReport)
    assert report.session_count == 1
    assert report.distilled_count == 1
    [bad] = report.skipped_lines
    assert bad.line_no == 2
    assert "JSON" in bad.reason or "json" in bad.reason
    assert len(llm.calls) == 1


# --------------------------------------------------------------------------
# Edge: session id mismatch on the dataclass should still produce a report
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_ingest_result_is_a_frozen_dataclass() -> None:
    result = SessionIngestResult(
        session_id="s-1",
        distilled=False,
        prior_seq=0,
        new_seq=0,
        candidate_count=0,
        curation=None,
    )
    with pytest.raises(AttributeError):
        result.session_id = "s-2"  # type: ignore[misc]
