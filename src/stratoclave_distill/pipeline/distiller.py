"""One-shot session distillation.

The Distiller is the first LLM-bound stage of the Stage B pipeline:

- it takes a sequence of :class:`NormalizedTurn` records (filtered to a
  single session and to ``seq`` strictly above the watermark by the CLI
  layer);
- builds a single prompt asking the model to emit a strict JSON object
  with three sections (purpose / digest / learnings);
- parses that JSON, mints UUIDs / timestamps, and returns dataclasses
  ready for the Curator and the persistence layer to consume.

The Distiller deliberately does **no** persistence and **no**
conflict resolution. Its only side effects are LLM and embedding
provider calls. This keeps the unit tests pure: a stub LLM emits a
canned JSON blob, a stub embedding handles vector shape, and the
asserts are entirely on the returned dataclasses.

The LLM contract is documented inline in :func:`build_distill_prompt`
and is intentionally narrow so a smaller model can satisfy it
reliably. The schema mirrors the persistence layer one-to-one.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from stratoclave_distill.core.errors import LLMError
from stratoclave_distill.core.types import (
    Learning,
    NormalizedTurn,
    SessionDigest,
    SessionPurpose,
)
from stratoclave_distill.providers.embedding import EmbeddingProvider
from stratoclave_distill.providers.llm import LLMMessage, LLMProvider

_VALID_SCOPES: frozenset[str] = frozenset({"session", "project", "group", "shared"})


@dataclass(frozen=True, slots=True)
class CandidateLearning:
    """A learning the Distiller emitted, *before* Curator decides INSERT/MERGE/SUPERSEDE.

    Carries the embedding alongside the public :class:`Learning` shape so
    the Curator can use it both for similarity search (against existing
    rows) and for persistence on INSERT. The ``learning_id`` on the inner
    :class:`Learning` is a freshly minted UUID; the Curator may choose to
    discard it (MERGE / SUPERSEDE) without ever persisting it.
    """

    learning: Learning
    embedding: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class DistillationResult:
    """Everything one Distiller call produced for a single session.

    The CLI layer is responsible for:

    - calling :meth:`PurposeStore.upsert(purpose)`,
    - calling :meth:`DigestStore.upsert(digest, embedding=digest_embedding)`,
    - feeding ``candidate_learnings`` into the Curator, and
    - advancing the watermark to ``last_seq`` on success.
    """

    purpose: SessionPurpose
    digest: SessionDigest
    digest_embedding: tuple[float, ...]
    candidate_learnings: tuple[CandidateLearning, ...]
    last_seq: int


_SYSTEM_PROMPT = (
    "You are stratoclave-distill, a careful summarizer of one engineering chat session. "
    "Output ONLY a single JSON object (no prose, no code fences) with the exact shape:\n"
    '{"purpose": {"purpose": str, "domain_tags": [str, ...], '
    '"success_score": float|null, "polluted": bool, "pollution_reason": str|null}, '
    '"digest": {"summary_md": str, "bm25_text": str}, '
    '"learnings": ['
    '{"scope": "session"|"project"|"group"|"shared", '
    '"rule": str, "why": str, "triggers": object, '
    '"evidence_count": int, "confidence": float, "bm25_text": str}, ...]}\n'
    "Be terse. Prefer at most a handful of high-signal learnings; emit an empty array "
    "if nothing reusable was demonstrated. Never invent facts; if uncertain, set "
    '"polluted": true and explain in "pollution_reason".'
)


def _format_turn(turn: NormalizedTurn) -> str:
    """Render one turn as a single human-readable line for the prompt.

    Keeps tool calls visible (they often carry the most reusable signal)
    while truncating the input dict to its keys to stay token-cheap. The
    Distiller's job is summarization, not faithful tool replay.
    """

    head = f"[{turn.seq:04d} {turn.role}]"
    if turn.tool_name:
        keys = ", ".join(sorted((turn.tool_input or {}).keys())) or "-"
        head = f"{head} tool={turn.tool_name}({keys})"
    text = turn.text_content.strip()
    if text:
        return f"{head} {text}"
    return head


def build_distill_prompt(
    turns: Sequence[NormalizedTurn],
    *,
    session_id: str,
    prior_purpose: SessionPurpose | None,
) -> tuple[LLMMessage, ...]:
    """Build the system + user messages handed to :meth:`LLMProvider.complete`.

    Exposed at module scope so unit tests can assert on the exact prompt
    shape (and so the prompt is reviewable in code review without running
    the pipeline).
    """

    transcript = "\n".join(_format_turn(t) for t in turns) or "(empty session)"
    purpose_hint = (
        f"Prior purpose for this session: {prior_purpose.purpose!r}\n"
        if prior_purpose is not None
        else ""
    )
    user = f"Session id: {session_id}\n{purpose_hint}Transcript ({len(turns)} turns):\n{transcript}"
    return (
        LLMMessage(role="system", content=_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user),
    )


def _require_str(obj: Mapping[str, object], key: str, *, allow_empty: bool = False) -> str:
    value = obj.get(key)
    if not isinstance(value, str):
        raise LLMError(f"distill JSON: {key!r} must be a string")
    if not allow_empty and not value:
        raise LLMError(f"distill JSON: {key!r} must be non-empty")
    return value


def _require_bool(obj: Mapping[str, object], key: str) -> bool:
    value = obj.get(key)
    if not isinstance(value, bool):
        raise LLMError(f"distill JSON: {key!r} must be a boolean")
    return value


def _coerce_optional_str(obj: Mapping[str, object], key: str) -> str | None:
    value = obj.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise LLMError(f"distill JSON: {key!r} must be a string or null")
    return value or None


def _coerce_optional_float(obj: Mapping[str, object], key: str) -> float | None:
    value = obj.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LLMError(f"distill JSON: {key!r} must be a number or null")
    return float(value)


def _coerce_str_tuple(obj: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = obj.get(key, [])
    if not isinstance(value, list):
        raise LLMError(f"distill JSON: {key!r} must be an array of strings")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise LLMError(f"distill JSON: {key!r} elements must be strings")
        out.append(item)
    return tuple(out)


def _parse_purpose(
    obj: Mapping[str, object],
    *,
    session_id: str,
    derived_from_version: str,
    now: str,
) -> SessionPurpose:
    return SessionPurpose(
        session_id=session_id,
        purpose=_require_str(obj, "purpose"),
        domain_tags=_coerce_str_tuple(obj, "domain_tags"),
        success_score=_coerce_optional_float(obj, "success_score"),
        polluted=_require_bool(obj, "polluted"),
        pollution_reason=_coerce_optional_str(obj, "pollution_reason"),
        derived_from_version=derived_from_version,
        derived_at=now,
        last_updated_at=now,
    )


def _parse_digest(
    obj: Mapping[str, object],
    *,
    session_id: str,
    version_id: str,
    now: str,
) -> SessionDigest:
    return SessionDigest(
        digest_id=str(uuid.uuid4()),
        session_id=session_id,
        version_id=version_id,
        summary_md=_require_str(obj, "summary_md", allow_empty=True),
        bm25_text=_require_str(obj, "bm25_text", allow_empty=True),
        extracted_at=now,
    )


def _parse_learning(
    obj: Mapping[str, object],
    *,
    session_id: str,
    version_id: str,
    now: str,
) -> Learning:
    scope_raw = _require_str(obj, "scope")
    if scope_raw not in _VALID_SCOPES:
        raise LLMError(
            f"distill JSON: learning scope must be one of {sorted(_VALID_SCOPES)!r}, "
            f"got {scope_raw!r}"
        )
    triggers = obj.get("triggers", {})
    if not isinstance(triggers, dict):
        raise LLMError("distill JSON: learning 'triggers' must be an object")
    evidence_raw = obj.get("evidence_count", 1)
    if isinstance(evidence_raw, bool) or not isinstance(evidence_raw, int):
        raise LLMError("distill JSON: learning 'evidence_count' must be an integer")
    confidence_raw = obj.get("confidence", 0.5)
    if isinstance(confidence_raw, bool) or not isinstance(confidence_raw, (int, float)):
        raise LLMError("distill JSON: learning 'confidence' must be a number")
    return Learning(
        learning_id=str(uuid.uuid4()),
        scope=scope_raw,  # type: ignore[arg-type]
        rule=_require_str(obj, "rule"),
        why=_require_str(obj, "why"),
        triggers=dict(triggers),
        source_session=session_id,
        source_version=version_id,
        evidence_count=evidence_raw,
        confidence=float(confidence_raw),
        bm25_text=_require_str(obj, "bm25_text", allow_empty=True),
        created_at=now,
        updated_at=now,
    )


def _parse_response(
    text: str,
    *,
    session_id: str,
    version_id: str,
    now: str,
) -> tuple[SessionPurpose, SessionDigest, tuple[Learning, ...]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"distill JSON could not be parsed: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise LLMError("distill JSON top-level value must be an object")

    purpose_obj = payload.get("purpose")
    if not isinstance(purpose_obj, dict):
        raise LLMError("distill JSON: 'purpose' must be an object")
    digest_obj = payload.get("digest")
    if not isinstance(digest_obj, dict):
        raise LLMError("distill JSON: 'digest' must be an object")
    learnings_raw = payload.get("learnings", [])
    if not isinstance(learnings_raw, list):
        raise LLMError("distill JSON: 'learnings' must be an array")

    purpose = _parse_purpose(
        purpose_obj, session_id=session_id, derived_from_version=version_id, now=now
    )
    digest = _parse_digest(digest_obj, session_id=session_id, version_id=version_id, now=now)
    learnings: list[Learning] = []
    for idx, item in enumerate(learnings_raw):
        if not isinstance(item, dict):
            raise LLMError(f"distill JSON: learnings[{idx}] must be an object")
        learnings.append(
            _parse_learning(item, session_id=session_id, version_id=version_id, now=now)
        )
    return purpose, digest, tuple(learnings)


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class Distiller:
    """One-shot LLM extractor for a single session.

    Parameters
    ----------
    llm:
        Provider used for the single completion call. Any object that
        satisfies :class:`stratoclave_distill.providers.llm.LLMProvider`
        works; tests inject :class:`StubLLM` to exercise the parser
        without standing up a real backend.
    embedder:
        Used to compute the digest embedding plus one embedding per
        emitted learning. The pipeline asserts that every returned
        vector has length :attr:`EmbeddingProvider.dimension`.
    version_id:
        Identifier for the prompt / library revision that produced this
        run. Stored on every emitted row so a future re-distillation
        can supersede earlier versions deterministically.
    max_tokens / temperature:
        Forwarded to the LLM call. Defaults match the v0.1 reference
        profile (1024 tokens, temperature 0).
    clock:
        Returns the ISO-8601 UTC timestamp stamped on every emitted
        dataclass. Injectable so unit tests can assert on exact values
        without monkeypatching ``datetime``.
    """

    __slots__ = ("_clock", "_embedder", "_llm", "_max_tokens", "_temperature", "_version_id")

    def __init__(
        self,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        *,
        version_id: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        clock: Callable[[], str] = _utc_now,
    ) -> None:
        if not version_id:
            raise LLMError("Distiller requires a non-empty version_id")
        self._llm = llm
        self._embedder = embedder
        self._version_id = version_id
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._clock = clock

    @property
    def version_id(self) -> str:
        return self._version_id

    async def distill(
        self,
        turns: Sequence[NormalizedTurn],
        *,
        session_id: str,
        prior_purpose: SessionPurpose | None = None,
    ) -> DistillationResult:
        """Run the LLM call and parse the response into dataclasses.

        ``turns`` must already be filtered to a single session and to
        the seq window the caller wants to distill. The Distiller does
        not consult any persistence layer and does not advance any
        watermark; ``last_seq`` is simply ``max(t.seq for t in turns)``
        (or 0 if ``turns`` is empty), surfaced so the caller can
        advance the watermark on success.
        """

        if not session_id:
            raise LLMError("distill requires a non-empty session_id")
        for turn in turns:
            if turn.session_id != session_id:
                raise LLMError(
                    f"distill: turn {turn.turn_id!r} has session_id {turn.session_id!r}, "
                    f"expected {session_id!r}"
                )

        messages = build_distill_prompt(turns, session_id=session_id, prior_purpose=prior_purpose)
        raw = await self._llm.complete(
            messages, max_tokens=self._max_tokens, temperature=self._temperature
        )
        now = self._clock()
        purpose, digest, learnings = _parse_response(
            raw, session_id=session_id, version_id=self._version_id, now=now
        )

        embed_inputs: list[str] = [digest.summary_md or digest.bm25_text or "(empty)"]
        for learning in learnings:
            embed_inputs.append(learning.bm25_text or learning.rule)

        vectors = await self._embedder.embed(embed_inputs)
        if len(vectors) != len(embed_inputs):
            raise LLMError(
                f"embedder returned {len(vectors)} vectors, expected {len(embed_inputs)}"
            )
        expected_dim = self._embedder.dimension
        for vec in vectors:
            if len(vec) != expected_dim:
                raise LLMError(
                    f"embedder returned vector of length {len(vec)}, "
                    f"expected dimension {expected_dim}"
                )

        digest_embedding = tuple(vectors[0])
        candidates = tuple(
            CandidateLearning(learning=learnings[i], embedding=tuple(vectors[i + 1]))
            for i in range(len(learnings))
        )
        last_seq = max((t.seq for t in turns), default=0)

        return DistillationResult(
            purpose=purpose,
            digest=digest,
            digest_embedding=digest_embedding,
            candidate_learnings=candidates,
            last_seq=last_seq,
        )


__all__ = [
    "CandidateLearning",
    "DistillationResult",
    "Distiller",
    "build_distill_prompt",
]
