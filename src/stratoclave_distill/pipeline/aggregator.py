"""Stage 3 group rollup: ``Aggregator``.

The Aggregator produces a single :class:`GroupLearning` row by handing
the LLM every active learning that shares a ``group_id`` and asking for
one Markdown summary plus one BM25-friendly text. The shape of the
output mirrors the Distiller: one LLM completion, one embedding call,
deterministic dataclass output, *no* persistence inside the class
(persistence is the caller's job, the same way Distiller hands its
result to the CLI).

Design notes
------------

- ``Aggregator.run`` accepts the contributing :class:`Learning` rows
  directly. Selecting which rows to feed (e.g. ``LearningStore.list_active``
  filtered to one group_id) is the orchestration layer's job. This keeps
  the unit tests pure: a stub LLM emits a canned response and the
  asserts run on the returned dataclass without involving any store.
- The LLM contract is intentionally tiny: a single JSON object
  ``{"summary_md": str, "bm25_text": str}``. We do not ask the LLM to
  re-derive ``contributing_learnings``: that list is authoritative on
  the caller side and is stamped into the result post-parse.
- Re-aggregation does not delete the previous row: each run produces a
  fresh ``group_learning_id`` so the audit trail survives. The Retriever
  consumes only the latest rollup per ``group_id``.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from stratoclave_distill.core.errors import LLMError
from stratoclave_distill.core.types import GroupLearning, Learning
from stratoclave_distill.providers.embedding import EmbeddingProvider
from stratoclave_distill.providers.llm import LLMMessage, LLMProvider


@dataclass(frozen=True, slots=True)
class AggregationResult:
    """Output of one :meth:`Aggregator.run` call.

    The CLI / orchestration layer is expected to call
    :meth:`GroupLearningStore.upsert(group_learning, embedding=embedding)`
    with both fields. The embedding is surfaced separately so the public
    :class:`GroupLearning` dataclass stays JSON-friendly.
    """

    group_learning: GroupLearning
    embedding: tuple[float, ...]


_SYSTEM_PROMPT = (
    "You are stratoclave-distill, summarizing a *group* of related learnings into "
    "ONE compact rollup. Output ONLY a single JSON object (no prose, no code "
    'fences) with the exact shape: {"summary_md": str, "bm25_text": str}.\n'
    "summary_md MUST be Markdown with one heading-less paragraph followed by a "
    "bullet list of the most reusable rules. bm25_text MUST be a flat text "
    "version of the same content with no Markdown formatting and no bullets, "
    "suitable for full-text search. Both fields MUST be non-empty. Do not "
    "invent rules that are not present in the inputs; if the inputs disagree, "
    "note the disagreement instead of choosing a side."
)


def _format_learning(learning: Learning) -> str:
    """Render one learning as a single line for the prompt.

    Keeps the prompt token-cheap while preserving the metadata the LLM
    actually uses to weight the rollup: scope (canonical / experimental
    matters), claim_type, evidence_count, and the rule + why pair.
    """

    claim = learning.claim_type or "signal"
    evidence = learning.evidence_count
    return (
        f"- ({learning.scope}|{claim}|n={evidence}) {learning.rule.strip()} "
        f"-- why: {learning.why.strip() or '(no why)'}"
    )


def build_aggregate_prompt(
    learnings: Sequence[Learning],
    *,
    group_id: str,
) -> tuple[LLMMessage, ...]:
    """Build the system + user messages for :meth:`Aggregator.run`.

    Exposed at module scope so tests can assert on the exact prompt
    shape and reviewers do not have to run the pipeline to read it.
    """

    body = "\n".join(_format_learning(rule) for rule in learnings) or "(no learnings)"
    user = (
        f"group_id: {group_id}\n"
        f"contributing learnings ({len(learnings)} rows):\n{body}\n"
        "Produce the JSON rollup."
    )
    return (
        LLMMessage(role="system", content=_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user),
    )


def _require_str(obj: Mapping[str, object], key: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str):
        raise LLMError(f"aggregate JSON: {key!r} must be a string")
    if not value.strip():
        raise LLMError(f"aggregate JSON: {key!r} must be non-empty")
    return value


def _parse_response(raw: str) -> tuple[str, str]:
    """Parse the LLM JSON object into ``(summary_md, bm25_text)``."""

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError(f"aggregate response was not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise LLMError("aggregate response must be a JSON object")
    return _require_str(payload, "summary_md"), _require_str(payload, "bm25_text")


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class Aggregator:
    """One-shot group rollup producer.

    Parameters
    ----------
    llm:
        Provider used for the single completion call. Tests inject
        :class:`StubLLM` to exercise the parser without a real backend.
    embedder:
        Used to compute the rollup embedding; the returned vector must
        have length :attr:`EmbeddingProvider.dimension`.
    max_tokens / temperature:
        Forwarded to the LLM call. Defaults match the Distiller.
    clock:
        Returns the ISO-8601 UTC timestamp stamped on the emitted
        :class:`GroupLearning`. Injectable for deterministic tests.
    """

    __slots__ = ("_clock", "_embedder", "_llm", "_max_tokens", "_temperature")

    def __init__(
        self,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        clock: Callable[[], str] = _utc_now,
    ) -> None:
        self._llm = llm
        self._embedder = embedder
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._clock = clock

    async def run(
        self,
        learnings: Sequence[Learning],
        *,
        group_id: str,
    ) -> AggregationResult:
        """Run one rollup for ``group_id``.

        ``learnings`` must be the contributing rows the caller wants
        summarized; the Aggregator does not consult any store. Raises
        :class:`LLMError` when the inputs are degenerate (no learnings,
        empty group_id) so the CLI surfaces a clean exit.
        """

        if not group_id:
            raise LLMError("aggregate requires a non-empty group_id")
        if not learnings:
            raise LLMError("aggregate requires at least one learning")
        for learning in learnings:
            if learning.group_id != group_id:
                raise LLMError(
                    f"aggregate: learning {learning.learning_id!r} has group_id "
                    f"{learning.group_id!r}, expected {group_id!r}"
                )

        messages = build_aggregate_prompt(learnings, group_id=group_id)
        raw = await self._llm.complete(
            messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        summary_md, bm25_text = _parse_response(raw)

        vectors = await self._embedder.embed([bm25_text])
        if len(vectors) != 1:
            raise LLMError(f"embedder returned {len(vectors)} vectors, expected 1 for the rollup")
        vector = tuple(vectors[0])
        if len(vector) != self._embedder.dimension:
            raise LLMError(
                f"embedder returned dim {len(vector)}, expected {self._embedder.dimension}"
            )

        contributing = tuple(rule.learning_id for rule in learnings)
        rollup = GroupLearning(
            group_learning_id=str(uuid.uuid4()),
            group_id=group_id,
            summary_md=summary_md,
            contributing_learnings=contributing,
            bm25_text=bm25_text,
            created_at=self._clock(),
        )
        return AggregationResult(group_learning=rollup, embedding=vector)


__all__ = [
    "AggregationResult",
    "Aggregator",
    "build_aggregate_prompt",
]
