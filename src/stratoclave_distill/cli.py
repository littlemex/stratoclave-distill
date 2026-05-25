"""Command-line entrypoint for stratoclave-distill.

Stage A shipped ``version`` and ``check-config``. Stage B added ``ingest``
for processing a JSONL transcript end-to-end. Stage B+ extended ``ingest``
with ``--branch-from`` / ``--at-seq`` / ``--branch-kind`` and introduced
the ``branch`` subcommand group (``branch close``, ``branch list``).
Stage C lands the remaining real subcommands:

- ``query``: hybrid search over learnings, with optional Markdown packing
  via the :class:`ContextPacker`. Supports ``--dry-run`` for fixture
  validation just like ``ingest``.
- ``export``: dump a single session (purpose + digest + learnings,
  optionally conflicts and gaps) as JSON for snapshotting.
- ``gc``: archive-row cleanup with a ``--dry-run`` default so destructive
  operations always require an explicit ``--apply``.

``ingest`` and ``query`` run in two modes:

- ``--dry-run`` (the default for tests and fixture validation) wires
  in-memory stores + stub providers so no database or LLM credentials
  are required;
- the production mode (no ``--dry-run``) reads :class:`DistillerConfig`
  from the environment and stands up the asyncpg-backed stores plus the
  real LLM / Embedding providers.

The ``branch``, ``export``, and ``gc`` subcommand groups always use
:class:`DistillerConfig` to open a real Postgres connection — there is
no dry-run for branch / export / gc state because there is nothing to
inspect in an in-memory store between CLI invocations.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict

from stratoclave_distill import __version__
from stratoclave_distill.config import DistillerConfig
from stratoclave_distill.core.errors import DistillError
from stratoclave_distill.core.types import (
    GroupLearning,
    Learning,
    LearningConflict,
    LearningScope,
    SessionDigest,
    SessionGap,
    SessionPurpose,
)
from stratoclave_distill.db.memory import (
    InMemoryConflictStore,
    InMemoryDigestStore,
    InMemoryGapStore,
    InMemoryLearningStore,
    InMemoryPurposeStore,
    InMemoryWatermarkStore,
)
from stratoclave_distill.db.stores import GroupLearningSearchHit, LearningSearchHit
from stratoclave_distill.pipeline import (
    AggregationResult,
    Aggregator,
    BranchPlan,
    Curator,
    Distiller,
    IngestReport,
    IngestRunner,
)
from stratoclave_distill.providers.embedding import StubEmbedding, build_embedding_provider
from stratoclave_distill.providers.llm import StubLLM, build_llm_provider
from stratoclave_distill.retrieval import (
    ContextPacker,
    RetrievalResult,
    Retriever,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stratoclave-distill",
        description="Session distillation, learning aggregation, and hybrid search.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="Print the package version and exit.")

    cfg_parser = sub.add_parser(
        "check-config",
        help="Validate environment-driven configuration without touching the database.",
    )
    cfg_parser.add_argument(
        "--show-defaults",
        action="store_true",
        help="Include defaulted fields in the JSON output.",
    )

    ingest_parser = sub.add_parser(
        "ingest",
        help="Distill a JSONL transcript end-to-end and emit a report.",
    )
    ingest_parser.add_argument(
        "path",
        help="Path to the JSONL transcript (one normalized turn per line).",
    )
    ingest_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run with in-memory stores and stub providers; never touches the database. "
            "Useful for fixture and prompt validation."
        ),
    )
    ingest_parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort on the first malformed line or distill error instead of recording it.",
    )
    ingest_parser.add_argument(
        "--version-id",
        default="dry-run",
        help="Distiller version_id stamped on emitted rows (only used with --dry-run).",
    )
    ingest_parser.add_argument(
        "--prod-version-id",
        default=None,
        help=(
            "Distiller version_id stamped on emitted rows in production mode. "
            "Defaults to a date-based identifier."
        ),
    )
    ingest_parser.add_argument(
        "--branch-from",
        default=None,
        metavar="PARENT_SESSION_ID",
        help=(
            "Treat the ingested session as a branch off PARENT_SESSION_ID. "
            "Requires --branch-session and --at-seq. Only applied when the "
            "child session has no purpose row yet; subsequent ingests preserve "
            "the existing topology."
        ),
    )
    ingest_parser.add_argument(
        "--branch-session",
        default=None,
        metavar="SESSION_ID",
        help=("Session id of the branch being ingested. Required when --branch-from is set."),
    )
    ingest_parser.add_argument(
        "--at-seq",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Parent watermark the branch starts from. Turns with seq <= N in "
            "the JSONL are skipped because the parent already distilled them. "
            "Required when --branch-from is set."
        ),
    )
    ingest_parser.add_argument(
        "--branch-kind",
        default="experiment",
        choices=("main", "experiment"),
        help="Topology label for the new branch (default: experiment).",
    )

    branch_parser = sub.add_parser(
        "branch",
        help="Stage B+ branch management (close / list).",
    )
    branch_sub = branch_parser.add_subparsers(dest="branch_command", required=True)

    close_parser = branch_sub.add_parser(
        "close",
        help="Mark a branch session as closed (logical close, no row deletion).",
    )
    close_parser.add_argument("session_id", help="Session id of the branch to close.")

    list_parser = branch_sub.add_parser(
        "list",
        help="List branches in tree or JSON form.",
    )
    list_parser.add_argument(
        "--tree",
        action="store_true",
        help="Render branches as a parent->child tree (default when no --json).",
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="Render branches as a flat JSON array. Mutually exclusive with --tree.",
    )

    # ----- query ---------------------------------------------------------
    query_parser = sub.add_parser(
        "query",
        help="Hybrid search over learnings; canonical / emerging lanes.",
    )
    query_parser.add_argument(
        "text",
        help="Query text driving both BM25 and vector search.",
    )
    query_parser.add_argument(
        "--lane",
        choices=("canonical", "emerging", "both"),
        default="both",
        help="Which retrieval lane(s) to surface (default: both).",
    )
    query_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Per-lane result limit (default: 5).",
    )
    query_parser.add_argument(
        "--scope",
        default=None,
        choices=("session", "project", "group", "shared", "experiment"),
        help="Optional scope filter forwarded to search_hybrid.",
    )
    query_parser.add_argument(
        "--gap-session-id",
        default=None,
        metavar="SESSION_ID",
        help="If set, surface unresolved gaps for SESSION_ID (otherwise global).",
    )
    query_parser.add_argument(
        "--pack",
        action="store_true",
        help=(
            "Run the result through ContextPacker and emit Markdown instead "
            "of JSON. Use --token-budget to control the cap."
        ),
    )
    query_parser.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help=(
            "Token budget for --pack output. Defaults to DISTILL_CONTEXT_BUDGET_DEFAULT (config)."
        ),
    )
    query_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run with empty in-memory stores and stub embeddings; "
            "useful for argument-shape validation without a database."
        ),
    )

    # ----- export --------------------------------------------------------
    export_parser = sub.add_parser(
        "export",
        help="Dump a session's purpose / digest / learnings as JSON.",
    )
    export_parser.add_argument("session_id", help="Session to export.")
    export_parser.add_argument(
        "--include-archived",
        action="store_true",
        help="Include archived learnings in the exported document.",
    )
    export_parser.add_argument(
        "--include-side-relations",
        action="store_true",
        help=(
            "Include open conflicts and unresolved gaps tied to the session "
            "(or to its learnings) in the exported document."
        ),
    )

    # ----- aggregate -----------------------------------------------------
    aggregate_parser = sub.add_parser(
        "aggregate",
        help="Stage D group rollups (run / list).",
    )
    aggregate_sub = aggregate_parser.add_subparsers(dest="aggregate_command", required=True)

    aggregate_run = aggregate_sub.add_parser(
        "run",
        help="Run the Aggregator over one group_id and emit a fresh group_learning.",
    )
    aggregate_run.add_argument(
        "--group-id",
        required=True,
        help="group_id to roll up; all active learnings tagged with this group_id are fed in.",
    )
    aggregate_run.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run with in-memory stores and stub providers. The CLI seeds two "
            "fixture learnings into the in-memory store so the prompt + parser "
            "path is exercised without a database / LLM."
        ),
    )

    aggregate_list = aggregate_sub.add_parser(
        "list",
        help="List the latest rollup per group_id, or the full history of one group.",
    )
    aggregate_list.add_argument(
        "--group",
        default=None,
        metavar="GROUP_ID",
        help="If set, list every rollup for this group_id (newest first).",
    )

    # ----- gc ------------------------------------------------------------
    gc_parser = sub.add_parser(
        "gc",
        help=(
            "Garbage-collect long-archived rows. Defaults to --dry-run; "
            "supply --apply to actually delete."
        ),
    )
    gc_parser.add_argument(
        "--older-than-days",
        type=int,
        default=90,
        metavar="N",
        help="Operate on rows whose archived_at / resolved_at is older than N days.",
    )
    gc_parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete the matching rows. Without --apply this is a dry-run.",
    )

    return parser


def _cmd_check_config(show_defaults: bool) -> int:
    try:
        cfg = DistillerConfig.from_env()
    except DistillError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    payload: dict[str, object] = {
        "database_url": cfg.database_url,
        "llm_provider": cfg.llm_provider,
        "llm_model": cfg.llm_model,
        "embedding_provider": cfg.embedding_provider,
        "embedding_model": cfg.embedding_model,
        "embedding_dim": cfg.embedding_dim,
    }
    if show_defaults:
        payload.update(
            auto_turns=cfg.auto_turns,
            workers=cfg.workers,
            hnsw_m=cfg.hnsw_m,
            hnsw_efc=cfg.hnsw_efc,
            hnsw_ef=cfg.hnsw_ef,
            tau_merge=cfg.tau_merge,
            tau_conflict=cfg.tau_conflict,
            rrf_k=cfg.rrf_k,
            context_budget_default=cfg.context_budget_default,
        )
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


def _serialize_report(report: IngestReport) -> dict[str, object]:
    """Render an :class:`IngestReport` as a JSON-friendly dict.

    Curation decisions are flattened to action counts so the CLI output stays
    readable for large batches; full per-decision detail is available
    programmatically via :class:`IngestReport`.
    """

    sessions: list[dict[str, object]] = []
    for s in report.sessions:
        actions: dict[str, int] = {}
        if s.curation is not None:
            for d in s.curation.decisions:
                actions[d.action] = actions.get(d.action, 0) + 1
        sessions.append(
            {
                "session_id": s.session_id,
                "distilled": s.distilled,
                "prior_seq": s.prior_seq,
                "new_seq": s.new_seq,
                "candidate_count": s.candidate_count,
                "actions": actions,
                "error": s.error,
            }
        )
    return {
        "session_count": report.session_count,
        "distilled_count": report.distilled_count,
        "error_count": report.error_count,
        "sessions": sessions,
        "skipped_lines": [asdict(line) for line in report.skipped_lines],
    }


async def _run_dry_run_ingest(
    path: str,
    *,
    strict: bool,
    version_id: str,
    embedding_dim: int = 8,
    branch_plan: BranchPlan | None = None,
) -> IngestReport:
    """Execute an ingest with stub providers and in-memory stores.

    Mirrors the wiring used by ``tests/unit/pipeline/test_ingest.py`` so the
    CLI's dry-run path is the same code path the tests exercise.
    """

    llm = StubLLM(
        responses=[
            json.dumps(
                {
                    "purpose": {
                        "purpose": "(dry-run placeholder)",
                        "domain_tags": [],
                        "success_score": None,
                        "polluted": False,
                        "pollution_reason": None,
                    },
                    "digest": {"summary_md": "", "bm25_text": ""},
                    "learnings": [],
                }
            )
        ]
        * 1024  # generous so a many-session batch can drain
    )
    embedder = StubEmbedding(dimension=embedding_dim)
    distiller = Distiller(llm, embedder, version_id=version_id)
    learnings = InMemoryLearningStore()
    curator = Curator(learnings)
    runner = IngestRunner(
        distiller=distiller,
        curator=curator,
        watermarks=InMemoryWatermarkStore(),
        purposes=InMemoryPurposeStore(),
        digests=InMemoryDigestStore(),
        strict=strict,
    )
    return await runner.run_path(path, branch_plan=branch_plan)


async def _run_prod_ingest(
    path: str,
    *,
    strict: bool,
    version_id: str | None,
    branch_plan: BranchPlan | None = None,
) -> IngestReport:
    """Execute an ingest against asyncpg-backed stores and real providers."""

    from stratoclave_distill.db.asyncpg import (
        AsyncpgDigestStore,
        AsyncpgLearningStore,
        AsyncpgPurposeStore,
        AsyncpgWatermarkStore,
        pool_context,
    )

    cfg = DistillerConfig.from_env()
    llm = build_llm_provider(cfg)
    embedder = build_embedding_provider(cfg)
    if embedder.dimension != cfg.embedding_dim:
        raise DistillError(
            f"embedding provider reports dimension {embedder.dimension} "
            f"but DISTILL_EMBEDDING_DIM is {cfg.embedding_dim}"
        )
    effective_version = version_id or _default_prod_version_id()
    distiller = Distiller(llm, embedder, version_id=effective_version)

    async with pool_context(cfg.database_url) as pool:
        learnings = AsyncpgLearningStore(pool)
        curator = Curator(
            learnings,
            tau_merge=cfg.tau_merge,
            tau_conflict=cfg.tau_conflict,
            rrf_k=cfg.rrf_k,
        )
        runner = IngestRunner(
            distiller=distiller,
            curator=curator,
            watermarks=AsyncpgWatermarkStore(pool),
            purposes=AsyncpgPurposeStore(pool),
            digests=AsyncpgDigestStore(pool),
            strict=strict,
        )
        return await runner.run_path(path, branch_plan=branch_plan)


def _default_prod_version_id() -> str:
    from datetime import UTC, datetime

    return "ingest-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _build_branch_plan(
    *,
    branch_from: str | None,
    branch_session: str | None,
    at_seq: int | None,
    branch_kind: str,
) -> BranchPlan | None:
    """Validate the branching CLI flags and build a :class:`BranchPlan`.

    All-or-nothing: ``--branch-from``, ``--branch-session``, and ``--at-seq``
    must be supplied together. Returns ``None`` when no branching flags
    were provided so callers can keep the legacy ingest path unchanged.
    """

    flags_set = sum(x is not None for x in (branch_from, branch_session, at_seq))
    if flags_set == 0:
        return None
    if branch_from is None or branch_session is None or at_seq is None:
        raise DistillError(
            "--branch-from, --branch-session, and --at-seq must be supplied together"
        )
    return BranchPlan(
        session_id=branch_session,
        parent_session_id=branch_from,
        at_seq=at_seq,
        branch_kind=branch_kind,  # type: ignore[arg-type]
    )


def _cmd_ingest(
    *,
    path: str,
    dry_run: bool,
    strict: bool,
    version_id: str,
    prod_version_id: str | None,
    branch_from: str | None,
    branch_session: str | None,
    at_seq: int | None,
    branch_kind: str,
) -> int:
    try:
        plan = _build_branch_plan(
            branch_from=branch_from,
            branch_session=branch_session,
            at_seq=at_seq,
            branch_kind=branch_kind,
        )
        if dry_run:
            report = asyncio.run(
                _run_dry_run_ingest(path, strict=strict, version_id=version_id, branch_plan=plan)
            )
        else:
            report = asyncio.run(
                _run_prod_ingest(path, strict=strict, version_id=prod_version_id, branch_plan=plan)
            )
    except DistillError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    except ValueError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    sys.stdout.write(json.dumps(_serialize_report(report), indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


# --------------------------------------------------------------------------
# branch close / branch list
# --------------------------------------------------------------------------


def _utc_now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


async def _run_branch_close(session_id: str) -> dict[str, object]:
    from stratoclave_distill.db.asyncpg import (
        AsyncpgPurposeStore,
        pool_context,
    )

    cfg = DistillerConfig.from_env()
    timestamp = _utc_now_iso()
    async with pool_context(cfg.database_url) as pool:
        store = AsyncpgPurposeStore(pool)
        existing = await store.get(session_id)
        if existing is None:
            raise DistillError(f"branch close: no session_purposes row for {session_id!r}")
        if existing.branch_state == "closed":
            return {
                "session_id": session_id,
                "branch_state": "closed",
                "closed_at": existing.closed_at,
                "noop": True,
            }
        await store.set_branch_state(
            session_id,
            branch_state="closed",
            closed_at=timestamp,
            last_updated_at=timestamp,
        )
    return {
        "session_id": session_id,
        "branch_state": "closed",
        "closed_at": timestamp,
        "noop": False,
    }


def _serialize_purpose(p: SessionPurpose) -> dict[str, object]:
    return {
        "session_id": p.session_id,
        "purpose": p.purpose,
        "branch_kind": p.branch_kind,
        "branch_state": p.branch_state,
        "parent_session_id": p.parent_session_id,
        "branched_at_seq": p.branched_at_seq,
        "closed_at": p.closed_at,
        "polluted": p.polluted,
        "last_updated_at": p.last_updated_at,
    }


def _format_purpose_label(p: SessionPurpose) -> str:
    head = p.purpose if len(p.purpose) <= 60 else p.purpose[:57] + "..."
    kind = "exp" if p.branch_kind == "experiment" else "session"
    short_id = p.session_id[:8] if len(p.session_id) > 8 else p.session_id
    return f'{kind} {short_id}... [{p.branch_state}]   "{head}"'


def _render_branch_tree(rows: Sequence[SessionPurpose]) -> str:
    """Render branches as ``main\\n  child\\n    grandchild`` ASCII tree.

    Roots are sessions with ``parent_session_id is None``. Children are
    grouped under their parent and sorted by ``last_updated_at`` so the
    output is stable across runs.
    """

    children: dict[str | None, list[SessionPurpose]] = {}
    for row in rows:
        children.setdefault(row.parent_session_id, []).append(row)
    for bucket in children.values():
        bucket.sort(key=lambda r: (r.last_updated_at, r.session_id))

    lines: list[str] = ["main"]

    def walk(parent_id: str | None, depth: int) -> None:
        kids = children.get(parent_id, ())
        for kid in kids:
            indent = "  " * depth
            lines.append(f"{indent}{_format_purpose_label(kid)}")
            walk(kid.session_id, depth + 1)

    walk(None, 1)

    counts: dict[str, int] = {"open": 0, "closed": 0, "promoted": 0}
    for row in rows:
        counts[row.branch_state] = counts.get(row.branch_state, 0) + 1
    summary = ", ".join(
        f"{counts.get(state, 0)} {state}" for state in ("open", "closed", "promoted")
    )
    lines.append("")
    lines.append(f"statistics: {summary}")
    return "\n".join(lines)


async def _run_branch_list(*, as_json: bool) -> str:
    from stratoclave_distill.db.asyncpg import (
        AsyncpgPurposeStore,
        pool_context,
    )

    cfg = DistillerConfig.from_env()
    async with pool_context(cfg.database_url) as pool:
        store = AsyncpgPurposeStore(pool)
        rows = await store.list_branches()
    if as_json:
        payload = [_serialize_purpose(r) for r in rows]
        return json.dumps(payload, indent=2, sort_keys=True)
    return _render_branch_tree(rows)


def _cmd_branch_close(session_id: str) -> int:
    try:
        result = asyncio.run(_run_branch_close(session_id))
    except DistillError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


def _cmd_branch_list(*, tree: bool, as_json: bool) -> int:
    if tree and as_json:
        sys.stderr.write("error: --tree and --json are mutually exclusive\n")
        return 2
    use_json = as_json
    try:
        output = asyncio.run(_run_branch_list(as_json=use_json))
    except DistillError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    sys.stdout.write(output)
    sys.stdout.write("\n")
    return 0


# --------------------------------------------------------------------------
# query subcommand
# --------------------------------------------------------------------------


def _serialize_hit(hit: LearningSearchHit) -> dict[str, object]:
    """Render a :class:`LearningSearchHit` as a JSON-friendly dict.

    Keeps both the raw cosine and the RRF-fused score so downstream
    callers (or human readers) can pick the signal they need.
    """

    learning = hit.learning
    return {
        "learning_id": learning.learning_id,
        "scope": learning.scope,
        "claim_type": learning.claim_type,
        "rule": learning.rule,
        "why": learning.why,
        "evidence_count": learning.evidence_count,
        "created_at": learning.created_at,
        "cosine": hit.cosine,
        "vector_rank": hit.vector_rank,
        "bm25_rank": hit.bm25_rank,
        "rrf_score": hit.rrf_score,
    }


def _serialize_group_hit(hit: GroupLearningSearchHit) -> dict[str, object]:
    g = hit.group
    return {
        "group_learning_id": g.group_learning_id,
        "group_id": g.group_id,
        "summary_md": g.summary_md,
        "contributing_learnings": list(g.contributing_learnings),
        "created_at": g.created_at,
        "cosine": hit.cosine,
        "vector_rank": hit.vector_rank,
        "bm25_rank": hit.bm25_rank,
        "rrf_score": hit.rrf_score,
    }


def _serialize_retrieval(result: RetrievalResult) -> dict[str, object]:
    return {
        "query_text": result.query_text,
        "canonical": [_serialize_hit(h) for h in result.canonical],
        "emerging": [_serialize_hit(h) for h in result.emerging],
        "conflicts": [asdict(c) for c in result.conflicts],
        "gaps": [asdict(g) for g in result.gaps],
        "groups": [_serialize_group_hit(h) for h in result.groups],
    }


async def _run_dry_run_query(
    *,
    text: str,
    lane: str,
    limit: int,
    scope: LearningScope | None,
    gap_session_id: str | None,
    embedding_dim: int = 8,
) -> RetrievalResult:
    """Run a query with empty in-memory stores + stub embeddings.

    Returns an empty :class:`RetrievalResult`; the value is mostly to
    exercise the argument-validation and formatting paths from tests
    without a database.
    """

    embedder = StubEmbedding(dimension=embedding_dim)
    learnings = InMemoryLearningStore()
    conflicts = InMemoryConflictStore()
    gaps = InMemoryGapStore()
    retriever = Retriever(
        store=learnings,
        embedder=embedder,
        top_k_canonical=limit,
        top_k_emerging=limit,
        conflict_store=conflicts,
        gap_store=gaps,
    )
    result = await retriever.retrieve(
        text,
        scope=scope,
        gap_session_id=gap_session_id,
    )
    if lane == "canonical":
        return RetrievalResult(
            query_text=result.query_text,
            canonical=result.canonical,
            emerging=(),
            conflicts=result.conflicts,
            gaps=result.gaps,
        )
    if lane == "emerging":
        return RetrievalResult(
            query_text=result.query_text,
            canonical=(),
            emerging=result.emerging,
            conflicts=result.conflicts,
            gaps=result.gaps,
        )
    return result


async def _run_prod_query(
    *,
    text: str,
    lane: str,
    limit: int,
    scope: LearningScope | None,
    gap_session_id: str | None,
) -> RetrievalResult:
    """Run a query against the asyncpg-backed stores and real embedder."""

    from stratoclave_distill.db.asyncpg import (
        AsyncpgConflictStore,
        AsyncpgGapStore,
        AsyncpgLearningStore,
        pool_context,
    )

    cfg = DistillerConfig.from_env()
    embedder = build_embedding_provider(cfg)
    if embedder.dimension != cfg.embedding_dim:
        raise DistillError(
            f"embedding provider reports dimension {embedder.dimension} "
            f"but DISTILL_EMBEDDING_DIM is {cfg.embedding_dim}"
        )
    async with pool_context(cfg.database_url) as pool:
        learnings = AsyncpgLearningStore(pool)
        conflict_store = AsyncpgConflictStore(pool)
        gap_store = AsyncpgGapStore(pool)
        retriever = Retriever(
            store=learnings,
            embedder=embedder,
            top_k_canonical=limit,
            top_k_emerging=limit,
            rrf_k=cfg.rrf_k,
            conflict_store=conflict_store,
            gap_store=gap_store,
        )
        result = await retriever.retrieve(
            text,
            scope=scope,
            gap_session_id=gap_session_id,
        )
    if lane == "canonical":
        return RetrievalResult(
            query_text=result.query_text,
            canonical=result.canonical,
            emerging=(),
            conflicts=result.conflicts,
            gaps=result.gaps,
        )
    if lane == "emerging":
        return RetrievalResult(
            query_text=result.query_text,
            canonical=(),
            emerging=result.emerging,
            conflicts=result.conflicts,
            gaps=result.gaps,
        )
    return result


def _cmd_query(
    *,
    text: str,
    lane: str,
    limit: int,
    scope: str | None,
    gap_session_id: str | None,
    pack: bool,
    token_budget: int | None,
    dry_run: bool,
) -> int:
    if limit < 1:
        sys.stderr.write(f"error: --limit must be >= 1, got {limit}\n")
        return 2
    typed_scope: LearningScope | None = scope  # type: ignore[assignment]
    try:
        if dry_run:
            result = asyncio.run(
                _run_dry_run_query(
                    text=text,
                    lane=lane,
                    limit=limit,
                    scope=typed_scope,
                    gap_session_id=gap_session_id,
                )
            )
        else:
            result = asyncio.run(
                _run_prod_query(
                    text=text,
                    lane=lane,
                    limit=limit,
                    scope=typed_scope,
                    gap_session_id=gap_session_id,
                )
            )
    except DistillError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    except ValueError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2

    if pack:
        budget = token_budget
        if budget is None:
            budget = _resolve_default_budget()
        packer = ContextPacker(token_budget=budget)
        sys.stdout.write(packer.pack(result).markdown)
        if not packer.pack(result).markdown.endswith("\n"):
            sys.stdout.write("\n")
        return 0

    sys.stdout.write(json.dumps(_serialize_retrieval(result), indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


def _resolve_default_budget() -> int:
    """Read ``context_budget_default`` from env, fall back to a safe default.

    Calling :meth:`DistillerConfig.from_env` requires ``DATABASE_URL``,
    which is overkill when the caller only wants the budget. We attempt
    it but fall back to 2000 (the library default) on any config error.
    """

    try:
        return DistillerConfig.from_env().context_budget_default
    except DistillError:
        return 2000


# --------------------------------------------------------------------------
# export subcommand
# --------------------------------------------------------------------------


def _serialize_purpose_full(p: SessionPurpose) -> dict[str, object]:
    return {
        "session_id": p.session_id,
        "purpose": p.purpose,
        "domain_tags": list(p.domain_tags),
        "success_score": p.success_score,
        "polluted": p.polluted,
        "pollution_reason": p.pollution_reason,
        "branch_kind": p.branch_kind,
        "branch_state": p.branch_state,
        "parent_session_id": p.parent_session_id,
        "branched_at_seq": p.branched_at_seq,
        "closed_at": p.closed_at,
        "derived_from_version": p.derived_from_version,
        "derived_at": p.derived_at,
        "last_updated_at": p.last_updated_at,
    }


def _serialize_digest(d: SessionDigest) -> dict[str, object]:
    return {
        "digest_id": d.digest_id,
        "session_id": d.session_id,
        "version_id": d.version_id,
        "summary_md": d.summary_md,
        "bm25_text": d.bm25_text,
        "extracted_at": d.extracted_at,
    }


def _serialize_learning(learning: Learning) -> dict[str, object]:
    return {
        "learning_id": learning.learning_id,
        "scope": learning.scope,
        "claim_type": learning.claim_type,
        "rule": learning.rule,
        "why": learning.why,
        "triggers": dict(learning.triggers),
        "project_key": learning.project_key,
        "group_id": learning.group_id,
        "source_session": learning.source_session,
        "source_version": learning.source_version,
        "evidence_count": learning.evidence_count,
        "confidence": learning.confidence,
        "archived_at": learning.archived_at,
        "superseded_by": learning.superseded_by,
        "bm25_text": learning.bm25_text,
        "created_at": learning.created_at,
        "updated_at": learning.updated_at,
    }


async def _run_export(
    *,
    session_id: str,
    include_archived: bool,
    include_side_relations: bool,
) -> dict[str, object]:
    from stratoclave_distill.db.asyncpg import (
        AsyncpgConflictStore,
        AsyncpgDigestStore,
        AsyncpgGapStore,
        AsyncpgLearningStore,
        AsyncpgPurposeStore,
        pool_context,
    )

    cfg = DistillerConfig.from_env()
    async with pool_context(cfg.database_url) as pool:
        purposes = AsyncpgPurposeStore(pool)
        digests = AsyncpgDigestStore(pool)
        learnings_store = AsyncpgLearningStore(pool)
        conflict_store = AsyncpgConflictStore(pool)
        gap_store = AsyncpgGapStore(pool)

        purpose = await purposes.get(session_id)
        if purpose is None:
            raise DistillError(f"export: no session_purposes row for {session_id!r}")
        digest = await digests.get(session_id)

        all_learnings = await learnings_store.list_active()
        session_learnings = [
            learning for learning in all_learnings if learning.source_session == session_id
        ]
        if include_archived:
            # ``list_active`` excludes archived rows; we currently have no
            # bulk listing for archived rows, so we leave a placeholder
            # message and document the limitation.
            archived: list[Learning] = []
            archived_note = (
                "include_archived requested but archived listing is "
                "not yet exposed by LearningStore; future work."
            )
        else:
            archived = []
            archived_note = None

        conflicts: list[LearningConflict] = []
        gaps: list[SessionGap] = []
        if include_side_relations:
            for learning in session_learnings:
                conflicts.extend(await conflict_store.list_for(learning.learning_id))
            gaps_seq = await gap_store.list_unresolved(session_id=session_id)
            gaps = list(gaps_seq)

    payload: dict[str, object] = {
        "session_id": session_id,
        "purpose": _serialize_purpose_full(purpose),
        "digest": _serialize_digest(digest) if digest is not None else None,
        "learnings": [_serialize_learning(learning) for learning in session_learnings],
    }
    if include_archived:
        payload["archived_learnings"] = [_serialize_learning(learning) for learning in archived]
        if archived_note is not None:
            payload["archived_note"] = archived_note
    if include_side_relations:
        payload["conflicts"] = [asdict(c) for c in conflicts]
        payload["gaps"] = [asdict(g) for g in gaps]
    return payload


def _cmd_export(
    *,
    session_id: str,
    include_archived: bool,
    include_side_relations: bool,
) -> int:
    try:
        payload = asyncio.run(
            _run_export(
                session_id=session_id,
                include_archived=include_archived,
                include_side_relations=include_side_relations,
            )
        )
    except DistillError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


# --------------------------------------------------------------------------
# aggregate subcommand (Stage D)
# --------------------------------------------------------------------------


def _serialize_group_learning(g: GroupLearning) -> dict[str, object]:
    return {
        "group_learning_id": g.group_learning_id,
        "group_id": g.group_id,
        "summary_md": g.summary_md,
        "contributing_learnings": list(g.contributing_learnings),
        "bm25_text": g.bm25_text,
        "created_at": g.created_at,
    }


def _serialize_aggregation(result: AggregationResult) -> dict[str, object]:
    return {
        "group_learning": _serialize_group_learning(result.group_learning),
        "embedding_dim": len(result.embedding),
    }


def _dry_run_aggregate_responder(group_id: str) -> str:
    """Canned LLM response for ``aggregate run --dry-run``.

    Echoes the requested ``group_id`` so smoke tests can assert the
    dry-run envelope is wired through.
    """

    return json.dumps(
        {
            "summary_md": (
                f"Group {group_id} rollup (dry-run).\n\n"
                "- placeholder norm: stub aggregator output for fixture validation"
            ),
            "bm25_text": f"group {group_id} rollup placeholder norm",
        }
    )


async def _run_dry_run_aggregate(
    *,
    group_id: str,
    embedding_dim: int = 8,
) -> AggregationResult:
    """Aggregate with stub providers and an in-memory learnings store.

    Seeds two fixture learnings tagged with ``group_id`` so the prompt
    path exercises non-empty input. No database / LLM credentials are
    required.
    """

    learnings_store = InMemoryLearningStore()
    seeds = [
        Learning(
            learning_id=f"L-dry-{i}",
            scope="group",
            rule=f"dry-run rule {i}",
            why="seeded by aggregate --dry-run",
            group_id=group_id,
            bm25_text=f"dry-run rule {i}",
            claim_type="norm",
            evidence_count=3,
            created_at="2026-05-25T00:00:00Z",
            updated_at="2026-05-25T00:00:00Z",
        )
        for i in range(2)
    ]
    for seed in seeds:
        await learnings_store.insert(seed, embedding=[1.0] + [0.0] * (embedding_dim - 1))

    llm = StubLLM(responses=[_dry_run_aggregate_responder(group_id)])
    embedder = StubEmbedding(dimension=embedding_dim)
    aggregator = Aggregator(llm, embedder)
    return await aggregator.run(seeds, group_id=group_id)


async def _run_prod_aggregate(*, group_id: str) -> AggregationResult:
    """Aggregate against asyncpg-backed stores and the configured LLM.

    Pulls every active learning whose ``group_id`` matches the request,
    runs the Aggregator, and persists the rollup via
    :meth:`GroupLearningStore.upsert`.
    """

    from stratoclave_distill.db.asyncpg import (
        AsyncpgGroupLearningStore,
        AsyncpgLearningStore,
        pool_context,
    )

    cfg = DistillerConfig.from_env()
    llm = build_llm_provider(cfg)
    embedder = build_embedding_provider(cfg)
    if embedder.dimension != cfg.embedding_dim:
        raise DistillError(
            f"embedding provider reports dimension {embedder.dimension} "
            f"but DISTILL_EMBEDDING_DIM is {cfg.embedding_dim}"
        )

    aggregator = Aggregator(llm, embedder)
    async with pool_context(cfg.database_url) as pool:
        learnings_store = AsyncpgLearningStore(pool)
        group_store = AsyncpgGroupLearningStore(pool)
        all_active = await learnings_store.list_active()
        learnings = [learning for learning in all_active if learning.group_id == group_id]
        if not learnings:
            raise DistillError(f"aggregate: no active learnings tagged group_id={group_id!r}")
        result = await aggregator.run(learnings, group_id=group_id)
        await group_store.upsert(result.group_learning, embedding=list(result.embedding))
    return result


async def _run_aggregate_list(*, group_id: str | None) -> list[dict[str, object]]:
    from stratoclave_distill.db.asyncpg import (
        AsyncpgGroupLearningStore,
        pool_context,
    )

    cfg = DistillerConfig.from_env()
    async with pool_context(cfg.database_url) as pool:
        store = AsyncpgGroupLearningStore(pool)
        if group_id is not None:
            rows = await store.list_by_group(group_id, latest_only=False)
        else:
            rows = await store.list_latest_per_group()
    return [_serialize_group_learning(g) for g in rows]


def _cmd_aggregate_run(*, group_id: str, dry_run: bool) -> int:
    if not group_id:
        sys.stderr.write("error: --group-id must be a non-empty string\n")
        return 2
    try:
        if dry_run:
            result = asyncio.run(_run_dry_run_aggregate(group_id=group_id))
        else:
            result = asyncio.run(_run_prod_aggregate(group_id=group_id))
    except DistillError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    except ValueError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    sys.stdout.write(json.dumps(_serialize_aggregation(result), indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


def _cmd_aggregate_list(*, group_id: str | None) -> int:
    try:
        rows = asyncio.run(_run_aggregate_list(group_id=group_id))
    except DistillError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    sys.stdout.write(json.dumps(rows, indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


# --------------------------------------------------------------------------
# gc subcommand
# --------------------------------------------------------------------------


async def _run_gc(*, older_than_days: int, apply: bool) -> dict[str, object]:
    """Survey + optionally delete long-archived audit rows.

    The archived ``learnings`` rows are kept by Stage B/B+ as audit trail.
    Stage C exposes a CLI to drop rows older than ``older_than_days``.
    For Stage C we surface a count-only dry-run by default and gate the
    real DELETE behind ``--apply`` to honor production-safety norms.
    """

    from datetime import UTC, datetime, timedelta

    cfg = DistillerConfig.from_env()
    cutoff = datetime.now(UTC) - timedelta(days=older_than_days)
    cutoff_iso = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")

    from stratoclave_distill.db.asyncpg import pool_context

    async with pool_context(cfg.database_url) as pool, pool.acquire() as conn:
        archived_learnings = await conn.fetchval(
            "SELECT count(*) FROM learnings WHERE archived_at IS NOT NULL AND archived_at < $1",
            cutoff,
        )
        resolved_conflicts = await conn.fetchval(
            "SELECT count(*) FROM learning_conflicts WHERE resolution <> 'open' AND noted_at < $1",
            cutoff,
        )
        resolved_gaps = await conn.fetchval(
            "SELECT count(*) FROM session_gaps WHERE resolution <> 'open' AND noted_at < $1",
            cutoff,
        )
        survey: dict[str, object] = {
            "older_than_days": older_than_days,
            "cutoff": cutoff_iso,
            "archived_learnings_eligible": int(archived_learnings or 0),
            "resolved_conflicts_eligible": int(resolved_conflicts or 0),
            "resolved_gaps_eligible": int(resolved_gaps or 0),
            "applied": False,
        }
        if not apply:
            return survey
        async with conn.transaction():
            deleted_learnings = await conn.fetchval(
                "WITH deleted AS ("
                "DELETE FROM learnings "
                "WHERE archived_at IS NOT NULL AND archived_at < $1 "
                "RETURNING 1) SELECT count(*) FROM deleted",
                cutoff,
            )
            deleted_conflicts = await conn.fetchval(
                "WITH deleted AS ("
                "DELETE FROM learning_conflicts "
                "WHERE resolution <> 'open' AND noted_at < $1 "
                "RETURNING 1) SELECT count(*) FROM deleted",
                cutoff,
            )
            deleted_gaps = await conn.fetchval(
                "WITH deleted AS ("
                "DELETE FROM session_gaps "
                "WHERE resolution <> 'open' AND noted_at < $1 "
                "RETURNING 1) SELECT count(*) FROM deleted",
                cutoff,
            )
        survey.update(
            {
                "applied": True,
                "archived_learnings_deleted": int(deleted_learnings or 0),
                "resolved_conflicts_deleted": int(deleted_conflicts or 0),
                "resolved_gaps_deleted": int(deleted_gaps or 0),
            }
        )
        return survey


def _cmd_gc(*, older_than_days: int, apply: bool) -> int:
    if older_than_days < 0:
        sys.stderr.write(f"error: --older-than-days must be >= 0, got {older_than_days}\n")
        return 2
    try:
        result = asyncio.run(_run_gc(older_than_days=older_than_days, apply=apply))
    except DistillError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "version":
        sys.stdout.write(f"{__version__}\n")
        return 0
    if args.command == "check-config":
        return _cmd_check_config(show_defaults=args.show_defaults)
    if args.command == "ingest":
        return _cmd_ingest(
            path=args.path,
            dry_run=args.dry_run,
            strict=args.strict,
            version_id=args.version_id,
            prod_version_id=args.prod_version_id,
            branch_from=args.branch_from,
            branch_session=args.branch_session,
            at_seq=args.at_seq,
            branch_kind=args.branch_kind,
        )
    if args.command == "branch":
        if args.branch_command == "close":
            return _cmd_branch_close(args.session_id)
        if args.branch_command == "list":
            return _cmd_branch_list(tree=args.tree, as_json=args.json)
        parser.error(f"unknown branch command: {args.branch_command}")  # pragma: no cover
    if args.command == "query":
        return _cmd_query(
            text=args.text,
            lane=args.lane,
            limit=args.limit,
            scope=args.scope,
            gap_session_id=args.gap_session_id,
            pack=args.pack,
            token_budget=args.token_budget,
            dry_run=args.dry_run,
        )
    if args.command == "export":
        return _cmd_export(
            session_id=args.session_id,
            include_archived=args.include_archived,
            include_side_relations=args.include_side_relations,
        )
    if args.command == "aggregate":
        if args.aggregate_command == "run":
            return _cmd_aggregate_run(group_id=args.group_id, dry_run=args.dry_run)
        if args.aggregate_command == "list":
            return _cmd_aggregate_list(group_id=args.group)
        parser.error(f"unknown aggregate command: {args.aggregate_command}")  # pragma: no cover
    if args.command == "gc":
        return _cmd_gc(older_than_days=args.older_than_days, apply=args.apply)
    parser.error(f"unknown command: {args.command}")  # pragma: no cover
    return 2  # pragma: no cover - parser.error raises SystemExit


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
