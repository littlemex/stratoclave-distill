"""Command-line entrypoint for stratoclave-distill.

Stage A shipped ``version`` and ``check-config``. Stage B added ``ingest``
for processing a JSONL transcript end-to-end. Stage B+ extends ``ingest``
with ``--branch-from`` / ``--at-seq`` / ``--branch-kind`` and introduces
the ``branch`` subcommand group (``branch close``, ``branch list``).
The remaining real subcommands (``query`` / ``export`` / ``gc``) land
in Stage C.

``ingest`` runs in two modes:

- ``--dry-run`` (the default for tests and fixture validation) wires
  in-memory stores + stub providers so no database or LLM credentials
  are required;
- the production mode (no ``--dry-run``) reads :class:`DistillerConfig`
  from the environment and stands up the asyncpg-backed stores plus the
  real LLM / Embedding providers.

The ``branch`` subcommand group always uses :class:`DistillerConfig` to
open a real Postgres connection — there is no dry-run for branch state
because there is nothing to inspect in an in-memory store between CLI
invocations.
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
from stratoclave_distill.core.types import SessionPurpose
from stratoclave_distill.db.memory import (
    InMemoryDigestStore,
    InMemoryLearningStore,
    InMemoryPurposeStore,
    InMemoryWatermarkStore,
)
from stratoclave_distill.pipeline import (
    BranchPlan,
    Curator,
    Distiller,
    IngestReport,
    IngestRunner,
)
from stratoclave_distill.providers.embedding import StubEmbedding, build_embedding_provider
from stratoclave_distill.providers.llm import StubLLM, build_llm_provider


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
    parser.error(f"unknown command: {args.command}")  # pragma: no cover
    return 2  # pragma: no cover - parser.error raises SystemExit


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
