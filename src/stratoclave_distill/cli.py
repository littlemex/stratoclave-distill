"""Command-line entrypoint for stratoclave-distill.

Stage A shipped ``version`` and ``check-config``. Stage B adds ``ingest``
for processing a JSONL transcript end-to-end. The remaining real
subcommands (``query`` / ``export`` / ``gc``) land in Stage C.

``ingest`` runs in two modes:

- ``--dry-run`` (the default for tests and fixture validation) wires
  in-memory stores + stub providers so no database or LLM credentials
  are required;
- the production mode (no ``--dry-run``) reads :class:`DistillerConfig`
  from the environment and stands up the asyncpg-backed stores plus the
  real LLM / Embedding providers.
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
    path: str, *, strict: bool, version_id: str, embedding_dim: int = 8
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
    return await runner.run_path(path)


async def _run_prod_ingest(path: str, *, strict: bool, version_id: str | None) -> IngestReport:
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
        return await runner.run_path(path)


def _default_prod_version_id() -> str:
    from datetime import UTC, datetime

    return "ingest-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _cmd_ingest(
    *,
    path: str,
    dry_run: bool,
    strict: bool,
    version_id: str,
    prod_version_id: str | None,
) -> int:
    try:
        if dry_run:
            report = asyncio.run(_run_dry_run_ingest(path, strict=strict, version_id=version_id))
        else:
            report = asyncio.run(_run_prod_ingest(path, strict=strict, version_id=prod_version_id))
    except DistillError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    sys.stdout.write(json.dumps(_serialize_report(report), indent=2, sort_keys=True))
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
        )
    parser.error(f"unknown command: {args.command}")  # pragma: no cover
    return 2  # pragma: no cover - parser.error raises SystemExit


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
