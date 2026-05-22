"""Command-line entrypoint for stratoclave-distill.

Stage A only ships ``version`` and ``check-config`` so users can verify their
install and environment without needing a database. Real subcommands
(``ingest`` / ``query`` / ``export`` / ``gc``) land in Stage B and Stage C.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from stratoclave_distill import __version__
from stratoclave_distill.config import DistillerConfig
from stratoclave_distill.core.errors import DistillError


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "version":
        sys.stdout.write(f"{__version__}\n")
        return 0
    if args.command == "check-config":
        return _cmd_check_config(show_defaults=args.show_defaults)
    parser.error(f"unknown command: {args.command}")  # pragma: no cover
    return 2  # pragma: no cover - parser.error raises SystemExit


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
