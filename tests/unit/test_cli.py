"""Smoke tests for the CLI entrypoint.

These tests assert the exit code and stdout/stderr contracts so packaging
changes (e.g. switching script wiring) cannot regress them silently. The
CLI grows substantially in Stage B and C; this file is the harness those
later subcommands extend.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stratoclave_distill import __version__
from stratoclave_distill.cli import main


def test_version_command_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["version"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == __version__


def test_version_flag_is_supported(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_check_config_emits_json(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+asyncpg://distill:distill@localhost:5432/distill"
    )
    monkeypatch.setenv("DISTILL_LLM_PROVIDER", "stub")
    monkeypatch.setenv("DISTILL_EMBEDDING_PROVIDER", "stub")
    monkeypatch.setenv("DISTILL_EMBEDDING_DIM", "8")

    rc = main(["check-config"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["llm_provider"] == "stub"
    assert payload["embedding_dim"] == 8


def test_check_config_returns_nonzero_when_misconfigured(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    rc = main(["check-config"])
    assert rc == 2
    assert "DATABASE_URL is required" in capsys.readouterr().err


def test_check_config_show_defaults_includes_threshold_fields(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+asyncpg://distill:distill@localhost:5432/distill"
    )
    monkeypatch.setenv("DISTILL_LLM_PROVIDER", "stub")
    monkeypatch.setenv("DISTILL_EMBEDDING_PROVIDER", "stub")
    monkeypatch.setenv("DISTILL_EMBEDDING_DIM", "8")

    rc = main(["check-config", "--show-defaults"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["tau_merge"] == pytest.approx(0.95)
    assert payload["rrf_k"] == 60


# --------------------------------------------------------------------------
# ingest subcommand
# --------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def test_ingest_dry_run_emits_report(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
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

    rc = main(["ingest", str(path), "--dry-run"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["session_count"] == 1
    assert payload["distilled_count"] == 1
    assert payload["error_count"] == 0
    [session] = payload["sessions"]
    assert session["session_id"] == "s-1"
    assert session["new_seq"] == 2
    assert session["distilled"] is True
    assert payload["skipped_lines"] == []


def test_ingest_without_dry_run_surfaces_config_error(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production ingest reads DistillerConfig.from_env; missing DATABASE_URL surfaces."""

    monkeypatch.delenv("DATABASE_URL", raising=False)
    path = tmp_path / "session.jsonl"
    path.write_text("", encoding="utf-8")

    rc = main(["ingest", str(path)])
    assert rc == 2
    assert "DATABASE_URL is required" in capsys.readouterr().err


def test_ingest_dry_run_reports_skipped_lines(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = tmp_path / "session.jsonl"
    path.write_text(
        json.dumps(
            {
                "turn_id": "t-1",
                "session_id": "s-1",
                "seq": 1,
                "role": "user",
                "text_content": "hello",
                "occurred_at": "2026-05-22T00:00:00Z",
            }
        )
        + "\nnot json at all\n",
        encoding="utf-8",
    )

    rc = main(["ingest", str(path), "--dry-run"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["distilled_count"] == 1
    assert len(payload["skipped_lines"]) == 1
    assert payload["skipped_lines"][0]["line_no"] == 2


def test_ingest_dry_run_strict_aborts_on_bad_line(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = tmp_path / "session.jsonl"
    path.write_text("not json at all\n", encoding="utf-8")

    rc = main(["ingest", str(path), "--dry-run", "--strict"])
    assert rc == 2
    assert "error:" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Stage B+ ingest --branch-from
# --------------------------------------------------------------------------


def test_ingest_branch_flags_require_full_set(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = tmp_path / "session.jsonl"
    _write_jsonl(
        path,
        [
            {
                "turn_id": "t-1",
                "session_id": "s-child",
                "seq": 6,
                "role": "user",
                "text_content": "hi",
                "occurred_at": "2026-05-22T00:00:00Z",
            }
        ],
    )

    rc = main(
        [
            "ingest",
            str(path),
            "--dry-run",
            "--branch-from",
            "s-parent",
        ]
    )
    assert rc == 2
    assert "must be supplied together" in capsys.readouterr().err


def test_ingest_branch_dry_run_passes_through(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = tmp_path / "session.jsonl"
    _write_jsonl(
        path,
        [
            {
                "turn_id": "t-1",
                "session_id": "s-child",
                "seq": 6,
                "role": "user",
                "text_content": "hi",
                "occurred_at": "2026-05-22T00:00:00Z",
            },
            {
                "turn_id": "t-2",
                "session_id": "s-child",
                "seq": 7,
                "role": "assistant",
                "text_content": "ok",
                "occurred_at": "2026-05-22T00:00:01Z",
            },
        ],
    )

    rc = main(
        [
            "ingest",
            str(path),
            "--dry-run",
            "--branch-from",
            "s-parent",
            "--branch-session",
            "s-child",
            "--at-seq",
            "5",
            "--branch-kind",
            "experiment",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["distilled_count"] == 1
    [session] = payload["sessions"]
    assert session["session_id"] == "s-child"
    assert session["prior_seq"] == 5  # bumped by the branch plan
    assert session["new_seq"] == 7


def test_branch_list_requires_database_url(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    rc = main(["branch", "list"])
    assert rc == 2
    assert "DATABASE_URL is required" in capsys.readouterr().err


def test_branch_close_requires_database_url(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    rc = main(["branch", "close", "s-child"])
    assert rc == 2
    assert "DATABASE_URL is required" in capsys.readouterr().err


def test_branch_list_rejects_both_tree_and_json(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+asyncpg://distill:distill@localhost:5432/distill"
    )
    monkeypatch.setenv("DISTILL_LLM_PROVIDER", "stub")
    monkeypatch.setenv("DISTILL_EMBEDDING_PROVIDER", "stub")
    monkeypatch.setenv("DISTILL_EMBEDDING_DIM", "8")
    rc = main(["branch", "list", "--tree", "--json"])
    assert rc == 2
    assert "mutually exclusive" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Stage C: query subcommand
# --------------------------------------------------------------------------


def test_query_dry_run_emits_empty_lanes_as_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["query", "flaky tests", "--dry-run"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["query_text"] == "flaky tests"
    assert payload["canonical"] == []
    assert payload["emerging"] == []
    assert payload["conflicts"] == []
    assert payload["gaps"] == []


def test_query_dry_run_pack_emits_markdown(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(
        [
            "query",
            "flaky tests",
            "--dry-run",
            "--pack",
            "--token-budget",
            "200",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # Title + query echo are rendered even when both lanes are empty.
    assert out.startswith("# Distilled context\n")
    assert "_query_: flaky tests" in out


def test_query_rejects_zero_limit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["query", "anything", "--dry-run", "--limit", "0"])
    assert rc == 2
    assert "--limit must be" in capsys.readouterr().err


def test_query_lane_filter_canonical_only_drops_emerging(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["query", "x", "--dry-run", "--lane", "canonical"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["canonical"] == []
    assert payload["emerging"] == []


def test_query_prod_path_requires_database_url(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    rc = main(["query", "anything"])
    assert rc == 2
    assert "DATABASE_URL is required" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Stage C: export subcommand
# --------------------------------------------------------------------------


def test_export_requires_database_url(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    rc = main(["export", "s-1"])
    assert rc == 2
    assert "DATABASE_URL is required" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Stage C: gc subcommand
# --------------------------------------------------------------------------


def test_gc_rejects_negative_age(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["gc", "--older-than-days", "-1"])
    assert rc == 2
    assert "must be" in capsys.readouterr().err


def test_gc_requires_database_url(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    rc = main(["gc"])
    assert rc == 2
    assert "DATABASE_URL is required" in capsys.readouterr().err


# --------------------------------------------------------------------------
# aggregate (Stage D)
# --------------------------------------------------------------------------


def test_aggregate_run_dry_run_emits_group_learning(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["aggregate", "run", "--group-id", "g-cli-test", "--dry-run"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    g = payload["group_learning"]
    assert g["group_id"] == "g-cli-test"
    assert g["contributing_learnings"] == ["L-dry-0", "L-dry-1"]
    assert g["summary_md"].startswith("Group g-cli-test rollup")
    assert payload["embedding_dim"] == 8


def test_aggregate_run_rejects_empty_group_id(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["aggregate", "run", "--group-id", "", "--dry-run"])
    assert rc == 2
    assert "must be a non-empty string" in capsys.readouterr().err


def test_aggregate_list_requires_database_url(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    rc = main(["aggregate", "list"])
    assert rc == 2
    assert "DATABASE_URL is required" in capsys.readouterr().err
