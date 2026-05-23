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
