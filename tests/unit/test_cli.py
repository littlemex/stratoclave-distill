"""Smoke tests for the CLI entrypoint.

These tests assert the exit code and stdout/stderr contracts so packaging
changes (e.g. switching script wiring) cannot regress them silently. The
CLI grows substantially in Stage B and C; this file is the harness those
later subcommands extend.
"""

from __future__ import annotations

import json

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
