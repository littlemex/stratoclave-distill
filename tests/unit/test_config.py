"""Tests for :class:`DistillerConfig`.

Covered behaviours (every one is a real requirement of the runtime):

- ``DATABASE_URL`` is mandatory; missing or empty must raise.
- Provider names are validated against the enum so a typo is caught early.
- Numeric env vars round-trip through their type and reject garbage.
- The threshold invariant ``0 <= tau_conflict <= tau_merge <= 1`` holds.
- Explicit kwargs override env values (priority order).
- ``from_env`` rejects unknown override keys instead of silently dropping them.
"""

from __future__ import annotations

import pytest

from stratoclave_distill import DistillerConfig
from stratoclave_distill.core.errors import ConfigError


def _base_env() -> dict[str, str]:
    return {"DATABASE_URL": "postgresql+asyncpg://distill@localhost/distill"}


def test_database_url_is_required() -> None:
    with pytest.raises(ConfigError, match="DATABASE_URL is required"):
        DistillerConfig.from_env({})


def test_blank_database_url_is_rejected() -> None:
    with pytest.raises(ConfigError, match="DATABASE_URL is required"):
        DistillerConfig.from_env({"DATABASE_URL": ""})


def test_unknown_llm_provider_is_rejected() -> None:
    env = _base_env() | {"DISTILL_LLM_PROVIDER": "cohere"}
    with pytest.raises(ConfigError, match="unsupported llm_provider"):
        DistillerConfig.from_env(env)


def test_unknown_embedding_provider_is_rejected() -> None:
    env = _base_env() | {"DISTILL_EMBEDDING_PROVIDER": "vertex"}
    with pytest.raises(ConfigError, match="unsupported embedding_provider"):
        DistillerConfig.from_env(env)


def test_non_integer_int_env_var_raises() -> None:
    env = _base_env() | {"DISTILL_AUTO_TURNS": "not-a-number"}
    with pytest.raises(ConfigError, match="DISTILL_AUTO_TURNS must be an integer"):
        DistillerConfig.from_env(env)


def test_non_float_threshold_env_var_raises() -> None:
    env = _base_env() | {"DISTILL_TAU_MERGE": "abc"}
    with pytest.raises(ConfigError, match="DISTILL_TAU_MERGE must be a float"):
        DistillerConfig.from_env(env)


def test_threshold_invariant_is_enforced() -> None:
    env = _base_env() | {"DISTILL_TAU_MERGE": "0.5", "DISTILL_TAU_CONFLICT": "0.8"}
    with pytest.raises(ConfigError, match="tau thresholds"):
        DistillerConfig.from_env(env)


def test_workers_must_be_at_least_one() -> None:
    env = _base_env() | {"DISTILL_WORKERS": "0"}
    with pytest.raises(ConfigError, match="workers must be >= 1"):
        DistillerConfig.from_env(env)


def test_explicit_overrides_take_precedence_over_env() -> None:
    env = _base_env() | {"DISTILL_LLM_MODEL": "from-env"}
    cfg = DistillerConfig.from_env(env, llm_model="from-kwarg")
    assert cfg.llm_model == "from-kwarg"


def test_unknown_overrides_are_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown configuration overrides: typo"):
        DistillerConfig.from_env(_base_env(), typo="value")


def test_defaults_are_consistent_with_design() -> None:
    cfg = DistillerConfig.from_env(_base_env())
    assert cfg.llm_provider == "anthropic"
    assert cfg.embedding_provider == "voyage"
    assert cfg.embedding_dim == 1024
    assert cfg.auto_turns == 20
    assert cfg.workers == 2
    assert cfg.hnsw_m == 16
    assert cfg.hnsw_efc == 64
    assert cfg.tau_merge == pytest.approx(0.95)
    assert cfg.tau_conflict == pytest.approx(0.80)
    assert cfg.rrf_k == 60
    assert cfg.context_budget_default == 2000


def test_field_names_returns_all_fields() -> None:
    cfg = DistillerConfig.from_env(_base_env())
    names = cfg.field_names()
    assert "database_url" in names
    assert "embedding_dim" in names
    assert "tau_conflict" in names


def test_negative_embedding_dim_rejected() -> None:
    with pytest.raises(ConfigError, match="embedding_dim must be positive"):
        DistillerConfig(database_url="x", embedding_dim=0)


def test_direct_construct_rejects_blank_database_url() -> None:
    """``__post_init__`` enforces the same invariant as ``from_env``."""

    with pytest.raises(ConfigError, match="database_url must be a non-empty string"):
        DistillerConfig(database_url="")


def test_direct_construct_rejects_invalid_rrf_k() -> None:
    with pytest.raises(ConfigError, match="rrf_k must be >= 1"):
        DistillerConfig(database_url="x", rrf_k=0)


def test_direct_construct_rejects_invalid_context_budget() -> None:
    with pytest.raises(ConfigError, match="context_budget_default must be >= 1"):
        DistillerConfig(database_url="x", context_budget_default=0)


def test_from_env_int_override_kwarg_path() -> None:
    """Override path for int fields converts the kwarg via ``int()``."""

    cfg = DistillerConfig.from_env(_base_env(), workers=4, hnsw_m=32, rrf_k=120)
    assert cfg.workers == 4
    assert cfg.hnsw_m == 32
    assert cfg.rrf_k == 120


def test_from_env_float_override_kwarg_path() -> None:
    cfg = DistillerConfig.from_env(_base_env(), tau_merge=0.9, tau_conflict=0.5)
    assert cfg.tau_merge == pytest.approx(0.9)
    assert cfg.tau_conflict == pytest.approx(0.5)


def test_from_env_optional_str_override_kwarg_path() -> None:
    """When a kwarg supplies an optional string, the env value is ignored."""

    env = _base_env() | {"DISTILL_LLM_API_KEY": "from-env"}
    cfg = DistillerConfig.from_env(env, llm_api_key="from-kwarg")
    assert cfg.llm_api_key == "from-kwarg"


def test_from_env_optional_str_override_none_clears_value() -> None:
    """``llm_api_key=None`` (kwarg) must override a populated env var."""

    env = _base_env() | {"DISTILL_LLM_API_KEY": "from-env"}
    cfg = DistillerConfig.from_env(env, llm_api_key=None)
    assert cfg.llm_api_key is None


def test_from_env_optional_str_override_empty_string_clears_value() -> None:
    cfg = DistillerConfig.from_env(_base_env(), llm_base_url="")
    assert cfg.llm_base_url is None


def test_from_env_uses_os_environ_when_no_mapping_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling ``from_env()`` with no mapping reads from ``os.environ``."""

    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://distill@localhost/distill")
    monkeypatch.setenv("DISTILL_LLM_PROVIDER", "stub")
    monkeypatch.setenv("DISTILL_EMBEDDING_PROVIDER", "stub")
    monkeypatch.setenv("DISTILL_EMBEDDING_DIM", "16")
    cfg = DistillerConfig.from_env()
    assert cfg.llm_provider == "stub"
    assert cfg.embedding_dim == 16


def test_blank_int_env_var_falls_back_to_default() -> None:
    """An empty string for a numeric env var yields the default, not an error."""

    cfg = DistillerConfig.from_env(_base_env() | {"DISTILL_AUTO_TURNS": ""})
    assert cfg.auto_turns == 20
