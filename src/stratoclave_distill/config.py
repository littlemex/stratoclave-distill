"""Runtime configuration for stratoclave-distill.

Configuration is sourced in this priority order, highest first:

1. Explicit kwargs passed to :class:`DistillerConfig`.
2. Environment variables documented in ``docs/DESIGN.md`` section 6.10.
3. Library defaults defined in this module.

Hard-coded paths, URLs, and credentials are deliberately absent: every
provider-specific setting must come from configuration. See
``docs/PROJECT_RULES.md`` for the no-hardcode policy.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Literal, cast

from stratoclave_distill.core.errors import ConfigError

LLMProviderName = Literal["anthropic", "openai", "stub"]
EmbeddingProviderName = Literal["voyage", "openai", "stub"]


_DEFAULT_LLM_MODEL = "claude-haiku-4-5-20251001"
_DEFAULT_EMBEDDING_MODEL = "voyage-3"
_DEFAULT_EMBEDDING_DIM = 1024
_DEFAULT_AUTO_TURNS = 20
_DEFAULT_WORKERS = 2
_DEFAULT_HNSW_M = 16
_DEFAULT_HNSW_EFC = 64
_DEFAULT_HNSW_EF = 64
_DEFAULT_TAU_MERGE = 0.95
_DEFAULT_TAU_CONFLICT = 0.80
_DEFAULT_RRF_K = 60
_DEFAULT_CONTEXT_BUDGET = 2000


def _read_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc


def _read_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a float, got {raw!r}") from exc


@dataclass(frozen=True, slots=True)
class DistillerConfig:
    """Frozen runtime configuration for the distill pipeline.

    The defaults are chosen so that calling ``DistillerConfig.from_env({})``
    with the bare-minimum environment (``DATABASE_URL`` and one provider key)
    yields a working configuration. Anything fancier can be overridden via
    kwargs or env vars.
    """

    database_url: str
    llm_provider: LLMProviderName = "anthropic"
    llm_model: str = _DEFAULT_LLM_MODEL
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    embedding_provider: EmbeddingProviderName = "voyage"
    embedding_model: str = _DEFAULT_EMBEDDING_MODEL
    embedding_dim: int = _DEFAULT_EMBEDDING_DIM
    embedding_api_key: str | None = None
    auto_turns: int = _DEFAULT_AUTO_TURNS
    workers: int = _DEFAULT_WORKERS
    hnsw_m: int = _DEFAULT_HNSW_M
    hnsw_efc: int = _DEFAULT_HNSW_EFC
    hnsw_ef: int = _DEFAULT_HNSW_EF
    tau_merge: float = _DEFAULT_TAU_MERGE
    tau_conflict: float = _DEFAULT_TAU_CONFLICT
    rrf_k: int = _DEFAULT_RRF_K
    context_budget_default: int = _DEFAULT_CONTEXT_BUDGET
    extra: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.database_url:
            raise ConfigError("database_url must be a non-empty string")
        if self.embedding_dim <= 0:
            raise ConfigError(f"embedding_dim must be positive, got {self.embedding_dim}")
        if not 0.0 <= self.tau_conflict <= self.tau_merge <= 1.0:
            raise ConfigError(
                "tau thresholds must satisfy 0 <= tau_conflict <= tau_merge <= 1, "
                f"got tau_conflict={self.tau_conflict}, tau_merge={self.tau_merge}"
            )
        if self.workers < 1:
            raise ConfigError(f"workers must be >= 1, got {self.workers}")
        if self.rrf_k < 1:
            raise ConfigError(f"rrf_k must be >= 1, got {self.rrf_k}")
        if self.context_budget_default < 1:
            raise ConfigError(
                f"context_budget_default must be >= 1, got {self.context_budget_default}"
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **overrides: object) -> DistillerConfig:
        """Build a config from a mapping (defaults to ``os.environ``).

        Explicit ``overrides`` take precedence over the env mapping. This is
        the entrypoint that the CLI and tests use; calling code should never
        read ``os.environ`` directly to keep configuration auditable.
        """

        src: Mapping[str, str] = os.environ if env is None else env

        def pop_str(key: str, env_key: str, default: str = "") -> str:
            if key in overrides:
                value = overrides.pop(key)
                return "" if value is None else str(value)
            return src.get(env_key, default)

        def pop_optional_str(key: str, env_key: str) -> str | None:
            if key in overrides:
                value = overrides.pop(key)
                if value is None:
                    return None
                text = str(value)
                return text or None
            raw = src.get(env_key)
            return raw or None

        def pop_int(key: str, env_key: str, default: int) -> int:
            if key in overrides:
                value = overrides.pop(key)
                return int(cast(int, value))
            return _read_int(src, env_key, default)

        def pop_float(key: str, env_key: str, default: float) -> float:
            if key in overrides:
                value = overrides.pop(key)
                return float(cast(float, value))
            return _read_float(src, env_key, default)

        database_url = pop_str("database_url", "DATABASE_URL")
        if not database_url:
            raise ConfigError("DATABASE_URL is required")

        llm_provider = pop_str("llm_provider", "DISTILL_LLM_PROVIDER", "anthropic")
        if llm_provider not in ("anthropic", "openai", "stub"):
            raise ConfigError(
                f"unsupported llm_provider {llm_provider!r}; "
                "expected one of: anthropic, openai, stub"
            )

        embedding_provider = pop_str("embedding_provider", "DISTILL_EMBEDDING_PROVIDER", "voyage")
        if embedding_provider not in ("voyage", "openai", "stub"):
            raise ConfigError(
                f"unsupported embedding_provider {embedding_provider!r}; "
                "expected one of: voyage, openai, stub"
            )

        cfg = cls(
            database_url=database_url,
            llm_provider=cast(LLMProviderName, llm_provider),
            llm_model=pop_str("llm_model", "DISTILL_LLM_MODEL", _DEFAULT_LLM_MODEL),
            llm_base_url=pop_optional_str("llm_base_url", "DISTILL_LLM_BASE_URL"),
            llm_api_key=pop_optional_str("llm_api_key", "DISTILL_LLM_API_KEY"),
            embedding_provider=cast(EmbeddingProviderName, embedding_provider),
            embedding_model=pop_str(
                "embedding_model", "DISTILL_EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL
            ),
            embedding_dim=pop_int("embedding_dim", "DISTILL_EMBEDDING_DIM", _DEFAULT_EMBEDDING_DIM),
            embedding_api_key=pop_optional_str("embedding_api_key", "DISTILL_EMBEDDING_API_KEY"),
            auto_turns=pop_int("auto_turns", "DISTILL_AUTO_TURNS", _DEFAULT_AUTO_TURNS),
            workers=pop_int("workers", "DISTILL_WORKERS", _DEFAULT_WORKERS),
            hnsw_m=pop_int("hnsw_m", "DISTILL_HNSW_M", _DEFAULT_HNSW_M),
            hnsw_efc=pop_int("hnsw_efc", "DISTILL_HNSW_EFC", _DEFAULT_HNSW_EFC),
            hnsw_ef=pop_int("hnsw_ef", "DISTILL_HNSW_EF", _DEFAULT_HNSW_EF),
            tau_merge=pop_float("tau_merge", "DISTILL_TAU_MERGE", _DEFAULT_TAU_MERGE),
            tau_conflict=pop_float("tau_conflict", "DISTILL_TAU_CONFLICT", _DEFAULT_TAU_CONFLICT),
            rrf_k=pop_int("rrf_k", "DISTILL_RRF_K", _DEFAULT_RRF_K),
            context_budget_default=pop_int(
                "context_budget_default",
                "DISTILL_CONTEXT_BUDGET_DEFAULT",
                _DEFAULT_CONTEXT_BUDGET,
            ),
        )
        if overrides:
            unknown = ", ".join(sorted(overrides))
            raise ConfigError(f"unknown configuration overrides: {unknown}")
        return cfg

    def field_names(self) -> tuple[str, ...]:
        """Return all dataclass field names. Useful for diagnostics."""

        return tuple(f.name for f in fields(self))
