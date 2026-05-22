"""Embedding provider abstractions.

Each provider takes a list of strings and returns a list of dense vectors,
all of the same length. The pipeline asserts that ``len(vector) ==
DistillerConfig.embedding_dim`` so a misconfigured model is caught
immediately.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from stratoclave_distill.config import DistillerConfig
from stratoclave_distill.core.errors import ConfigError, EmbeddingError


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Minimal embedding transport surface."""

    @property
    def model(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text.

        MUST raise :class:`EmbeddingError` on transport / format issues.
        """


class StubEmbedding:
    """Deterministic embedding for unit tests.

    The vector is derived from the SHA-256 of the input so that two identical
    inputs yield identical vectors, but unrelated inputs land in different
    regions of the vector space. This is enough for testing dedup, hybrid
    fusion, and dimension assertions without paying for a real model.
    """

    __slots__ = ("_calls", "_dimension", "_model")

    def __init__(self, *, dimension: int = 8, model: str = "stub") -> None:
        if dimension < 1:
            raise ConfigError(f"StubEmbedding dimension must be >= 1, got {dimension}")
        self._dimension = dimension
        self._model = model
        self._calls: list[tuple[str, ...]] = []

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def calls(self) -> tuple[tuple[str, ...], ...]:
        return tuple(self._calls)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self._calls.append(tuple(texts))
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw: list[float] = []
        for i in range(self._dimension):
            byte = digest[i % len(digest)]
            mixed = byte ^ ((i * 31) & 0xFF)
            raw.append((mixed - 128.0) / 128.0)
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]


class VoyageEmbedding:
    """Voyage AI adapter (lazy SDK import).

    Voyage's Python SDK is sync; we offload to a thread inside ``embed`` so
    the pipeline can keep its async surface even when the model client is
    synchronous.
    """

    __slots__ = ("_api_key", "_client", "_dimension", "_model")

    def __init__(self, *, model: str, dimension: int, api_key: str | None) -> None:
        if not model:
            raise ConfigError("VoyageEmbedding requires a model id")
        if dimension < 1:
            raise ConfigError(f"VoyageEmbedding dimension must be >= 1, got {dimension}")
        self._model = model
        self._dimension = dimension
        self._api_key = api_key
        self._client: object | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        import asyncio

        client = self._get_client()
        try:
            result = await asyncio.to_thread(
                lambda: client.embed(  # type: ignore[attr-defined]
                    list(texts), model=self._model, input_type="document"
                )
            )
        except Exception as exc:  # pragma: no cover
            raise EmbeddingError(f"Voyage embed failed: {exc}") from exc
        vectors = getattr(result, "embeddings", None)
        if vectors is None:
            raise EmbeddingError("Voyage response did not include embeddings")
        return [list(map(float, v)) for v in vectors]

    def _get_client(self) -> object:
        if self._client is None:
            try:
                import voyageai
            except ImportError as exc:
                raise EmbeddingError(
                    "voyageai SDK not installed. Install stratoclave-distill[voyage]."
                ) from exc
            self._client = (
                voyageai.Client(api_key=self._api_key) if self._api_key else voyageai.Client()
            )
        return self._client


class OpenAIEmbedding:
    """OpenAI embeddings adapter (lazy SDK import)."""

    __slots__ = ("_api_key", "_base_url", "_client", "_dimension", "_model")

    def __init__(
        self,
        *,
        model: str,
        dimension: int,
        api_key: str | None,
        base_url: str | None = None,
    ) -> None:
        if not model:
            raise ConfigError("OpenAIEmbedding requires a model id")
        if dimension < 1:
            raise ConfigError(f"OpenAIEmbedding dimension must be >= 1, got {dimension}")
        self._model = model
        self._dimension = dimension
        self._api_key = api_key
        self._base_url = base_url
        self._client: object | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        client = self._get_client()
        try:
            response = await client.embeddings.create(  # type: ignore[attr-defined]
                model=self._model, input=list(texts)
            )
        except Exception as exc:  # pragma: no cover
            raise EmbeddingError(f"OpenAI embed failed: {exc}") from exc
        try:
            return [list(map(float, item.embedding)) for item in response.data]
        except Exception as exc:  # pragma: no cover
            raise EmbeddingError(f"unexpected OpenAI embed response: {exc}") from exc

    def _get_client(self) -> object:
        if self._client is None:
            try:
                import openai
            except ImportError as exc:
                raise EmbeddingError(
                    "openai SDK not installed. Install stratoclave-distill[openai]."
                ) from exc
            kwargs: dict[str, object] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = openai.AsyncOpenAI(**kwargs)
        return self._client


def build_embedding_provider(cfg: DistillerConfig) -> EmbeddingProvider:
    """Construct the configured :class:`EmbeddingProvider`."""

    if cfg.embedding_provider == "voyage":
        return VoyageEmbedding(
            model=cfg.embedding_model,
            dimension=cfg.embedding_dim,
            api_key=cfg.embedding_api_key,
        )
    if cfg.embedding_provider == "openai":
        return OpenAIEmbedding(
            model=cfg.embedding_model,
            dimension=cfg.embedding_dim,
            api_key=cfg.embedding_api_key,
        )
    if cfg.embedding_provider == "stub":
        return StubEmbedding(dimension=cfg.embedding_dim, model=cfg.embedding_model)
    raise ConfigError(f"unknown embedding_provider: {cfg.embedding_provider!r}")
