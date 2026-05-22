"""Wire-level tests for the real embedding adapters.

Mirrors :mod:`tests.unit.providers.test_llm_wire` but for embeddings:
prove that the adapter forwards model/input correctly, parses the SDK
response, and wraps SDK errors in :class:`EmbeddingError`. Tests run
without ``voyageai`` / ``openai`` installed by injecting fake clients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from stratoclave_distill.core.errors import EmbeddingError
from stratoclave_distill.providers import OpenAIEmbedding, VoyageEmbedding


@dataclass
class _VoyageResult:
    embeddings: list[list[float]]


class _VoyageFakeClient:
    """Sync API mirror of ``voyageai.Client.embed``."""

    def __init__(
        self, embeddings: list[list[float]] | None = None, raise_with: Exception | None = None
    ) -> None:
        self._embeddings = embeddings
        self._raise_with = raise_with
        self.calls: list[dict[str, Any]] = []

    def embed(self, texts: list[str], *, model: str, input_type: str) -> _VoyageResult:
        self.calls.append({"texts": texts, "model": model, "input_type": input_type})
        if self._raise_with is not None:
            raise self._raise_with
        assert self._embeddings is not None
        return _VoyageResult(embeddings=self._embeddings)


@pytest.mark.asyncio
async def test_voyage_embed_returns_floats_and_forwards_args() -> None:
    fake = _VoyageFakeClient(embeddings=[[1.0, 2.0], [3.0, 4.0]])
    emb = VoyageEmbedding(model="voyage-3", dimension=2, api_key="vk-1")
    object.__setattr__(emb, "_client", fake)

    out = await emb.embed(["hello", "world"])
    assert out == [[1.0, 2.0], [3.0, 4.0]]
    [call] = fake.calls
    assert call["texts"] == ["hello", "world"]
    assert call["model"] == "voyage-3"
    assert call["input_type"] == "document"


@pytest.mark.asyncio
async def test_voyage_embed_coerces_numeric_types_to_float() -> None:
    """SDK may return Decimal / numpy.float32; the adapter must hand back plain floats."""

    fake = _VoyageFakeClient(embeddings=[[1, 2, 3]])  # ints, on purpose
    emb = VoyageEmbedding(model="voyage-3", dimension=3, api_key=None)
    object.__setattr__(emb, "_client", fake)

    [vec] = await emb.embed(["x"])
    assert vec == [1.0, 2.0, 3.0]
    assert all(isinstance(v, float) for v in vec)


@pytest.mark.asyncio
async def test_voyage_embed_wraps_sdk_exception() -> None:
    fake = _VoyageFakeClient(raise_with=RuntimeError("boom"))
    emb = VoyageEmbedding(model="voyage-3", dimension=2, api_key=None)
    object.__setattr__(emb, "_client", fake)
    with pytest.raises(EmbeddingError, match=r"Voyage embed failed.*boom"):
        await emb.embed(["a"])


@pytest.mark.asyncio
async def test_voyage_embed_rejects_missing_embeddings() -> None:
    """A response without ``.embeddings`` must surface as a clean error."""

    class _BadResultClient:
        def embed(self, texts: list[str], *, model: str, input_type: str) -> object:
            return object()  # no .embeddings

    emb = VoyageEmbedding(model="voyage-3", dimension=2, api_key=None)
    object.__setattr__(emb, "_client", _BadResultClient())
    with pytest.raises(EmbeddingError, match="did not include embeddings"):
        await emb.embed(["a"])


@pytest.mark.asyncio
async def test_voyage_get_client_raises_when_sdk_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "voyageai":
            raise ImportError("No module named 'voyageai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    emb = VoyageEmbedding(model="voyage-3", dimension=4, api_key=None)
    with pytest.raises(EmbeddingError, match="voyageai SDK not installed"):
        await emb.embed(["a"])


@dataclass
class _OpenAIEmbeddingItem:
    embedding: list[float]


@dataclass
class _OpenAIEmbeddingResponse:
    data: list[_OpenAIEmbeddingItem]


class _OpenAIEmbeddings:
    def __init__(
        self,
        response: _OpenAIEmbeddingResponse | None = None,
        raise_with: Exception | None = None,
    ) -> None:
        self._response = response
        self._raise_with = raise_with
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _OpenAIEmbeddingResponse:
        self.calls.append(kwargs)
        if self._raise_with is not None:
            raise self._raise_with
        assert self._response is not None
        return self._response


@dataclass
class _OpenAIEmbeddingClient:
    embeddings: _OpenAIEmbeddings = field(default_factory=lambda: _OpenAIEmbeddings())


@pytest.mark.asyncio
async def test_openai_embed_returns_floats_and_forwards_args() -> None:
    response = _OpenAIEmbeddingResponse(
        data=[
            _OpenAIEmbeddingItem(embedding=[0.1, 0.2, 0.3]),
            _OpenAIEmbeddingItem(embedding=[0.4, 0.5, 0.6]),
        ]
    )
    embeddings = _OpenAIEmbeddings(response=response)
    fake = _OpenAIEmbeddingClient(embeddings=embeddings)
    emb = OpenAIEmbedding(
        model="text-embedding-3-small", dimension=3, api_key="sk-x", base_url="https://x"
    )
    object.__setattr__(emb, "_client", fake)

    out = await emb.embed(["a", "b"])
    assert out == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    [call] = embeddings.calls
    assert call["model"] == "text-embedding-3-small"
    assert call["input"] == ["a", "b"]


@pytest.mark.asyncio
async def test_openai_embed_wraps_sdk_exception() -> None:
    embeddings = _OpenAIEmbeddings(raise_with=RuntimeError("server overloaded"))
    fake = _OpenAIEmbeddingClient(embeddings=embeddings)
    emb = OpenAIEmbedding(model="text-embedding-3-small", dimension=3, api_key=None)
    object.__setattr__(emb, "_client", fake)
    with pytest.raises(EmbeddingError, match=r"OpenAI embed failed.*server overloaded"):
        await emb.embed(["a"])


@pytest.mark.asyncio
async def test_openai_embed_rejects_unexpected_response_shape() -> None:
    """If ``response.data`` is missing, the adapter must surface a clean error."""

    class _BadEmbeddings:
        async def create(self, **kwargs: Any) -> object:
            return object()  # no .data

    fake = _OpenAIEmbeddingClient(embeddings=_BadEmbeddings())  # type: ignore[arg-type]
    emb = OpenAIEmbedding(model="text-embedding-3-small", dimension=3, api_key=None)
    object.__setattr__(emb, "_client", fake)
    with pytest.raises(EmbeddingError, match="unexpected OpenAI embed response"):
        await emb.embed(["a"])


@pytest.mark.asyncio
async def test_openai_embedding_get_client_raises_when_sdk_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "openai":
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    emb = OpenAIEmbedding(model="text-embedding-3-small", dimension=3, api_key=None)
    with pytest.raises(EmbeddingError, match="openai SDK not installed"):
        await emb.embed(["a"])


def test_voyage_get_client_constructs_sdk_with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    constructed: list[Any] = []

    class _FakeVoyageClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            constructed.append({"args": args, "kwargs": kwargs})

    class _FakeVoyageModule:
        Client = _FakeVoyageClient

    monkeypatch.setitem(sys.modules, "voyageai", _FakeVoyageModule())
    emb = VoyageEmbedding(model="voyage-3", dimension=4, api_key="vk-1")
    client = emb._get_client()
    assert isinstance(client, _FakeVoyageClient)
    assert constructed == [{"args": (), "kwargs": {"api_key": "vk-1"}}]
    # Second call hits the cache.
    assert emb._get_client() is client


def test_voyage_get_client_omits_api_key_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    constructed: list[Any] = []

    class _FakeVoyageClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            constructed.append({"args": args, "kwargs": kwargs})

    class _FakeVoyageModule:
        Client = _FakeVoyageClient

    monkeypatch.setitem(sys.modules, "voyageai", _FakeVoyageModule())
    emb = VoyageEmbedding(model="voyage-3", dimension=4, api_key=None)
    emb._get_client()
    assert constructed == [{"args": (), "kwargs": {}}]


def test_openai_embedding_get_client_constructs_sdk_with_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    constructed: list[dict[str, Any]] = []

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            constructed.append(kwargs)

    class _FakeOpenAIModule:
        AsyncOpenAI = _FakeAsyncOpenAI

    monkeypatch.setitem(sys.modules, "openai", _FakeOpenAIModule())
    emb = OpenAIEmbedding(
        model="text-embedding-3-small",
        dimension=3,
        api_key="sk-x",
        base_url="https://y",
    )
    client = emb._get_client()
    assert isinstance(client, _FakeAsyncOpenAI)
    assert constructed == [{"api_key": "sk-x", "base_url": "https://y"}]
    assert emb._get_client() is client


def test_openai_embedding_get_client_omits_unset_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    constructed: list[dict[str, Any]] = []

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            constructed.append(kwargs)

    class _FakeOpenAIModule:
        AsyncOpenAI = _FakeAsyncOpenAI

    monkeypatch.setitem(sys.modules, "openai", _FakeOpenAIModule())
    emb = OpenAIEmbedding(model="text-embedding-3-small", dimension=3, api_key=None)
    emb._get_client()
    assert constructed == [{}]
