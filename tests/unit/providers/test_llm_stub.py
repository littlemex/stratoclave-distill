"""Tests for the LLM provider abstractions.

The stub is the contract test: any real provider that ships later must
emit identical errors and accept identical inputs. The lazy SDK
imports for :class:`AnthropicLLM` / :class:`OpenAILLM` are exercised
indirectly: instantiation must succeed without the SDK installed; only
``complete`` raises. We assert that contract here.
"""

from __future__ import annotations

import pytest

from stratoclave_distill import DistillerConfig
from stratoclave_distill.core.errors import ConfigError, LLMError
from stratoclave_distill.providers import (
    AnthropicLLM,
    LLMMessage,
    OpenAILLM,
    StubLLM,
    build_llm_provider,
)


@pytest.mark.asyncio
async def test_stub_returns_scripted_response() -> None:
    stub = StubLLM(responses=["hi"])
    out = await stub.complete([LLMMessage(role="user", content="ping")])
    assert out == "hi"


@pytest.mark.asyncio
async def test_stub_records_calls_in_order() -> None:
    stub = StubLLM(responses=["a", "b"])
    await stub.complete([LLMMessage(role="user", content="one")])
    await stub.complete([LLMMessage(role="user", content="two")])
    assert len(stub.calls) == 2
    assert stub.calls[0][0].content == "one"
    assert stub.calls[1][0].content == "two"


@pytest.mark.asyncio
async def test_stub_raises_when_exhausted() -> None:
    stub = StubLLM(responses=["only-one"])
    await stub.complete([LLMMessage(role="user", content="ok")])
    with pytest.raises(LLMError, match="exhausted"):
        await stub.complete([LLMMessage(role="user", content="boom")])


def test_stub_requires_either_responses_or_responder() -> None:
    with pytest.raises(ConfigError):
        StubLLM()  # type: ignore[call-arg]


def test_stub_rejects_both_responses_and_responder() -> None:
    with pytest.raises(ConfigError):
        StubLLM(responses=["x"], responder=lambda _msgs: "y")


@pytest.mark.asyncio
async def test_stub_responder_callable_is_invoked() -> None:
    stub = StubLLM(responder=lambda msgs: f"echo:{msgs[-1].content}")
    out = await stub.complete([LLMMessage(role="user", content="hello")])
    assert out == "echo:hello"


def test_anthropic_constructor_does_not_import_sdk() -> None:
    """Should be cheap to construct in environments without anthropic installed."""

    obj = AnthropicLLM(model="claude-haiku-4-5-20251001")
    assert obj.model == "claude-haiku-4-5-20251001"


def test_openai_constructor_does_not_import_sdk() -> None:
    obj = OpenAILLM(model="gpt-4o-mini")
    assert obj.model == "gpt-4o-mini"


def test_anthropic_requires_model_id() -> None:
    with pytest.raises(ConfigError):
        AnthropicLLM(model="")


def test_openai_requires_model_id() -> None:
    with pytest.raises(ConfigError):
        OpenAILLM(model="")


def test_build_llm_provider_dispatches_by_provider_name() -> None:
    cfg = DistillerConfig(database_url="x", llm_provider="anthropic")
    assert isinstance(build_llm_provider(cfg), AnthropicLLM)
    cfg2 = DistillerConfig(database_url="x", llm_provider="openai")
    assert isinstance(build_llm_provider(cfg2), OpenAILLM)
    cfg3 = DistillerConfig(database_url="x", llm_provider="stub")
    assert isinstance(build_llm_provider(cfg3), StubLLM)


def test_build_llm_provider_rejects_unknown_provider() -> None:
    cfg = DistillerConfig(database_url="x", llm_provider="anthropic")
    bogus = type(cfg)(database_url="x")
    object.__setattr__(bogus, "llm_provider", "nope")
    with pytest.raises(ConfigError):
        build_llm_provider(bogus)


def test_stub_llm_exposes_model_property() -> None:
    stub = StubLLM(responses=["x"], model="custom-model")
    assert stub.model == "custom-model"
