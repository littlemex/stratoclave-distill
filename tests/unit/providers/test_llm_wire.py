"""Wire-level tests for the real LLM adapters.

These tests use injected fake SDK clients so we can prove the adapter
correctly:

- builds the SDK request payload from a sequence of :class:`LLMMessage`
  (system/user/assistant separation for Anthropic, role passthrough for
  OpenAI),
- forwards ``max_tokens`` / ``temperature`` to the SDK,
- extracts the assistant text from the SDK response shape,
- wraps any SDK exception (transport, schema) in :class:`LLMError` so
  callers stay backend-agnostic.

The fake clients live in this module so the tests run without the real
``anthropic`` / ``openai`` packages installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from stratoclave_distill.core.errors import LLMError
from stratoclave_distill.providers import AnthropicLLM, LLMMessage, OpenAILLM


@dataclass
class _AnthropicTextBlock:
    text: str
    type: str = "text"


@dataclass
class _AnthropicResponse:
    content: list[Any]


class _AnthropicMessages:
    def __init__(
        self, response: _AnthropicResponse | None = None, raise_with: Exception | None = None
    ) -> None:
        self._response = response
        self._raise_with = raise_with
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _AnthropicResponse:
        self.calls.append(kwargs)
        if self._raise_with is not None:
            raise self._raise_with
        assert self._response is not None
        return self._response


@dataclass
class _AnthropicFakeClient:
    messages: _AnthropicMessages = field(default_factory=lambda: _AnthropicMessages())


@dataclass
class _OpenAIChoice:
    message: Any


@dataclass
class _OpenAIMessage:
    content: str | None


@dataclass
class _OpenAIResponse:
    choices: list[_OpenAIChoice]


class _OpenAICompletions:
    def __init__(
        self, response: _OpenAIResponse | None = None, raise_with: Exception | None = None
    ) -> None:
        self._response = response
        self._raise_with = raise_with
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _OpenAIResponse:
        self.calls.append(kwargs)
        if self._raise_with is not None:
            raise self._raise_with
        assert self._response is not None
        return self._response


@dataclass
class _OpenAIChat:
    completions: _OpenAICompletions


@dataclass
class _OpenAIFakeClient:
    chat: _OpenAIChat


@pytest.mark.asyncio
async def test_anthropic_complete_returns_concatenated_text_blocks() -> None:
    response = _AnthropicResponse(
        content=[_AnthropicTextBlock(text="hello "), _AnthropicTextBlock(text="world")]
    )
    fake = _AnthropicFakeClient(messages=_AnthropicMessages(response=response))
    llm = AnthropicLLM(model="claude-haiku-4-5-20251001")
    object.__setattr__(llm, "_client", fake)

    out = await llm.complete(
        [LLMMessage(role="user", content="hi")], max_tokens=42, temperature=0.3
    )
    assert out == "hello world"

    [call] = fake.messages.calls
    assert call["model"] == "claude-haiku-4-5-20251001"
    assert call["max_tokens"] == 42
    assert call["temperature"] == pytest.approx(0.3)
    assert call["system"] is None
    assert call["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_anthropic_complete_collapses_system_blocks() -> None:
    response = _AnthropicResponse(content=[_AnthropicTextBlock(text="ok")])
    fake = _AnthropicFakeClient(messages=_AnthropicMessages(response=response))
    llm = AnthropicLLM(model="claude-haiku-4-5-20251001")
    object.__setattr__(llm, "_client", fake)

    await llm.complete(
        [
            LLMMessage(role="system", content="be terse"),
            LLMMessage(role="system", content="answer in english"),
            LLMMessage(role="user", content="hi"),
            LLMMessage(role="assistant", content="hello"),
            LLMMessage(role="user", content="what is two plus two"),
        ]
    )

    [call] = fake.messages.calls
    assert call["system"] == "be terse\n\nanswer in english"
    assert call["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "what is two plus two"},
    ]


@pytest.mark.asyncio
async def test_anthropic_complete_skips_non_text_content_blocks() -> None:
    @dataclass
    class _ToolUseBlock:
        type: str = "tool_use"
        id: str = "abc"

    response = _AnthropicResponse(
        content=[
            _AnthropicTextBlock(text="answer:"),
            _ToolUseBlock(),
            _AnthropicTextBlock(text=" 4"),
        ]
    )
    fake = _AnthropicFakeClient(messages=_AnthropicMessages(response=response))
    llm = AnthropicLLM(model="claude-haiku-4-5-20251001")
    object.__setattr__(llm, "_client", fake)

    out = await llm.complete([LLMMessage(role="user", content="add 2+2")])
    assert out == "answer: 4"


@pytest.mark.asyncio
async def test_anthropic_complete_wraps_sdk_exception() -> None:
    fake = _AnthropicFakeClient(
        messages=_AnthropicMessages(raise_with=RuntimeError("network down"))
    )
    llm = AnthropicLLM(model="claude-haiku-4-5-20251001")
    object.__setattr__(llm, "_client", fake)

    with pytest.raises(LLMError, match=r"Anthropic completion failed.*network down"):
        await llm.complete([LLMMessage(role="user", content="hi")])


@pytest.mark.asyncio
async def test_openai_complete_returns_choice_text() -> None:
    response = _OpenAIResponse(choices=[_OpenAIChoice(message=_OpenAIMessage(content="hi there"))])
    completions = _OpenAICompletions(response=response)
    fake = _OpenAIFakeClient(chat=_OpenAIChat(completions=completions))
    llm = OpenAILLM(model="gpt-4o-mini")
    object.__setattr__(llm, "_client", fake)

    out = await llm.complete(
        [
            LLMMessage(role="system", content="be terse"),
            LLMMessage(role="user", content="ping"),
        ],
        max_tokens=10,
        temperature=0.0,
    )
    assert out == "hi there"

    [call] = completions.calls
    assert call["model"] == "gpt-4o-mini"
    assert call["max_tokens"] == 10
    assert call["temperature"] == pytest.approx(0.0)
    assert call["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "ping"},
    ]


@pytest.mark.asyncio
async def test_openai_complete_handles_null_message_content() -> None:
    response = _OpenAIResponse(choices=[_OpenAIChoice(message=_OpenAIMessage(content=None))])
    completions = _OpenAICompletions(response=response)
    fake = _OpenAIFakeClient(chat=_OpenAIChat(completions=completions))
    llm = OpenAILLM(model="gpt-4o-mini")
    object.__setattr__(llm, "_client", fake)

    out = await llm.complete([LLMMessage(role="user", content="ping")])
    assert out == ""


@pytest.mark.asyncio
async def test_openai_complete_wraps_sdk_exception() -> None:
    completions = _OpenAICompletions(raise_with=RuntimeError("rate limit"))
    fake = _OpenAIFakeClient(chat=_OpenAIChat(completions=completions))
    llm = OpenAILLM(model="gpt-4o-mini")
    object.__setattr__(llm, "_client", fake)

    with pytest.raises(LLMError, match=r"OpenAI completion failed.*rate limit"):
        await llm.complete([LLMMessage(role="user", content="hi")])


@pytest.mark.asyncio
async def test_anthropic_get_client_raises_when_sdk_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lazy SDK import must surface as :class:`LLMError`, not :class:`ImportError`."""

    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "anthropic":
            raise ImportError("No module named 'anthropic'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    llm = AnthropicLLM(model="claude-haiku-4-5-20251001")
    with pytest.raises(LLMError, match="Anthropic SDK not installed"):
        await llm.complete([LLMMessage(role="user", content="hi")])


@pytest.mark.asyncio
async def test_openai_get_client_raises_when_sdk_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "openai":
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    llm = OpenAILLM(model="gpt-4o-mini")
    with pytest.raises(LLMError, match="OpenAI SDK not installed"):
        await llm.complete([LLMMessage(role="user", content="hi")])


def test_anthropic_get_client_constructs_sdk_with_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lazy import path must forward ``api_key`` and ``base_url`` to the SDK."""

    import sys

    constructed: list[dict[str, Any]] = []

    class _FakeAsyncAnthropic:
        def __init__(self, **kwargs: Any) -> None:
            constructed.append(kwargs)

    class _FakeAnthropicModule:
        AsyncAnthropic = _FakeAsyncAnthropic

    monkeypatch.setitem(sys.modules, "anthropic", _FakeAnthropicModule())
    llm = AnthropicLLM(model="claude-haiku-4-5-20251001", api_key="sk-test", base_url="https://x")
    client = llm._get_client()
    assert isinstance(client, _FakeAsyncAnthropic)
    assert constructed == [{"api_key": "sk-test", "base_url": "https://x"}]
    # Second call returns the cached client without reconstructing.
    assert llm._get_client() is client
    assert len(constructed) == 1


def test_anthropic_get_client_omits_unset_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Optional kwargs (api_key/base_url) must not be forwarded when ``None``."""

    import sys

    constructed: list[dict[str, Any]] = []

    class _FakeAsyncAnthropic:
        def __init__(self, **kwargs: Any) -> None:
            constructed.append(kwargs)

    class _FakeAnthropicModule:
        AsyncAnthropic = _FakeAsyncAnthropic

    monkeypatch.setitem(sys.modules, "anthropic", _FakeAnthropicModule())
    llm = AnthropicLLM(model="claude-haiku-4-5-20251001")
    llm._get_client()
    assert constructed == [{}]


def test_openai_get_client_constructs_sdk_with_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    constructed: list[dict[str, Any]] = []

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            constructed.append(kwargs)

    class _FakeOpenAIModule:
        AsyncOpenAI = _FakeAsyncOpenAI

    monkeypatch.setitem(sys.modules, "openai", _FakeOpenAIModule())
    llm = OpenAILLM(model="gpt-4o-mini", api_key="sk-x", base_url="https://y")
    client = llm._get_client()
    assert isinstance(client, _FakeAsyncOpenAI)
    assert constructed == [{"api_key": "sk-x", "base_url": "https://y"}]
    assert llm._get_client() is client


def test_openai_get_client_omits_unset_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    constructed: list[dict[str, Any]] = []

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            constructed.append(kwargs)

    class _FakeOpenAIModule:
        AsyncOpenAI = _FakeAsyncOpenAI

    monkeypatch.setitem(sys.modules, "openai", _FakeOpenAIModule())
    llm = OpenAILLM(model="gpt-4o-mini")
    llm._get_client()
    assert constructed == [{}]
