"""LLM provider abstractions.

Every provider implementation must be importable without the optional
provider-specific package installed. The actual SDK import happens lazily on
first call, so a CI image that only installs the ``[dev]`` extras can still
load the module to exercise the stub.

A provider's job is narrow: take a list of role-tagged messages and return a
single string completion. The pipeline layer (:mod:`stratoclave_distill.pipeline`)
owns prompt construction, JSON parsing, retries, and rate limiting so that
each provider stays a thin transport adapter.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from stratoclave_distill.config import DistillerConfig
from stratoclave_distill.core.errors import ConfigError, LLMError

LLMRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class LLMMessage:
    """A single chat turn passed to :meth:`LLMProvider.complete`."""

    role: LLMRole
    content: str


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal LLM transport surface used by the distill pipeline."""

    @property
    def model(self) -> str:
        """Return the model identifier this provider was configured with."""

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Run a non-streaming completion and return the assistant text.

        Implementations MUST raise :class:`stratoclave_distill.core.errors.LLMError`
        on transport failures or malformed responses, never the SDK's own
        exception types. This keeps callers backend-agnostic.
        """


class StubLLM:
    """In-memory deterministic LLM used by unit tests and offline tooling.

    The stub records every call so that tests can assert on how the pipeline
    constructed prompts. Responses can be queued either as a ``Sequence[str]``
    (consumed in order, raising once exhausted) or via a ``responder`` callable
    for richer fixture logic.
    """

    __slots__ = ("_calls", "_model", "_responder", "_responses")

    def __init__(
        self,
        responses: Sequence[str] | None = None,
        *,
        responder: Callable[[Sequence[LLMMessage]], str] | None = None,
        model: str = "stub",
    ) -> None:
        if responses is None and responder is None:
            raise ConfigError("StubLLM requires either responses or responder")
        if responses is not None and responder is not None:
            raise ConfigError("StubLLM accepts responses xor responder, not both")
        self._responses: list[str] = list(responses or [])
        self._responder = responder
        self._model = model
        self._calls: list[tuple[LLMMessage, ...]] = []

    @property
    def model(self) -> str:
        return self._model

    @property
    def calls(self) -> tuple[tuple[LLMMessage, ...], ...]:
        """All prompts the pipeline has issued so far, in call order."""

        return tuple(self._calls)

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        del max_tokens, temperature  # stub ignores generation knobs
        self._calls.append(tuple(messages))
        if self._responder is not None:
            return self._responder(messages)
        if not self._responses:
            raise LLMError("StubLLM exhausted its scripted responses")
        return self._responses.pop(0)


class AnthropicLLM:
    """Anthropic Messages API adapter (lazy SDK import).

    The SDK is imported inside ``complete`` so that tests can construct the
    object without ``anthropic`` installed. Production callers should install
    ``stratoclave-distill[anthropic]``.
    """

    __slots__ = ("_api_key", "_base_url", "_client", "_model")

    def __init__(
        self, *, model: str, api_key: str | None = None, base_url: str | None = None
    ) -> None:
        if not model:
            raise ConfigError("AnthropicLLM requires a model id")
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._client: object | None = None

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        client = self._get_client()
        system_blocks = [m.content for m in messages if m.role == "system"]
        chat = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]
        try:
            response = await client.messages.create(  # type: ignore[attr-defined]
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                system="\n\n".join(system_blocks) if system_blocks else None,
                messages=chat,
            )
        except Exception as exc:  # pragma: no cover - depends on SDK errors
            raise LLMError(f"Anthropic completion failed: {exc}") from exc

        try:
            blocks = response.content
            return "".join(
                getattr(b, "text", "") for b in blocks if getattr(b, "type", "") == "text"
            )
        except Exception as exc:  # pragma: no cover
            raise LLMError(f"unexpected Anthropic response shape: {exc}") from exc

    def _get_client(self) -> object:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise LLMError(
                    "Anthropic SDK not installed. Install stratoclave-distill[anthropic]."
                ) from exc
            kwargs: dict[str, object] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = anthropic.AsyncAnthropic(**kwargs)
        return self._client


class OpenAILLM:
    """OpenAI Chat Completions adapter (lazy SDK import)."""

    __slots__ = ("_api_key", "_base_url", "_client", "_model")

    def __init__(
        self, *, model: str, api_key: str | None = None, base_url: str | None = None
    ) -> None:
        if not model:
            raise ConfigError("OpenAILLM requires a model id")
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._client: object | None = None

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        client = self._get_client()
        try:
            response = await client.chat.completions.create(  # type: ignore[attr-defined]
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": m.role, "content": m.content} for m in messages],
            )
        except Exception as exc:  # pragma: no cover
            raise LLMError(f"OpenAI completion failed: {exc}") from exc
        try:
            choice = response.choices[0]
            return str(choice.message.content or "")
        except Exception as exc:  # pragma: no cover
            raise LLMError(f"unexpected OpenAI response shape: {exc}") from exc

    def _get_client(self) -> object:
        if self._client is None:
            try:
                import openai
            except ImportError as exc:
                raise LLMError(
                    "OpenAI SDK not installed. Install stratoclave-distill[openai]."
                ) from exc
            kwargs: dict[str, object] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = openai.AsyncOpenAI(**kwargs)
        return self._client


def build_llm_provider(cfg: DistillerConfig) -> LLMProvider:
    """Construct the configured :class:`LLMProvider`.

    Returns :class:`StubLLM` only when ``llm_provider == "stub"``; this is
    used in unit tests that assemble a real :class:`DistillerConfig` and the
    rest of the pipeline but want to inject scripted LLM output.
    """

    if cfg.llm_provider == "anthropic":
        return AnthropicLLM(model=cfg.llm_model, api_key=cfg.llm_api_key, base_url=cfg.llm_base_url)
    if cfg.llm_provider == "openai":
        return OpenAILLM(model=cfg.llm_model, api_key=cfg.llm_api_key, base_url=cfg.llm_base_url)
    if cfg.llm_provider == "stub":
        return StubLLM(responses=[""], model=cfg.llm_model)
    raise ConfigError(f"unknown llm_provider: {cfg.llm_provider!r}")
