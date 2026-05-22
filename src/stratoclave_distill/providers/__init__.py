"""LLM and embedding provider abstractions for stratoclave-distill.

The :class:`LLMProvider` and :class:`EmbeddingProvider` Protocols are the
seams that let tests inject deterministic stubs and that let production code
swap between Anthropic / OpenAI / Voyage without touching pipeline logic.
"""

from stratoclave_distill.providers.embedding import (
    EmbeddingProvider,
    OpenAIEmbedding,
    StubEmbedding,
    VoyageEmbedding,
    build_embedding_provider,
)
from stratoclave_distill.providers.llm import (
    AnthropicLLM,
    LLMMessage,
    LLMProvider,
    OpenAILLM,
    StubLLM,
    build_llm_provider,
)

__all__ = [
    "AnthropicLLM",
    "EmbeddingProvider",
    "LLMMessage",
    "LLMProvider",
    "OpenAIEmbedding",
    "OpenAILLM",
    "StubEmbedding",
    "StubLLM",
    "VoyageEmbedding",
    "build_embedding_provider",
    "build_llm_provider",
]
