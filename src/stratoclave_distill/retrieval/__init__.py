"""Hybrid retrieval and ContextPacker.

Stage B+ delivered the :class:`Retriever` that splits hits into
canonical / emerging lanes. Stage C composes a :class:`ContextPacker`
on top of :class:`RetrievalResult` to produce a budgeted Markdown
fragment grouped by lane and ``claim_type``, suitable for splicing
directly into a turn-level prompt.
"""

from stratoclave_distill.retrieval.packer import (
    DEFAULT_CHARS_PER_TOKEN,
    ContextPacker,
    TokenCounter,
    approximate_token_count,
)
from stratoclave_distill.retrieval.retriever import (
    RetrievalResult,
    Retriever,
    hits_for_learning,
    learning_ids,
)

__all__ = [
    "DEFAULT_CHARS_PER_TOKEN",
    "ContextPacker",
    "RetrievalResult",
    "Retriever",
    "TokenCounter",
    "approximate_token_count",
    "hits_for_learning",
    "learning_ids",
]
