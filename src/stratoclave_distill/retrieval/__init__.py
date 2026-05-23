"""Hybrid retrieval and ContextPacker (lands in Stage C).

The :class:`Retriever` exposed here is Stage B+'s canonical / emerging
lane separator. Stage C's ContextPacker will compose on top of
:class:`RetrievalResult`.
"""

from stratoclave_distill.retrieval.retriever import (
    RetrievalResult,
    Retriever,
    hits_for_learning,
    learning_ids,
)

__all__ = [
    "RetrievalResult",
    "Retriever",
    "hits_for_learning",
    "learning_ids",
]
