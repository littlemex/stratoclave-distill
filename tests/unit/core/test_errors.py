"""Unit tests for the error hierarchy.

The pipeline's recovery logic catches these classes by base type, so the
inheritance chain has to stay stable.
"""

from __future__ import annotations

import pytest

from stratoclave_distill.core.errors import (
    ConfigError,
    DistillError,
    EmbeddingError,
    IngestError,
    LLMError,
    NotFoundError,
    SchemaError,
)


@pytest.mark.parametrize(
    "subclass",
    [ConfigError, SchemaError, IngestError, LLMError, EmbeddingError, NotFoundError],
)
def test_all_subclasses_descend_from_distill_error(subclass: type) -> None:
    assert issubclass(subclass, DistillError)


def test_distill_error_is_an_exception() -> None:
    assert issubclass(DistillError, Exception)
    with pytest.raises(DistillError):
        raise DistillError("boom")
