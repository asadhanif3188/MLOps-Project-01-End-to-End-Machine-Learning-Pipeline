"""Tests for the custom exception hierarchy.

The value of :mod:`exceptions` is entirely in its *taxonomy*: every
pipeline-specific failure must be catchable as a single :class:`PipelineError`
base, while each subclass stays a distinct type. These tests pin that contract
down so a future refactor cannot silently reparent an exception and break the
"catch ``PipelineError`` at the stage boundary" strategy that ``stage_runner``
relies on.
"""

import pytest

from exceptions import (
    ConfigError,
    DataError,
    ModelError,
    PipelineError,
    TrackingError,
)

SUBCLASSES = [ConfigError, DataError, ModelError, TrackingError]


@pytest.mark.unit
@pytest.mark.parametrize("exc_type", SUBCLASSES)
def test_subclass_of_pipeline_error(exc_type: type[PipelineError]) -> None:
    """Every pipeline exception derives from the common base."""
    assert issubclass(exc_type, PipelineError)


@pytest.mark.unit
def test_pipeline_error_is_an_exception() -> None:
    """The base is a plain ``Exception`` — not something exotic."""
    assert issubclass(PipelineError, Exception)


@pytest.mark.unit
@pytest.mark.parametrize("exc_type", SUBCLASSES)
def test_catchable_as_pipeline_error(exc_type: type[PipelineError]) -> None:
    """A subclass instance is caught by an ``except PipelineError`` — the exact
    guarantee ``stage_runner`` depends on."""
    with pytest.raises(PipelineError):
        raise exc_type("boom")


@pytest.mark.unit
@pytest.mark.parametrize("exc_type", SUBCLASSES)
def test_message_is_preserved(exc_type: type[PipelineError]) -> None:
    """The actionable message survives round-tripping through ``str``."""
    assert str(exc_type("something actionable")) == "something actionable"


@pytest.mark.unit
def test_subclasses_are_distinct_types() -> None:
    """Catching one subclass must not accidentally catch a sibling: a
    ``ModelError`` is *not* a ``ConfigError``, so an ``except ConfigError`` must
    let it propagate rather than swallow it."""
    with pytest.raises(ModelError):
        try:
            raise ModelError("model")
        except ConfigError:  # pragma: no cover - must never match
            pytest.fail("ModelError was wrongly caught as ConfigError")
