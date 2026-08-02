"""Tests for the uniform stage entry-point error handling.

:func:`run_stage` is the one place the pipeline catches broadly, on purpose, at
the process boundary. Its contract has three edges worth pinning:

* a clean run returns normally (no exit),
* an *expected* failure (``PipelineError``) and an *unexpected* one (any other
  ``Exception``) both exit non-zero — a failed stage must stop ``dvc repro``, and
* interpreter control-flow signals (``KeyboardInterrupt``, ``SystemExit``) are
  **not** swallowed, so Ctrl-C and explicit exits still work.

These are exactly the properties that would silently rot without a test.
"""

import pytest

from exceptions import DataError
from stage_runner import run_stage


@pytest.mark.unit
def test_clean_run_does_not_exit() -> None:
    """A stage whose ``main`` succeeds returns normally."""
    calls: list[str] = []
    run_stage("demo", lambda: calls.append("ran"))
    assert calls == ["ran"]


@pytest.mark.unit
def test_pipeline_error_exits_nonzero() -> None:
    """An expected failure is caught and turned into a non-zero exit."""

    def main() -> None:
        raise DataError("dataset missing")

    with pytest.raises(SystemExit) as exit_info:
        run_stage("demo", main)
    assert exit_info.value.code == 1


@pytest.mark.unit
def test_unexpected_error_exits_nonzero() -> None:
    """An unexpected bug is also contained at the boundary (exit 1), not leaked."""

    def main() -> None:
        raise ValueError("some unforeseen bug")

    with pytest.raises(SystemExit) as exit_info:
        run_stage("demo", main)
    assert exit_info.value.code == 1


@pytest.mark.unit
def test_pipeline_error_is_logged_with_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The failure is logged once, at ERROR, with the stage name and message —
    the actionable record an operator reads instead of a raw traceback."""

    def main() -> None:
        raise DataError("dataset missing")

    with caplog.at_level("ERROR"), pytest.raises(SystemExit):
        run_stage("train", main)

    assert "train stage failed" in caplog.text
    assert "dataset missing" in caplog.text


@pytest.mark.unit
def test_keyboard_interrupt_is_not_swallowed() -> None:
    """Ctrl-C must propagate untouched, not be caught as a stage failure."""

    def main() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_stage("demo", main)


@pytest.mark.unit
def test_system_exit_is_not_swallowed() -> None:
    """An explicit ``sys.exit`` from within a stage passes through unchanged."""

    def main() -> None:
        raise SystemExit(2)

    with pytest.raises(SystemExit) as exit_info:
        run_stage("demo", main)
    assert exit_info.value.code == 2
