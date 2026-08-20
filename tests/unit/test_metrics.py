"""Unit tests for :mod:`pipeline_metrics` and its wiring into the stages.

These pin the operational-metrics contract (Sprint 8, PR 3 — ADR-030) with an
INJECTED fake gateway (no ``prometheus_client``, no network, no Pushgateway):

* emission is DISABLED unless ``PUSHGATEWAY_URL`` is set (local runs / CI / tests
  do no network I/O),
* a successful stage emits ``success=1`` with a real duration, a failing stage
  emits ``success=0`` and still fails the run,
* label cardinality is bounded (unknown stage names are refused),
* the batch lifecycle reset clears every stage group, and
* every push/delete is best-effort — a gateway error never fails the pipeline.

The gateway I/O itself is injected, so these need nothing installed and never
touch a socket — the same property the pipeline contract requires of unit tests.
"""

import time

import pytest

import fetch_dataset
import pipeline_metrics
import stage_runner
from exceptions import DataError
from pipeline_metrics import (
    PIPELINE_STAGES,
    push_stage_metrics,
    reset_pipeline_metrics,
    resolve_gateway_url,
    time_stage,
)

pytestmark = pytest.mark.unit

_URL = "http://pushgateway.monitoring:9091"


# --------------------------------------------------------------------------- #
# resolve_gateway_url
# --------------------------------------------------------------------------- #
def test_resolve_prefers_explicit_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(pipeline_metrics.PUSHGATEWAY_URL_ENV, "http://from-env:9091")
    assert resolve_gateway_url("http://explicit:9091") == "http://explicit:9091"


def test_resolve_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(pipeline_metrics.PUSHGATEWAY_URL_ENV, "  http://from-env:9091 ")
    assert resolve_gateway_url() == "http://from-env:9091"


def test_resolve_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(pipeline_metrics.PUSHGATEWAY_URL_ENV, raising=False)
    assert resolve_gateway_url() is None


def test_resolve_none_when_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(pipeline_metrics.PUSHGATEWAY_URL_ENV, "   ")
    assert resolve_gateway_url() is None


# --------------------------------------------------------------------------- #
# push_stage_metrics
# --------------------------------------------------------------------------- #
def test_push_disabled_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no gateway configured, nothing is pushed (local/CI/test default)."""
    monkeypatch.delenv(pipeline_metrics.PUSHGATEWAY_URL_ENV, raising=False)
    calls: list[tuple] = []
    push_stage_metrics("train", 1.5, success=True, push=lambda *a: calls.append(a))
    assert calls == []


def test_push_enabled_calls_pusher_with_values() -> None:
    calls: list[tuple] = []
    push_stage_metrics(
        "train", 2.5, success=True, url=_URL, push=lambda *a: calls.append(a)
    )
    assert calls == [(_URL, "train", 2.5, True)]


def test_push_failure_records_zero() -> None:
    calls: list[tuple] = []
    push_stage_metrics(
        "evaluate", 0.25, success=False, url=_URL, push=lambda *a: calls.append(a)
    )
    assert calls == [(_URL, "evaluate", 0.25, False)]


def test_push_unknown_stage_refused() -> None:
    """An out-of-set stage name is refused to keep label cardinality bounded."""
    calls: list[tuple] = []
    push_stage_metrics(
        "not-a-stage", 1.0, success=True, url=_URL, push=lambda *a: calls.append(a)
    )
    assert calls == []


def test_push_swallows_gateway_error() -> None:
    """A gateway/network error is best-effort: swallowed, not raised."""

    def boom(*_args: object) -> None:
        raise OSError("connection refused")

    # Must not raise.
    push_stage_metrics("train", 1.0, success=True, url=_URL, push=boom)


def test_push_swallows_missing_client() -> None:
    """A missing prometheus_client (ImportError) is also swallowed."""

    def boom(*_args: object) -> None:
        raise ImportError("No module named 'prometheus_client'")

    push_stage_metrics("train", 1.0, success=True, url=_URL, push=boom)


# --------------------------------------------------------------------------- #
# reset_pipeline_metrics (batch lifecycle)
# --------------------------------------------------------------------------- #
def test_reset_disabled_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(pipeline_metrics.PUSHGATEWAY_URL_ENV, raising=False)
    deleted: list[str] = []
    reset_pipeline_metrics(delete=lambda _url, stage: deleted.append(stage))
    assert deleted == []


def test_reset_clears_every_stage_group() -> None:
    deleted: list[str] = []
    reset_pipeline_metrics(url=_URL, delete=lambda _url, stage: deleted.append(stage))
    assert deleted == list(PIPELINE_STAGES)


def test_reset_is_best_effort_across_stages() -> None:
    """One group failing to clear must not stop the others or raise."""
    deleted: list[str] = []

    def flaky_delete(_url: str, stage: str) -> None:
        if stage == "split":
            raise OSError("gone")
        deleted.append(stage)

    reset_pipeline_metrics(url=_URL, delete=flaky_delete)
    # Every stage except the one that raised was still attempted.
    assert deleted == [s for s in PIPELINE_STAGES if s != "split"]


# --------------------------------------------------------------------------- #
# time_stage (the shared timer/emitter)
# --------------------------------------------------------------------------- #
def test_time_stage_success_emits_positive_duration() -> None:
    calls: list[tuple] = []
    with time_stage("train", url=_URL, push=lambda *a: calls.append(a)):
        time.sleep(0.001)
    assert len(calls) == 1
    url, stage, duration, success = calls[0]
    assert (url, stage, success) == (_URL, "train", True)
    assert duration > 0


def test_time_stage_failure_emits_zero_and_reraises() -> None:
    calls: list[tuple] = []
    with (
        pytest.raises(ValueError, match="boom"),
        time_stage("split", url=_URL, push=lambda *a: calls.append(a)),
    ):
        raise ValueError("boom")
    assert len(calls) == 1
    url, stage, _duration, success = calls[0]
    assert (url, stage, success) == (_URL, "split", False)


def test_time_stage_keyboardinterrupt_does_not_emit() -> None:
    """Interpreter control-flow signals are not a stage failure — no emission."""
    calls: list[tuple] = []
    with (
        pytest.raises(KeyboardInterrupt),
        time_stage("train", url=_URL, push=lambda *a: calls.append(a)),
    ):
        raise KeyboardInterrupt
    assert calls == []


def test_time_stage_systemexit_does_not_emit() -> None:
    calls: list[tuple] = []
    with (
        pytest.raises(SystemExit),
        time_stage("train", url=_URL, push=lambda *a: calls.append(a)),
    ):
        raise SystemExit(2)
    assert calls == []


# --------------------------------------------------------------------------- #
# Wiring: stage_runner.run_stage emits per-stage metrics
# --------------------------------------------------------------------------- #
def _record_pushes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, bool]]:
    """Capture (stage, success) for every push_stage_metrics call via time_stage."""
    recorded: list[tuple[str, bool]] = []

    def recorder(stage: str, _duration: float, *, success: bool, **_kw: object) -> None:
        recorded.append((stage, success))

    monkeypatch.setattr(pipeline_metrics, "push_stage_metrics", recorder)
    return recorded


def test_run_stage_success_emits_success(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = _record_pushes(monkeypatch)
    stage_runner.run_stage("train", lambda: None)
    assert recorded == [("train", True)]


def test_run_stage_pipeline_error_emits_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = _record_pushes(monkeypatch)

    def main() -> None:
        raise DataError("dataset missing")

    with pytest.raises(SystemExit):
        stage_runner.run_stage("split", main)
    assert recorded == [("split", False)]


def test_run_stage_unexpected_error_emits_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = _record_pushes(monkeypatch)

    def main() -> None:
        raise ValueError("bug")

    with pytest.raises(SystemExit):
        stage_runner.run_stage("evaluate", main)
    assert recorded == [("evaluate", False)]


# --------------------------------------------------------------------------- #
# Wiring: fetch_dataset.main resets the gateway then emits its own stage
# --------------------------------------------------------------------------- #
def test_fetch_dataset_main_resets_and_emits_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The init container clears prior-run metrics up front, then records itself.

    Exercised on the failure path (no DATASET_S3_URI): the reset must still run
    (it is the once-per-run cleanup), and the fetch_dataset stage must be recorded
    as failed before the non-zero exit.
    """
    reset_calls: list[bool] = []
    recorded = _record_pushes(monkeypatch)
    monkeypatch.setattr(
        fetch_dataset,
        "reset_pipeline_metrics",
        lambda **_kw: reset_calls.append(True),
    )
    monkeypatch.delenv("DATASET_S3_URI", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        fetch_dataset.main()

    assert excinfo.value.code == 1
    assert reset_calls == [True]
    assert recorded == [("fetch_dataset", False)]
