"""Unit tests for :mod:`tracking`'s bounded-retry integration (Sprint 8 PR 13).

:mod:`tracking` is the one module allowed to import MLflow, and MLflow is
*deliberately absent* from the unit environment (ADR-006 dec. 4), so the stages'
tests stub :mod:`tracking` wholesale rather than exercise it. That leaves the
retry wiring this PR adds — the policy passed to :func:`retry.retry_call`, the
``MlflowException`` → :class:`TrackingError` conversion, and the fresh-run-per-
attempt behaviour — asserted only in comments and the ADR.

These tests close that gap **without** adding MLflow to the unit suite: they
install a minimal fake ``mlflow`` into :data:`sys.modules` *before* importing
:mod:`tracking`, so the real production functions run against a controllable
stub. The retry back-off is collapsed to zero delay so the tests are instant.

They pin the three behaviours the reliability contract depends on:

* a *transient* ``MlflowException`` is absorbed and the call ultimately succeeds
  (the improvement), with a **fresh run opened per attempt**;
* a *persistent* ``MlflowException`` is re-raised as :class:`TrackingError` with
  the original exception **chained** (the failure is surfaced, not hidden);
* a non-``MlflowException`` error is **not** retried and is **not** wrapped —
  only the tracking boundary's own error type is translated.
"""

import importlib
import sys
import types
from typing import Any

import pytest

from exceptions import TrackingError

pytestmark = pytest.mark.unit


class _FakeMlflowException(Exception):
    """Stand-in for ``mlflow.exceptions.MlflowException``."""


class _FakeRun:
    """A fake ``mlflow.start_run()`` context manager that records its outcome."""

    def __init__(self, statuses: list[str]):
        self._statuses = statuses

    def __enter__(self) -> "_FakeRun":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        # Mirror MLflow: a run exited via an exception is FAILED, else FINISHED.
        self._statuses.append("FAILED" if exc_type is not None else "FINISHED")
        return False  # never suppress — the failure must propagate to retry_call


class _FakeMlflow:
    """A minimal, controllable stand-in for the ``mlflow`` top-level module.

    ``fail_times`` is how many *attempts* fail before one succeeds: the first
    ``log_metric`` of each attempt raises until the budget is spent, which aborts
    that attempt inside its run context (so each failed attempt marks exactly one
    FAILED run, and a fresh run is opened on the next attempt).
    """

    def __init__(self, *, fail_times: int):
        self._fail_budget = fail_times
        self.run_statuses: list[str] = []
        self.calls = {
            "set_tracking_uri": 0,
            "set_experiment": 0,
            "start_run": 0,
            "log_metric": 0,
            "log_param": 0,
            "log_text": 0,
            "log_model": 0,
        }
        # log_training_run branches on the artifact-store scheme; "file" takes the
        # no-registration path, keeping the stub simple.
        self.sklearn = types.SimpleNamespace(log_model=self._log_model)

    def set_tracking_uri(self, uri: str) -> None:
        self.calls["set_tracking_uri"] += 1

    def set_experiment(self, name: str) -> None:
        self.calls["set_experiment"] += 1

    def start_run(self) -> _FakeRun:
        self.calls["start_run"] += 1
        return _FakeRun(self.run_statuses)

    def log_metric(self, key: str, value: float) -> None:
        self.calls["log_metric"] += 1
        if self._fail_budget > 0:
            self._fail_budget -= 1
            raise _FakeMlflowException("transient: connection refused")

    def log_param(self, key: str, value: Any) -> None:
        self.calls["log_param"] += 1

    def log_text(self, text: str, filename: str) -> None:
        self.calls["log_text"] += 1

    def get_artifact_uri(self) -> str:
        return "file:///tmp/artifacts"

    def _log_model(self, *args: Any, **kwargs: Any) -> None:
        self.calls["log_model"] += 1


def _install_tracking(monkeypatch: pytest.MonkeyPatch, fake: _FakeMlflow) -> Any:
    """Install ``fake`` as the ``mlflow`` package and import :mod:`tracking` fresh.

    Also collapses the retry back-off to zero so the 5-attempt policy runs
    instantly, and forces a re-import so the module binds to *this* fake.
    """
    mlflow_mod = types.ModuleType("mlflow")
    for attr in (
        "set_tracking_uri",
        "set_experiment",
        "start_run",
        "log_metric",
        "log_param",
        "log_text",
        "get_artifact_uri",
    ):
        setattr(mlflow_mod, attr, getattr(fake, attr))
    mlflow_mod.sklearn = fake.sklearn  # type: ignore[attr-defined]

    exceptions_mod = types.ModuleType("mlflow.exceptions")
    exceptions_mod.MlflowException = _FakeMlflowException  # type: ignore[attr-defined]
    models_mod = types.ModuleType("mlflow.models")
    models_mod.infer_signature = lambda *a, **k: "sig"  # type: ignore[attr-defined]
    sklearn_mod = types.ModuleType("mlflow.sklearn")
    sklearn_mod.log_model = fake.sklearn.log_model  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mlflow", mlflow_mod)
    monkeypatch.setitem(sys.modules, "mlflow.exceptions", exceptions_mod)
    monkeypatch.setitem(sys.modules, "mlflow.models", models_mod)
    monkeypatch.setitem(sys.modules, "mlflow.sklearn", sklearn_mod)

    monkeypatch.delitem(sys.modules, "tracking", raising=False)
    tracking = importlib.import_module("tracking")

    # Instant retries: keep the real attempt count, zero the waits.
    monkeypatch.setattr(tracking, "_TRACKING_BASE_DELAY_SECONDS", 0.0)
    monkeypatch.setattr(tracking, "_TRACKING_MAX_DELAY_SECONDS", 0.0)
    return tracking


def _metrics() -> dict[str, float]:
    return {"accuracy": 0.9}


# --------------------------------------------------------------------------- #
# The improvement: a transient blip is absorbed, fresh run per attempt
# --------------------------------------------------------------------------- #
def test_log_evaluation_absorbs_transient_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeMlflow(fail_times=2)  # 2 transient failures, then success
    tracking = _install_tracking(monkeypatch, fake)

    # No exception: the two blips are ridden out within the 5-attempt budget.
    tracking.log_evaluation("http://mlflow:5000", _metrics(), experiment_name="exp")

    assert fake.calls["start_run"] == 3  # a fresh run opened for each attempt
    assert fake.run_statuses == ["FAILED", "FAILED", "FINISHED"]  # no half-write


def test_log_training_run_absorbs_transient_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeMlflow(fail_times=1)
    tracking = _install_tracking(monkeypatch, fake)

    tracking.log_training_run(
        "http://mlflow:5000",
        experiment_name="exp",
        model=object(),
        signature="sig",
        metrics=_metrics(),
        params={"n_estimators": 100},
        text_artifacts={"report.txt": "ok"},
        registered_model_name="model",
    )

    assert fake.calls["start_run"] == 2  # one failed attempt + one success
    assert fake.run_statuses == ["FAILED", "FINISHED"]
    assert fake.calls["log_model"] == 1  # the successful attempt logged the model


# --------------------------------------------------------------------------- #
# Persistent outage: re-raised as TrackingError, original chained, bounded
# --------------------------------------------------------------------------- #
def test_log_evaluation_persistent_failure_becomes_trackingerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeMlflow(fail_times=10**6)  # never recovers
    tracking = _install_tracking(monkeypatch, fake)

    with pytest.raises(TrackingError) as excinfo:
        tracking.log_evaluation("http://mlflow:5000", _metrics(), experiment_name="exp")

    # The persistent failure is SURFACED, not hidden: original exception chained,
    # message preserves the tracking URI and the actionable hint.
    assert isinstance(excinfo.value.__cause__, _FakeMlflowException)
    assert "http://mlflow:5000" in str(excinfo.value)
    assert "reachable MLflow tracking server" in str(excinfo.value)
    # Bounded: exactly the 5-attempt policy, no unbounded loop.
    assert fake.calls["start_run"] == 5


def test_log_training_run_persistent_failure_becomes_trackingerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeMlflow(fail_times=10**6)
    tracking = _install_tracking(monkeypatch, fake)

    with pytest.raises(TrackingError) as excinfo:
        tracking.log_training_run(
            "http://mlflow:5000",
            experiment_name="exp",
            model=object(),
            signature="sig",
            metrics=_metrics(),
            params={},
            text_artifacts={},
            registered_model_name="model",
        )

    assert isinstance(excinfo.value.__cause__, _FakeMlflowException)
    assert fake.calls["start_run"] == 5  # bounded to the attempt budget


# --------------------------------------------------------------------------- #
# Only the tracking boundary's own error type is translated / retried
# --------------------------------------------------------------------------- #
def test_non_mlflow_error_is_not_retried_or_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeMlflow(fail_times=0)
    tracking = _install_tracking(monkeypatch, fake)

    # A non-MlflowException raised inside the boundary must propagate unchanged
    # (not retried, not converted to TrackingError) — it is not a transient
    # tracking fault the policy claims to absorb.
    def _boom(uri: str) -> None:
        raise ValueError("programming error")

    # Patch the attribute the imported module actually calls (``tracking.mlflow``),
    # not the fake object it was built from.
    monkeypatch.setattr(tracking.mlflow, "set_tracking_uri", _boom)

    with pytest.raises(ValueError, match="programming error"):
        tracking.log_evaluation("http://mlflow:5000", _metrics(), experiment_name="exp")

    assert fake.calls["start_run"] == 0  # failed before the run; never retried
