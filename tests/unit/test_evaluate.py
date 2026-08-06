"""Unit tests for the evaluation stage's pure ML computation.

Like the train tests, these exercise :func:`evaluate.compute_metrics` **without
MLflow, a tracking server, a network, or ``MLFLOW_TRACKING_URI``**. The model is
a tiny in-memory stub, so the test needs no trained artifact on disk either —
only scikit-learn (for ``accuracy_score``), never MLflow.
"""

import pytest

pytest.importorskip("sklearn")

import pandas as pd

from evaluate import compute_metrics
from exceptions import ModelError


class _StubModel:
    """A stand-in estimator that returns pre-set predictions from ``predict``."""

    def __init__(self, predictions: list[int]) -> None:
        self._predictions = predictions

    def predict(self, X: pd.DataFrame) -> list[int]:
        return self._predictions


class _BadModel:
    """A model whose ``predict`` fails the way a schema mismatch would."""

    def predict(self, X: pd.DataFrame) -> list[int]:
        raise ValueError("feature names mismatch")


def _features() -> pd.DataFrame:
    return pd.DataFrame({"FeatureA": [1, 2, 3, 4], "FeatureB": [5, 6, 7, 8]})


@pytest.mark.unit
def test_compute_metrics_scores_perfect_prediction() -> None:
    """A model that reproduces the labels exactly scores accuracy 1.0."""
    X = _features()
    y = pd.Series([0, 1, 0, 1], name="Outcome")

    metrics = compute_metrics(_StubModel([0, 1, 0, 1]), X, y, "dataset")

    assert metrics == {"accuracy": 1.0}


@pytest.mark.unit
def test_compute_metrics_scores_partial_prediction() -> None:
    """Accuracy reflects the fraction of correct predictions (2 of 4 => 0.5)."""
    X = _features()
    y = pd.Series([0, 1, 0, 1], name="Outcome")

    metrics = compute_metrics(_StubModel([0, 1, 1, 0]), X, y, "dataset")

    assert metrics["accuracy"] == pytest.approx(0.5)


@pytest.mark.unit
def test_compute_metrics_predict_failure_raises_model_error() -> None:
    """A failed prediction surfaces as a typed ``ModelError`` naming the source,
    not a raw ``ValueError``."""
    X = _features()
    y = pd.Series([0, 1, 0, 1], name="Outcome")

    with pytest.raises(ModelError, match="dataset"):
        compute_metrics(_BadModel(), X, y, "dataset")


@pytest.mark.unit
def test_compute_metrics_needs_no_tracking_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit isolation check: scoring runs with ``MLFLOW_TRACKING_URI`` unset;
    the computation never touches the tracking boundary."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    X = _features()
    y = pd.Series([0, 1, 0, 1], name="Outcome")

    metrics = compute_metrics(_StubModel([0, 1, 0, 1]), X, y, "dataset")

    assert metrics["accuracy"] == 1.0
