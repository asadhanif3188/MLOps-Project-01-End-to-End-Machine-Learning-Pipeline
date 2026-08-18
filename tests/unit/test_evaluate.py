"""Unit tests for the evaluation stage.

Two tiers, both **without MLflow, a tracking server, a network, or a real
DagsHub endpoint** (ADR-006 boundary):

* *pure computation* — :func:`evaluate.compute_metrics`, exercised against tiny
  in-memory model stubs, so it needs no artifact on disk; and
* *the stage orchestrator* — :func:`evaluate.evaluate`, exercised through real
  files (a pickled model + a labelled CSV) with the MLflow boundary swapped for
  the ``stub_tracking`` recorder, so the read → score → **persist metrics** path
  is validated and the metrics artifact is actually produced and readable.

Both tiers require only scikit-learn (skipped when it is absent), never MLflow.
"""

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sklearn")

import pandas as pd

from evaluate import compute_metrics, evaluate
from exceptions import ConfigError, DataError, ModelError
from pipeline_io import save_pickle


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


# --------------------------------------------------------------------------- #
# evaluate — the stage orchestrator (read -> load -> score -> persist -> track)
# --------------------------------------------------------------------------- #


@pytest.fixture
def trained_model_path(tmp_path: Path, training_frame: pd.DataFrame) -> Path:
    """A real fitted ``RandomForestClassifier`` pickled to disk.

    Fit on the same feature schema (``Glucose``/``BloodPressure``) the evaluate
    stage will score against, so the stage runs against a genuine, loadable model
    artifact rather than a stub.
    """
    from sklearn.ensemble import RandomForestClassifier

    features = training_frame.drop(columns=["Outcome"])
    labels = training_frame["Outcome"]
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(features, labels)

    path = tmp_path / "model.pkl"
    save_pickle(model, str(path))
    return path


@pytest.mark.unit
def test_evaluate_produces_readable_metrics_artifact(
    tmp_path: Path,
    training_csv: Path,
    trained_model_path: Path,
    stub_tracking: dict[str, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stage's core contract: a valid model + dataset yields a metrics file
    that exists, parses as JSON, has the expected ``accuracy`` key, and holds a
    valid probability — with the tracking boundary stubbed (no MLflow)."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://stub")
    metrics_path = tmp_path / "metrics" / "metrics.json"  # nested dir must be created

    evaluate(str(training_csv), str(trained_model_path), "Outcome", str(metrics_path))

    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert set(payload) == {"accuracy"}
    assert isinstance(payload["accuracy"], float)
    assert 0.0 <= payload["accuracy"] <= 1.0
    # The persisted metrics were also handed across the (stubbed) boundary, under
    # the configured (default) experiment.
    assert stub_tracking["evaluation"] == [
        {
            "tracking_uri": "http://stub",
            "metrics": payload,
            "experiment_name": "mlops-pipeline",
        }
    ]


@pytest.mark.unit
def test_evaluate_fails_fast_when_tracking_uri_unset(
    tmp_path: Path,
    training_csv: Path,
    trained_model_path: Path,
    stub_tracking: dict[str, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing ``MLFLOW_TRACKING_URI`` raises ``ConfigError`` before the model is
    loaded/scored, so no partial metrics artifact is written."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    metrics_path = tmp_path / "metrics.json"

    with pytest.raises(ConfigError, match="not set"):
        evaluate(
            str(training_csv), str(trained_model_path), "Outcome", str(metrics_path)
        )
    assert not metrics_path.exists()


@pytest.mark.unit
def test_evaluate_missing_model_raises_model_error(
    tmp_path: Path,
    training_csv: Path,
    stub_tracking: dict[str, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An absent model artifact fails predictably as a ``ModelError`` (the stage
    depends on train having run first)."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://stub")
    metrics_path = tmp_path / "metrics.json"

    with pytest.raises(ModelError, match="not found"):
        evaluate(
            str(training_csv),
            str(tmp_path / "absent.pkl"),
            "Outcome",
            str(metrics_path),
        )
    assert not metrics_path.exists()


@pytest.mark.unit
def test_evaluate_missing_target_column_raises_data_error(
    tmp_path: Path,
    trained_model_path: Path,
    stub_tracking: dict[str, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evaluation data lacking the configured target column fails as a
    ``DataError`` naming the column, and writes no metrics."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://stub")
    data = tmp_path / "no_target.csv"
    pd.DataFrame({"Glucose": [1, 2, 3], "BloodPressure": [4, 5, 6]}).to_csv(
        data, index=False
    )
    metrics_path = tmp_path / "metrics.json"

    with pytest.raises(DataError, match="Outcome"):
        evaluate(str(data), str(trained_model_path), "Outcome", str(metrics_path))
    assert not metrics_path.exists()
