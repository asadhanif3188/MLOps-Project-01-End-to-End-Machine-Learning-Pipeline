"""Unit tests for the training stage.

Two tiers live here, both **without MLflow, a tracking server, a network, or a
real DagsHub endpoint** (ADR-006 boundary):

* *pure computation* — :func:`train.run_training` and its helpers, exercised on
  in-memory data; and
* *the stage orchestrator* — :func:`train.train`, exercised through real files
  with the MLflow boundary swapped for the ``stub_tracking`` recorder, so the
  read → compute → **persist** path is validated (the model artifact is actually
  produced and loadable) without ever importing MLflow.

Both tiers require only scikit-learn (skipped when it is absent), never MLflow.
"""

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sklearn")

import pandas as pd

from exceptions import ConfigError, DataError
from pipeline_io import load_pickle
from train import build_param_grid, run_training, train


def _labelled_data() -> tuple[pd.DataFrame, pd.Series]:
    """A small, balanced two-class dataset large enough for 3-fold CV.

    30 rows, 15 per class, deterministically constructed (no randomness), so the
    only source of non-determinism a test could observe is the model itself.
    """
    n = 30
    features = pd.DataFrame(
        {
            "FeatureA": list(range(n)),
            "FeatureB": [x % 5 for x in range(n)],
        }
    )
    target = pd.Series([0, 1] * (n // 2), name="Outcome")
    return features, target


@pytest.mark.unit
def test_build_param_grid_tunes_only_regularization() -> None:
    """The configured hyperparameters (n_estimators, max_depth, random_state)
    govern the estimator directly, so the search grid tunes only the leaf/split
    regularization — not those configured values."""
    grid = build_param_grid()
    assert set(grid) == {"min_samples_split", "min_samples_leaf"}
    assert "n_estimators" not in grid
    assert "max_depth" not in grid


@pytest.mark.unit
def test_run_training_is_deterministic_given_seed() -> None:
    """Same data + same random_state => identical model selection and accuracy.

    This is the reproducibility guarantee: both the split and the estimator are
    seeded, so nothing downstream drifts between runs."""
    X, y = _labelled_data()

    first = run_training(X, y, random_state=42, n_estimators=10, max_depth=3)
    second = run_training(X, y, random_state=42, n_estimators=10, max_depth=3)

    assert first.accuracy == second.accuracy
    assert first.best_params == second.best_params


@pytest.mark.unit
def test_run_training_applies_configured_hyperparameters() -> None:
    """The configured n_estimators/max_depth are the ones that end up on the
    fitted model (they are no longer inert)."""
    X, y = _labelled_data()

    result = run_training(X, y, random_state=0, n_estimators=7, max_depth=4)

    assert result.best_params["n_estimators"] == 7
    assert result.best_params["max_depth"] == 4
    assert result.model.n_estimators == 7
    assert result.model.max_depth == 4


@pytest.mark.unit
def test_run_training_returns_usable_model_and_valid_accuracy() -> None:
    """The returned model predicts on real features and the reported accuracy is
    a probability — evidence the artifact is fit and loadable/usable."""
    X, y = _labelled_data()

    result = run_training(X, y, random_state=42, n_estimators=10, max_depth=3)

    predictions = result.model.predict(X)
    assert len(predictions) == len(X)
    assert 0.0 <= result.accuracy <= 1.0


@pytest.mark.unit
def test_run_training_needs_no_tracking_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit isolation check: with ``MLFLOW_TRACKING_URI`` unset, the pure
    computation still runs — it never touches the tracking boundary."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    X, y = _labelled_data()

    result = run_training(X, y, random_state=42, n_estimators=10, max_depth=3)

    assert result.accuracy == pytest.approx(result.accuracy)  # completed, no raise


# --------------------------------------------------------------------------- #
# train — the stage orchestrator (read -> compute -> persist -> track)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_train_produces_loadable_model_artifact(
    tmp_path: Path,
    training_csv: Path,
    stub_tracking: dict[str, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stage's core contract: valid processed data yields a model artifact
    that exists on disk, unpickles, and exposes a working ``predict`` — all with
    the tracking boundary stubbed (no MLflow, no network)."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://stub")
    model_path = tmp_path / "models" / "model.pkl"  # nested dir must be created

    train(
        str(training_csv),
        str(model_path),
        "Outcome",
        random_state=42,
        n_estimators=10,
        max_depth=3,
    )

    assert model_path.exists()
    model = load_pickle(str(model_path))
    unseen = pd.DataFrame({"Glucose": [1, 2], "BloodPressure": [3, 4]})
    predictions = model.predict(unseen)
    assert len(predictions) == 2


@pytest.mark.unit
def test_train_logs_run_across_the_stubbed_boundary(
    tmp_path: Path,
    training_csv: Path,
    stub_tracking: dict[str, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The computed result is handed to the tracking layer: exactly one run is
    logged, carrying the accuracy metric and the *configured* hyperparameters —
    evidence the config genuinely reaches MLflow (not inert)."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://stub")
    model_path = tmp_path / "model.pkl"

    train(
        str(training_csv),
        str(model_path),
        "Outcome",
        random_state=42,
        n_estimators=7,
        max_depth=4,
    )

    assert len(stub_tracking["training"]) == 1
    run = stub_tracking["training"][0]
    assert run["tracking_uri"] == "http://stub"
    assert run["experiment_name"] == "mlops-pipeline"
    assert run["metrics"]["accuracy"] == pytest.approx(run["metrics"]["accuracy"])
    assert run["params"]["best_n_estimators"] == 7
    assert run["params"]["best_max_depth"] == 4


@pytest.mark.unit
def test_train_fails_fast_when_tracking_uri_unset(
    tmp_path: Path,
    training_csv: Path,
    stub_tracking: dict[str, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing ``MLFLOW_TRACKING_URI`` is caught *before* the expensive fit, so
    the stage raises ``ConfigError`` and writes no partial model artifact."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    model_path = tmp_path / "model.pkl"

    with pytest.raises(ConfigError, match="not set"):
        train(
            str(training_csv),
            str(model_path),
            "Outcome",
            random_state=42,
            n_estimators=10,
            max_depth=3,
        )
    assert not model_path.exists()


@pytest.mark.unit
def test_train_missing_target_column_raises_data_error(
    tmp_path: Path,
    stub_tracking: dict[str, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Processed data lacking the configured target column fails predictably as a
    ``DataError`` naming the column, and produces no model."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://stub")
    data = tmp_path / "no_target.csv"
    pd.DataFrame({"Glucose": [1, 2, 3], "BloodPressure": [4, 5, 6]}).to_csv(
        data, index=False
    )
    model_path = tmp_path / "model.pkl"

    with pytest.raises(DataError, match="Outcome"):
        train(
            str(data),
            str(model_path),
            "Outcome",
            random_state=42,
            n_estimators=10,
            max_depth=3,
        )
    assert not model_path.exists()
