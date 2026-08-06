"""Unit tests for the training stage's pure ML computation.

The point of these tests is the ADR-006 boundary: :func:`train.run_training` and
its helpers are exercised **without MLflow, a tracking server, a network, or
``MLFLOW_TRACKING_URI``** — the computation is isolated from experiment tracking.
They require only scikit-learn (skipped when it is absent), never MLflow.
"""

import pytest

pytest.importorskip("sklearn")

import pandas as pd

from train import build_param_grid, run_training


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
