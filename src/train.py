"""Model training stage.

Structured as four separable concerns, wired together only in :func:`train` /
:func:`main`:

* **ML computation** — :func:`run_training` (with :func:`build_param_grid` and
  :func:`hyperparameter_tuning`) splits, tunes, fits, and scores. It takes a
  feature matrix and target and returns a :class:`TrainingResult`; it performs
  no file IO, reads no environment, and imports no MLflow, so it is
  deterministic given its inputs and unit-testable without a tracking server.
* **Artifact persistence** — the fitted model is pickled via
  :func:`pipeline_io.save_pickle` (the DVC-tracked output this stage owns).
* **MLflow tracking** — delegated to :mod:`tracking`, imported lazily at the
  boundary so importing this module does not require MLflow.
* **Orchestration** — :func:`train` / :func:`main` read config and data, invoke
  the computation, persist the model, then log the run.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

from logging_config import configure_logging, get_logger
from mlflow_config import resolve_experiment_name, resolve_tracking_uri
from pipeline_io import ensure_columns, load_params, read_csv, save_pickle
from stage_runner import run_stage

logger = get_logger("train")

# Held-out fraction for the in-training accuracy estimate. Fixed here (not
# configured) because it governs an internal validation split *within the
# training set* for tuning/reporting — it is not the pipeline's held-out
# evaluation dataset, which the ``split`` stage carves off before training and
# ``evaluate`` consumes. Training therefore never sees the held-out rows.
TEST_SIZE = 0.20

# Registry name used when MLflow's artifact store is remote (see tracking).
REGISTERED_MODEL_NAME = "Best Random Forest Classifier"


@dataclass(frozen=True)
class TrainingResult:
    """The outputs of the MLflow-free training computation.

    Attributes:
        model: The fitted best estimator (ready to pickle and to predict with).
        accuracy: Accuracy on the internal held-out split.
        best_params: The hyperparameters that produced ``model`` — the configured
            ``n_estimators``/``max_depth`` plus the tuned
            ``min_samples_split``/``min_samples_leaf``.
        confusion_matrix: Text rendering of the confusion matrix.
        classification_report: Text rendering of the classification report.
    """

    model: RandomForestClassifier
    accuracy: float
    best_params: dict[str, Any]
    confusion_matrix: str
    classification_report: str


def build_param_grid() -> dict[str, list[int]]:
    """Return the grid searched by :func:`hyperparameter_tuning`.

    Only the leaf/split regularization is tuned here. ``n_estimators``,
    ``max_depth``, and ``random_state`` come from ``params.yaml`` and are set
    directly on the base estimator, so the configured values genuinely govern the
    model instead of being shadowed by a hardcoded grid (resolving the previously
    inert ``train.*`` hyperparameters).
    """
    return {
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }


def hyperparameter_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    base_estimator: RandomForestClassifier,
    param_grid: dict[str, list[int]],
) -> GridSearchCV:
    """Grid-search ``param_grid`` around ``base_estimator`` with 3-fold CV.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        base_estimator: The seeded estimator whose fixed hyperparameters
            (``n_estimators``, ``max_depth``, ``random_state``) come from config.
        param_grid: The parameters to tune (see :func:`build_param_grid`).

    Returns:
        The fitted :class:`GridSearchCV`.
    """
    grid_search = GridSearchCV(
        estimator=base_estimator, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2
    )
    logger.info("Hyperparameter tuning started")
    grid_search.fit(X_train, y_train)
    logger.info(
        "Hyperparameter tuning completed; best params: %s", grid_search.best_params_
    )
    return grid_search


def run_training(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int,
    n_estimators: int,
    max_depth: int | None,
    test_size: float = TEST_SIZE,
) -> TrainingResult:
    """Split, tune, fit, and score — the stage's pure ML computation.

    Deterministic given ``X``, ``y``, and ``random_state``: both the train/test
    split and the Random Forest are seeded, so repeated calls yield the same
    model and accuracy. Performs no IO and makes no MLflow calls.

    Args:
        X: Feature matrix.
        y: Target vector.
        random_state: Seed applied to both the split and the estimator.
        n_estimators: Number of trees (set on the estimator).
        max_depth: Maximum tree depth, or ``None`` for unbounded.
        test_size: Held-out fraction for the internal accuracy estimate.

    Returns:
        A :class:`TrainingResult` with the fitted model and its metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    base_estimator = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
    )
    grid_search = hyperparameter_tuning(
        X_train, y_train, base_estimator, build_param_grid()
    )
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))

    best_params: dict[str, Any] = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": grid_search.best_params_["min_samples_split"],
        "min_samples_leaf": grid_search.best_params_["min_samples_leaf"],
    }
    return TrainingResult(
        model=best_model,
        accuracy=accuracy,
        best_params=best_params,
        confusion_matrix=str(confusion_matrix(y_test, y_pred)),
        classification_report=str(classification_report(y_test, y_pred)),
    )


def train(
    data_path: str,
    model_path: str,
    target: str,
    random_state: int,
    n_estimators: int,
    max_depth: int | None,
) -> None:
    """Orchestrate the train stage: read → compute → persist → track.

    Stage contract:
        * Input:  the training dataset (``data_path``, the ``split`` stage's
          ``train_output``), containing ``target`` plus feature columns. This is
          the training half of the held-out split, never the evaluation half, so
          the model is not fitted on the rows it will be scored against.
        * Output: the pickled model artifact at ``model_path`` (owned here).
        * Configuration: ``target`` and the training hyperparameters from the
          ``train`` section of ``params.yaml``.

    Args:
        data_path: Path to the training CSV dataset (the ``split`` train output).
        model_path: Path to save the pickled model.
        target: Name of the label column; every other column is a feature.
        random_state: Seed for the split and the estimator (reproducibility).
        n_estimators: Number of trees.
        max_depth: Maximum tree depth, or ``None`` for unbounded.

    Raises:
        DataError: If the dataset cannot be read/lacks ``target``, or the model
            cannot be written.
        ConfigError: If ``MLFLOW_TRACKING_URI`` is unset or names a local file
            store without ``MLFLOW_ALLOW_FILE_STORE`` (see :mod:`mlflow_config`).
        TrackingError: If MLflow tracking fails.
        ModelError: If the trained model cannot be serialized.
    """
    logger.info("Train stage started (data=%s, model=%s)", data_path, model_path)

    data = read_csv(data_path)
    ensure_columns(data, [target], data_path)
    X = data.drop(columns=[target])
    y = data[target]

    # Validate the tracking config up front — fail fast before the expensive fit.
    tracking_uri = resolve_tracking_uri()
    experiment_name = resolve_experiment_name()

    result = run_training(
        X,
        y,
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    logger.info("Best model accuracy: %.4f", result.accuracy)

    # Persist the owned artifact before the network boundary, so the DVC output
    # exists independently of MLflow availability.
    save_pickle(result.model, model_path)
    logger.info("Model saved to %s", model_path)

    # Cross the tracking boundary last; the lazy import keeps MLflow out of this
    # module's import graph so the computation above stays testable without it.
    from tracking import build_signature, log_training_run

    log_training_run(
        tracking_uri,
        experiment_name=experiment_name,
        model=result.model,
        signature=build_signature(X, y),
        metrics={"accuracy": result.accuracy},
        params={
            "best_n_estimators": result.best_params["n_estimators"],
            "best_max_depth": result.best_params["max_depth"],
            "best_samples_split": result.best_params["min_samples_split"],
            "best_samples_leaf": result.best_params["min_samples_leaf"],
        },
        text_artifacts={
            "confusion_matrix.txt": result.confusion_matrix,
            "classification_report.txt": result.classification_report,
        },
        registered_model_name=REGISTERED_MODEL_NAME,
    )

    logger.info("Train stage completed")


def main() -> None:
    """Entry point: load environment, configure logging, run the stage."""
    load_dotenv()
    configure_logging()

    params = load_params(
        "params.yaml",
        "train",
        required=(
            "input",
            "output",
            "target",
            "random_state",
            "n_estimators",
            "max_depth",
        ),
    )

    train(
        params["input"],
        params["output"],
        params["target"],
        params["random_state"],
        params["n_estimators"],
        params["max_depth"],
    )


if __name__ == "__main__":
    run_stage("train", main)
