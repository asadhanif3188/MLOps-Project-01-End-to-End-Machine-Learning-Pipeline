"""Model evaluation stage.

Separates its concerns like the train stage:

* **ML computation** — :func:`compute_metrics` loads no files and calls no
  MLflow; it takes a model and a labelled feature matrix and returns metrics, so
  it is unit-testable without a tracking server.
* **Artifact persistence** — metrics are written via
  :func:`pipeline_io.write_json` (the DVC-tracked output this stage owns).
* **MLflow tracking** — delegated to :mod:`tracking`, imported lazily.
* **Orchestration** — :func:`evaluate` / :func:`main` read config, load the
  dataset and model, compute metrics, persist them, then log to MLflow.

Evaluation methodology: the model is scored over the *entire* dataset at
``evaluate.data``. That path is ``data/processed/test.csv`` — the **held-out**
partition carved off by the ``split`` stage and never seen by ``train`` — so the
reported accuracy is a genuine out-of-sample generalization estimate, not an
in-sample figure. The held-out boundary is explicit in the DVC graph (``split``
owns the train/test files; ``train`` depends only on the train file and
``evaluate`` only on the test file) and enforced by the ``contract`` tests. See
pipeline-contract.md §8.
"""

from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score

from exceptions import ModelError
from logging_config import configure_logging, get_logger
from pipeline_io import (
    ensure_columns,
    load_params,
    load_pickle,
    read_csv,
    require_env,
    write_json,
)
from stage_runner import run_stage

logger = get_logger("evaluate")


def compute_metrics(
    model: Any, X: pd.DataFrame, y: pd.Series, source: str
) -> dict[str, float]:
    """Score ``model`` on labelled data ``(X, y)`` — the pure evaluation compute.

    Predicts over every row of ``X`` and returns the accuracy against ``y``.
    Performs no file IO and makes no MLflow calls, so it is unit-testable with a
    plain in-memory model and no tracking server.

    Args:
        model: A fitted estimator exposing ``predict``.
        X: Feature matrix to predict on.
        y: True labels aligned with ``X``.
        source: Human-readable origin of the data, used in the error message.

    Returns:
        A metrics mapping, currently ``{"accuracy": <float>}``.

    Raises:
        ModelError: If the model fails to predict on ``X`` (e.g. its feature
            schema does not match what the model was trained on).
    """
    try:
        predictions = model.predict(X)
    except (ValueError, AttributeError) as exc:
        raise ModelError(
            f"Model failed to predict on {source!r}: {exc}. The model and dataset "
            f"features may be incompatible."
        ) from exc
    return {"accuracy": float(accuracy_score(y, predictions))}


def evaluate(data_path: str, model_path: str, target: str, metrics_path: str) -> None:
    """Orchestrate the evaluate stage: read → compute → persist → track.

    Stage contract:
        * Inputs: the model artifact (``model_path``, the ``train`` output) and
          the held-out evaluation dataset (``data_path``, the ``split`` stage's
          ``test_output``), which must contain ``target`` plus the features the
          model expects. Both are explicit paths — there is no hidden dataset
          loading and no dependency on train's in-memory state — and the dataset
          is the held-out partition ``train`` never fitted on.
        * Output: the metrics artifact at ``metrics_path`` (owned here), written
          as JSON before the MLflow boundary.
        * Configuration: ``target`` and the paths from the ``evaluate`` section
          of ``params.yaml``.

    Args:
        data_path: Path to the evaluation CSV dataset.
        model_path: Path to the pickled model file.
        target: Name of the label column (must match the column ``train`` fit on).
        metrics_path: Path to write the metrics artifact (JSON).

    Raises:
        DataError: If the dataset cannot be read/lacks ``target``, or the metrics
            cannot be written.
        ConfigError: If ``MLFLOW_TRACKING_URI`` is not set.
        ModelError: If the model cannot be loaded or fails to predict.
        TrackingError: If logging to MLflow fails.
    """
    logger.info("Evaluate stage started (data=%s, model=%s)", data_path, model_path)

    data = read_csv(data_path)
    ensure_columns(data, [target], data_path)
    X = data.drop(columns=[target])
    y = data[target]

    tracking_uri = require_env("MLFLOW_TRACKING_URI")

    model = load_pickle(model_path)
    metrics = compute_metrics(model, X, y, data_path)

    # Persist the owned artifact before the network boundary, so the declared
    # metrics output exists independently of MLflow availability.
    write_json(metrics, metrics_path)

    # Cross the tracking boundary last; the lazy import keeps MLflow out of this
    # module's import graph so the computation above stays testable without it.
    from tracking import log_evaluation

    log_evaluation(tracking_uri, metrics)

    logger.info("Evaluate stage completed; model accuracy: %.4f", metrics["accuracy"])


def main() -> None:
    """Entry point: load environment, configure logging, run the stage."""
    load_dotenv()
    configure_logging()

    params = load_params(
        "params.yaml", "evaluate", required=("data", "model", "target", "metrics")
    )

    evaluate(
        params["data"],
        params["model"],
        params["target"],
        params["metrics"],
    )


if __name__ == "__main__":
    run_stage("evaluate", main)
