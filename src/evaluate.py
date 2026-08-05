"""Model evaluation stage: loads trained model and logs accuracy metrics."""

import mlflow
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from sklearn.metrics import accuracy_score

from exceptions import ModelError, TrackingError
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


def evaluate(data_path: str, model_path: str, metrics_path: str) -> None:
    """Load trained model and evaluate accuracy on dataset.

    Args:
        data_path: Path to CSV dataset.
        model_path: Path to pickled model file.
        metrics_path: Path to write the metrics artifact (JSON).

    Raises:
        DataError: If the dataset cannot be read or lacks the ``Outcome`` column,
            or if the metrics artifact cannot be written.
        ConfigError: If ``MLFLOW_TRACKING_URI`` is not set.
        ModelError: If the model cannot be loaded or fails to predict.
        TrackingError: If logging the metric to MLflow fails.
    """
    logger.info("Evaluate stage started (data=%s, model=%s)", data_path, model_path)

    data = read_csv(data_path)
    ensure_columns(data, ["Outcome"], data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    tracking_uri = require_env("MLFLOW_TRACKING_URI")

    model = load_pickle(model_path)

    try:
        predictions = model.predict(X)
    except (ValueError, AttributeError) as exc:
        raise ModelError(
            f"Model loaded from {model_path!r} failed to predict on {data_path!r}: "
            f"{exc}. The model and dataset features may be incompatible."
        ) from exc

    model_accuracy_score = accuracy_score(y, predictions)

    # Persist metrics as a first-class, DVC-tracked artifact before the network
    # boundary, so the pipeline's declared ``metrics`` output exists independently
    # of MLflow availability.
    write_json({"accuracy": float(model_accuracy_score)}, metrics_path)

    try:
        mlflow.set_tracking_uri(tracking_uri)
        with mlflow.start_run():
            mlflow.log_metric("accuracy", model_accuracy_score)
    except MlflowException as exc:
        raise TrackingError(
            f"MLflow tracking failed against {tracking_uri!r}: {exc}. Check the "
            f"tracking URI and your DagsHub credentials / network connection."
        ) from exc

    logger.info("Evaluate stage completed; model accuracy: %.4f", model_accuracy_score)


def main() -> None:
    """Entry point: load environment, configure logging, run the stage."""
    load_dotenv()
    configure_logging()

    params = load_params("params.yaml", "test", required=("data", "model", "metrics"))

    evaluate(
        params["data"],
        params["model"],
        params["metrics"],
    )


if __name__ == "__main__":
    run_stage("evaluate", main)
