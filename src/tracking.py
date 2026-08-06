"""MLflow experiment-tracking boundary.

Every MLflow call in the pipeline — and therefore every experiment-tracking
network interaction — goes through one of these functions, so that:

* the stages' **ML computation is free of MLflow**: :mod:`train` and
  :mod:`evaluate` compute models and metrics without importing MLflow at module
  load, and can be exercised by unit tests with no tracking server, network, or
  credentials (see ADR-006 decision 4); and
* MLflow failures surface as the pipeline's own
  :class:`~exceptions.TrackingError`, with the original exception chained,
  exactly like the filesystem/serialization boundaries in :mod:`pipeline_io`.

This is the one module allowed to depend on MLflow. The stages import it
*lazily*, at the point they cross the tracking boundary, so importing a stage
does not require MLflow to be installed — only running it does.
"""

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature

from exceptions import TrackingError

# Reused for both stages' failure messages: the actionable next steps are the
# same whichever call fails.
_TRACKING_HINT = (
    "Check the tracking URI and your DagsHub credentials / network connection."
)


def build_signature(model_input: Any, model_output: Any) -> Any:
    """Infer an MLflow model signature (input/output schema) from example data.

    A thin, network-free wrapper over :func:`mlflow.models.infer_signature`,
    kept here so the training stage never imports MLflow directly.

    Args:
        model_input: Example feature data (e.g. the training ``X``).
        model_output: Example target data (e.g. the training ``y``).

    Returns:
        The inferred MLflow ``ModelSignature`` (typed ``Any``: MLflow ships no
        type information, so its concrete type is opaque to the checker).
    """
    return infer_signature(model_input, model_output)


def log_evaluation(tracking_uri: str, metrics: Mapping[str, float]) -> None:
    """Log evaluation ``metrics`` to MLflow under a fresh run.

    Args:
        tracking_uri: The MLflow tracking URI (e.g. the DagsHub endpoint).
        metrics: Metric name → value pairs to log.

    Raises:
        TrackingError: If MLflow rejects the connection or a logging call.
    """
    try:
        mlflow.set_tracking_uri(tracking_uri)
        with mlflow.start_run():
            for name, value in metrics.items():
                mlflow.log_metric(name, value)
    except MlflowException as exc:
        raise TrackingError(
            f"MLflow tracking failed against {tracking_uri!r}: {exc}. {_TRACKING_HINT}"
        ) from exc


def log_training_run(
    tracking_uri: str,
    *,
    model: Any,
    signature: Any,
    metrics: Mapping[str, float],
    params: Mapping[str, Any],
    text_artifacts: Mapping[str, str],
    registered_model_name: str,
) -> None:
    """Log a full training run to MLflow: metrics, params, text artifacts, model.

    The model is registered under ``registered_model_name`` when the run's
    artifact store is remote (scheme other than ``file``); against a local file
    store it is logged without registration, since the Model Registry requires a
    real backend. This preserves the exact behavior the training stage had before
    the tracking boundary was extracted.

    Args:
        tracking_uri: The MLflow tracking URI.
        model: The fitted estimator to log.
        signature: The model's MLflow signature (see :func:`build_signature`).
        metrics: Metric name → value pairs to log.
        params: Parameter name → value pairs to log.
        text_artifacts: Filename → text-content pairs to log as run artifacts.
        registered_model_name: Registry name used when the store is remote.

    Raises:
        TrackingError: If MLflow rejects the connection or a logging call.
    """
    try:
        mlflow.set_tracking_uri(tracking_uri)
        with mlflow.start_run():
            for name, value in metrics.items():
                mlflow.log_metric(name, value)
            for name, value in params.items():
                mlflow.log_param(name, value)
            for filename, text in text_artifacts.items():
                mlflow.log_text(text, filename)

            # The Model Registry needs a real backend; a local ``file`` artifact
            # store cannot register, so fall back to a plain model log there.
            if urlparse(mlflow.get_artifact_uri()).scheme != "file":
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name=registered_model_name,
                    signature=signature,
                )
            else:
                mlflow.sklearn.log_model(model, "model", signature=signature)
    except MlflowException as exc:
        raise TrackingError(
            f"MLflow tracking failed against {tracking_uri!r}: {exc}. {_TRACKING_HINT}"
        ) from exc
