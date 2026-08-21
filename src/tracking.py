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
from logging_config import get_logger
from retry import retry_call

logger = get_logger("tracking")

# Reused for both stages' failure messages: the actionable next steps are the
# same whichever call fails. The tracking server is the project's in-cluster
# MLflow platform (ADR-026); the hint stays backend-agnostic so it is accurate
# whether the URI points at the in-cluster Service, a port-forward, or a local
# `mlflow server`.
_TRACKING_HINT = (
    "Check that MLFLOW_TRACKING_URI points at a reachable MLflow tracking server "
    "and that the server is healthy."
)

# Bounded retry policy for a *mid-run* MLflow blip (Sprint 8 PR 13; design of
# record ADR-037; evidence in docs/proof/sprint-08-mlflow-failure-tests-evidence.md
# § 8 candidate #1). A tracking call inside the `train`/`evaluate` stage that hits
# a transient outage — the canonical case being a ~30-60s rolling restart of the
# stateless in-cluster MLflow Deployment — would otherwise fail the whole Job and
# DISCARD the completed preprocess/split/train compute. These attempts ride out
# that blip: 5 attempts with back-off 5s, 10s, 20s, 30s (clamped) ≈ 65s of waiting,
# comfortably longer than a rolling restart yet a tiny fraction of the Job's 1800s
# activeDeadlineSeconds outer stall-guard.
#
# This is DELIBERATELY bounded, not "retry forever": after the 5th attempt the
# underlying MlflowException is re-raised and converted to TrackingError exactly as
# before, so a *persistent* outage still fails the run fast and loud (the fail-fast
# start-of-run `wait-for-mlflow` gate is unchanged). The Job's backoffLimit=2 is a
# coarser, whole-run retry on top of this fine-grained, work-preserving one.
_TRACKING_ATTEMPTS = 5
_TRACKING_BASE_DELAY_SECONDS = 5.0
_TRACKING_MAX_DELAY_SECONDS = 30.0


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


def log_evaluation(
    tracking_uri: str, metrics: Mapping[str, float], *, experiment_name: str
) -> None:
    """Log evaluation ``metrics`` to MLflow under a fresh run.

    Args:
        tracking_uri: The MLflow tracking URI — the in-cluster MLflow Service, a
            local server, or a port-forward (resolved by :mod:`mlflow_config`).
        metrics: Metric name → value pairs to log.
        experiment_name: Experiment to log the run under; created if it does not
            exist (resolved by :func:`mlflow_config.resolve_experiment_name`).

    Raises:
        TrackingError: If MLflow rejects the connection or a logging call and the
            failure persists across the bounded retry policy.
    """

    def _log() -> None:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            for name, value in metrics.items():
                mlflow.log_metric(name, value)

    try:
        retry_call(
            _log,
            attempts=_TRACKING_ATTEMPTS,
            base_delay=_TRACKING_BASE_DELAY_SECONDS,
            max_delay=_TRACKING_MAX_DELAY_SECONDS,
            retry_on=(MlflowException,),
            logger=logger,
            description="MLflow evaluation logging",
        )
    except MlflowException as exc:
        raise TrackingError(
            f"MLflow tracking failed against {tracking_uri!r}: {exc}. {_TRACKING_HINT}"
        ) from exc


def log_training_run(
    tracking_uri: str,
    *,
    experiment_name: str,
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
        experiment_name: Experiment to log the run under; created if it does not
            exist (resolved by :func:`mlflow_config.resolve_experiment_name`).
        model: The fitted estimator to log.
        signature: The model's MLflow signature (see :func:`build_signature`).
        metrics: Metric name → value pairs to log.
        params: Parameter name → value pairs to log.
        text_artifacts: Filename → text-content pairs to log as run artifacts.
        registered_model_name: Registry name used when the store is remote.

    Raises:
        TrackingError: If MLflow rejects the connection or a logging call and the
            failure persists across the bounded retry policy.
    """

    def _log() -> None:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
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

    # Each retry runs `_log` from the top, opening a FRESH MLflow run — a retried
    # attempt never resumes a half-written run. A transient failure can therefore
    # leave at most one incomplete run behind before a later attempt succeeds; that
    # bounded, self-healing duplication is an accepted trade for not discarding the
    # (far more expensive) model training on a momentary tracking blip. See ADR-037.
    try:
        retry_call(
            _log,
            attempts=_TRACKING_ATTEMPTS,
            base_delay=_TRACKING_BASE_DELAY_SECONDS,
            max_delay=_TRACKING_MAX_DELAY_SECONDS,
            retry_on=(MlflowException,),
            logger=logger,
            description="MLflow training-run logging",
        )
    except MlflowException as exc:
        raise TrackingError(
            f"MLflow tracking failed against {tracking_uri!r}: {exc}. {_TRACKING_HINT}"
        ) from exc
