"""Custom exception hierarchy for the ML pipeline.

Every pipeline-specific failure derives from :class:`PipelineError`, giving each
stage a single base class to catch at its entry point. The subclasses mark the
*distinct failure boundaries* the pipeline crosses — configuration, data IO,
model serialization, and experiment tracking — so callers and logs can tell
*what kind* of thing went wrong without parsing message strings.

These classes carry no behavior beyond their type and message: the value is in
the taxonomy. See ``docs/exception-strategy.md`` for the design rationale,
error-propagation rules, and logging conventions that go with this hierarchy.

This module is intentionally dependency-free (standard library only) so it can
be imported from anywhere in the pipeline without creating import cycles.
"""


class PipelineError(Exception):
    """Base class for all expected, pipeline-specific errors.

    Catching :class:`PipelineError` at a stage entry point handles every failure
    the pipeline deliberately raises, while letting genuinely *unexpected*
    exceptions (bugs) surface on a separate path.
    """


class ConfigError(PipelineError):
    """Configuration is missing or invalid.

    Raised for absent or malformed ``params.yaml`` entries and for unset
    required environment variables (e.g. ``MLFLOW_TRACKING_URI``).
    """


class DataError(PipelineError):
    """A dataset cannot be read or does not have the expected shape.

    Raised when an input file is missing, empty, unparseable, cannot be written,
    or is missing a required column.
    """


class ModelError(PipelineError):
    """A model artifact cannot be serialized, deserialized, or used.

    Raised when a pickled model cannot be written or loaded, or when a loaded
    model fails to produce predictions.
    """


class TrackingError(PipelineError):
    """Experiment tracking (MLflow) failed.

    Raised when the tracking backend is unreachable, rejects a logging call, or
    is otherwise misconfigured at the point of a network interaction. The backend
    is the project's in-cluster MLflow Tracking Server (ADR-026); before that it
    was DagsHub-hosted MLflow, removed from the runtime path in Sprint 7.
    """
