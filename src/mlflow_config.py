"""Resolution and validation of the pipeline's MLflow tracking configuration.

This is the pipeline's *configuration model* for experiment tracking. It is kept
separate from :mod:`tracking` (which imports MLflow) on purpose: the stages
resolve and validate the tracking target **before** the lazy MLflow import, so a
misconfiguration fails fast — with a typed :class:`~exceptions.ConfigError` and
an actionable message — without paying MLflow's import cost, and the resolution
logic stays unit-testable with no tracking server, network, or credentials.

The model is deliberately minimal, reflecting ADR-026: the pipeline logs to the
project's own **in-cluster MLflow Tracking Server** (PostgreSQL + S3), which runs
internal-only with no client authentication. So the *only* required input is the
tracking URI; there is no username, password, or token in the runtime path (the
former DagsHub SaaS and its credential Secret are gone). Where the URI comes from
is environment-specific and never hardcoded in Python:

* **in-cluster** — the Kustomize ``ConfigMap`` (``k8s/base/configmap.yaml``)
  injects ``MLFLOW_TRACKING_URI`` pointing at the MLflow ``Service`` DNS name;
* **local dev** — ``.env`` / the shell (see ``.env.example``), typically a
  ``kubectl port-forward`` to the same server, a local ``mlflow server``, or an
  explicitly opted-in local file store for offline work.

The one piece of real validation beyond "is it set" is the **file-store guard**:
a ``file:`` (or scheme-less path) tracking URI records runs to the local
filesystem. In a cluster that filesystem is the pod's ephemeral storage, so every
run, metric, and artifact is silently lost when the pod exits — the "transient
offline file store" failure mode the project called out as a limitation. Rather
than let that happen implicitly, a file store is rejected unless the developer
opts in via ``MLFLOW_ALLOW_FILE_STORE``, turning a silent footgun into an
explicit, offline-only choice while preserving local usability.
"""

import os
from urllib.parse import urlparse

from exceptions import ConfigError
from pipeline_io import require_env

# Environment variables that make up the tracking configuration model. The names
# match exactly what the K8s ConfigMap injects and what ``.env.example`` documents.
TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
ALLOW_FILE_STORE_ENV = "MLFLOW_ALLOW_FILE_STORE"
EXPERIMENT_NAME_ENV = "MLFLOW_EXPERIMENT_NAME"

# Experiment the stages log under when ``MLFLOW_EXPERIMENT_NAME`` is not set.
# Naming the experiment (rather than defaulting to MLflow's catch-all "Default")
# groups this project's train/evaluate runs together on the shared server.
DEFAULT_EXPERIMENT_NAME = "mlops-pipeline"

# URI schemes that MLflow treats as a *local* file store. A scheme-less value
# (e.g. ``./mlruns`` or an absolute path) is a local path to MLflow too, so the
# empty string is included.
_FILE_STORE_SCHEMES = frozenset({"", "file"})

# Truthy spellings accepted for the opt-in flag, matched case-insensitively.
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _flag_enabled(value: str | None) -> bool:
    """Return whether an opt-in flag env var is set to a truthy value."""
    return value is not None and value.strip().lower() in _TRUE_VALUES


def is_file_store(tracking_uri: str) -> bool:
    """Return whether ``tracking_uri`` names a local file store rather than a server.

    A ``file:`` URI or a scheme-less path is a local file store; ``http(s)://``
    (and any other network scheme) is a remote tracking server.

    Args:
        tracking_uri: The MLflow tracking URI to classify.

    Returns:
        ``True`` for a local file store, ``False`` for a server URI.
    """
    return urlparse(tracking_uri).scheme.lower() in _FILE_STORE_SCHEMES


def resolve_tracking_uri() -> str:
    """Resolve and validate the MLflow tracking URI from the environment.

    Reads ``MLFLOW_TRACKING_URI`` (required) and enforces the file-store guard: a
    local file store is rejected unless ``MLFLOW_ALLOW_FILE_STORE`` opts in. No
    MLflow import, network call, or credential is involved, so this can run — and
    fail fast — before the stage crosses the (lazy) tracking boundary.

    Returns:
        The validated tracking URI, ready to hand to :mod:`tracking`.

    Raises:
        ConfigError: If ``MLFLOW_TRACKING_URI`` is unset/empty, or names a local
            file store without ``MLFLOW_ALLOW_FILE_STORE`` set.
    """
    uri = require_env(TRACKING_URI_ENV)

    if is_file_store(uri) and not _flag_enabled(os.environ.get(ALLOW_FILE_STORE_ENV)):
        raise ConfigError(
            f"{TRACKING_URI_ENV}={uri!r} points at a local file store, which records "
            f"runs to the local filesystem instead of the shared MLflow tracking "
            f"server. In a cluster that filesystem is the pod's ephemeral storage, so "
            f"every run, metric, and artifact is lost when the pod exits. Point "
            f"{TRACKING_URI_ENV} at the MLflow server (in-cluster: the tracking "
            f"Service supplied by the pipeline ConfigMap; locally: a "
            f"'kubectl port-forward' to it or your own 'mlflow server'), or set "
            f"{ALLOW_FILE_STORE_ENV}=true to deliberately use a local file store for "
            f"offline development."
        )
    return uri


def resolve_experiment_name() -> str:
    """Return the MLflow experiment the stages should log under.

    Reads the optional ``MLFLOW_EXPERIMENT_NAME`` and falls back to
    :data:`DEFAULT_EXPERIMENT_NAME`. Unlike the tracking URI this is not required:
    a sensible default keeps the pipeline runnable out of the box while still
    grouping runs under a named experiment rather than MLflow's "Default".

    Returns:
        The experiment name (a non-empty string).
    """
    value = os.environ.get(EXPERIMENT_NAME_ENV, "").strip()
    return value or DEFAULT_EXPERIMENT_NAME
