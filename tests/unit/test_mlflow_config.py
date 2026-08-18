"""Unit tests for the MLflow tracking configuration model (:mod:`mlflow_config`).

These exercise the configuration *behavior* the pipeline relies on to log to the
in-cluster MLflow server (ADR-026), with no MLflow import, network, or tracking
server involved:

* the tracking URI is required and returned verbatim for a server URI;
* a local file store is rejected unless explicitly opted into, so a run cannot
  silently record to ephemeral pod storage instead of the shared server; and
* the opt-in flag is parsed the way an operator would reasonably spell it.
"""

import pytest

from exceptions import ConfigError
from mlflow_config import (
    ALLOW_FILE_STORE_ENV,
    DEFAULT_EXPERIMENT_NAME,
    EXPERIMENT_NAME_ENV,
    TRACKING_URI_ENV,
    is_file_store,
    resolve_experiment_name,
    resolve_tracking_uri,
)

# --------------------------------------------------------------------------- #
# is_file_store — classifying a URI as a local store vs a server
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    "uri",
    [
        "http://mlflow.mlops.svc.cluster.local:5000",
        "https://mlflow.example.com",
        "http://127.0.0.1:5000",
    ],
)
def test_server_uris_are_not_file_stores(uri: str) -> None:
    """An ``http(s)://`` tracking URI is a server, not a local file store."""
    assert is_file_store(uri) is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "uri",
    [
        "file:./mlruns",
        "file:/var/mlruns",
        "./mlruns",  # scheme-less path — MLflow treats it as a local store too
        "mlruns",
    ],
)
def test_file_and_scheme_less_uris_are_file_stores(uri: str) -> None:
    """``file:`` URIs and bare paths are local file stores."""
    assert is_file_store(uri) is True


# --------------------------------------------------------------------------- #
# resolve_tracking_uri — the required, validated server URI
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_resolve_returns_server_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured server URI is returned unchanged (the common in-cluster case)."""
    uri = "http://mlflow.mlops.svc.cluster.local:5000"
    monkeypatch.setenv(TRACKING_URI_ENV, uri)
    monkeypatch.delenv(ALLOW_FILE_STORE_ENV, raising=False)

    assert resolve_tracking_uri() == uri


@pytest.mark.unit
def test_resolve_unset_raises_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unset tracking URI fails fast as a ``ConfigError``."""
    monkeypatch.delenv(TRACKING_URI_ENV, raising=False)
    with pytest.raises(ConfigError, match="not set"):
        resolve_tracking_uri()


@pytest.mark.unit
def test_resolve_empty_raises_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty tracking URI is treated as unset."""
    monkeypatch.setenv(TRACKING_URI_ENV, "")
    with pytest.raises(ConfigError, match="not set"):
        resolve_tracking_uri()


# --------------------------------------------------------------------------- #
# resolve_tracking_uri — the file-store guard
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize("uri", ["file:./mlruns", "./mlruns"])
def test_resolve_rejects_file_store_by_default(
    uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file store without the opt-in flag is rejected with an actionable error,
    naming both the flag and the in-cluster Service so a silent, ephemeral-storage
    run cannot happen by accident."""
    monkeypatch.setenv(TRACKING_URI_ENV, uri)
    monkeypatch.delenv(ALLOW_FILE_STORE_ENV, raising=False)

    with pytest.raises(ConfigError, match="local file store") as excinfo:
        resolve_tracking_uri()
    assert ALLOW_FILE_STORE_ENV in str(excinfo.value)


@pytest.mark.unit
@pytest.mark.parametrize("flag", ["true", "TRUE", "1", "yes", "on", " True "])
def test_resolve_allows_file_store_when_opted_in(
    flag: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the opt-in flag set (any reasonable truthy spelling), a local file
    store is permitted for offline development."""
    monkeypatch.setenv(TRACKING_URI_ENV, "file:./mlruns")
    monkeypatch.setenv(ALLOW_FILE_STORE_ENV, flag)

    assert resolve_tracking_uri() == "file:./mlruns"


@pytest.mark.unit
@pytest.mark.parametrize("flag", ["false", "0", "no", "", "maybe"])
def test_resolve_file_store_flag_falsey_still_rejects(
    flag: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-truthy flag value does not enable the file store."""
    monkeypatch.setenv(TRACKING_URI_ENV, "file:./mlruns")
    monkeypatch.setenv(ALLOW_FILE_STORE_ENV, flag)

    with pytest.raises(ConfigError, match="local file store"):
        resolve_tracking_uri()


@pytest.mark.unit
def test_resolve_flag_does_not_affect_server_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opt-in flag is irrelevant to a server URI — it is returned regardless."""
    uri = "http://mlflow.mlops.svc.cluster.local:5000"
    monkeypatch.setenv(TRACKING_URI_ENV, uri)
    monkeypatch.setenv(ALLOW_FILE_STORE_ENV, "true")

    assert resolve_tracking_uri() == uri


# --------------------------------------------------------------------------- #
# resolve_experiment_name — optional, defaulted experiment grouping
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_experiment_name_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no override, runs group under the named default (not MLflow's Default)."""
    monkeypatch.delenv(EXPERIMENT_NAME_ENV, raising=False)
    assert resolve_experiment_name() == DEFAULT_EXPERIMENT_NAME


@pytest.mark.unit
def test_experiment_name_override_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit experiment name is honoured verbatim."""
    monkeypatch.setenv(EXPERIMENT_NAME_ENV, "sprint-07-integration")
    assert resolve_experiment_name() == "sprint-07-integration"


@pytest.mark.unit
@pytest.mark.parametrize("value", ["", "   "])
def test_experiment_name_blank_falls_back_to_default(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blank/whitespace override falls back to the default rather than creating
    an unnamed experiment."""
    monkeypatch.setenv(EXPERIMENT_NAME_ENV, value)
    assert resolve_experiment_name() == DEFAULT_EXPERIMENT_NAME
