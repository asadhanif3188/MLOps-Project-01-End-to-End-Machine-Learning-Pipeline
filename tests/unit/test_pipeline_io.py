"""Tests for the IO / config / serialization layer.

:mod:`pipeline_io` is the pipeline's single most critical component: every
filesystem, config, and model boundary flows through it, and its whole reason
for existing is to convert low-level failures into the pipeline's own typed,
actionable errors. So these tests cover both halves of the contract —

* the **happy path** (values round-trip correctly), and
* the **error path** (the *right* typed exception is raised, with the original
  cause chained).

The error paths matter more than the happy paths here: they are the behaviour
the module was written to guarantee, and the easiest thing to regress.
"""

from pathlib import Path

import pandas as pd
import pytest

from exceptions import ConfigError, DataError, ModelError
from pipeline_io import (
    ensure_columns,
    load_params,
    load_pickle,
    read_csv,
    require_env,
    save_pickle,
    write_csv,
    write_json,
)

# --------------------------------------------------------------------------- #
# load_params
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_load_params_returns_stage_section(params_file: Path) -> None:
    section = load_params(str(params_file), "train", required=("input", "output"))
    assert section["random_state"] == 42
    assert section["n_estimators"] == 100


@pytest.mark.unit
def test_load_params_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_params(str(tmp_path / "nope.yaml"), "train")


@pytest.mark.unit
def test_load_params_invalid_yaml_raises_config_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("train: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_params(str(bad), "train")


@pytest.mark.unit
def test_load_params_missing_section_raises_config_error(params_file: Path) -> None:
    # ``deploy`` is deliberately not a pipeline stage, so the section is absent.
    with pytest.raises(ConfigError, match="no 'deploy' section"):
        load_params(str(params_file), "deploy")


@pytest.mark.unit
def test_load_params_returns_evaluate_section(params_file: Path) -> None:
    """The evaluate stage's config contract: the ``evaluate`` section (renamed
    from ``test`` in Sprint 4 PR 3) exposes its declared input/output keys."""
    section = load_params(
        str(params_file), "evaluate", required=("data", "model", "target", "metrics")
    )
    assert section["target"] == "Outcome"
    assert section["metrics"] == "metrics/metrics.json"


@pytest.mark.unit
def test_load_params_non_mapping_section_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "params.yaml"
    path.write_text("train:\n  - just\n  - a\n  - list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_params(str(path), "train")


@pytest.mark.unit
def test_load_params_missing_required_key_raises_config_error(
    params_file: Path,
) -> None:
    with pytest.raises(ConfigError, match="missing key"):
        load_params(str(params_file), "train", required=("input", "does_not_exist"))


@pytest.mark.unit
def test_load_params_allows_required_key_with_none_value(tmp_path: Path) -> None:
    """Presence-only validation: a required key may legitimately be ``None``
    (e.g. ``max_depth: null`` meaning unbounded), and must not be rejected."""
    path = tmp_path / "params.yaml"
    path.write_text("train:\n  max_depth:\n", encoding="utf-8")  # value is null
    section = load_params(str(path), "train", required=("max_depth",))
    assert section["max_depth"] is None


# --------------------------------------------------------------------------- #
# require_env
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_require_env_returns_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    assert require_env("MLFLOW_TRACKING_URI") == "http://localhost:5000"


@pytest.mark.unit
def test_require_env_unset_raises_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(ConfigError, match="not set"):
        require_env("MLFLOW_TRACKING_URI")


@pytest.mark.unit
def test_require_env_empty_raises_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")
    with pytest.raises(ConfigError, match="not set"):
        require_env("MLFLOW_TRACKING_URI")


# --------------------------------------------------------------------------- #
# read_csv
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_read_csv_round_trips(csv_path: Path, sample_dataframe: pd.DataFrame) -> None:
    loaded = read_csv(str(csv_path))
    pd.testing.assert_frame_equal(loaded, sample_dataframe)


@pytest.mark.unit
def test_read_csv_missing_raises_data_error(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="not found"):
        read_csv(str(tmp_path / "missing.csv"))


@pytest.mark.unit
def test_read_csv_empty_raises_data_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(DataError, match="empty"):
        read_csv(str(empty))


# --------------------------------------------------------------------------- #
# write_csv
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_write_csv_creates_parent_dirs_and_round_trips(
    tmp_path: Path, sample_dataframe: pd.DataFrame
) -> None:
    """The nested directory does not exist yet — ``write_csv`` must create it."""
    dest = tmp_path / "nested" / "deeper" / "out.csv"
    write_csv(sample_dataframe, str(dest))
    assert dest.exists()
    pd.testing.assert_frame_equal(pd.read_csv(dest), sample_dataframe)


@pytest.mark.unit
def test_write_csv_honors_header_and_index_flags(
    tmp_path: Path, sample_dataframe: pd.DataFrame
) -> None:
    """``write_csv`` must honor the ``header``/``index`` flags: with both off, the
    first field of the first line is a data value, not the column name."""
    dest = tmp_path / "no_header.csv"
    write_csv(sample_dataframe, str(dest), header=False, index=False)
    first_field = dest.read_text(encoding="utf-8").splitlines()[0].split(",")[0]
    assert first_field == "85"  # a data value, not the "Glucose" header


# --------------------------------------------------------------------------- #
# write_json
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_write_json_creates_parent_dirs_and_round_trips(tmp_path: Path) -> None:
    """``write_json`` must create missing parents and write readable JSON."""
    import json

    dest = tmp_path / "metrics" / "metrics.json"
    write_json({"accuracy": 0.75}, str(dest))
    assert dest.exists()
    assert json.loads(dest.read_text(encoding="utf-8")) == {"accuracy": 0.75}


@pytest.mark.unit
def test_write_json_non_serializable_raises_data_error(tmp_path: Path) -> None:
    """A value the JSON encoder cannot handle surfaces as a typed ``DataError``,
    not a raw ``TypeError``."""
    dest = tmp_path / "metrics.json"
    with pytest.raises(DataError):
        write_json({"bad": {1, 2, 3}}, str(dest))  # a set is not JSON-serializable


# --------------------------------------------------------------------------- #
# ensure_columns
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_ensure_columns_passes_when_present(sample_dataframe: pd.DataFrame) -> None:
    # Should simply return None without raising.
    assert ensure_columns(sample_dataframe, ["Outcome"], "sample") is None


@pytest.mark.unit
def test_ensure_columns_missing_raises_with_column_name(
    sample_dataframe: pd.DataFrame,
) -> None:
    with pytest.raises(DataError, match="Target"):
        ensure_columns(sample_dataframe, ["Outcome", "Target"], "sample")


# --------------------------------------------------------------------------- #
# save_pickle / load_pickle
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_pickle_round_trips(tmp_path: Path) -> None:
    obj = {"weights": [1, 2, 3], "name": "rf"}
    path = tmp_path / "nested" / "model.pkl"  # parent created by save_pickle
    save_pickle(obj, str(path))
    assert load_pickle(str(path)) == obj


@pytest.mark.unit
def test_load_pickle_missing_raises_model_error(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="not found"):
        load_pickle(str(tmp_path / "absent.pkl"))


@pytest.mark.unit
def test_load_pickle_corrupt_raises_model_error(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.pkl"
    corrupt.write_bytes(b"this is not a pickle stream")
    with pytest.raises(ModelError, match="corrupt or not a valid pickle"):
        load_pickle(str(corrupt))


@pytest.mark.unit
def test_save_pickle_unpicklable_raises_model_error(tmp_path: Path) -> None:
    """A lambda cannot be pickled by the default pickler — the failure must
    surface as a typed ``ModelError``, not a raw ``PicklingError``."""
    with pytest.raises(ModelError):
        save_pickle(lambda x: x, str(tmp_path / "bad.pkl"))


@pytest.mark.unit
def test_load_pickle_preserves_original_cause(tmp_path: Path) -> None:
    """Error translation must chain the underlying exception (``raise ... from``)
    so tracebacks stay debuggable — a core promise of this module."""
    try:
        load_pickle(str(tmp_path / "absent.pkl"))
    except ModelError as exc:
        assert isinstance(exc.__cause__, FileNotFoundError)
    else:  # pragma: no cover - the call above must raise
        pytest.fail("expected ModelError")
