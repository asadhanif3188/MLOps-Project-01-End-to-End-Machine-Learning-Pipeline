"""Stage-level tests for the preprocessing stage.

These exercise :func:`preprocess.preprocess` — the orchestrator — through real
files in a temp dir, rather than its IO helpers in isolation (those live in
``test_pipeline_io``). Preprocess has no MLflow boundary, so no tracking stub is
needed; it is the simplest stage and the only one with *no* prior behavioural
coverage.

The behaviours pinned here are the stage's contract (see pipeline-contract.md):
a valid raw dataset flows through to a **headered** processed output that
preserves every row and column, and a malformed or missing input fails as a
typed :class:`~exceptions.DataError` rather than a raw pandas traceback.
"""

from pathlib import Path

import pandas as pd
import pytest

from exceptions import DataError
from preprocess import preprocess


@pytest.mark.unit
def test_preprocess_writes_processed_output_that_round_trips(
    tmp_path: Path, sample_dataframe: pd.DataFrame
) -> None:
    """A valid raw dataset produces a processed artifact identical in schema and
    content — same columns (features + ``Outcome``) and same rows."""
    raw = tmp_path / "raw.csv"
    sample_dataframe.to_csv(raw, index=False)
    out = tmp_path / "processed" / "data.csv"  # nested dir must be created

    preprocess(str(raw), str(out))

    assert out.exists()
    result = pd.read_csv(out)
    pd.testing.assert_frame_equal(result, sample_dataframe)


@pytest.mark.unit
def test_preprocess_output_carries_the_header_row(
    tmp_path: Path, sample_dataframe: pd.DataFrame
) -> None:
    """The processed output must be *headered*: downstream stages select the
    target/feature columns by name, so the first line has to be the column names,
    not a data row. This is the exact D8 regression the stage was fixed to avoid.
    """
    raw = tmp_path / "raw.csv"
    sample_dataframe.to_csv(raw, index=False)
    out = tmp_path / "processed.csv"

    preprocess(str(raw), str(out))

    first_field = out.read_text(encoding="utf-8").splitlines()[0].split(",")[0]
    assert first_field == "Glucose"  # the column name, not the value 85


@pytest.mark.unit
def test_preprocess_missing_input_raises_data_error(tmp_path: Path) -> None:
    """A missing raw dataset fails predictably as a typed ``DataError`` naming the
    path — not a bare ``FileNotFoundError``."""
    out = tmp_path / "processed.csv"
    with pytest.raises(DataError, match="not found"):
        preprocess(str(tmp_path / "does_not_exist.csv"), str(out))
    assert not out.exists()  # nothing written on failure


@pytest.mark.unit
def test_preprocess_empty_input_raises_data_error(tmp_path: Path) -> None:
    """An empty (malformed) input file surfaces as a typed ``DataError``."""
    raw = tmp_path / "empty.csv"
    raw.write_text("", encoding="utf-8")
    out = tmp_path / "processed.csv"

    with pytest.raises(DataError, match="empty"):
        preprocess(str(raw), str(out))
    assert not out.exists()
