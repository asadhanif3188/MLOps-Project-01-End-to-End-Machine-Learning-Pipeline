"""Shared fixtures for the pipeline test suite.

Fixtures here are deliberately small and dependency-free: they build the
handful of inputs the pipeline's IO layer actually consumes — a params mapping,
a CSV on disk, a labelled DataFrame — using pytest's built-in ``tmp_path`` so
every test gets an isolated filesystem and nothing touches the real ``data/``
or ``models/`` trees.

Anything that needs a *network* (MLflow, DagsHub) is intentionally absent: those
boundaries are covered by smoke tests and left to integration testing, not
reimplemented as mocks here.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """A tiny, well-formed dataset shaped like the pipeline's real input.

    Two feature columns plus the binary ``Outcome`` target the train and
    evaluate stages expect. Small enough to reason about, valid enough to flow
    through the IO helpers unchanged.
    """
    return pd.DataFrame(
        {
            "Glucose": [85, 168, 90, 130],
            "BloodPressure": [66, 74, 68, 70],
            "Outcome": [0, 1, 0, 1],
        }
    )


@pytest.fixture
def csv_path(tmp_path: Path, sample_dataframe: pd.DataFrame) -> Path:
    """Path to ``sample_dataframe`` written as a real CSV in a temp dir."""
    path = tmp_path / "dataset.csv"
    sample_dataframe.to_csv(path, index=False)
    return path


@pytest.fixture
def params_file(tmp_path: Path) -> Path:
    """A representative ``params.yaml`` on disk, mirroring the real one."""
    params = {
        "preprocess": {
            "input": "data/raw/data.csv",
            "output": "data/processed/data.csv",
        },
        "train": {
            "input": "data/raw/data.csv",
            "output": "models/model.pkl",
            "random_state": 42,
            "n_estimators": 100,
            "max_depth": 5,
        },
    }
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(params), encoding="utf-8")
    return path
