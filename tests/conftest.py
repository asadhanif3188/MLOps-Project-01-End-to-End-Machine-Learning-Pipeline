"""Shared fixtures for the pipeline test suite.

Fixtures here are deliberately small and dependency-free: they build the
handful of inputs the pipeline's IO layer actually consumes — a params mapping,
a CSV on disk, a labelled DataFrame — using pytest's built-in ``tmp_path`` so
every test gets an isolated filesystem and nothing touches the real ``data/``
or ``models/`` trees.

The one *network* boundary — MLflow / DagsHub — is not reimplemented as a rich
mock; instead the :func:`stub_tracking` fixture swaps the whole :mod:`tracking`
module for an in-memory recorder, so a stage's read → compute → persist path can
run end to end while the tracking call is neutralized (see its docstring).
"""

import sys
import types
from pathlib import Path
from typing import Any

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
        "split": {
            "input": "data/processed/data.csv",
            "train_output": "data/processed/train.csv",
            "test_output": "data/processed/test.csv",
            "target": "Outcome",
            "test_size": 0.2,
            "random_state": 42,
        },
        "train": {
            "input": "data/processed/train.csv",
            "output": "models/model.pkl",
            "target": "Outcome",
            "random_state": 42,
            "n_estimators": 100,
            "max_depth": 5,
        },
        "evaluate": {
            "data": "data/processed/test.csv",
            "model": "models/model.pkl",
            "target": "Outcome",
            "metrics": "metrics/metrics.json",
        },
    }
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(params), encoding="utf-8")
    return path


@pytest.fixture
def training_frame() -> pd.DataFrame:
    """A balanced 30-row dataset suitable for the train stage's 3-fold CV.

    The 4-row :func:`sample_dataframe` is too small to split into three folds, so
    the train/evaluate *stage* tests use this one. It is deterministically
    constructed (no randomness) and 15/15 class-balanced, so the only source of
    non-determinism a test can observe is the model, which is itself seeded.
    """
    n = 30
    return pd.DataFrame(
        {
            "Glucose": list(range(n)),
            "BloodPressure": [x % 5 for x in range(n)],
            "Outcome": [0, 1] * (n // 2),
        }
    )


@pytest.fixture
def training_csv(tmp_path: Path, training_frame: pd.DataFrame) -> Path:
    """Path to ``training_frame`` written as a processed CSV in a temp dir.

    Shaped like the ``preprocess`` output the train/evaluate stages consume: a
    headered CSV whose columns can be selected by name.
    """
    path = tmp_path / "processed.csv"
    training_frame.to_csv(path, index=False)
    return path


@pytest.fixture
def stub_tracking(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Replace the lazily-imported :mod:`tracking` module with an in-memory stub.

    The stages import :mod:`tracking` *lazily*, at the point they cross the
    MLflow boundary (``from tracking import ...`` inside :func:`train.train` /
    :func:`evaluate.evaluate`). Pre-inserting a stub into ``sys.modules`` makes
    that import resolve to this recorder instead of the real module — so a
    stage's read → compute → persist path runs and can be asserted on, while
    **MLflow is never imported, no network is touched, and no DagsHub credentials
    are needed**. Because the boundary is already lazy (ADR-006 decision 4), this
    requires *no* change to production code.

    Returns:
        A record of the calls each stage made across the boundary, keyed
        ``"training"`` / ``"evaluation"`` / ``"signature"``, so a test can assert
        the stage handed the tracking layer the expected payload.
    """
    calls: dict[str, list[Any]] = {"training": [], "evaluation": [], "signature": []}
    module = types.ModuleType("tracking")

    def build_signature(model_input: Any, model_output: Any) -> str:
        calls["signature"].append((model_input, model_output))
        return "stub-signature"

    def log_training_run(tracking_uri: str, **kwargs: Any) -> None:
        calls["training"].append({"tracking_uri": tracking_uri, **kwargs})

    def log_evaluation(tracking_uri: str, metrics: Any, **kwargs: Any) -> None:
        calls["evaluation"].append(
            {"tracking_uri": tracking_uri, "metrics": dict(metrics), **kwargs}
        )

    module.build_signature = build_signature  # type: ignore[attr-defined]
    module.log_training_run = log_training_run  # type: ignore[attr-defined]
    module.log_evaluation = log_evaluation  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tracking", module)
    return calls
