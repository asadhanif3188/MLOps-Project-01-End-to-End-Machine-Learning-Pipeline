"""Reproducibility test: the committed fixture reproduces equivalent outputs.

CI proves reproducibility by running ``dvc repro`` against the fixture pipeline
(real DVC, real committed lock). This test is the portable, in-suite complement:
it runs the **same production stage code** on the **same committed fixture
dataset** (``tests/fixtures/pipeline/data/raw/data.csv``) and asserts the results
are reproducible — the determinism the reproducibility claim rests on, checked on
every ``pytest`` run with **no DVC, no MLflow, no network, and no credentials**.

What it establishes (ADR-008, pipeline-contract §7):

* **Fixture execution.** The unmodified ``preprocess → split → train → evaluate``
  stage code runs end to end against the committed fixture, producing every
  declared artifact — so the fixture genuinely exercises the pipeline, not a mock.
* **Determinism.** Re-running the seeded split and training yields a
  byte-identical model and identical metrics. Equal inputs + params + seed +
  code ⇒ equal outputs (requirement: "repeated fixture execution produces
  equivalent expected outputs").
* **Held-out evaluation.** ``train`` fits on the training split while the metric
  is scored on the disjoint held-out split (out-of-sample).

The MLflow boundary is neutralized by the ``stub_tracking`` fixture (no artifact
depends on it), exactly as the fixture pipeline's ``_run_stage.py`` wrapper does
for ``dvc repro``.
"""

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

pytest.importorskip("sklearn")

from evaluate import evaluate
from preprocess import preprocess
from split import split, split_dataset
from train import run_training, train

# The committed fixture dataset — the single source of truth the fixture DVC
# pipeline (tests/fixtures/pipeline/dvc.yaml) also reproduces from.
_FIXTURE_RAW = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "pipeline"
    / "data"
    / "raw"
    / "data.csv"
)

# The fixture's declared parameters (mirrors tests/fixtures/pipeline/params.yaml).
_TARGET = "Outcome"
_TEST_SIZE = 0.25
_RANDOM_STATE = 42
_N_ESTIMATORS = 25
_MAX_DEPTH = 3


@pytest.fixture(scope="module")
def fixture_frame() -> pd.DataFrame:
    """The committed fixture dataset as a DataFrame."""
    assert _FIXTURE_RAW.is_file(), (
        f"fixture dataset missing at {_FIXTURE_RAW}; regenerate with "
        "`python tests/fixtures/pipeline/generate_data.py`"
    )
    return pd.read_csv(_FIXTURE_RAW)


@pytest.mark.integration
def test_fixture_split_and_training_are_deterministic(
    fixture_frame: pd.DataFrame,
) -> None:
    """The seeded split and training reproduce equivalent outputs across runs.

    Runs the pure ML compute (``split_dataset`` then ``run_training``) twice with
    the fixture's parameters and asserts the second run reproduces the first: the
    same train/held-out partition, a byte-identical pickled model, and identical
    accuracy. This is the determinism the committed ``dvc.lock`` records.
    """

    def _split() -> tuple[pd.DataFrame, pd.DataFrame]:
        result = split_dataset(
            fixture_frame,
            target=_TARGET,
            test_size=_TEST_SIZE,
            random_state=_RANDOM_STATE,
        )
        return result.train, result.test

    train_a, test_a = _split()
    train_b, test_b = _split()

    # The partition is deterministic: identical rows, in identical order.
    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(test_a, test_b)

    # The held-out guarantee: the two partitions share no row and cover the whole
    # dataset — so training never sees an evaluation row.
    assert set(train_a.index).isdisjoint(test_a.index)
    assert len(train_a) + len(test_a) == len(fixture_frame)

    def _train_model(train_df: pd.DataFrame) -> tuple[bytes, float]:
        result = run_training(
            train_df.drop(columns=[_TARGET]),
            train_df[_TARGET],
            random_state=_RANDOM_STATE,
            n_estimators=_N_ESTIMATORS,
            max_depth=_MAX_DEPTH,
        )
        return pickle.dumps(result.model), result.accuracy

    model_a, accuracy_a = _train_model(train_a)
    model_b, accuracy_b = _train_model(train_b)

    # Byte-identical model and identical in-training accuracy across independent
    # runs — the seeded pipeline is reproducible, not merely close.
    assert model_a == model_b
    assert accuracy_a == accuracy_b


@pytest.mark.integration
def test_fixture_pipeline_runs_end_to_end_held_out(
    tmp_path: Path,
    stub_tracking: dict[str, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full fixture pipeline runs through the real stage entry points.

    Drives ``preprocess → split → train → evaluate`` — the actual stage
    orchestration, over files — starting from the committed fixture raw dataset,
    and checks that every declared artifact is produced and consumable and that
    the evaluation metric is scored on the held-out split (out-of-sample). This is
    what ``dvc repro tests/fixtures/pipeline/dvc.yaml`` does in CI, run here
    without DVC so it is portable to every environment.
    """
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://stub")

    processed = tmp_path / "processed.csv"
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    model = tmp_path / "model.pkl"
    metrics = tmp_path / "metrics.json"

    preprocess(str(_FIXTURE_RAW), str(processed))
    split(
        str(processed),
        str(train_csv),
        str(test_csv),
        target=_TARGET,
        test_size=_TEST_SIZE,
        random_state=_RANDOM_STATE,
    )
    train(
        str(train_csv),
        str(model),
        _TARGET,
        random_state=_RANDOM_STATE,
        n_estimators=_N_ESTIMATORS,
        max_depth=_MAX_DEPTH,
    )
    evaluate(str(test_csv), str(model), _TARGET, str(metrics))

    # Every declared stage output exists.
    for artifact in (processed, train_csv, test_csv, model, metrics):
        assert artifact.exists(), f"stage output missing: {artifact.name}"

    # Held-out boundary, end to end: the file train fitted on and the file
    # evaluate scored share no row. Compare whole rows (the fixture's feature
    # values are not individually unique) and confirm the split is exhaustive.
    train_rows = set(pd.read_csv(train_csv).itertuples(index=False, name=None))
    test_rows = set(pd.read_csv(test_csv).itertuples(index=False, name=None))
    assert train_rows.isdisjoint(test_rows)
    assert len(train_rows) + len(test_rows) == len(pd.read_csv(_FIXTURE_RAW))

    payload = json.loads(metrics.read_text(encoding="utf-8"))
    assert set(payload) == {"accuracy"}
    assert 0.0 <= payload["accuracy"] <= 1.0

    # Both boundary-crossing stages logged through the stub — MLflow never imported.
    assert len(stub_tracking["training"]) == 1
    assert len(stub_tracking["evaluation"]) == 1
