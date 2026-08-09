"""Integration test: the four stages run together, end to end.

Every other test isolates a single component. This one proves the stages
*compose* — that the artifact each stage owns is the artifact the next stage can
actually consume:

    raw CSV --preprocess--> processed CSV --split--> train CSV + held-out CSV
                                                       │            │
                                              train ◀──┘            │
                                                │                   │
                                              model --evaluate------┘--> metrics

That composition is a real property no single-stage test checks: preprocess's
*headered* output has to be splittable by column name, the split's two files have
to be disjoint and separately loadable, train's pickled model has to load and
predict against evaluate's feature schema — and, critically, ``train`` must fit
on the training split while ``evaluate`` scores the *held-out* split, so the
metric is out-of-sample.

External service requirements: **none.** This test needs only scikit-learn and
a temp filesystem. The MLflow / DagsHub boundary is replaced by the
``stub_tracking`` recorder (see ``conftest.py``), so there is no network call,
no tracking server, and no credentials — which is why it is marked
``integration`` yet still runs in CI offline. Live ``dvc repro`` and real MLflow
reproducibility belong to a later PR, not here.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

pytest.importorskip("sklearn")

from evaluate import evaluate
from preprocess import preprocess
from split import split
from train import train


@pytest.mark.integration
def test_full_pipeline_produces_consumable_artifacts(
    tmp_path: Path,
    training_frame: pd.DataFrame,
    stub_tracking: dict[str, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """preprocess -> split -> train -> evaluate produces each declared artifact,
    each stage's output is consumed by the next without manual fix-up, and
    training and evaluation run on disjoint data (held-out evaluation)."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://stub")

    raw = tmp_path / "raw" / "data.csv"
    raw.parent.mkdir(parents=True)
    training_frame.to_csv(raw, index=False)

    processed = tmp_path / "processed" / "data.csv"
    train_data = tmp_path / "processed" / "train.csv"
    test_data = tmp_path / "processed" / "test.csv"
    model = tmp_path / "models" / "model.pkl"
    metrics = tmp_path / "metrics" / "metrics.json"

    preprocess(str(raw), str(processed))
    split(
        str(processed),
        str(train_data),
        str(test_data),
        target="Outcome",
        test_size=0.25,
        random_state=42,
    )
    train(
        str(train_data),
        str(model),
        "Outcome",
        random_state=42,
        n_estimators=10,
        max_depth=3,
    )
    evaluate(str(test_data), str(model), "Outcome", str(metrics))

    # Every declared stage output exists and is readable.
    assert processed.exists()
    assert train_data.exists()
    assert test_data.exists()
    assert model.exists()
    assert metrics.exists()

    # The held-out guarantee, end to end: the file train fitted on and the file
    # evaluate scored share no row (identity carried by the unique Glucose value).
    train_ids = set(pd.read_csv(train_data)["Glucose"])
    test_ids = set(pd.read_csv(test_data)["Glucose"])
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == set(training_frame["Glucose"])

    payload = json.loads(metrics.read_text(encoding="utf-8"))
    assert set(payload) == {"accuracy"}
    assert 0.0 <= payload["accuracy"] <= 1.0

    # Both boundary-crossing stages logged exactly one run through the stub —
    # proof the tracking calls fired without any real MLflow being importable.
    assert len(stub_tracking["training"]) == 1
    assert len(stub_tracking["evaluation"]) == 1
