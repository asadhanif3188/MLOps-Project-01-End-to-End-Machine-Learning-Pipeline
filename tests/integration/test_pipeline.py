"""Integration test: the three stages run together, end to end.

Every other test isolates a single component. This one proves the stages
*compose* — that the artifact each stage owns is the artifact the next stage can
actually consume:

    raw CSV --preprocess--> processed CSV --train--> model --evaluate--> metrics

That composition is a real property no single-stage test checks: preprocess's
*headered* output has to be selectable by column name in train, and train's
pickled model has to load and predict against evaluate's feature schema.

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
from train import train


@pytest.mark.integration
def test_full_pipeline_produces_consumable_artifacts(
    tmp_path: Path,
    training_frame: pd.DataFrame,
    stub_tracking: dict[str, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """preprocess -> train -> evaluate produces each declared artifact, and each
    stage's output is consumed by the next without manual fix-up."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://stub")

    raw = tmp_path / "raw" / "data.csv"
    raw.parent.mkdir(parents=True)
    training_frame.to_csv(raw, index=False)

    processed = tmp_path / "processed" / "data.csv"
    model = tmp_path / "models" / "model.pkl"
    metrics = tmp_path / "metrics" / "metrics.json"

    preprocess(str(raw), str(processed))
    train(
        str(processed),
        str(model),
        "Outcome",
        random_state=42,
        n_estimators=10,
        max_depth=3,
    )
    evaluate(str(processed), str(model), "Outcome", str(metrics))

    # Every declared stage output exists and is readable.
    assert processed.exists()
    assert model.exists()
    assert metrics.exists()

    payload = json.loads(metrics.read_text(encoding="utf-8"))
    assert set(payload) == {"accuracy"}
    assert 0.0 <= payload["accuracy"] <= 1.0

    # Both boundary-crossing stages logged exactly one run through the stub —
    # proof the tracking calls fired without any real MLflow being importable.
    assert len(stub_tracking["training"]) == 1
    assert len(stub_tracking["evaluation"]) == 1
