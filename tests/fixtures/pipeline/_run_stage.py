"""Offline entry point for the fixture pipeline's stages.

``dvc repro`` drives each fixture stage through this wrapper instead of calling
``src/<stage>.py`` directly. It runs the **real** production stage ``main()`` —
the same read → compute → persist code the production pipeline runs — but first
replaces the lazily-imported :mod:`tracking` module with an in-memory no-op, so
the MLflow / DagsHub boundary is never crossed.

Why this is honest, not a shortcut:

* Tracking is a *side effect to an external service*; it produces **no
  DVC-tracked artifact**. The pipeline's reproduced outputs (processed data, the
  train/held-out split, the model, the metrics) come entirely from the seeded ML
  computation, which runs here unchanged. Neutralizing tracking cannot change a
  single reproduced byte — it only removes the network dependency.
* This is the *same* boundary substitution the unit and integration tests use
  (``conftest.stub_tracking``), sanctioned by ADR-006 decision 4 ("external
  services must not be required for ordinary validation"). Applying it in a
  subprocess is what lets ``dvc repro`` run the whole graph with **no MLflow, no
  network, and no credentials** — the property that makes the fixture pipeline
  safe for CI. See ADR-008.

Usage (from ``dvc.yaml``): ``python _run_stage.py <preprocess|split|train|evaluate>``
"""

import importlib
import os
import sys
import types
from pathlib import Path

# The production stages live in ``src/`` and import their siblings by bare module
# name (``from exceptions import ...``). Put ``src/`` first on the path so those
# resolve exactly as they do in production, regardless of this wrapper's location.
_SRC = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(_SRC))


def _install_offline_tracking() -> None:
    """Pre-insert a no-op :mod:`tracking` so the stages' lazy import finds it.

    The stages do ``from tracking import ...`` only at the point they cross the
    MLflow boundary; seeding ``sys.modules["tracking"]`` here makes that import
    resolve to this recorder-free stub, so MLflow is never imported.
    """
    module = types.ModuleType("tracking")
    module.build_signature = lambda *_args, **_kwargs: "offline-fixture-signature"
    module.log_training_run = lambda *_args, **_kwargs: None
    module.log_evaluation = lambda *_args, **_kwargs: None
    sys.modules["tracking"] = module


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python _run_stage.py <stage-name>")
    stage = sys.argv[1]

    _install_offline_tracking()
    # ``train``/``evaluate`` call ``require_env('MLFLOW_TRACKING_URI')`` before the
    # (now stubbed) tracking call. Provide a placeholder so that guard passes; its
    # value is never used because tracking is a no-op. ``setdefault`` respects any
    # value already set by the environment.
    os.environ.setdefault("MLFLOW_TRACKING_URI", "offline://fixture")

    module = importlib.import_module(stage)
    module.main()


if __name__ == "__main__":
    main()
