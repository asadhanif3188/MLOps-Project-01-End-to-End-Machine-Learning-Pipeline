"""Smoke tests: the cheapest possible signal that the pipeline is intact.

These do not exercise behaviour — they assert that every module imports and that
each stage exposes the entry point the DVC pipeline and ``stage_runner`` expect.
A green smoke run means "nothing is fundamentally broken"; a red one usually
means a syntax error, a bad import, or a renamed function — caught in
milliseconds, before any heavier test even starts.

Two tiers of module live here:

* **Core modules** depend only on the test environment's own libraries
  (stdlib, pandas, yaml, dotenv), so they must *always* import.
* **Stage modules** additionally pull in scikit-learn to import (MLflow is now
  reached lazily, through the :mod:`tracking` boundary, so it is *not* needed to
  import a stage — only to run one). scikit-learn is declared in
  ``requirements.txt`` and present in CI, but a lean dev environment may omit it,
  so the import test *skips* rather than fails when it is absent.
* **The tracking module** is the one module that imports MLflow at load, so its
  import check is gated on MLflow being installed.
"""

import importlib
import importlib.util

import pytest

# First-party modules that import cleanly with only the dev/test dependencies.
CORE_MODULES = [
    "exceptions",
    "logging_config",
    "pipeline_io",
    "stage_runner",
    "preprocess",
]

# The four DVC stages, each expected to expose a zero-argument ``main`` that
# ``stage_runner.run_stage`` can call. ``split``, ``train`` and ``evaluate``
# import scikit-learn at load; ``requires`` names a module that must be
# importable first.
STAGES = [
    pytest.param("preprocess", None, id="preprocess"),
    pytest.param("split", "sklearn", id="split"),
    pytest.param("train", "sklearn", id="train"),
    pytest.param("evaluate", "sklearn", id="evaluate"),
]


def _skip_if_runtime_missing(requires: str | None) -> None:
    """Skip when an optional heavy runtime dependency is not installed."""
    if requires and importlib.util.find_spec(requires) is None:
        pytest.skip(f"runtime dependency {requires!r} not installed")


@pytest.mark.smoke
@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_core_module_imports(module_name: str) -> None:
    """Every core pipeline module imports without error."""
    assert importlib.import_module(module_name) is not None


@pytest.mark.smoke
@pytest.mark.parametrize(("module_name", "requires"), STAGES)
def test_stage_exposes_main(module_name: str, requires: str | None) -> None:
    """Each stage exposes a callable ``main`` entry point."""
    _skip_if_runtime_missing(requires)
    module = importlib.import_module(module_name)
    assert callable(getattr(module, "main", None))


@pytest.mark.smoke
def test_tracking_module_imports() -> None:
    """The MLflow boundary module imports wherever MLflow is installed."""
    _skip_if_runtime_missing("mlflow")
    assert importlib.import_module("tracking") is not None
