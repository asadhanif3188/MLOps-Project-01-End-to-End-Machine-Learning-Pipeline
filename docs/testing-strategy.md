# Testing Strategy

This document describes how the ML pipeline is **tested**: the philosophy behind
the suite, how the `tests/` tree is laid out, and where testing goes next. It is
the reference for contributors adding or changing tests, and the companion to the
[Exception Strategy](exception-strategy.md) and [Type Safety](type-safety.md)
documents.

It establishes the *testing foundation* called for by engineering-review finding
[**H-3**](reviews/sprint-02-engineering-review.md) ("No automated tests") and
Sprint-02 Epic 6: a `pytest` setup with smoke tests, focused critical-component
tests, reusable fixtures, and shared configuration — deliberately scoped for
**quality over quantity**, not coverage percentage.

---

## 1. Philosophy

The suite is built on a few explicit choices. Each one trades breadth for signal.

- **Test the contracts, not the lines.** We test the guarantees the code makes —
  "a missing file becomes a typed `DataError`", "`run_stage` exits non-zero on
  failure" — rather than chasing a coverage number. A test earns its place by
  catching a regression a reader would actually care about.
- **Quality over quantity.** A small suite of meaningful, fast, deterministic
  tests is worth more than a large one that is slow, flaky, or asserts trivia.
  We do **not** target a coverage percentage; we target the behaviours most
  likely to break and most expensive to break silently.
- **Prioritize the critical component.** The pipeline's IO/config/serialization
  layer ([`src/pipeline_io.py`](../src/pipeline_io.py)) is where every stage
  meets the filesystem, and where the typed-error contract lives — so it gets the
  deepest coverage. Its *error paths* matter more than its happy paths, because
  the error paths are the entire reason the module exists.
- **Fast, isolated, deterministic.** Every test runs in milliseconds, touches
  only a pytest-provided temp directory, and needs no network, no MLflow, and no
  real `data/` or `models/` tree. Order-independence is a hard requirement.
- **No mocks for what we don't own the behaviour of.** Rather than reimplement
  MLflow or the network as mocks, we draw the test boundary at the pipeline's own
  code: the ML computation is separated from the MLflow boundary
  ([`tracking.py`](../src/tracking.py)), and the one place a stage crosses that
  boundary is replaced in tests by the `stub_tracking` in-memory recorder — so no
  test imports MLflow or touches the network. See the
  [roadmap](#4-future-testing-roadmap).
- **Skip honestly, never fake.** When a test genuinely needs an optional runtime
  dependency that isn't installed (e.g. `mlflow`), it **skips** with a visible
  reason — it does not silently pass. A green run means what it says.

### What is tested now — and what is not

| Area | Status | Rationale |
|------|--------|-----------|
| Exception hierarchy ([`exceptions.py`](../src/exceptions.py)) | ✅ Unit | The taxonomy is a contract `run_stage` relies on; cheap to pin. |
| IO / config / serialization ([`pipeline_io.py`](../src/pipeline_io.py)) | ✅ Unit (deep) | The critical component: pure logic, typed error boundaries. |
| Stage entry-point handling ([`stage_runner.py`](../src/stage_runner.py)) | ✅ Unit | Exit codes and non-swallowing are core operational guarantees. |
| Module import / wiring (all of `src/`) | ✅ Smoke | Catches syntax/import breakage in milliseconds. |
| Stage *bodies* (`preprocess`, `train`, `evaluate` compute logic) | ✅ Unit (Sprint 4) | The ML compute (`run_training`, `compute_metrics`) was separated from IO + MLflow (review finding H-6), so it is now unit-tested offline. |
| Stage composition (`preprocess → train → evaluate`) | ✅ Integration (Sprint 4) | One end-to-end run through real temp files with MLflow stubbed; proves each stage's output is consumable by the next. |
| Pipeline definition (`dvc.yaml`/`params.yaml`/`src` consistency) | ✅ Contract (Sprint 4) | Static checks of lineage, parameter consistency, and single-owner artifacts — the CI-enforceable half of the pipeline contract. |
| End-to-end `dvc repro` execution | ⬜ Deferred | Needs a committed fixture dataset + `dvc.lock`; the raw dataset is remote-only. See §4. |

> This is now a four-tier suite (smoke · unit · integration · contract), not just
> a foundation. The one remaining gap above is intentional and tracked.

---

## 2. Directory layout

Tests live under a top-level [`tests/`](../tests/) directory, split by the *kind*
of test rather than mirroring the `src/` module tree. Configuration is centralized
in [`pyproject.toml`](../pyproject.toml); shared fixtures in
[`tests/conftest.py`](../tests/conftest.py).

```text
tests/
├── conftest.py              # shared fixtures (sample/training data, params, MLflow stub)
├── smoke/
│   └── test_smoke.py        # every module imports; each stage exposes main()
├── unit/
│   ├── test_exceptions.py   # the exception taxonomy is a stable contract
│   ├── test_pipeline_io.py  # the critical component — happy + error paths
│   ├── test_stage_runner.py # exit codes, logging, non-swallowing at the boundary
│   ├── test_preprocess.py   # preprocess produces a headed, consumable CSV
│   ├── test_train.py        # run_training: seeded, deterministic, params applied
│   └── test_evaluate.py     # compute_metrics: accuracy in [0, 1], schema errors
├── integration/
│   └── test_pipeline.py     # preprocess → train → evaluate through real files (MLflow stubbed)
└── contract/
    └── test_pipeline_contract.py # dvc.yaml/params.yaml/src agree with the pipeline contract
```

| Location | Contains | Marker |
|----------|----------|--------|
| `tests/smoke/` | Import-and-wiring checks — the cheapest signal that nothing is fundamentally broken. | `smoke` |
| `tests/unit/` | Isolated tests of one component or one stage's pure compute, no external services. | `unit` |
| `tests/integration/` | The three stages run together through real temp files, MLflow stubbed. | `integration` |
| `tests/contract/` | Static `dvc.yaml`/`params.yaml`/`src` consistency checks (pure parsing). | `contract` |
| `tests/conftest.py` | Fixtures shared across the suite. | — |

### Configuration

Pytest is configured under `[tool.pytest.ini_options]` in
[`pyproject.toml`](../pyproject.toml):

- **`pythonpath = ["src"]`** — mirrors how the stages actually run. They live in
  `src/` and import siblings by bare module name (`from exceptions import ...`),
  so the tests import them the same way, with no package prefix and no editable
  install.
- **`testpaths = ["tests"]`** — a bare `pytest` discovers only the suite.
- **`addopts = "-ra --strict-markers"`** — `-ra` summarizes every non-pass
  (skips, xfails) so reasons stay visible; `--strict-markers` turns a mistyped
  marker into an error instead of a silent no-op.
- **`markers`** — a declared taxonomy (`smoke`, `unit`, `integration`,
  `contract`) so slices can be selected with `-m` and the marker set stays honest.

### Fixtures

[`conftest.py`](../tests/conftest.py) provides three small, dependency-free
fixtures, all built on pytest's `tmp_path` so nothing touches the real data tree:

| Fixture | Provides |
|---------|----------|
| `sample_dataframe` | A tiny, well-formed DataFrame with the binary `Outcome` target the stages expect. |
| `csv_path` | `sample_dataframe` written to a real CSV in a temp directory. |
| `params_file` | A representative `params.yaml` on disk, mirroring the real one. |
| `training_frame` / `training_csv` | A balanced 30-row dataset (and its CSV) large enough for the train stage's 3-fold CV. |
| `stub_tracking` | Swaps the lazily-imported `tracking` module for an in-memory recorder, so a stage's read → compute → persist path runs without importing MLflow or touching the network. |

The real network boundary (MLflow, DagsHub) is never contacted: the ML compute
imports no MLflow, and `stub_tracking` neutralizes the one place a stage crosses
the tracking boundary. No live service or credential is required by any test.

---

## 3. Running the tests

From the repository root, with dev dependencies installed
(`pip install -r requirements-dev.txt`):

```bash
# The whole suite
python -m pytest

# Only the fast smoke tests
python -m pytest -m smoke

# Only the unit tests
python -m pytest -m unit

# The offline pipeline-definition contract checks (no data, no network)
python -m pytest -m contract

# The end-to-end integration test (MLflow stubbed)
python -m pytest -m integration

# With coverage, for insight (not as a gate)
python -m pytest --cov=src --cov-report=term-missing
```

Tests that need an uninstalled optional runtime dependency (e.g. `mlflow` for the
`train`/`evaluate` import smoke test) **skip** with a printed reason rather than
fail, so the suite stays green in a lean dev environment while still exercising
those imports wherever the full runtime is installed (such as CI).

> **Coverage is a lens, not a target.** Run `--cov` to *find* untested behaviour
> worth testing — do not add tests merely to move the number.

---

## 4. Future testing roadmap

Testing grows with the pipeline. The stages below align with the project
[roadmap](roadmap.md) and the engineering-review findings.

### Delivered in Sprint 4 (v1.3.0)

- ✅ **Decoupled stage logic from IO/MLflow (review finding
  [H-6](reviews/sprint-02-engineering-review.md)).** The pure logic
  (`run_training`, `compute_metrics`) is separated from side effects (MLflow via
  `tracking.py`, disk via `pipeline_io`), so `train` and `evaluate` are
  unit-tested without a network.
- ✅ **Stage-body unit tests.** `preprocess` output shape, `train` producing a
  deterministic fitted estimator on the `training_frame` fixture, and `evaluate`
  returning an accuracy in `[0, 1]` (plus a schema-mismatch `ModelError`).
- ✅ **End-to-end integration test.** `preprocess → train → evaluate` through real
  temp files with MLflow stubbed (`integration` marker).
- ✅ **Contract tests.** Static `dvc.yaml`/`params.yaml`/`src` consistency checks
  (`contract` marker) — moved earlier than originally planned because they are the
  offline, CI-safe way to enforce the pipeline contract.
- ✅ **Tiered markers + CI.** `smoke`/`unit`/`integration`/`contract` markers; the
  full suite (plus Ruff and offline DVC integrity) runs on every push and pull
  request (see [CI/CD](ci-cd.md)).

### Still ahead

- **`logging_config` behaviour.** Idempotent handler attachment, `LOG_LEVEL` /
  `LOG_DIR` resolution, and the noisy-logger caps.
- **End-to-end `dvc repro` execution test.** A real `dvc repro` over a committed
  fixture dataset with a committed `dvc.lock`, so CI validates *execution*, not
  just the definition. Blocked today by the remote-only raw dataset.
- **mypy as a CI gate** and **branch protection** requiring green checks (carried
  from Sprint 3).
- **Longer term (Roadmap v4+):** property-based tests (e.g. Hypothesis) for the IO
  helpers, data-validation tests for input schema drift, and model-quality gates
  (accuracy regression thresholds before a model is promoted). A genuine held-out
  evaluation split is the prerequisite for meaningful quality gates.

---

## 5. Adding a test — checklist

When you add or change behaviour:

1. **Decide the tier.** Wiring/import concern → `smoke`. Single component or one
   stage's pure compute → `unit`. Multiple stages composed → `integration`.
   `dvc.yaml`/`params.yaml`/`src` consistency → `contract`. Mark it accordingly
   (`@pytest.mark.smoke` / `.unit` / `.integration` / `.contract`).
2. **Test the contract.** Assert the guarantee (the typed error, the exit code,
   the returned shape) — not incidental implementation detail.
3. **Cover the error path.** For anything that raises a typed
   [`PipelineError`](exception-strategy.md), test that the *right* type is raised
   and, where it matters, that the original cause is chained.
4. **Reuse fixtures.** Prefer `sample_dataframe` / `csv_path` / `params_file`
   over hand-rolling inputs; add a new fixture to `conftest.py` only if it will
   be shared.
5. **Keep it fast and isolated.** Use `tmp_path`; no network, no real data tree,
   no dependence on test order.
6. **Skip, don't fake.** If a test needs an optional runtime dependency, skip
   with a clear reason when it is absent rather than mocking around it.

---

## Related Documentation

- [Exception Strategy](exception-strategy.md) — the typed errors these tests assert.
- [Type Safety](type-safety.md) — typing conventions and the mypy configuration.
- [Architecture](architecture.md) — system overview the tests exercise.
- [Roadmap](roadmap.md) — versioned milestones, including the testing epics.
- [Engineering Review](reviews/sprint-02-engineering-review.md) — findings H-3 and H-6.
