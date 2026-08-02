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
  code and leave true external integrations to smoke tests and (later)
  integration testing. See the [roadmap](#4-future-testing-roadmap).
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
| Stage *bodies* (`train`, `evaluate` model logic) | ⬜ Deferred | Coupled to MLflow + network; needs the decoupling in review finding H-6. See §4. |
| End-to-end `dvc repro` | ⬜ Deferred | Belongs in CI integration testing (Roadmap v3). |

> This is a **foundation**, not a finished pyramid. The gaps above are
> intentional and tracked, not overlooked.

---

## 2. Directory layout

Tests live under a top-level [`tests/`](../tests/) directory, split by the *kind*
of test rather than mirroring the `src/` module tree. Configuration is centralized
in [`pyproject.toml`](../pyproject.toml); shared fixtures in
[`tests/conftest.py`](../tests/conftest.py).

```text
tests/
├── conftest.py              # shared fixtures (sample data, CSV + params on disk)
├── smoke/
│   └── test_smoke.py        # every module imports; each stage exposes main()
└── unit/
    ├── test_exceptions.py   # the exception taxonomy is a stable contract
    ├── test_pipeline_io.py  # the critical component — happy + error paths
    └── test_stage_runner.py # exit codes, logging, non-swallowing at the boundary
```

| Location | Contains | Marker |
|----------|----------|--------|
| `tests/smoke/` | Import-and-wiring checks — the cheapest signal that nothing is fundamentally broken. | `smoke` |
| `tests/unit/` | Isolated tests of one component, no external services. | `unit` |
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
- **`markers`** — a small, declared taxonomy (`smoke`, `unit`) so slices can be
  selected with `-m` and the marker set stays honest.

### Fixtures

[`conftest.py`](../tests/conftest.py) provides three small, dependency-free
fixtures, all built on pytest's `tmp_path` so nothing touches the real data tree:

| Fixture | Provides |
|---------|----------|
| `sample_dataframe` | A tiny, well-formed DataFrame with the binary `Outcome` target the stages expect. |
| `csv_path` | `sample_dataframe` written to a real CSV in a temp directory. |
| `params_file` | A representative `params.yaml` on disk, mirroring the real one. |

Anything requiring a network (MLflow, DagsHub) is intentionally **absent** from
the fixtures — those boundaries are left to smoke and integration testing rather
than reimplemented as mocks.

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
[roadmap](roadmap.md) and the open engineering-review findings.

### Near term — deepen unit coverage (Roadmap v2)

- **Decouple stage logic from IO to make it testable (review finding
  [H-6](reviews/sprint-02-engineering-review.md)).** Separate the pure logic
  (split, fit, score) from side effects (MLflow, disk) so `train` and `evaluate`
  can be unit-tested without a network. This is the structural unlock for the
  next item.
- **Stage-body unit tests.** Once decoupled: assert `preprocess` output
  shape/schema on a fixture, that `train` produces a fitted estimator on a tiny
  synthetic dataset, and that `evaluate` returns an accuracy in `[0, 1]`.
- **`logging_config` behaviour.** Idempotent handler attachment, `LOG_LEVEL` /
  `LOG_DIR` resolution, and the noisy-logger caps.

### Mid term — integration & CI (Roadmap v3)

- **End-to-end pipeline test.** A `dvc repro` run over a small synthetic dataset,
  asserting the produced artifacts — with MLflow pointed at a local/file backend
  or a lightweight double.
- **Continuous integration.** Run `pytest` (and `mypy`, linting) on every pull
  request, so the suite guards each change instead of running ad hoc. Publish a
  coverage report for *visibility* — still not as a hard gate unless the team
  later decides otherwise.
- **Test markers for tiering.** Introduce an `integration` marker so CI can run
  fast unit/smoke tests on every push and slower integration tests on a schedule.

### Longer term — as the project matures (Roadmap v4+)

- **Property-based tests** (e.g. Hypothesis) for the IO helpers' invariants.
- **Data validation / contract tests** for input schema drift.
- **Model quality gates** — regression thresholds on accuracy in CI before a
  model is promoted.

---

## 5. Adding a test — checklist

When you add or change behaviour:

1. **Decide the tier.** Wiring/import concern → `smoke`. Single component in
   isolation → `unit`. Mark it accordingly (`@pytest.mark.smoke` / `.unit`).
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
