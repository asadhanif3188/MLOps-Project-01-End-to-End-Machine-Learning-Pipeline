# Design Principles

This document explains the reasoning behind the project's core design and
technology choices — the *why* behind the *what*. It complements two neighbours:

- [Engineering Philosophy](philosophy.md) — the values that guide how the
  repository is built.
- [Architecture Decision Records](decisions/) — the formal record of specific
  decisions, with alternatives and consequences.

Design principles sit between the two: they articulate the rationale for the
shape of the system in plain engineering terms. Where a choice is a deliberate
**baseline** rather than a benchmarked conclusion, that is stated honestly, and
the path to revisiting it is noted.

---

## Why a Batch Pipeline?

The system is designed as a **batch, file-based pipeline** because the problem is
a supervised training task over a static tabular dataset. Batch execution matches
the workload: data is read, transformed, used to train and evaluate a model, and
the resulting artifacts are versioned.

**Why not streaming?** Streaming solves a different problem — continuous,
low-latency processing of unbounded data. This project has no real-time data
source and no online-inference requirement today, so streaming would add
operational complexity (brokers, stateful processing, exactly-once semantics)
with no corresponding benefit. Online/streaming inference is a forward-looking
concern tied to the serving objectives in the [roadmap](roadmap.md) (v5–v6).

## Why Random Forest?

A **Random Forest classifier** is the current model because it is a strong,
low-friction baseline for tabular classification: it handles non-linear
relationships, is robust to unscaled features, resists overfitting through
ensembling, and exposes feature importances for interpretability. Paired with
`GridSearchCV`, it gives a defensible first result without heavy tuning.

**Why not XGBoost or CatBoost?** Gradient-boosted trees frequently outperform
Random Forests on tabular data and are strong candidates for a future iteration.
They are not used yet because the current goal is a clear, reproducible baseline
rather than leaderboard accuracy. Model selection has not been benchmarked on
this dataset.

> **TODO:** Benchmark Random Forest against gradient-boosted alternatives
> (XGBoost, CatBoost, LightGBM) and record the outcome as an ADR when a model
> decision is made.

## Why Python?

**Python** is the implementation language because it is the lingua franca of the
ML ecosystem: scikit-learn, MLflow, DVC, and the surrounding tooling are all
first-class in Python, which minimizes integration friction and maximizes
reproducibility across environments.

## Why DVC?

**DVC** versions data, models, and the pipeline graph. Git is excellent for code
but poor for large binaries and for expressing stage dependencies. DVC provides
content-addressed artifact versioning, a declarative stage graph (`dvc.yaml`),
and remote storage — so `dvc repro` re-runs only the stages whose inputs changed.
The full rationale and alternatives are in
[ADR-003](decisions/ADR-003-why-dvc.md).

## Why MLflow?

**MLflow** provides experiment tracking and an optional model registry. Training
produces many candidate models; MLflow records their parameters, metrics, and
artifacts so runs are comparable and results are not trapped on one machine.
Hosting it on DagsHub keeps tracking and DVC storage on one platform. The full
rationale and alternatives are in [ADR-002](decisions/ADR-002-why-mlflow.md).

## Why Modular Code?

The pipeline is split into **one module per stage** (`preprocess`, `train`,
`evaluate`). Each module has a single responsibility, can be run and reasoned
about independently, and maps 1:1 onto a DVC stage. This keeps the codebase easy
to navigate, makes failures easy to localize, and enables DVC to cache and skip
work at stage granularity. The structural decision is recorded in
[ADR-001](decisions/ADR-001-repository-structure.md).

## Why YAML Configuration?

Pipeline parameters live in **`params.yaml`**, separate from code. YAML is
human-readable, diff-friendly, and natively understood by DVC — which tracks
individual parameters as stage dependencies, so a parameter change alone can
trigger a re-run. Keeping configuration out of source means experiments vary by
editing data, not logic, which reinforces reproducibility.

## Why Centralized Logging?

All stages log through a **single configuration module**
(`src/logging_config.py`) built on the standard library's `logging` package,
replacing the ad-hoc `print()` calls of the baseline implementation. One module
means one format, one level policy, and one set of destinations (console plus a
rotating file) — and switching to `DEBUG` is an environment-variable change, not
a code change. The standard library was chosen over third-party loggers because
the pipeline's needs (levels, handlers, rotation) are fully met without adding a
dependency. Details: [Logging Strategy](logging.md).

## Why a Typed Exception Hierarchy?

Expected failures are instances of a **small typed hierarchy** rooted at
`PipelineError` (`ConfigError`, `DataError`, `ModelError`, `TrackingError`),
raised at centralized IO/config boundaries (`src/pipeline_io.py`) and handled
once at a uniform stage entry point (`src/stage_runner.py`). Types — not message
strings — say *what kind* of thing went wrong; messages say what to do next;
`raise ... from` preserves the cause; and a failed stage exits non-zero so
`dvc repro` and CI stop instead of continuing on bad state. Details:
[Exception Strategy](exception-strategy.md).

## Why Strict Static Typing?

Every public function in `src/` carries **complete type annotations**, checked
by a strict [mypy](https://mypy.readthedocs.io/) configuration with no
suppressions. Types make contracts visible to callers and IDEs and catch
mismatches before the pipeline runs. `Any` is confined to the two genuinely
dynamic boundaries (YAML parameters and unpickled artifacts), where the wider
type is a documented choice rather than missing rigor. Details:
[Type Safety](type-safety.md).

## Why a Contract-Focused Test Suite?

The `pytest` suite deliberately targets **contracts over coverage numbers**:
smoke tests pin import/wiring health, and unit tests go deep on the critical
IO/error layer where regressions are most expensive and most silent. Tests are
fast, isolated, and deterministic — no network, no MLflow, no real data tree.
Stage bodies coupled to MLflow are consciously deferred until they are
decoupled, rather than faked with mocks. Details:
[Testing Strategy](testing-strategy.md).

## Why Ruff (One Tool for Lint and Format)?

**Ruff** provides both the linter and the formatter, replacing the traditional
flake8 + isort + black stack with a single fast tool configured in one place
(`pyproject.toml`). Combined with mypy, pytest, and pre-commit hooks, this gives
the project **one source of truth per concern** — line length, lint rules, type
strictness, and test configuration each live in exactly one file, and every
entry point (CLI, Makefile, hooks, editor) defers to it. The decision and its
alternatives are recorded in
[ADR-004](decisions/ADR-004-python-quality-toolchain.md); day-to-day usage is in
the [Developer Guide](developer-guide.md).

---

## Summary

| Choice | Rationale | Reference |
|--------|-----------|-----------|
| Batch pipeline | Matches a static, supervised tabular workload | This document |
| Random Forest | Strong, interpretable tabular baseline | This document (benchmark pending) |
| Python | First-class ML ecosystem support | This document |
| DVC | Data/pipeline versioning and reproducibility | [ADR-003](decisions/ADR-003-why-dvc.md) |
| MLflow (DagsHub) | Experiment tracking and model registry | [ADR-002](decisions/ADR-002-why-mlflow.md) |
| Modular code | Single-responsibility stages, 1:1 with DVC | [ADR-001](decisions/ADR-001-repository-structure.md) |
| YAML config | Readable, diff-able, DVC-tracked parameters | This document |
| Centralized logging (stdlib) | One config, console + rotating file, env-controlled | [Logging Strategy](logging.md) |
| Typed exceptions | Failure kind by type; log once; fail loudly | [Exception Strategy](exception-strategy.md) |
| Strict typing (mypy) | Visible contracts, checked before runtime | [Type Safety](type-safety.md) |
| Contract-focused tests (pytest) | Signal over coverage percentage | [Testing Strategy](testing-strategy.md) |
| Ruff + pre-commit toolchain | One source of truth per concern | [ADR-004](decisions/ADR-004-python-quality-toolchain.md) |

---

## Related Documentation

- [Engineering Philosophy](philosophy.md)
- [Architecture](architecture.md)
- [Architecture Decision Records](decisions/)
- [Roadmap](roadmap.md)
