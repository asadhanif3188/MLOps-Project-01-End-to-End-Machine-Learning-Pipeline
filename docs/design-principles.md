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

---

## Related Documentation

- [Engineering Philosophy](philosophy.md)
- [Architecture](architecture.md)
- [Architecture Decision Records](decisions/)
- [Roadmap](roadmap.md)
