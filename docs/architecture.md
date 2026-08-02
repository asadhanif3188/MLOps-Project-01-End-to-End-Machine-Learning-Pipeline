# Architecture

The **End-to-End Machine Learning Pipeline** is architected around
reproducibility: a Random Forest training workflow orchestrated with
[DVC](https://dvc.org/) and instrumented with [MLflow](https://mlflow.org/)
(hosted on [DagsHub](https://dagshub.com/)). This document explains how the
pieces fit together.

> **Scope note.** This document reflects the repository as it exists today. Where
> a component is planned but not yet implemented, it is marked with an explicit
> `TODO`. Diagrams referenced here live under [`diagrams/`](diagrams/) and are
> placeholders until produced in a later sprint.

---

## 1. System Overview

The system is a **batch, file-based ML pipeline**. It transforms a raw CSV
dataset into a trained, serialized model while tracking experiments and
versioning data and artifacts.

There is no online serving component today; the pipeline runs on demand (locally
or via `dvc repro`) and produces:

- a preprocessed dataset (`data/processed/data.csv`),
- a serialized model (`models/model.pkl`),
- experiment metadata (parameters, metrics, artifacts) logged to a remote
  MLflow tracking server on DagsHub.

The dataset is the **Pima Indians Diabetes** dataset; the task is binary
classification of the `Outcome` column.

**Key properties:**

| Property | Current state |
|----------|---------------|
| Execution model | Batch, on-demand |
| Orchestration | DVC pipeline (`dvc.yaml`) |
| Experiment tracking | MLflow via DagsHub (remote) |
| Data/artifact versioning | DVC with S3-compatible remote (DagsHub) |
| Model type | `RandomForestClassifier` (scikit-learn) |
| Serving | ❌ Not implemented — see [roadmap](roadmap.md) |

---

## 2. Component Diagram

<!-- TODO: Add the rendered component diagram under
     diagrams/system-architecture/ and embed it here once produced. -->

> 📌 **Diagram placeholder:** [`diagrams/system-architecture/`](diagrams/system-architecture/)

The major components and their relationships:

```text
            params.yaml ─────────────┐ (parameters)
                                      ▼
data/raw/data.csv ──▶ [ preprocess ] ──▶ data/processed/data.csv
                                      │
                                      ▼
                        [ train ] ──▶ models/model.pkl ──▶ [ evaluate ]
                                      │                          │
                                      ▼                          ▼
                              MLflow (DagsHub) ◀── metrics, params, artifacts
```

**Components:**

- **`src/preprocess.py`** — reads the raw dataset and writes a processed CSV.
- **`src/train.py`** — trains a Random Forest with `GridSearchCV`, logs to
  MLflow, and serializes the best estimator to `models/model.pkl`.
- **`src/evaluate.py`** — loads the serialized model and logs an accuracy metric
  to MLflow.
- **DVC** — defines stage dependencies and outputs (`dvc.yaml`) and versions
  data/model artifacts to the remote.
- **MLflow / DagsHub** — remote experiment tracking and (optionally) model
  registry.

The stages share a small infrastructure layer (introduced in Sprint 2):

- **`src/logging_config.py`** — centralized logging configuration (console +
  rotating file); see [§7](#7-observability--logging).
- **`src/exceptions.py`** — the typed exception hierarchy rooted at
  `PipelineError`.
- **`src/pipeline_io.py`** — IO/config/serialization helpers that wrap every
  filesystem and config boundary with consistent, typed error handling.
- **`src/stage_runner.py`** — the uniform stage entry point: logs any failure
  once (with traceback) and exits non-zero.

---

## 3. Pipeline Flow

The pipeline is defined in [`dvc.yaml`](../dvc.yaml) as three stages executed in
dependency order.

> 📌 **Diagram placeholder:** [`diagrams/pipeline-flow/`](diagrams/pipeline-flow/)

### Stage 1 — `preprocess`

- **Command:** `python src/preprocess.py`
- **Inputs:** `data/raw/data.csv`, `src/preprocess.py`,
  params `preprocess.input`, `preprocess.output`
- **Output:** `data/processed/data.csv`
- **Behavior:** reads the raw CSV and re-writes it (without header/index).

### Stage 2 — `train`

- **Command:** `python src/train.py`
- **Inputs:** `data/raw/data.csv`, `src/train.py`, and params
  `train.data`, `train.model`, `train.random_state`, `train.n_estimators`,
  `train.max_depth`
- **Output:** `models/model.pkl`
- **Behavior:** splits data (`train_test_split`, 20% test), runs `GridSearchCV`
  (3-fold) over a hyperparameter grid, logs parameters/metrics/artifacts to
  MLflow, optionally registers the model, and pickles the best estimator.

### Stage 3 — `evaluate`

- **Command:** `python src/evaluate.py`
- **Inputs:** `data/raw/data.csv`, `models/model.pkl`, `src/evaluate.py`
- **Output:** none tracked by DVC; logs an `accuracy` metric to MLflow.

> ⚠️ **TODO / known gaps to resolve later (documented, not fixed here):**
>
> 1. **Preprocess output is unused downstream.** The `train` and `evaluate`
>    stages both depend on `data/raw/data.csv`, and `train.py` reads
>    `params['train']['input']` (the raw file). The `data/processed/data.csv`
>    produced by `preprocess` is therefore not consumed.
> 2. **Param name mismatch.** `dvc.yaml` references `train.data` and
>    `train.model`, but `params.yaml` defines `train.input` and `train.output`.
>    `train.py` reads `input`/`output`.
> 3. **Evaluation on training data.** `evaluate.py` computes accuracy over the
>    full dataset rather than a held-out split.
>
> These are engineering concerns for a future sprint and are recorded here only
> to keep the architecture description faithful.

---

## 4. Technology Choices

| Concern | Technology | Rationale (see ADR) |
|---------|-----------|---------------------|
| Pipeline orchestration & data versioning | DVC (+ `dvc-s3`) | [ADR-003](decisions/ADR-003-why-dvc.md) |
| Experiment tracking / model registry | MLflow via DagsHub | [ADR-002](decisions/ADR-002-why-mlflow.md) |
| Repository structure | Stage-per-script `src/` layout | [ADR-001](decisions/ADR-001-repository-structure.md) |
| Model | `RandomForestClassifier` (scikit-learn) | Baseline for tabular classification |
| Config | `params.yaml` (declarative parameters) | Separates config from code |
| Secrets | `python-dotenv` + `.env` | Keeps credentials out of source |
| Logging | Python `logging` (stdlib), centralized config | [Logging Strategy](logging.md) |
| Error handling | Typed exception hierarchy (`PipelineError`) | [Exception Strategy](exception-strategy.md) |
| Lint/format/type/test toolchain | Ruff + mypy + pytest + pre-commit | [ADR-004](decisions/ADR-004-python-quality-toolchain.md) |

For the reasoning behind these choices, see [Design Principles](design-principles.md).
Full dependency lists: [`requirements.txt`](../requirements.txt) (runtime) and
[`requirements-dev.txt`](../requirements-dev.txt) (development tooling).

---

## 5. Data Flow

> 📌 **Diagram placeholder:** [`diagrams/pipeline-flow/`](diagrams/pipeline-flow/)

1. **Ingestion.** The raw dataset is tracked by DVC
   (`data/raw/data.csv.dvc`) and retrieved from the S3-compatible DagsHub remote
   via `dvc pull`.
2. **Preprocessing.** `preprocess` reads the raw CSV and writes
   `data/processed/data.csv` (a DVC-tracked output).
3. **Training.** `train` reads the dataset, tunes and fits the model, and writes
   `models/model.pkl` (DVC-tracked, git-ignored on disk).
4. **Tracking.** During training and evaluation, parameters, metrics
   (`accuracy`), and artifacts (confusion matrix, classification report, model)
   are pushed to the remote MLflow server on DagsHub.
5. **Evaluation.** `evaluate` loads the pickled model and logs an accuracy
   metric.

**Data classification & storage:**

- Raw and processed data are **not** stored in Git; they are versioned by DVC and
  pushed to the remote.
- Models are git-ignored (`models/` in `.gitignore`) and versioned by DVC.
- Credentials live only in `.env` (see [`.env.example`](../.env.example)).

---

## 6. Future Cloud Architecture

> This section describes the **target** state. None of it is implemented today;
> each item is a forward-looking objective aligned with the
> [roadmap](roadmap.md).

- **CI/CD (Roadmap v3).** Automated lint/test and `dvc repro` on pull requests.
  <!-- TODO: define CI provider and pipeline once chosen. -->
- **Containerization & orchestration (Roadmap v4).** Package the pipeline as a
  container and run stages on Kubernetes. The container design is settled in the
  [Containerization Strategy](containerization.md) and
  [ADR-005](decisions/ADR-005-containerization-strategy.md); implementation is a
  later sprint.
  <!-- TODO: implement the Dockerfile per ADR-005; define orchestration approach. -->
- **Cloud deployment (Roadmap v5).** Managed object storage for the DVC remote,
  a hosted MLflow tracking server, and a model-serving endpoint.
  <!-- TODO: select cloud provider and serving mechanism. -->
- **Enterprise MLOps (Roadmap v6).** Monitoring, drift detection, automated
  retraining, and governance.
  <!-- TODO: define monitoring and retraining triggers. -->

> 📌 **Diagram placeholders:**
> [`diagrams/deployment-architecture/`](diagrams/deployment-architecture/),
> [`diagrams/kubernetes-architecture/`](diagrams/kubernetes-architecture/),
> [`diagrams/cicd-flow/`](diagrams/cicd-flow/)

---

## 7. Observability & Logging

All pipeline stages emit structured logs through a single, centralized
configuration in [`src/logging_config.py`](../src/logging_config.py). This
replaces the ad-hoc `print()` statements previously used for diagnostics
(engineering review finding **H-1**).

- **Configuration.** `configure_logging()` sets up console **and** rotating-file
  handlers on the root logger at each stage's entry point; `get_logger(name)`
  returns a per-stage logger.
- **Destinations.** Logs stream to the console and persist to
  `logs/pipeline.log` (git-ignored, rotating at 5 MB × 3 backups).
- **Levels.** Controlled by the `LOG_LEVEL` environment variable (default
  `INFO`): `INFO` for lifecycle, `WARNING` for recoverable issues, `ERROR` for
  failures, `DEBUG` for development only.
- **Lifecycle logs.** Each stage emits deliberate start/completion logs plus key
  outcomes (e.g. best model accuracy, model saved) — enough to trace a run
  without over-logging.

For the full policy — format, level guidance, and per-stage log inventory — see
[Logging Strategy](logging.md).

### Error handling

Failures are handled through a small typed exception hierarchy
([`src/exceptions.py`](../src/exceptions.py)) rooted at `PipelineError`, with
IO/config/model boundaries centralized in
[`src/pipeline_io.py`](../src/pipeline_io.py) and a uniform stage entry point
([`src/stage_runner.py`](../src/stage_runner.py)) that logs each failure once —
with the full traceback — and exits non-zero so `dvc repro` and CI stop on error
(engineering review finding **H-2**). See [Exception Strategy](exception-strategy.md)
for the hierarchy, propagation rules, and user-facing error contract.

---

## Related Documentation

- [Project Structure](project-structure.md)
- [Roadmap](roadmap.md)
- [Engineering Philosophy](philosophy.md)
- [Architecture Decision Records](decisions/)
- [Logging Strategy](logging.md)
- [Exception Strategy](exception-strategy.md)
- [Type Safety](type-safety.md)
- [Testing Strategy](testing-strategy.md)
- [Developer Guide](developer-guide.md)
