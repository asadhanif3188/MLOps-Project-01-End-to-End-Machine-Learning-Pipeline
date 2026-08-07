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
- a metrics artifact (`metrics/metrics.json`, DVC-tracked),
- experiment metadata (parameters, metrics, artifacts) logged to a remote
  MLflow tracking server on DagsHub.

The stage inputs, outputs, artifact ownership, evaluation boundary, and
reproducibility expectations are specified in the
[Pipeline Contract](pipeline-contract.md) and enforced by the `contract` test
suite and CI (see [§3](#3-pipeline-flow) and [CI/CD](ci-cd.md)).

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
| Packaging | Multi-stage Docker image (non-root `runtime`) — [Containerization](containerization.md) |
| Local dev environment | Docker Compose (`dev` service) — [Docker Development](docker-development.md) |
| Continuous integration | GitHub Actions: lint, test, image build/validate — [CI/CD](ci-cd.md) |
| Serving | ❌ Not implemented — see [roadmap](roadmap.md) |

---

## 2. Component Diagram

<!-- TODO: Add the rendered component diagram under
     diagrams/system-architecture/ and embed it here once produced. -->

> 📌 **Diagram placeholder:** [`diagrams/system-architecture/`](diagrams/system-architecture/)

The major components and their relationships:

```text
        params.yaml ───────────────────────────┐ (parameters)
                                                ▼
data/raw/data.csv ─▶ [ preprocess ] ─▶ data/processed/data.csv
                                                │
                            ┌───────────────────┴───────────────┐
                            ▼                                    ▼
                       [ train ] ─▶ models/model.pkl ─▶ [ evaluate ] ─▶ metrics/metrics.json
                            │                                    │
                            ▼                                    ▼
                     MLflow (DagsHub) ◀── metrics, params, artifacts
```

**Components:**

- **`src/preprocess.py`** — reads the raw dataset and writes the processed CSV
  (with header) that `train` consumes.
- **`src/train.py`** — trains a seeded Random Forest on the processed dataset
  (`GridSearchCV` tunes leaf/split regularization; `n_estimators`/`max_depth`/
  `random_state` come from `params.yaml`), serializes the best estimator to
  `models/model.pkl`, and logs the run to MLflow.
- **`src/evaluate.py`** — loads the model, scores it, writes
  `metrics/metrics.json`, and logs the accuracy to MLflow.
- **`src/tracking.py`** — the single MLflow boundary: every experiment-tracking
  call goes through it, and the stages import it lazily so their ML computation
  stays free of MLflow and unit-testable without it.
- **DVC** — defines stage dependencies and outputs (`dvc.yaml`) and versions
  data/model artifacts to the remote.
- **MLflow / DagsHub** — remote experiment tracking and (optionally) model
  registry.

The stages share a small infrastructure layer (introduced in Sprint 2, extended
in Sprint 4 with the tracking boundary):

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
- **Output:** `data/processed/data.csv` (written **with** its header row).

### Stage 2 — `train`

- **Command:** `python src/train.py`
- **Inputs:** `data/processed/data.csv`, `src/train.py`, and params
  `train.input`, `train.output`, `train.target`, `train.random_state`,
  `train.n_estimators`, `train.max_depth`
- **Output:** `models/model.pkl`
- **Behavior:** consumes the **processed** dataset, splits it
  (`train_test_split`, 20% test, seeded by `train.random_state`), fits a
  `RandomForestClassifier` whose `n_estimators`/`max_depth`/`random_state` come
  from config, runs a small `GridSearchCV` (3-fold) tuning leaf/split
  regularization, logs parameters/metrics/artifacts to MLflow, conditionally
  registers the model, and pickles the best estimator. Seeded → deterministic
  given the same inputs and parameters.

### Stage 3 — `evaluate`

- **Command:** `python src/evaluate.py`
- **Inputs:** `data/processed/data.csv`, `models/model.pkl`, `src/evaluate.py`,
  and params `evaluate.data`, `evaluate.model`, `evaluate.target`,
  `evaluate.metrics`
- **Output:** `metrics/metrics.json` (DVC-tracked metrics artifact, `cache:
  false`); also logs the `accuracy` metric to MLflow.

> ℹ️ **Known limitation — in-sample evaluation.** `evaluate` currently scores the
> model over the **full** processed dataset — the same data `train` fits on — so
> the reported `accuracy` is an **in-sample** figure and must not be cited as a
> generalization estimate. Because the evaluation dataset is already an explicit,
> configured input, introducing a genuine held-out split is a
> configuration/graph change rather than a code change. This is deviation **D5**
> in the [Pipeline Contract](pipeline-contract.md#8-evaluation-boundary); it and
> the absence of a committed `dvc.lock` (D7) are the pipeline's remaining
> reproducibility limitations after Sprint 4.

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
| Containerization | Multi-stage Docker (`python:3.12-slim`) | [ADR-005](decisions/ADR-005-containerization-strategy.md) |
| Continuous integration | GitHub Actions | [CI/CD](ci-cd.md) |

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
3. **Training.** `train` reads the **processed** dataset, tunes and fits the
   model, and writes `models/model.pkl` (DVC-tracked, git-ignored on disk).
4. **Tracking.** During training and evaluation, parameters, metrics
   (`accuracy`), and artifacts (confusion matrix, classification report, model)
   are pushed to the remote MLflow server on DagsHub through the `tracking`
   boundary.
5. **Evaluation.** `evaluate` loads the pickled model, scores it on the
   processed dataset, writes `metrics/metrics.json`, and logs the accuracy
   metric.

**Data classification & storage:**

- Raw and processed data are **not** stored in Git; they are versioned by DVC and
  pushed to the remote.
- Models are git-ignored (`models/` in `.gitignore`) and versioned by DVC.
- Credentials live only in `.env` (see [`.env.example`](../.env.example)).

---

## 6. Containerization & Continuous Integration

Sprint 3 packaged the pipeline for reproducible execution and automated its
quality gates. Both are **implemented and in the repository today**.

### Containerization

- A single multi-stage [`Dockerfile`](../Dockerfile) builds three targets from
  one source of truth: `builder` (installs the dependency virtualenv),
  `development` (adds the Ruff/mypy/pytest/pre-commit toolchain), and `runtime`
  (the default — a lean, **non-root** production image on `python:3.12-slim`).
- A [`.dockerignore`](../.dockerignore) keeps the build context small and free of
  secrets, data, and history.
- State (`data/`, `models/`, `logs/`) is mounted as volumes and credentials are
  injected at run time — nothing sensitive is baked into the image.
- The full rationale is in the [Containerization Strategy](containerization.md)
  and [ADR-005](decisions/ADR-005-containerization-strategy.md).

### Local development environment

- A [`docker-compose.yml`](../docker-compose.yml) provides a `dev` service (the
  `development` image, working tree bind-mounted for live edits) and a
  profile-gated `pipeline` service that runs the production image's `dvc repro`.
  A contributor runs `docker compose up` and works entirely in-container. See the
  [Docker Development Workflow](docker-development.md).

### Continuous integration

- A [GitHub Actions workflow](../.github/workflows/ci.yml) validates every push to
  `main` and every pull request: checkout → Python 3.12 → install dependencies →
  Ruff (lint + format check) → pytest (including the `contract` tests) → **DVC
  pipeline integrity** (`dvc dag` + `dvc status`, offline, analytics disabled) →
  Docker build of the `runtime` image → build validation (non-root UID, core
  imports, `dvc` entrypoint). The DVC and contract checks (added in Sprint 4)
  fail a PR when the graph, parameter contract, or artifact lineage is broken —
  without contacting the DagsHub remote. It is validation-only — it does not
  deploy, push images, or use Kubernetes. See [CI/CD](ci-cd.md).

> 📌 **Diagram placeholders:**
> [`diagrams/cicd-flow/`](diagrams/cicd-flow/),
> [`diagrams/deployment-architecture/`](diagrams/deployment-architecture/)

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
- [Containerization Strategy](containerization.md)
- [Docker Development Workflow](docker-development.md)
- [CI/CD](ci-cd.md)
