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

> ⚠️ **TODO / known gap:** The docstring in `preprocess.py` states that the
> `Unnamed: 0` column is dropped, but the current implementation does not perform
> that transformation. Reconcile the code and its documentation in a future
> engineering sprint (out of scope for documentation-only work).

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

For the reasoning behind these choices, see [Design Principles](design-principles.md).
Full dependency list: [`requirements.txt`](../requirements.txt).

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
  container and run stages on Kubernetes.
  <!-- TODO: define container/base image and orchestration approach. -->
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

## Related Documentation

- [Project Structure](project-structure.md)
- [Roadmap](roadmap.md)
- [Engineering Philosophy](philosophy.md)
- [Architecture Decision Records](decisions/)
