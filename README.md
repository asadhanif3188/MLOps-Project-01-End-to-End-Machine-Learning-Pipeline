# MLOps Project 01 - End-to-End Machine Learning Pipeline 

[![CI](https://github.com/asadhanif3188/MLOps-Project-01-End-to-End-Machine-Learning-Pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/asadhanif3188/MLOps-Project-01-End-to-End-Machine-Learning-Pipeline/actions/workflows/ci.yml)

This project demonstrates how to build an end-to-end machine learning pipeline using DVC (Data Version Control) for data and model versioning, and MLflow for experiment tracking. 

The pipeline focuses on training a Random Forest Classifier on the Pima Indians Diabetes Dataset, with clear stages for data preprocessing, model training, and evaluation.

## Tools/Technologies used for Project
Following tools have been used to complete the project. 
1. Git / GitHub / GitLab
2. DagsHub
3. DVC
4. MLFlow

### Data Version Control (DVC):
- DVC is used to track and version the dataset, models, and pipeline stages, ensuring reproducibility across different environments.
- The pipeline is structured into stages (**preprocessing**, **training**, **evaluation**) that can be automatically re-executed if any dependencies change (e.g., data, scripts, or parameters).
- DVC also allows remote data storage (e.g., DagsHub, S3) for large datasets and models.

### Experiment Tracking with MLflow:
- MLflow is used to track experiment metrics, parameters, and artifacts.
- It logs the hyperparameters of the model (e.g., n_estimators, max_depth) and performance metrics like accuracy.
- MLflow helps compare different runs and models to optimize the machine learning pipeline.

## Pipeline Stages

### Preprocessing:

- `src/preprocess.py` reads the raw dataset (`data/raw/data.csv`) and writes the processed dataset to `data/processed/data.csv`, preserving the header row so downstream stages can select the target and feature columns by name.
- It is the single owner of the processed dataset; `split` consumes that output directly (`preprocess` is the only stage that reads the raw file).

### Splitting:

- `src/split.py` partitions the processed dataset into a training set (`data/processed/train.csv`) and a **held-out** evaluation set (`data/processed/test.csv`) with a stratified, seeded `train_test_split`.
- It is the single owner of both partitions and asserts they are **disjoint** (no row is used for both training and evaluation) and **exhaustive** (no row is lost). `split.random_state` makes the exact held-out rows reproducible. This is what makes evaluation genuinely out-of-sample (see [ADR-007](docs/decisions/ADR-007-held-out-evaluation.md)).

### Training:

- `src/train.py` trains a Random Forest Classifier on the **training** partition (`data/processed/train.csv`), never the held-out set.
- `n_estimators`, `max_depth`, and `random_state` are read from `params.yaml` and applied to the estimator; a small `GridSearchCV` then tunes the leaf/split regularization. Its internal validation split (taken within the training set) and the estimator are seeded, so training is deterministic given the same inputs and parameters.
- The best model is saved to `models/model.pkl`, and hyperparameters, metrics, and artifacts are logged to MLflow.

### Evaluation:

- `src/evaluate.py` loads the trained model, scores it on the **held-out** partition (`data/processed/test.csv`), writes the accuracy to the DVC-tracked metrics artifact `metrics/metrics.json`, and logs the metric to MLflow.
- **Evaluation boundary:** because `train` fits on `train.csv` and `evaluate` scores on the disjoint `test.csv`, the reported accuracy is a genuine **out-of-sample** figure (deviation D5 resolved). The remaining caveats are quality refinements — a single split rather than cross-validation, and a small held-out set — documented in the [pipeline contract](docs/pipeline-contract.md#8-evaluation-boundary).



## Goals
- **Reproducibility:** By using DVC, the pipeline ensures that the same data, parameters, and code can reproduce the same results, making the workflow reliable and consistent.
- **Experimentation:** MLflow allows users to easily track different experiments (with varying hyperparameters) and compare the performance of models.
- **Collaboration:** DVC and MLflow enable smooth collaboration in a team environment, where different users can work on the same project and track changes seamlessly.

## Running with Docker

The pipeline ships a production-grade, multi-stage [`Dockerfile`](Dockerfile) that
runs the same environment on any machine. The image is **non-root**, built on
`python:3.12-slim`, and keeps data, models, and credentials **out** of the image —
they are mounted and injected at run time.

**Build the image:**
```bash
docker build \
  --build-arg VCS_REF="$(git rev-parse --short HEAD)" \
  --build-arg BUILD_VERSION="1.3.1" \
  -t ml-pipeline:local .
```

**Run the pipeline** (mount state, supply credentials via `.env` — see
[`.env.example`](.env.example)):
```bash
docker run --rm \
  --env-file .env \
  -v "$(pwd)/data":/app/data \
  -v "$(pwd)/models":/app/models \
  -v "$(pwd)/logs":/app/logs \
  ml-pipeline:local
```

Override the command to run a single stage, e.g. `... ml-pipeline:local python src/preprocess.py`.
A development image with the full lint/type/test toolchain is available via
`docker build --target development -t ml-pipeline:dev .`.

**Local development with Docker Compose** — a new contributor needs only Docker:

```bash
cp .env.example .env          # add MLflow / DagsHub credentials
docker compose up -d          # build + start the dev environment
docker compose exec dev bash  # shell in; run `make check`, `dvc repro`, ...
```

Your working tree is bind-mounted, so host edits are live inside the container.
Run the production pipeline image on demand with
`docker compose --profile pipeline run --rm pipeline`.

> Full build/run instructions (including hardened, read-only execution), the
> design rationale, and the decision record are in
> [docs/containerization.md](docs/containerization.md) and
> [ADR-005](docs/decisions/ADR-005-containerization-strategy.md). The day-to-day
> Compose workflow is documented in
> [docs/docker-development.md](docs/docker-development.md).

## Continuous Integration

Every push to `main` and every pull request is validated by a GitHub Actions
pipeline ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)). It runs the
same quality gates you run locally and then builds and validates the container
image:

1. Checkout → set up Python 3.12 → install dependencies
2. **Ruff** lint + format check
3. **mypy** strict type check (the same `[tool.mypy]` gate enforced locally and by pre-commit)
4. **Pytest** suite (smoke, unit, integration, contract)
5. **DVC pipeline integrity** — parse the graph and check status, offline
6. **Fixture pipeline reproduction** — a real `dvc repro` with byte-identical artifact validation
7. **Docker build** of the production image (built and validated — never pushed)

CI is validation only — it does not deploy, publish images, or use Kubernetes.
See [docs/ci-cd.md](docs/ci-cd.md) for the stages, failure strategy, and how to
reproduce each gate locally.

## Kubernetes

The pipeline is a **finite batch workload**, so its Kubernetes model is a
run-to-completion **`Job`** (in a dedicated `mlops` namespace) — not a
`Deployment`, and with no fake HTTP API invented to justify a `Service`. The
manifests live under [`k8s/`](k8s/) as a Kustomize `base/` + `overlays/local/`
layout. The `Job` runs the real `dvc repro` command on the locally built
`ml-pipeline:local` image with a finite-run lifecycle (`restartPolicy: Never`,
`backoffLimit: 2`, `activeDeadlineSeconds: 1800`). Render with:

```bash
kustomize build k8s/overlays/local   # or: kubectl kustomize k8s/overlays/local
```

A step-by-step local run (build → side-load → apply → inspect → logs → re-run) is
in [`k8s/README.md`](k8s/README.md).

> **Status:** namespace, workload model, and the **runnable** Job + local runbook
> are in place (Sprint 5 PR 1–2), and PR 3 adds externalized **configuration** (a
> `ConfigMap`: `LOG_LEVEL`, `MLFLOW_TRACKING_URI`), a **Secret** template for the
> DagsHub credentials (created out-of-band — never committed), and a
> least-privilege **ServiceAccount** with the API-token automount off (the workload
> needs no cluster API access). The Job was **executed on a local Docker Desktop
> cluster** (2026-08-12) and its lifecycle verified end to end (3 attempts →
> terminal `Failed`); the pipeline does **not** complete yet — `dvc repro` aborts
> with `/app is not a git repository`, so a *green* in-cluster run still needs an
> SCM in the image + mounted data. Security hardening, CPU/memory limits, and CI
> validation are deferred to later PRs. Rationale:
> [Kubernetes Architecture](docs/kubernetes-architecture.md) and
> [ADR-009](docs/decisions/ADR-009-kubernetes-workload-model.md); run details:
> [`k8s/README.md`](k8s/README.md).

### How the DVC Stages Are Defined

The pipeline graph lives in [`dvc.yaml`](dvc.yaml); the commands below reproduce
it from scratch and match the corrected lineage
`raw → preprocess → processed → split → {train.csv → train → model, test.csv} → evaluate → metrics`.
Note that `split` owns the two partitions, `train` depends on the **training**
partition and `evaluate` on the disjoint **held-out** partition (never the raw
file), and `evaluate` declares `metrics/metrics.json` as an uncached metrics output.

### Bash Commands
```
dvc stage add -n preprocess \
    -p preprocess.input,preprocess.output \
    -d src/preprocess.py -d data/raw/data.csv \
    -o data/processed/data.csv \
    python src/preprocess.py
```	
	
```
dvc stage add -n split \
    -p split.input,split.train_output,split.test_output,split.target,split.test_size,split.random_state \
    -d src/split.py -d data/processed/data.csv \
    -o data/processed/train.csv -o data/processed/test.csv \
    python src/split.py
```	

```
dvc stage add -n train \
    -p train.input,train.output,train.target,train.random_state,train.n_estimators,train.max_depth \
    -d src/train.py -d data/processed/train.csv \
    -o models/model.pkl \
    python src/train.py
```	

```
dvc stage add -n evaluate \
    -p evaluate.data,evaluate.model,evaluate.target,evaluate.metrics \
    -d src/evaluate.py -d models/model.pkl -d data/processed/test.csv \
    -M metrics/metrics.json \
    python src/evaluate.py
```

### Windows Commands
```
dvc stage add -n preprocess -p preprocess.input,preprocess.output -d src/preprocess.py -d data/raw/data.csv -o data/processed/data.csv python src/preprocess.py
```	
	
```
dvc stage add -n split -p split.input,split.train_output,split.test_output,split.target,split.test_size,split.random_state -d src/split.py -d data/processed/data.csv -o data/processed/train.csv -o data/processed/test.csv python src/split.py
```	

```
dvc stage add -n train -p train.input,train.output,train.target,train.random_state,train.n_estimators,train.max_depth -d src/train.py -d data/processed/train.csv -o models/model.pkl python src/train.py
```	

```
dvc stage add -n evaluate -p evaluate.data,evaluate.model,evaluate.target,evaluate.metrics -d src/evaluate.py -d models/model.pkl -d data/processed/test.csv -M metrics/metrics.json python src/evaluate.py
```