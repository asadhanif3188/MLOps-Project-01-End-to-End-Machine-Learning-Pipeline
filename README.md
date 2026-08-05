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

- The preprocess.py script reads the raw dataset (data/raw/data.csv), performs basic preprocessing (such as renaming columns), and outputs the processed data to data/processed/data.csv.
- This stage ensures that data is consistently processed across runs.

### Training:

- The train.py script trains a Random Forest Classifier, tuning hyperparameters with a grid search, on the dataset (data/raw/data.csv).
- The model is saved as models/model.pkl.
- Hyperparameters and the model itself are logged into MLflow for tracking and comparison.

### Evaluation:

- The evaluate.py script loads the trained model and evaluates its performance (accuracy) on the dataset.
- The evaluation metrics are logged to MLflow for tracking.



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
  --build-arg BUILD_VERSION="1.2.0" \
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
3. **Pytest** suite
4. **Docker build** of the production image (built and validated — never pushed)

CI is validation only — it does not deploy, publish images, or use Kubernetes.
See [docs/ci-cd.md](docs/ci-cd.md) for the stages, failure strategy, and how to
reproduce each gate locally.

### For Adding DVC Stages

### Bash Commands
```
dvc stage add -n preprocess \
    -p preprocess.input,preprocess.output \
    -d src/preprocess.py -d data/raw/data.csv \
    -o data/processed/data.csv \
    python src/preprocess.py
```	
	
```
dvc stage add -n train \
    -p train.data,train.model,train.random_state,train.n_estimators,train.max_depth \
    -d src/train.py -d data/raw/data.csv \
    -o models/model.pkl \
    python src/train.py
```	

```
dvc stage add -n evaluate \
    -d src/evaluate.py -d models/model.pkl -d data/raw/data.csv \
    python src/evaluate.py
```

### Windows Commands
```
dvc stage add -n preprocess -p preprocess.input,preprocess.output -d src/preprocess.py -d data/raw/data.csv -o data/processed/data.csv python src/preprocess.py
```	
	
```
dvc stage add -n train -p train.data,train.model,train.random_state,train.n_estimators,train.max_depth -d src/train.py -d data/raw/data.csv -o models/model.pkl python src/train.py
```	

```
dvc stage add -n evaluate -d src/evaluate.py -d models/model.pkl -d data/raw/data.csv python src/evaluate.py
```