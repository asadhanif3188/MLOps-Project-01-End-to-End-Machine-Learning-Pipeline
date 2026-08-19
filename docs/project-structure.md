# Project Structure

This document explains how the repository is organized so that a new contributor
can understand it within about ten minutes. It reflects the repository as it
exists today; planned-but-absent items are marked with `TODO`.

For the reasoning behind this layout, see
[ADR-001: Repository Structure](decisions/ADR-001-repository-structure.md).

---

## Top-Level Layout

```text
.
├── src/                  # Pipeline code
│   ├── preprocess.py     #   stage: raw CSV → processed CSV (headered)
│   ├── split.py          #   stage: processed CSV → train.csv + held-out test.csv
│   ├── train.py          #   stage: seeded train + tune, persist model, track
│   ├── evaluate.py       #   stage: score model on held-out test.csv → metrics.json, track
│   ├── tracking.py       #   MLflow experiment-tracking boundary (lazy-imported)
│   ├── logging_config.py #   centralized logging configuration
│   ├── exceptions.py     #   typed exception hierarchy (PipelineError, …)
│   ├── pipeline_io.py    #   IO/config/serialization helpers (typed errors)
│   └── stage_runner.py   #   uniform stage entry point (log once, exit non-zero)
├── tests/                # Pytest suite (smoke · unit · integration · contract)
│   ├── conftest.py       #   shared fixtures (incl. MLflow stub)
│   ├── smoke/            #   fast import/wiring checks
│   ├── unit/             #   isolated component + stage-compute tests
│   ├── integration/      #   full preprocess→train→evaluate run (MLflow stubbed)
│   └── contract/         #   static dvc.yaml/params.yaml/src consistency checks
├── data/                 # Datasets (DVC-tracked; contents not stored in Git)
│   ├── raw/              #   raw input data (data.csv, tracked via data.csv.dvc)
│   └── processed/        #   preprocessed output (consumed by train/evaluate)
├── metrics/              # Metrics artifact (metrics.json — evaluate output)
├── models/               # Serialized model artifacts (DVC-tracked, git-ignored)
├── logs/                 # Rotating pipeline logs (git-ignored)
├── docs/                 # Project documentation (see docs/README.md portal)
│   ├── README.md         #   documentation index
│   ├── architecture.md
│   ├── pipeline-contract.md #  stage contracts, artifact ownership, eval boundary
│   ├── roadmap.md
│   ├── project-structure.md
│   ├── design-principles.md
│   ├── philosophy.md
│   ├── logging.md        #   logging strategy
│   ├── exception-strategy.md
│   ├── type-safety.md
│   ├── testing-strategy.md
│   ├── developer-guide.md
│   ├── containerization.md #   container image design (ADR-005)
│   ├── docker-development.md #  local Docker Compose dev workflow
│   ├── ci-cd.md            #   CI pipeline + CD roadmap
│   ├── github-workflow.md
│   ├── versioning.md
│   ├── release-checklist.md
│   ├── repository-metadata.md
│   ├── decisions/        #   Architecture Decision Records (ADRs)
│   ├── reviews/          #   engineering & release-readiness reviews
│   ├── retrospectives/   #   per-sprint retrospectives
│   ├── proof/            #   sprint proof-impact assessments
│   ├── diagrams/         #   diagram placeholders (by category)
│   └── screenshots/      #   screenshot placeholders (by category)
├── .github/              # GitHub config
│   ├── workflows/        #   CI pipeline (ci.yml — lint, test, image build)
│   ├── ISSUE_TEMPLATE/
│   └── pull_request_template.md
├── .vscode/              # VS Code workspace settings & recommended extensions
├── .dvc/                 # DVC internal config (remote definition)
├── Dockerfile            # Multi-stage image (builder/development/runtime)
├── .dockerignore         # Files excluded from the Docker build context
├── docker-compose.yml    # Local dev workflow (dev + pipeline services)
├── dvc.yaml              # DVC pipeline definition (stages, deps, params, outs)
├── params.yaml           # Declarative pipeline parameters
├── pyproject.toml        # Tool configuration (ruff, mypy, pytest)
├── .pre-commit-config.yaml # Git hook definitions (ruff, hygiene, mypy, tests)
├── Makefile              # Developer command shortcuts (`make help`)
├── requirements.txt      # Runtime Python dependencies
├── requirements-dev.txt  # Development dependencies (ruff, mypy, pytest, pre-commit)
├── .env.example          # Template for required environment variables
├── .editorconfig         # Editor/formatting conventions
├── CHANGELOG.md          # Notable changes (Keep a Changelog format)
├── CONTRIBUTING.md       # Contribution guidelines
├── CODE_OF_CONDUCT.md    # Contributor Covenant
├── SECURITY.md           # Security policy
├── SUPPORT.md            # How to get help
├── LICENSE               # MIT License
└── README.md             # Project overview and quick start
```

---

## Directory & File Responsibilities

### `src/` — Pipeline logic
One Python module per pipeline stage, each runnable directly and mapped 1:1 to a
DVC stage:

- **`preprocess.py`** — reads the raw dataset and writes the processed CSV (with
  header). Reads its paths from the `preprocess` section of `params.yaml`.
- **`split.py`** — partitions the processed dataset into `data/processed/train.csv`
  and a **held-out** `data/processed/test.csv` with a stratified, seeded
  `train_test_split`, asserting the two are disjoint and exhaustive. Its pure
  computation (`split_dataset`) is IO-free and unit-testable. Reads its paths and
  seed from the `split` section of `params.yaml`.
- **`train.py`** — consumes the **training** partition (`train.csv`, never the
  held-out set) and trains a seeded `RandomForestClassifier` (configured
  `n_estimators`/`max_depth`/`random_state` plus a small `GridSearchCV`),
  serializes the best model to `models/model.pkl`, and logs the run to MLflow. Its
  ML computation (`run_training`) is separated from IO and tracking so it is
  unit-testable without a tracking server.
- **`evaluate.py`** — loads the model, scores it on the **held-out** partition
  (`test.csv`), writes the metrics artifact `metrics/metrics.json`, and logs the
  accuracy to MLflow. Its scoring (`compute_metrics`) is likewise MLflow-free.
- **`tracking.py`** — the single MLflow experiment-tracking boundary. The stages
  import it lazily, so importing (or unit-testing) a stage needs neither MLflow
  nor credentials.

The stages share the infrastructure modules (added in Sprint 2; the tracking
boundary added in Sprint 4):

- **`logging_config.py`** — the single source of truth for logging: console +
  rotating-file handlers, `LOG_LEVEL`/`LOG_DIR` environment control. See
  [Logging Strategy](logging.md).
- **`exceptions.py`** — the typed exception hierarchy (`PipelineError` →
  `ConfigError`, `DataError`, `ModelError`, `TrackingError`). See
  [Exception Strategy](exception-strategy.md).
- **`pipeline_io.py`** — IO, configuration, and serialization helpers; every
  filesystem/config boundary goes through here so failures surface as the typed
  exceptions with actionable messages.
- **`stage_runner.py`** — the uniform `run_stage(...)` entry point: logs any
  failure once (with the full traceback) and exits non-zero so `dvc repro` and
  CI stop on error.

All modules carry complete type annotations checked by a strict mypy
configuration (see [Type Safety](type-safety.md)).

> The pipeline-correctness gaps this section previously flagged (raw data
> consumed instead of the `preprocess` output, `dvc.yaml`/`params.yaml` parameter
> mismatch, and in-sample evaluation) were resolved in Sprint 4 and the subsequent
> held-out evaluation milestone — a dedicated `split` stage now feeds `train` and
> `evaluate` disjoint partitions (see
> [pipeline-contract §8](pipeline-contract.md#8-evaluation-boundary)). The wiring is
> enforced by the `contract` tests below and by CI. A committed `dvc.lock` and a
> real in-CI `dvc repro` now exist too — proven via a self-contained fixture
> pipeline ([ADR-008](decisions/ADR-008-fixture-reproducibility.md)). The remaining
> limitation is documented, not a gap: reproducing the *production* run end to end
> needs the remote-only dataset, live MLflow, and digest-pinned deps
> ([pipeline-contract §7](pipeline-contract.md#7-reproducibility-expectations),
> level 4).

### `tests/` — Automated tests
A `pytest` suite in four tiers (see [Testing Strategy](testing-strategy.md)):

- **`conftest.py`** — shared fixtures, including `stub_tracking`, which swaps the
  lazily-imported `tracking` module for an in-memory recorder so stage tests run
  without importing MLflow or touching the network.
- **`smoke/`** — fast import/wiring checks across all of `src/`.
- **`unit/`** — isolated tests of the infrastructure modules (`exceptions.py`,
  `pipeline_io.py`, `stage_runner.py`) **and** the stages' pure ML-compute
  functions (`preprocess`, `split`, `train`, `evaluate`), with no network or
  external services.
- **`integration/`** — an end-to-end `preprocess → split → train → evaluate` run through
  real temp files with MLflow stubbed, proving each stage's output is consumable
  by the next, plus a reproducibility test proving the seeded stages reproduce
  equivalent outputs on the committed fixture dataset.
- **`contract/`** — static checks that `dvc.yaml`, `params.yaml`, and `src/`
  agree with the [Pipeline Contract](pipeline-contract.md) (parameter
  consistency, single-owner artifacts, declared lineage, acyclic graph), and that
  the committed fixture `dvc.lock` is not structurally stale. Pure parsing — no
  data, network, or credentials.
- **`fixtures/pipeline/`** — a self-contained fixture DVC pipeline (committed
  dataset + `params.yaml` + `dvc.yaml` + `dvc.lock`) that CI reproduces with a real
  `dvc repro` to prove reproducible execution offline
  ([ADR-008](decisions/ADR-008-fixture-reproducibility.md)).

Run with `make test` (or `python -m pytest`); markers `smoke`, `unit`,
`integration`, and `contract` select slices of the suite.

### `data/` — Datasets (DVC-tracked)
- **`raw/`** — original input data. `data.csv` is versioned via
  `data.csv.dvc`; the actual file is pulled from the DVC remote, not committed.
- **`processed/`** — output of the `preprocess` stage.
- Contents are excluded from Git and managed by DVC.

### `models/` — Model artifacts
Serialized model(s), e.g. `model.pkl`. Git-ignored (`models/` in `.gitignore`)
and versioned by DVC.

### `metrics/` — Metrics artifact
The `evaluate` stage's `metrics/metrics.json`, declared in `dvc.yaml` as an
uncached DVC metric (`cache: false`) so the file itself is committed and diffable
while DVC still tracks it as a stage output.

### `docs/` — Documentation
- **`README.md`** — the documentation index / portal.
- **`architecture.md`** — system architecture and data flow.
- **`pipeline-contract.md`** — stage inputs/outputs, artifact ownership, the
  evaluation boundary, and reproducibility expectations (Sprint 4).
- **`roadmap.md`** — versioned milestones.
- **`project-structure.md`** — this document.
- **`design-principles.md`** — rationale behind core design and technology choices.
- **`philosophy.md`** — engineering principles.
- **`logging.md`**, **`exception-strategy.md`**, **`type-safety.md`**,
  **`testing-strategy.md`**, **`developer-guide.md`** — engineering strategy
  references introduced in Sprint 2 (observability, error handling, typing,
  testing, and day-to-day tooling).
- **`containerization.md`**, **`docker-development.md`**, **`ci-cd.md`** —
  Sprint 3 references: the container image design, the local Docker Compose
  development workflow, and the CI pipeline.
- **`github-workflow.md`**, **`versioning.md`**, **`release-checklist.md`** —
  process and governance (branching, SemVer, releases).
- **`repository-metadata.md`** — recommended repository description and topics.
- **`decisions/`** — Architecture Decision Records (ADR-001..006 + index).
- **`reviews/`** — engineering and release-readiness reviews (the Sprint 2
  production-readiness review plus the per-sprint final validations).
- **`retrospectives/`** — per-sprint look-backs (planned vs delivered,
  decisions, lessons, deferred work).
- **`proof/`** — sprint proof-impact assessments (what the project can
  credibly claim after each sprint, evidence-based).
- **`diagrams/`** — placeholder subfolders for diagrams (system architecture,
  pipeline flow, deployment, Kubernetes, CI/CD).
- **`screenshots/`** — placeholder subfolders for screenshots (MLflow UI, DVC
  pipeline, project execution, folder structure, training logs, Docker build,
  CI pipeline).

### `.github/` — GitHub configuration
Issue templates (`bug_report.md`, `feature_request.md`, `documentation.md`) and
the pull request template used to standardize contributions, plus
**`workflows/ci.yml`** — the GitHub Actions continuous-integration pipeline
(lint, test, and container image build/validation). See [CI/CD](ci-cd.md).

### Containerization & CI
- **`Dockerfile`** — a single multi-stage build with three targets: `builder`
  (installs the dependency virtualenv), `development` (adds the quality
  toolchain), and `runtime` (the default — a lean, non-root production image).
  See [Containerization Strategy](containerization.md) and
  [ADR-005](decisions/ADR-005-containerization-strategy.md).
- **`.dockerignore`** — keeps the build context small and free of secrets, data,
  and history.
- **`docker-compose.yml`** — the local development workflow: a `dev` service
  (working tree bind-mounted for live edits) and a profile-gated `pipeline`
  service that runs the production image. See
  [Docker Development](docker-development.md).

### Configuration & pipeline definition
- **`dvc.yaml`** — the pipeline graph: stages with their dependencies,
  parameters, and outputs.
- **`params.yaml`** — declarative parameters consumed by the stage scripts,
  keeping configuration out of code.
- **`.dvc/config`** — defines the S3-compatible DagsHub remote.

### Environment & tooling
- **`requirements.txt`** — runtime Python dependencies (`dvc`, `scikit-learn`,
  `mlflow`, `dvc-s3`, `boto3`, `python-dotenv`).
- **`requirements-dev.txt`** — development dependencies (`ruff`, `mypy`,
  `pytest`, `pytest-cov`, `pre-commit`); includes the runtime dependencies.
- **`pyproject.toml`** — single source of truth for tool configuration:
  `[tool.ruff]` (lint + format), `[tool.mypy]` (strict type checking), and
  `[tool.pytest.ini_options]` (test runner and markers).
- **`.pre-commit-config.yaml`** — git hooks running Ruff, file-hygiene checks,
  and mypy at commit time, plus the test suite at push time.
- **`Makefile`** — memorable entry points for the tooling (`make help` lists
  them: format, lint, typecheck, test, coverage, check, repro, …).
- **`.vscode/`** — workspace settings and recommended extensions aligned with
  the toolchain.
- **`.env.example`** — template for the MLflow tracking URI and optional runtime
  settings (no DagsHub credentials since Sprint 7); copy to `.env` (never commit
  real secrets).
- **`.editorconfig`** — cross-editor formatting conventions.

See the [Developer Guide](developer-guide.md) for how these fit together and
[ADR-004](decisions/ADR-004-python-quality-toolchain.md) for why this toolchain
was chosen.

### Repository hygiene
- **`LICENSE`** (MIT), **`CONTRIBUTING.md`**, **`CODE_OF_CONDUCT.md`**,
  **`CHANGELOG.md`** — standard open-source project files.

---

## Code Organization Principles

- **Stage = module.** Each pipeline stage is a single, self-contained script
  under `src/` that can be run on its own or via `dvc repro`.
- **Config over code.** Parameters live in `params.yaml`, not in source.
- **Data and code are separate.** Code lives in Git; data and models live in DVC.
- **Secrets stay out of Git.** Credentials are provided via `.env`.

## Naming Conventions

- **Modules:** lowercase, action-oriented names matching their DVC stage
  (`preprocess.py`, `train.py`, `evaluate.py`).
- **DVC stages:** match their script's purpose (`preprocess`, `train`,
  `evaluate`).
- **Docs:** kebab-case filenames (`project-structure.md`).
- **ADRs:** `ADR-NNN-short-title.md`, numbered sequentially.
- **Diagram/screenshot folders:** kebab-case by category
  (`system-architecture/`, `mlflow-ui/`).
- **Python style:** enforced by Ruff (lint + format, 88-character lines,
  import sorting) and strict mypy — configured in `pyproject.toml` and run via
  pre-commit hooks. See the [Developer Guide](developer-guide.md) and
  [ADR-004](decisions/ADR-004-python-quality-toolchain.md).
- **Tests:** `tests/<marker>/test_<subject>.py` (e.g.
  `tests/unit/test_pipeline_io.py`), marked `smoke`, `unit`, `integration`, or
  `contract`.

---

## Related Documentation

- [Architecture](architecture.md)
- [Roadmap](roadmap.md)
- [Engineering Philosophy](philosophy.md)
- [ADR-001: Repository Structure](decisions/ADR-001-repository-structure.md)
