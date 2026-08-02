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
│   ├── preprocess.py     #   stage: raw CSV → processed CSV
│   ├── train.py          #   stage: train + tune model, log to MLflow
│   ├── evaluate.py       #   stage: evaluate model, log metric
│   ├── logging_config.py #   centralized logging configuration
│   ├── exceptions.py     #   typed exception hierarchy (PipelineError, …)
│   ├── pipeline_io.py    #   IO/config/serialization helpers (typed errors)
│   └── stage_runner.py   #   uniform stage entry point (log once, exit non-zero)
├── tests/                # Pytest suite
│   ├── conftest.py       #   shared fixtures
│   ├── smoke/            #   fast import/wiring checks
│   └── unit/             #   isolated component tests
├── data/                 # Datasets (DVC-tracked; contents not stored in Git)
│   ├── raw/              #   raw input data (data.csv, tracked via data.csv.dvc)
│   └── processed/        #   preprocessed output
├── models/               # Serialized model artifacts (DVC-tracked, git-ignored)
├── logs/                 # Rotating pipeline logs (git-ignored)
├── docs/                 # Project documentation (see docs/README.md portal)
│   ├── README.md         #   documentation index
│   ├── architecture.md
│   ├── roadmap.md
│   ├── project-structure.md
│   ├── design-principles.md
│   ├── philosophy.md
│   ├── logging.md        #   logging strategy
│   ├── exception-strategy.md
│   ├── type-safety.md
│   ├── testing-strategy.md
│   ├── developer-guide.md
│   ├── github-workflow.md
│   ├── versioning.md
│   ├── release-checklist.md
│   ├── repository-metadata.md
│   ├── decisions/        #   Architecture Decision Records (ADRs)
│   ├── reviews/          #   engineering reviews (sprint-02 production readiness)
│   ├── diagrams/         #   diagram placeholders (by category)
│   └── screenshots/      #   screenshot placeholders (by category)
├── .github/              # Issue/PR templates and GitHub config
│   ├── ISSUE_TEMPLATE/
│   └── pull_request_template.md
├── .vscode/              # VS Code workspace settings & recommended extensions
├── .dvc/                 # DVC internal config (remote definition)
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

- **`preprocess.py`** — reads the raw dataset and writes the processed CSV. Reads
  its paths from the `preprocess` section of `params.yaml`.
- **`train.py`** — trains a `RandomForestClassifier` using `GridSearchCV`, logs
  parameters/metrics/artifacts to MLflow, and serializes the best model to
  `models/model.pkl`.
- **`evaluate.py`** — loads the serialized model and logs an accuracy metric to
  MLflow.

The stages share four infrastructure modules (added in Sprint 2):

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

> ⚠️ **Known gaps** (documented in [architecture.md](architecture.md), to be
> fixed in a future sprint): the `train`/`evaluate` scripts read `data/raw`
> rather than the `preprocess` output, and `dvc.yaml` param names don't match
> `params.yaml`.

### `tests/` — Automated tests
A `pytest` suite (see [Testing Strategy](testing-strategy.md)):

- **`conftest.py`** — shared fixtures.
- **`smoke/`** — fast import/wiring checks across all of `src/`.
- **`unit/`** — isolated tests of the critical components (`exceptions.py`,
  `pipeline_io.py`, `stage_runner.py`) with no network or external services.

Run with `make test` (or `python -m pytest`); markers `smoke` and `unit` select
slices of the suite.

### `data/` — Datasets (DVC-tracked)
- **`raw/`** — original input data. `data.csv` is versioned via
  `data.csv.dvc`; the actual file is pulled from the DVC remote, not committed.
- **`processed/`** — output of the `preprocess` stage.
- Contents are excluded from Git and managed by DVC.

### `models/` — Model artifacts
Serialized model(s), e.g. `model.pkl`. Git-ignored (`models/` in `.gitignore`)
and versioned by DVC.

### `docs/` — Documentation
- **`README.md`** — the documentation index / portal.
- **`architecture.md`** — system architecture and data flow.
- **`roadmap.md`** — versioned milestones.
- **`project-structure.md`** — this document.
- **`design-principles.md`** — rationale behind core design and technology choices.
- **`philosophy.md`** — engineering principles.
- **`logging.md`**, **`exception-strategy.md`**, **`type-safety.md`**,
  **`testing-strategy.md`**, **`developer-guide.md`** — engineering strategy
  references introduced in Sprint 2 (observability, error handling, typing,
  testing, and day-to-day tooling).
- **`github-workflow.md`**, **`versioning.md`**, **`release-checklist.md`** —
  process and governance (branching, SemVer, releases).
- **`repository-metadata.md`** — recommended repository description and topics.
- **`decisions/`** — Architecture Decision Records (ADR-001..004 + index).
- **`reviews/`** — engineering reviews (the Sprint 2 production-readiness
  review that drove the engineering-excellence work).
- **`diagrams/`** — placeholder subfolders for diagrams (system architecture,
  pipeline flow, deployment, Kubernetes, CI/CD).
- **`screenshots/`** — placeholder subfolders for screenshots (MLflow UI, DVC
  pipeline, project execution, folder structure, training logs).

### `.github/` — GitHub configuration
Issue templates (`bug_report.md`, `feature_request.md`, `documentation.md`) and
the pull request template used to standardize contributions.

### Configuration & pipeline definition
- **`dvc.yaml`** — the pipeline graph: stages with their dependencies,
  parameters, and outputs.
- **`params.yaml`** — declarative parameters consumed by the stage scripts,
  keeping configuration out of code.
- **`.dvc/config`** — defines the S3-compatible DagsHub remote.

### Environment & tooling
- **`requirements.txt`** — runtime Python dependencies (`dvc`, `dagshub`,
  `scikit-learn`, `mlflow`, `dvc-s3`, `python-dotenv`).
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
- **`.env.example`** — template listing required MLflow/DagsHub credentials;
  copy to `.env` (never commit real secrets).
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
  `tests/unit/test_pipeline_io.py`), marked `smoke` or `unit`.

---

## Related Documentation

- [Architecture](architecture.md)
- [Roadmap](roadmap.md)
- [Engineering Philosophy](philosophy.md)
- [ADR-001: Repository Structure](decisions/ADR-001-repository-structure.md)
