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
├── src/                  # Pipeline stage scripts (one module per stage)
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── data/                 # Datasets (DVC-tracked; contents not stored in Git)
│   ├── raw/              #   raw input data (data.csv, tracked via data.csv.dvc)
│   └── processed/        #   preprocessed output
├── models/               # Serialized model artifacts (DVC-tracked, git-ignored)
├── docs/                 # Project documentation (see docs/README.md portal)
│   ├── README.md         #   documentation index
│   ├── architecture.md
│   ├── roadmap.md
│   ├── project-structure.md
│   ├── design-principles.md
│   ├── philosophy.md
│   ├── github-workflow.md
│   ├── versioning.md
│   ├── release-checklist.md
│   ├── repository-metadata.md
│   ├── decisions/        #   Architecture Decision Records (ADRs)
│   ├── diagrams/         #   diagram placeholders (by category)
│   └── screenshots/      #   screenshot placeholders (by category)
├── .github/              # Issue/PR templates and GitHub config
│   ├── ISSUE_TEMPLATE/
│   └── pull_request_template.md
├── .dvc/                 # DVC internal config (remote definition)
├── dvc.yaml              # DVC pipeline definition (stages, deps, params, outs)
├── params.yaml           # Declarative pipeline parameters
├── requirements.txt      # Python dependencies
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

> ⚠️ **Known gaps** (documented in [architecture.md](architecture.md), to be
> fixed in a future sprint): the `train`/`evaluate` scripts read `data/raw`
> rather than the `preprocess` output, and `dvc.yaml` param names don't match
> `params.yaml`.

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
- **`github-workflow.md`**, **`versioning.md`**, **`release-checklist.md`** —
  process and governance (branching, SemVer, releases).
- **`repository-metadata.md`** — recommended repository description and topics.
- **`decisions/`** — Architecture Decision Records (ADR-001..003 + index).
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
- **`requirements.txt`** — Python dependencies (`dvc`, `dagshub`,
  `scikit-learn`, `mlflow`, `dvc-s3`, `python-dotenv`).
- **`.env.example`** — template listing required MLflow/DagsHub credentials;
  copy to `.env` (never commit real secrets).
- **`.editorconfig`** — cross-editor formatting conventions.

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

> <!-- TODO: adopt and document a formal Python style toolchain (black, isort,
> ruff) and any function/variable conventions in Roadmap v2. -->

---

## Related Documentation

- [Architecture](architecture.md)
- [Roadmap](roadmap.md)
- [Engineering Philosophy](philosophy.md)
- [ADR-001: Repository Structure](decisions/ADR-001-repository-structure.md)
