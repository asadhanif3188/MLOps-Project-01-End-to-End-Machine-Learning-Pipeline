# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Kubernetes architectural foundation** (Sprint 5, PR 1) — the containerized
  pipeline is now expressed as a Kubernetes **batch workload**, without
  manufacturing a fake online service. Foundation only: it establishes the
  structure and workload model; configuration/secrets, security hardening,
  resource limits, CI validation, and a demonstrated cluster run are deferred to
  later Sprint 5 PRs.
  - A [`k8s/`](k8s/) directory using a Kustomize `base/` + `overlays/local/`
    layout: an `mlops` **Namespace** (the environment boundary) and the pipeline
    modelled as a run-to-completion **`batch/v1` Job** (`restartPolicy: Never`,
    bounded `backoffLimit`) that runs `dvc repro`. Both `kustomize build k8s/base`
    and `kustomize build k8s/overlays/local` render successfully; the local
    overlay maps the workload to the locally built `ml-pipeline:local` image.
  - [ADR-009](docs/decisions/ADR-009-kubernetes-workload-model.md) recording the
    **Job-vs-Deployment** decision — batch vs service semantics, completion/retry
    lifecycle, why a Deployment (and a fake HTTP API) were rejected, and what the
    decision explicitly does *not* imply.
  - [docs/kubernetes-architecture.md](docs/kubernetes-architecture.md) describing
    why Kubernetes is introduced, the workload architecture, the configuration and
    security boundaries (as design contracts for later PRs), and a clear
    local-validation-vs-production-deferred split.
  - A batch-workload architecture diagram (Mermaid + ASCII) under
    [docs/diagrams/kubernetes-architecture/](docs/diagrams/kubernetes-architecture/).
  - [`k8s/README.md`](k8s/README.md) with render/apply instructions and a table of
    what is deliberately deferred to which PR.

### Changed

- Documentation updated to reflect the Kubernetes foundation: [roadmap](docs/roadmap.md)
  v4 marked in progress, [architecture](docs/architecture.md) notes the workload
  model, the [ADR index](docs/decisions/README.md) and
  [diagrams index](docs/diagrams/README.md) list the new records. No application
  logic changed; existing tests are unaffected.

## [1.3.1] - 2026-08-09

Proof Hardening — close the three limitations 1.3.0 documented rather than hid,
so the repository can *demonstrate* what it previously only *claimed*: evaluation
that is genuinely out-of-sample, reproducible execution proven by a real
`dvc repro`, and a type contract enforced on every pull request (not just
locally). No new pipeline capability — the goal is to make the existing
guarantees checkable.

### Added

- A dedicated **`split` stage** (`src/split.py`) that partitions the processed
  dataset into a training set (`data/processed/train.csv`) and a **held-out**
  evaluation set (`data/processed/test.csv`) with a stratified, seeded
  `train_test_split`. It is the single owner of both partitions and asserts they
  are **disjoint** (no row trains and evaluates) and **exhaustive** (no row is
  lost); `split.random_state` makes the exact held-out rows reproducible
  ([ADR-007](docs/decisions/ADR-007-held-out-evaluation.md)).
- [ADR-007](docs/decisions/ADR-007-held-out-evaluation.md) recording held-out
  evaluation as an engineering requirement.
- A self-contained **fixture DVC pipeline** (`tests/fixtures/pipeline/`) — the
  same `src/` stage code run against a small committed fixture dataset with the
  MLflow boundary stubbed offline (`_run_stage.py`) — carrying a **committed
  `dvc.lock`** so `declared pipeline + params + inputs + code = reproducible
  execution` is a checkable artifact, not a claim (resolves deviation D7,
  `dvc.lock`).
- [ADR-008](docs/decisions/ADR-008-fixture-reproducibility.md) recording the
  fixture-reproducibility strategy.
- `tests/contract/test_fixture_lock_contract.py`: a contract test asserting the
  committed fixture `dvc.lock` stays consistent with the fixture pipeline
  definition.

### Changed

- **Evaluation is now out-of-sample.** `train` fits only on the training
  partition (`data/processed/train.csv`) and `evaluate` scores the disjoint
  held-out partition (`data/processed/test.csv`), so the reported accuracy is a
  genuine generalization figure. The DVC lineage becomes
  `raw → preprocess → processed → split → {train.csv → train → model, test.csv} → evaluate → metrics`.
- The test suite grew from **84 to 101 tests** across the four tiers (smoke,
  unit, integration, contract), adding coverage for the `split` stage, the
  held-out boundary, and the fixture lock contract. All tiers remain
  deterministic and offline.

### Fixed

- **In-sample evaluation eliminated (deviation D5).** Because `train` no longer
  sees the held-out rows, evaluation accuracy is no longer measured on data the
  model was fit on.

### CI

- New **mypy type-check gate** (`python -m mypy`) in the `quality` job. CI now
  runs the same strict `[tool.mypy]` configuration a developer runs locally and
  that pre-commit runs at commit time, making the type contract a binding
  server-side gate: a type regression can no longer reach `main` on green
  pre-commit alone. mypy was already installed (requirements-dev.txt), so this
  adds no dependency install.
- New **fixture pipeline reproduction** step: a real `dvc repro` over the
  fixture pipeline runs all stages from a clean checkout, asserts the workspace
  is up to date against its committed lock, then force-re-runs and requires
  **byte-identical** `model.pkl` and `metrics.json` — a deterministic-artifact
  check that proves reproducibility rather than asserting it
  ([ADR-008](docs/decisions/ADR-008-fixture-reproducibility.md)).

## [1.3.0] - 2026-08-06

Sprint 4 — Pipeline Correctness & Reproducibility: turn attention from the
infrastructure *around* the ML pipeline to the pipeline itself. Correct the DVC
dependency graph so it models the real lineage, make configuration consistent
across `dvc.yaml`/`params.yaml`/code, separate the ML computation from the MLflow
boundary so stage logic is unit-testable, seed the training run for
determinism, and enforce the pipeline contract automatically in CI. Documentation
is reconciled to the as-built pipeline, and remaining limitations (in-sample
evaluation, no committed `dvc.lock`, name-pinned dependencies) are documented
rather than hidden.

### Added

- `src/tracking.py` — the single MLflow experiment-tracking boundary. All
  MLflow calls (and therefore every tracking network interaction) go through it;
  the stages import it lazily at the tracking boundary, so importing or
  unit-testing a stage requires neither MLflow nor credentials
  ([ADR-006](docs/decisions/ADR-006-pipeline-reproducibility.md) decision 4).
- A DVC-tracked **metrics artifact**: `evaluate` now writes
  `metrics/metrics.json`, declared under `metrics:` (`cache: false`) in
  `dvc.yaml` — accuracy is a first-class, versioned output, not only an MLflow
  entry and a log line (resolves deviation D4).
- Contract test suite (`tests/contract/test_pipeline_contract.py`) and a new
  `contract` pytest marker: eight pure-parsing checks that assert
  `dvc.yaml`/`params.yaml`/`src` agree with the pipeline contract (parameter
  consistency, no orphaned params, single-owner artifacts, the declared
  lineage `raw → preprocess → processed → train → model → evaluate → metrics`,
  and an acyclic graph). No data, network, or credentials.
- End-to-end integration test (`tests/integration/test_pipeline.py`): runs
  `preprocess → train → evaluate` through real temp files with MLflow stubbed,
  proving each stage's output is consumable by the next.
- Stage-level unit tests for `preprocess`, `train`, and `evaluate`
  (`tests/unit/`), exercising the extracted pure-compute functions without any
  external service.
- Pipeline contract ([docs/pipeline-contract.md](docs/pipeline-contract.md)) and
  [ADR-006](docs/decisions/ADR-006-pipeline-reproducibility.md) recording
  reproducibility and stage contracts as engineering requirements.
- Sprint 4 release documents: the
  [final engineering review](docs/reviews/sprint-04-final-review.md), the
  [retrospective](docs/retrospectives/sprint-04-retrospective.md), and the
  [proof-impact assessment](docs/proof/sprint-04-proof-impact.md).

### Changed

- **Corrected the DVC pipeline graph** (`dvc.yaml`): `train` now depends on the
  processed dataset (`data/processed/data.csv`) instead of the raw file, and
  `evaluate` depends on the processed dataset **and** the model. The graph is now
  the single linear chain the architecture always described.
- **Reconciled the parameter contract** (`params.yaml`): `train` uses
  `input`/`output`/`target`/`random_state`/`n_estimators`/`max_depth`; the
  evaluation section was renamed `test:` → `evaluate:` with explicit
  `data`/`model`/`target`/`metrics`. `dvc.yaml` param keys and the code now
  reference the same authoritative names, with no orphaned parameters.
- **Refactored the ML stages into separable concerns** (`train.py`,
  `evaluate.py`): a pure ML computation (`run_training` / `compute_metrics`) that
  performs no IO and imports no MLflow, artifact persistence via `pipeline_io`,
  and the lazily-imported `tracking` boundary. Existing MLflow behavior
  (metrics, params, artifacts, conditional model registration) is preserved, not
  removed, to make testing easier.
- `preprocess` now writes the processed CSV **with its header row**
  (`header=True`), so downstream stages can select `Outcome`/feature columns by
  name — the prerequisite for `train` consuming the processed dataset (resolves
  deviation D8).

### Fixed

- **Preprocess output is now consumed downstream** — the orphaned
  `data/processed/data.csv` produced by `preprocess` is `train`'s input
  (deviation D1).
- **Configuration drift eliminated** — `dvc.yaml` no longer references
  `train.data`/`train.model` (which did not exist in `params.yaml`), and the
  `evaluate` stage's parameters are declared in the graph (deviations D2, D3).
- **Training is now deterministic** — `train_test_split` and the
  `RandomForestClassifier` are seeded from `train.random_state`, and the
  configured `n_estimators`/`max_depth` are applied to the estimator instead of
  being inert (deviation D7, seeding portion).

### CI

- New `quality`-job step **"DVC pipeline integrity (graph + status, offline)"**:
  runs `dvc dag` (parseable, acyclic graph) and local `dvc status`, with
  `DVC_NO_ANALYTICS=true` so the step makes no network call and never touches the
  DagsHub remote. `dvc repro --dry` is deliberately not used — it requires the
  remote-only raw dataset; the guarantees it would give are enforced offline by
  the contract tests. `dvc` was already installed (a runtime dependency), so this
  adds no new install.
- The contract tests run as part of the existing `pytest` step, so a broken
  stage contract, an inconsistent parameter, or a mis-wired lineage fails a pull
  request. Sprint 3's `docker` job (build + non-root/imports/entrypoint
  validation) and least-privilege `contents: read` permissions are unchanged.

### Testing

- The suite grew to **84 tests** across four tiers: `smoke` (import/wiring),
  `unit` (isolated component and stage-compute tests), `integration` (full
  three-stage run, MLflow stubbed), and `contract` (static
  `dvc.yaml`/`params.yaml`/`src` consistency). All tiers are deterministic and
  run offline; unit tests require no live MLflow, network, or credentials.
- A `stub_tracking` fixture swaps the lazily-imported `tracking` module for an
  in-memory recorder, so stage read → compute → persist paths run end-to-end in
  tests without importing MLflow.

### Documentation

- Reconciled [docs/pipeline-contract.md](docs/pipeline-contract.md) from a
  design contract (CURRENT vs TARGET) to the **as-built** pipeline, with the
  remaining deviations (D5 in-sample evaluation, D7 `dvc.lock`) called out
  explicitly.
- Updated [architecture.md](docs/architecture.md),
  [project-structure.md](docs/project-structure.md), and
  [roadmap.md](docs/roadmap.md) to match the corrected pipeline, the new
  `tracking.py` module, the four-tier test layout, and the CI pipeline-integrity
  step.
- Rewrote the stale sections of the root [README.md](README.md): the training
  stage no longer claims to read raw data or grid-search all hyperparameters, the
  evaluation description reflects the metrics artifact and the in-sample boundary,
  and the DVC-stage snippets match the corrected `dvc.yaml`.

## [1.2.0] - 2026-08-05

Sprint 3 — Containerization & Continuous Integration: make the pipeline portable
and self-validating. Ship a production-grade container image and a Compose-based
development workflow, and add a GitHub Actions CI pipeline that lints, tests, and
builds the image on every push and pull request.

### Added

- Production-grade, multi-stage `Dockerfile` with three named targets from a
  single source of truth — `builder` (dependency compilation into an isolated
  virtualenv), `development` (builder + Ruff/mypy/pytest/pre-commit toolchain),
  and `runtime` (lean, non-root production image, the default target) — built on
  `python:3.12-slim-bookworm` with BuildKit cache mounts and OCI provenance
  labels.
- `.dockerignore` that keeps data, models, credentials, and local tooling out of
  the build context and image layers.
- `docker-compose.yml` development workflow: a bind-mounted `dev` service for the
  inner loop and an on-demand `pipeline` profile that runs the production image,
  plus `.env.example` for MLflow / DagsHub credentials.
- Continuous integration pipeline (`.github/workflows/ci.yml`): a `quality` job
  (Ruff lint + format check, pytest) gating a `docker` job that builds the
  `runtime` image and validates it (non-root UID 10001, core imports resolve,
  DVC entrypoint present).
- CI status badge on the root `README.md`.
- ADR-005 recording the containerization strategy (Docker/OCI, multi-stage
  build, `slim` base, non-root, twelve-factor config, externalized state).

### Changed

- ADR-005 status moved from "Accepted (design only — not yet implemented)" to
  "Accepted", with scope and consequences updated to reflect that the design was
  implemented in Sprint 3.
- Documentation refreshed for Sprint 3 (architecture, roadmap, project structure,
  and the documentation index) to describe the container image, Compose workflow,
  and CI pipeline.

### Security

- Production image runs as a dedicated non-root user (UID/GID 10001) with a
  `nologin` shell; build toolchain and compilers are confined to the `builder`
  stage and never reach runtime.
- Secrets and data are injected at run time (via `--env-file` / mounted volumes),
  never baked into image layers; base image pinned by codename
  (`python:3.12-slim-bookworm`), with digest pinning recorded as a follow-up.
- CI workflow granted least-privilege `contents: read` only, structurally
  preventing it from pushing images or writing to the repository.

### CI

- CI is validation only — it lints, tests, and builds/validates the image on
  every push to `main` and every pull request; it does not deploy, publish
  images, or use Kubernetes (continuous delivery is deferred to Roadmap v3+).
- Concurrency control cancels superseded in-flight runs per ref; pip and
  BuildKit (`type=gha`) layer caches speed repeat runs.

### Documentation

- `docs/containerization.md` — containerization strategy and as-built
  build/run instructions (including hardened, read-only execution).
- `docs/docker-development.md` — day-to-day Docker Compose development workflow.
- `docs/ci-cd.md` — CI stages, failure strategy, local reproduction, and the
  future continuous-delivery roadmap.

## [1.1.0] - 2026-08-02

Sprint 2 — Engineering Excellence: raise the baseline pipeline to a maintainable,
professionally engineered codebase (logging, error handling, typing, tests, and
a quality toolchain), driven by a principal-engineer production-readiness review.

### Added

- Principal-engineer production-readiness review
  (`docs/reviews/sprint-02-engineering-review.md`) whose findings (H-1..H-6)
  drove the Sprint 2 engineering-excellence work.
- Centralized logging framework: `src/logging_config.py` (console + rotating
  file handlers, `LOG_LEVEL`/`LOG_DIR` environment control) replacing `print()`
  across all pipeline stages, with `docs/logging.md` documenting the strategy
  (review finding H-1).
- Standardized exception handling: a typed hierarchy in `src/exceptions.py`
  (`PipelineError` → `ConfigError`, `DataError`, `ModelError`,
  `TrackingError`), centralized IO/config/serialization boundaries in
  `src/pipeline_io.py`, a uniform stage entry point in `src/stage_runner.py`
  (log once, exit non-zero), and `docs/exception-strategy.md` (review finding
  H-2).
- Complete type annotations across `src/` with a strict mypy configuration in
  `pyproject.toml`, documented in `docs/type-safety.md`.
- Testing foundation: a `pytest` suite under `tests/` (smoke and unit tests)
  with shared fixtures (`tests/conftest.py`) and configuration in
  `pyproject.toml`; `pytest`/`pytest-cov` added to `requirements-dev.txt`; and
  `docs/testing-strategy.md` documenting the philosophy, layout, and roadmap
  (review finding H-3).
- Developer experience tooling: Ruff linter and formatter (configured in
  `pyproject.toml`), a `.pre-commit-config.yaml` running Ruff, file-hygiene
  checks, mypy, and (at push time) the test suite; a `Makefile` with helpful
  development commands (`make help`); VS Code workspace settings and recommended
  extensions under `.vscode/`; `ruff`/`pre-commit` added to
  `requirements-dev.txt`; and `docs/developer-guide.md` documenting local
  development, formatting, linting, testing, and the pre-commit workflow.
- ADR-004 recording the Python quality toolchain decision (Ruff, mypy, pytest,
  pre-commit).
- Sprint 2 final engineering-validation review
  (`docs/reviews/sprint-02-final-review.md`).

### Changed

- Pipeline stage scripts (`preprocess.py`, `train.py`, `evaluate.py`)
  refactored for organization and readability: corrected import grouping,
  removed redundant intermediates, and reconciled stale docstrings.
- Core documentation updated to reflect the Sprint 2 engineering work:
  `docs/architecture.md` (shared infrastructure modules, expanded technology
  table), `docs/roadmap.md` (v2 delivered vs. remaining scope),
  `docs/project-structure.md` (new modules, `tests/`, tooling files), and
  `docs/design-principles.md` (logging, exceptions, typing, testing, and
  toolchain rationale).
- Reconciled previously stale documentation with the delivered work:
  `docs/philosophy.md` and `docs/decisions/ADR-001-repository-structure.md`
  (testing and tooling now exist, delivered without a package layout), and
  fixed broken in-page anchors and obsolete release/versioning notes surfaced
  during final validation.

## [1.0.0] - 2026-08-01

Sprint 1 — Professional Repository Transformation: establish repository
governance, engineering documentation, and the ADR framework on top of the
foundation pipeline.

### Added

- Documentation scaffolding under `docs/` (architecture, roadmap, project
  structure, and architecture decision records).
- Placeholder directories for diagrams and screenshots.
- Repository hygiene files: `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `CHANGELOG.md`, and `.editorconfig`.
- First drafts of the core documentation: expanded `docs/architecture.md`,
  `docs/roadmap.md`, and `docs/project-structure.md`; first drafts of ADR-001,
  ADR-002, and ADR-003; and a new `docs/philosophy.md` describing engineering
  principles.
- Repository governance and GitHub metadata: issue templates (bug, feature,
  documentation) and PR template under `.github/`; `SECURITY.md` and
  `SUPPORT.md`; and documentation for the GitHub workflow, semantic versioning,
  release checklist, repository metadata recommendations, and a documentation
  index (`docs/README.md`).
- `docs/design-principles.md` explaining the rationale behind core design and
  technology choices (batch pipeline, Random Forest, Python, DVC, MLflow,
  modular code, YAML configuration).

### Changed

- Roadmap v1 renamed from "Course Implementation" to "Foundation Release"; v5
  expanded to "Production Cloud Platform" and v6 objectives broadened.
- ADR-001/002/003 finalized (status Accepted, dated) with a more confident
  engineering voice and no placeholder markers.
- Recommended repository description updated to
  "Production-Oriented MLOps Pipeline using DVC, MLflow and Python".

[Unreleased]: https://github.com/asadhanif3188/MLOps-Project-01-End-to-End-Machine-Learning-Pipeline/compare/v1.3.1...HEAD
[1.3.1]: https://github.com/asadhanif3188/MLOps-Project-01-End-to-End-Machine-Learning-Pipeline/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/asadhanif3188/MLOps-Project-01-End-to-End-Machine-Learning-Pipeline/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/asadhanif3188/MLOps-Project-01-End-to-End-Machine-Learning-Pipeline/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/asadhanif3188/MLOps-Project-01-End-to-End-Machine-Learning-Pipeline/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/asadhanif3188/MLOps-Project-01-End-to-End-Machine-Learning-Pipeline/releases/tag/v1.0.0
