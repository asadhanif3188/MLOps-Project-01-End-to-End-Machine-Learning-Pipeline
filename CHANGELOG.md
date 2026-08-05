# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_No unreleased changes yet._

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

[Unreleased]: https://github.com/asadhanif3188/MLOps-Project-01-End-to-End-Machine-Learning-Pipeline/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/asadhanif3188/MLOps-Project-01-End-to-End-Machine-Learning-Pipeline/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/asadhanif3188/MLOps-Project-01-End-to-End-Machine-Learning-Pipeline/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/asadhanif3188/MLOps-Project-01-End-to-End-Machine-Learning-Pipeline/releases/tag/v1.0.0
