# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- Principal-engineer production-readiness review
  (`docs/reviews/sprint-02-engineering-review.md`) whose findings drove the
  Sprint 2 engineering-excellence work.
- Centralized logging framework: `src/logging_config.py` (console + rotating
  file handlers, `LOG_LEVEL`/`LOG_DIR` environment control) replacing `print()`
  across all pipeline stages, with `docs/logging.md` documenting the strategy.
- Standardized exception handling: a typed hierarchy in `src/exceptions.py`
  (`PipelineError` → `ConfigError`, `DataError`, `ModelError`,
  `TrackingError`), centralized IO/config/serialization boundaries in
  `src/pipeline_io.py`, a uniform stage entry point in `src/stage_runner.py`
  (log once, exit non-zero), and `docs/exception-strategy.md`.
- Complete type annotations across `src/` with a strict mypy configuration in
  `pyproject.toml`, documented in `docs/type-safety.md`.
- ADR-004 recording the Python quality toolchain decision (Ruff, mypy, pytest,
  pre-commit).
- Testing foundation: a `pytest` suite under `tests/` (smoke and unit tests)
  with shared fixtures (`tests/conftest.py`) and configuration in
  `pyproject.toml`; `pytest`/`pytest-cov` added to `requirements-dev.txt`; and
  `docs/testing-strategy.md` documenting the philosophy, layout, and roadmap.
- Developer experience tooling: Ruff linter and formatter (configured in
  `pyproject.toml`), a `.pre-commit-config.yaml` running Ruff, file-hygiene
  checks, mypy, and (at push time) the test suite; a `Makefile` with helpful
  development commands (`make help`); VS Code workspace settings and recommended
  extensions under `.vscode/`; `ruff`/`pre-commit` added to
  `requirements-dev.txt`; and `docs/developer-guide.md` documenting local
  development, formatting, linting, testing, and the pre-commit workflow.

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
- Roadmap v1 renamed from "Course Implementation" to "Foundation Release"; v5
  expanded to "Production Cloud Platform" and v6 objectives broadened.
- ADR-001/002/003 finalized (status Accepted, dated) with a more confident
  engineering voice and no placeholder markers.
- Recommended repository description updated to
  "Production-Oriented MLOps Pipeline using DVC, MLflow and Python".

<!-- TODO: Add releases as the project matures. -->
