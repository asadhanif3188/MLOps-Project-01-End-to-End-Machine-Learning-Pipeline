# Documentation

Welcome to the documentation for the **End-to-End Machine Learning Pipeline**.
This page is the entry point to all project documentation.

For a general project introduction and quick start, see the
[repository README](../README.md).

---

## Overview

This project is a reproducible, batch ML pipeline that trains a Random Forest
classifier on the Pima Indians Diabetes dataset. It uses **DVC** for
data/pipeline versioning and **MLflow** (hosted on **DagsHub**) for experiment
tracking. Start with the [architecture](architecture.md) for a system-level view,
or the [project structure](project-structure.md) to navigate the repository.

---

## Core Documentation

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | System overview, components, pipeline, and data flow |
| [Logging Strategy](logging.md) | How the pipeline emits, formats, and persists logs |
| [Exception Strategy](exception-strategy.md) | Exception hierarchy, error propagation, and error logging |
| [Testing Strategy](testing-strategy.md) | Testing philosophy, `tests/` layout, and the testing roadmap |
| [Developer Guide](developer-guide.md) | Local setup and the day-to-day tooling (format, lint, type-check, test, pre-commit) |
| [Containerization Strategy](containerization.md) | Design for containerizing the pipeline (dev/prod images, security, K8s/CI readiness) — design only, not yet implemented |
| [Type Safety](type-safety.md) | Typing conventions, dynamic boundaries, and the mypy configuration |
| [Roadmap](roadmap.md) | Versioned milestones (v1–v6) with objectives and outcomes |
| [Architecture Decision Records](decisions/) | Records of significant technical decisions |
| [Project Structure](project-structure.md) | Every top-level directory and its responsibility |
| [Design Principles](design-principles.md) | Rationale behind core design and technology choices |
| [Engineering Philosophy](philosophy.md) | Principles guiding the project |

### Architecture Decision Records

| ADR | Title |
|-----|-------|
| [ADR-001](decisions/ADR-001-repository-structure.md) | Repository Structure |
| [ADR-002](decisions/ADR-002-why-mlflow.md) | Why MLflow |
| [ADR-003](decisions/ADR-003-why-dvc.md) | Why DVC |
| [ADR-004](decisions/ADR-004-python-quality-toolchain.md) | Python Quality Toolchain (Ruff, mypy, pytest, pre-commit) |
| [ADR-005](decisions/ADR-005-containerization-strategy.md) | Containerization Strategy |

### Engineering Reviews

| Review | Description |
|--------|-------------|
| [Sprint 2 — Production Readiness](reviews/sprint-02-engineering-review.md) | Principal-engineer review that drove the Sprint 2 engineering-excellence work (findings H-1..H-6) |
| [Sprint 2 — Final Validation](reviews/sprint-02-final-review.md) | Release-readiness validation for v1.1.0: checks, remaining debt, risks, and Sprint 3 recommendations |

---

## Process & Governance

| Document | Description |
|----------|-------------|
| [Release Process](release-checklist.md) | Step-by-step checklist for cutting a release |
| [Versioning](versioning.md) | Semantic Versioning policy with project examples |
| [GitHub Workflow](github-workflow.md) | Branching, commits, PRs, labels, milestones, releases |
| [Repository Metadata](repository-metadata.md) | Recommended description, topics, and presentation |

### Repository-Root Documents

| Document | Description |
|----------|-------------|
| [Contributing](../CONTRIBUTING.md) | How to contribute |
| [Code of Conduct](../CODE_OF_CONDUCT.md) | Community standards |
| [Security Policy](../SECURITY.md) | Reporting vulnerabilities and best practices |
| [Support](../SUPPORT.md) | How to get help |
| [Changelog](../CHANGELOG.md) | Notable changes |
| [License](../LICENSE) | MIT License |

---

## Visual Assets

| Location | Description |
|----------|-------------|
| [Diagrams](diagrams/) | Architecture and flow diagrams (placeholders) |
| [Screenshots](screenshots/) | UI and execution screenshots (placeholders) |

> Diagrams and screenshots are currently placeholders and will be added in a
> later sprint.

---

## Future Documents

Planned documentation, to be added as the project matures (see the
[roadmap](roadmap.md)):

- **Pipeline Usage Guide** — running and reproducing stages with DVC. <!-- TODO -->
- **CI/CD Documentation** — once continuous integration is added (Roadmap v3).
  <!-- TODO -->
- **Deployment Guide** — for cloud/Kubernetes deployment (Roadmap v4–v5).
  <!-- TODO -->
- **Monitoring & Operations** — for enterprise MLOps (Roadmap v6). <!-- TODO -->
