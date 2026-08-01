# Roadmap

This roadmap organizes the project's evolution into versioned milestones. Each
version states its **objectives** and **expected outcomes** rather than
prescribing implementation details. Concrete technical decisions will be captured
as [Architecture Decision Records](decisions/) when they are made.

> **Legend:** ✅ Done · 🚧 In progress · ⬜ Planned

| Version | Theme | Status |
|---------|-------|--------|
| [v1](#version-1--course-implementation) | Course Implementation | ✅ |
| [v2](#version-2--engineering-improvements) | Engineering Improvements | 🚧 |
| [v3](#version-3--cicd) | CI/CD | ⬜ |
| [v4](#version-4--kubernetes) | Kubernetes | ⬜ |
| [v5](#version-5--cloud-deployment) | Cloud Deployment | ⬜ |
| [v6](#version-6--enterprise-mlops) | Enterprise MLOps | ⬜ |

---

## Version 1 — Course Implementation ✅

**Objective:** Establish a working, reproducible ML pipeline as a learning
baseline.

**Scope delivered:**

- Three-stage DVC pipeline: preprocess → train → evaluate.
- Random Forest classifier with `GridSearchCV` hyperparameter tuning.
- MLflow experiment tracking hosted on DagsHub.
- DVC data/model versioning with an S3-compatible DagsHub remote.

**Expected outcome:** A runnable pipeline that reproduces training results and
tracks experiments — the foundation everything else builds on.

---

## Version 2 — Engineering Improvements 🚧

**Objective:** Elevate the repository from a course project to a professional,
maintainable codebase.

**Objectives:**

- Professional documentation (this sprint): architecture, roadmap, project
  structure, ADRs, engineering philosophy, and a rewritten README.
- Repository hygiene: LICENSE, CONTRIBUTING, CODE_OF_CONDUCT, CHANGELOG,
  `.editorconfig`.
- Code quality: type hints, formatting/linting (black, isort, ruff), and
  structured logging in place of `print`.
- Correctness fixes surfaced during documentation (tracked as TODOs):
  reconcile `dvc.yaml`/`params.yaml` parameter names, make the `preprocess`
  output feed downstream stages, and evaluate on a held-out split.
- Introduce automated tests (pytest) with meaningful coverage.

**Expected outcome:** A repository that reads as an actively maintained,
professionally engineered project and is safe to change with confidence.

> <!-- TODO: confirm which v2 items land in this sprint vs. later sub-sprints. -->

---

## Version 3 — CI/CD ⬜

**Objective:** Automate quality gates and pipeline reproduction.

**Objectives:**

- Continuous integration: run linting and tests on every pull request.
- Automated pipeline validation (e.g., `dvc repro` / `dvc status`) in CI.
- Enforced formatting and basic security scanning.

**Expected outcome:** Every change is automatically validated before merge,
reducing regressions and manual effort.

> <!-- TODO: select CI provider and define the pipeline once decided (record as an ADR). -->

---

## Version 4 — Kubernetes ⬜

**Objective:** Make the pipeline portable and horizontally runnable.

**Objectives:**

- Containerize the pipeline for consistent execution environments.
- Run pipeline stages as orchestrated workloads on Kubernetes.
- Externalize configuration and secrets appropriately for a cluster.

**Expected outcome:** The pipeline runs reproducibly on any conformant cluster,
independent of a developer's local machine.

> <!-- TODO: define container base image, orchestration approach, and secret handling (record as ADRs). -->

---

## Version 5 — Cloud Deployment ⬜

**Objective:** Operate the pipeline and model in a cloud environment.

**Objectives:**

- Managed cloud object storage for the DVC remote.
- A hosted MLflow tracking server (or managed equivalent).
- A model-serving endpoint for inference.

**Expected outcome:** Training and serving run on managed cloud infrastructure
with clear separation of environments.

> <!-- TODO: select cloud provider and serving mechanism (record as ADRs). -->

---

## Version 6 — Enterprise MLOps ⬜

**Objective:** Add the operational maturity expected of production ML systems.

**Objectives:**

- Monitoring and alerting for pipeline and model health.
- Data/model drift detection and automated retraining triggers.
- Governance: model lineage, approvals, and auditability.

**Expected outcome:** A production-grade MLOps platform demonstrating engineering
judgment beyond model building.

> <!-- TODO: define monitoring stack, drift metrics, and retraining triggers (record as ADRs). -->

---

## Related Documentation

- [Architecture](architecture.md)
- [Engineering Philosophy](philosophy.md)
- [Architecture Decision Records](decisions/)
