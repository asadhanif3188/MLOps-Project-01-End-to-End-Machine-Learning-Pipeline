# Roadmap

The roadmap organizes the project's evolution into versioned milestones. Each
version states its **objectives** and **expected outcome** rather than
prescribing implementation details. Concrete technical decisions are captured as
[Architecture Decision Records](decisions/) when they are made.

> **Legend:** ✅ Done · 🚧 In progress · ⬜ Planned

| Version | Theme | Status |
|---------|-------|--------|
| [v1](#version-1--foundation-release) | Foundation Release | ✅ |
| [v2](#version-2--engineering-improvements) | Engineering Improvements | 🚧 |
| [v3](#version-3--cicd) | CI/CD | ⬜ |
| [v4](#version-4--kubernetes) | Kubernetes | ⬜ |
| [v5](#version-5--production-cloud-platform) | Production Cloud Platform | ⬜ |
| [v6](#version-6--enterprise-mlops) | Enterprise MLOps | ⬜ |

---

## Version 1 — Foundation Release ✅

**Objective:** Establish a working, reproducible ML pipeline as the baseline for
future engineering enhancements.

> The initial implementation is based on a guided educational project. It serves
> as the honest baseline the rest of this roadmap builds on — not as the
> project's long-term identity.

**Scope delivered:**

- Three-stage DVC pipeline: preprocess → train → evaluate.
- Random Forest classifier with `GridSearchCV` hyperparameter tuning.
- MLflow experiment tracking hosted on DagsHub.
- DVC data/model versioning with an S3-compatible DagsHub remote.

**Expected outcome:** A runnable pipeline that reproduces training results and
tracks experiments — the foundation everything else builds on.

---

## Version 2 — Engineering Improvements 🚧

**Objective:** Elevate the repository from a baseline implementation to a
professional, maintainable codebase.

**Objectives:**

- Professional documentation (this sprint): architecture, roadmap, project
  structure, ADRs, engineering philosophy, design principles, and a rewritten
  README.
- Repository hygiene and governance: LICENSE, CONTRIBUTING, CODE_OF_CONDUCT,
  CHANGELOG, `.editorconfig`, security and support policies, issue/PR templates.
- Code quality: type hints, formatting/linting (black, isort, ruff), and
  structured logging in place of `print`.
- Correctness fixes surfaced during documentation: reconcile
  `dvc.yaml`/`params.yaml` parameter names, feed the `preprocess` output into
  downstream stages, and evaluate on a held-out split.
- Automated tests (pytest) with meaningful coverage.

**Expected outcome:** A repository that reads as an actively maintained,
professionally engineered project and is safe to change with confidence.

> **TODO:** Confirm which v2 items land in this sprint versus later sub-sprints.

---

## Version 3 — CI/CD ⬜

**Objective:** Automate quality gates and pipeline reproduction.

**Objectives:**

- Continuous integration: run linting and tests on every pull request.
- Automated pipeline validation (e.g., `dvc repro` / `dvc status`) in CI.
- Enforced formatting and basic security scanning.
- Branch protection requiring green checks before merge.

**Expected outcome:** Every change is automatically validated before merge,
reducing regressions and manual effort.

> **TODO:** Select the CI provider and ratify the pipeline design as an ADR.

---

## Version 4 — Kubernetes ⬜

**Objective:** Make the pipeline portable and horizontally runnable.

**Objectives:**

- Containerize the pipeline for consistent execution environments.
- Run pipeline stages as orchestrated workloads on Kubernetes.
- Externalize configuration and secrets for a cluster (ConfigMaps/Secrets).
- Define resource requests/limits for reproducible scheduling.

**Expected outcome:** The pipeline runs reproducibly on any conformant cluster,
independent of a developer's local machine.

> **TODO:** Ratify container base image, orchestration approach, and secret
> handling as ADRs.

---

## Version 5 — Production Cloud Platform ⬜

**Objective:** Provision and operate the pipeline on managed cloud
infrastructure defined as code.

**Objectives:**

- **Infrastructure as Code** with Terraform (versioned, reviewable).
- **AWS** as the target cloud provider.
- **Remote state** management for Terraform (e.g., S3 backend with locking).
- **IAM** roles and least-privilege access for pipeline and storage.
- **CI/CD** integration to plan/apply infrastructure changes.
- **Monitoring** and centralized logging for pipeline and infrastructure health.

**Expected outcome:** Production-deployable infrastructure, provisioned
reproducibly from code with clear separation of environments.

> **TODO:** Ratify the cloud provider, Terraform module structure, and serving
> mechanism as ADRs before implementation.

---

## Version 6 — Enterprise MLOps ⬜

**Objective:** Add the operational maturity expected of production ML systems.

**Objectives:**

- **Model serving** via a versioned inference endpoint with rollback.
- **Monitoring and alerting** for data quality, latency, and model performance.
- **Drift detection** (data and concept drift) with automated retraining
  triggers.
- **Governance:** model lineage, approval gates, and auditability.
- **Feature and artifact management** across environments.

**Expected outcome:** A production-grade MLOps platform demonstrating engineering
judgment well beyond model building.

> **TODO:** Ratify the monitoring stack, drift metrics, and retraining triggers
> as ADRs.

---

## Related Documentation

- [Architecture](architecture.md)
- [Design Principles](design-principles.md)
- [Engineering Philosophy](philosophy.md)
- [Architecture Decision Records](decisions/)
