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
| [v3](#version-3--cicd) | CI/CD | 🚧 |
| [v4](#version-4--kubernetes) | Kubernetes | 🚧 |
| [v5](#version-5--production-cloud-platform) | Production Cloud Platform | ⬜ |
| [v6](#version-6--enterprise-mlops) | Enterprise MLOps | ⬜ |

---

## Version 1 — Foundation Release

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

## Version 2 — Engineering Improvements

**Objective:** Elevate the repository from a baseline implementation to a
professional, maintainable codebase.

**Delivered:**

- ✅ Professional documentation under `docs/`: architecture, roadmap, project
  structure, ADRs, engineering philosophy, and design principles.
- ✅ Repository hygiene and governance: LICENSE, CONTRIBUTING, CODE_OF_CONDUCT,
  CHANGELOG, `.editorconfig`, security and support policies, issue/PR templates.
- ✅ Principal-engineer production-readiness review
  ([reviews/sprint-02-engineering-review.md](reviews/sprint-02-engineering-review.md)),
  whose findings drove the rest of the sprint.
- ✅ Code organization and readability refactor of the pipeline stages.
- ✅ Centralized logging in place of `print`
  ([Logging Strategy](logging.md), review finding H-1).
- ✅ Standardized exception handling: typed hierarchy, wrapped IO/config
  boundaries, uniform stage entry point
  ([Exception Strategy](exception-strategy.md), review finding H-2).
- ✅ Complete type annotations with a strict mypy configuration
  ([Type Safety](type-safety.md)).
- ✅ Testing foundation: pytest smoke and unit suites with shared fixtures
  ([Testing Strategy](testing-strategy.md), review finding H-3).
- ✅ Developer experience: Ruff linting/formatting, pre-commit hooks, Makefile,
  VS Code workspace settings
  ([Developer Guide](developer-guide.md),
  [ADR-004](decisions/ADR-004-python-quality-toolchain.md)).

**Delivered in Sprint 4 (`v1.3.0` — Pipeline Correctness & Reproducibility):**

- ✅ Correctness fixes surfaced during documentation: reconciled
  `dvc.yaml`/`params.yaml` parameter names and fed the `preprocess` output into
  `train` (and evaluation source). Now enforced by the `contract` tests and CI
  (see the [Pipeline Contract](pipeline-contract.md)).
- ✅ Decoupled the stage bodies from MLflow/network via the `tracking` boundary,
  so `train` and `evaluate` logic is now unit-tested
  ([Testing Strategy](testing-strategy.md)).
- ✅ Corrected the stale root `README.md` claims (training input, hyperparameter
  tuning, evaluation output, DVC-stage snippets) to match the as-built pipeline.

**Still remaining:**

- ✅ Evaluate on a genuine **held-out split** — done. A dedicated `split` stage now
  partitions the processed dataset so `train` and `evaluate` consume disjoint data
  and the reported accuracy is out-of-sample (deviation **D5** resolved;
  [ADR-007](decisions/ADR-007-held-out-evaluation.md),
  [pipeline contract §8](pipeline-contract.md#8-evaluation-boundary)).
- ✅ Commit a `dvc.lock` and run `dvc repro` in CI against a fixture dataset —
  done. A self-contained fixture pipeline reproduces the same four stages and the
  same `src/` code offline (no remote, no MLflow, no credentials); CI runs a real
  `dvc repro`, asserts the committed lock is up to date, and requires byte-identical
  outputs on a forced re-run. This upgrades reproducibility from *logical* to
  *drift-gated* and closes the `dvc.lock`/execution portion of deviation **D7**
  ([ADR-008](decisions/ADR-008-fixture-reproducibility.md),
  [pipeline contract §7](pipeline-contract.md#7-reproducibility-expectations)).
  End-to-end reproduction of the *production* run (remote data + live MLflow +
  digest-pinned deps) remains a documented limitation, not a gap.

**Expected outcome:** A repository that reads as an actively maintained,
professionally engineered project and is safe to change with confidence.

---

## Version 3 — CI/CD

**Objective:** Automate quality gates and pipeline reproduction.

**Objectives:**

- ✅ Continuous integration: run linting, strict type checking (mypy), and tests
  on every pull request (GitHub Actions — see [CI/CD](ci-cd.md) and
  [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)).
- ✅ Build and validate the container image in CI (build only — **not** pushed).
- ✅ Automated pipeline validation in CI — **definition** validated offline in
  Sprint 4 (`dvc dag` + local `dvc status` + the `contract` tests), and a real
  **execution** check now runs a scoped `dvc repro` against a committed fixture
  dataset + `dvc.lock` (proof-hardening milestone;
  [ADR-008](decisions/ADR-008-fixture-reproducibility.md)). The production raw
  dataset stays remote-only, so the production run itself is still not executed in
  CI — a documented limitation, not a gap.
- 🚧 Basic security/supply-chain scanning (image scan, SBOM, signing).
- 🚧 Publish the container image on release
  ([Containerization Strategy](containerization.md), [ADR-005](decisions/ADR-005-containerization-strategy.md)).
- 🚧 Branch protection requiring green checks before merge.

**Expected outcome:** Every change is automatically validated before merge,
reducing regressions and manual effort.

> **Status:** CI (checkout → setup Python → install → Ruff → mypy → pytest → DVC
> pipeline integrity → fixture `dvc repro` → Docker build → build validation) is
> implemented on **GitHub Actions**. It validates only — no deploy, no image push,
> no Kubernetes. Offline pipeline-definition validation landed in Sprint 4, and a
> real fixture-pipeline `dvc repro` execution check landed in the proof-hardening
> milestone ([ADR-008](decisions/ADR-008-fixture-reproducibility.md)); the
> remaining items above (publish, scan, branch protection) are the path to
> continuous *delivery*; see
> [CI/CD § Future CD roadmap](ci-cd.md#future-cd-roadmap).
>
> **TODO:** Ratify the CD approach (registry, signing, deploy target) as an ADR.

---

## Version 4 — Kubernetes

**Objective:** Make the pipeline portable and horizontally runnable.

**Objectives:**

- ✅ Containerize the pipeline for consistent execution environments —
  **implemented** in Sprint 3: multi-stage [`Dockerfile`](../Dockerfile),
  [`.dockerignore`](../.dockerignore), and a [`docker-compose.yml`](../docker-compose.yml)
  local dev workflow ([Containerization Strategy](containerization.md),
  [Docker Development](docker-development.md),
  [ADR-005](decisions/ADR-005-containerization-strategy.md)).
- 🚧 Run pipeline stages as an orchestrated Kubernetes workload — **runnable
  workload landed** (Sprint 5, PR 1–2): an `mlops` namespace and the pipeline
  modelled as a **runnable** `batch/v1` **Job** (not a Deployment) — the real
  `ml-pipeline:local` image, the real `dvc repro` command, and a finite-run
  lifecycle (`restartPolicy: Never`, `backoffLimit: 2`,
  `activeDeadlineSeconds: 1800`) — structured with Kustomize under
  [`k8s/`](../k8s/), with a local run runbook
  ([Kubernetes Architecture](kubernetes-architecture.md),
  [ADR-009](decisions/ADR-009-kubernetes-workload-model.md)). Executed on a local
  Docker Desktop cluster (2026-08-12): the Job lifecycle is verified end to end (3
  attempts → terminal `Failed`), though the pipeline is not yet green in-cluster
  (`dvc repro` needs an SCM + data + credentials — PR 3).
- ✅ Externalize configuration and secrets for a cluster — **implemented** (Sprint
  5, PR 3): a `ConfigMap` for non-secret runtime config (`LOG_LEVEL`,
  `MLFLOW_TRACKING_URI`), a **Secret template** for the MLflow/DagsHub credentials
  (`MLFLOW_TRACKING_USERNAME`/`_PASSWORD`, created out-of-band — never committed),
  and a least-privilege `ServiceAccount` with `automountServiceAccountToken: false`
  (the workload needs no Kubernetes API access, so no RBAC is granted). Wired into
  the Job via `envFrom` and verified on a local cluster.
- ✅ Harden the workload with a Kubernetes `securityContext` — **implemented**
  (Sprint 5, PR 4): non-root with an explicit uid/gid `10001` (required — the
  image's `USER` is a name), `allowPrivilegeEscalation: false`, all Linux
  capabilities dropped, and seccomp `RuntimeDefault`, applied at the correct
  pod/container scopes and verified enforced on a local cluster. Read-only root
  filesystem is **evaluated and deliberately deferred** (DVC writes state in-tree
  at the repo root); restricted Pod Security Standard compliance is **not** claimed
  ([ADR-010](decisions/ADR-010-kubernetes-security-hardening.md)).
- ✅ Define resource requests/limits for reproducible scheduling — **implemented**
  (Sprint 5, PR 5): `requests: cpu 250m / mem 256Mi`, `limits: cpu 1 / mem 512Mi`
  (Burstable QoS), chosen from *measured* usage of the real image — the CPU limit
  doubles as the memory-safety control because `GridSearchCV(n_jobs=-1)` sizes
  joblib's worker fan-out from the cgroup CPU quota. The finite-run lifecycle,
  the deliberate absence of health probes, and the failure modes are documented;
  values are **not** production-certified
  ([ADR-011](decisions/ADR-011-kubernetes-resource-lifecycle.md)).

**Expected outcome:** The pipeline runs reproducibly on any conformant cluster,
independent of a developer's local machine.

> **Status.** The workload model is ratified in
> [ADR-009](decisions/ADR-009-kubernetes-workload-model.md) and the `k8s/`
> manifests define a **runnable** Job (namespace + Job + Kustomize + local
> runbook), validated by offline rendering and field assertions **and by an
> executed run on a local Docker Desktop cluster** (2026-08-12) that verified the
> Job lifecycle end to end. PR 3 adds externalized **config/secrets + identity** (a
> `ConfigMap`, an out-of-band `Secret` template, and a least-privilege
> `ServiceAccount` with token automount off), and PR 4 added an **enforced
> `securityContext`** (non-root uid/gid `10001`, no privilege escalation, all
> capabilities dropped, seccomp `RuntimeDefault`; read-only root filesystem
> deferred with evidence — [ADR-010](decisions/ADR-010-kubernetes-security-hardening.md)),
> and PR 5 added **resource requests/limits chosen from measured usage** plus the
> documented lifecycle/probe/failure-mode decisions
> ([ADR-011](decisions/ADR-011-kubernetes-resource-lifecycle.md)) — all verified on
> the local cluster. Still to come in Sprint 5: a green in-cluster `dvc repro`
> (SCM + data + credentials) and CI manifest validation.
> The container **base image** is ratified in
> [ADR-005](decisions/ADR-005-containerization-strategy.md).

---

## Version 5 — Production Cloud Platform

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

## Version 6 — Enterprise MLOps

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
