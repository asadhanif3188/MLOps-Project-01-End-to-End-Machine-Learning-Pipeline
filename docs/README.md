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
| [Pipeline Contract](pipeline-contract.md) | Stage inputs/outputs, artifact ownership, evaluation boundary, external-service boundaries, and reproducibility expectations (current vs target) |
| [Logging Strategy](logging.md) | How the pipeline emits, formats, and persists logs |
| [Exception Strategy](exception-strategy.md) | Exception hierarchy, error propagation, and error logging |
| [Testing Strategy](testing-strategy.md) | Testing philosophy, `tests/` layout, and the testing roadmap |
| [Developer Guide](developer-guide.md) | Local setup and the day-to-day tooling (format, lint, type-check, test, pre-commit) |
| [Containerization Strategy](containerization.md) | Design and build of the container image (dev/prod stages, security, K8s/CI readiness) |
| [Docker Development Workflow](docker-development.md) | Running the local dev environment with Docker Compose (startup, logs, rebuild, troubleshooting) |
| [CI/CD](ci-cd.md) | Continuous integration pipeline (stages, failure strategy, local validation) and the future CD roadmap |
| [Kubernetes Architecture](kubernetes-architecture.md) | The pipeline as a Kubernetes batch `Job`: workload model, boundaries, and the local-vs-production split |
| [Kubernetes Operations](kubernetes-operations.md) | Day-2 operations runbook: deploy/observe/logs/re-run/cleanup, a troubleshooting matrix, and the honest observability posture (local only) |
| [Kubernetes Security](kubernetes-security.md) | Identity, `securityContext`, secret handling, controls→evidence checklist, and what is explicitly not claimed |
| [Cloud Operations](cloud-operations.md) | AWS/EKS lifecycle runbook (auth → init → plan → apply → verify → run → evidence → **destroy → verify clean**), the **AWS cost drivers**, safe teardown, and the honest limitations of the ephemeral validation environment |
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
| [ADR-006](decisions/ADR-006-pipeline-reproducibility.md) | Pipeline Reproducibility as an Engineering Requirement |
| [ADR-007](decisions/ADR-007-held-out-evaluation.md) | Held-Out Evaluation via a Dedicated `split` Stage |
| [ADR-008](decisions/ADR-008-fixture-reproducibility.md) | Fixture-Based Pipeline Reproducibility |
| [ADR-009](decisions/ADR-009-kubernetes-workload-model.md) | Kubernetes Workload Model — `Job`, not `Deployment` |
| [ADR-010](decisions/ADR-010-kubernetes-security-hardening.md) | Kubernetes Workload Security Hardening |
| [ADR-011](decisions/ADR-011-kubernetes-resource-lifecycle.md) | Kubernetes Resource & Lifecycle Management |
| [ADR-012](decisions/ADR-012-kubernetes-manifest-validation.md) | Automated Kubernetes Manifest Validation in CI |
| [ADR-013](decisions/ADR-013-kubernetes-runtime-execution.md) | Kubernetes Runtime Execution (DVC/dataset/MLflow runtime contract) |
| [ADR-014](decisions/ADR-014-terraform-architecture.md) | Terraform Architecture & Foundation |
| [ADR-015](decisions/ADR-015-aws-network-architecture.md) | AWS Network Architecture (VPC, Subnets, AZs, NAT) |
| [ADR-016](decisions/ADR-016-aws-iam-foundation.md) | AWS IAM Foundation for EKS |
| [ADR-017](decisions/ADR-017-eks-platform.md) | Amazon EKS Platform (Cluster, Node Group, Addons) |
| [ADR-018](decisions/ADR-018-aws-eks-deployment-overlay.md) | AWS EKS Deployment Overlay (Cloud Runtime Integration) |
| [ADR-019](decisions/ADR-019-terraform-ci-validation.md) | Terraform CI Validation (no AWS credentials) |
| [ADR-020](decisions/ADR-020-cloud-lifecycle-cost-control.md) | Cloud Environment Lifecycle & Cost Control (Provision → Prove → Destroy) |
| [ADR-021](decisions/ADR-021-terraform-managed-ecr.md) | Terraform-Managed Container Registry (Amazon ECR) — closes H-01 |

### Engineering Reviews

| Review | Description |
|--------|-------------|
| [Sprint 2 — Production Readiness](reviews/sprint-02-engineering-review.md) | Principal-engineer review that drove the Sprint 2 engineering-excellence work (findings H-1..H-6) |
| [Sprint 2 — Final Validation](reviews/sprint-02-final-review.md) | Release-readiness validation for v1.1.0: checks, remaining debt, risks, and Sprint 3 recommendations |
| [Sprint 3 — Final Validation](reviews/sprint-03-final-review.md) | Release-readiness validation for v1.2.0 (containerization & CI): checks, remaining debt, risks, and Sprint 4 recommendations |
| [Sprint 4 — Final Validation](reviews/sprint-04-final-review.md) | Release-readiness validation for v1.3.0 (pipeline correctness & reproducibility): correctness, lineage, config consistency, stage contracts, testability, CI, and known limitations |

### Retrospectives

| Retrospective | Description |
|---------------|-------------|
| [Sprint 3 — v1.2.0](retrospectives/sprint-03-retrospective.md) | Containerization & CI: what was planned, delivered, decided, and deliberately deferred, plus lessons learned |
| [Sprint 4 — v1.3.0](retrospectives/sprint-04-retrospective.md) | Pipeline correctness & reproducibility: planned vs delivered, changes during implementation, decisions, lessons, and deferred work |
| [Sprint 5 — v1.4.0](retrospectives/sprint-05-retrospective.md) | Kubernetes platform engineering: workload model → security → resources → CI validation → operations/proof → **green in-cluster execution** (PR 8); the "structurally valid ≠ runtime-complete" lesson |

### Proof Assessments

| Assessment | Description |
|------------|-------------|
| [Sprint 4 — Proof Impact](proof/sprint-04-proof-impact.md) | Evidence-based statement of what the project can credibly claim after Sprint 4 that it could not after Sprint 3, with remaining limitations |
| [Sprint 5 — Proof Impact](proof/sprint-05-proof-impact.md) | Evidence-based statement of the Kubernetes platform-engineering claims after Sprint 5 (workload model, security, resources, validation, operations) with a conservative Before/After and explicit known limitations |
| [Sprint 6 — Proof Impact](proof/sprint-06-proof-impact.md) | Evidence-based statement of the cloud-platform claims after Sprint 6 (Terraform IaC, least-privilege cloud IAM, credential-free CI gate, a green run on real EKS, live-pod security, verified teardown) with a conservative Before/After and explicit known limitations |
| [Sprint 6 — Runtime Evidence](proof/sprint-06-runtime-evidence.md) | Redacted record of the PR 7 cloud integration test: 29 resources applied, Job `Complete` (exit 0) on real EKS, six security controls verified live, then destroyed and verified clean |

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
- **Deployment Guide** — *local* Kubernetes deployment
  ([`k8s/README.md`](../k8s/README.md), [Kubernetes Operations](kubernetes-operations.md))
  and an *ephemeral cloud/EKS* lifecycle ([Cloud Operations](cloud-operations.md))
  now exist. A **production** deployment (persistent, HA, GitOps) remains future work
  (Roadmap v5). <!-- TODO -->
- **Production Monitoring & Operations** — an observability stack (metrics, tracing,
  alerting) for enterprise MLOps (Roadmap v6). The current local-cluster operations
  posture is documented in [Kubernetes Operations](kubernetes-operations.md). <!-- TODO -->
