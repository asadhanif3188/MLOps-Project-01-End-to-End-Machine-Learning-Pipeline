# Documentation

Welcome to the documentation for the **End-to-End Machine Learning Pipeline**.
This page is the entry point to all project documentation.

For a general project introduction and quick start, see the
[repository README](../README.md).

---

## Overview

This project is a reproducible, batch ML pipeline that trains a Random Forest
classifier on the Pima Indians Diabetes dataset. It uses **DVC** for
data/pipeline versioning and the project's **in-cluster MLflow platform**
(self-hosted server + PostgreSQL + S3; [ADR-026](decisions/ADR-026-in-cluster-mlflow-platform.md))
for experiment tracking. Start with the [architecture](architecture.md) for a
system-level view, or the [project structure](project-structure.md) to navigate the
repository.

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
| [Observability & Operations](observability.md) | 🚧 The Sprint 8 observability *architecture* (design only): the Prometheus/Grafana stack, the four-layer signal catalogue, how an ephemeral batch `Job`'s metrics stay queryable, SLO-style objectives, deferred areas, and the PR 2–6 delivery plan |
| [Monitoring Operations](monitoring-operations.md) | 🚧 Runbook for the Sprint 8 metrics foundation (Prometheus + kube-state-metrics + node-exporter, PR 2) **and the pipeline's operational metrics** (Pushgateway per-stage duration/success, PR 3): deploy, reach Prometheus, run a PromQL query, troubleshoot, and clean up (manifests defined & validated; not yet runtime-proven) |
| [Cloud Operations](cloud-operations.md) | AWS/EKS lifecycle runbook (auth → init → plan → apply → verify → run → evidence → **destroy → verify clean**), the **AWS cost drivers**, safe teardown, and the honest limitations of the ephemeral validation environment |
| [MLflow Platform](mlflow-platform.md) | The in-cluster MLflow tracking platform (server + PostgreSQL + S3): deploy, operate, the persistence test, and the AWS (EKS Pod Identity) notes |
| [Dataset](dataset.md) | Dataset identity, version, and integrity — the S3 runtime-retrieval source of truth and how the `fetch-dataset` init container verifies it |
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
| [ADR-022](decisions/ADR-022-eks-secure-api-access.md) | Secure-by-Default EKS API Access (private default, no `0.0.0.0/0`) — closes H-02 |
| [ADR-023](decisions/ADR-023-eks-access-control.md) | EKS Access Control via Explicit Access Entries (no creator-admin) — closes H-03 |
| [ADR-024](decisions/ADR-024-vpc-cni-pod-identity.md) | VPC CNI Identity via EKS Pod Identity (off the node role) — closes M-01 |
| [ADR-025](decisions/ADR-025-eks-secrets-kms-encryption.md) | EKS Secrets Envelope Encryption with a Customer-Managed KMS Key — closes M-02 |
| [ADR-026](decisions/ADR-026-in-cluster-mlflow-platform.md) | In-Cluster MLflow Platform (server + PostgreSQL + S3) |
| [ADR-027](decisions/ADR-027-s3-dataset-runtime-retrieval.md) | S3 Dataset Runtime Retrieval via EKS Pod Identity — closes M-04 |
| [ADR-028](decisions/ADR-028-observability-architecture.md) | Observability Architecture (Prometheus + Grafana; four-layer model; batch-Job metrics via kube-state-metrics) |
| [ADR-029](decisions/ADR-029-monitoring-foundation.md) | Monitoring Foundation (Sprint 8 PR 2) — minimal hand-written Prometheus + KSM + node-exporter; ephemeral TSDB; read-only RBAC; node-exporter PSA exception |
| [ADR-030](decisions/ADR-030-pipeline-operational-metrics.md) | Pipeline Operational Metrics (Sprint 8 PR 3) — per-stage duration/success via Pushgateway; bounded cardinality; per-run reset lifecycle; operational-vs-MLflow boundary |
| [ADR-031](decisions/ADR-031-mlflow-postgres-monitoring.md) | MLflow & PostgreSQL Monitoring (Sprint 8 PR 4) — blackbox `/health` (Layer 3) + postgres-exporter with a dedicated read-only role (Layer 4) + scoped kubelet PVC-fill scrape; no DB credentials in config |

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
| [Sprint 7 — v1.6.0](retrospectives/sprint-07-retrospective.md) | Cloud-native MLOps hardening: closing all seven Sprint 6 HIGH/MEDIUM findings (ECR IaC, secure-by-default EKS API, access entries, CNI Pod Identity, KMS Secrets, in-cluster MLflow, S3 dataset), proven on real EKS; the "secure defaults have an operational tax" lesson |

### Proof Assessments

| Assessment | Description |
|------------|-------------|
| [Sprint 4 — Proof Impact](proof/sprint-04-proof-impact.md) | Evidence-based statement of what the project can credibly claim after Sprint 4 that it could not after Sprint 3, with remaining limitations |
| [Sprint 5 — Proof Impact](proof/sprint-05-proof-impact.md) | Evidence-based statement of the Kubernetes platform-engineering claims after Sprint 5 (workload model, security, resources, validation, operations) with a conservative Before/After and explicit known limitations |
| [Sprint 6 — Proof Impact](proof/sprint-06-proof-impact.md) | Evidence-based statement of the cloud-platform claims after Sprint 6 (Terraform IaC, least-privilege cloud IAM, credential-free CI gate, a green run on real EKS, live-pod security, verified teardown) with a conservative Before/After and explicit known limitations |
| [Sprint 6 — Runtime Evidence](proof/sprint-06-runtime-evidence.md) | Redacted record of the PR 7 cloud integration test: 29 resources applied, Job `Complete` (exit 0) on real EKS, six security controls verified live, then destroyed and verified clean |
| [Sprint 7 — Proof Impact](proof/sprint-07-proof-impact.md) | Evidence-based statement of the hardened cloud-native claims after Sprint 7 (Terraform-managed ECR, secure-by-default EKS API, access entries, VPC CNI Pod Identity, KMS-encrypted Secrets, in-cluster MLflow, S3 dataset) with a conservative Before/After and explicit deferred items |
| [Sprint 7 — Runtime Evidence](proof/sprint-07-runtime-evidence.md) | Redacted record of the full-platform EKS run: 63 resources applied, EKS Pod Identity workload identity, in-cluster MLflow (PostgreSQL + SSE-KMS S3), Job `Complete` (exit 0), then destroyed and verified clean |
| [Sprint 7 — Release Gate](proof/sprint-07-release-gate.md) | Final Sprint 7 release-readiness gate: full local toolchain results, individual assessment of all seven Sprint 6 HIGH/MEDIUM findings (7/7 closed), runtime-chain verification, no-GitOps/no-remote-state confirmation; verdict **CONDITIONAL PASS**, no blockers, recommended **v1.6.0** |

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
  alerting) for enterprise MLOps (Roadmap v6). The Sprint 8 observability
  *architecture* is now defined in [Observability & Operations](observability.md)
  and [ADR-028](decisions/ADR-028-observability-architecture.md) (design only — no
  component deployed yet); the current local-cluster operations posture is
  documented in [Kubernetes Operations](kubernetes-operations.md). <!-- TODO -->
