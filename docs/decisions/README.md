# Architecture Decision Records (ADRs)

This directory records significant architectural and engineering decisions using
short, dated records. Each ADR captures the **context**, the **decision**, the
**alternatives considered**, and the resulting **consequences**.

## Index

| ID | Title | Status | Date |
|----|-------|--------|------|
| [ADR-001](ADR-001-repository-structure.md) | Repository Structure | Accepted | 2026-08-01 |
| [ADR-002](ADR-002-why-mlflow.md) | Why MLflow | Accepted | 2026-08-01 |
| [ADR-003](ADR-003-why-dvc.md) | Why DVC | Accepted | 2026-08-01 |
| [ADR-004](ADR-004-python-quality-toolchain.md) | Python Quality Toolchain (Ruff, mypy, pytest, pre-commit) | Accepted | 2026-08-02 |
| [ADR-005](ADR-005-containerization-strategy.md) | Containerization Strategy | Accepted (design) | 2026-08-02 |
| [ADR-006](ADR-006-pipeline-reproducibility.md) | Pipeline Reproducibility as an Engineering Requirement | Accepted (design) | 2026-08-05 |
| [ADR-007](ADR-007-held-out-evaluation.md) | Held-Out Evaluation via a Dedicated `split` Stage | Accepted | 2026-08-09 |
| [ADR-008](ADR-008-fixture-reproducibility.md) | Fixture-Based Pipeline Reproducibility (Committed `dvc.lock` + CI `dvc repro`) | Accepted | 2026-08-09 |
| [ADR-009](ADR-009-kubernetes-workload-model.md) | Kubernetes Workload Model — `Job`, not `Deployment` | Accepted (design) | 2026-08-12 |
| [ADR-010](ADR-010-kubernetes-security-hardening.md) | Kubernetes Workload Security Hardening (Pod/Container `securityContext`) | Accepted | 2026-08-12 |
| [ADR-011](ADR-011-kubernetes-resource-lifecycle.md) | Kubernetes Resource & Lifecycle Management (Requests/Limits, Backoff, No Probes) | Accepted | 2026-08-12 |

## Template

New ADRs should follow the structure used in the existing records:
Context, Decision, Alternatives Considered, and Consequences.
