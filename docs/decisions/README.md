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
| [ADR-012](ADR-012-kubernetes-manifest-validation.md) | Automated Kubernetes Manifest Validation in CI (kustomize + kubeconform + project checks) | Accepted | 2026-08-12 |
| [ADR-013](ADR-013-kubernetes-runtime-execution.md) | Kubernetes Runtime Execution — the DVC/dataset/MLflow runtime contract | Accepted | 2026-08-14 |
| [ADR-014](ADR-014-terraform-architecture.md) | Terraform Architecture & Foundation (structure, versions, tagging, local state) | Accepted (validated) | 2026-08-14 |
| [ADR-015](ADR-015-aws-network-architecture.md) | AWS Network Architecture (VPC, Subnets, AZs, NAT) | Accepted (validated) | 2026-08-14 |
| [ADR-016](ADR-016-aws-iam-foundation.md) | AWS IAM Foundation for EKS (Cluster & Node Roles) | Accepted (validated) | 2026-08-14 |
| [ADR-017](ADR-017-eks-platform.md) | Amazon EKS Platform (Cluster, Managed Node Group, Core Addons) | Accepted (validated) | 2026-08-14 |
| [ADR-018](ADR-018-aws-eks-deployment-overlay.md) | AWS EKS Deployment Overlay (Cloud Runtime Integration) | Accepted (validated) | 2026-08-14 |
| [ADR-019](ADR-019-terraform-ci-validation.md) | Terraform CI Validation (fmt/init/validate/lint/IaC scan, no AWS) | Accepted | 2026-08-14 |
| [ADR-020](ADR-020-cloud-lifecycle-cost-control.md) | Cloud Environment Lifecycle & Cost Control (Ephemeral, Provision → Prove → Destroy) | Accepted | 2026-08-15 |
| [ADR-021](ADR-021-terraform-managed-ecr.md) | Terraform-Managed Container Registry (Amazon ECR) — closes H-01 | Accepted (design) | 2026-08-17 |
| [ADR-022](ADR-022-eks-secure-api-access.md) | Secure-by-Default EKS API Access (private default, no `0.0.0.0/0`) — closes H-02 | Accepted (design) | 2026-08-17 |
| [ADR-023](ADR-023-eks-access-control.md) | Explicit EKS Access Entries (scoped policies, no creator-admin) — closes H-03 | Accepted (design) | 2026-08-17 |
| [ADR-024](ADR-024-vpc-cni-pod-identity.md) | VPC CNI Identity via EKS Pod Identity (dedicated role, off the node role) — closes M-01 | Accepted (design) | 2026-08-17 |
| [ADR-025](ADR-025-eks-secrets-kms-encryption.md) | EKS Secret Envelope Encryption with a Customer-Managed KMS Key — closes M-02 | Accepted (design) | 2026-08-17 |
| [ADR-026](ADR-026-in-cluster-mlflow-platform.md) | Persistent In-Cluster MLflow Tracking Platform (PostgreSQL + S3); removes DagsHub | Accepted (validated) | 2026-08-18 |
| [ADR-027](ADR-027-s3-dataset-runtime-retrieval.md) | S3-Backed Runtime Dataset Retrieval (init container + Pod Identity) — closes M-04 | Accepted (validated) | 2026-08-19 |

> **"Accepted (validated)"** marks the design ADRs (014–018) whose configuration was
> **provisioned and exercised** in the Sprint 6 PR 7 runtime test (2026-08-15) and
> then torn down — they are no longer design-only. See the
> [runtime evidence](../proof/sprint-06-runtime-evidence.md).

## Template

New ADRs should follow the structure used in the existing records:
Context, Decision, Alternatives Considered, and Consequences.
