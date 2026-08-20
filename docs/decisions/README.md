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
| [ADR-028](ADR-028-observability-architecture.md) | Observability Architecture (Prometheus + Grafana; four-layer model; batch-Job metrics via kube-state-metrics) | Accepted (design) | 2026-08-20 |
| [ADR-029](ADR-029-monitoring-foundation.md) | Monitoring Foundation (Sprint 8 PR 2) — minimal hand-written Prometheus + kube-state-metrics + node-exporter; ephemeral TSDB; read-only RBAC; node-exporter PSA exception | Accepted (design) | 2026-08-20 |
| [ADR-030](ADR-030-pipeline-operational-metrics.md) | Pipeline Operational Metrics (Sprint 8 PR 3) — per-stage duration/success via Pushgateway; bounded cardinality; per-run reset lifecycle; operational-vs-MLflow boundary; supersedes ADR-028's "Pushgateway deferred" | Accepted (design) | 2026-08-20 |
| [ADR-031](ADR-031-mlflow-postgres-monitoring.md) | MLflow & PostgreSQL Monitoring (Sprint 8 PR 4) — blackbox-exporter (Layer 3 `/health`) + postgres-exporter with a dedicated `pg_monitor`-only role (Layer 4 up/connections/size) + scoped kubelet volume-stats scrape (PVC-fill); no DB creds in config; eight scrape jobs | Accepted (design) | 2026-08-20 |
| [ADR-032](ADR-032-grafana-dashboards.md) | Grafana Dashboards (Sprint 8 PR 5) — three purpose-built, version-controlled dashboards (EKS/Platform Health, MLOps Pipeline Operations, MLflow Platform Health) mapped to operational questions; file-provisioned datasource + dashboards; hardened internal-only Grafana; stable PromQL, bounded windows; model quality stays in MLflow | Accepted (design) | 2026-08-20 |
| [ADR-033](ADR-033-alerting.md) | Alerting (Sprint 8 PR 6) — eight high-signal Prometheus alert rules encoding the § 6 objectives + § 3 catalogue (pipeline failure via the terminal Failed condition — batch-correct, never "not Running"; OOMKill; MLflow/Postgres down; PVC-fill; memory headroom; crash-looping); severity + summary/description + runbook_url per rule; documented thresholds; promtool unit-tested in CI; no arbitrary alerts; Alertmanager routing deferred | Accepted (design) | 2026-08-20 |
| [ADR-034](ADR-034-network-policies.md) | Least-privilege NetworkPolicies (Sprint 8 PR 7) — evidence-mapped communication matrix; default-deny + explicit allow in both namespaces (PostgreSQL: two peers, zero egress; pipeline can't reach the DB); DNS/Pod Identity/scrape graph preserved; static contract (validate.py §8/M12) + runtime harness with enforcement canary; VPC CNI `enableNetworkPolicy` enabled on EKS; AWS S3 egress bounded to internet-only:443 with the precise "which bucket" control delegated to IAM + a recommended VPC S3 endpoint (documented limitation); no service mesh | Accepted (design) | 2026-08-20 |
| [ADR-035](ADR-035-container-image-scanning.md) | Container-image vulnerability scanning (Sprint 8 PR 8) — Trivy scans both shipped images (`mlops-pipeline` + the `mlflow-server` layer) in the `docker` job, credential-free/AWS-independent; gate on **fixable** HIGH/CRITICAL (`--ignore-unfixed`), non-fixable reported not muted (auto-promotes when a fix ships); specific time-boxed exceptions only (`.trivyignore.yaml`: id + rationale + `expired_at`), no blanket ignore; JSON+table artifact; pinned checksum-verified binary shared with the IaC scan; complements ECR scan-on-push; SBOM/signing deferred | Accepted (design) | 2026-08-20 |
| [ADR-036](ADR-036-sbom-and-image-provenance.md) | SBOM & immutable image provenance (Sprint 8 PR 9) — **CycloneDX** SBOM (Trivy, reused) for both images + a CI assertion that the image's OCI `revision` label **==** the commit SHA (git→image); operator `release-image.sh` captures the immutable ECR **sha256 digest** (cross-checked vs `aws ecr describe-images`) and records the **git commit → tag → digest** chain; **opt-in** digest-pinned deploy in the renderer + `verify-deployed-digest.sh` runtime imageID check; SBOM is a CI artifact, never committed; **cosign signing optional** (keyless `--sign`), enforced gate deferred; credential-free CI preserved (push/verify are operator steps) | Accepted (design) | 2026-08-20 |

> **"Accepted (validated)"** marks the design ADRs (014–018) whose configuration was
> **provisioned and exercised** in the Sprint 6 PR 7 runtime test (2026-08-15) and
> then torn down — they are no longer design-only. See the
> [runtime evidence](../proof/sprint-06-runtime-evidence.md).

## Template

New ADRs should follow the structure used in the existing records:
Context, Decision, Alternatives Considered, and Consequences.
