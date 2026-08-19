# Sprint 7 — Proof-Impact Assessment (Cloud-Native MLOps Hardening)

- **Date:** 2026-08-19
- **Status:** Unreleased (Sprint 7). **No release tag is cut in this PR** — this
  assessment covers the work staged under `[Unreleased]` in the
  [CHANGELOG](../../CHANGELOG.md); a version tag is a separate release step.
- **Headline:** The Sprint 6 cloud platform — a Terraform-defined EKS environment the
  pipeline ran on once — has been **hardened into a cloud-native MLOps platform**: the
  container registry is Terraform-managed (**two** ECR repos), the EKS API is
  **private by default** with **explicit access entries** (no creator-admin), the VPC
  CNI and both application workloads draw AWS access from **EKS Pod Identity** (no
  static keys), Kubernetes Secrets are **KMS envelope-encrypted**, the dataset is
  retrieved at runtime from a **private, SSE-KMS S3 bucket** (not a ConfigMap), and
  experiment tracking runs on an **in-cluster MLflow platform** (self-hosted server +
  PostgreSQL + S3) instead of external DagsHub. The whole platform was **provisioned
  from scratch on real EKS 1.35, ran the pipeline to completion (Job `Complete`, exit
  0), and was destroyed and verified clean.** What is **not** claimed remains
  production — see [§5](#5-what-still-cannot-be-claimed).
- **Related:** [Sprint 7 Release Gate](sprint-07-release-gate.md),
  [Sprint 7 Runtime Evidence](sprint-07-runtime-evidence.md),
  [Cloud Operations](../cloud-operations.md),
  [MLflow Platform](../mlflow-platform.md), [Dataset](../dataset.md),
  [`terraform/README.md`](../../terraform/README.md),
  [ADR-021](../decisions/ADR-021-terraform-managed-ecr.md),
  [ADR-022](../decisions/ADR-022-eks-secure-api-access.md),
  [ADR-023](../decisions/ADR-023-eks-access-control.md),
  [ADR-024](../decisions/ADR-024-vpc-cni-pod-identity.md),
  [ADR-025](../decisions/ADR-025-eks-secrets-kms-encryption.md),
  [ADR-026](../decisions/ADR-026-in-cluster-mlflow-platform.md),
  [ADR-027](../decisions/ADR-027-s3-dataset-runtime-retrieval.md)

> This document answers one question with evidence:
> **What can this project credibly claim after Sprint 7 that it could not after
> Sprint 6?** "Credibly" is the operative word — every claim below points to a
> Terraform file, an ADR, a manifest, or the redacted [runtime
> evidence](sprint-07-runtime-evidence.md) record a reviewer can check.
> [§5](#5-what-still-cannot-be-claimed) lists what the repository does **not**
> support, so the credible claims are not diluted by overreach.

---

## 1. The shift, in one line

| | Claim licensed by the repository |
|-|----------------------------------|
| **After Sprint 6** | "I defined a managed cloud platform (VPC + IAM + EKS) as Terraform and **ran the pipeline on real EKS to completion** — but the ECR repo was created out-of-band, the EKS API was public, cluster access came from creator-admin, the VPC CNI used the node role, Secrets had no customer-managed encryption, the dataset arrived via a **ConfigMap**, and experiment tracking pointed at **DagsHub** (exercised only against an offline file store)." |
| **After Sprint 7** | "I **closed every HIGH/MEDIUM Sprint 6 review finding** and made the platform cloud-native: **Terraform-managed ECR** (two repos), a **private-by-default EKS API** that rejects `0.0.0.0/0`, **explicit EKS access entries** with creator-admin off, **EKS Pod Identity** for the VPC CNI and both app workloads (no static keys), **KMS-encrypted Secrets**, a **private SSE-KMS S3 dataset** retrieved at runtime by an init container, and an **in-cluster MLflow platform** (server + PostgreSQL + S3) replacing DagsHub — then **provisioned the whole thing on real EKS 1.35 from scratch, ran the Job to completion (exit 0) with all AWS access on workload identity, and destroyed it verified-clean.** Still bounded, honestly, to a short-lived single-operator validation environment: not production, not HA, not multi-region, no GitOps, no remote state, no DR, no production observability." |

Sprint 6's claim was about **provisioning and running on managed cloud**. Sprint 7's
claim is about **hardening that platform to a defensible security and data
architecture** — least-exposure control plane, workload identity everywhere,
customer-managed encryption, a professional cloud data path, and self-hosted tracking
— evidenced by a full-platform run and honest about what is still deferred.

---

## 2. New credible claims, with evidence

Each row was **not** defensible after Sprint 6 and **is** after Sprint 7. Runtime
figures are from the full-platform EKS run (`us-east-1`, 2026-08-19), account ID and
operator IP redacted; the environment was destroyed the same session.

### 2.1 "The container registry is Terraform-managed, not created out-of-band." (H-01)
- Two private ECR repositories (`mlops-pipeline` + `mlflow-server`), **immutable
  tags**, scan-on-push, AES256 at rest, a retention lifecycle policy, and
  `force_delete` so `terraform destroy` reclaims them — the manual `aws ecr
  create-repository` / `delete-repository --force` steps are gone.
- **Evidence:** [`terraform/ecr.tf`](../../terraform/ecr.tf),
  [ADR-021](../decisions/ADR-021-terraform-managed-ecr.md);
  [runtime evidence §3](sprint-07-runtime-evidence.md#3-ecr-image-verification)
  (2 images pushed, immutable tags).

### 2.2 "The EKS API server is private by default and can never be opened to the world." (H-02)
- `endpoint_public_access` defaults **false**; public access is an opt-in that
  **requires** a scoped CIDR allow-list; an unrestricted `0.0.0.0/0` is **rejected**
  by variable validation + `lifecycle` preconditions (fails `plan`, `apply`, and the
  offline `terraform test`), not merely discouraged.
- **Evidence:** [`terraform/eks.tf`](../../terraform/eks.tf) (`vpc_config`,
  preconditions), [ADR-022](../decisions/ADR-022-eks-secure-api-access.md);
  [runtime evidence §14–15](sprint-07-runtime-evidence.md#14-job-completion--15-pod-security-context)
  (endpoint private + scoped `/32`, never `0.0.0.0/0`).

### 2.3 "Cluster access is by explicit, scoped access entries — no implicit creator-admin." (H-03)
- `authentication_mode = API` (access entries only — no `aws-auth` ConfigMap
  backdoor) and `bootstrap_cluster_creator_admin_permissions = false`, so the
  provisioning principal gets **no** implicit admin; human/automation access is
  granted only by declared `aws_eks_access_entry` + policy associations.
- **Evidence:** [`terraform/eks.tf`](../../terraform/eks.tf) (`access_config`, access
  entries), [ADR-023](../decisions/ADR-023-eks-access-control.md);
  [runtime evidence §2](sprint-07-runtime-evidence.md#2-eks-verification).

### 2.4 "The VPC CNI and both app workloads use EKS Pod Identity — no static AWS keys." (M-01)
- The VPC CNI's `aws-node` service account assumes a **dedicated** role via EKS Pod
  Identity (CNI permissions removed from the node instance profile); the MLflow server
  and the pipeline likewise assume their own least-privilege roles. Four Pod Identity
  associations, zero static keys on the cluster.
- **Evidence:** [`terraform/eks.tf`](../../terraform/eks.tf),
  [`terraform/iam.tf`](../../terraform/iam.tf),
  [`terraform/s3.tf`](../../terraform/s3.tf),
  [`terraform/datasets.tf`](../../terraform/datasets.tf),
  [ADR-024](../decisions/ADR-024-vpc-cni-pod-identity.md);
  [runtime evidence §4](sprint-07-runtime-evidence.md#4-workload-identity-eks-pod-identity)
  (4 live associations, no static keys).

### 2.5 "Kubernetes Secrets are envelope-encrypted with a customer-managed KMS key." (M-02)
- A dedicated symmetric CMK (annual rotation on) wired into the cluster's
  `encryption_config` (`resources = ["secrets"]`) with a least-privilege key policy —
  the association, not merely a key that exists, is what makes the encryption real.
- **Evidence:** [`terraform/kms.tf`](../../terraform/kms.tf),
  [`terraform/eks.tf`](../../terraform/eks.tf) (`encryption_config`),
  [ADR-025](../decisions/ADR-025-eks-secrets-kms-encryption.md);
  [runtime evidence §14–15](sprint-07-runtime-evidence.md#14-job-completion--15-pod-security-context)
  (EKS Secrets KMS-encrypted).

### 2.6 "Experiment tracking runs on an in-cluster MLflow platform, not external DagsHub." (PR 6)
- A stateless MLflow Tracking Server (Deployment, `--serve-artifacts`) backed by a
  **PostgreSQL** StatefulSet (metadata) and an **S3** artifact store (MinIO locally,
  Amazon S3 on EKS via Pod Identity), ClusterIP-internal, credential-free to clients.
  Persistence is proven by surviving stateful-pod recreation.
- **Evidence:** [`k8s/base/mlflow/`](../../k8s/base/mlflow/),
  [`terraform/s3.tf`](../../terraform/s3.tf), [MLflow Platform](../mlflow-platform.md),
  [ADR-026](../decisions/ADR-026-in-cluster-mlflow-platform.md);
  [runtime evidence §6, §9–13](sprint-07-runtime-evidence.md#6-in-cluster-mlflow-platform)
  (2 runs `FINISHED`, metadata in PostgreSQL, 7 artifacts in SSE-KMS S3).

### 2.7 "The runtime dataset comes from a private, encrypted, versioned S3 bucket — not a ConfigMap." (M-04)
- A private, all-public-access-blocked, versioned, SSE-KMS S3 bucket holds the
  dataset; the pipeline's `fetch-dataset` init container downloads it via a read-only
  Pod Identity role and **verifies its SHA-256** against a pinned identity before any
  training runs — no ConfigMap, no baked-in data, no hostPath.
- **Evidence:** [`terraform/datasets.tf`](../../terraform/datasets.tf),
  [`k8s/overlays/aws/job-cloud.yaml`](../../k8s/overlays/aws/job-cloud.yaml),
  [Dataset](../dataset.md),
  [ADR-027](../decisions/ADR-027-s3-dataset-runtime-retrieval.md);
  [runtime evidence §5–6](sprint-07-runtime-evidence.md#5-s3-dataset-runtime-retrieval-integrity-pinned)
  (sha256 == pinned identity, verified in-cluster).

### 2.8 "The full hardened platform runs the pipeline to completion on real EKS, then destroys clean."
- `terraform apply` created **63** resources from a clean-slate account (a second
  `apply` added **2 more** — the operator access entry + policy association — for the
  documented kubectl access path, bringing the environment to **65** managed
  resources); the cluster came up **ACTIVE** (control plane **v1.35**, 1 node
  **Ready** in a private subnet); the Job reached **`Complete`** with the successful
  pod **exit 0**, all four DVC stages ran, and the security controls were verified on
  the live pod. Teardown: `terraform destroy` → **65 destroyed**, state empty,
  buckets/repos gone, verified three ways.
- **Evidence:** [Sprint 7 Runtime Evidence](sprint-07-runtime-evidence.md) (§§1–15),
  [Cloud Operations](../cloud-operations.md),
  [ADR-020](../decisions/ADR-020-cloud-lifecycle-cost-control.md).

---

## 3. Proof / evidence

The dimensions Sprint 7 set out to establish, each mapped to its enforcing artifact
and its executed/verified evidence.

| Dimension | Enforcing artifact | Verified evidence |
|---|---|---|
| **Terraform-managed ECR** (H-01) | [`ecr.tf`](../../terraform/ecr.tf) ([ADR-021](../decisions/ADR-021-terraform-managed-ecr.md)) | 2 repos, immutable tags, scan-on-push, retention; 2 images pushed ([evidence §3](sprint-07-runtime-evidence.md#3-ecr-image-verification)). |
| **Secure-by-default EKS API** (H-02) | [`eks.tf`](../../terraform/eks.tf) preconditions ([ADR-022](../decisions/ADR-022-eks-secure-api-access.md)) | Private default; `0.0.0.0/0` rejected; live endpoint private + scoped `/32`. |
| **Explicit access entries** (H-03) | [`eks.tf`](../../terraform/eks.tf) `access_config` ([ADR-023](../decisions/ADR-023-eks-access-control.md)) | `authentication_mode = API`, creator-admin off ([evidence §2](sprint-07-runtime-evidence.md#2-eks-verification)). |
| **Workload identity (Pod Identity)** (M-01) | [`iam.tf`](../../terraform/iam.tf) + associations ([ADR-024](../decisions/ADR-024-vpc-cni-pod-identity.md)) | 4 live associations, no static keys ([evidence §4](sprint-07-runtime-evidence.md#4-workload-identity-eks-pod-identity)). |
| **KMS Secret encryption** (M-02) | [`kms.tf`](../../terraform/kms.tf) + `encryption_config` ([ADR-025](../decisions/ADR-025-eks-secrets-kms-encryption.md)) | 3 CMKs (rotation on); EKS Secrets KMS-encrypted. |
| **In-cluster MLflow** (PR 6) | [`k8s/base/mlflow/`](../../k8s/base/mlflow/) ([ADR-026](../decisions/ADR-026-in-cluster-mlflow-platform.md)) | Server Ready, 2 runs `FINISHED`, PostgreSQL + SSE-KMS S3 ([evidence §6, §13](sprint-07-runtime-evidence.md#6-in-cluster-mlflow-platform)). |
| **S3 dataset runtime retrieval** (M-04) | [`datasets.tf`](../../terraform/datasets.tf) ([ADR-027](../decisions/ADR-027-s3-dataset-runtime-retrieval.md)) | sha256 == pinned identity, in-cluster ([evidence §5](sprint-07-runtime-evidence.md#5-s3-dataset-runtime-retrieval-integrity-pinned)). |
| **DVC data-flow correctness** (PR 9) | `dvc.yaml` + contract tests | Declared DAG == traced execution ([DVC correction evidence](sprint-07-dvc-dataflow-correction-evidence.md)). |
| **Full-platform cloud run** | the executed Job | Job **`Complete`**, pod **exit 0**, 4 stages, all AWS via Pod Identity. |
| **Ephemeral lifecycle & teardown** | runbook + destroy ([ADR-020](../decisions/ADR-020-cloud-lifecycle-cost-control.md)) | `destroy` → **65 destroyed**; verified clean three ways. |
| **Offline security contracts** | `terraform test` + `k8s/validate.py` | `terraform test` **42/42** (clean checkout; see [release gate §6](sprint-07-release-gate.md#6-the-terraform-test-observation-non-blocking)); `k8s/validate.py` 158/158 (PR 11). |

---

## 4. Before / After (conservative)

"✅" only where the repository has the artifact **and** the evidence; "⬜" where
deferred; "❌" where not attempted/claimed.

| Capability | After Sprint 6 | After Sprint 7 |
|---|---|---|
| Green run on real managed EKS | ✅ | ✅ (full hardened platform — [evidence](sprint-07-runtime-evidence.md)) |
| Terraform-managed container registry (ECR) | ❌ (out-of-band) | ✅ (2 repos — [ADR-021](../decisions/ADR-021-terraform-managed-ecr.md)) |
| Private-by-default EKS API, no `0.0.0.0/0` | ❌ (public) | ✅ ([ADR-022](../decisions/ADR-022-eks-secure-api-access.md)) |
| Explicit EKS access entries, no creator-admin | ❌ | ✅ ([ADR-023](../decisions/ADR-023-eks-access-control.md)) |
| Workload identity via EKS Pod Identity, no static keys | ❌ (CNI on node role) | ✅ ([ADR-024](../decisions/ADR-024-vpc-cni-pod-identity.md)) |
| KMS-encrypted Kubernetes Secrets | ❌ (AWS-owned key) | ✅ ([ADR-025](../decisions/ADR-025-eks-secrets-kms-encryption.md)) |
| In-cluster MLflow (server + PostgreSQL + S3) | ❌ (DagsHub SaaS) | ✅ ([ADR-026](../decisions/ADR-026-in-cluster-mlflow-platform.md)) |
| Runtime dataset from private SSE-KMS S3 (not ConfigMap) | ❌ (ConfigMap) | ✅ ([ADR-027](../decisions/ADR-027-s3-dataset-runtime-retrieval.md)) |
| Experiment tracking exercised live on the cloud run | ❌ (offline file store) | ✅ (2 runs `FINISHED`, artifacts in S3) |
| Terraform remote state backend (S3 + locking) | ❌ | ⬜ not claimed (local state by design — [ADR-014](../decisions/ADR-014-terraform-architecture.md)) |
| GitOps / continuous delivery to the cluster | ❌ | ⬜ not claimed (CI validates, does not deploy) |
| Multi-region / disaster recovery | ❌ | ⬜ not claimed |
| Enterprise HA/DR (control plane, DB, NAT) | ❌ | ⬜ not claimed (single node/NAT, single-writer DB) |
| Production observability (metrics/tracing/alerting) | ❌ | ⬜ not claimed |
| Model serving / inference endpoint | ❌ | ⬜ not claimed (roadmap v6) |

---

## 5. What still **cannot** be claimed

Documented so none is accidentally implied. These remain **deferred**:

- **❌ GitOps / continuous delivery.** No Argo CD / Flux; deployment is an operator
  running `kubectl apply`. CI validates manifests and IaC statically and deploys
  nothing.
- **❌ Terraform remote state.** Local state only (`terraform/terraform.tfstate`,
  git-ignored); a remote S3 + lock-table backend is a documented migration path, not
  implemented ([ADR-014](../decisions/ADR-014-terraform-architecture.md)).
- **❌ Multi-region.** One region (`us-east-1`); nothing spans or fails over between
  regions.
- **❌ Enterprise HA / disaster recovery.** Single node, single NAT gateway, a
  single-writer PostgreSQL StatefulSet, single Job — no HA topology, no backup/restore,
  no state replication, no RTO/RPO. Teardown is intentional deletion, not a recovery
  drill.
- **❌ Full observability stack.** No Prometheus/Grafana, tracing, alerting, or log
  aggregation; diagnosis is `kubectl` + structured logs.
- **❌ Production deployment / production-certified capacity.** The environment is
  ephemeral (`provision → prove → destroy`), single-operator, own-account; sizing is
  the minimum to prove the claim.
- **⬜ Read-only root filesystem** and **❌ Restricted Pod Security Standard
  certification** — deferred with recorded evidence
  ([ADR-010](../decisions/ADR-010-kubernetes-security-hardening.md)); controls are
  applied and a live API server admitted the pod, but no admission-policy engine
  ratifies the profile.
- **⬜ DVC data/model remote migration off DagsHub.** Sprint 7 removed DagsHub from
  the **experiment-tracking** path; the DVC **data/model** remote in
  [`.dvc/config`](../../.dvc/config) still uses DagsHub — a separate versioning concern
  (roadmap v5), not on any in-cluster runtime path.

---

## 6. Known limitations (explicit)

- **Not production** — a short-lived, single-operator validation environment.
- **Single environment** — one account, one region, local Terraform state, no
  dev/staging/prod split.
- **Limited scale** — 1 (default 2) `t3.medium` on-demand node, one shared NAT, two
  AZs, one batch `Job`, a ~23 KiB dataset.
- **No GitOps, no remote state, no multi-region, no HA/DR, no production
  observability** — all deferred (see [§5](#5-what-still-cannot-be-claimed)).
- **Transient first-pod failure on a cold cluster** — MLflow's first-boot DB migration
  can exceed the `wait-for-mlflow` init budget; the Job's `backoffLimit` retry covers
  it, but the first pod burns a retry ([runtime evidence § Failures & fixes](sprint-07-runtime-evidence.md#failures--fixes)).
- **Restricted PSS not admission-ratified**; **read-only root filesystem deferred** —
  both with recorded evidence (ADR-010).

---

## 7. The honest one-paragraph statement

> Building on a Terraform-defined EKS platform proven to run the pipeline in Sprint 6,
> I **closed every HIGH and MEDIUM finding from the Sprint 6 review** and hardened the
> platform into a cloud-native MLOps architecture: the container registry is now
> **Terraform-managed** (two immutable-tag ECR repos), the EKS API is **private by
> default** and structurally refuses `0.0.0.0/0`, cluster access is by **explicit
> access entries** with no creator-admin, the VPC CNI and both application workloads
> draw AWS credentials from **EKS Pod Identity** with **no static keys**, Kubernetes
> Secrets are **envelope-encrypted with a customer-managed KMS key**, the dataset is
> retrieved at runtime from a **private, SSE-KMS, versioned S3 bucket** and
> integrity-checked before use, and experiment tracking runs on an **in-cluster MLflow
> platform** (server + PostgreSQL + S3) that replaced external DagsHub. I then
> **provisioned the entire platform on real EKS 1.35 from a clean-slate account, ran
> the same pipeline to completion (Job `Complete`, exit 0, all four stages, all AWS
> access on workload identity, runs logged to PostgreSQL with artifacts in SSE-KMS
> S3), verified the security controls on the live pod, and destroyed the environment
> verified-clean.** Deliberately, I do **not** claim GitOps, Terraform remote state,
> multi-region, enterprise HA/DR, production observability, or a production deployment
> — each is documented as deferred. The claim is about **cloud-native platform
> hardening evidenced on an ephemeral validation environment**, bounded and torn down —
> not about operating production cloud infrastructure.

That paragraph is fully supported by the repository and its redacted evidence. Its
restraint — naming the ephemeral, single-region, deferred-GitOps/remote-state/HA/DR/
observability boundaries out loud — is part of what makes the rest of it credible.
