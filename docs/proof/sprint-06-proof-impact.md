# Sprint 6 — Proof-Impact Assessment (Terraform Cloud Platform Foundation)

> **Superseded by Sprint 7.** This is a **dated snapshot** of the credible claims
> *after Sprint 6*. Several of its stated limitations were subsequently closed in
> Sprint 7 — the dataset moved from a ConfigMap to S3 runtime retrieval (M-04,
> [ADR-027](../decisions/ADR-027-s3-dataset-runtime-retrieval.md)), the offline MLflow
> file store became the in-cluster MLflow platform
> ([ADR-026](../decisions/ADR-026-in-cluster-mlflow-platform.md)), and the EKS API /
> access / CNI / KMS findings (H-02, H-03, M-01, M-02) were remediated. Read this as
> history; for the current credible-claims boundary see the
> [Sprint 7 Proof-Impact Assessment](sprint-07-proof-impact.md). The historical
> figures below (29 resources, ConfigMap dataset, offline file store) were correct
> for Sprint 6 and are left unchanged.

- **Date:** 2026-08-15
- **Status:** Unreleased (Sprint 6). **No release tag is cut in this PR** — this
  assessment covers the work staged under `[Unreleased]` in the
  [CHANGELOG](../../CHANGELOG.md); a version tag is a separate release step.
- **Headline:** The containerized MLOps pipeline — proven green on a **local**
  cluster in Sprint 5 — has now been **provisioned onto a real, managed cloud
  Kubernetes platform (Amazon EKS) defined entirely as Terraform, run to completion
  there (Job `Complete`, exit 0, 52s), with all Sprint 5 security controls
  re-verified on the live pod, and then destroyed and verified clean.** The cloud
  platform is expressed as reviewable IaC (VPC/IAM/EKS), statically gated in CI with
  **no AWS credentials**, and operated through a documented ephemeral lifecycle
  (`provision → prove → destroy`). What is **not** claimed is production: it is a
  short-lived, single-operator, single-region validation environment — see
  [§5](#5-what-still-cannot-be-claimed).
- **Related:** [Cloud Operations](../cloud-operations.md),
  [Runtime Evidence](sprint-06-runtime-evidence.md),
  [`terraform/README.md`](../../terraform/README.md),
  [ADR-014](../decisions/ADR-014-terraform-architecture.md),
  [ADR-015](../decisions/ADR-015-aws-network-architecture.md),
  [ADR-016](../decisions/ADR-016-aws-iam-foundation.md),
  [ADR-017](../decisions/ADR-017-eks-platform.md),
  [ADR-018](../decisions/ADR-018-aws-eks-deployment-overlay.md),
  [ADR-019](../decisions/ADR-019-terraform-ci-validation.md),
  [ADR-020](../decisions/ADR-020-cloud-lifecycle-cost-control.md)

> This document answers one question with evidence:
> **What can this project credibly claim after Sprint 6 that it could not after
> Sprint 5?** "Credibly" is the operative word — every claim below points to a
> Terraform file, an ADR, an executed run, or a redacted evidence record a reviewer
> can check. [§5](#5-what-still-cannot-be-claimed) lists what the repository does
> **not** support, so the credible claims are not diluted by overreach.

---

## 1. The shift, in one line

| | Claim licensed by the repository |
|-|----------------------------------|
| **After Sprint 5** | "I expressed the containerized pipeline as a hardened Kubernetes batch `Job` and **ran it to completion (exit 0) on a local cluster**, with enforced security controls, measured resources, and static manifest validation in CI — bounded to a laptop, with a local-validation dataset and an in-pod MLflow file store." |
| **After Sprint 6** | "I additionally **defined a managed cloud platform (AWS VPC + least-privilege IAM + Amazon EKS) entirely as Terraform**, gated it in CI with **static, credential-free** validation, then **provisioned it in my own account and ran the *same* Job on real EKS to completion (exit 0, 52s)** — re-verifying every Sprint 5 security control on the **live** pod — and **destroyed the environment and verified it clean**, operated through a documented, cost-controlled ephemeral lifecycle. Bounded, honestly, to a short-lived single-operator validation environment: not production, not HA, not multi-region, no GitOps, no DR." |

Sprint 5's claim is about **platform-engineering judgment on a local cluster**.
Sprint 6's claim is about **cloud-platform engineering**: infrastructure as
reviewable code, a least-privilege cloud identity, a real managed control plane, a
credential-safe CI boundary, an executed cloud run that preserves the security
posture, and a disciplined provision-prove-destroy lifecycle with its costs and
limits stated.

---

## 2. New credible claims, with evidence

Each row was **not** defensible after Sprint 5 and **is** after Sprint 6.

### 2.1 "I defined the cloud platform as reviewable Infrastructure as Code."
- A structured Terraform root module — versions/provider/variables/outputs, a VPC
  with public/private subnets across AZs, routing, IGW + single NAT, two
  least-privilege IAM roles, and a managed EKS cluster + node group + core addons —
  **29 resources**, no unrelated infrastructure.
- **Evidence:** [`terraform/`](../../terraform/),
  [`terraform/README.md`](../../terraform/README.md),
  [ADR-014](../decisions/ADR-014-terraform-architecture.md)…[ADR-017](../decisions/ADR-017-eks-platform.md);
  `terraform validate` → **"Success! The configuration is valid."**,
  `terraform fmt -check -recursive` clean.

### 2.2 "I gave the platform a least-privilege cloud identity, with no static credentials."
- Two dedicated IAM roles (cluster + node), single-service trust each, only the
  AWS-managed policies EKS requires — **no `AdministratorAccess`, no
  project-authored wildcard, no IAM users/keys**. Terraform authenticates via the
  standard AWS credential chain; nothing credential-shaped is committed.
- **Evidence:** [`terraform/iam.tf`](../../terraform/iam.tf),
  [ADR-016](../decisions/ADR-016-aws-iam-foundation.md),
  [SECURITY.md](../../SECURITY.md); role ARNs are `sensitive` outputs (they embed the
  account ID).

### 2.3 "I gated the IaC in CI without giving CI any power over AWS."
- A `terraform-validate` job — `fmt -check` → `init -backend=false` → `validate` →
  TFLint → Trivy IaC scan (fails on CRITICAL/HIGH) — running on every push/PR with
  `permissions: contents: read`, **no AWS credentials, no OIDC identity**, and
  **never** `plan`/`apply`. Trivy suppressions are a justified, ADR-cross-referenced
  triage record.
- **Evidence:** [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml),
  [`terraform/.trivyignore`](../../terraform/.trivyignore),
  [ADR-019](../decisions/ADR-019-terraform-ci-validation.md), [docs/ci-cd.md](../ci-cd.md).

### 2.4 "I integrated the *existing* workload with EKS by reuse, not duplication or weakening."
- A thin `k8s/overlays/aws` over the unchanged base that layers **only** the three
  genuine cloud differences (ECR image, `imagePullPolicy: Always`, dataset mount) —
  the pod/container `securityContext`, `resources`, ServiceAccount, and token
  automount are **byte-identical** to the local overlay. No account ID in git (a
  `000000000000` placeholder).
- **Evidence:** [`k8s/overlays/aws/`](../../k8s/overlays/aws/),
  [ADR-018](../decisions/ADR-018-aws-eks-deployment-overlay.md); both overlays pass
  the same **45-check** `python k8s/validate.py` and render clean under
  `kubeconform -strict`.

### 2.5 "I provisioned real EKS in my own account and ran the pipeline to completion on it." (PR 7)
- `terraform apply` created the 29 resources; the EKS cluster came up **ACTIVE**
  (control plane **v1.35.6-eks**, 1 node **Ready** in a private subnet); the image
  was pulled from ECR via the node role (no pod credential); the Job reached
  **`Complete`** (1/1, **52s**), the pod **`Succeeded`** with **exit 0**, and all
  four stages ran (preprocess 768 → split 614/154 → train **0.7398** → evaluate
  **0.7078** — matching the Sprint 5 local metrics).
- **Evidence:** [Runtime Evidence §§1–4](sprint-06-runtime-evidence.md) (account ID
  and operator IP redacted).

### 2.6 "The Sprint 5 security controls hold on the *live* cloud pod, not just on paper." (PR 7)
- Read directly from the running pod: `runAsNonRoot: true`, uid/gid `10001`, seccomp
  `RuntimeDefault`, `allowPrivilegeEscalation: false`, `capabilities.drop: [ALL]`,
  measured requests/limits (`250m/256Mi`–`1/512Mi`, Burstable), and
  `automountServiceAccountToken: false` — **all 6 controls verified on EKS**,
  inherited verbatim from the committed base.
- **Evidence:** [Runtime Evidence §5](sprint-06-runtime-evidence.md#5-security-result--sprint-5-controls-verified-on-the-live-pod).

### 2.7 "I operated a cost-controlled ephemeral lifecycle and tore the environment down, verified clean." (PR 8)
- A documented `provision → prove → destroy` runbook with ranked cost drivers and a
  three-angle cleanup verification. The real run's teardown: `terraform destroy` →
  **"Destroy complete! Resources: 29 destroyed."**, local state **empty** ("No
  resources are represented"), ECR repository deleted, working tree clean. **No
  ongoing cost, no leftover diff.**
- **Evidence:** [Cloud Operations](../cloud-operations.md),
  [ADR-020](../decisions/ADR-020-cloud-lifecycle-cost-control.md),
  [Runtime Evidence § Teardown](sprint-06-runtime-evidence.md#teardown).

---

## 3. Proof / evidence

The dimensions Sprint 6 set out to establish, each mapped to its enforcing artifact
and its executed/verified evidence. Cloud evidence is from the **PR 7** run against
real EKS (`us-east-1`, 2026-08-15), account ID and operator IP redacted; the
environment was destroyed the same day.

| Dimension | Enforcing artifact | Verified evidence |
|---|---|---|
| **Infrastructure as Code** | Terraform root module ([ADR-014](../decisions/ADR-014-terraform-architecture.md)) | `terraform validate` **Success**, `fmt -check` clean; `apply` created **29** resources (18 net + 6 IAM + 5 EKS), `0 changed, 0 destroyed`. |
| **Cloud network** | VPC/subnets/NAT ([ADR-015](../decisions/ADR-015-aws-network-architecture.md)) | VPC `10.0.0.0/16`, 2 public + 2 private subnets across 2 AZs, 1 IGW, **1 shared NAT** + EIP; nodes in private subnets. |
| **Least-privilege IAM** | cluster + node roles ([ADR-016](../decisions/ADR-016-aws-iam-foundation.md)) | Two single-trust roles, AWS-managed policies only; node role's `AmazonEC2ContainerRegistryReadOnly` authorized the ECR pull — no pod credential. |
| **Managed EKS platform** | control plane + node group + addons ([ADR-017](../decisions/ADR-017-eks-platform.md)) | Cluster **ACTIVE**, **v1.35.6-eks**, 1 `t3.medium` node **Ready** (AL2023, containerd 2.2.5); coredns/kube-proxy/vpc-cni. |
| **Credential-safe CI gate** | `terraform-validate` ([ADR-019](../decisions/ADR-019-terraform-ci-validation.md)) | `fmt`/`init -backend=false`/`validate`/TFLint/Trivy on every push/PR; **no AWS creds**, never `plan`/`apply`. |
| **Workload↔EKS integration** | `k8s/overlays/aws` ([ADR-018](../decisions/ADR-018-aws-eks-deployment-overlay.md)) | Base reused unchanged; **45/45** static checks; security fields byte-identical to local overlay. |
| **Cloud runtime execution** | the executed Job (PR 7) | Job **`Complete`**, pod **exit 0**, **52s**, 4 stages, metrics parity (train 0.7398 / eval 0.7078). |
| **Live security posture** | live-pod `securityContext` (PR 7) | All **6** Sprint 5 controls verified on the running EKS pod; token automount off; Burstable QoS. |
| **Ephemeral lifecycle & cost** | runbook + teardown ([ADR-020](../decisions/ADR-020-cloud-lifecycle-cost-control.md)) | `destroy` → **29 destroyed**; state empty; ECR deleted; tree clean — verified three ways. |

---

## 4. Before / After (conservative)

Capability status, stated conservatively — "✅" only where the repository has the
artifact **and** the evidence; "⬜" where deferred; "❌" where not attempted/claimed.

| Capability | After Sprint 5 | After Sprint 6 |
|---|---|---|
| Green in-cluster pipeline run (local) | ✅ | ✅ (unchanged) |
| Cloud infrastructure as Terraform (VPC/IAM/EKS) | ❌ | ✅ |
| Least-privilege cloud IAM, no static credentials | ❌ | ✅ |
| Static, credential-free IaC validation in CI | ❌ | ✅ |
| EKS deployment overlay reusing base, no security weakening | ❌ | ✅ |
| **Green run on real managed EKS** (Job Complete, exit 0) | ❌ | ✅ (PR 7 — [runtime evidence](sprint-06-runtime-evidence.md)) |
| Sprint 5 security controls verified on a **live cloud** pod | ❌ | ✅ (PR 7) |
| Cost-controlled ephemeral lifecycle + verified teardown | ❌ | ✅ (PR 8 — [ADR-020](../decisions/ADR-020-cloud-lifecycle-cost-control.md)) |
| Real DagsHub MLflow connectivity exercised on the cloud run | ❌ | ❌ not claimed (offline file store used; config-validated) |
| Remote Terraform state backend (S3 + locking) | ❌ | ❌ not claimed (local state by design — [ADR-014](../decisions/ADR-014-terraform-architecture.md)) |
| Production cloud deployment / HA / multi-region / DR | ❌ | ❌ not claimed |
| GitOps / continuous delivery to the cluster | ❌ | ❌ not claimed (CI validates, does not deploy) |
| Production observability (metrics/tracing/alerting) | ❌ | ❌ not claimed |
| Model serving / inference endpoint | ❌ | ❌ not claimed |

---

## 5. What still **cannot** be claimed

Documented so none is accidentally implied:

- **✅ A green run on real managed EKS.** Achieved in PR 7 — but in a **short-lived,
  single-operator validation environment** (1 node, 1 NAT, 2 AZs), with a
  ~23 KiB dataset via ConfigMap and an **offline MLflow file store**. What is still
  **❌ not** claimed follows.
- **❌ Production deployment.** The environment is ephemeral (`provision → prove →
  destroy`), not an operated service. It was destroyed the same day.
- **❌ High availability.** Single node, single NAT gateway, single Job — no HA
  topology is provisioned or claimed.
- **❌ Multi-region / disaster recovery.** One region (`us-east-1`); no backup/restore,
  no state replication, no RTO/RPO — teardown is intentional deletion, not a recovery
  drill.
- **❌ GitOps / continuous delivery.** CI **validates** IaC and manifests statically;
  it holds no AWS credentials and deploys nothing.
- **❌ Production observability.** No metrics/tracing/alerting/log-aggregation stack;
  diagnosis is `kubectl` + structured logs.
- **❌ Real MLflow/DagsHub connectivity from the cluster.** The recorded run used a
  transient offline file store; the tracking path is configuration-validated, not
  connectivity-tested ([runtime evidence](sprint-06-runtime-evidence.md#limitations)).
- **❌ Remote Terraform state / team workflow.** Local state, single operator, by
  design; a remote backend is a documented migration path.
- **❌ Restricted Pod Security Standard certification**, and **⬜ read-only root
  filesystem** — deferred with evidence ([ADR-010](../decisions/ADR-010-kubernetes-security-hardening.md)).
- **❌ Production-certified capacity / cost-optimization at scale.** Sizing is the
  minimum to prove the claim, not a production or scale envelope.

---

## 6. Known limitations (explicit)

- **Not production** — a short-lived, single-operator validation environment.
- **Single environment** — one account, one region, one environment; local Terraform
  state, no dev/staging/prod split.
- **Limited scale** — 1 (default 2) `t3.medium` on-demand node, one batch `Job`, a
  ConfigMap dataset (not production storage).
- **No GitOps** (Argo CD / Flux) — operator `kubectl apply`; CI never deploys.
- **No HA proof** — single node/NAT/Job, no HA topology.
- **No production observability** — `kubectl` + structured logs only.
- **No multi-region** — single region, no cross-region anything.
- **No disaster-recovery proof** — no backup/restore, replication, or RTO/RPO.
- **Real DagsHub tracking not exercised on the cloud run** — offline file store used;
  connectivity is config-validated.
- **Read-only root filesystem deferred** — with recorded evidence (ADR-010).
- **Restricted PSS compliance not claimed** — controls applied and API-admitted, but
  no admission-policy engine ratifies the profile.

---

## 7. The honest one-paragraph statement

> Building on a hardened Kubernetes `Job` proven green on a local cluster, I defined
> the cloud platform it runs on — an AWS VPC, least-privilege IAM, and a managed
> Amazon EKS cluster — **entirely as reviewable Terraform**, and gated that IaC in CI
> with static, **credential-free** validation that never plans or applies. I then
> **provisioned the platform in my own account and ran the same containerized MLOps
> pipeline on real EKS to completion** (Job `Complete`, pod exit 0, 52 seconds, all
> four stages, metrics matching the local run), **re-verifying every Sprint 5
> security control on the live pod** — non-root uid/gid 10001, dropped capabilities,
> no privilege escalation, seccomp `RuntimeDefault`, measured resources, no API token
> — with the workload integrated by **reusing the base overlay unchanged, weakening
> nothing**. Then I **destroyed the environment and verified it clean** three ways
> (empty state, absent cluster, deleted repository), operated through a documented,
> cost-controlled `provision → prove → destroy` lifecycle. Deliberately, I do **not**
> claim a production deployment, high availability, multi-region, disaster recovery,
> GitOps, production observability, real MLflow connectivity on the cloud run, or a
> remote state backend — each is documented as deferred or out of scope. The claim is
> about **cloud-platform engineering evidenced on an ephemeral validation
> environment**, bounded and torn down — not about operating production cloud
> infrastructure.

That paragraph is fully supported by the repository and its redacted evidence. Its
restraint — naming the ephemeral, single-region, offline-tracking boundaries out loud
— is part of what makes the rest of it credible.
