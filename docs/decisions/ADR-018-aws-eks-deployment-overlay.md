# ADR-018: AWS EKS Deployment Overlay (Cloud Runtime Integration)

- **Status:** Accepted (design)
- **Date:** 2026-08-14
- **Deciders:** Asad Hanif
- **Related:** [`k8s/overlays/aws/`](../../k8s/overlays/aws/),
  [`k8s/base/`](../../k8s/base/),
  [`k8s/overlays/local/`](../../k8s/overlays/local/),
  [`k8s/validate.py`](../../k8s/validate.py),
  [ADR-009 (Kubernetes Workload Model — Job)](ADR-009-kubernetes-workload-model.md),
  [ADR-010 (Kubernetes Security Hardening)](ADR-010-kubernetes-security-hardening.md),
  [ADR-013 (Kubernetes Runtime Execution)](ADR-013-kubernetes-runtime-execution.md),
  [ADR-015 (AWS Network Architecture)](ADR-015-aws-network-architecture.md),
  [ADR-016 (AWS IAM Foundation)](ADR-016-aws-iam-foundation.md),
  [ADR-017 (Amazon EKS Platform)](ADR-017-eks-platform.md),
  [Sprint 6 plan](../../Sprint-06-Terraform-Cloud-Platform-Foundation.md)

> **Scope note.** This ADR ratifies the *Kubernetes-side integration* delivered in
> Sprint 6, PR 5: a new `k8s/overlays/aws` Kustomize overlay that targets the
> Terraform-provisioned EKS platform ([ADR-017](ADR-017-eks-platform.md)) by
> **reusing the existing Sprint 5 base unchanged**. It adds only cloud-specific
> configuration and **no new workload**. The *real* execution on EKS — `terraform
> apply` → `kubectl apply -k k8s/overlays/aws` → a green `Job` — is PR 7; this PR
> stops at a statically validated, deployable overlay.

## Context

Sprint 5 built the workload as a Kustomize **base** (`k8s/base`) — Namespace,
ServiceAccount, ConfigMaps, and a hardened batch `Job` — specialized for a local
cluster by a single **local overlay** (`k8s/overlays/local`) that pins the image
to `ml-pipeline:local` and layers two local-only runtime inputs: a dataset mount
and an offline MLflow file store ([ADR-013](ADR-013-kubernetes-runtime-execution.md)).

Sprint 6 provisioned a real cloud Kubernetes platform in Terraform (EKS control
plane, one managed node group in the private subnets, core addons —
[ADR-017](ADR-017-eks-platform.md)). The remaining gap is *connecting the
existing workload to that platform without duplicating it.* The base already
carries every Sprint 5 security control (`runAsNonRoot` + uid/gid `10001`,
`allowPrivilegeEscalation: false`, `capabilities: drop [ALL]`, seccomp
`RuntimeDefault`, token automount off, measured resource requests/limits); the
integration must **preserve all of them** and must not weaken the workload to make
EKS work.

The design constraint from the sprint: introduce an **AWS overlay** that contains
**only cloud-specific configuration**, **reuse the base**, **do not copy the
workload**, keep cloud differences **isolated**, and add none of the sprint
non-goals (ingress, service mesh, GitOps, observability, unrelated cloud services).

## Decision

Add `k8s/overlays/aws` as a **thin overlay over `../../base`** — the same base the
local overlay uses — and express **only** the three genuine local-vs-cloud
differences. The base is not modified; the local overlay is not modified. Every
security control, resource setting, the ServiceAccount, and the finite-run
lifecycle are **inherited from the base verbatim**, so the AWS overlay changes no
security field.

**1. Image source — a registry (Amazon ECR), not a side-loaded image.**
Worker nodes run in the **private subnets** ([ADR-015](ADR-015-aws-network-architecture.md))
and cannot see a locally built image, so on EKS the image is **pulled from a
registry**. Amazon ECR is the natural choice: the node role already carries
`AmazonEC2ContainerRegistryReadOnly` ([ADR-016](ADR-016-aws-iam-foundation.md)),
so the **kubelet authenticates the pull with the node's instance role** — no
pod-level credential and no IRSA. The overlay's `images` transformer repoints
`ml-pipeline` to an ECR reference pinned to an explicit, immutable tag
(`1.3.1`, the image `BUILD_VERSION`), never `:latest`. The account ID and region
are a committed **placeholder** (`000000000000` / `us-east-1`) that the operator
sets for their **own** account in PR 7 without editing the file
(`kustomize edit set image ml-pipeline=<account>.dkr.ecr.<region>.amazonaws.com/mlops-pipeline:1.3.1`).

**2. Image pull policy + runtime dataset — the `job-cloud.yaml` patch.**
A single strategic-merge patch adds:
- `imagePullPolicy: Always` — the base sets none (a side-loaded local image needs
  no pull); on EKS an explicit policy belongs here, and `Always` guarantees the
  node runs exactly the image just pushed to ECR for the validation run (removing
  the "stale cached image" class of failure the k8s README documents). The pull
  cost is negligible for a one-shot batch `Job`.
- a **read-only dataset mount at `/app/data/raw`** from an out-of-band ConfigMap
  (`mlops-pipeline-dataset`, `optional: true`) — identical mechanism to the local
  overlay. Dataset provisioning is a **per-environment** concern *by design* (the
  base "does not presume how the data arrives"), so each overlay declaring its own
  source is the established pattern, not base duplication. The bundled dataset is
  tiny (~23 KiB), so the same portable ConfigMap mechanism works on EKS.

**3. MLflow backend — deliberately *not* patched.**
The local overlay overrides `MLFLOW_TRACKING_URI` to an in-pod file store for an
offline run. The AWS overlay **omits** that override, so the base ConfigMap's real
DagsHub endpoint is used, with credentials from the out-of-band Secret
(`mlops-pipeline-secret`, `optional`). The cloud run therefore exercises the
**real** MLflow tracking path. No credential is committed (see
`base/secret.example.yaml`); the Secret is created out-of-band for PR 7.

**Validation tooling.** `k8s/validate.py` already accepts an overlay argument, so
it validates the AWS overlay unchanged; CI now runs the render + `kubeconform`
schema gate, the `k8s/validate.py` security/runtime-contract gate, and the opt-in
server-side dry-run **for both `local` and `aws`** overlays. The AWS overlay
passes the full contract (45/45), and its rendered pod `securityContext`,
container `securityContext`, `resources`, `serviceAccountName`, and
`automountServiceAccountToken` are **byte-identical** to the local overlay's —
mechanical proof that the cloud overlay weakens nothing.

## What is intentionally *not* in this overlay

- **No copy of the workload** — no Namespace/ServiceAccount/ConfigMap/Job is
  redefined; all come from `../../base`.
- **No security changes** — the overlay sets no `securityContext`, no
  `serviceAccountName`, no capability or seccomp field; it only inherits them.
- **No ingress, no Service, no service mesh, no GitOps, no observability, no
  Load Balancer Controller / CSI addons** — all sprint non-goals; a batch `Job`
  has no inbound surface and needs none of them.
- **No new AWS services** (S3 bucket, EFS, RDS, …) — dataset delivery reuses the
  existing out-of-band ConfigMap mechanism rather than introducing cloud storage
  in this PR.
- **No static credentials, no committed kubeconfig, no committed ECR account ID**
  — the ECR reference is a placeholder; real credentials/identity come from the
  operator's AWS profile at deploy time (PR 7).

## Alternatives Considered

1. **Move the dataset mount into the base (shared by both overlays).**
   - *Rejected* — the base deliberately "does not presume how the data arrives"
     ([ADR-013](ADR-013-kubernetes-runtime-execution.md)); dataset provisioning is
     a per-environment concern. Pushing it to the base would contradict that
     design and the ADR-013 scoping of the ConfigMap dataset as a *validation*
     mechanism. Each overlay declaring its own source keeps the base clean.
2. **Extract the dataset mount into a shared Kustomize *component*.**
   - *Rejected for now* — it removes a ~10-line repetition at the cost of a new
     `k8s/components/` structure not in the sprint's target layout and a refactor
     of the working local overlay (its patch bundles dataset + MLflow override),
     risking the "preserve local overlay functionality" acceptance criterion. The
     small, documented per-overlay stanza is the lower-risk, more legible choice;
     a component remains a clean future refactor if a third overlay appears.
3. **`imagePullPolicy: IfNotPresent` instead of `Always`.**
   - *Not chosen* — with an immutable version tag `IfNotPresent` is defensible and
     cheaper, but `Always` removes the stale-image failure mode entirely during a
     one-shot validation where the operator may re-push the same tag; the pull
     cost is trivial for a single batch run.
4. **A real cloud dataset path now (S3 / PVC / `dvc pull`).**
   - *Deferred* — that introduces a new AWS storage service (a Sprint 6 non-goal
     for this PR) and more moving parts than a validation run needs. The
     ConfigMap mechanism is honestly scoped as validation-only, exactly as in the
     local overlay; a production data path is a documented follow-up.
5. **Bake a real ECR account ID / region into the committed overlay.**
   - *Rejected* — the verification account is supplied per-operator (and, in this
     repo's working environment, must not be assumed); a committed placeholder set
     via `kustomize edit set image` at deploy time keeps no account identity in
     git and keeps the overlay account-agnostic.
6. **A separate `aws-*` base or a forked manifest set.**
   - *Rejected* — that is precisely the "copy the whole workload" the sprint
     forbids; it would double the maintenance surface and let the cloud and local
     definitions drift.

## Consequences

**Positive**

- The existing workload is now **deployable to EKS** by reusing the base — a
  single thin overlay, three isolated cloud differences, zero workload
  duplication.
- **Every Sprint 5 security control is provably preserved** — the AWS overlay's
  rendered security/resource/identity fields are byte-identical to the local
  overlay's, and both pass the same 45-check static contract in CI.
- **Infrastructure and workload stay cleanly separated** — Terraform provisions
  the platform (ADR-017), Kustomize configures the workload; the overlay adds no
  AWS resources.
- **No secret exposure** — ECR pull uses the node role; MLflow credentials stay in
  an out-of-band Secret; the committed overlay contains only a placeholder image
  reference.

**Trade-offs and follow-ups**

- The dataset stanza is **repeated** in the local and AWS overlays (~10 lines).
  This is the accepted cost of keeping dataset provisioning per-environment and
  the base clean; a shared component is a documented future refactor.
- The ConfigMap dataset remains a **validation mechanism, not production
  storage** on AWS too; a real cloud data path (S3/PVC/`dvc pull`) is a follow-up.
- **Runtime is not proven here.** PR 5 delivers a statically validated, admissible
  overlay; the green in-cluster run on EKS (Job Complete, exit 0) is PR 7.

## What This Decision Does *Not* Imply

- It does **not** modify the base or the local overlay; local behavior is
  unchanged.
- It does **not** add ingress, a Service, a mesh, GitOps, observability, or any
  new AWS service.
- It does **not** claim the workload has run on EKS — only that it is configured
  and statically validated to. The real execution and its evidence are PR 7.
- It does **not** introduce static credentials, a committed kubeconfig, or a
  committed account ID.
