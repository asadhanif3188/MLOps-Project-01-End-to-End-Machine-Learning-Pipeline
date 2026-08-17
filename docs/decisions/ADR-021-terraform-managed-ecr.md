# ADR-021: Terraform-Managed Container Registry (Amazon ECR)

- **Status:** Accepted (design) — closes Sprint 6 finding **H-01**
- **Date:** 2026-08-17
- **Deciders:** Asad Hanif
- **Related:** [`terraform/ecr.tf`](../../terraform/ecr.tf),
  [`terraform/variables.tf`](../../terraform/variables.tf),
  [`terraform/outputs.tf`](../../terraform/outputs.tf),
  [`terraform/tests/ecr.tftest.hcl`](../../terraform/tests/ecr.tftest.hcl),
  [`terraform/README.md`](../../terraform/README.md),
  [`docs/cloud-operations.md`](../cloud-operations.md),
  [`k8s/overlays/aws/kustomization.yaml`](../../k8s/overlays/aws/kustomization.yaml),
  [ADR-014 (Terraform Architecture & Foundation)](ADR-014-terraform-architecture.md),
  [ADR-016 (AWS IAM Foundation for EKS)](ADR-016-aws-iam-foundation.md),
  [ADR-017 (Amazon EKS Platform)](ADR-017-eks-platform.md),
  [ADR-018 (AWS EKS Deployment Overlay)](ADR-018-aws-eks-deployment-overlay.md),
  [ADR-019 (Terraform CI Validation)](ADR-019-terraform-ci-validation.md),
  [ADR-020 (Cloud Lifecycle & Cost Control)](ADR-020-cloud-lifecycle-cost-control.md)

> **Scope note.** This ADR ratifies making the **Amazon ECR** repository — the
> private registry that stores the workload image the EKS nodes pull — **fully
> Terraform-managed**, closing Sprint 6 finding **H-01**. It changes only the
> registry's ownership, lifecycle, and teardown; it does **not** alter the network,
> IAM, or EKS resources, introduce remote state or GitOps, or add Kubernetes changes
> beyond a comment in the AWS overlay.

## Context

Through Sprint 6 the AWS platform was fully Terraform-managed —
[network](ADR-015-aws-network-architecture.md),
[IAM](ADR-016-aws-iam-foundation.md), and [EKS](ADR-017-eks-platform.md) — **with one
exception**: the ECR repository that holds the workload image. The
[cloud-operations runbook](../cloud-operations.md) created it out-of-band with
`aws ecr create-repository` before the image push, and the
[teardown](ADR-020-cloud-lifecycle-cost-control.md) deleted it separately with
`aws ecr delete-repository --force`. A Sprint 6 review recorded this as finding
**H-01 — "ECR outside Terraform"** (HIGH, must-fix).

Leaving one live AWS resource outside Terraform state is a real gap, not a cosmetic
one:

- **No single source of truth.** The registry was not described by the IaC, so its
  configuration (tag mutability, scanning, encryption, retention) was implicit and
  unversioned — whatever the CLI defaulted to on the day it was run.
- **Split lifecycle.** `terraform apply`/`destroy` did not create or remove it. The
  runbook and [ADR-020](ADR-020-cloud-lifecycle-cost-control.md) had to carry a manual
  step, and a "clean" teardown depended on the operator remembering to delete it
  separately — exactly the kind of out-of-band drift the project otherwise avoids.
- **No contract enforcement.** Because the repository was not in the config, the
  Terraform CI gate ([ADR-019](ADR-019-terraform-ci-validation.md)) could not assert
  anything about it (private, immutable tags, scanning on).

The constraint: bring ECR under Terraform **using the project's existing
conventions**, preserve the security posture, add no remote state and no GitOps, and
do not rewrite the existing modules.

## Decision

Add an `aws_ecr_repository` (plus an `aws_ecr_lifecycle_policy`) to the root module in
[`terraform/ecr.tf`](../../terraform/ecr.tf), managed like every other resource.

1. **Repository name = `project_name` (`mlops-pipeline`), overridable.** ECR is a
   per-project artifact store, so — unlike the environment-scoped EKS/IAM/network
   resources that use `local.name_prefix` — the repository is **not** environment
   suffixed. Defaulting to `project_name` keeps it in lock-step with the image
   reference already committed in [`k8s/overlays/aws`](../../k8s/overlays/aws), so no
   committed manifest changes. A `ecr_repository_name` variable overrides it if
   needed.

2. **Immutable tags** (`image_tag_mutability = "IMMUTABLE"`). A version tag such as
   `1.3.1` can never be repointed at a different image, so a deployed digest is
   reproducible — matching the "explicit, immutable version, never `:latest`"
   convention the AWS overlay already relies on ([ADR-018](ADR-018-aws-eks-deployment-overlay.md)).

3. **Scan on push stays ON** (`scan_on_push = true`). Image vulnerability scanning is
   a security feature; it is enabled, not disabled to simplify.

4. **Private by construction.** No repository policy granting public or cross-account
   access is authored, so the registry is reachable only by the account's own
   principals — the node role carries `AmazonEC2ContainerRegistryReadOnly` from
   [ADR-016](ADR-016-aws-iam-foundation.md), which is sufficient for the pull. **ECR
   is never made public.**

5. **Encrypted at rest with the AWS-managed key** (`AES256`, ECR's default). A
   customer-managed **KMS CMK** is the documented hardening follow-up, tracked with
   the EKS-secrets KMS work (finding M-02); it is deliberately out of this PR's H-01
   scope so the change stays minimal.

6. **Lifecycle policy caps storage.** A single `expire` rule keeps the most recent
   `ecr_max_image_count` images (default **10**) and reaps older ones, so the registry
   cannot grow unbounded across repeated validation pushes.

7. **`force_delete = true`.** `terraform destroy` removes the repository and any
   images it still holds in the same pass. This is what replaces the manual
   `aws ecr delete-repository --force` teardown step for this **short-lived,
   single-operator** validation environment ([ADR-020](ADR-020-cloud-lifecycle-cost-control.md)).

8. **Sensitive URL/ARN outputs.** `ecr_repository_url` and `ecr_repository_arn` embed
   the AWS account ID — which the project treats as sensitive (see the
   `aws_account_id` output) — so both are marked `sensitive`; `ecr_repository_name` is
   not. Operators read the URL with `terraform output -raw ecr_repository_url` when
   pushing the image or pointing the Kustomize overlay at the registry, so no account
   ID reaches git or logs.

9. **Contract pinned by an offline test.** A native `terraform test` suite
   ([`tests/ecr.tftest.hcl`](../../terraform/tests/ecr.tftest.hcl)) uses
   `mock_provider "aws"` (`command = plan`, no AWS, no credentials) to assert the name,
   immutability, scan-on-push, encryption, and retention policy, and runs as a step in
   the existing `terraform-validate` CI job — the same no-cloud spirit as
   [ADR-019](ADR-019-terraform-ci-validation.md). (`required_version` is raised to
   `>= 1.7.0` for `mock_provider`.)

The manual `aws ecr create-repository` / `aws ecr delete-repository` steps are removed
from the runbook and from [ADR-020](ADR-020-cloud-lifecycle-cost-control.md).

## Alternatives Considered

1. **Leave ECR out-of-band, only document it better.**
   - *Rejected* — this is precisely finding H-01. Documentation does not give the
     registry a versioned config, a managed lifecycle, or a CI contract, and it keeps
     the manual teardown step that a "clean" environment depends on.
2. **Environment-scope the name via `local.name_prefix` (`mlops-pipeline-dev`).**
   - *Rejected* — it would break the image reference committed in the AWS overlay and
     the runbook for no benefit: the registry is a per-project artifact store, not a
     per-environment resource. A dedicated `ecr_repository_name` variable covers the
     rare case where a distinct name is wanted.
3. **Add a customer-managed KMS CMK for the registry now.**
   - *Deferred* — AES256 (the AWS-managed default) already encrypts at rest, and a CMK
     is the same hardening decision as KMS-encrypting EKS secrets (M-02). Bundling it
     here would widen H-01's scope; it is tracked as a follow-up. *(The EKS-secrets
     side of that decision is now settled by
     [ADR-025](ADR-025-eks-secrets-kms-encryption.md) — a dedicated CMK closes M-02.
     ECR keeps the AES256 default: a **separate**, still-open follow-up if a
     persistent registry ever warrants its own CMK — each key stays single-purpose.)*
4. **`force_delete = false` (protect against accidental image loss).**
   - *Rejected for this scope* — the environment is ephemeral and destroyed after
     evidence capture ([ADR-020](ADR-020-cloud-lifecycle-cost-control.md)); a repo that
     `destroy` cannot remove would reintroduce a manual cleanup step, the exact problem
     H-01 fixes. A persistent/production registry would set this `false`.
5. **Mutable tags for convenience.**
   - *Rejected* — mutable tags break reproducibility and the overlay's static
     image-pinning contract. Immutability is the safer default and costs nothing here.

## Consequences

**Positive**

- **H-01 closed.** The entire AWS platform, ECR included, is now described by the IaC
  and has one lifecycle: `terraform apply` creates it, `terraform destroy` removes it.
- **Cleaner, safer teardown.** No separate `aws ecr delete-repository --force` step;
  `destroy` (with `force_delete`) removes the repository and its images, so an
  environment that reports clean *is* clean.
- **Stronger posture, enforced.** Private, immutable-tagged, scanned, encrypted, and
  storage-capped by construction — and the `terraform test` suite fails CI if any of
  those regress.
- **No secrets in git.** The URL/ARN (which carry the account ID) are `sensitive`
  outputs read on demand; the committed overlay keeps its `000000000000` placeholder.

**Trade-offs and follow-ups**

- **AWS-managed encryption, not a CMK.** Acceptable for a validation registry that
  stores no production secrets; a customer-managed KMS CMK is the documented follow-up
  (with M-02). Trivy flags the missing CMK only at LOW severity, below the
  CRITICAL/HIGH CI gate, so no suppression is needed.
- **`force_delete` favours teardown over data protection** — correct for an ephemeral
  environment, wrong for a persistent one; flip it (and revisit the lifecycle count)
  if this registry ever becomes long-lived.
- **`required_version` raised to `>= 1.7.0`** for `mock_provider`. CI already pins
  Terraform 1.9.8, so this only removes 1.6.x from the supported floor.

## What This Decision Does *Not* Imply

- It does **not** introduce remote state, GitOps, or any CI/CD delivery — image build
  and push stay an operator step (the runbook), and CI still runs **no** `plan`/`apply`
  ([ADR-019](ADR-019-terraform-ci-validation.md)).
- It does **not** make the registry public, add cross-account access, or grant any new
  IAM permission — the existing node-role read-only pull is unchanged
  ([ADR-016](ADR-016-aws-iam-foundation.md)).
- It does **not** change the network, IAM, or EKS resources, nor any Kubernetes
  security field — only a comment in the AWS overlay is updated to reference the
  Terraform output.
