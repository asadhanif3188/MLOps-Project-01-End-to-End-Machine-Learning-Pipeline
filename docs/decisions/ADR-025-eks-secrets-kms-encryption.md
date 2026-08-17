# ADR-025: EKS Secret Envelope Encryption with a Customer-Managed KMS Key — closes M-02

- **Status:** Accepted (design)
- **Date:** 2026-08-17
- **Deciders:** Asad Hanif
- **Related:** [`terraform/kms.tf`](../../terraform/kms.tf),
  [`terraform/eks.tf`](../../terraform/eks.tf),
  [`terraform/variables.tf`](../../terraform/variables.tf),
  [`terraform/outputs.tf`](../../terraform/outputs.tf),
  [`terraform/tests/eks_secrets_encryption.tftest.hcl`](../../terraform/tests/eks_secrets_encryption.tftest.hcl),
  [`terraform/.trivyignore`](../../terraform/.trivyignore),
  [`terraform/README.md`](../../terraform/README.md),
  [ADR-017 (Amazon EKS Platform)](ADR-017-eks-platform.md) — **this ADR resolves
  ADR-017's deferred "no customer-managed KMS envelope encryption of secrets"
  follow-up**,
  [ADR-021 (Terraform-Managed ECR)](ADR-021-terraform-managed-ecr.md) — **shares
  the "AWS-managed default now, CMK later" decision that this ADR settles for EKS
  secrets**,
  [ADR-019 (Terraform CI Validation)](ADR-019-terraform-ci-validation.md),
  [ADR-020 (Cloud Lifecycle & Cost Control)](ADR-020-cloud-lifecycle-cost-control.md)

> **Scope note.** This ADR ratifies adding a **dedicated customer-managed KMS key**
> and wiring it into the EKS cluster's **`encryption_config`** so Kubernetes
> **Secret** objects are envelope-encrypted, closing Sprint 6 finding **M-02**. It
> changes *how secrets are encrypted at rest* and adds two resources (a KMS key and
> its alias); it does **not** re-architect the cluster, network, node group, IAM
> roles, or the CNI identity model (ADR-015/-016/-017/-022/-023/-024 otherwise
> stand), and it introduces no GitOps, no remote state, no hardcoded ARNs/account
> IDs, and no committed credentials.

## Context

By default, Amazon EKS encrypts all cluster data in etcd **at rest** using an
**AWS-owned** key that the customer neither sees nor controls. Kubernetes
**Secret** objects are stored in etcd, so they inherit that baseline encryption —
but with an AWS-owned key there is **no customer-controlled envelope-encryption
layer**: nothing the operator can audit in CloudTrail, scope with a key policy, or
revoke. The Sprint 6 review flagged the absence of customer-managed envelope
encryption of Kubernetes Secrets as finding **M-02**, and [ADR-017](ADR-017-eks-platform.md)
recorded "enable customer-managed KMS envelope encryption" as an explicit
**deferred** follow-up (it adds a billable KMS key for marginal benefit on a
cluster that, at the time, stored no real secrets).

EKS supports a second, customer-controlled encryption layer via the **KMS secrets
encryption provider** (a Kubernetes [encryption-at-rest provider](https://kubernetes.io/docs/tasks/administer-cluster/encrypt-data/)).
When enabled, the API server generates a **data key** for each Secret, encrypts the
Secret's data with it (envelope encryption), and calls **AWS KMS** to wrap that
data key with a **customer-managed key (CMK)**. The result: Secret plaintext is
never written to etcd, and access to the wrapping key is governed by a key policy
we own and logged in CloudTrail.

The requirement for this PR: a Terraform-managed CMK dedicated to EKS secrets, an
appropriate key policy, key rotation, the EKS encryption configuration that
actually **uses** the key, the minimum IAM permissions, resource tags, removal of
the now-obsolete Trivy suppression, static validation, and this ADR — **without**
wildcard KMS permissions (unless unavoidable and justified), without exposing key
material, and without weakening encryption merely to satisfy CI.

A crucial distinction this design keeps front-of-mind: **"a KMS key exists" is not
"EKS Secrets are configured to use it."** The finding is only closed when the
cluster's `encryption_config` points at the CMK — a key sitting unused next to the
cluster changes nothing.

## Decision

Create a **dedicated, Terraform-managed customer-managed KMS key** for EKS secret
envelope encryption and enable it on the cluster:

1. **Dedicated CMK** (`aws_kms_key.eks_secrets`, [`terraform/kms.tf`](../../terraform/kms.tf)) —
   symmetric (the only key type EKS envelope encryption supports), with a
   human-readable **alias** (`alias/<project>-<environment>-eks-secrets`).
2. **Automatic key rotation ENABLED** (`enable_key_rotation = true`) — AWS rotates
   the backing key material annually with no re-encryption and no config change.
   It is a no-cost, no-downtime hardening and is unconditional (not a variable):
   there is no reason to run this key without it.
3. **Least-privilege key policy**, three statements:
   - **`EnableIAMRootAdministration`** — the AWS-canonical "prevent lock-out"
     statement present in every default KMS key policy: it delegates access control
     for this key to IAM **within this account** (the account-root principal). This
     is the **one justified wildcard** (`kms:*`): without it the key can become
     permanently unmanageable, which AWS explicitly warns against. It grants **no
     other account or service** any use of the key.
   - **`AllowEKSClusterRoleToUseTheKey`** — grants the **EKS cluster (control-plane)
     role** exactly the cryptographic operations the secrets provider needs:
     `kms:Encrypt`, `kms:Decrypt`, `kms:DescribeKey`, `kms:ListGrants`. An explicit
     action list — **no `kms:*`**, and deliberately no wildcard action such as
     `kms:ReEncrypt*` (EKS does not need it).
   - **`AllowEKSClusterRoleToCreateGrants`** — `kms:CreateGrant`, constrained by the
     `kms:GrantIsForAWSResource` condition so the role can only create grants for
     AWS services. EKS uses this grant to let the managed control plane use the CMK.
4. **EKS encryption configuration** (`aws_eks_cluster.this.encryption_config`,
   [`terraform/eks.tf`](../../terraform/eks.tf)) — `resources = ["secrets"]`
   (`secrets` is the only resource type EKS supports), `provider.key_arn` = the CMK.
   **This is the half that makes M-02 real**: it associates the key with the cluster
   so Secrets are actually envelope-encrypted, rather than a key merely existing.
5. **IAM permissions via the key policy, not a separate identity policy.** For KMS,
   a key policy that grants a same-account principal is **authoritative and
   sufficient on its own** — an additional identity-based IAM policy is not
   required. Granting through the key policy keeps the permission **scoped to
   exactly this one key**; a customer-managed IAM policy would be a second place to
   maintain and a wider surface. So **no new IAM policy is attached** to the cluster
   role; the required permissions live in statements 2–3 above.
6. **Tags** — the CMK carries the common `default_tags` set plus a `Name` tag, like
   every other resource.
7. **One tunable, `var.kms_key_deletion_window_days`** (default **7**, the
   minimum) — how long the key lingers in `PendingDeletion` after `terraform
   destroy`. Short by design for this ephemeral validation cluster (ADR-020); a
   persistent key would use a longer window as an accidental-deletion safety net.
   There is deliberately **no "enable encryption" toggle** — a switch to turn the
   security control off is exactly what M-02 is about.

The dependency chain is linear and acyclic: **cluster role → KMS key** (the key
policy names the role) **→ EKS cluster** (its `encryption_config` names the key).

## Mechanism selection

| | **Customer-managed CMK (chosen)** | AWS-owned key (status quo) | AWS-managed key (`aws/eks`) |
|---|---|---|---|
| Envelope layer for Secrets | **Yes** | No (etcd baseline only) | Partial |
| Key policy we control | **Yes** | No | No |
| CloudTrail visibility of key use | **Yes** | No | Limited |
| Revocable / disablable by us | **Yes** | No | No |
| Rotation we can assert | **Yes (annual)** | AWS-internal | AWS-internal |
| Cost | ~1 CMK/month + per-request | Free | Free |
| Closes M-02 | **Yes** | No | Not the customer-managed control the finding asks for |

A **customer-managed CMK** is the only option that gives an auditable,
policy-scoped, revocable envelope layer — which is precisely what the finding
requires. The AWS-owned default is the status quo M-02 flags; the AWS-managed
`aws/eks` key still is not customer-controlled. The marginal monthly cost is
accepted (and bounded by the ephemeral-cluster lifecycle, ADR-020).

## Alternatives considered

1. **Keep the AWS-owned default and re-justify the Trivy suppression.**
   - *Rejected* — that is the finding, not a fix. The suppression was always framed
     as "documented hardening deferred", and this PR is that hardening. Re-muting it
     would leave M-02 open.
2. **Use the AWS-managed `aws/eks` key instead of a CMK.**
   - *Rejected* — it removes the very properties the finding asks for: a key policy
     we author, CloudTrail attribution, and the ability to disable/revoke. It is not
     a customer-managed control.
3. **Grant KMS permissions with a customer-managed IAM policy on the cluster role
   (instead of the key policy).**
   - *Rejected* — redundant for same-account KMS (the key policy is authoritative),
     and it widens the grant's surface and creates drift risk between two policy
     documents. The key-policy grant is tighter and colocated with the key.
4. **A single shared KMS key for ECR + EKS secrets (and anything else).**
   - *Rejected* — one key per purpose keeps blast radius and key policies small and
     legible, consistent with the "one role per purpose" stance (ADR-016). ECR
     currently uses the AES256 default (ADR-021); if it later adopts a CMK it gets
     its own, not this one.
5. **Encrypt more `resources` than `secrets`.**
   - *Not applicable* — EKS's `encryption_config` supports only `"secrets"`; there
     is no broader option to choose.
6. **Make encryption an opt-in variable (`enable_secrets_encryption`).**
   - *Rejected* — a toggle that can disable a security control is the anti-pattern
     M-02 is about. Encryption is unconditional; only the key's deletion window is
     configurable.

## Consequences

**Positive**

- **M-02 closed in configuration**: Kubernetes Secrets are envelope-encrypted with
  a customer-managed, rotated, least-privilege CMK that the cluster actually uses.
- The obsolete `AVD-AWS-0039` Trivy suppression is **removed, not re-justified** —
  with real CMK encryption configured the scanner passes on its own, so any future
  regression (someone dropping `encryption_config`) becomes a **blocking CI
  failure**, not silent drift.
- An **offline contract test** (`tests/eks_secrets_encryption.tftest.hcl`, mocked
  provider) pins the whole contract: encryption covers `secrets`, a KMS provider
  key is wired, rotation is on, the deletion window is valid, the alias is correct,
  and the key policy grants **no bare `"*"` principal** use of the key.
- Evidence outputs make live verification one command: `eks_secrets_kms_key_arn`
  and `eks_secrets_kms_key_alias` describe the CMK, and
  **`eks_secrets_encryption_key_arn` reads the key ARN back off the cluster's
  applied `encryption_config`** — proving the cluster uses the key, not merely that
  a key exists.

**Trade-offs and follow-ups**

- **Cost.** A CMK bills a small monthly charge plus per-request KMS costs on Secret
  reads/writes. Bounded by the ephemeral-cluster lifecycle (ADR-020) and negligible
  at this scale.
- **One-way enablement.** Once a cluster has secrets encryption enabled it **cannot
  be disabled**, and the **key cannot be swapped**, without replacing the cluster.
  This is acceptable and intended here — the ephemeral cluster is created with
  encryption on from the first `apply` — but it is why enabling it is a deliberate,
  ADR-ratified step rather than a casual toggle. On an **already-running** cluster,
  adding `encryption_config` is an in-place, one-way update; existing Secrets are
  re-encrypted on their next write (a `kubectl replace`/rotation forces it).
- **`aws-cn`/GovCloud.** KMS and EKS secrets encryption are available in all
  standard partitions; the partition is resolved (not hardcoded) for the root ARN,
  so the key policy is correct in any partition.
- **Not a substitute for application-layer secret hygiene.** Envelope encryption
  protects Secrets **at rest in etcd**; it does not change how the workload receives
  its MLflow credential (still delivered out-of-band at run time) or replace
  in-cluster RBAC. Those remain governed by ADR-023 (access) and the workload's own
  handling.

## What this decision does *not* imply

- It does **not** claim production key management (no external key store, no
  multi-Region keys, no HSM/`XKS`) — a single regional CMK for a validation cluster.
- It does **not** add or change any IAM **role**; the cluster role's KMS access is
  granted through the key policy, and no new identity-based policy is created.
- It does **not** encrypt anything beyond Kubernetes Secrets (EKS supports only
  `secrets`), nor does it alter the ECR encryption decision (ADR-021).
- It does **not** provision or expose any key material, plaintext secret, or
  credential; only ARNs/IDs/aliases are output, and the ARNs are marked sensitive.
