# ADR-019: Terraform CI Validation (fmt / init / validate / lint / IaC scan, no AWS)

- **Status:** Accepted
- **Date:** 2026-08-14
- **Deciders:** Asad Hanif
- **Related:** [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml),
  [`terraform/`](../../terraform/),
  [`terraform/.tflint.hcl`](../../terraform/.tflint.hcl),
  [`terraform/.trivyignore`](../../terraform/.trivyignore),
  [ADR-014 (Terraform Architecture & Foundation)](ADR-014-terraform-architecture.md),
  [ADR-015 (AWS Network Architecture)](ADR-015-aws-network-architecture.md),
  [ADR-016 (AWS IAM Foundation)](ADR-016-aws-iam-foundation.md),
  [ADR-017 (Amazon EKS Platform)](ADR-017-eks-platform.md),
  [ADR-012 (Kubernetes Manifest Validation in CI)](ADR-012-kubernetes-manifest-validation.md),
  [docs/ci-cd.md](../ci-cd.md)

> **Scope note.** This ADR ratifies the *CI quality gates* for the Terraform IaC
> added across Sprint 6 (PRs 1–4): a new `terraform-validate` job that
> **statically** validates `terraform/` on every push/PR. It adds **no** cloud
> automation — no `terraform plan` against a real account, no `apply`, no AWS
> credentials in CI. The boundary is the point: prove the IaC is well-formed,
> valid, linted, and free of insecure configuration **without** giving CI any
> power to touch AWS.

## Context

Sprint 6 grew a real Terraform codebase (VPC/subnets/NAT, two IAM roles, an EKS
cluster + node group + addons). CI already gates the Python package
([`quality`](../ci-cd.md)), the container image (`docker`), and the Kubernetes
manifests (`k8s-validate`, [ADR-012](ADR-012-kubernetes-manifest-validation.md)),
but the Terraform had **no automated gate** — formatting, a typo'd reference, a
broken type, or an insecure default could reach `main` unreviewed by a machine.

The obvious "complete" IaC gate — `terraform plan` — is exactly what CI must
**not** run here. `plan` reads provider data sources (`aws_caller_identity`,
`aws_region`, `aws_availability_zones`; see [`main.tf`](../../terraform/main.tf)
and [`network.tf`](../../terraform/network.tf)), so it requires **live AWS
credentials**. Wiring long-lived AWS keys into GitHub Actions to make `plan` run
would (a) put standing cloud credentials in a system that executes untrusted
pull-request code, and (b) contradict this repo's standing rule that cloud
provisioning is a deliberate, operator-driven, own-account activity
([ADR-014](ADR-014-terraform-architecture.md), and the network/IAM/EKS ADRs). The
credentials in the working environment during development belong to a **client**,
which makes an accidental or automated plan/apply especially unacceptable.

The constraint from the sprint: add Terraform quality gates to CI **without**
creating unsafe automatic AWS provisioning — no long-lived AWS credentials in
Actions, no `apply` from PRs, no secret exposure in logs, and no weakening of the
existing (read-only, no-publish) CI security posture.

## Decision

Add a single `terraform-validate` job to `ci.yml` that runs on every push/PR **in
parallel** with the other static validators. Every step is **static source
analysis that never contacts AWS**; tool versions are **pinned** for reproducible
runs. The job declares `permissions: contents: read` and holds **no AWS
identity**.

**The gate chain (all offline):**

1. **`terraform fmt -check -recursive`** — canonical formatting is enforced
   (checked, never rewritten); drift fails with an actionable `::error::` message.
2. **`terraform init -backend=false`** — installs the pinned provider from the
   committed `.terraform.lock.hcl` **without** configuring a backend or reading
   state, and **without** AWS credentials. It is the prerequisite for `validate`.
3. **`terraform validate`** — syntax, types, references, and provider-schema
   conformance. Makes **no** AWS API calls. The primary IaC correctness gate.
4. **TFLint** (pinned, language best-practices preset **+ AWS ruleset**; config in
   [`terraform/.tflint.hcl`](../../terraform/.tflint.hcl)) — catches deprecated
   syntax, unused declarations, naming issues, and AWS-specific mistakes (invalid
   instance types, malformed values). Static; contacts no cloud.
5. **Trivy IaC scan** (`trivy config`, pinned) — scans `terraform/` for insecure
   configuration (public exposure, missing encryption, over-broad IAM). **Fails
   the job on CRITICAL/HIGH.** Reads only the source — no AWS access.

**`terraform plan` is deliberately excluded** because it needs AWS credentials;
the boundary is documented (here, in [docs/ci-cd.md](../ci-cd.md), and in
[terraform/README.md § Planning](../../terraform/README.md)) rather than papered
over with credentials. `apply` is never run by CI under any trigger.

**Trivy suppressions are a justified triage record, not a blanket mute.** This
config intentionally carries a few exposures that suit a **short-lived,
single-operator validation cluster** — the EKS public API endpoint defaulting to
an open CIDR for first-run `kubectl` access, no customer-managed KMS envelope
encryption of secrets, and the auto-assigned public IPs on the **public** subnets
(which host NAT / future public LBs; the EKS nodes themselves are in the private
subnets with no public IPs — [ADR-015](ADR-015-aws-network-architecture.md)). Each
is ratified in [ADR-017](ADR-017-eks-platform.md)/[ADR-015](ADR-015-aws-network-architecture.md)
and is suppressed **with written justification and an ADR cross-reference** in
[`terraform/.trivyignore`](../../terraform/.trivyignore). Any **new** CRITICAL/HIGH
the scanner surfaces is therefore a real, blocking regression — not expected
noise. For a persistent/production cluster the correct action is to **remediate**
(narrow the API CIDR, disable the public endpoint, add a CMK), not to keep
suppressing.

**Least privilege is preserved and restated.** The job keeps the workflow's
`contents: read` permission, adds no `packages`/`id-token`/write scope, and
introduces no secrets. TFLint's plugin download uses the repo-scoped, read-only
`GITHUB_TOKEN` (to avoid anonymous API rate limits) — which carries **no AWS
identity**. No AWS output value is printed; the sensitive `aws_account_id` output
is unaffected because no `plan`/`apply`/`output` runs in CI.

## What is intentionally *not* in this job

- **No `terraform plan`** — needs AWS credentials; excluded and documented.
- **No `terraform apply`, ever** — CI provisions nothing, under any trigger.
- **No AWS credentials, no OIDC role, no `id-token` permission** — the job has no
  cloud identity of any kind.
- **No remote backend / state access** — `init -backend=false`; state is never
  read or written in CI.
- **No new write scope** — `permissions: contents: read`, matching the rest of the
  pipeline; nothing can be published or mutated.

## Alternatives Considered

1. **Run `terraform plan` in CI with long-lived AWS access keys.**
   - *Rejected* — standing cloud credentials in a system that runs untrusted PR
     code is precisely the insecure automatic provisioning the sprint forbids. It
     also risks acting on the wrong (client) account. The boundary is documented
     instead.
2. **Run `plan` via short-lived OIDC (GitHub → AWS `AssumeRoleWithWebIdentity`).**
   - *Deferred* — OIDC federation is the *correct* future answer (no static keys),
     but it still grants CI a real, plan-capable AWS identity and requires an IAM
     OIDC provider + role + trust policy that don't exist yet and belong to the
     operator's own account. It is a documented follow-up, out of scope for a
     "no unsafe provisioning" PR; the offline gate delivers most of the value now.
3. **`terraform plan` with a mocked/offline provider (no real AWS).**
   - *Rejected* — Terraform has no first-class offline-plan mode; the data sources
     resolve against a real account. Faking it would be brittle and misleading. A
     provider-only `init` + `validate` gives the honest offline guarantee.
4. **TFLint only, or Trivy only.**
   - *Rejected* — they are complementary. TFLint catches language/AWS-API
     correctness (bad instance types, deprecated args); Trivy catches security
     misconfiguration (public exposure, missing encryption). Both are cheap,
     credential-free, and justified for an AWS-heavy config.
5. **Fail CI on every Trivy CRITICAL/HIGH with no suppressions.**
   - *Rejected* — it would red the build on the *intentional*, ADR-ratified
     validation-cluster exposures, forcing either a weakening of the documented
     design or an undocumented mute. A small, **justified** `.trivyignore` keeps
     the gate meaningful (new findings still block) and honest (each exception is
     explained and points to its ADR).
6. **Report-only scanners (never fail the build).**
   - *Rejected* — a non-blocking scanner is security theatre. The gate blocks on
     unignored CRITICAL/HIGH so a genuine regression cannot merge.
7. **A separate workflow file for Terraform.**
   - *Rejected for now* — one `ci.yml` with parallel jobs keeps triggers,
     concurrency, and the least-privilege permission block in one place, matching
     how `k8s-validate` was added. A split is a clean future refactor if the infra
     pipeline grows its own lifecycle.

## Consequences

**Positive**

- Terraform now has the same class of **binding, server-side** quality gate the
  Python, container, and Kubernetes layers already have — format, validity, lint,
  and security misconfiguration are all enforced before merge.
- **No cloud attack surface added.** CI gains zero AWS power: no credentials, no
  identity, no `plan`/`apply`, read-only permissions. Safe on fork PRs.
- **Reproducible.** Pinned Terraform/TFLint/Trivy versions mean a green run is
  stable across runner-image changes; bumps are deliberate.
- **Security posture is legible.** The `.trivyignore` doubles as a reviewed,
  cross-referenced record of every accepted exposure and why.

**Trade-offs and follow-ups**

- **`plan`-time errors are not caught in CI** (e.g. an invalid AZ count for a
  region, a value only a real provider rejects). That check remains an operator
  step against their own account; short-lived **OIDC** federation is the
  documented path to a credential-safe `plan` in CI later.
- **Trivy was validated on the Linux CI runner, not on the Windows dev host**
  (local endpoint security blocked the scanner binary); `fmt`/`init`/`validate`
  and TFLint were run locally. The `.trivyignore` set was compiled by static
  review and is finalized/enforced by the runner — new findings there are real.
- The suppression list must be **re-triaged when the network/EKS/IAM config
  changes**; it is scoped to the current validation-cluster design.

## What This Decision Does *Not* Imply

- It does **not** give CI the ability to provision, plan, or mutate AWS — the job
  has no cloud identity.
- It does **not** weaken any existing gate; it adds one and preserves
  `permissions: contents: read` everywhere.
- It does **not** accept the suppressed findings as safe for production — they are
  scoped to a short-lived validation cluster and are remediation items for any
  persistent environment.
- It does **not** close the OIDC/`plan`-in-CI question — that remains a documented
  future enhancement.
