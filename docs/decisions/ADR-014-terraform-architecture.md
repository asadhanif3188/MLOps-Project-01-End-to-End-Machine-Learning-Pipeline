# ADR-014: Terraform Architecture & Foundation

- **Status:** Accepted (design)
- **Date:** 2026-08-14
- **Deciders:** Asad Hanif
- **Related:** [`terraform/`](../../terraform/),
  [`terraform/README.md`](../../terraform/README.md),
  [Sprint 6 plan](../../Sprint-06-Terraform-Cloud-Platform-Foundation.md),
  [ADR-009 (Kubernetes Workload Model)](ADR-009-kubernetes-workload-model.md),
  [`k8s/`](../../k8s/)

> **Scope note.** This ADR ratifies the *Terraform foundation* delivered in
> Sprint 6, PR 1: the `terraform/` project structure, version/provider
> constraints, naming and tagging strategy, and the local-state posture. It
> declares **no AWS resources**. The AWS/EKS platform choice (why AWS, why EKS,
> node sizing) is reserved for a later ADR (ADR-015), and the
> infrastructure/workload separation gets its own record (ADR-016) if the
> decision needs more depth than the summary here. This record covers only the
> IaC foundation and how state is handled.

## Context

Sprints 1–5 established the pipeline locally: Python engineering, DVC, MLflow,
Docker, CI, testing, and a security-hardened Kubernetes `Job` that runs the
workload to completion ([ADR-009](ADR-009-kubernetes-workload-model.md)–ADR-013).
The remaining conventional proof gap is **cloud infrastructure provisioned as
code**. Sprint 6 closes it with Terraform + AWS + EKS.

The first decision is the **foundation**: how the IaC project is structured,
constrained, and made safe *before* any resource that costs money is declared.
Getting this wrong is expensive later — inconsistent tagging is hard to
retrofit, an unpinned provider makes plans irreproducible, and a single leaked
credential or committed state file compromises the whole account. So PR 1
deliberately provisions nothing and instead nails down the contract every later
PR inherits.

Constraints that shaped the decision:

- The repository must remain **safe to publish publicly** — no credentials, no
  state, no account identifiers.
- The environment is **portfolio-scoped**: small, short-lived, single-operator,
  and torn down after evidence capture. It is not a team production platform.
- Infrastructure and workload responsibilities must stay **separated** —
  Terraform for AWS, Kustomize for Kubernetes workloads, application code for ML.

## Decision

Introduce a single-root Terraform project under **`terraform/`** with the
conventional file split (`versions.tf`, `providers.tf`, `variables.tf`,
`main.tf`, `outputs.tf`, `terraform.tfvars.example`, `README.md`), and adopt the
following contract.

**Version constraints.** Terraform core is pinned to the 1.x line
(`>= 1.6.0, < 2.0.0`) and the AWS provider with a pessimistic `~> 5.60`
constraint. Reproducibility of a `plan` starts with resolving the same tool and
provider everywhere.

**Provider & authentication.** The provider hard-codes **no** credentials. It
resolves them from the standard AWS chain (environment, profile, or assumed
role) at plan/apply time. Region is a variable (default `us-east-1`), not
baked-in provider config.

**Naming & tagging strategy.** A `local.name_prefix` of
`"<project_name>-<environment>"` gives every future resource a consistent,
environment-scoped name. A `local.common_tags` set (`Project`, `Environment`,
`ManagedBy=terraform`, `Owner`, `Repository`, plus optional `additional_tags`)
is applied globally through the provider's `default_tags`, so tagging is correct
by construction rather than remembered per resource. `additional_tags` is merged
so it can *extend* but not override the reserved keys.

**Foundation declares no resources.** `main.tf` contains only locals and two
context data sources (`aws_caller_identity`, `aws_region`) that back outputs
confirming which account/region Terraform targets. The configuration is
`validate`-clean, and an accidental `apply` creates nothing and costs nothing.

**No premature modules.** A single small root module is the honest shape for a
resource-free foundation. `modules/` is added only when a later PR has a real
reusable boundary to extract (network, EKS) — not for appearance.

**State handling — local now, remote-ready.** The first implementation uses the
**default local backend**. `terraform.tfstate` is gitignored and must never be
committed (it can hold resolved account IDs and, for some resources, secrets).
Local state is acceptable here because the environment is single-operator and
short-lived; a remote backend would add S3/lock-table resources whose only
purpose is the portfolio itself. The path to team/production use is documented
and non-disruptive: an S3 backend (versioned, encrypted) + state locking + KMS +
least-privilege IAM, reached via a `backend` block and
`terraform init -migrate-state`.

**Secret handling.** `*.tfvars` carry non-secret configuration only;
`terraform.tfvars.example` holds placeholders and is the one `*.tfvars*` file
committed (via a `!*.tfvars.example` negation). `.gitignore` blocks
`.terraform/`, `*.tfstate*`, `*.tfvars`, crash logs, plan artifacts, and
override files. The `aws_account_id` output is marked `sensitive`.

## Alternatives Considered

1. **Provision resources in PR 1 (jump straight to a VPC/EKS).**
   - *Rejected* — mixes "establish the IaC contract" with "spend money on
     infrastructure." A resource-free foundation is reviewable on its own, keeps
     PR 1 zero-cost, and lets naming/tagging/state decisions land before any
     resource depends on them.
2. **Create a `modules/` scaffold now (network/, iam/, eks/ stubs).**
   - *Rejected* — empty modules are structure for appearance. The Sprint 6 plan
     explicitly says not to create unnecessary modules. Extract a module when a
     real boundary exists.
3. **Remote S3 + DynamoDB backend from the start.**
   - *Deferred* — a remote backend is the right production answer, but here it
     would provision AWS resources solely to host portfolio state, contradicting
     the cost-control principle. Documented as the explicit team/production
     upgrade path instead.
4. **Per-resource tags instead of provider `default_tags`.**
   - *Rejected* — per-resource tagging drifts. `default_tags` makes the common
     set automatic and consistent across everything the provider creates.
5. **A separate top-level `infra/` or embedding Terraform inside `k8s/`.**
   - *Rejected* — `terraform/` at the repository root is the conventional,
     discoverable location and keeps infrastructure cleanly separate from the
     `k8s/` workload configuration, reinforcing the separation of concerns.

## Consequences

**Positive**

- The IaC contract (versions, provider, naming, tagging, state, secrets) is
  fixed and reviewable before any resource depends on it.
- PR 1 is **zero-cost and safe to publish** — no resources, no credentials, no
  state committed.
- Every later resource inherits consistent names and tags automatically.
- The local→remote state path is documented, so scaling to a team is a
  backend swap, not a redesign.

**Trade-offs and follow-ups**

- **Foundation only.** No VPC/IAM/EKS exists yet; `terraform plan` proposes no
  changes. Real infrastructure begins in PR 2.
- **Local state is single-operator.** It is deliberately not suitable for
  concurrent team use; that is the documented remote-backend upgrade.
- **Provider lock committed.** `.terraform.lock.hcl` is committed (and
  intentionally not gitignored) so the AWS provider version and checksums are
  reproducible from this first PR, per HashiCorp's recommendation.
- **`terraform fmt`/`validate` are run manually for now.** Wiring them (and
  `plan`, lint, and security scans) into CI is PR 6 by design; PR 1 does not
  modify the existing CI workflow.

## What This Decision Does *Not* Imply

- It does **not** choose or justify AWS/EKS specifically, node sizing, or the
  cluster topology — that is ADR-015 with PR 4.
- It does **not** provision, or claim to provision, any cloud infrastructure.
  The foundation is validation-clean, not applied.
- It does **not** imply a production platform: local state, single operator, and
  short-lived scope are explicit. Production HA, DR, and multi-region remain out
  of scope for Sprint 6.
- It does **not** move any workload configuration into Terraform. The Kubernetes
  workload stays in [`k8s/`](../../k8s/) (Kustomize); Terraform stops at
  infrastructure.
