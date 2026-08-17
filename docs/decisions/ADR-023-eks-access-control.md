# ADR-023: Explicit EKS Access Entries — closes H-03

- **Status:** Accepted (design)
- **Date:** 2026-08-17
- **Deciders:** Asad Hanif
- **Related:** [`terraform/eks.tf`](../../terraform/eks.tf),
  [`terraform/variables.tf`](../../terraform/variables.tf),
  [`terraform/outputs.tf`](../../terraform/outputs.tf),
  [`terraform/terraform.tfvars.example`](../../terraform/terraform.tfvars.example),
  [`terraform/tests/eks_access_control.tftest.hcl`](../../terraform/tests/eks_access_control.tftest.hcl),
  [`terraform/README.md`](../../terraform/README.md),
  [ADR-017 (Amazon EKS Platform)](ADR-017-eks-platform.md) — **this ADR supersedes
  ADR-017's cluster-access decision**,
  [ADR-016 (AWS IAM Foundation)](ADR-016-aws-iam-foundation.md),
  [ADR-019 (Terraform CI Validation)](ADR-019-terraform-ci-validation.md),
  [ADR-022 (Secure-by-Default EKS API Access)](ADR-022-eks-secure-api-access.md)

> **Scope note.** This ADR ratifies the change that replaces the **automatic
> cluster-creator admin** bootstrap with **explicit, scoped EKS access entries**,
> closing Sprint 6 finding **H-03**. It changes the cluster *access model and
> guardrails* only; it does **not** re-architect the cluster, network, or IAM
> roles (ADR-015/-016/-017 stand), and it introduces no GitOps, no remote state,
> no hardcoded ARNs/account IDs, and no committed credentials.

## Context

[ADR-017](ADR-017-eks-platform.md) provisioned the managed EKS platform with
`authentication_mode = API_AND_CONFIG_MAP` **and**
`bootstrap_cluster_creator_admin_permissions = true`. That was convenient — the IAM
principal that ran `terraform apply` automatically became **cluster-admin
(`system:masters`)**, so `kubectl` worked immediately without editing `aws-auth`.

It is also the finding. The grant is **implicit and identity-ambiguous**: cluster
administration is tied to *whoever happened to create the cluster* (a personal
workstation identity, a CI role, an assumed SSO session — whatever ran `apply`),
not to a declared, reviewable set of principals. There is no record in the
configuration of *who* can administer the cluster or *what* they can do, the grant
is always the broadest possible (full cluster-admin), and it cannot be scoped or
revoked without recreating the cluster or hand-editing access out-of-band. Keeping
`API_AND_CONFIG_MAP` also leaves the legacy `aws-auth` ConfigMap as a second,
harder-to-audit access path.

The Sprint 6 engineering review flagged this as finding **H-03** (cluster creator
admin privileges). The requirement for this PR: **replace** the creator-admin
bootstrap with **explicit EKS access entries**, grant only the identities the
project actually needs, use the **narrowest practical** managed EKS access policy,
**avoid cluster-admin for convenience**, keep the identity configuration
**configurable without hardcoding personal credentials**, **detect the old insecure
bootstrap setting** with validation, and **not** create an unusable cluster.

## Decision

Make cluster access **explicit, scoped, and declared**, in the existing root
module, changing the access configuration and adding guardrails only. The access
chain becomes: **AWS IAM identity → EKS access entry → scoped EKS access policy →
Kubernetes permissions.**

1. **No creator-admin, by default and by tripwire.**
   `bootstrap_cluster_creator_admin_permissions` is now driven by the variable
   `cluster_bootstrap_creator_admin_permissions`, which defaults to **`false`**.
   Its `validation` **rejects `true`** — the old insecure setting cannot be
   reintroduced (even deliberately) without failing `plan`, `apply`, and
   `terraform test`. This is the "detect the old insecure bootstrap setting"
   requirement made executable.

2. **Access entries are the only path, by default.**
   `cluster_authentication_mode` defaults to **`API`** (access entries only — no
   `aws-auth` ConfigMap backdoor). `API_AND_CONFIG_MAP` remains available for a
   migration; **`CONFIG_MAP`** (aws-auth only, access entries ignored) is
   **rejected** by validation because it bypasses the access-entry model.

3. **Explicit, configurable identities — no hardcoded ARNs.** A new
   `cluster_access_entries` map (default **`{}`**) declares each principal and the
   scoped policy it receives. It is populated from a **git-ignored
   `terraform.tfvars`**; the committed `terraform.tfvars.example` carries
   `<account-id>`/`<role>` **placeholders** only. Each entry becomes an
   `aws_eks_access_entry` (registers the principal) plus an
   `aws_eks_access_policy_association` (grants the scoped policy). Validation
   enforces a valid IAM role/user ARN, a policy from the AWS-managed EKS
   access-policy set, and a non-empty namespace list when namespace-scoped.

4. **Narrowest practical policy; no cluster-admin for convenience.** The per-entry
   `policy` defaults to **`AmazonEKSAdminPolicy`** — scoped admin across
   namespaces, **not** `system:masters` — which is sufficient for the single
   operator to create the workload namespace, run the batch `Job`, and inspect it.
   `AmazonEKSViewPolicy`/`AmazonEKSEditPolicy` (optionally namespace-scoped) are
   supported and documented for read-only/deploy-only identities.
   `AmazonEKSClusterAdminPolicy` (full cluster-admin) is accepted but documented as
   a **last resort**; the validation whitelist prevents associating an arbitrary
   broad IAM policy.

5. **Terraform/CI needs no admin entry — documented.** This root module manages
   only AWS-API resources (there is **no `kubernetes`/`helm` provider**), so
   Terraform never calls the Kubernetes API and is granted **no** access entry. CI
   runs `fmt`/`validate`/`test` offline and never touches the cluster. Access
   entries exist solely for the humans/automation that run `kubectl`.

6. **The cluster stays usable.** An empty `cluster_access_entries` is safe, not
   broken: EKS **automatically** creates the managed node group's own access entry,
   so nodes join and addons run regardless; only *human* `kubectl` access requires
   an explicit entry. The operator adds their principal in `terraform.tfvars`
   before validating — mirroring the empty public-CIDR default of ADR-022.

7. **Executable contract tests.** A new offline suite,
   [`tests/eks_access_control.tftest.hcl`](../../terraform/tests/eks_access_control.tftest.hcl),
   runs under `mock_provider "aws"` (`command = plan`, no AWS, no credentials) and
   asserts: no creator-admin by default; `API` auth by default; the bootstrap
   tripwire rejects `true`; `CONFIG_MAP`, invalid principal ARNs, unknown policies,
   and empty namespace scopes are rejected; and scoped cluster- and namespace-level
   entries plan cleanly with the correct EKS cluster-access-policy ARN. This is the
   "detect insecure config / prove the secure model" requirement made runnable, in
   the same no-AWS spirit as ADR-019 and ADR-022.

Cluster access therefore remains **fully configurable** for the ephemeral
validation workflow (declare your operator role, choose a scoped policy), but the
*insecure* shapes — implicit creator-admin, aws-auth-only auth, an unscoped or
arbitrary policy — are all rejected or unrepresentable.

## Alternatives Considered

1. **Keep `bootstrap_cluster_creator_admin_permissions = true` but document it.**
   - *Rejected* — this is precisely the finding: access implicitly tied to the
     creating principal, always full cluster-admin, unreviewable. Documentation
     does not make it explicit or scoped.
2. **Hardcode the operator's ARN (or a default admin entry) so a fresh `apply` is
   immediately usable.**
   - *Rejected* — it would commit a personal ARN/account ID (forbidden) and
     recreate an implicit default admin. An empty default with a git-ignored
     override keeps ARNs out of the repo and forces a conscious grant.
3. **Add a `precondition`/`check` that fails when no access entries are configured
   (force at least one admin).**
   - *Rejected* — "no entries yet" is a legitimate intermediate state (the operator
     populates `terraform.tfvars` before validating), and the cluster is still
     functional (nodes join via the auto-created node access entry). A hard
     precondition would block the private-by-default `apply`; a `check` block fails
     the credential-free `terraform test` gate (check-block failures fail test
     runs) and would break the ECR/API-security suites, which is why the usability
     guardrail is docs + secure-empty-default instead.
4. **Keep `authentication_mode = API_AND_CONFIG_MAP`.**
   - *Not the default* — it leaves the legacy `aws-auth` ConfigMap as a second,
     harder-to-audit access path. `API` (access entries only) is the single,
     explicit model. `API_AND_CONFIG_MAP` is still selectable for a genuine
     migration; `CONFIG_MAP`-only is rejected.
5. **Default the per-entry policy to `AmazonEKSClusterAdminPolicy` for
   convenience.**
   - *Rejected* — that is cluster-admin-for-convenience, exactly what the task
     forbids. `AmazonEKSAdminPolicy` (scoped admin) is sufficient to operate the
     batch `Job`; cluster-admin is a documented last resort.
6. **Grant Terraform/CI its own cluster-admin access entry.**
   - *Rejected as unnecessary* — the module has no Kubernetes/Helm provider and
     never calls the Kubernetes API, so it needs no entry. Adding one would create
     a standing admin identity for no functional reason. (If a future PR adds a
     `kubernetes` provider that manages in-cluster resources, it would get a
     **scoped** entry, documented then.)

## Consequences

**Positive**

- Cluster administration is **explicit and reviewable**: the configuration states
  which principals have access and at what scope, instead of "whoever ran `apply`".
  H-03 is closed at the source.
- The insecure shapes are **unrepresentable**: creator-admin bootstrap (`true`),
  `CONFIG_MAP`-only auth, invalid principals, and non-managed policies all fail
  before any resource is created.
- Access follows **least privilege by default** (`AmazonEKSAdminPolicy`, not
  cluster-admin) and can be narrowed further (View/Edit, namespace-scoped).
- The guarantee is **executable and regression-proof** — the `terraform test` suite
  fails CI if a future change weakens the model.
- **No secrets or personal identifiers** enter the repo: ARNs live only in a
  git-ignored `terraform.tfvars`.

**Trade-offs and follow-ups**

- **Explicit grant required before `kubectl` works.** A fresh `apply` with no
  `cluster_access_entries` yields a cluster no human can administer until the
  operator adds their principal. This is intended (secure default) and documented
  at the variable, in `terraform.tfvars.example`, and in `terraform/README.md`.
- **Behavioural change for existing callers.** Anyone who relied on automatic
  creator-admin must now declare their identity explicitly. This is deliberate: the
  implicit convenience is removed.
- **Live-validation note.** The access entries and policy associations are asserted
  offline by the contract suite; confirming them against a running cluster
  (`aws eks list-access-entries` / `list-associated-access-policies`) requires a
  live `apply`, which is an operator-driven action against their own account (per
  ADR-014/-019/-020), not something CI performs.

## What This Decision Does *Not* Imply

- It does **not** re-architect the network, IAM roles, or cluster shape —
  ADR-015/-016 and the rest of ADR-017 stand; only the access model changes.
- It does **not** claim a production, HA, or hardened-to-completion platform — the
  cluster is still the short-lived validation environment of ADR-020.
- It does **not** introduce GitOps, Terraform remote state, hardcoded account IDs
  or ARNs, or any Kubernetes workload change.
- It does **not** remove the ability to grant broad access — it makes any grant an
  explicit, scoped, declared choice, with cluster-admin reserved as a documented
  last resort.
