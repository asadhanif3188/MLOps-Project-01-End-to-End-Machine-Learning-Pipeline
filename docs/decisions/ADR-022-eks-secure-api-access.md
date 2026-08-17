# ADR-022: Secure-by-Default EKS API Access — closes H-02

- **Status:** Accepted (design)
- **Date:** 2026-08-17
- **Deciders:** Asad Hanif
- **Related:** [`terraform/eks.tf`](../../terraform/eks.tf),
  [`terraform/variables.tf`](../../terraform/variables.tf),
  [`terraform/tests/eks_api_security.tftest.hcl`](../../terraform/tests/eks_api_security.tftest.hcl),
  [`terraform/.trivyignore`](../../terraform/.trivyignore),
  [`terraform/README.md`](../../terraform/README.md),
  [ADR-017 (Amazon EKS Platform)](ADR-017-eks-platform.md) — **this ADR supersedes
  ADR-017's endpoint-posture decision**,
  [ADR-019 (Terraform CI Validation)](ADR-019-terraform-ci-validation.md),
  [ADR-020 (Cloud Lifecycle & Cost Control)](ADR-020-cloud-lifecycle-cost-control.md),
  [ADR-021 (Terraform-Managed ECR)](ADR-021-terraform-managed-ecr.md)

> **Scope note.** This ADR ratifies the change that makes the **EKS control-plane
> API endpoint secure by default**, closing Sprint 6 finding **H-02**. It changes
> the endpoint *defaults and guardrails* only; it does **not** re-architect the
> cluster, network, or IAM (ADR-015/-016/-017 stand), and it introduces no GitOps,
> no remote state, and no unrelated Kubernetes changes.

## Context

[ADR-017](ADR-017-eks-platform.md) provisioned the managed EKS platform and, for
first-run convenience, made the Kubernetes API server reachable **both privately
and publicly by default**, with the public source range
(`cluster_endpoint_public_access_cidrs`) **defaulting to `0.0.0.0/0`**. That was
convenient for validating a throwaway cluster with `kubectl` from any workstation,
but it is an **insecure default**: a fresh `apply` with no operator overrides
stands up a control plane whose API is reachable from the entire internet. The
posture was only "safe" if the operator remembered to narrow the CIDR — security
by documentation, not by construction.

The Sprint 6 engineering review flagged this as finding **H-02** (EKS API access
insecure by default). The two consequences were also visible in the IaC scan
surface: `terraform/.trivyignore` had to **suppress** AVD-AWS-0040 ("public access
enabled") and AVD-AWS-0041 ("open public access CIDRs") to keep CI green — i.e. we
were muting the scanner for a real exposure rather than removing the exposure.

The requirement for this PR: make the API access **secure by default**, prefer
**private** endpoint access, keep public access **configurable but explicit
opt-in**, **never** allow an unrestricted `0.0.0.0/0`, and make the secure default
an **executable** guarantee (validation/preconditions + contract tests) rather
than a comment. The fix must not simply hide the insecure setting.

## Decision

Make the endpoint posture **secure by default and self-enforcing**, in the
existing root module, changing defaults and adding guardrails only.

1. **Private by default.** `cluster_endpoint_private_access` defaults **`true`**
   (unchanged) and `cluster_endpoint_public_access` now defaults **`false`**
   (was `true`). A default `apply` yields a **private-only** API server — no
   internet exposure without a deliberate choice.

2. **Empty public allow-list by default.** `cluster_endpoint_public_access_cidrs`
   now defaults to **`[]`** (was `["0.0.0.0/0"]`). With public access off, there
   is nothing to expose.

3. **`0.0.0.0/0` is impossible, by validation.** The CIDR variable has two
   `validation` blocks: every entry must be a valid CIDR, and **no entry may be a
   `/0`** (which includes `0.0.0.0/0`). An unrestricted range fails `plan`,
   `apply`, and `terraform test` — it can never be configured, even deliberately.

4. **Public opt-in requires a scoped allow-list, by precondition.** A `lifecycle`
   **precondition** on `aws_eks_cluster.this` rejects
   `cluster_endpoint_public_access = true` combined with an **empty** CIDR list —
   because EKS treats an empty `public_access_cidrs` as `0.0.0.0/0`. So opting into
   public access *forces* an explicit, scoped operator CIDR.

5. **The API can't be made unreachable, by precondition.** A second precondition
   rejects disabling **both** endpoints (`private = false` **and** `public =
   false`), which would strand the control plane.

6. **Executable contract tests.** A new offline suite,
   [`tests/eks_api_security.tftest.hcl`](../../terraform/tests/eks_api_security.tftest.hcl),
   runs under `mock_provider "aws"` (`command = plan`, no AWS, no credentials) and
   asserts: private-by-default; `0.0.0.0/0` rejected; any `/0` rejected; an invalid
   CIDR rejected; public-without-CIDRs rejected; both-endpoints-off rejected; and
   the scoped public opt-in plans cleanly. This is the "detect insecure config"
   requirement made runnable, in the same no-AWS spirit as ADR-019.

7. **Remove the suppressions, don't re-justify them.** Because the default is now
   private with no open CIDR, Trivy no longer raises AVD-AWS-0040/0041, so both
   entries are **deleted** from `.trivyignore`. The scan now *confirms* the secure
   posture instead of being told to ignore an insecure one. (AVD-AWS-0039 for KMS
   envelope encryption and AVD-AWS-0164 for public-subnet IPs remain — they are
   unrelated, still-intentional trade-offs.)

The public endpoint therefore remains **fully configurable** for the ephemeral
validation workflow (opt in, scope to your `/32`), but the *insecure* shapes —
default-public, open-CIDR, empty-list-public, both-off — are all rejected by the
configuration itself.

## Alternatives Considered

1. **Keep both endpoints on by default but validate the CIDR isn't `0.0.0.0/0`.**
   - *Rejected* — it still exposes a public endpoint by default and only narrows
     the range. "Secure by default" means no public exposure unless opted in;
     private-by-default is the stronger, correct posture.
2. **Private-only with no public option at all.**
   - *Rejected* — the project's ephemeral `provision → prove → destroy` workflow
     (ADR-020) validates the cluster with `kubectl` from the operator's
     workstation, which a private-only endpoint blocks without a bastion/VPN
     (out of scope for a portfolio validation run). Keeping a **scoped** public
     opt-in preserves that workflow without a standing insecure default.
3. **Only flip the defaults; keep the `0.0.0.0/0` guard as documentation.**
   - *Rejected* — the finding is specifically that safety relied on documentation.
     The guardrails must be executable (validation + preconditions + tests), so a
     future edit that reintroduces the exposure fails CI.
4. **Cross-variable `validation` referencing other variables (TF ≥ 1.9) instead of
   resource preconditions.**
   - *Not chosen* — resource `lifecycle` preconditions express the cross-variable
     invariants ("public ⇒ non-empty CIDRs", "not both off") and evaluate during
     `plan`/`terraform test` without raising the module's `required_version`
     floor. They also read naturally at the resource they protect.
5. **Keep suppressing AVD-AWS-0040/0041 with updated wording.**
   - *Rejected* — that would be hiding the setting, exactly what the task forbids.
     A genuine fix lets the scanner pass on its own, so the suppressions are
     removed.

## Consequences

**Positive**

- A fresh `apply` is **private by default** — no internet-reachable API without an
  explicit, scoped opt-in. H-02 is closed at the source, not masked.
- The insecure shapes are **unrepresentable**: `0.0.0.0/0`/`/0`, public-with-empty
  CIDRs, and both-endpoints-off all fail before any resource is created.
- The guarantee is **executable and regression-proof** — the `terraform test`
  suite fails CI if a future change weakens the posture.
- The **IaC scan surface shrinks honestly**: two suppressions are removed because
  the exposure is gone, so Trivy now validates the secure default.

**Trade-offs and follow-ups**

- **Private-only needs in-VPC reachability.** With the secure default, `kubectl`
  from a workstation outside the VPC will not reach the API. Operators either use
  the **scoped public opt-in** (`public_access = true` + their `/32`) for a
  validation run, or reach the private endpoint via a **bastion/VPN/SSM/in-VPC
  runner**. This is documented at the variable, in `terraform/README.md`, and in
  the cloud-operations runbook.
- **Behavioural change for existing callers.** Anyone who relied on the old
  default-public behaviour must now opt in explicitly. This is intended: the
  insecure convenience is removed deliberately.
- **KMS envelope encryption of secrets still deferred** (AVD-AWS-0039) — unrelated
  to H-02 and tracked separately (ADR-017 follow-up).

## What This Decision Does *Not* Imply

- It does **not** re-architect the network, IAM, or cluster shape — ADR-015/-016
  and the rest of ADR-017 stand; only the endpoint posture changes.
- It does **not** claim a production, HA, or hardened-to-completion platform — the
  cluster is still the short-lived validation environment of ADR-020, and KMS
  envelope encryption remains a documented follow-up.
- It does **not** introduce GitOps, Terraform remote state, or any Kubernetes
  workload change.
- It does **not** remove the ability to use a public endpoint — it makes that use
  an explicit, CIDR-scoped, never-unrestricted opt-in.
