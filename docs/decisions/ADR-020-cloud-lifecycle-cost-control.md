# ADR-020: Cloud Environment Lifecycle & Cost Control (Ephemeral, Provision → Prove → Destroy)

- **Status:** Accepted — validated by the [Sprint 6 PR 7 runtime evidence](../proof/sprint-06-runtime-evidence.md) (2026-08-15)
- **Date:** 2026-08-15
- **Deciders:** Asad Hanif
- **Related:** [`docs/cloud-operations.md`](../cloud-operations.md),
  [`terraform/README.md`](../../terraform/README.md),
  [`terraform/variables.tf`](../../terraform/variables.tf),
  [Sprint 6 — Runtime Evidence](../proof/sprint-06-runtime-evidence.md),
  [Sprint 6 — Proof Impact](../proof/sprint-06-proof-impact.md),
  [ADR-014 (Terraform Architecture & Foundation)](ADR-014-terraform-architecture.md),
  [ADR-015 (AWS Network Architecture)](ADR-015-aws-network-architecture.md),
  [ADR-017 (Amazon EKS Platform)](ADR-017-eks-platform.md),
  [ADR-018 (AWS EKS Deployment Overlay)](ADR-018-aws-eks-deployment-overlay.md),
  [ADR-019 (Terraform CI Validation)](ADR-019-terraform-ci-validation.md),
  [Sprint 6 plan](../../Sprint-06-Terraform-Cloud-Platform-Foundation.md)

> **Scope note.** This ADR ratifies the *lifecycle and cost strategy* for the AWS
> platform delivered across Sprint 6 (PR 8, the final PR): the environment is a
> **short-lived, single-operator validation environment** that is provisioned to
> prove one claim, has its evidence captured, and is then **destroyed and verified
> clean**. It ratifies *how the environment is operated and paid for and torn down*,
> and consolidates the cost decisions the earlier PRs made in passing. It adds **no
> new infrastructure** — the sizing/topology decisions live in ADR-015/-017; this
> record ties them to a deliberate cost ceiling and a mandatory teardown.

## Context

By the end of Sprint 6 the repository can provision a real, managed Kubernetes
platform on AWS ([ADR-014](ADR-014-terraform-architecture.md) →
[ADR-018](ADR-018-aws-eks-deployment-overlay.md)) and — as of the PR 7 integration
test — has actually run the MLOps `Job` on it to completion
([runtime evidence](../proof/sprint-06-runtime-evidence.md)). That capability
introduces something the earlier, offline sprints never had: **a way to spend real
money and leave it spending.** An EKS control plane, an EC2 node, and a NAT gateway
each bill **per hour they exist**, independent of whether any work is running.

The project's purpose for this infrastructure is a **portfolio proof** — "the
pipeline runs on managed cloud, with the Sprint 5 security controls intact" — not an
operated production service. A proof needs the environment to exist only long enough
to capture evidence. Left running, the environment would accrue cost indefinitely
for **zero** additional proof, and an *unverified* teardown could silently leave a
NAT gateway or node billing after the operator believes they are done.

The credentials present in the development tool environment belong to a **client**
account, which makes an accidental provision-and-forget especially unacceptable
(see [ADR-019](ADR-019-terraform-ci-validation.md) and the standing rule that all
provisioning is a deliberate, operator-driven, own-account step). The constraint for
this PR: define and document a lifecycle that makes the environment **cheap by
construction and gone by default**, without adding infrastructure and without
overstating what the environment is.

## Decision

Adopt an **ephemeral, evidence-driven environment lifecycle** with a small,
deliberate cost envelope, and document it as the operator's runbook
([`docs/cloud-operations.md`](../cloud-operations.md)).

1. **`provision → prove → destroy` is the whole lifecycle.** The unit of work is the
   *entire environment*, not a service kept alive. It is stood up to prove one thing,
   its evidence is captured, and it is destroyed. Continuous operation is explicitly
   a non-goal.

2. **Teardown is mandatory and the primary operation** — `terraform destroy` **plus
   verification that AWS resources are gone**, never the destroy line alone.
   Verification is checked from three angles: Terraform state empty, the AWS API
   showing no cluster/NAT/EIP, and the ECR repository absent. Cleanup must not be
   *claimed* until verification passes; a *delayed* teardown must be stated as
   still-billing, not reported as clean.

   > **Update (Sprint 7, PR 1 — [ADR-021](ADR-021-terraform-managed-ecr.md)):** the
   > ECR repository is now **Terraform-managed** with `force_delete`, so
   > `terraform destroy` removes it in the same pass — the separate
   > `aws ecr delete-repository --force` step is gone. The three-angle verification
   > (including confirming the repository is absent) still stands; only the deletion
   > mechanism changed from out-of-band to Terraform-owned.

3. **Small by construction — the minimum that still proves the claim.** Time-alive is
   the dominant cost lever, and every sizing knob is set to the cheapest value that
   still demonstrates the pipeline running:
   - **1–2 `t3.medium` on-demand node(s)**, no GPU, no Cluster Autoscaler
     ([ADR-017](ADR-017-eks-platform.md)); the PR 7 run used **1**.
   - **A single shared NAT gateway**, not one per AZ
     ([ADR-015](ADR-015-aws-network-architecture.md)).
   - **Two AZs**, the EKS control-plane minimum — not a resilience target.
   - **Local Terraform state**, no remote backend — a backend would add billable AWS
     resources (S3 + lock table) whose only purpose is the portfolio
     ([ADR-014](ADR-014-terraform-architecture.md)).

4. **The cost drivers are documented and ranked**, with time-alive named as the lever
   (control plane and node per-hour; NAT per-hour + per-GB; EBS/CloudWatch/ECR minor;
   IAM/VPC/subnets/IGW/addons free) in
   [`docs/cloud-operations.md § 4`](../cloud-operations.md#4-aws-cost-drivers). Figures
   are order-of-magnitude and point to live AWS pricing; the goal is to rank drivers
   and justify promptness, not to quote a bill.

5. **The endpoint is locked to the operator and no static credentials exist.**
   `cluster_endpoint_public_access_cidrs` is set to the operator's own `/32`, not
   `0.0.0.0/0`; all AWS access is via the standard credential chain and short-lived
   `update-kubeconfig` tokens; no account ID, state, kubeconfig, or secret is
   committed (the ECR image stays a `000000000000` placeholder in git).

This ADR **ratifies decisions already exercised** — the PR 7 run provisioned 29
resources, ran the Job to completion (exit 0, 52s), and destroyed all 29 the same
day, verified clean. This record makes that lifecycle the *documented default*, not
a one-off.

## What is intentionally *not* in this decision

- **No new infrastructure** — no scheduler/auto-destroy Lambda, no budget-alarm
  stack, no cost-monitoring service. Those are themselves billable and would
  contradict "small by construction"; promptness is enforced by procedure and the
  runbook, not by more cloud.
- **No always-on environment**, no remote state backend, no multi-environment
  (dev/staging/prod) split — a single ephemeral environment is the honest shape.
- **No production cost-optimization at scale** — this is not Reserved Instances /
  Savings Plans / Spot-fleet engineering; it is "keep it tiny and delete it quickly."
- **No claim of HA, DR, multi-region, or production observability** — none are
  provisioned; see [`docs/cloud-operations.md § 7`](../cloud-operations.md#7-limitations).

## Alternatives Considered

1. **Keep the environment running (a persistent cluster).**
   - *Rejected* — it bills per hour for a proof that is already captured. A portfolio
     proof does not need a standing cluster; the evidence doc is the durable
     artifact, not the live infrastructure.
2. **Automate teardown with a scheduled auto-destroy (e.g. a Lambda + EventBridge).**
   - *Rejected for now* — it adds billable, stateful infrastructure and its own
     failure modes to manage a single short-lived environment. For one operator
     running one proof, a documented mandatory-teardown procedure is simpler and
     safer. Auto-destroy is a reasonable enhancement only for a team running many
     ephemeral environments.
3. **Add AWS Budgets / billing alarms as IaC.**
   - *Deferred* — a real cost guardrail for a persistent or team environment, but it
     is more infrastructure for a run measured in minutes, and this PR's rule is "add
     no new infrastructure." Documented as a follow-up for any longer-lived
     environment.
4. **Trust `terraform destroy`'s success line as proof of cleanup.**
   - *Rejected* — `destroy` can partially fail. Independent verification (state empty
     + AWS API + ECR) is required precisely because the cost of a false "clean" is
     ongoing silent billing. (At the time of this ADR the ECR repo was also outside
     Terraform state entirely; Sprint 7 PR 1 brought it under management, but the
     verification discipline is unchanged.)
5. **Use a remote state backend (S3 + DynamoDB) from the start.**
   - *Rejected here, ratified in [ADR-014](ADR-014-terraform-architecture.md)* — it
     would add billable AWS resources whose only purpose is a single-operator
     portfolio environment. Local state is the honest choice for this scope; the
     migration path is documented.
6. **Larger/production-shaped topology (multi-node, NAT-per-AZ, three AZs) to look
   more "real".**
   - *Rejected* — it multiplies cost without strengthening the (narrow, honest)
     claim, and would misrepresent a validation run as a production deployment. The
     small size *is* the honest claim.

## Consequences

**Positive**

- **Cheap by construction, gone by default.** The dominant costs are per-hour and the
  environment's lifetime is minutes-to-hours, so a full run costs a rounding error —
  and the mandatory, *verified* teardown means it does not keep costing after.
- **Honest scope.** The lifecycle and its limits are documented in one runbook; the
  environment is never presented as more than a short-lived validation proof.
- **Safe.** No standing credentials, an operator-locked endpoint, verified cleanup,
  and no account identifiers in git — consistent with the whole-sprint security
  posture ([ADR-019](ADR-019-terraform-ci-validation.md), [SECURITY.md](../../SECURITY.md)).
- **Reproducible.** Anyone with their own account can follow
  [`docs/cloud-operations.md`](../cloud-operations.md) to reproduce the run and tear
  it down, using the same pinned Terraform provider lock.

**Trade-offs and follow-ups**

- **Teardown discipline is procedural, not enforced by the platform.** An operator
  who skips verification can still leak a billing resource. Auto-destroy and AWS
  Budgets alarms are the documented enhancements for a longer-lived or team context.
- **No persistent environment means no long-running observability/soak evidence** —
  acceptable because none is claimed; a persistent environment is future roadmap work
  (v5–v6), where a remote backend, budgets, and monitoring would all be revisited.
- **The out-of-band ECR repository is a manual teardown step** (outside Terraform
  state); the runbook calls it out explicitly so it is not forgotten.
  > **Resolved (Sprint 7, PR 1 — [ADR-021](ADR-021-terraform-managed-ecr.md)):** the
  > ECR repository is now Terraform-managed and removed by `terraform destroy`
  > (`force_delete`), eliminating this manual step.

## What This Decision Does *Not* Imply

- It does **not** claim a production, HA, DR, multi-region, or continuously-operated
  cloud platform — the environment is ephemeral and single-region by design.
- It does **not** add any new billable infrastructure; cost control is procedural and
  by-sizing, not a new service.
- It does **not** supersede the sizing ADRs ([ADR-015](ADR-015-aws-network-architecture.md),
  [ADR-017](ADR-017-eks-platform.md)) — it consolidates their cost rationale and binds
  it to a mandatory, verified teardown.
- It does **not** permit reporting cleanup as complete before
  [verification](../cloud-operations.md#52-verify-aws-resources-are-gone) passes.
