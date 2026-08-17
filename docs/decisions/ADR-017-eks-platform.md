# ADR-017: Amazon EKS Platform (Cluster, Managed Node Group, Core Addons)

- **Status:** Accepted (validated) — design ratified, then provisioned, exercised, and torn down in the [Sprint 6 PR 7 runtime test](../proof/sprint-06-runtime-evidence.md) (2026-08-15)
- **Date:** 2026-08-14
- **Deciders:** Asad Hanif
- **Related:** [`terraform/eks.tf`](../../terraform/eks.tf),
  [`terraform/variables.tf`](../../terraform/variables.tf),
  [`terraform/outputs.tf`](../../terraform/outputs.tf),
  [`terraform/README.md`](../../terraform/README.md),
  [ADR-014 (Terraform Architecture & Foundation)](ADR-014-terraform-architecture.md),
  [ADR-015 (AWS Network Architecture)](ADR-015-aws-network-architecture.md),
  [ADR-016 (AWS IAM Foundation)](ADR-016-aws-iam-foundation.md),
  [ADR-009 (Kubernetes Workload Model — Job)](ADR-009-kubernetes-workload-model.md),
  [Sprint 6 plan](../../Sprint-06-Terraform-Cloud-Platform-Foundation.md)

> **Scope note.** This ADR ratifies the *managed Kubernetes platform* delivered
> in Sprint 6, PR 4: an EKS control plane, one small managed node group, and the
> three core EKS addons. It consumes the network (ADR-015) and IAM roles
> (ADR-016) built by earlier PRs and adds **no application/workload resources** —
> the MLOps `Job` stays in Kustomize and is wired to the cluster in a later PR.
> This record covers the platform shape and why it is built this way.

## Context

[ADR-014](ADR-014-terraform-architecture.md) set the Terraform foundation,
[ADR-015](ADR-015-aws-network-architecture.md) the network, and
[ADR-016](ADR-016-aws-iam-foundation.md) the two IAM roles a managed EKS cluster
requires. The remaining foundation piece is the Kubernetes platform itself: a
control plane and worker capacity on which the existing Sprint 5 batch `Job`
([ADR-009](ADR-009-kubernetes-workload-model.md)) can run.

The design constraints are set by the sprint: the platform must be **real**
(provisioned and validated, not just `validate`-clean), **minimal and
cost-conscious** (short-lived portfolio validation, not production), and must
**not** pull in the sprint's explicit non-goals — GPUs, autoscaling, service
mesh, ingress stack, observability stack, unrelated AWS services, or workload
resources. It must reuse the ADR-016 roles rather than invent new IAM, and keep
infrastructure (Terraform) cleanly separated from workload configuration
(Kustomize).

## Decision

Author `terraform/eks.tf` in the root module (no premature `modules/`, per
ADR-014/-015/-016) with exactly three resource kinds: one `aws_eks_cluster`, one
`aws_eks_node_group`, and the three core `aws_eks_addon`s — everything
variable-driven so the cluster stays small by default but resizes without editing
resource definitions.

**Control plane.**

- *IAM:* assumes the dedicated control-plane role from ADR-016
  (`aws_iam_role.eks_cluster`); the node group assumes the node role. No new IAM
  is created here. The cluster `depends_on` the `AmazonEKSClusterPolicy`
  attachment so the role is ready before creation.
- *Kubernetes version:* pinned **explicitly** to **1.35** via
  `kubernetes_version`. An explicit, comfortably-supported pin (one minor below
  the region's newest/default at authoring time) is chosen over tracking the
  latest, so a `plan`/`apply` means the same version everywhere and upgrades are
  deliberate. EKS manages the patch level.
- *Networking:* the control plane places its cross-account ENIs across **both**
  the private and public subnets from ADR-015 (so it can reach nodes in the
  private subnets and support later public/internal LB discovery); worker nodes
  are placed in the **private subnets only**.
- *Endpoint/security:* **both** private and public API access are enabled — the
  private endpoint keeps in-VPC traffic off the public path; the public endpoint
  lets an operator validate with `kubectl`. The public source range
  (`cluster_endpoint_public_access_cidrs`) defaults to open **only for first-run
  validation** and is documented as "set to your operator CIDR" for real use.
  > **Superseded by [ADR-022](ADR-022-eks-secure-api-access.md) (2026-08-17, closes
  > H-02).** This "public-on, `0.0.0.0/0`-by-default" posture was the H-02 finding.
  > The endpoint is now **private by default** (`cluster_endpoint_public_access =
  > false`, CIDR list `[]`); public access is a **scoped, explicit opt-in** that can
  > never be `0.0.0.0/0`, enforced by variable validation and cluster preconditions
  > and pinned by `tests/eks_api_security.tftest.hcl`. The rest of this ADR stands.

  Access uses **EKS access entries** (`authentication_mode =
  API_AND_CONFIG_MAP`) with `bootstrap_cluster_creator_admin_permissions = true`,
  so the creating principal gets cluster-admin without hand-editing `aws-auth`.
  Control-plane logging ships the **security-relevant** types (`api`, `audit`,
  `authenticator`) to CloudWatch, toggleable to `[]`.

**Managed node group.**

- *Sizing:* a **fixed pair of `t3.medium`** (2 vCPU / 4 GiB) **on-demand** nodes
  on the **Amazon Linux 2023** EKS AMI (`AL2023_x86_64_STANDARD` — AL2 is
  deprecated for recent Kubernetes), 20 GiB root volume. That comfortably hosts
  the single batch `Job` plus EKS system pods (CoreDNS, and the `aws-node`/
  `kube-proxy` DaemonSets) and nothing more.
- *No autoscaling:* `min = max = desired = 2` and **no Cluster Autoscaler** is
  installed, so the group does not scale on its own — the sprint forbids
  autoscaling unless genuinely necessary, and a fixed batch validation workload
  does not need it.
- *No SSH:* no `remote_access` block — nodes are managed, not logged into,
  removing an access surface. Nodes have no public IP (private subnets) and reach
  the internet outbound only via the ADR-015 NAT gateway.
- The node group `depends_on` the three ADR-016 policy attachments so nodes never
  launch before their permissions exist.

**Core addons only.** `vpc-cni`, `coredns`, and `kube-proxy` — the minimum for a
functioning cluster. `addon_version` is left unset so EKS installs the default
version for the pinned Kubernetes version (deterministic, in lock-step with the
control plane); `resolve_conflicts_on_create/update = OVERWRITE` lets the managed
addons take over the self-managed defaults. The `vpc-cni` addon runs with the
node role's `AmazonEKS_CNI_Policy` (ADR-016) — no separate IRSA role in this PR.
CoreDNS needs schedulable nodes, so all three `depend_on` the node group.

**Outputs.** All EKS outputs are **non-sensitive** — cluster name, endpoint URL,
version, cluster security-group ID, OIDC issuer URL, node-group name, and a
ready-to-run `configure_kubectl` command. No kubeconfig, token, or certificate is
emitted; operators fetch short-lived credentials with `aws eks
update-kubeconfig`. The API endpoint URL is not a secret — access is gated by IAM
and the public-CIDR allow-list.

**Naming & tagging.** Cluster `…-eks`, node group `…-eks-ng`, addons
`…-eks-addon-<name>`, all under the ADR-014 `"<project>-<environment>-…"` prefix
and inheriting the common tag set via `default_tags`; only a `Name` tag (and a
`role` node label) is set per resource.

## What is intentionally *not* provisioned

- **No GPU nodes** — sprint non-goal; the workload is CPU-only.
- **No autoscaling / Cluster Autoscaler** — the node group is fixed-size.
- **No service mesh** (Istio/Linkerd), **no ingress controller/stack**, **no
  observability stack** (Prometheus/Grafana/logging agents) — all sprint
  non-goals; the batch `Job` has no inbound surface and needs none of them.
- **No optional addons** — EBS/EFS CSI drivers, AWS Load Balancer Controller,
  etc. The `Job` uses no persistent volumes or load balancers.
- **No additional AWS services** (RDS, ElastiCache, Lambda, …) and **no
  application/workload resources** — infrastructure stays in Terraform, the
  workload stays in Kustomize (separation of concerns).
- **No customer-managed KMS envelope encryption of secrets** — a recognized
  hardening, deferred to keep the PR minimal and avoid an extra billable
  resource; recorded as a follow-up. EKS still encrypts etcd at rest with an
  AWS-owned key by default.
- **No SSH/remote node access** and **no static credentials or kubeconfig**.

## Alternatives Considered

1. **Self-managed nodes / launch template instead of a managed node group.**
   - *Rejected* — managed node groups handle bootstrap, the instance profile,
     draining, and version updates for us, which is exactly the operational
     burden a portfolio validation cluster should avoid. The sprint explicitly
     prefers managed node groups.
2. **A community EKS module (e.g. `terraform-aws-modules/eks`).**
   - *Rejected for now* — the module is excellent but large and opinionated; for
     a deliberately minimal, well-understood cluster, a handful of first-party
     resources is more legible and keeps the "no premature modules" stance of
     ADR-014. The module remains a reasonable future refactor if reuse appears.
3. **Track the newest/default Kubernetes version automatically.**
   - *Rejected* — an explicit pin is reproducible and makes upgrades a reviewed,
     deliberate change rather than drift. 1.35 is current and comfortably
     supported.
4. **SPOT capacity for the nodes.**
   - *Not the default* — SPOT is cheaper and is offered as a variable
     (`node_capacity_type`), but ON_DEMAND gives predictable capacity for a
     one-shot validation run where a mid-run interruption would just add noise.
5. **Public-only or private-only API endpoint.**
   - *Rejected* — private-only would block `kubectl` validation from an operator
     workstation without a bastion/VPN (out of scope); public-only would forgo
     the in-VPC private path. Enabling both, with a documented public-CIDR
     allow-list, is the balanced default.
   - > **Revised by [ADR-022](ADR-022-eks-secure-api-access.md).** The default is
     > now **private**, with public access as a scoped opt-in — the secure-by-default
     > posture that closes H-02. The "balanced default" above relied on the operator
     > narrowing an open CIDR; that safety is now enforced by the configuration.
6. **Enable customer-managed KMS envelope encryption now.**
   - *Deferred* — it adds a KMS key (a billable resource) and complexity for
     marginal benefit on a short-lived cluster with no real secrets. Documented
     as a follow-up rather than silently omitted.
7. **Add optional addons (EBS CSI, ALB controller) "in case".**
   - *Rejected* — unused addons are footprint and cost the workload does not
     need; they can be added when a workload justifies them.

## Consequences

**Positive**

- A real, Terraform-managed EKS platform exists: `apply` yields an Active
  cluster with Ready nodes and working `kubectl`, satisfying the PR 4 acceptance
  criteria and the sprint's "evidence, not claims" principle.
- The platform reuses the ADR-016 roles exactly as designed; infrastructure and
  workload stay cleanly separated (Terraform vs. Kustomize).
- The footprint is minimal and every knob (version, sizing, endpoint, logging) is
  an explicit, documented variable — easy to review and to resize.
- Connection details are exported without exposing any secret; access is
  IAM-gated.

**Trade-offs and follow-ups**

- **Cost while running.** The control plane bills hourly and the two on-demand
  nodes bill hourly, on top of the NAT gateway — the cluster is **short-lived by
  design** and must be `terraform destroy`ed after evidence capture (PR 8).
- **Public endpoint defaults open** for first-run validation; a real environment
  must restrict `cluster_endpoint_public_access_cidrs`. This is documented at the
  variable and in the README, not left implicit.
  > **Resolved by [ADR-022](ADR-022-eks-secure-api-access.md) (H-02):** the public
  > endpoint no longer defaults open — it defaults **off** (private-only), and when
  > opted into it must be scoped to an explicit CIDR (never `0.0.0.0/0`), enforced
  > by validation, preconditions, and a contract test rather than left to operator
  > discipline.
- **CNI runs on the node role**, and **secrets lack KMS envelope encryption** —
  both recognized hardenings deferred with rationale (OIDC/IRSA and KMS
  respectively).
- **No HA guarantees beyond the managed control plane** — a single small node
  group across two AZs is a validation cluster, not a production platform.

## What This Decision Does *Not* Imply

- It does **not** claim a production, HA, or multi-region EKS platform — it is a
  deliberately small validation cluster (see the sprint's "claims still not
  allowed").
- It does **not** add GPUs, autoscaling, a mesh, ingress, observability, extra
  AWS services, or any application/workload resource.
- It does **not** create static AWS credentials or store a kubeconfig; access is
  via short-lived, IAM-derived credentials.
- It does **not** finalize node-level IAM hardening (CNI-via-IRSA) or secret
  envelope encryption — both are documented follow-ups.
