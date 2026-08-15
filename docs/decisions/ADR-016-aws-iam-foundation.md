# ADR-016: AWS IAM Foundation for EKS (Cluster & Node Roles)

- **Status:** Accepted (validated) — design ratified, then provisioned, exercised, and torn down in the [Sprint 6 PR 7 runtime test](../proof/sprint-06-runtime-evidence.md) (2026-08-15)
- **Date:** 2026-08-14
- **Deciders:** Asad Hanif
- **Related:** [`terraform/iam.tf`](../../terraform/iam.tf),
  [`terraform/README.md`](../../terraform/README.md),
  [ADR-014 (Terraform Architecture & Foundation)](ADR-014-terraform-architecture.md),
  [ADR-015 (AWS Network Architecture)](ADR-015-aws-network-architecture.md),
  [Sprint 6 plan](../../Sprint-06-Terraform-Cloud-Platform-Foundation.md),
  [`SECURITY.md`](../../SECURITY.md)

> **Scope note.** This ADR ratifies the *IAM foundation* delivered in Sprint 6,
> PR 3: the two IAM roles a managed EKS cluster needs — a control-plane role and
> a worker-node role — with their trust relationships and the AWS-managed policy
> attachments EKS requires. It provisions **no EKS, no EC2, no application
> resources, and no static credentials**. The EKS cluster and node group that
> consume these roles arrive in the next PR. This record covers only the IAM
> shape and why it is built this way.

## Context

[ADR-014](ADR-014-terraform-architecture.md) set the resource-free Terraform
foundation and [ADR-015](ADR-015-aws-network-architecture.md) laid the network.
Before a managed EKS cluster can be created, two IAM identities must already
exist:

- **A control-plane role** that the EKS service assumes to manage AWS resources
  on the cluster's behalf (creating cluster ENIs in the subnets from ADR-015,
  managing the cluster security group, etc.).
- **A worker-node role** that each EC2 instance in the managed node group
  assumes so the kubelet can join the cluster, the Amazon VPC CNI can wire pod
  networking, and the container runtime can pull images from ECR.

EKS defines a small, well-known set of AWS-managed policies for each. The design
question is therefore **not** "what permissions to invent" but "how to grant
exactly what EKS requires and nothing more, safely, as code" — while honouring
the sprint's least-privilege and no-static-credentials requirements and the
repository's existing security posture ([`SECURITY.md`](../../SECURITY.md),
[ADR-010](ADR-010-kubernetes-security-hardening.md)).

## Decision

Author `terraform/iam.tf` in the root module (no premature `modules/`, per
ADR-014/-015) with **two dedicated roles**, each trusted by exactly one AWS
service principal, and **only AWS-managed policy attachments** — no inline or
custom policy, and no wildcard authored by this project.

**EKS cluster (control-plane) role.**

- *Purpose:* the identity the EKS managed control plane assumes to operate the
  cluster's AWS resources.
- *Trust relationship:* `sts:AssumeRole` allowed to the **`eks.amazonaws.com`**
  service principal only — no account, user, or role principal is trusted, so
  nothing but the EKS service can assume it.
- *Permissions:* the single AWS-managed **`AmazonEKSClusterPolicy`**, which is
  the only policy EKS requires on this role.

**EKS worker-node role.**

- *Purpose:* the instance-profile identity for the managed node group's EC2
  nodes.
- *Trust relationship:* `sts:AssumeRole` allowed to the **`ec2.amazonaws.com`**
  service principal only (nodes are EC2 instances).
- *Permissions:* the three AWS-managed policies EKS documents as required —
  **`AmazonEKSWorkerNodePolicy`** (join the cluster), **`AmazonEKS_CNI_Policy`**
  (VPC CNI pod networking), and **`AmazonEC2ContainerRegistryReadOnly`**
  (read-only image pulls from ECR).

**No static credentials.** No access key, secret key, IAM user, login profile,
or long-lived credential is created. The roles are assumed by AWS services via
their trust policies; Terraform itself authenticates through the standard AWS
credential chain (ADR-014). Nothing secret enters the repository, state aside.

**Partition-correct ARNs.** Managed-policy ARNs are built from
`data.aws_partition.current.partition` rather than a hard-coded `aws`, so the
configuration is correct in GovCloud/China partitions without edits.

**Naming & tagging.** Roles follow the `"<project>-<environment>-…"` prefix and
inherit the common tag set via `default_tags` (ADR-014); only a `Name` tag is
set per resource.

**Outputs.** Role **names** are exported plainly; role **ARNs** are exported but
marked `sensitive`, because an IAM role ARN embeds the AWS account ID, which this
project already treats as sensitive (the `aws_account_id` output). This keeps the
account ID out of plan/CI logs while still handing the EKS PR the identities it
needs.

## What is intentionally *not* permitted

- **No `AdministratorAccess` / `PowerUserAccess`** — nowhere in this PR.
- **No project-authored wildcard.** Any `*` actions/resources live *inside*
  AWS-owned managed policies (notably `AmazonEKS_CNI_Policy`, which needs several
  `ec2:*NetworkInterface`/`ec2:Describe*` actions for the CNI); those are
  AWS-maintained and are the documented minimum for the CNI to function. This PR
  writes no inline policy of its own.
- **`AmazonEKSVPCResourceController` is intentionally omitted** from the cluster
  role. It is only needed for "security groups for pods", which the batch-`Job`
  workload does not use.
- **`AmazonSSMManagedInstanceCore` is intentionally omitted** from the node role.
  No SSM/interactive node access is required; leaving it off keeps nodes to the
  EKS-mandated minimum.
- **No IAM users, groups, or access keys**, and **no instance profile secrets**.

## Alternatives Considered

1. **One shared role for cluster and nodes.**
   - *Rejected* — the two identities have different trust principals
     (`eks.amazonaws.com` vs `ec2.amazonaws.com`) and different permission needs.
     Merging them would over-grant both and blur purpose. Dedicated roles are the
     least-privilege, AWS-recommended shape.
2. **Hand-write custom inline policies instead of AWS-managed policies.**
   - *Rejected* — EKS's required permissions evolve with the service; AWS keeps
     the managed policies current. Re-deriving them by hand would be broader (to
     be safe), more brittle, and higher-maintenance than attaching the
     AWS-maintained set. Managed policies are the documented, least-surprising
     choice here.
3. **Give the VPC CNI its own IRSA role instead of the node role.**
   - *Deferred* — moving `AmazonEKS_CNI_Policy` to a dedicated IAM-Roles-for-
     Service-Accounts role removes CNI permissions from the node instance
     profile (a real hardening). It depends on the cluster's OIDC provider, which
     does not exist until the EKS PR. Recorded as a follow-up; the standard
     node-role attachment is used now.
4. **Attach `AmazonEKSVPCResourceController` / `AmazonSSMManagedInstanceCore`
   pre-emptively "in case".**
   - *Rejected* — neither is needed by this workload; adding unused permissions
     is exactly the over-grant least privilege forbids. They can be added if a
     future workload justifies them.
5. **Export role ARNs as non-sensitive outputs.**
   - *Rejected* — an IAM role ARN contains the AWS account ID, which the project
     deliberately keeps out of logs (ADR-014's sensitive `aws_account_id`).
     Marking the ARN outputs `sensitive` is the consistent choice; role names
     remain plain for convenience.

## Consequences

**Positive**

- The EKS PR has exactly the two identities it needs, each least-privilege and
  purpose-dedicated, and can consume the role outputs directly.
- No static credentials, users, or access keys exist — the account's attack
  surface from this PR is limited to two service-assumable roles.
- Permissions are AWS-maintained; the project owns no wildcard and no inline
  policy to keep current.
- Naming, tagging, and account-ID handling stay consistent with ADR-014/-015.

**Trade-offs and follow-ups**

- **CNI permissions sit on the node role** for now; the IRSA hardening is
  deferred to the EKS PR (OIDC-dependent), as above.
- **The CNI managed policy contains broad `ec2:*NetworkInterface` actions.** That
  breadth is AWS-owned and required for pod networking, not a project choice — it
  is documented rather than removed.
- **IAM only.** No EKS, EC2, security groups, or workload exist yet; those are
  later PRs. An accidental `apply` of this PR creates only IAM roles and policy
  attachments — which are **free** (IAM has no per-resource charge).

## What This Decision Does *Not* Imply

- It does **not** provision EKS, a node group, EC2 instances, or any compute —
  only the IAM roles they will assume.
- It does **not** create any static AWS credential, IAM user, or access key, and
  does **not** store a kubeconfig.
- It does **not** grant administrative or wildcard permissions authored by this
  project; the only broad actions are inside AWS-maintained managed policies and
  are documented.
- It does **not** finalize node-level IAM hardening (CNI-via-IRSA) — that is a
  documented follow-up tied to the EKS OIDC provider.
