# ADR-024: VPC CNI Identity via EKS Pod Identity — closes M-01

- **Status:** Accepted (design)
- **Date:** 2026-08-17
- **Deciders:** Asad Hanif
- **Related:** [`terraform/iam.tf`](../../terraform/iam.tf),
  [`terraform/eks.tf`](../../terraform/eks.tf),
  [`terraform/outputs.tf`](../../terraform/outputs.tf),
  [`terraform/tests/eks_cni_identity.tftest.hcl`](../../terraform/tests/eks_cni_identity.tftest.hcl),
  [`terraform/README.md`](../../terraform/README.md),
  [ADR-016 (AWS IAM Foundation)](ADR-016-aws-iam-foundation.md) — **this ADR closes
  ADR-016's deferred CNI-via-IRSA follow-up**,
  [ADR-017 (Amazon EKS Platform)](ADR-017-eks-platform.md) — **supersedes ADR-017's
  "CNI runs on the node role" decision**,
  [ADR-019 (Terraform CI Validation)](ADR-019-terraform-ci-validation.md),
  [ADR-023 (Explicit EKS Access Entries)](ADR-023-eks-access-control.md)

> **Scope note.** This ADR ratifies moving the Amazon VPC CNI's AWS permissions
> **off the worker-node IAM role** and onto a **dedicated role bound to the
> `aws-node` service account via EKS Pod Identity**, closing Sprint 6 finding
> **M-01**. It changes *which identity the CNI uses* and adds one EKS addon; it
> does **not** re-architect the cluster, network, node group, or the other IAM
> roles (ADR-015/-016/-017 otherwise stand), and it introduces no GitOps, no
> remote state, no hardcoded ARNs/account IDs, and no committed credentials.

## Context

[ADR-016](ADR-016-aws-iam-foundation.md) attached three AWS-managed policies to
the single worker-node IAM role: `AmazonEKSWorkerNodePolicy` (join the cluster),
`AmazonEC2ContainerRegistryReadOnly` (pull images), and **`AmazonEKS_CNI_Policy`**
— the pod-networking permissions (`ec2:*NetworkInterface`, `ec2:AssignPrivateIpAddresses`,
`ec2:Describe*`, …) the Amazon VPC CNI plugin (`aws-node`) uses to attach ENIs and
assign pod IPs. ADR-016 explicitly recorded moving the CNI policy to its own role
as a **deferred hardening** (it appeared to need the cluster OIDC provider, which
did not exist yet), and ADR-017 confirmed "the `vpc-cni` addon runs with the node
role's `AmazonEKS_CNI_Policy`."

That is the finding. On EKS, every pod on a node can reach the node's instance
profile through the **instance metadata service (IMDS)** unless IMDS is blocked.
So attaching `AmazonEKS_CNI_Policy` to the node role grants those ENI-manipulation
permissions not just to the CNI, but to **any container scheduled on the node** —
including the project's own workload `Job` and any future or compromised pod. The
CNI's permissions are relatively powerful (they can create/attach/detach network
interfaces), so this is a real privilege-aggregation and blast-radius problem: a
single pod compromise inherits network-plumbing rights it never needed. The Sprint
6 review flagged this as finding **M-01** (EKS CNI permissions on the node role).

The requirement for this PR: give the VPC CNI its **own** IAM role with a proper
trust relationship and **only** the CNI permissions, associate it to the CNI
service account, **remove** the now-redundant CNI permissions from the node role
while **preserving** everything else the node needs — and do it **without breaking
EKS networking**.

## Decision

Bind the VPC CNI's service account to a dedicated IAM role using **EKS Pod
Identity**, and strip the CNI policy from the node role. The credential chain
becomes: **`aws-node` ServiceAccount → EKS Pod Identity association → dedicated VPC
CNI role → `AmazonEKS_CNI_Policy`.**

1. **Dedicated VPC CNI role, CNI permissions only.** A new
   `aws_iam_role.vpc_cni` (`…-vpc-cni-role`) carries exactly one attachment,
   `AmazonEKS_CNI_Policy` — the **same** AWS-managed policy, moved verbatim from
   the node role, not re-authored. No wildcard or inline policy is written by this
   project; the breadth remains AWS-owned and AWS-maintained, now scoped to a
   single consumer.

2. **Trust scoped to Pod Identity.** The role's trust policy allows only the EKS
   Pod Identity service principal `pods.eks.amazonaws.com`, with `sts:AssumeRole`
   **and** `sts:TagSession` (Pod Identity tags the session with the pod's
   identity). No EC2, account, user, or human principal is trusted — the role is
   unusable except by a pod that EKS Pod Identity has explicitly associated.

3. **Association to the CNI service account.** An
   `aws_eks_pod_identity_association` maps `(cluster, namespace = kube-system,
   service_account = aws-node) → …-vpc-cni-role`. This is the EKS-native binding;
   the Pod Identity agent injects role credentials into the `aws-node` pods at
   runtime via the standard `AWS_CONTAINER_CREDENTIALS_FULL_URI` mechanism, so the
   CNI stops using the node instance profile automatically.

4. **The `eks-pod-identity-agent` addon.** Pod Identity requires an on-cluster
   agent (a hostNetwork DaemonSet) to serve credentials. It is added as a fourth
   EKS addon alongside `vpc-cni`/`coredns`/`kube-proxy`, with `addon_version`
   unset (default for the pinned Kubernetes version) — consistent with the other
   addons (ADR-017).

5. **Node role reduced, other components preserved.** `AmazonEKS_CNI_Policy` is
   **removed** from the node role. The node role keeps `AmazonEKSWorkerNodePolicy`
   (the kubelet joins the cluster) and `AmazonEC2ContainerRegistryReadOnly` (the
   container runtime pulls images). Both are **node-level** permissions the CNI
   change does not touch: image pulls are performed by the kubelet/runtime, not by
   the CNI, so ECR read access correctly stays on the node. No other component
   (kube-proxy, CoreDNS, the workload `Job`) depends on `AmazonEKS_CNI_Policy`.

6. **Networking cannot deadlock — ordering is explicit.** `aws-node` and the
   pod-identity-agent are **both hostNetwork** DaemonSets, so neither needs the CNI
   to obtain a pod IP: the agent starts, serves credentials to `aws-node`,
   `aws-node` initialises the CNI, and the node reaches `Ready`. Terraform
   ordering makes this deterministic and acyclic —
   **association → agent addon → node group**: the association (a control-plane API
   call needing no nodes) is created first; the agent addon depends on the
   association and reaches `ACTIVE` with zero nodes (a 0-desired DaemonSet is
   healthy), so it is installed before any node launches; and the **node group
   depends on both the association and the agent addon**, so nodes only launch once
   credentials can actually be served (the "before_compute" ordering the upstream
   EKS tooling uses). Crucially the agent addon is **not** gated on the node group
   — that reverse edge would deadlock, because the node group's `Ready`-wait itself
   needs the CNI, which needs the agent.

7. **Executable contract tests.** A new offline suite,
   [`tests/eks_cni_identity.tftest.hcl`](../../terraform/tests/eks_cni_identity.tftest.hcl),
   runs under `mock_provider "aws"` (`command = plan`, no AWS, no credentials) and
   asserts the M-01 contract: `AmazonEKS_CNI_Policy` is **not** on the node role
   (exactly two node-role attachments remain); the node role **keeps** its worker
   and ECR policies; the dedicated CNI role carries `AmazonEKS_CNI_Policy`; the CNI
   role trusts `pods.eks.amazonaws.com` (with `sts:TagSession`) and **not** EC2;
   the association targets `kube-system/aws-node`; and the `eks-pod-identity-agent`
   addon is installed. This is the regression guard, in the same no-AWS spirit as
   ADR-019/-022/-023.

## Mechanism selection: Pod Identity vs IRSA

Both **EKS Pod Identity** and **IAM Roles for Service Accounts (IRSA)** solve the
same problem — giving a Kubernetes service account an IAM role instead of using the
node instance profile. The project standardised on **Pod Identity**:

| Criterion | EKS Pod Identity (**chosen**) | IRSA |
|-----------|-------------------------------|------|
| Cluster OIDC provider | **Not required** | Required (`aws_iam_openid_connect_provider`) |
| TLS thumbprint / `tls` provider | **None** | Needs the OIDC issuer's CA thumbprint |
| Trust policy | Simple service-principal trust on `pods.eks.amazonaws.com` | Federated trust with `sub`/`aud` conditions per service account |
| Binding | Declarative association `(cluster, ns, sa) → role` | Trust-policy condition + SA annotation |
| Cross-cluster role reuse | The same role works for any cluster you associate | OIDC trust is issuer-specific |
| On-cluster component | `eks-pod-identity-agent` addon | Pod-identity webhook is built-in |
| Fit with this project | Matches the EKS-native access-entry model already adopted (ADR-023); supported at K8s 1.35 / provider `~> 5.60` | The older, OIDC-dependent path ADR-016 assumed |

Pod Identity is the current AWS-recommended mechanism, needs the least additional
machinery (no OIDC provider, no `tls` provider, no thumbprint to maintain), and is
philosophically consistent with the explicit, EKS-native access model this project
adopted for cluster access in ADR-023. The VPC CNI has supported Pod Identity since
CNI **v1.16**, far below the default CNI version shipped with the pinned Kubernetes
**1.35** control plane, so there is no version obstacle. Per the task, the two
mechanisms are **not** both added — only Pod Identity.

## Alternatives Considered

1. **Leave `AmazonEKS_CNI_Policy` on the node role (do nothing).**
   - *Rejected* — this is the finding: every pod on the node inherits ENI
     permissions via IMDS. The whole point of M-01 is to stop that.
2. **Use IRSA instead of Pod Identity.**
   - *Rejected (for this project)* — functionally equivalent for the CNI, but it
     requires provisioning the cluster OIDC provider and handling the issuer TLS
     thumbprint (a `tls` provider and periodic-thumbprint concern), for no benefit
     over Pod Identity here. See the comparison above. Recorded as a viable
     alternative, not the chosen one.
3. **Block IMDS access from pods (hop limit / `httpTokens`) instead of moving the
   policy.**
   - *Rejected as a substitute* — restricting IMDS is a complementary hardening,
     but on its own it does not give the CNI a *scoped* identity: the CNI itself
     still reads the node profile, so the node role must still hold the broad CNI
     policy. Moving the policy addresses the finding at the identity layer; IMDS
     restriction can be layered on later and does not conflict.
4. **Write a hand-crafted least-privilege CNI policy instead of the AWS-managed
   one.**
   - *Rejected* — the CNI's required permissions evolve with the plugin; AWS keeps
     `AmazonEKS_CNI_Policy` current. Re-deriving it by hand would be broader (to be
     safe), more brittle, and higher-maintenance. The goal of M-01 is *isolation*
     of the CNI permissions to a single consumer, which the dedicated role
     achieves; narrowing the policy contents is AWS's job.
5. **Gate the pod-identity-agent addon on the node group for "correct" ordering.**
   - *Rejected* — it deadlocks: the node group cannot become `Ready` until the CNI
     has credentials, which needs the agent. The agent (hostNetwork) is created
     independently of the node group precisely to avoid this; the association is
     what the node group waits on.

## Consequences

**Positive**

- **CNI permissions are isolated.** `AmazonEKS_CNI_Policy` lives on exactly one
  role, assumable only by the `aws-node` pod via Pod Identity. A compromised
  workload pod no longer inherits ENI-manipulation rights from the node profile —
  M-01 is closed at the identity layer.
- **The node role is minimal and honest.** It now holds only node-level
  permissions (join + image-pull); its policy list matches what the node actually
  does.
- **No new project-authored permissions.** The same AWS-managed policy is moved,
  not rewritten; the project still owns no wildcard or inline IAM policy.
- **Regression-proof.** The `terraform test` suite fails CI if a future change
  re-attaches the CNI policy to the node role or breaks the association.
- **Consistent with the platform's direction.** Pod Identity mirrors the
  EKS-native access-entry choice (ADR-023); the two together move the platform off
  the legacy node-profile / aws-auth patterns.

**Trade-offs and follow-ups**

- **One extra addon.** `eks-pod-identity-agent` is a small hostNetwork DaemonSet
  on every node; negligible resource cost, but it is one more managed component to
  keep in lock-step with the cluster version.
- **Bootstrap ordering is subtle.** The no-deadlock ordering (agent addon depends
  on the association; node group depends on both the association and the agent
  addon; agent **not** gated on the node group) is deliberate and documented in
  `eks.tf`; a naive "make the addon depend on the node group for consistency" edit
  would reintroduce a deadlock. The contract tests and comments guard against it.
- **Live-validation note.** The isolation is asserted offline by the contract
  suite; confirming it on a running cluster (`aws iam list-attached-role-policies`
  on both roles, `aws eks list-pod-identity-associations`, and inspecting the
  `aws-node` pod's injected credentials while networking stays healthy) requires a
  live `apply`, an operator-driven action against one's own account (per
  ADR-014/-019/-020), not something CI performs. The exact commands are in
  `terraform/README.md` § VPC CNI identity.

## What This Decision Does *Not* Imply

- It does **not** re-architect the network, cluster, node group, or the cluster/
  node IAM roles — only the CNI's identity changes and one addon is added.
- It does **not** address the remaining Sprint 6 hardenings (e.g. **M-02** KMS
  envelope encryption of secrets); those are tracked separately.
- It does **not** introduce GitOps, Terraform remote state, hardcoded account IDs
  or ARNs, an OIDC provider, or a `tls`/`kubernetes`/`helm` provider.
- It does **not** claim a production, HA, or hardened-to-completion platform — the
  cluster remains the short-lived validation environment of ADR-020.
