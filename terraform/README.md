# Terraform — AWS Infrastructure as Code

This directory holds the **Infrastructure as Code (IaC)** for the project's AWS
platform. It is introduced in **Sprint 6, PR 1** as a *foundation*: a
professionally structured, validation-clean Terraform project that declares
**no billable AWS resources yet**. Networking, IAM, and the EKS cluster are
provisioned by later Sprint 6 PRs on top of this foundation.

Design of record: [ADR-014 — Terraform Architecture & Foundation](../docs/decisions/ADR-014-terraform-architecture.md).

> **The full stack is now provisioned and proven.** As of Sprint 6 PR 7, this
> configuration was `apply`-ed in the operator's own account, the MLOps `Job` ran to
> completion on the resulting EKS cluster (exit 0), and the environment was then
> **destroyed and verified clean** — see the [runtime evidence](../docs/proof/sprint-06-runtime-evidence.md).
> For the end-to-end operator runbook, the **AWS cost drivers**, and the **safe
> teardown** procedure, see [docs/cloud-operations.md](../docs/cloud-operations.md)
> and [ADR-020](../docs/decisions/ADR-020-cloud-lifecycle-cost-control.md). This
> README is the Terraform *reference*; that runbook is the *lifecycle*.

> **Separation of concerns.** Terraform owns **cloud infrastructure** (VPC, IAM,
> EKS, node capacity). Kubernetes workload configuration stays in
> [`k8s/`](../k8s/) (Kustomize). Application/ML logic stays in [`src/`](../src/).
> Terraform deliberately stops at infrastructure and does not embed Kubernetes
> manifests.

---

## Purpose

- Provide a reproducible, version-controlled definition of the AWS infrastructure.
- Establish version constraints, provider configuration, naming, and a common
  tagging strategy that every later resource inherits.
- Keep the repository **safe to publish**: no credentials, state, or secrets are
  ever committed.

## Directory structure

```text
terraform/
├── README.md                 # this file
├── versions.tf               # Terraform + AWS provider version constraints
├── providers.tf              # AWS provider config (region + default_tags)
├── variables.tf              # input variables (region, project, environment, tags, network)
├── outputs.tf                # outputs (context + network: VPC, subnets, NAT, …)
├── main.tf                   # locals (name prefix + common tags) + context data sources
├── network.tf                # VPC, subnets, IGW, NAT, route tables (Sprint 6, PR 2)
├── iam.tf                     # EKS cluster + node roles, dedicated VPC CNI role, trust, policy attachments (Sprint 6 PR 3; CNI role Sprint 7 PR 4)
├── eks.tf                     # EKS cluster, node group, core addons, access entries, VPC CNI Pod Identity (Sprint 6 PR 4; Sprint 7 PR 3–4)
├── ecr.tf                     # ECR repository + lifecycle policy (Sprint 7, PR 1 — closes H-01)
├── tests/                     # offline `terraform test` contract suite (mock_provider, no AWS)
│   ├── ecr.tftest.hcl              # asserts the ECR security + lifecycle contract (H-01)
│   ├── eks_api_security.tftest.hcl # asserts the secure-by-default EKS API posture (H-02)
│   ├── eks_access_control.tftest.hcl  # asserts explicit EKS access entries (H-03)
│   └── eks_cni_identity.tftest.hcl    # asserts VPC CNI identity isolation via Pod Identity (M-01)
└── terraform.tfvars.example  # copyable placeholders — NO secrets
```

There is intentionally **no `modules/` directory yet**. A single small root
module is the honest shape at this stage; modules will be introduced only when a
later PR has a genuine, *reusable* boundary to extract. The network is
instantiated exactly once (one VPC, one environment), so it lives in
`network.tf` in the root module rather than a `modules/network` used a single
time — extracting a module for one caller adds indirection without reuse. See
[ADR-015](../docs/decisions/ADR-015-aws-network-architecture.md).

## Authentication expectations

Terraform authenticates to AWS using the **standard AWS credential chain** — it
does **not** read credentials from any file in this repository. Provide
credentials by one of:

- environment variables — `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
  `AWS_SESSION_TOKEN`, `AWS_REGION`; or
- a named profile — `export AWS_PROFILE=<profile>` (configured via
  `aws configure`); or
- an assumed IAM role / SSO session (`aws sso login`).

Verify your identity before planning:

```bash
aws sts get-caller-identity
```

Region comes from the `aws_region` variable (default `us-east-1`), not from
hard-coded provider config.

## Initialization

```bash
cd terraform
terraform init
```

`init` downloads the pinned AWS provider and prepares the working directory.
It creates a local `.terraform/` directory (provider binaries — gitignored) and
a `.terraform.lock.hcl` dependency lock file. The lock file **is committed** so
the exact provider version and checksums are reproducible for everyone (this is
HashiCorp's recommendation and `init`'s own advice); it is intentionally not
gitignored.

## Validation

```bash
terraform fmt -recursive -check   # formatting is canonical
terraform init -backend=false     # providers only — no backend, no state, no AWS
terraform validate                # configuration is internally consistent
terraform test                    # offline contract suite (mock_provider — no AWS)
tflint --init && tflint           # language preset + AWS ruleset (.tflint.hcl)
trivy config .                    # IaC misconfiguration scan (CRITICAL/HIGH gate)
```

Every command above performs **no AWS API calls** and needs **no credentials** —
they check formatting, syntax, types, references, provider schema, contract
assertions, lint rules, and insecure configuration purely from the source.
`validate` is the primary correctness gate; `terraform test` runs the
`tests/*.tftest.hcl` suite with a **mocked AWS provider** (`command = plan`, nothing
provisioned) to pin the ECR security/lifecycle contract (H-01) **and the
secure-by-default EKS API-access posture** (H-02 — private by default, no
unrestricted `0.0.0.0/0`, public access only as a scoped opt-in); `fmt`/`init
-backend=false`/`validate`/`test`/TFLint/Trivy are the same checks CI runs on every
push and PR in the **`terraform-validate`** job, so passing them locally guarantees
that gate is green. Design of record:
[ADR-019](../docs/decisions/ADR-019-terraform-ci-validation.md) and
[docs/ci-cd.md § Job 4](../docs/ci-cd.md).

> **CI never runs `terraform plan` or `apply`.** The `terraform-validate` job holds
> no AWS credentials and no cloud identity; it validates the IaC statically and
> stops there. A real `plan` (below) is an operator step against their **own**
> account. Trivy's few suppressed findings are intentional, ADR-ratified
> validation-cluster exposures, each justified in
> [`.trivyignore`](.trivyignore).

## Planning

```bash
terraform plan
```

`plan` **does** contact AWS (to resolve the context and Availability-Zone data
sources), so valid credentials are required. A plan proposes the
**network** (VPC, subnets, internet gateway, NAT gateway, route tables — 18
resources with the defaults), the **IAM foundation** (three EKS roles — cluster,
node, and the dedicated VPC CNI role — and their four managed-policy attachments —
7 resources, incl. Sprint 7 PR 4), the **EKS platform** (one cluster, one managed
node group, three core addons, the `eks-pod-identity-agent` addon, and the VPC CNI
Pod Identity association — 7 resources), and the **container registry** (one ECR
repository + its lifecycle policy — 2 resources, Sprint 7 PR 1): **34 resources**
in total. Each configured operator identity in `cluster_access_entries` adds
**2 more** (an access entry + its policy association, Sprint 7 PR 3); the default
empty map adds none. The **EKS control
plane and worker nodes are now the dominant hourly cost**, alongside the NAT
gateway; IAM, ECR, and access entries are effectively free at this scale.
`validate` alone contacts nothing.

> CI never runs `terraform plan` **or** `apply`: `plan` needs the AWS credentials
> above (to resolve the data sources), which CI deliberately does not hold, so the
> `terraform-validate` job stops at static checks. Cloud provisioning is a
> deliberate, credentialed, operator-driven operation against your **own** account
> — see [ADR-014](../docs/decisions/ADR-014-terraform-architecture.md),
> [ADR-019](../docs/decisions/ADR-019-terraform-ci-validation.md), and the
> Sprint 6 CI/CD boundary in [docs/ci-cd.md](../docs/ci-cd.md).

## Network architecture

PR 2 provisions the minimum, EKS-ready network — a single VPC spread across
Availability Zones with a public and a private subnet per AZ. Nodes run in the
private subnets and reach the internet **outbound only** through NAT; the
workload is a batch `Job` with no inbound surface, so no public ingress is
created. Design of record:
[ADR-015](../docs/decisions/ADR-015-aws-network-architecture.md).

```text
                         Internet
                            │
                      ┌─────┴─────┐
                      │    IGW     │
                      └─────┬─────┘
            VPC 10.0.0.0/16 │
   ┌────────────────────────┼────────────────────────┐
   │  AZ a                   │            AZ b         │
   │  ┌──────────────┐   ┌───┴───┐    ┌──────────────┐ │
   │  │ public /24   │──▶│  NAT  │◀───│ public /24   │ │  kubernetes.io/role/elb
   │  │ 10.0.0.0/24  │   └───┬───┘    │ 10.0.1.0/24  │ │
   │  └──────────────┘       │        └──────────────┘ │
   │  ┌──────────────┐       │        ┌──────────────┐ │
   │  │ private /24  │───────┴───────▶│ private /24  │ │  kubernetes.io/role/internal-elb
   │  │ 10.0.8.0/24  │  (egress via   │ 10.0.9.0/24  │ │  ← EKS worker nodes
   │  │  EKS nodes   │     NAT)        │  EKS nodes   │ │
   │  └──────────────┘                └──────────────┘ │
   └───────────────────────────────────────────────────┘
```

**Subnet strategy.** One **public** and one **private** subnet per AZ. Public
subnets (`map_public_ip_on_launch = true`) host the NAT gateway and any future
public load balancers and are tagged `kubernetes.io/role/elb = 1`. Private
subnets host the **EKS worker nodes** with no public IPs and are tagged
`kubernetes.io/role/internal-elb = 1`. Per-AZ CIDRs are derived from the VPC
CIDR with `cidrsubnet(cidr, 8, …)` (a `/16` → `/24`s); public subnets take the
low indices and private subnets are offset by 8 so the ranges never overlap.
Placing nodes in private subnets is a security default (nodes are not directly
reachable from the internet) that matches the Sprint 5 workload's posture.

**AZ strategy.** AZ names are **discovered at plan time**
(`aws_availability_zones`, standard non-opt-in zones only), never hard-coded, so
the configuration is region-portable. `az_count` defaults to **2** — the EKS
minimum for the managed control plane — and is capped at 3. Changing region or
AZ count requires no edit to any resource.

**Routing rationale.** A single public route table (default route → internet
gateway) is shared by the public subnets. Each AZ gets its **own** private route
table (default route → NAT gateway); per-AZ private tables mean enabling one NAT
per AZ later (`single_nat_gateway = false`) needs no restructuring — each table
simply points at its AZ-local gateway.

**Cost considerations.** The **NAT gateway is the dominant cost** in this PR: it
bills per hour *and* per GB processed, whereas the VPC, subnets, route tables,
and internet gateway are free. Choices that keep this cheap:

- **A single shared NAT gateway** (`single_nat_gateway = true`, the default) —
  one gateway serves both AZs instead of one per AZ. This is the main
  cost/resilience trade-off and is a documented variable.
- **Two AZs, not three** (`az_count = 2`) — the EKS minimum; fewer subnets and,
  with per-AZ NAT, fewer gateways.
- **NAT can be removed entirely** (`enable_nat_gateway = false`) for a
  deliberately ultra-low-cost run, provided nodes are placed in public subnets
  instead.
- The environment is **short-lived** — provision, capture evidence, then
  `terraform destroy` (PR 8). The NAT gateway is the resource most important to
  tear down promptly to stop hourly billing.

Elastic IPs attached to a NAT gateway are free while attached; the NAT hourly
and data-processing charges are the meaningful line items.

## IAM foundation

PR 3 adds the two IAM roles a managed EKS cluster needs — a control-plane role
and a worker-node role — with their trust relationships and the AWS-managed
policy attachments EKS requires. It creates **no EKS, EC2, or application
resources, and no static credentials**. Design of record:
[ADR-016](../docs/decisions/ADR-016-aws-iam-foundation.md).

| Role | Purpose | Trusted principal | Attached AWS-managed policies |
|------|---------|-------------------|-------------------------------|
| `…-eks-cluster-role` | Identity the EKS **control plane** assumes to manage the cluster's AWS resources (cluster ENIs, cluster security group). | `eks.amazonaws.com` | `AmazonEKSClusterPolicy` |
| `…-eks-node-role` | Instance-profile identity each **EC2 worker node** assumes to join the cluster and pull images (node-level permissions only). | `ec2.amazonaws.com` | `AmazonEKSWorkerNodePolicy`, `AmazonEC2ContainerRegistryReadOnly` |
| `…-vpc-cni-role` | Dedicated identity the **Amazon VPC CNI** (`aws-node`) assumes via **EKS Pod Identity** to wire pod networking — isolated from the node role (finding **M-01**). | `pods.eks.amazonaws.com` | `AmazonEKS_CNI_Policy` |

**Least privilege.** Each role is dedicated to one purpose and its trust policy
names exactly one AWS service principal — nothing else can assume it. Permissions
come **only** from the AWS-managed policies EKS documents as required; this project
authors **no inline policy and no wildcard of its own**. The one broad grant —
the `ec2:*NetworkInterface`/`ec2:Describe*` actions the Amazon VPC CNI needs — is
inside AWS-owned `AmazonEKS_CNI_Policy`, is AWS-maintained, and is the documented
minimum for pod networking rather than a project choice. That policy now sits on
the **dedicated VPC CNI role**, not the node role (finding **M-01**), so only the
`aws-node` pod — not every workload sharing a node's instance profile — can wield
it. See **VPC CNI identity** below and
[ADR-024](../docs/decisions/ADR-024-vpc-cni-pod-identity.md).

**Intentionally not permitted.** No `AdministratorAccess` or `PowerUserAccess`;
no IAM users, groups, or access keys; no static credentials or kubeconfig.
`AmazonEKSVPCResourceController` (security-groups-for-pods, unused by the
batch-`Job` workload) and `AmazonSSMManagedInstanceCore` (interactive node
access) are deliberately omitted. The CNI-permission-on-the-node-role hardening
that [ADR-016](../docs/decisions/ADR-016-aws-iam-foundation.md) deferred is now
**done** via EKS Pod Identity (ADR-024) — see below.

**Outputs.** Role **names** are exported plainly; role **ARNs** are marked
`sensitive` because an IAM role ARN embeds the AWS account ID — consistent with
the sensitive `aws_account_id` output. Retrieve an ARN with
`terraform output -raw eks_cluster_role_arn` when wiring EKS.

**Cost.** IAM roles and policy attachments are **free** — they add no billable
resource to the plan.

## VPC CNI identity (closes M-01)

The Amazon VPC CNI (`aws-node`) needs AWS permissions to attach ENIs and assign
pod IPs. Attaching `AmazonEKS_CNI_Policy` to the **node role** — the original
setup — gave those permissions to the EC2 instance profile, so **every pod on the
node** could reach them through the instance metadata service (IMDS), not just the
CNI. That over-grant is Sprint 6 finding **M-01**. Design of record:
[ADR-024](../docs/decisions/ADR-024-vpc-cni-pod-identity.md).

**What changed.** `AmazonEKS_CNI_Policy` is **removed from the node role** and
attached to a **dedicated `…-vpc-cni-role`**, assumed **only** by the `aws-node`
service account (namespace `kube-system`) through **EKS Pod Identity**:

```text
  aws-node ServiceAccount (kube-system)
        │  EKS Pod Identity association
        ▼
  …-vpc-cni-role      ── trust: pods.eks.amazonaws.com (sts:AssumeRole + sts:TagSession)
        │
        ▼
  AmazonEKS_CNI_Policy   (the ONLY place this policy is now attached)
```

The node role keeps only what the **node itself** needs
(`AmazonEKSWorkerNodePolicy` to join, `AmazonEC2ContainerRegistryReadOnly` to pull
images); neither is a CNI permission, so removing CNI does not affect them.

**Mechanism — Pod Identity, not IRSA.** EKS Pod Identity is the current AWS-native
way to bind an IAM role to a Kubernetes service account. It needs **no cluster
OIDC provider**, no TLS-thumbprint bookkeeping, and no extra Terraform provider —
just the `eks-pod-identity-agent` addon (a hostNetwork DaemonSet) plus a small
association resource. It is consistent with the access-entry model this project
already adopted for cluster access (ADR-023). The IRSA alternative (and why it was
not chosen) is in [ADR-024](../docs/decisions/ADR-024-vpc-cni-pod-identity.md).

**Networking is not broken.** `aws-node` and the pod-identity-agent are **both
hostNetwork** DaemonSets, so neither needs the CNI to start: the agent comes up,
serves credentials to `aws-node`, `aws-node` wires the CNI, and the node becomes
Ready. Terraform ordering guarantees this without a dependency cycle — the
association (a control-plane call) is created before nodes launch, the agent addon
depends on the association, and the node group depends on the association;
crucially the agent addon is **not** gated on the node group (that would
deadlock). See the comments in [`eks.tf`](eks.tf).

**Verifying on a live cluster.** After `apply`:

```bash
# The CNI policy is NOT on the node role (expect: no AmazonEKS_CNI_Policy)
aws iam list-attached-role-policies --role-name "$(terraform output -raw eks_node_role_name)"

# It IS on the dedicated CNI role (expect: AmazonEKS_CNI_Policy)
aws iam list-attached-role-policies --role-name "$(terraform output -raw vpc_cni_role_name)"

# The aws-node service account is bound to the CNI role via Pod Identity
aws eks list-pod-identity-associations \
  --cluster-name "$(terraform output -raw eks_cluster_name)" --namespace kube-system

# aws-node pods carry the injected Pod Identity credentials (expect AWS_CONTAINER_* env)
kubectl -n kube-system get pods -l k8s-app=aws-node
kubectl -n kube-system exec <aws-node-pod> -c aws-node -- env | grep AWS_CONTAINER

# Networking is healthy: nodes Ready, aws-node Running, a test pod gets an IP
kubectl get nodes
kubectl -n kube-system get ds aws-node
```

## EKS platform

PR 4 provisions the managed Kubernetes platform: an **EKS control plane**, one
small **managed node group**, and only the **three core addons** a functioning
cluster needs. It consumes the PR 2 network and the PR 3 IAM roles and adds no
networking or IAM of its own beyond the cluster security group EKS creates
automatically. Design of record:
[ADR-017](../docs/decisions/ADR-017-eks-platform.md).

```text
        EKS control plane (managed)              Kubernetes 1.35
        role: …-eks-cluster-role                 endpoint: PRIVATE by default
        ENIs across public + private subnets     (public = scoped opt-in) · logs → CloudWatch
                    │
                    ▼
        Managed node group  …-eks-ng             role: …-eks-node-role
        2 × t3.medium (ON_DEMAND, AL2023)        private subnets only, egress via NAT
        fixed size (min = max = desired = 2)     no autoscaler, no GPU, no SSH
                    │
                    ▼
        Core addons: vpc-cni · coredns · kube-proxy
        + eks-pod-identity-agent  (delivers the VPC CNI role — M-01)
```

**Kubernetes version.** Pinned explicitly to **1.35** (`kubernetes_version`) for
reproducibility — a deliberately chosen, comfortably-supported version rather
than "whatever is newest". EKS manages the patch version; the addon versions
track the control-plane version (their `addon_version` is left to the EKS
default per Kubernetes version).

**Node sizing.** A **fixed pair of `t3.medium`** (2 vCPU / 4 GiB) on-demand nodes
on the Amazon Linux 2023 EKS AMI, 20 GiB root volume each. That is enough for the
single batch `Job` plus EKS system pods (CoreDNS ×2, and the `aws-node`/
`kube-proxy` DaemonSets per node) with room to schedule, and nothing more — no
GPUs (sprint non-goal) and no oversizing. `min = max = desired = 2` and **no
Cluster Autoscaler is installed**, so the group is effectively fixed-size; all
sizes are variables (`node_instance_types`, `node_desired_size`, …) for easy
resize. (The **PR 7 validation run used a single node** — `node_*_size = 1` via a
git-ignored `terraform.tfvars` — which comfortably hosted the one batch `Job` plus
the EKS system pods; the default of 2 leaves scheduling headroom.)

**Endpoint & security — secure by default (closes finding H-02).** The API server
is **private by default**: `cluster_endpoint_private_access` defaults **true** and
`cluster_endpoint_public_access` defaults **false**, so out of the box the control
plane is reachable only from inside the VPC and **nothing is exposed to the
internet**. Public access is an **explicit opt-in** and can never be unrestricted —
the configuration enforces this itself, not just in docs:

- `cluster_endpoint_public_access_cidrs` defaults to `[]` and its validation
  **rejects any `/0`** (including `0.0.0.0/0`);
- a **precondition** rejects enabling public access with an **empty** CIDR list
  (EKS would otherwise treat empty as `0.0.0.0/0`);
- a **precondition** rejects **disabling both** endpoints (an unreachable API).

To validate with `kubectl` from a workstation, opt in by setting
`cluster_endpoint_public_access = true` **and** a scoped
`cluster_endpoint_public_access_cidrs = ["<your-ip>/32"]`. **Operational note:** a
private-only endpoint is **not reachable from a workstation outside the VPC** —
reaching it needs a bastion/VPN, an SSM session, or a CI/ops runner inside the VPC;
the scoped public opt-in is the documented path for the short-lived validation run.
These rules are pinned by the offline
[`tests/eks_api_security.tftest.hcl`](tests/eks_api_security.tftest.hcl) contract
suite. Design of record: [ADR-022](../docs/decisions/ADR-022-eks-secure-api-access.md).

Cluster access uses **explicit EKS access entries** — see
[§ EKS access management](#eks-access-management) below (closes finding H-03). The
creating principal gets **no** implicit cluster-admin; access is declared, scoped,
and independent of who ran `apply`. Control-plane logging ships the
**security-relevant** types (`api`, `audit`, `authenticator`) to CloudWatch; set
`cluster_enabled_log_types = []` to remove that small cost. Nodes have **no public
IP and no SSH remote-access** configured.

**Intentionally not provisioned.** No GPU nodes, no Cluster Autoscaler /
autoscaling, no service mesh, no ingress controller, no observability stack
(Prometheus/Grafana/logging agents), no optional addons (EBS/EFS CSI, ALB
controller), no additional AWS services, and **no application/workload
resources** — the Kubernetes `Job` stays in Kustomize
([`k8s/`](../k8s/)), wired to EKS in a later PR. Envelope encryption of secrets
with a customer-managed KMS key is a documented follow-up, not included here.

**Outputs.** All EKS outputs are **non-sensitive** connection/inspection details
— `eks_cluster_name`, `eks_cluster_endpoint`, `eks_cluster_version`,
`eks_cluster_security_group_id`, `eks_cluster_oidc_issuer_url`,
`eks_node_group_name`, and a ready-to-run `configure_kubectl` command. No
kubeconfig, token, or certificate is emitted; operators fetch short-lived
credentials with:

```bash
aws eks update-kubeconfig --region <region> --name <eks_cluster_name>
```

**Cost.** The EKS **control plane bills at a flat hourly rate** and the
**on-demand node(s) bill per hour**; together with the NAT gateway these are the
meaningful line items. The cluster is **short-lived by design** — provision,
verify, capture evidence, then `terraform destroy`. Cheaper knobs:
`node_capacity_type = "SPOT"`, fewer/smaller nodes, or `node_desired_size = 1`
(the PR 7 run used 1). The ranked cost drivers and the destroy-then-verify teardown
are in [docs/cloud-operations.md](../docs/cloud-operations.md#4-aws-cost-drivers).

## EKS access management

PR 3 replaces the old *"whoever ran `apply` becomes cluster-admin"* bootstrap with
**explicit EKS access entries** (closes Sprint 6 finding **H-03**). Access is now a
declared chain — the creating principal receives nothing implicitly:

```text
AWS IAM identity (role/user)
        │  cluster_access_entries[<key>].principal_arn
        ▼
EKS access entry            aws_eks_access_entry
        │
        ▼
scoped EKS access policy    aws_eks_access_policy_association
        │  policy (View / Edit / Admin / …) + access_scope (cluster | namespace)
        ▼
Kubernetes RBAC permissions
```

Design of record: [ADR-023](../docs/decisions/ADR-023-eks-access-control.md).

**Secure defaults (enforced, not just documented).**

- `cluster_authentication_mode` defaults to **`API`** — access entries are the
  *only* path; there is no `aws-auth` ConfigMap backdoor. `API_AND_CONFIG_MAP` is
  allowed for a migration; `CONFIG_MAP` (aws-auth only) is **rejected** by
  validation because it bypasses access entries.
- `cluster_bootstrap_creator_admin_permissions` defaults to **`false`** and is a
  **tripwire**: its validation **rejects `true`**, so the old insecure
  creator-admin bootstrap cannot be reintroduced — a `plan`/`apply`/`terraform
  test` fails if anyone sets it.
- `cluster_access_entries` defaults to **`{}`** — no ARNs are committed.

**How access is granted.** Add an entry to `cluster_access_entries` in your
**git-ignored `terraform.tfvars`** (never in the repo — see
[`terraform.tfvars.example`](terraform.tfvars.example)):

```hcl
cluster_access_entries = {
  operator = {
    principal_arn = "arn:aws:iam::<account-id>:role/<operator-role>"
    policy        = "AmazonEKSAdminPolicy"   # scoped admin (NOT cluster-admin)
    access_scope  = "cluster"                # or "namespace" + namespaces = [...]
  }
}
```

`apply` then creates an `aws_eks_access_entry` (registers the principal) and an
`aws_eks_access_policy_association` (grants the scoped policy). Prefer an **IAM
role** as the principal over a user.

**What permissions are granted (choose the narrowest that works).**

| `policy` | Grants | Use for |
|---|---|---|
| `AmazonEKSViewPolicy` | read-only | auditors, CI observers |
| `AmazonEKSEditPolicy` | create/update most namespaced resources | deploy-only automation |
| `AmazonEKSAdminPolicy` **(default)** | scoped admin across namespaces — **not** `system:masters` | the operator who provisions & runs the validation workload |
| `AmazonEKSAdminViewPolicy` | view incl. sensitive resources | deeper read-only |
| `AmazonEKSClusterAdminPolicy` | full cluster-admin (`system:masters`) | **avoid** — last resort only |

`access_scope = "namespace"` (with a non-empty `namespaces` list) narrows an entry
to specific namespaces; `"cluster"` is the default because a single operator that
stands up and tears down the whole validation cluster needs to create namespaces
and manage cluster-scoped resources. **Do not reach for
`AmazonEKSClusterAdminPolicy` for convenience** — the default `AmazonEKSAdminPolicy`
covers operating the batch `Job`, and the policy set is restricted by validation to
the AWS-managed EKS access policies (an arbitrary broad IAM policy cannot be
associated).

**How access is revoked.** Remove the entry from `cluster_access_entries` and
`apply` — Terraform destroys both the access entry and its policy association, and
the principal immediately loses cluster access. To change a grant, edit the entry's
`policy`/`access_scope` and `apply`. (Out-of-band, `aws eks
disassociate-access-policy` / `delete-access-entry` do the same, but the
Terraform-managed path keeps state and config in sync.)

**Does Terraform / CI need an admin entry?** **No.** This root module manages only
AWS-API resources — there is **no `kubernetes`/`helm` provider**, so Terraform never
calls the Kubernetes API and needs no cluster access entry. CI runs `fmt`/`validate`/
`test` offline and never touches the cluster. Access entries exist purely for the
humans/automation that run `kubectl`.

**Not creating an unusable cluster.** An empty `cluster_access_entries` is **safe,
not broken**: EKS still auto-creates the managed node group's own access entry, so
nodes join and addons run; only *human* `kubectl` access needs an explicit entry.
Add your operator principal before validating. This mirrors the empty public-CIDR
default of [ADR-022](../docs/decisions/ADR-022-eks-secure-api-access.md).

These rules are pinned by the offline
[`tests/eks_access_control.tftest.hcl`](tests/eks_access_control.tftest.hcl)
contract suite: no creator-admin by default, `API` auth by default, the bootstrap
tripwire, rejected `CONFIG_MAP`/invalid-ARN/unknown-policy/empty-namespace-scope
inputs, and clean plans for scoped cluster- and namespace-level entries.

## Container registry (ECR)

`ecr.tf` provisions the private **Amazon ECR** repository that stores the workload
image the EKS nodes pull at run time (Sprint 7, PR 1 — closes Sprint 6 finding
**H-01**). Through Sprint 6 this repository was created and destroyed **out-of-band**
with `aws ecr create-repository` / `aws ecr delete-repository --force`, leaving one
live AWS resource outside Terraform state. It is now managed like everything else —
provisioned, tagged, versioned, and torn down by `terraform apply`/`destroy`.

- **Name.** Defaults to `project_name` (`mlops-pipeline`), not the environment-scoped
  `name_prefix`: the registry is a per-project artifact store and the name must match
  the image reference committed in [`k8s/overlays/aws`](../k8s/overlays/aws). Override
  with `ecr_repository_name` if needed.
- **Immutable tags** (`image_tag_mutability = "IMMUTABLE"`) — a version tag such as
  `1.3.1` can never be repointed, so a deployed digest is reproducible and the
  overlay's static image-pinning contract holds. This matches the "explicit,
  immutable version, never `:latest`" convention the AWS overlay already relies on.
- **Scan on push** (`scan_on_push = true`) — image vulnerability scanning stays on; it
  is a security feature, not disabled to simplify.
- **Private** — no repository policy granting public or cross-account access is
  authored, so the repository is reachable only by the account's own principals (the
  node role's `AmazonEC2ContainerRegistryReadOnly` from ADR-016). ECR is **never
  public**.
- **Encrypted at rest** with the AWS-managed key (`AES256`, ECR's default). A
  customer-managed **KMS CMK** is the documented hardening follow-up (tracked with the
  EKS-secrets KMS work), deliberately out of this PR's H-01 scope.
- **Lifecycle policy** — retains the most recent `ecr_max_image_count` images
  (default **10**) and expires older ones, so registry storage cannot grow unbounded
  across repeated validation pushes.
- **`force_delete = true`** — `terraform destroy` removes the repository (and any
  images it still holds) in the same pass, which is what replaces the old manual
  `aws ecr delete-repository --force` teardown step for this ephemeral environment.
- **Outputs.** `ecr_repository_url` and `ecr_repository_arn` are marked **`sensitive`**
  (they embed the account ID, treated as sensitive project-wide); read them with
  `terraform output -raw ecr_repository_url` when pushing the image or pointing the
  Kustomize overlay at the registry. `ecr_repository_name` is non-sensitive.

Design of record: [ADR-021](../docs/decisions/ADR-021-terraform-managed-ecr.md). The
contract is pinned by the offline `terraform test` suite
([`tests/ecr.tftest.hcl`](tests/ecr.tftest.hcl)).

## State handling

- **Local state for now.** This first implementation uses Terraform's default
  **local backend**: state is written to `terraform.tfstate` in this directory
  during a controlled, single-operator workflow. That file is **gitignored and
  must never be committed** — it can contain resolved values (account ID,
  resource attributes) and, for some resources, sensitive material.
- **Why local.** For a portfolio-scoped, short-lived, single-operator
  environment, a remote backend would add AWS resources (an S3 bucket + lock
  table) whose only purpose is the portfolio itself. That is deferred until it
  materially improves the proof.
- **What changes for a team / production.** A remote backend with locking,
  encryption, and restricted access:

  ```text
  S3 (versioned, encrypted) state
        + DynamoDB (or S3 native) state locking
        + KMS encryption
        + least-privilege IAM on the state bucket
  ```

  Migrating is a `backend` block plus `terraform init -migrate-state`; the
  configuration in this directory is otherwise unaffected.

## Security rules

- **Never commit** AWS access keys, secret keys, session tokens, account IDs,
  `terraform.tfvars`, `*.tfstate`/`*.tfstate.*`, `.terraform/`, crash logs, or a
  kubeconfig with embedded credentials. The repository `.gitignore` enforces all
  of these.
- **No credentials in variables.** `*.tfvars` files carry non-secret
  configuration only. Credentials come from the AWS credential chain above.
- `terraform.tfvars.example` contains **placeholders only** and is safe to commit.
- The `aws_account_id` output is marked **`sensitive`** so it is not printed to
  logs; retrieve it explicitly with `terraform output -raw aws_account_id` when
  needed.
- Run a secret scan before pushing and inspect the diff for any of the above.

## What each PR provisions

**Sprint 6 — AWS platform foundation**

| PR | Adds |
|----|------|
| **PR 1** | Terraform foundation — versions, provider, variables, outputs, naming + tagging, docs. No AWS resources. |
| **PR 2** | AWS network foundation — VPC, public/private subnets across AZs, routing, internet + NAT gateways, EKS subnet tags. |
| **PR 3** | Least-privilege IAM — EKS cluster role, node role, policy attachments, trust relationships. |
| **PR 4** | Managed EKS cluster + node group, three core addons, non-sensitive connection outputs. |
| **PR 5** | Kubernetes AWS overlay wiring the existing workload to EKS (in [`k8s/`](../k8s/), not here). |
| **PR 6** | Terraform CI validation gates — `fmt`/`init -backend=false`/`validate`, TFLint, Trivy IaC scan. **No** `plan`/`apply`, no AWS credentials in CI (in [`ci.yml`](../.github/workflows/ci.yml); see [ADR-019](../docs/decisions/ADR-019-terraform-ci-validation.md)). |
| **PR 7** ✅ | Real cloud integration test — apply → run the MLOps Job on EKS → capture evidence. **Executed 2026-08-15** ([runtime evidence](../docs/proof/sprint-06-runtime-evidence.md)); 29 resources applied, Job `Complete` (exit 0), then destroyed and verified clean. |
| **PR 8** | Cost controls, teardown, and lifecycle documentation — the [cloud-operations runbook](../docs/cloud-operations.md), ranked [cost drivers](../docs/cloud-operations.md#4-aws-cost-drivers), the [safe-teardown](../docs/cloud-operations.md#5-safe-teardown) sequence, and [ADR-020](../docs/decisions/ADR-020-cloud-lifecycle-cost-control.md). No new infrastructure. |

**Sprint 7 — Cloud-native hardening**

| PR | Adds |
|----|------|
| **PR 1** | Terraform-managed **ECR** — the private repository + lifecycle policy that were previously created out-of-band, closing finding **H-01**. Immutable tags, scan-on-push, encrypted, retention-capped, `force_delete` for clean teardown; sensitive URL/ARN outputs; offline `terraform test` contract suite. See [ADR-021](../docs/decisions/ADR-021-terraform-managed-ecr.md). |
| **PR 2** | **Secure-by-default EKS API access**, closing finding **H-02**. The control-plane endpoint is **private by default** (public off, CIDR list empty); public access is a scoped, explicit opt-in that **can never be `0.0.0.0/0`** — enforced by variable validation and cluster preconditions, and pinned by a new offline `terraform test` suite. The two obsolete Trivy suppressions for the old open endpoint are **removed**, not re-justified. See [ADR-022](../docs/decisions/ADR-022-eks-secure-api-access.md). |
| **PR 3** | **Explicit EKS access entries**, closing finding **H-03**. Replaces the automatic *"cluster creator becomes cluster-admin"* bootstrap with declared, scoped access entries (`API` auth by default, creator-admin `false` and rejected by a tripwire validation, scoped `AmazonEKSAdminPolicy` default). Identities come from a git-ignored `terraform.tfvars`; pinned by an offline `terraform test` suite. See [ADR-023](../docs/decisions/ADR-023-eks-access-control.md). |
| **PR 4 (this PR)** | **VPC CNI identity isolation**, closing finding **M-01**. Moves `AmazonEKS_CNI_Policy` off the worker-node instance profile onto a **dedicated VPC CNI role** assumed only by the `aws-node` service account via **EKS Pod Identity** (`eks-pod-identity-agent` addon + association); the node role keeps only its own (join + ECR-pull) permissions. Pinned by a new offline `terraform test` suite. See [ADR-024](../docs/decisions/ADR-024-vpc-cni-pod-identity.md). |

As of PR 4, `terraform apply` in this directory creates a **billable, running
EKS platform** — the control plane and the on-demand node(s) bill hourly, on top
of the NAT gateway (PR 3's IAM roles remain free). Provision deliberately and run
`terraform destroy` promptly after evidence capture — the full runbook, the ranked
cost drivers, and the **destroy-then-verify** teardown procedure are in
[docs/cloud-operations.md](../docs/cloud-operations.md)
([ADR-020](../docs/decisions/ADR-020-cloud-lifecycle-cost-control.md)).
