# Terraform — AWS Infrastructure as Code

This directory holds the **Infrastructure as Code (IaC)** for the project's AWS
platform. It is introduced in **Sprint 6, PR 1** as a *foundation*: a
professionally structured, validation-clean Terraform project that declares
**no billable AWS resources yet**. Networking, IAM, and the EKS cluster are
provisioned by later Sprint 6 PRs on top of this foundation.

Design of record: [ADR-014 — Terraform Architecture & Foundation](../docs/decisions/ADR-014-terraform-architecture.md).

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
terraform fmt -recursive -check   # formatting is canonical (CI gate in a later PR)
terraform validate                # configuration is internally consistent
```

`validate` performs **no AWS API calls** and needs no credentials — it checks
syntax, types, and references. This is the primary correctness gate for the
foundation PR.

## Planning

```bash
terraform plan
```

`plan` **does** contact AWS (to resolve the context and Availability-Zone data
sources), so valid credentials are required. As of **PR 2** a plan proposes the
**network** described below — a VPC, subnets, an internet gateway, a NAT
gateway, and route tables (18 resources with the defaults). Applying it creates
billable resources (chiefly the NAT gateway); `validate` alone does not.

> Normal pull-request CI never runs `terraform apply`. Cloud provisioning is a
> deliberate, controlled operation — see [ADR-014](../docs/decisions/ADR-014-terraform-architecture.md)
> and the Sprint 6 CI/CD boundary.

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

## What later Sprint 6 PRs will provision

| PR | Adds |
|----|------|
| **PR 1** | Terraform foundation — versions, provider, variables, outputs, naming + tagging, docs. No AWS resources. |
| **PR 2 (this PR)** | AWS network foundation — VPC, public/private subnets across AZs, routing, internet + NAT gateways, EKS subnet tags. |
| **PR 3** | Least-privilege IAM — EKS cluster role, node role, policy attachments, trust relationships. |
| **PR 4** | Managed EKS cluster + node group, required addons, connection outputs. |
| **PR 5** | Kubernetes AWS overlay wiring the existing workload to EKS (in [`k8s/`](../k8s/), not here). |
| **PR 6** | Terraform CI validation gates (`fmt`/`init`/`validate`/`plan`, lint/security scans). |
| **PR 7** | Real cloud integration test — apply → run the MLOps Job on EKS → capture evidence. |
| **PR 8** | Cost controls, teardown (`terraform destroy`), and lifecycle documentation. |

As of PR 2, `terraform apply` in this directory **does** create billable
resources (chiefly the NAT gateway). Provision deliberately and run
`terraform destroy` after evidence capture — see the cost notes above and the
teardown procedure in PR 8.
