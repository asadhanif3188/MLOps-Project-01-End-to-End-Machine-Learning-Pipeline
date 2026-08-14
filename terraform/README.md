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
├── variables.tf              # input variables (region, project, environment, tags)
├── outputs.tf                # context outputs (region, account, naming, tags)
├── main.tf                   # locals (name prefix + common tags) + context data sources
└── terraform.tfvars.example  # copyable placeholders — NO secrets
```

There is intentionally **no `modules/` directory yet**. A single small root
module is the honest shape for a resource-free foundation; modules will be
introduced only when a later PR has a genuine, reusable boundary to extract
(e.g. network or EKS). Creating empty modules now would be structure for
appearance, which this project avoids.

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

`plan` **does** contact AWS (to resolve the account/region context data
sources), so valid credentials are required. In this foundation PR a plan
proposes **no resource changes** — the only data read is caller identity and
region. Meaningful plans (VPC, IAM, EKS) begin in later PRs.

> Normal pull-request CI never runs `terraform apply`. Cloud provisioning is a
> deliberate, controlled operation — see [ADR-014](../docs/decisions/ADR-014-terraform-architecture.md)
> and the Sprint 6 CI/CD boundary.

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
| **PR 1 (this PR)** | Terraform foundation — versions, provider, variables, outputs, naming + tagging, docs. No AWS resources. |
| **PR 2** | AWS network foundation — VPC, subnets, routing, gateways, tags. |
| **PR 3** | Least-privilege IAM — EKS cluster role, node role, policy attachments, trust relationships. |
| **PR 4** | Managed EKS cluster + node group, required addons, connection outputs. |
| **PR 5** | Kubernetes AWS overlay wiring the existing workload to EKS (in [`k8s/`](../k8s/), not here). |
| **PR 6** | Terraform CI validation gates (`fmt`/`init`/`validate`/`plan`, lint/security scans). |
| **PR 7** | Real cloud integration test — apply → run the MLOps Job on EKS → capture evidence. |
| **PR 8** | Cost controls, teardown (`terraform destroy`), and lifecycle documentation. |

Until PR 2, `terraform apply` in this directory creates nothing and incurs no
cost.
