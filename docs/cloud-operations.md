# Cloud Operations — Runbook, Cost, and Teardown

The operator's runbook for the **AWS/EKS cloud lifecycle** of this project: how to
provision the Terraform-defined platform, run the existing MLOps `Job` on it,
capture evidence, and — the point of this document as much as any other — **tear it
all down and verify nothing is left billing**. It also documents the **AWS cost
drivers** and **why the environment is deliberately small**.

> **Scope — an ephemeral validation environment, honestly bounded.** Everything
> here provisions a **short-lived, single-operator validation environment**, not a
> production platform. It is `provision → prove → destroy`, run from **one
> operator's own AWS account** with the standard credential chain. There is **no**
> production topology (single node, single NAT, two AZs), **no** high availability,
> **no** GitOps reconciler, **no** production observability stack, **no**
> multi-region or disaster-recovery posture, and **no** always-on service — those
> are roadmap items ([roadmap.md](roadmap.md) v5–v6), not capabilities this
> repository has. See [§7 Limitations](#7-limitations). The real run this runbook
> describes was executed once and torn down; the current full-platform evidence is in
> [Sprint 7 — Runtime Evidence](proof/sprint-07-runtime-evidence.md) (the earlier
> Sprint 6 EKS run is in [Sprint 6 — Runtime Evidence](proof/sprint-06-runtime-evidence.md)).

**Design of record:**
[ADR-020 — Cloud Environment Lifecycle & Cost Control](decisions/ADR-020-cloud-lifecycle-cost-control.md),
building on [ADR-014](decisions/ADR-014-terraform-architecture.md) (Terraform),
[ADR-015](decisions/ADR-015-aws-network-architecture.md) (network),
[ADR-016](decisions/ADR-016-aws-iam-foundation.md) (IAM),
[ADR-017](decisions/ADR-017-eks-platform.md) (EKS), and
[ADR-018](decisions/ADR-018-aws-eks-deployment-overlay.md) (AWS overlay). For the
Terraform reference see [`terraform/README.md`](../terraform/README.md); for the
in-cluster day-2 operations see [Kubernetes Operations](kubernetes-operations.md).

---

## 1. Operational model — an ephemeral environment

The unit of work is **the whole environment**, not a long-running service. An
operator does not "keep it up"; they stand it up, prove one thing on it, capture
the evidence, and destroy it:

```text
  prerequisites ─▶ authenticate ─▶ init ─▶ plan ─▶ apply ─▶ verify EKS
        ▲                                                         │
        │                                                   configure kubectl
        │                                                         │
        │                                                    run the Job
        │                                                         │
        │                                                   capture evidence
        │                                                         │
   (nothing left)  ◀── verify cleanup ◀── terraform destroy ◀─────┘
```

The most important step is the **last** one. Every billable resource — the EKS
control plane, the EC2 node, the NAT gateway — bills **per hour it exists**, so the
environment's lifetime is the cost lever, not its size. Provision deliberately,
work quickly, and `terraform destroy` promptly (see [§5](#5-safe-teardown)).

---

## 2. Prerequisites

| Tool | Purpose | Notes |
|---|---|---|
| **Terraform** ≥ 1.5 | provision/destroy the AWS platform | version pinned in [`terraform/versions.tf`](../terraform/versions.tf); provider locked in [`terraform/.terraform.lock.hcl`](../terraform/.terraform.lock.hcl) |
| **AWS CLI v2** | identity, kubeconfig, ECR login | `aws sts get-caller-identity` must resolve to **your own** account |
| **kubectl** | drive the cluster | client within one minor of the cluster (Kubernetes **1.35**) |
| **Docker** | build + push the workload image | build `--platform linux/amd64` to match the AL2023 x86_64 node |
| **An AWS account you own** | pays for the environment | **not** a shared/client account; you are spending real money for the environment's lifetime |
| The git-ignored **dataset** and (optional) **`.env`** | runtime inputs | dataset uploaded out-of-band to the S3 dataset bucket (retrieved at runtime by the `fetch-dataset` init container via Pod Identity — [ADR-027](decisions/ADR-027-s3-dataset-runtime-retrieval.md)); `.env` for local, non-cluster runs |

This is a **credentialed, operator-driven** procedure. **CI never runs any of it** —
the `terraform-validate` job holds no AWS credentials and stops at static checks
([ADR-019](decisions/ADR-019-terraform-ci-validation.md),
[SECURITY.md](../SECURITY.md)).

---

## 3. The runbook

All commands run from the repository root unless a `cd`/`-chdir` says otherwise.
Region defaults to `us-east-1` (the `aws_region` variable).

### 3.1 Authenticate to AWS

Terraform and the AWS CLI use the **standard AWS credential chain** — no credential
is ever read from this repository. Provide credentials by environment variables, a
named profile, or an SSO/assumed-role session, then **verify the identity is your
own account** before doing anything billable:

```bash
aws sts get-caller-identity      # confirm Account + ARN are YOURS
```

> **Guardrail.** If this does not resolve to the account you intend to pay with,
> stop and fix your credentials — do **not** provision. Whose account the
> credentials belong to matters more than whether credentials merely exist.

### 3.2 `terraform init`

```bash
terraform -chdir=terraform init
```

Downloads the pinned AWS provider (recorded in the committed
`.terraform.lock.hcl`) and prepares the local working directory. Uses the default
**local backend** — state is written to `terraform/terraform.tfstate`, which is
**git-ignored and must never be committed** (see [§6](#6-security)).

### 3.3 `terraform plan`

Set your own values in `terraform/terraform.tfvars` first (copy
[`terraform.tfvars.example`](../terraform/terraform.tfvars.example)). The EKS API
server is **private by default** (finding H-02); to drive it with `kubectl` from
your workstation, **opt into public access scoped to your own IP** and keep the
node group small:

```hcl
# terraform/terraform.tfvars  (git-ignored — never committed)
cluster_endpoint_public_access       = true               # opt in (default is false / private-only)
cluster_endpoint_public_access_cidrs = ["<YOUR_IP>/32"]   # REQUIRED when opting in — never 0.0.0.0/0
node_desired_size = 1                                      # 1 node is enough for the Job
node_min_size     = 1
node_max_size     = 1
```

> Terraform **rejects** `0.0.0.0/0` (any `/0`) and rejects enabling public access
> with an empty CIDR list, so you cannot accidentally stand up an open endpoint. If
> you prefer to keep the endpoint fully private, leave these unset and reach the API
> from **inside the VPC** (bastion/VPN/SSM/in-VPC runner) instead — see
> [§7 Limitations](#7-limitations) and [ADR-022](decisions/ADR-022-eks-secure-api-access.md).

```bash
terraform -chdir=terraform plan -out=tfplan
```

`plan` **contacts AWS** (to resolve caller identity and AZ data sources), so valid
credentials are required. With the defaults it proposes **~63 resources** across
network, IAM (six roles: cluster, node, VPC-CNI, EBS-CSI, MLflow-S3, dataset-reader),
EKS (control plane + node group + five addons including `eks-pod-identity-agent` and
`aws-ebs-csi-driver`), **two** ECR repositories (`mlops-pipeline` + `mlflow-server`)
with lifecycle policies, **three** customer-managed KMS keys (EKS Secrets, dataset,
MLflow artifacts), and **two** private SSE-KMS S3 buckets (dataset + MLflow artifacts)
with their Pod Identity associations. Review the plan; the EKS control plane, the
node, and the NAT gateway are the resources that will start billing on apply.

### 3.4 `terraform apply`

```bash
terraform -chdir=terraform apply tfplan
```

Creates the environment (~10–15 min, dominated by EKS control-plane provisioning).
**Billing starts now** — start a mental clock; the sooner you destroy, the cheaper
the run. Expect `Apply complete! Resources: 63 added, 0 changed, 0 destroyed.` (the
figure the Sprint 7 full-platform run recorded; an additional `apply` of a couple of
resources enables the documented operator access path — see
[Sprint 7 runtime evidence § 1](proof/sprint-07-runtime-evidence.md)). Earlier
snapshots recorded fewer resources: the Sprint 6 integration run predated the Sprint 7
hardening PRs.

### 3.5 Verify EKS

```bash
aws eks describe-cluster --name "$(terraform -chdir=terraform output -raw eks_cluster_name)" \
  --query cluster.status --output text          # -> ACTIVE
```

Expect the cluster **ACTIVE** and control plane at **v1.35.x**. Outputs expose only
non-sensitive connection details (name, endpoint, version, security-group id, OIDC
issuer, node-group name, and a ready-to-run `configure_kubectl`); **no kubeconfig,
token, or certificate is emitted**.

### 3.6 Configure `kubectl`

```bash
aws eks update-kubeconfig --region us-east-1 \
  --name "$(terraform -chdir=terraform output -raw eks_cluster_name)"
kubectl get nodes -o wide                        # -> 1 node Ready, private subnet, AL2023
```

`update-kubeconfig` fetches **short-lived** credentials via the AWS credential
chain; it writes a context, not a static secret. The API is reachable only from the
scoped operator CIDR you opted into in §3.3 — the endpoint is private by default and
never open to `0.0.0.0/0`. (If you kept the endpoint fully private, run this and
`kubectl` from inside the VPC instead.)

### 3.7 Publish the workload image to ECR

Nodes run in **private subnets** and cannot see a locally built image, so the image
is pulled from a registry. The node role carries
`AmazonEC2ContainerRegistryReadOnly`, so the kubelet authenticates the pull with the
node's instance role — **no pod credential or IRSA**.

The **ECR repository is created by `terraform apply`** above — it is
Terraform-managed (`terraform/ecr.tf`, [ADR-021](decisions/ADR-021-terraform-managed-ecr.md)),
so there is **no manual `aws ecr create-repository` step**. Read its URL from
Terraform (the account ID lives in state, not in git):

```bash
region=us-east-1
repo=$(terraform -chdir=terraform output -raw ecr_repository_url)   # <account>.dkr.ecr.<region>.amazonaws.com/mlops-pipeline
aws ecr get-login-password --region "$region" \
  | docker login --username AWS --password-stdin "${repo%%/*}"

docker build --platform linux/amd64 --provenance=false --sbom=false \
  -t "$repo:1.6.0" .
docker push "$repo:1.6.0"
```

The tag is an explicit, immutable version (`1.6.0`), never `:latest` — and the
repository enforces **immutable tags**, so that version can never be repointed. (The
MLflow server image is built and pushed to its own ECR repository the same way — see
[MLflow Platform](mlflow-platform.md).)

The AWS overlay stays **account-neutral in git**: the committed image, dataset, and
MLflow-artifact values are all `000000000000` placeholders. Rather than editing —
and later reverting — those tracked files, render a concrete manifest from the live
Terraform outputs with
[`scripts/render-cloud-manifests.sh`](../scripts/render-cloud-manifests.sh). It copies
the overlay to a temp directory, substitutes the ECR registry, the dataset and
MLflow-artifact S3 buckets, and the region from `terraform output`, runs
`kustomize build`, and refuses to emit if any placeholder survived. **No tracked file
is touched**, so there is nothing to `git checkout --` on teardown.

### 3.8 Run the workload

Supply the runtime dataset out-of-band by uploading it to the Terraform-provisioned
S3 bucket (never baked into the image, never committed); the `fetch-dataset` init
container retrieves it at runtime via EKS Pod Identity (Sprint 7 PR 8 —
[ADR-027](decisions/ADR-027-s3-dataset-runtime-retrieval.md)). Then render the overlay
and apply it:

```bash
aws s3 cp data/raw/data.csv "$(terraform -chdir=terraform output -raw dataset_s3_uri)" \
  --sse aws:kms --sse-kms-key-id "$(terraform -chdir=terraform output -raw dataset_kms_key_arn)"

# Render the account-neutral overlay into a deployable manifest from Terraform
# outputs (no tracked file is edited). Inspect it, then apply:
scripts/render-cloud-manifests.sh -o /tmp/aws-manifests.yaml
kubectl apply -f /tmp/aws-manifests.yaml        # Namespace, SA, ConfigMaps, MLflow, Job
#   …or render-and-apply in one step:  scripts/render-cloud-manifests.sh --apply
kubectl -n mlops wait --for=condition=complete --timeout=300s job/mlops-pipeline
```

(If you built the workload image under a non-default tag, pass it through:
`IMAGE_TAG=<tag> scripts/render-cloud-manifests.sh …`.)

The overlay also brings up the **in-cluster MLflow platform** (the tracking server,
its PostgreSQL metadata backend, and — on AWS — the Terraform-provisioned S3 artifact
bucket reached via EKS Pod Identity). Experiment tracking is **self-hosted**: the
pipeline logs to the `mlflow` Service over HTTP and needs **no MLflow credentials and
no DagsHub** ([ADR-026](decisions/ADR-026-in-cluster-mlflow-platform.md),
[MLflow Platform](mlflow-platform.md)). The only out-of-band Secret the platform needs
is the Postgres DB credential:

```bash
kubectl create secret generic mlflow-db-credentials -n mlops \
  --from-literal=POSTGRES_USER=mlflow \
  --from-literal=POSTGRES_PASSWORD="$(openssl rand -base64 24 | tr -d '/+=')"
```

Do **not** create an `mlflow-s3-credentials` Secret on AWS — the MLflow server draws
short-lived, pod-scoped S3 credentials from EKS Pod Identity. No security field
changes.

### 3.9 Capture evidence

Read the terminal state directly from the cluster — Job condition, pod exit code,
the enforced security context, and the stage logs:

```bash
kubectl -n mlops get job/mlops-pipeline -o jsonpath='{.status}{"\n"}'
pod=$(kubectl -n mlops get pods -o jsonpath='{.items[0].metadata.name}')
kubectl -n mlops get pod "$pod" -o jsonpath=\
'phase={.status.phase} exit={.status.containerStatuses[0].state.terminated.exitCode} qos={.status.qosClass}{"\n"}'
kubectl -n mlops logs job/mlops-pipeline          # preprocess -> split -> train -> evaluate
```

In the real run: **Job `Complete`, pod `Succeeded`, exit 0**, all four stages,
in-cluster MLflow runs logged (metadata in PostgreSQL, artifacts in SSE-KMS S3), and
security controls verified live. See the
[Sprint 7 runtime evidence](proof/sprint-07-runtime-evidence.md). **Redact** account
IDs, the operator IP, and anything credential-shaped from anything you keep.

### 3.10 Destroy

Proceed straight to [§5 Safe teardown](#5-safe-teardown) — it is the mandatory last
step, not an optional one.

---

## 4. AWS cost drivers

What actually costs money here, largest first. **Time alive is the dominant lever:**
every row marked "per hour" bills for the environment's whole lifetime regardless of
whether the Job is running, so a run measured in **hours** (provision → prove →
destroy) is the difference between negligible and not. Figures below are
**order-of-magnitude, `us-east-1`, and will drift** — always confirm against
[current AWS pricing](https://aws.amazon.com/pricing/); they are here to rank the
drivers, not to quote a bill.

| Driver | Billing shape | Relative weight | Notes |
|---|---|---|---|
| **EKS control plane** | flat **per-cluster-hour** (~$0.10/hr order-of-magnitude) | **High** — unavoidable while the cluster exists | One charge per cluster regardless of node count; starts at `apply`, stops at `destroy`. |
| **EC2 worker node(s)** | **per-instance-hour** (`t3.medium`, on-demand) | **High** — scales with node count | The run used **1** node; the default is 2. `SPOT` capacity or a smaller type reduces this. |
| **NAT gateway** | **per-hour** *plus* **per-GB processed** | **Medium–High** | One shared NAT (not one per AZ). The image pull from ECR and the S3 dataset/artifact traffic flow through it. Removable entirely (`enable_nat_gateway=false`) if nodes sit in public subnets. |
| **EBS volumes** | per-GB-month (gp3), prorated hourly | **Low** | 20 GiB root volume per node. |
| **Elastic IP** | free **while attached** to the running NAT | **~Zero** | Billable only if left allocated and unattached — which `destroy` prevents. |
| **CloudWatch Logs** | ingestion + storage | **Low** | Only `api`/`audit`/`authenticator` control-plane logs; set `cluster_enabled_log_types=[]` to drop. |
| **ECR storage** | per-GB-month | **Low** | One image (~hundreds of MB); Terraform-managed, so removed by `terraform destroy`. A lifecycle policy also caps retained images ([ADR-021](decisions/ADR-021-terraform-managed-ecr.md)). |
| **Data transfer / requests** | per-GB / per-request | **Low** | Small dataset, short run. |

**IAM roles, VPC, subnets, route tables, internet gateway, security groups, and the
three EKS addons are free** — they add no hourly charge.

### Runtime duration (the actual cost driver)

The **pipeline itself ran in 52 seconds**; the **environment** lived long enough to
provision, verify, run, capture evidence, and destroy — on the order of **tens of
minutes to a couple of hours**, not days. Because the expensive resources bill per
hour, cost is governed by *how long the environment exists*, which is why teardown
is treated as the primary operation. Leaving the cluster up overnight would cost
far more than the run itself, for **zero** additional proof.

### Why the environment is intentionally small

Every sizing choice is deliberately the **minimum that still proves the claim**, not
a production shape:

- **1 (or 2) `t3.medium` on-demand node(s)** — enough for the single batch `Job`
  plus EKS system pods, nothing more. No GPUs, no autoscaler
  ([ADR-017](decisions/ADR-017-eks-platform.md)).
- **A single shared NAT gateway**, not one per AZ — the main cost/resilience
  trade-off, taken toward cost for a validation run
  ([ADR-015](decisions/ADR-015-aws-network-architecture.md)).
- **Two AZs** — the EKS control-plane minimum, not a resilience target.
- **Short-lived** — provisioned to capture evidence, then destroyed; not run
  continuously ([ADR-020](decisions/ADR-020-cloud-lifecycle-cost-control.md)).
- **Local Terraform state**, no remote backend — a remote backend would add
  billable AWS resources (S3 + lock table) whose only purpose is the portfolio
  itself ([ADR-014](decisions/ADR-014-terraform-architecture.md)).

This is a portfolio proof that the pipeline runs on managed cloud infrastructure —
**not** a claim of production-grade capacity, resilience, or cost-optimization at
scale. The small size is the honest shape of that claim.

---

## 5. Safe teardown

Teardown is the mandatory final step. The sequence is **destroy, then verify gone** —
never assume `destroy` succeeded from the fact that you ran it.

### 5.1 The sequence

```bash
# 1. Remove the Kubernetes workload (frees the ECR pull / any LB before infra goes).
#    Delete via the same rendered manifest you applied (or `kubectl delete -k` if you
#    still have it — but the render flow leaves no tracked file to revert).
kubectl delete -f /tmp/aws-manifests.yaml   # Job + mlops namespace (cascades to ConfigMaps + pods)

# 2. Destroy ALL Terraform-managed infrastructure — including the ECR repositories.
terraform -chdir=terraform destroy           # review the plan, then confirm
#   -> Destroy complete! Resources: 65 destroyed.

# 3. Nothing to revert: render-cloud-manifests.sh never edits a tracked file, so the
#    committed overlay is already at its 000000000000-placeholder state.
```

`terraform destroy` removes everything in the state — EKS cluster, node group, NAT
gateway, EIP, VPC, subnets, route tables, internet gateway, IAM roles, the KMS keys
and S3 buckets, **and the two ECR repositories** (with their lifecycle policies). As
of Sprint 7 the registries are Terraform-managed
([ADR-021](decisions/ADR-021-terraform-managed-ecr.md)) and set to `force_delete`, and
the S3 buckets to `force_destroy`, so `destroy` removes them and any images/objects
they still hold in the same pass — the old separate `aws ecr delete-repository
--force` step is **no longer needed**. (The customer-managed KMS keys move to a
scheduled-deletion window rather than vanishing instantly — see the runtime evidence.)

### 5.2 Verify AWS resources are gone

Do not trust the destroy line alone. Confirm from **three** angles:

```bash
# a. Terraform's own view: the state must be empty.
terraform -chdir=terraform show           # -> "The state file is empty. No resources are represented."

# b. AWS's view: the cluster must not exist.
aws eks list-clusters --region us-east-1 --query clusters --output text   # -> mlops-pipeline-dev-eks absent
aws ec2 describe-nat-gateways --region us-east-1 \
  --filter Name=state,Values=available --query 'NatGateways[].NatGatewayId' --output text  # -> empty

# c. ECR: the repository is gone.
aws ecr describe-repositories --repository-names mlops-pipeline --region us-east-1  # -> RepositoryNotFoundException
```

| Check | Expected | Why it matters |
|---|---|---|
| `terraform show` | *"The state file is empty."* | All managed resources (incl. the ECR repository) are released. |
| `aws eks list-clusters` | cluster **absent** | The control plane (flat hourly) has stopped billing. |
| `aws ec2 describe-nat-gateways` (available) | **none** | The NAT gateway (hourly + per-GB) has stopped billing. |
| `aws ec2 describe-addresses` unattached | **none billable** | No orphaned Elastic IP (billable only while unattached). |
| `aws ecr describe-repositories` | **not found** | No lingering image storage. |
| Repository working tree | **clean** | No account ID / run-time edit left in git (`git status`). |

If any check still shows a resource, **re-run `terraform destroy`** and delete
stragglers explicitly before walking away — an overlooked NAT gateway or node bills
silently.

> **Do not falsely claim cleanup.** If teardown must be *delayed* (e.g. a follow-up
> depends on the live environment), say so explicitly and state that resources are
> still billing — do not report "torn down" until [§5.2](#52-verify-aws-resources-are-gone)
> passes. In the Sprint 7 run, teardown was **not** delayed: all 65 resources were
> destroyed and verified clean the same session
> ([runtime evidence § Teardown](proof/sprint-07-runtime-evidence.md#15-teardown)).

---

## 6. Security

The cloud lifecycle keeps the repository safe to publish (full posture in
[SECURITY.md](../SECURITY.md)):

- **No static AWS credentials, ever** — Terraform and `kubectl` use the standard
  credential chain; nothing credential-shaped is read from or written to the repo.
- **State and tfvars are git-ignored** — `*.tfstate*`, `*.tfvars` (except
  `*.tfvars.example`), `tfplan`, and `.terraformrc` are excluded by
  [`.gitignore`](../.gitignore); state can contain resolved account IDs and
  resource attributes and must never be committed.
- **No kubeconfig in git** — `update-kubeconfig` writes to the operator's home
  directory, not the repo.
- **No account ID in git** — the committed ECR reference is a `000000000000`
  placeholder; the operator's real account is supplied at deploy time by
  [`scripts/render-cloud-manifests.sh`](../scripts/render-cloud-manifests.sh), which
  renders into a temp directory and never writes a tracked file. Evidence docs redact
  account IDs and the operator IP.
- **Least-privilege IAM** — dedicated cluster/node roles, AWS-managed policies only,
  no `AdministratorAccess`, no project-authored wildcard
  ([ADR-016](decisions/ADR-016-aws-iam-foundation.md)).
- **API endpoint private by default** (H-02) — public access is off out of the box;
  opting in requires a scoped `cluster_endpoint_public_access_cidrs` (your own `/32`).
  An unrestricted `0.0.0.0/0` is **rejected** by Terraform validation/preconditions,
  not merely discouraged ([ADR-022](decisions/ADR-022-eks-secure-api-access.md)).
- **CI holds no AWS access** — provisioning is operator-only
  ([ADR-019](decisions/ADR-019-terraform-ci-validation.md)).

---

## 7. Limitations

Stated plainly so nothing here is over-read. This is a **validation environment**;
it is **not**:

- **Not production** — a short-lived, single-operator proof, provisioned and
  destroyed, not an operated service.
- **A single environment** — no separate dev/staging/prod; one account, one
  environment, local Terraform state.
- **Limited scale** — 1 (default 2) `t3.medium` on-demand node, one small batch
  `Job`, a ~23 KiB dataset retrieved at runtime from a private S3 bucket (Sprint 7
  PR 8, [ADR-027](decisions/ADR-027-s3-dataset-runtime-retrieval.md)).
- **No GitOps** — no Argo CD / Flux; deployment is an operator running `kubectl
  apply`. CI validates manifests statically and never deploys.
- **No high-availability proof** — single node, single NAT gateway, single Job; no
  HA topology is provisioned or claimed.
- **No production observability** — diagnosis is `kubectl` + structured logs; no
  Prometheus/Grafana, tracing, alerting, or log aggregation.
- **No multi-region** — one region (`us-east-1`); nothing spans or fails over
  between regions.
- **Private API endpoint needs in-VPC reachability** — the API server is private by
  default (H-02). Reaching it *without* the scoped public opt-in requires network
  access into the VPC (a bastion host, a VPN/Direct Connect, an SSM port-forward, or
  a CI/ops runner in-VPC). This project provisions **none** of those, so the
  documented path for a workstation validation run is the scoped public opt-in
  (`cluster_endpoint_public_access = true` + your `/32`); a standing private-only
  operations posture would need that in-VPC access added
  ([ADR-022](decisions/ADR-022-eks-secure-api-access.md)).
- **No disaster-recovery proof** — no backup/restore, no state replication, no RTO/RPO
  target; teardown is intentional deletion, not a recovery drill.
- **No Terraform remote state** — local state only; a remote backend (S3 + lock
  table) is deliberately deferred (it would add billable AWS resources whose only
  purpose is the portfolio itself — [ADR-014](decisions/ADR-014-terraform-architecture.md)).
- **`readOnlyRootFilesystem: false`** — deferred by design
  ([ADR-010](decisions/ADR-010-kubernetes-security-hardening.md)); DVC writes state
  in-tree.
- **Restricted Pod Security Standard compliance not claimed** — controls are applied
  and a real API server admitted the pod, but no admission-policy engine ratifies
  the profile.

The credible claim is narrow and evidenced: **the existing MLOps pipeline runs to
completion on a real, Terraform-provisioned EKS cluster — with Terraform-managed ECR,
KMS-encrypted S3 dataset and MLflow artifact stores, EKS Pod Identity workload
identity (no static keys), and the in-cluster PostgreSQL+S3 MLflow tracking server —
with the Sprint 5 security controls intact, and the environment is destroyed and
verified clean afterward.** Everything beyond that — GitOps, Terraform remote state,
multi-region, HA/DR, and production observability — is future work
([roadmap.md](roadmap.md) v5–v6), not a present capability.

---

## Related documentation

- [`terraform/README.md`](../terraform/README.md) — Terraform reference (resources, variables, state)
- [Sprint 7 — Runtime Evidence](proof/sprint-07-runtime-evidence.md) — the executed full-platform EKS run
- [Sprint 7 — Proof Impact](proof/sprint-07-proof-impact.md) — before/after credible claims
- [Sprint 6 — Runtime Evidence](proof/sprint-06-runtime-evidence.md) — the earlier Sprint 6 EKS run
- [Kubernetes Operations](kubernetes-operations.md) — in-cluster day-2 operations (local + cloud)
- [ADR-020](decisions/ADR-020-cloud-lifecycle-cost-control.md) · [ADR-014](decisions/ADR-014-terraform-architecture.md) · [ADR-015](decisions/ADR-015-aws-network-architecture.md) · [ADR-016](decisions/ADR-016-aws-iam-foundation.md) · [ADR-017](decisions/ADR-017-eks-platform.md) · [ADR-018](decisions/ADR-018-aws-eks-deployment-overlay.md) · [ADR-019](decisions/ADR-019-terraform-ci-validation.md)
- [SECURITY.md](../SECURITY.md) — repository security policy
