# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security

- **Secure-by-default EKS API access — closes finding H-02** (Sprint 7, PR 2) — the
  EKS control-plane endpoint is now **private by default** and public access can
  **never be unrestricted**, fixing an insecure default where the Kubernetes API
  server was reachable from `0.0.0.0/0` out of the box. The fix is enforced by the
  configuration itself, not by documentation. Design of record:
  [ADR-022](docs/decisions/ADR-022-eks-secure-api-access.md).
  - **Secure defaults** — in [`variables.tf`](terraform/variables.tf),
    `cluster_endpoint_public_access` now defaults to **`false`** (was `true`) and
    `cluster_endpoint_public_access_cidrs` to **`[]`** (was `["0.0.0.0/0"]`);
    `cluster_endpoint_private_access` stays `true`. A fresh `apply` is private-only.
  - **Executable guardrails** — the CIDR variable rejects **any `/0`** (including
    `0.0.0.0/0`) and any invalid CIDR; `lifecycle` **preconditions** on
    [`aws_eks_cluster`](terraform/eks.tf) reject enabling public access with an empty
    allow-list (EKS would treat empty as `0.0.0.0/0`) and reject disabling both
    endpoints (an unreachable API). Public access remains a **scoped, explicit
    opt-in**.
  - **Contract tests** — new [`terraform/tests/eks_api_security.tftest.hcl`](terraform/tests/eks_api_security.tftest.hcl)
    runs offline under `mock_provider "aws"` (`command = plan`, **no AWS, no
    credentials**) and asserts private-by-default, `0.0.0.0/0`/`/0`/invalid CIDRs
    rejected, public-without-CIDRs rejected, both-endpoints-off rejected, and the
    scoped opt-in planning cleanly. Wired into the existing `terraform test` CI step.
  - **Suppressions removed, not re-justified** — the two Trivy entries that excused
    the old open endpoint (`AVD-AWS-0040`, `AVD-AWS-0041`) are **deleted** from
    [`terraform/.trivyignore`](terraform/.trivyignore); with no default public
    exposure the scanner passes on its own.
  - **Docs** — [ADR-017](docs/decisions/ADR-017-eks-platform.md) endpoint posture
    superseded by ADR-022; [`terraform/README.md`](terraform/README.md),
    [`terraform.tfvars.example`](terraform/terraform.tfvars.example),
    [SECURITY.md](SECURITY.md), [architecture](docs/architecture.md),
    [ci-cd](docs/ci-cd.md), and the
    [cloud-operations runbook](docs/cloud-operations.md) updated for the private
    default, the scoped opt-in, and the in-VPC-reachability limitation. No
    credentials or account-specific values committed.

### Added

- **Terraform-managed Amazon ECR — closes finding H-01** (Sprint 7, PR 1) — brings
  the container registry that stores the workload image **fully under Terraform**, so
  the entire AWS platform now has one lifecycle (`apply` creates, `destroy` removes).
  Previously the repository was created/deleted out-of-band with the AWS CLI, leaving
  one live resource outside Terraform state. Design of record:
  [ADR-021](docs/decisions/ADR-021-terraform-managed-ecr.md).
  - **New [`terraform/ecr.tf`](terraform/ecr.tf)** — an `aws_ecr_repository` plus an
    `aws_ecr_lifecycle_policy`. **Private** (no public/cross-account policy),
    **immutable tags** (a version can never be repointed — matches the overlay's
    image-pinning contract), **scan-on-push ON**, **encrypted at rest** (AES256; a KMS
    CMK is the documented M-02 follow-up), a **lifecycle policy** retaining the most
    recent `ecr_max_image_count` images (default 10), and `force_delete` so
    `terraform destroy` removes it and its images in one pass.
  - **New variables & outputs** — `ecr_repository_name` (defaults to `project_name`,
    keeping it in lock-step with the committed overlay image) and `ecr_max_image_count`
    in [`variables.tf`](terraform/variables.tf); `ecr_repository_name` (non-sensitive)
    plus **`sensitive`** `ecr_repository_url` / `ecr_repository_arn` (they embed the
    account ID) in [`outputs.tf`](terraform/outputs.tf).
  - **No more manual ECR step** — the `aws ecr create-repository` and
    `aws ecr delete-repository --force` steps are removed from
    [`docs/cloud-operations.md`](docs/cloud-operations.md); the runbook now reads the
    registry URL from `terraform output -raw ecr_repository_url`. A default plan is now
    **31 resources** (was 29). [ADR-020](docs/decisions/ADR-020-cloud-lifecycle-cost-control.md)
    updated to record the manual teardown step as resolved.
  - **Offline contract tests** — new [`terraform/tests/ecr.tftest.hcl`](terraform/tests/ecr.tftest.hcl)
    runs under `mock_provider "aws"` (`command = plan`, **no AWS, no credentials**) and
    asserts the name, immutability, scan-on-push, encryption, and retention policy. Wired
    into the `terraform-validate` CI job as a `terraform test` step; `required_version`
    raised to `>= 1.7.0` for `mock_provider`.
  - **Security preserved** — ECR is **not** public, no new IAM permission is added (the
    node role's existing read-only ECR pull is unchanged), and no security feature is
    disabled to simplify. No credentials, account IDs, or environment-specific secrets
    are committed.

- **Cloud cost controls, teardown & lifecycle documentation** (Sprint 6, PR 8 —
  the final Sprint 6 PR) — completes the cloud lifecycle documentation and prepares
  the repository for the Sprint 6 release gate. **Adds no new infrastructure and
  makes no production claims.** Design of record:
  [ADR-020](docs/decisions/ADR-020-cloud-lifecycle-cost-control.md).
  - **New [`docs/cloud-operations.md`](docs/cloud-operations.md)** — the operator's
    cloud runbook: prerequisites, AWS authentication, `init` → `plan` → `apply`, EKS
    verification, `kubectl` configuration, ECR publish, workload execution, evidence
    capture, and **`terraform destroy` with three-angle cleanup verification**. It
    documents the **AWS cost drivers** (EKS control-plane and EC2 node per-hour, NAT
    per-hour + per-GB, minor EBS/CloudWatch/ECR, free IAM/VPC/addons) with
    **time-alive named as the dominant lever**, and **why the environment is
    intentionally small** (1–2 nodes, single NAT, two AZs, short-lived, local state).
  - **Ephemeral `provision → prove → destroy` lifecycle ratified** — the environment
    is a short-lived, single-operator **validation** environment; teardown is the
    mandatory final step, and cleanup is *verified* (Terraform state empty + AWS API
    shows no cluster/NAT + ECR repo absent), never claimed from the destroy line
    alone.
  - **Honest limitations documented** — not production, single environment, limited
    scale, **no GitOps, no HA proof, no production observability, no multi-region, no
    disaster-recovery proof** — in the runbook and the new proof assessment.
  - **New [`docs/proof/sprint-06-proof-impact.md`](docs/proof/sprint-06-proof-impact.md)**
    — an evidence-based before/after of what the project can credibly claim after
    Sprint 6 (cloud IaC, least-privilege cloud identity, credential-free CI gate, a
    **green run on real EKS**, live-pod security verification, verified teardown) that
    it could not after Sprint 5, with a conservative capability table and an explicit
    "what still cannot be claimed" section.
  - **Docs refreshed for factual consistency** — [`terraform/README.md`](terraform/README.md)
    (PR 7 executed / PR 8 lifecycle pointers, node-count note), [`docs/architecture.md`](docs/architecture.md)
    (Kubernetes now runs green locally **and on EKS**; new Cloud Platform section),
    the docs/ADR/roadmap indices, and [`SECURITY.md`](SECURITY.md).
  - **Final validation re-run green** — `terraform fmt -check -recursive` clean,
    `terraform validate` **Success**, `python k8s/validate.py` **45/45**, both
    Kustomize overlays render, `pytest` **100 passed / 1 skipped**. Repository
    security scan clean: no committed state, tfvars, kubeconfig, secrets, or account
    identifiers (the ECR image stays a `000000000000` placeholder).

- **Real cloud integration test — runtime proof on Amazon EKS** (Sprint 6, PR 7) —
  provisioned the Terraform-defined platform in the operator's **own** AWS account
  and ran the existing MLOps `Job` on **real managed EKS** to completion, then
  destroyed and verified the environment. Redacted evidence:
  [`docs/proof/sprint-06-runtime-evidence.md`](docs/proof/sprint-06-runtime-evidence.md).
  - **Infrastructure applied** — `terraform apply` created exactly the plan:
    **29 resources added, 0 changed, 0 destroyed** (VPC, IAM, EKS, a 1-node group).
  - **Green run on EKS** — EKS **ACTIVE** (control plane **v1.35.6-eks**, 1 node
    Ready in a private subnet); image pulled from **Amazon ECR** via the node role
    (no pod credential); Job **`Complete`** (1/1, **52s**), pod **`Succeeded`, exit
    0**, all four stages (preprocess 768 → split 614/154 → train **0.7398** →
    evaluate **0.7078**, matching the Sprint 5 local metrics).
  - **Sprint 5 security controls verified on the LIVE pod** — all six (non-root,
    uid/gid `10001`, seccomp `RuntimeDefault`, no privilege escalation, `drop:[ALL]`,
    measured requests/limits) read directly from the running pod; token automount
    off; Burstable QoS. Inherited **verbatim** from the committed base — the overlay
    weakened nothing.
  - **Honest deviation recorded** — the run used a **transient offline MLflow file
    store** (DagsHub credentials not supplied), so real tracking connectivity was
    **not** exercised; the patch was **not committed** and was reverted at teardown.
  - **Torn down and verified clean** — `terraform destroy` → **"Destroy complete!
    Resources: 29 destroyed."**, local state **empty**, ECR repository deleted,
    working tree clean. **No ongoing cost, no leftover diff.**
  - **No credentials/secrets/account IDs committed** — evidence redacts the account
    ID and operator IP; no kubeconfig, token, state, or tfvars entered git.

- **Terraform CI validation gates** (Sprint 6, PR 6) — a new
  **`terraform-validate`** job in [`ci.yml`](.github/workflows/ci.yml) that
  statically validates the `terraform/` IaC on every push/PR, **in parallel** with
  the existing gates and with **no AWS access**. Design of record:
  [ADR-019](docs/decisions/ADR-019-terraform-ci-validation.md).
  - **Offline gate chain (pinned tools)** — `terraform fmt -check -recursive` →
    `terraform init -backend=false` (providers only, no backend/state, no
    credentials) → `terraform validate` → **TFLint** (language preset + AWS
    ruleset, config in [`terraform/.tflint.hcl`](terraform/.tflint.hcl)) → **Trivy**
    `config` IaC misconfiguration scan (**fails on CRITICAL/HIGH**).
  - **No unsafe provisioning** — CI **never** runs `terraform plan` or `apply`,
    holds **no AWS credentials/OIDC identity**, and keeps `permissions:
    contents: read`. `plan` needs live credentials (it reads AWS data sources), so
    it is intentionally an operator-driven, own-account step; the boundary is
    **documented, not credentialed** ([terraform/README.md](terraform/README.md),
    [docs/ci-cd.md](docs/ci-cd.md)).
  - **Trivy suppressions are a justified triage record** — the few intentional,
    ADR-ratified validation-cluster exposures (open default API CIDR, no KMS
    envelope encryption, public-subnet public IPs) are suppressed **with
    rationale** in
    [`terraform/.trivyignore`](terraform/.trivyignore); any **new** critical/high
    finding blocks the merge.
  - **All existing CI jobs preserved** — `quality`, `docker`, `k8s-validate`, and
    the opt-in `k8s-cluster-dry-run` are unchanged; the least-privilege,
    no-publish posture is maintained.

- **AWS EKS Deployment Overlay** (Sprint 6, PR 5) — integrates the existing
  Sprint 5 Kubernetes workload with the Terraform-provisioned EKS platform by
  **reusing the base unchanged**, with **no workload duplication** and **no
  security weakening**. Design of record:
  [ADR-018](docs/decisions/ADR-018-aws-eks-deployment-overlay.md).
  - **New [`k8s/overlays/aws/`](k8s/overlays/aws/)** — a thin overlay over
    `../../base` that layers **only** the three genuine cloud differences: the
    image source (**Amazon ECR**, pulled via the node role's ECR read-only policy
    — no pod credential/IRSA), an explicit **`imagePullPolicy: Always`**, and the
    runtime **dataset mount** at `/app/data/raw`. It **deliberately does not**
    override MLflow, so the cloud run uses the **real DagsHub endpoint** + the
    out-of-band Secret (the local overlay's in-pod file store is local-only).
  - **Security preserved, provably** — the rendered pod/container
    `securityContext`, `resources`, `serviceAccountName`, and
    `automountServiceAccountToken` are **byte-identical** to the local overlay's
    (`runAsNonRoot` + uid/gid `10001`, `allowPrivilegeEscalation: false`,
    `capabilities: drop [ALL]`, seccomp `RuntimeDefault`, measured
    requests/limits). Both overlays pass the same **45-check** static contract.
  - **No account ID in git** — the ECR reference is a committed
    `000000000000`/`us-east-1` **placeholder**; the operator points it at their
    own account at deploy time via `kustomize edit set image` (no file edit).
  - **Validation tooling extended** — CI now runs the Kustomize render +
    `kubeconform -strict` schema gate, the `k8s/validate.py`
    security/runtime-contract gate, and the opt-in server-side dry-run admission
    for **both** the `local` and `aws` overlays.
  - **Base and local overlay unchanged** — local behavior is byte-for-byte
    preserved; no ingress, Service, mesh, GitOps, observability, or new AWS
    service is added. The **real EKS execution** (Job Complete, exit 0) is the
    Sprint 6 PR 7 integration test.
  - **Docs** — [`k8s/README.md`](k8s/README.md) § "AWS overlay — deploy to EKS"
    and [`docs/kubernetes-architecture.md`](docs/kubernetes-architecture.md)
    document the cloud-specific differences and the static-vs-runtime boundary.
- **Managed EKS Platform** (Sprint 6, PR 4) — provisions the real Kubernetes
  platform as Terraform, consuming the PR 2 network and PR 3 IAM roles. Design of
  record: [ADR-017](docs/decisions/ADR-017-eks-platform.md).
  - **New [`terraform/eks.tf`](terraform/eks.tf)** — an **EKS control plane**
    (assuming the PR 3 cluster role), one **managed node group** (assuming the PR
    3 node role), and the **three core addons** (`vpc-cni`, `coredns`,
    `kube-proxy`). With the defaults a plan adds **5 resources**, taking the full
    stack to **29** (18 network + 6 IAM + 5 EKS).
  - **Explicit Kubernetes version** — pinned to **1.35** (`kubernetes_version`)
    for reproducibility rather than tracking the newest; EKS manages the patch
    level and addon versions track the control plane.
  - **Small, cost-conscious nodes** — a **fixed pair of `t3.medium`** on-demand
    nodes on Amazon Linux 2023 (`min = max = desired = 2`, no Cluster Autoscaler,
    no GPU, no SSH), in the **private subnets** with egress via NAT. All sizing is
    variable-driven (`node_instance_types`, `node_capacity_type`, `node_*_size`,
    `node_disk_size`).
  - **Sensible endpoint/security** — **both** private and public API access
    enabled; the public source range (`cluster_endpoint_public_access_cidrs`)
    defaults open **for first-run validation only** and should be restricted for
    real use. _(Superseded within this same release by the secure-by-default EKS
    API posture — see the **Security** entry above (H-02): the endpoint now
    defaults to private-only and can never be `0.0.0.0/0`.)_ Access uses **EKS
    access entries** (`API_AND_CONFIG_MAP`) and
    bootstraps the creator as cluster admin; control-plane `api`/`audit`/
    `authenticator` logs ship to CloudWatch (toggleable).
  - **Non-sensitive EKS outputs** — `eks_cluster_name`, `eks_cluster_endpoint`,
    `eks_cluster_version`, `eks_cluster_security_group_id`,
    `eks_cluster_oidc_issuer_url`, `eks_node_group_name`, and a ready-to-run
    `configure_kubectl` command. No kubeconfig, token, or certificate is emitted.
  - **Explicit non-goals** — no autoscaling, GPUs, service mesh, ingress stack,
    observability stack, optional addons, extra AWS services, or workload
    resources; the `Job` stays in Kustomize. KMS secret envelope encryption and
    CNI-via-IRSA are documented follow-ups.
  - **Docs** — [`terraform/README.md`](terraform/README.md) § EKS platform and
    [`terraform/terraform.tfvars.example`](terraform/terraform.tfvars.example)
    document the architecture, node sizing, version, endpoint/security decisions,
    and cost implications.
- **AWS IAM Foundation** (Sprint 6, PR 3) — adds the least-privilege IAM roles a
  managed EKS cluster needs, building on the PR 2 network. Creates **no EKS,
  EC2, or application resources, and no static credentials**. Design of record:
  [ADR-016](docs/decisions/ADR-016-aws-iam-foundation.md).
  - **New [`terraform/iam.tf`](terraform/iam.tf)** — two dedicated IAM roles: an
    **EKS control-plane role** (trusted only by `eks.amazonaws.com`, attached
    `AmazonEKSClusterPolicy`) and an **EKS worker-node role** (trusted only by
    `ec2.amazonaws.com`, attached `AmazonEKSWorkerNodePolicy`,
    `AmazonEKS_CNI_Policy`, and `AmazonEC2ContainerRegistryReadOnly`). With the
    defaults a plan adds **6 resources** (2 roles + 4 attachments), all **free**.
  - **Least privilege by construction** — each role has a single-service trust
    policy and only AWS-managed policies; **no inline policy, no project-authored
    wildcard, no `AdministratorAccess`**. `AmazonEKSVPCResourceController` and
    `AmazonSSMManagedInstanceCore` are intentionally omitted; the CNI-via-IRSA
    hardening is deferred to the EKS PR. Managed-policy ARNs are partition-aware
    (`data.aws_partition`).
  - **New IAM outputs** consumed by the later EKS PR — `eks_cluster_role_name`,
    `eks_node_role_name` (plain) and `eks_cluster_role_arn`, `eks_node_role_arn`
    (marked **`sensitive`**, since a role ARN embeds the AWS account ID).
  - **Docs** — [`terraform/README.md`](terraform/README.md) § IAM foundation and
    [`SECURITY.md`](SECURITY.md) document the roles, trust, permissions, and what
    is intentionally not permitted.
- **AWS Network Foundation** (Sprint 6, PR 2) — provisions the minimum,
  EKS-ready network as Terraform, building on the PR 1 foundation. Design of
  record: [ADR-015](docs/decisions/ADR-015-aws-network-architecture.md).
  - **New [`terraform/network.tf`](terraform/network.tf)** — a single VPC
    (`10.0.0.0/16` by default) with a **public and a private subnet per
    Availability Zone**, an internet gateway, a **single shared NAT gateway**,
    and per-tier route tables. With the defaults a plan proposes **18 resources**
    (2 AZs). EKS worker nodes will run in the private subnets and reach the
    internet outbound-only through NAT; the batch-`Job` workload has no inbound
    surface, so no public ingress is created.
  - **AZs discovered at plan time** via `aws_availability_zones` (never
    hard-coded); `az_count` (default 2, the EKS minimum) and the VPC CIDR are
    variables, and derived subnet CIDRs use `cidrsubnet`.
  - **EKS subnet tags** — public subnets tagged `kubernetes.io/role/elb`,
    private subnets `kubernetes.io/role/internal-elb`, so Kubernetes load-balancer
    provisioning can discover them later.
  - **Cost-aware, variable-driven NAT** — `single_nat_gateway` (default `true`)
    shares one NAT across AZs; `enable_nat_gateway` can remove NAT entirely. The
    NAT gateway is the dominant cost and is documented in
    [`terraform/README.md`](terraform/README.md) § Network architecture.
  - **New network outputs** consumed by the later EKS PR — `vpc_id`,
    `vpc_cidr_block`, `availability_zones`, `public_subnet_ids`,
    `private_subnet_ids`, `internet_gateway_id`, `nat_gateway_ids`,
    `nat_public_ips`. No EKS, IAM, or application resources are created in this PR.
- **Terraform Foundation** (Sprint 6, PR 1) — introduces a professional
  Infrastructure-as-Code foundation for the project's AWS platform, declaring
  **no billable AWS resources yet**. Design of record:
  [ADR-014](docs/decisions/ADR-014-terraform-architecture.md).
  - **New [`terraform/`](terraform/) root module** with the conventional split:
    [`versions.tf`](terraform/versions.tf) (Terraform `>= 1.6.0, < 2.0.0`, AWS
    provider `~> 5.60`), [`providers.tf`](terraform/providers.tf) (region +
    `default_tags`, no hard-coded credentials),
    [`variables.tf`](terraform/variables.tf) (validated `aws_region`,
    `project_name`, `environment`, `owner`, `repository_url`, `additional_tags`),
    [`main.tf`](terraform/main.tf) (`name_prefix` + `common_tags` locals and
    account/region context data sources), [`outputs.tf`](terraform/outputs.tf)
    (region, sensitive account ID, name prefix, common tags), and
    [`terraform.tfvars.example`](terraform/terraform.tfvars.example) (placeholders
    only).
  - **Naming & tagging strategy** — a `"<project>-<environment>"` name prefix and
    a common tag set (`Project`, `Environment`, `ManagedBy`, `Owner`,
    `Repository`) applied globally via the provider's `default_tags`, inherited by
    every resource added in later PRs.
  - **No premature `modules/`** — a single small root module is the honest shape
    for a resource-free foundation; modules are extracted only when a real
    boundary (network, EKS) exists.
  - **Local-state posture, remote-ready** — the default local backend is used for
    controlled single-operator validation; `.gitignore` blocks `.terraform/`,
    `*.tfstate*`, `*.tfvars` (except the committed `*.tfvars.example`), crash
    logs, plan artifacts, and override files. The remote S3 + locking + KMS +
    least-privilege upgrade path for team/production use is documented.
  - **Security posture** — the repository stays safe to publish: no credentials,
    state, or account IDs are committed; the `aws_account_id` output is marked
    `sensitive`; authentication is delegated to the standard AWS credential chain.
  - **Docs** — [`terraform/README.md`](terraform/README.md) (purpose, structure,
    authentication, init/validate/plan, state handling, security rules, and what
    later Sprint 6 PRs provision) and
    [ADR-014](docs/decisions/ADR-014-terraform-architecture.md) added to the
    [ADR index](docs/decisions/README.md). Scope is strictly foundation: no VPC,
    IAM, or EKS yet, and the existing CI workflow is unchanged (Terraform CI gates
    are Sprint 6, PR 6).
- **Kubernetes Runtime Execution** (Sprint 5, PR 8) — closes the last Sprint 5 proof
  gap: the **complete ML pipeline now runs to completion inside the secured
  Kubernetes Job**. Design of record:
  [ADR-013](docs/decisions/ADR-013-kubernetes-runtime-execution.md).
  - **Capabilities delivered:**
    - **Kubernetes Job execution** — the workload runs as a secured `batch/v1` Job on
      a local cluster (Docker Desktop Kubernetes v1.34.3).
    - **DVC no-SCM runtime configuration** — `core.no_scm = true` via a mounted
      `config.local`, resolving the `/app is not a git repository` abort.
    - **Runtime dataset provisioning** — the dataset mounted read-only at
      `/app/data/raw` from an out-of-band ConfigMap (local-validation, not production).
    - **Local MLflow execution** — an in-pod file store
      (`file:///app/mlruns` + `MLFLOW_ALLOW_FILE_STORE=true`), fully offline.
    - **Successful end-to-end Job completion** — Job `Complete`, pod `Succeeded`,
      **exit 0**, all four stages (preprocess → split → train → evaluate).
    - **Failure/retry validation** — a controlled missing-dataset run demonstrated
      fail-fast → back-off → terminal `Failed (BackoffLimitExceeded)`, then restored.
    - **Runtime validation contract** — `k8s/validate.py` now statically asserts the
      no-SCM config + mount, the dataset mount + backing volume, the MLflow endpoint,
      and the `dvc` command (**43/43** checks; runs in CI).
  - **Root cause (diagnosed against the real image).** `dvc repro` aborted with
    `/app is not a git repository` — the runtime image ships no `.git` (by design),
    and DVC defaults to requiring an SCM. Two further blockers followed: the image
    ships no dataset, and MLflow needs an endpoint.
  - **Minimal runtime contract, as Kubernetes config only (no app/image change).**
    New base ConfigMap [`k8s/base/dvc-config.yaml`](k8s/base/dvc-config.yaml) carries
    `config.local` with **`core.no_scm = true`** (DVC's supported no-SCM mode),
    mounted read-only at `/app/.dvc/config.local` (subPath) — so the committed
    `.dvc/config` and the dev/CI Git+DVC workflow are untouched. New local-overlay
    patch [`k8s/overlays/local/job-runtime.yaml`](k8s/overlays/local/job-runtime.yaml)
    mounts the **dataset** read-only at `/app/data/raw` from an **out-of-band**
    ConfigMap (like the Secret; `optional: true` → graceful missing-input failure),
    and overrides MLflow to an **in-pod file store** (`file:///app/mlruns` +
    `MLFLOW_ALLOW_FILE_STORE=true`) so a local run needs no external MLflow or
    credentials. The base keeps the DagsHub endpoint + Secret for real use.
  - **Verified green end-to-end on a live cluster** (Docker Desktop Kubernetes
    v1.34.3, 2026-08-14): Job **`Complete`**, pod **`Succeeded`**, container **exit
    0**, first attempt (`RESTARTS: 0`), all four stages (preprocess 768 → split
    614/154 → train acc 0.7398 → evaluate acc 0.7078). **Failure test:** removing the
    dataset ConfigMap → fail-fast at preprocess → 3 fresh-pod attempts → terminal
    `Failed: BackoffLimitExceeded`; restoring it returns the Job to green.
  - **Security posture preserved and re-verified:** non-root/`10001`, seccomp
    `RuntimeDefault`, `allowPrivilegeEscalation: false`, `capabilities.drop [ALL]`,
    `automountServiceAccountToken: false`, QoS `Burstable`; the two added mounts are
    read-only ConfigMaps. `readOnlyRootFilesystem` stays `false` (now the *only*
    remaining item of the ADR-010 deferral).
  - **Static validation extended:** [`k8s/validate.py`](k8s/validate.py) gained a
    **Runtime execution contract** section (no-SCM ConfigMap + `/app/.dvc/config.local`
    mount, dataset mount at `/app/data/raw` backed by a declared volume, configured
    `MLFLOW_TRACKING_URI`, `dvc` command) — **43/43** checks pass; it runs in CI's
    `k8s-validate` job automatically.
  - **Docs updated:** [`k8s/README.md`](k8s/README.md) (runtime record + runbook +
    Docker-Desktop containerd image-load note + expanded troubleshooting),
    [operations](docs/kubernetes-operations.md), [security](docs/kubernetes-security.md),
    the [Sprint 5 proof](docs/proof/sprint-05-proof-impact.md), and
    [ADR-010](docs/decisions/ADR-010-kubernetes-security-hardening.md). The dataset
    ConfigMap and MLflow file store are documented as **local-validation mechanisms,
    not production storage**; no production/cloud deployment is claimed.
- **Kubernetes operations, security & proof documentation** (Sprint 5, PR 7) — the
  final Sprint 5 PR; documentation and evidence only, **no manifest or code change**.
  - A complete **deployment guide** in [`k8s/README.md`](k8s/README.md): prerequisites,
    local cluster setup, image build/load, optional secret setup, deploy, inspect,
    logs, a deployment **troubleshooting matrix**, cleanup, and re-run.
  - A new [**Kubernetes Operations** runbook](docs/kubernetes-operations.md): the
    operational model, routine day-2 operations (deploy/observe/logs/re-run/rotate
    secret/update config/tear down), a full **symptom → cause → investigation →
    remediation** matrix, a failure-handling playbook, and an **honest observability
    posture** (`kubectl` + structured logs; **no** production observability stack).
  - A new [**Kubernetes Security** document](docs/kubernetes-security.md): the
    two-layer (image + platform) enforcement, least-privilege identity, the pod/
    container `securityContext`, the read-only-root deferral, the secret/data-handling
    model, and a controls→evidence checklist with an explicit "what is **not** claimed"
    (no restricted-PSS certification, no `NetworkPolicy`, no production baseline).
  - A new [**Sprint 5 Proof-Impact Assessment**](docs/proof/sprint-05-proof-impact.md):
    a conservative **Before/After** capability model, evidence-linked new claims,
    a proof/evidence table across the six dimensions (workload model, security,
    resources, configuration, validation, operations), and explicit known limitations
    (**local cluster only; no production cloud deployment; no GitOps; no production
    observability/HA/serving**).
  - **The local deployment path was re-executed from a clean state** (2026-08-12,
    Docker Desktop Kubernetes v1.34.3) as evidence: render → `python k8s/validate.py`
    (**34/34**) → `kubectl apply -k` (4 objects, image resolved with no registry pull)
    → the designed **3-attempt** back-off lifecycle (`RESTARTS: 0`, Job `Failed`) →
    logs (`/app is not a git repository`) → `kubectl delete -k` (namespace `NotFound`).
    The live pod enforced QoS `Burstable`, the exact `securityContext` and `resources`,
    and an **empty** `volumes` (no API token). Consistent with every prior Sprint 5
    record: the Job **mechanism** and control **enforcement** are proven on a real
    cluster; a **green** pipeline run is **not** claimed (the image lacks an SCM).
  - Documentation reconciled to reflect the final Sprint 5 implementation:
    [kubernetes-architecture.md](docs/kubernetes-architecture.md) (scope → PR 7),
    [roadmap.md](docs/roadmap.md) (v4 operations & proof ✅), the root
    [README](README.md) (Kubernetes status + doc links), and the
    [docs index](docs/README.md) (Kubernetes docs, refreshed ADR list, Sprint 5 proof).
    **No release tag is cut in this PR.** No production Kubernetes expertise is claimed.
- **Automated Kubernetes manifest validation in CI** (Sprint 5, PR 6) — every push
  and pull request now statically validates the `k8s/` manifests, so a future edit
  cannot silently regress the PR 1–5 contract.
  - New CI job **`k8s-validate`** (runs in parallel with `quality`, no cluster): a
    minimal, pinned toolchain — `kustomize` renders `base/` + `overlays/local/`
    (Kustomize + YAML-syntax check), `kubeconform -strict` validates every object
    against the pinned upstream Kubernetes **schema** (rejecting unknown fields),
    and a new project script **`k8s/validate.py`** (stdlib + PyYAML) asserts the
    **security/required-field contract** with a PASS/FAIL line per check:
    `runAsNonRoot`, `allowPrivilegeEscalation: false`, seccomp `RuntimeDefault`,
    `capabilities: drop [ALL]`, explicit non-default ServiceAccount, token automount
    off, CPU/memory **requests and limits**, explicit **pinned** image, namespace
    pinning, and **secret hygiene** (no rendered `Secret`, no inline credentials, no
    secret fingerprints, template holds only placeholders).
  - New **opt-in** job **`k8s-cluster-dry-run`** (`workflow_dispatch` only): an
    ephemeral **kind** cluster + a **server-side dry-run**
    (`kubectl apply -k … --dry-run=server`) validates admissibility (schema,
    defaulting, Pod Security) **without** persisting or running the workload. Kept
    off the per-PR path to stay deterministic and fast.
  - Tool **and** schema versions are pinned (kustomize 5.4.3, kubeconform 0.6.7,
    k8s schema 1.31.0) and the downloaded binaries are checksum-verified.
  - Validated locally end to end: `kubeconform` passes for base + overlay (4/4
    each); `k8s/validate.py` passes 34/34; a temporarily flipped
    `allowPrivilegeEscalation`/`runAsNonRoot` is caught by `k8s/validate.py` and a
    string `activeDeadlineSeconds` is caught by `kubeconform -strict`; the
    server-side dry-run admits all objects on Docker Desktop. **CI validation is
    static** (plus the opt-in dry-run) — it does **not** deploy or run the workload.
  - [ADR-012](docs/decisions/ADR-012-kubernetes-manifest-validation.md) records the
    tiered design, the rejected alternatives (OPA/Gatekeeper, kube-linter, kubeval,
    per-PR kind), and what the decision does not imply.
- **Kubernetes resource & lifecycle management** (Sprint 5, PR 5) — the pipeline
  `Job` now declares CPU/memory requests and limits chosen from **measured** usage
  of the real image, and its finite-run lifecycle and probe decision are documented.
  - `resources.requests: {cpu: 250m, memory: 256Mi}` and
    `resources.limits: {cpu: "1", memory: 512Mi}` → **Burstable** QoS. Values are
    derived from `docker run` probes of the real `ml-pipeline:local` image, not
    guessed: the import floor is ~132 MiB, and peak memory scales with granted CPU
    (1 CPU → ~133 MiB/~2.5 s; 2 → ~419 MiB; unlimited → ~1785 MiB/~20 s) because
    `GridSearchCV(n_jobs=-1)` sizes joblib's worker fan-out from the cgroup CPU
    quota (`joblib.cpu_count()` returns `2` under `--cpus=2` while `os.cpu_count()`
    returns `20`). The **CPU limit therefore doubles as the memory-safety control**.
  - Lifecycle reviewed and documented: `restartPolicy: Never`, `backoffLimit: 2`
    (deterministic-failure aware; exponential back-off absorbs only transient
    faults), and `activeDeadlineSeconds: 1800` (an outer stall-guard, not an SLO).
  - **No health probes**, by design — a finite batch Job has no socket/Service and
    its health is terminal (exit code), which the Job controller reads directly; a
    probe would need an endpoint the app should not expose or would kill a healthy
    quiet run. Documented alongside the five failure modes (image pull, config,
    secret, application, resource exhaustion).
  - Validated: at the chosen limits the container completes (exit 0, ~133 MiB
    peak); at `--memory=64m` it is `OOMKilled` (exit 137, limit kernel-enforced);
    on a live Docker Desktop cluster (v1.34.3) the enforced `resources`, `Burstable`
    QoS, absence of probes, and 3-attempt back-off lifecycle were confirmed, with
    every attempt hitting the *same* pre-existing SCM blocker (exit 255) and **none
    `OOMKilled`** — no new failure mode. Values are **not** production-certified.
  - [ADR-011](docs/decisions/ADR-011-kubernetes-resource-lifecycle.md) records the
    method, the measurements, the alternatives, and what the decision does not imply.
- **Kubernetes architectural foundation** (Sprint 5, PR 1) — the containerized
  pipeline is now expressed as a Kubernetes **batch workload**, without
  manufacturing a fake online service. Foundation only: it establishes the
  structure and workload model; configuration/secrets, security hardening,
  resource limits, CI validation, and a demonstrated cluster run are deferred to
  later Sprint 5 PRs.
  - A [`k8s/`](k8s/) directory using a Kustomize `base/` + `overlays/local/`
    layout: an `mlops` **Namespace** (the environment boundary) and the pipeline
    modelled as a run-to-completion **`batch/v1` Job** (`restartPolicy: Never`,
    bounded `backoffLimit`) that runs `dvc repro`. Both `kustomize build k8s/base`
    and `kustomize build k8s/overlays/local` render successfully; the local
    overlay maps the workload to the locally built `ml-pipeline:local` image.
  - [ADR-009](docs/decisions/ADR-009-kubernetes-workload-model.md) recording the
    **Job-vs-Deployment** decision — batch vs service semantics, completion/retry
    lifecycle, why a Deployment (and a fake HTTP API) were rejected, and what the
    decision explicitly does *not* imply.
  - [docs/kubernetes-architecture.md](docs/kubernetes-architecture.md) describing
    why Kubernetes is introduced, the workload architecture, the configuration and
    security boundaries (as design contracts for later PRs), and a clear
    local-validation-vs-production-deferred split.
  - A batch-workload architecture diagram (Mermaid + ASCII) under
    [docs/diagrams/kubernetes-architecture/](docs/diagrams/kubernetes-architecture/).
  - [`k8s/README.md`](k8s/README.md) with render/apply instructions and a table of
    what is deliberately deferred to which PR.
- **Runnable Kubernetes batch workload** (Sprint 5, PR 2) — the foundation Job is
  promoted to the actual runnable workload:
  - The base `Job` now references the **real image the project builds**
    (`ml-pipeline`, name only) instead of a placeholder; the local overlay pins
    the tag to `ml-pipeline:local` (produced by `docker build -t ml-pipeline:local`
    and docker-compose). No registry path is invented — none exists yet.
  - A finite-run lifecycle appropriate for a batch pipeline: `restartPolicy: Never`,
    `backoffLimit: 2`, and a new `activeDeadlineSeconds: 1800` wall-clock safety
    ceiling (a completion-semantics guard against a stuck run — **not** a
    CPU/memory limit, which stays deferred to PR 5).
  - A local **runbook** in [`k8s/README.md`](k8s/README.md): build the image,
    side-load it (`kind load` / `minikube image load`), `kubectl apply -k`,
    inspect (`get`/`describe`), fetch logs, and delete/re-run.
  - Validated offline: `kustomize build` renders base and overlay, manifests parse
    as YAML, and field/scope assertions confirm the rendered image, command, and
    lifecycle fields — and that **no** deferred fields (resources, securityContext,
    env, volumes, ServiceAccount) leaked in.
  - **Executed on a local cluster** (2026-08-12, Docker Desktop Kubernetes
    v1.34.3): `kubectl apply -k k8s/overlays/local` created the namespace + Job,
    the local image resolved with no registry pull, and the Job ran its designed
    lifecycle — 3 attempts (initial pod + `backoffLimit: 2`), each a fresh pod
    (`restartPolicy: Never`), then `BackoffLimitExceeded` → terminal `Failed`. The
    pipeline does **not** complete: `dvc repro` aborts with `/app is not a git
    repository` (the image has no SCM). A *green* in-cluster run is PR 3 scope —
    an SCM (`git init`/`core.no_scm`), a mounted dataset, and credentials. The Job
    *mechanism* is proven; a *green pipeline run* is not claimed.
- **Kubernetes configuration, secrets & workload identity** (Sprint 5, PR 3) —
  externalized config and a least-privilege identity, wired into the Job, with no
  credentials committed:
  - A **`ConfigMap`** (`mlops-pipeline-config`) for the non-secret runtime config
    the code actually reads — `LOG_LEVEL` (`src/logging_config.py`) and
    `MLFLOW_TRACKING_URI` (`require_env` in `src/train.py`/`src/evaluate.py`). The
    URI is a public endpoint (the same host already committed as the DVC S3 remote
    in `.dvc/config`), not a credential.
  - A **`Secret` template** ([`k8s/base/secret.example.yaml`](k8s/base/secret.example.yaml))
    for the sensitive values — `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`
    — with placeholders only. It is **excluded from the Kustomize base**, so no
    render or apply can emit it; the real Secret is created out-of-band from a
    git-ignored `.env` (`kubectl create secret … --from-env-file`). **No credentials
    are committed.**
  - An explicit least-privilege **`ServiceAccount`** (`mlops-pipeline`) with
    **`automountServiceAccountToken: false`** (on both the account and the pod): the
    pipeline never calls the Kubernetes API, so no API token is mounted and **no
    `Role`/`RoleBinding`** is granted.
  - The `Job` now sets `serviceAccountName`, disables the token automount, and pulls
    config/credentials via `envFrom` (the ConfigMap unconditionally; the Secret as
    `optional: true` so `apply -k` works before the Secret exists).
  - Validated: base and overlay render via Kustomize; field assertions confirm the
    ConfigMap holds no credential keys, the Secret template is not emitted, and env
    names match the app. Verified on a local Docker Desktop cluster — the applied
    pod carried the ServiceAccount with an empty `volumes`/`volumeMounts` (no API
    token projected) and started with the optional Secret absent, failing only at
    the pre-existing SCM blocker. Existing tests unchanged (100 passed, 1 skip). No
    application behavior changed.
- **Kubernetes workload security hardening** (Sprint 5, PR 4) — an enforced
  `securityContext` on the pipeline `Job`, designed against the real image rather
  than copy-pasted:
  - **Pod level:** `runAsNonRoot: true` with an **explicit** `runAsUser`/`runAsGroup`
    `10001` — required, not cosmetic, because the image's `USER` is the *name*
    `appuser` (which the kubelet cannot verify as non-root, so `runAsNonRoot` alone
    would reject the pod) — and `seccompProfile.type: RuntimeDefault`.
  - **Container level:** `allowPrivilegeEscalation: false` (verified `NoNewPrivs: 1`)
    and `capabilities.drop: [ALL]` (the pipeline needs no Linux capabilities).
  - **`readOnlyRootFilesystem` is set explicitly to `false` and deferred with
    evidence:** `dvc repro` mutates DVC state in-tree at the `/app` repo root (its
    first write under a read-only root fails with
    `[Errno 30] Read-only file system: '/app/.dvc/tmp'`; it also rewrites
    `dvc.lock`, `.dvc/cache`, and needs a writable `.git`), so enabling it now would
    break the workload *earlier* than the pre-existing SCM blocker. Deferred to the
    green-in-cluster work per [ADR-010](docs/decisions/ADR-010-kubernetes-security-hardening.md).
  - [ADR-010](docs/decisions/ADR-010-kubernetes-security-hardening.md) records the
    decision, the pod-vs-container scoping, the read-only-root deferral, and why
    RBAC / `NetworkPolicy` are intentionally out of scope.
  - Validated: 21 rendered-manifest assertions (fields present, correct scope, no
    `privileged`/host-namespace footguns); `docker run` probes on the real image
    (imports clean under dropped caps + no-new-privs; behaviour-neutral — reaches
    the *same* SCM blocker; read-only-root failure reproduced); and a live Docker
    Desktop cluster **admitted** the Job and enforced the exact context (pod ran and
    terminated at the same blocker, token automount still off). **Restricted Pod
    Security Standard compliance is not claimed.** Tests unchanged (100 passed,
    1 skip); no application behavior changed.

### Changed

- Documentation updated to reflect the Kubernetes foundation: [roadmap](docs/roadmap.md)
  v4 marked in progress, [architecture](docs/architecture.md) notes the workload
  model, the [ADR index](docs/decisions/README.md) and
  [diagrams index](docs/diagrams/README.md) list the new records. No application
  logic changed; existing tests are unaffected.

## [1.3.1] - 2026-08-09

Proof Hardening — close the three limitations 1.3.0 documented rather than hid,
so the repository can *demonstrate* what it previously only *claimed*: evaluation
that is genuinely out-of-sample, reproducible execution proven by a real
`dvc repro`, and a type contract enforced on every pull request (not just
locally). No new pipeline capability — the goal is to make the existing
guarantees checkable.

### Added

- A dedicated **`split` stage** (`src/split.py`) that partitions the processed
  dataset into a training set (`data/processed/train.csv`) and a **held-out**
  evaluation set (`data/processed/test.csv`) with a stratified, seeded
  `train_test_split`. It is the single owner of both partitions and asserts they
  are **disjoint** (no row trains and evaluates) and **exhaustive** (no row is
  lost); `split.random_state` makes the exact held-out rows reproducible
  ([ADR-007](docs/decisions/ADR-007-held-out-evaluation.md)).
- [ADR-007](docs/decisions/ADR-007-held-out-evaluation.md) recording held-out
  evaluation as an engineering requirement.
- A self-contained **fixture DVC pipeline** (`tests/fixtures/pipeline/`) — the
  same `src/` stage code run against a small committed fixture dataset with the
  MLflow boundary stubbed offline (`_run_stage.py`) — carrying a **committed
  `dvc.lock`** so `declared pipeline + params + inputs + code = reproducible
  execution` is a checkable artifact, not a claim (resolves deviation D7,
  `dvc.lock`).
- [ADR-008](docs/decisions/ADR-008-fixture-reproducibility.md) recording the
  fixture-reproducibility strategy.
- `tests/contract/test_fixture_lock_contract.py`: a contract test asserting the
  committed fixture `dvc.lock` stays consistent with the fixture pipeline
  definition.

### Changed

- **Evaluation is now out-of-sample.** `train` fits only on the training
  partition (`data/processed/train.csv`) and `evaluate` scores the disjoint
  held-out partition (`data/processed/test.csv`), so the reported accuracy is a
  genuine generalization figure. The DVC lineage becomes
  `raw → preprocess → processed → split → {train.csv → train → model, test.csv} → evaluate → metrics`.
- The test suite grew from **84 to 101 tests** across the four tiers (smoke,
  unit, integration, contract), adding coverage for the `split` stage, the
  held-out boundary, and the fixture lock contract. All tiers remain
  deterministic and offline.

### Fixed

- **In-sample evaluation eliminated (deviation D5).** Because `train` no longer
  sees the held-out rows, evaluation accuracy is no longer measured on data the
  model was fit on.

### CI

- New **mypy type-check gate** (`python -m mypy`) in the `quality` job. CI now
  runs the same strict `[tool.mypy]` configuration a developer runs locally and
  that pre-commit runs at commit time, making the type contract a binding
  server-side gate: a type regression can no longer reach `main` on green
  pre-commit alone. mypy was already installed (requirements-dev.txt), so this
  adds no dependency install.
- New **fixture pipeline reproduction** step: a real `dvc repro` over the
  fixture pipeline runs all stages from a clean checkout, asserts the workspace
  is up to date against its committed lock, then force-re-runs and requires
  **byte-identical** `model.pkl` and `metrics.json` — a deterministic-artifact
  check that proves reproducibility rather than asserting it
  ([ADR-008](docs/decisions/ADR-008-fixture-reproducibility.md)).

## [1.3.0] - 2026-08-06

Sprint 4 — Pipeline Correctness & Reproducibility: turn attention from the
infrastructure *around* the ML pipeline to the pipeline itself. Correct the DVC
dependency graph so it models the real lineage, make configuration consistent
across `dvc.yaml`/`params.yaml`/code, separate the ML computation from the MLflow
boundary so stage logic is unit-testable, seed the training run for
determinism, and enforce the pipeline contract automatically in CI. Documentation
is reconciled to the as-built pipeline, and remaining limitations (in-sample
evaluation, no committed `dvc.lock`, name-pinned dependencies) are documented
rather than hidden.

### Added

- `src/tracking.py` — the single MLflow experiment-tracking boundary. All
  MLflow calls (and therefore every tracking network interaction) go through it;
  the stages import it lazily at the tracking boundary, so importing or
  unit-testing a stage requires neither MLflow nor credentials
  ([ADR-006](docs/decisions/ADR-006-pipeline-reproducibility.md) decision 4).
- A DVC-tracked **metrics artifact**: `evaluate` now writes
  `metrics/metrics.json`, declared under `metrics:` (`cache: false`) in
  `dvc.yaml` — accuracy is a first-class, versioned output, not only an MLflow
  entry and a log line (resolves deviation D4).
- Contract test suite (`tests/contract/test_pipeline_contract.py`) and a new
  `contract` pytest marker: eight pure-parsing checks that assert
  `dvc.yaml`/`params.yaml`/`src` agree with the pipeline contract (parameter
  consistency, no orphaned params, single-owner artifacts, the declared
  lineage `raw → preprocess → processed → train → model → evaluate → metrics`,
  and an acyclic graph). No data, network, or credentials.
- End-to-end integration test (`tests/integration/test_pipeline.py`): runs
  `preprocess → train → evaluate` through real temp files with MLflow stubbed,
  proving each stage's output is consumable by the next.
- Stage-level unit tests for `preprocess`, `train`, and `evaluate`
  (`tests/unit/`), exercising the extracted pure-compute functions without any
  external service.
- Pipeline contract ([docs/pipeline-contract.md](docs/pipeline-contract.md)) and
  [ADR-006](docs/decisions/ADR-006-pipeline-reproducibility.md) recording
  reproducibility and stage contracts as engineering requirements.
- Sprint 4 release documents: the
  [final engineering review](docs/reviews/sprint-04-final-review.md), the
  [retrospective](docs/retrospectives/sprint-04-retrospective.md), and the
  [proof-impact assessment](docs/proof/sprint-04-proof-impact.md).

### Changed

- **Corrected the DVC pipeline graph** (`dvc.yaml`): `train` now depends on the
  processed dataset (`data/processed/data.csv`) instead of the raw file, and
  `evaluate` depends on the processed dataset **and** the model. The graph is now
  the single linear chain the architecture always described.
- **Reconciled the parameter contract** (`params.yaml`): `train` uses
  `input`/`output`/`target`/`random_state`/`n_estimators`/`max_depth`; the
  evaluation section was renamed `test:` → `evaluate:` with explicit
  `data`/`model`/`target`/`metrics`. `dvc.yaml` param keys and the code now
  reference the same authoritative names, with no orphaned parameters.
- **Refactored the ML stages into separable concerns** (`train.py`,
  `evaluate.py`): a pure ML computation (`run_training` / `compute_metrics`) that
  performs no IO and imports no MLflow, artifact persistence via `pipeline_io`,
  and the lazily-imported `tracking` boundary. Existing MLflow behavior
  (metrics, params, artifacts, conditional model registration) is preserved, not
  removed, to make testing easier.
- `preprocess` now writes the processed CSV **with its header row**
  (`header=True`), so downstream stages can select `Outcome`/feature columns by
  name — the prerequisite for `train` consuming the processed dataset (resolves
  deviation D8).

### Fixed

- **Preprocess output is now consumed downstream** — the orphaned
  `data/processed/data.csv` produced by `preprocess` is `train`'s input
  (deviation D1).
- **Configuration drift eliminated** — `dvc.yaml` no longer references
  `train.data`/`train.model` (which did not exist in `params.yaml`), and the
  `evaluate` stage's parameters are declared in the graph (deviations D2, D3).
- **Training is now deterministic** — `train_test_split` and the
  `RandomForestClassifier` are seeded from `train.random_state`, and the
  configured `n_estimators`/`max_depth` are applied to the estimator instead of
  being inert (deviation D7, seeding portion).

### CI

- New `quality`-job step **"DVC pipeline integrity (graph + status, offline)"**:
  runs `dvc dag` (parseable, acyclic graph) and local `dvc status`, with
  `DVC_NO_ANALYTICS=true` so the step makes no network call and never touches the
  DagsHub remote. `dvc repro --dry` is deliberately not used — it requires the
  remote-only raw dataset; the guarantees it would give are enforced offline by
  the contract tests. `dvc` was already installed (a runtime dependency), so this
  adds no new install.
- The contract tests run as part of the existing `pytest` step, so a broken
  stage contract, an inconsistent parameter, or a mis-wired lineage fails a pull
  request. Sprint 3's `docker` job (build + non-root/imports/entrypoint
  validation) and least-privilege `contents: read` permissions are unchanged.

### Testing

- The suite grew to **84 tests** across four tiers: `smoke` (import/wiring),
  `unit` (isolated component and stage-compute tests), `integration` (full
  three-stage run, MLflow stubbed), and `contract` (static
  `dvc.yaml`/`params.yaml`/`src` consistency). All tiers are deterministic and
  run offline; unit tests require no live MLflow, network, or credentials.
- A `stub_tracking` fixture swaps the lazily-imported `tracking` module for an
  in-memory recorder, so stage read → compute → persist paths run end-to-end in
  tests without importing MLflow.

### Documentation

- Reconciled [docs/pipeline-contract.md](docs/pipeline-contract.md) from a
  design contract (CURRENT vs TARGET) to the **as-built** pipeline, with the
  remaining deviations (D5 in-sample evaluation, D7 `dvc.lock`) called out
  explicitly.
- Updated [architecture.md](docs/architecture.md),
  [project-structure.md](docs/project-structure.md), and
  [roadmap.md](docs/roadmap.md) to match the corrected pipeline, the new
  `tracking.py` module, the four-tier test layout, and the CI pipeline-integrity
  step.
- Rewrote the stale sections of the root [README.md](README.md): the training
  stage no longer claims to read raw data or grid-search all hyperparameters, the
  evaluation description reflects the metrics artifact and the in-sample boundary,
  and the DVC-stage snippets match the corrected `dvc.yaml`.

## [1.2.0] - 2026-08-05

Sprint 3 — Containerization & Continuous Integration: make the pipeline portable
and self-validating. Ship a production-grade container image and a Compose-based
development workflow, and add a GitHub Actions CI pipeline that lints, tests, and
builds the image on every push and pull request.

### Added

- Production-grade, multi-stage `Dockerfile` with three named targets from a
  single source of truth — `builder` (dependency compilation into an isolated
  virtualenv), `development` (builder + Ruff/mypy/pytest/pre-commit toolchain),
  and `runtime` (lean, non-root production image, the default target) — built on
  `python:3.12-slim-bookworm` with BuildKit cache mounts and OCI provenance
  labels.
- `.dockerignore` that keeps data, models, credentials, and local tooling out of
  the build context and image layers.
- `docker-compose.yml` development workflow: a bind-mounted `dev` service for the
  inner loop and an on-demand `pipeline` profile that runs the production image,
  plus `.env.example` for MLflow / DagsHub credentials.
- Continuous integration pipeline (`.github/workflows/ci.yml`): a `quality` job
  (Ruff lint + format check, pytest) gating a `docker` job that builds the
  `runtime` image and validates it (non-root UID 10001, core imports resolve,
  DVC entrypoint present).
- CI status badge on the root `README.md`.
- ADR-005 recording the containerization strategy (Docker/OCI, multi-stage
  build, `slim` base, non-root, twelve-factor config, externalized state).

### Changed

- ADR-005 status moved from "Accepted (design only — not yet implemented)" to
  "Accepted", with scope and consequences updated to reflect that the design was
  implemented in Sprint 3.
- Documentation refreshed for Sprint 3 (architecture, roadmap, project structure,
  and the documentation index) to describe the container image, Compose workflow,
  and CI pipeline.

### Security

- Production image runs as a dedicated non-root user (UID/GID 10001) with a
  `nologin` shell; build toolchain and compilers are confined to the `builder`
  stage and never reach runtime.
- Secrets and data are injected at run time (via `--env-file` / mounted volumes),
  never baked into image layers; base image pinned by codename
  (`python:3.12-slim-bookworm`), with digest pinning recorded as a follow-up.
- CI workflow granted least-privilege `contents: read` only, structurally
  preventing it from pushing images or writing to the repository.

### CI

- CI is validation only — it lints, tests, and builds/validates the image on
  every push to `main` and every pull request; it does not deploy, publish
  images, or use Kubernetes (continuous delivery is deferred to Roadmap v3+).
- Concurrency control cancels superseded in-flight runs per ref; pip and
  BuildKit (`type=gha`) layer caches speed repeat runs.

### Documentation

- `docs/containerization.md` — containerization strategy and as-built
  build/run instructions (including hardened, read-only execution).
- `docs/docker-development.md` — day-to-day Docker Compose development workflow.
- `docs/ci-cd.md` — CI stages, failure strategy, local reproduction, and the
  future continuous-delivery roadmap.

## [1.1.0] - 2026-08-02

Sprint 2 — Engineering Excellence: raise the baseline pipeline to a maintainable,
professionally engineered codebase (logging, error handling, typing, tests, and
a quality toolchain), driven by a principal-engineer production-readiness review.

### Added

- Principal-engineer production-readiness review
  (`docs/reviews/sprint-02-engineering-review.md`) whose findings (H-1..H-6)
  drove the Sprint 2 engineering-excellence work.
- Centralized logging framework: `src/logging_config.py` (console + rotating
  file handlers, `LOG_LEVEL`/`LOG_DIR` environment control) replacing `print()`
  across all pipeline stages, with `docs/logging.md` documenting the strategy
  (review finding H-1).
- Standardized exception handling: a typed hierarchy in `src/exceptions.py`
  (`PipelineError` → `ConfigError`, `DataError`, `ModelError`,
  `TrackingError`), centralized IO/config/serialization boundaries in
  `src/pipeline_io.py`, a uniform stage entry point in `src/stage_runner.py`
  (log once, exit non-zero), and `docs/exception-strategy.md` (review finding
  H-2).
- Complete type annotations across `src/` with a strict mypy configuration in
  `pyproject.toml`, documented in `docs/type-safety.md`.
- Testing foundation: a `pytest` suite under `tests/` (smoke and unit tests)
  with shared fixtures (`tests/conftest.py`) and configuration in
  `pyproject.toml`; `pytest`/`pytest-cov` added to `requirements-dev.txt`; and
  `docs/testing-strategy.md` documenting the philosophy, layout, and roadmap
  (review finding H-3).
- Developer experience tooling: Ruff linter and formatter (configured in
  `pyproject.toml`), a `.pre-commit-config.yaml` running Ruff, file-hygiene
  checks, mypy, and (at push time) the test suite; a `Makefile` with helpful
  development commands (`make help`); VS Code workspace settings and recommended
  extensions under `.vscode/`; `ruff`/`pre-commit` added to
  `requirements-dev.txt`; and `docs/developer-guide.md` documenting local
  development, formatting, linting, testing, and the pre-commit workflow.
- ADR-004 recording the Python quality toolchain decision (Ruff, mypy, pytest,
  pre-commit).
- Sprint 2 final engineering-validation review
  (`docs/reviews/sprint-02-final-review.md`).

### Changed

- Pipeline stage scripts (`preprocess.py`, `train.py`, `evaluate.py`)
  refactored for organization and readability: corrected import grouping,
  removed redundant intermediates, and reconciled stale docstrings.
- Core documentation updated to reflect the Sprint 2 engineering work:
  `docs/architecture.md` (shared infrastructure modules, expanded technology
  table), `docs/roadmap.md` (v2 delivered vs. remaining scope),
  `docs/project-structure.md` (new modules, `tests/`, tooling files), and
  `docs/design-principles.md` (logging, exceptions, typing, testing, and
  toolchain rationale).
- Reconciled previously stale documentation with the delivered work:
  `docs/philosophy.md` and `docs/decisions/ADR-001-repository-structure.md`
  (testing and tooling now exist, delivered without a package layout), and
  fixed broken in-page anchors and obsolete release/versioning notes surfaced
  during final validation.

## [1.0.0] - 2026-08-01

Sprint 1 — Professional Repository Transformation: establish repository
governance, engineering documentation, and the ADR framework on top of the
foundation pipeline.

### Added

- Documentation scaffolding under `docs/` (architecture, roadmap, project
  structure, and architecture decision records).
- Placeholder directories for diagrams and screenshots.
- Repository hygiene files: `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `CHANGELOG.md`, and `.editorconfig`.
- First drafts of the core documentation: expanded `docs/architecture.md`,
  `docs/roadmap.md`, and `docs/project-structure.md`; first drafts of ADR-001,
  ADR-002, and ADR-003; and a new `docs/philosophy.md` describing engineering
  principles.
- Repository governance and GitHub metadata: issue templates (bug, feature,
  documentation) and PR template under `.github/`; `SECURITY.md` and
  `SUPPORT.md`; and documentation for the GitHub workflow, semantic versioning,
  release checklist, repository metadata recommendations, and a documentation
  index (`docs/README.md`).
- `docs/design-principles.md` explaining the rationale behind core design and
  technology choices (batch pipeline, Random Forest, Python, DVC, MLflow,
  modular code, YAML configuration).

### Changed

- Roadmap v1 renamed from "Course Implementation" to "Foundation Release"; v5
  expanded to "Production Cloud Platform" and v6 objectives broadened.
- ADR-001/002/003 finalized (status Accepted, dated) with a more confident
  engineering voice and no placeholder markers.
- Recommended repository description updated to
  "Production-Oriented MLOps Pipeline using DVC, MLflow and Python".

[Unreleased]: https://github.com/asadhanif3188/MLOps-Project-01-End-to-End-Machine-Learning-Pipeline/compare/v1.3.1...HEAD
[1.3.1]: https://github.com/asadhanif3188/MLOps-Project-01-End-to-End-Machine-Learning-Pipeline/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/asadhanif3188/MLOps-Project-01-End-to-End-Machine-Learning-Pipeline/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/asadhanif3188/MLOps-Project-01-End-to-End-Machine-Learning-Pipeline/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/asadhanif3188/MLOps-Project-01-End-to-End-Machine-Learning-Pipeline/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/asadhanif3188/MLOps-Project-01-End-to-End-Machine-Learning-Pipeline/releases/tag/v1.0.0
