# Security Policy

## Supported Versions

This project is under active development. Security fixes are applied to the
latest released version and the `main` branch.

| Version | Supported |
|---------|-----------|
| Latest release | ✅ |
| `main` (development) | ✅ |
| Older releases | ❌ |

> <!-- TODO: refine this table once tagged releases exist (see docs/versioning.md). -->

## Reporting a Vulnerability

Please **do not report security vulnerabilities through public GitHub issues.**

Instead, report them privately using one of the following:

- GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
  ("Report a vulnerability" under the repository's **Security** tab), or
- <!-- TODO: add a dedicated security contact email if one is available. -->

When reporting, please include:

- A description of the vulnerability and its potential impact.
- Steps to reproduce, or a proof of concept.
- Any relevant environment details (OS, Python version, dependency versions).

## Responsible Disclosure

We ask that you:

- Give us a reasonable opportunity to investigate and address the issue before
  any public disclosure.
- Avoid accessing, modifying, or deleting data that does not belong to you.
- Act in good faith and avoid privacy violations, service disruption, or data
  destruction.

We will:

- Acknowledge your report as soon as reasonably possible.
- Keep you informed of progress toward a fix.
- Credit reporters who wish to be acknowledged, once the issue is resolved.

> <!-- TODO: define concrete response/acknowledgement time targets when the
> project's maintenance cadence is established. -->

## Security Best Practices

Contributors and users should follow these practices:

- **Never commit secrets.** Credentials (e.g. `MLFLOW_TRACKING_URI`,
  `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`) belong in a local
  `.env` file, which is git-ignored. Only `.env.example` (a template) is
  committed.
- **Rotate credentials** if they are ever exposed, and remove them from history.
- **Review dependencies.** Keep `requirements.txt` up to date and monitor for
  known vulnerabilities.
- **Container image vulnerability scanning.** Both shipped images — the
  `mlops-pipeline` runtime image and the `mlflow-server` image layered on it — are
  scanned with **Trivy** on every push/PR in the `docker` CI job, over their OS **and**
  Python packages. The gate fails on **fixable** HIGH/CRITICAL vulnerabilities (a
  patched version exists, so the fix is actionable: rebuild on a patched base or bump
  the package); **non-fixable** HIGH/CRITICAL are **reported, not muted**, and
  auto-promote to the gate the moment an upstream fix ships — so this is **not** a
  blanket ignore of HIGH/CRITICAL. Justified, **time-boxed** exceptions (a specific CVE
  id + rationale + `expired_at`, auto-expired by Trivy) live in
  [`.trivyignore.yaml`](.trivyignore.yaml); there is no blanket severity mute. The scan
  runs on the locally-built images — **never pulled from a registry** — so ordinary PR
  CI stays credential-free and AWS-independent, and it **complements** (does not
  replace) ECR `scan_on_push` ([ADR-021](docs/decisions/ADR-021-terraform-managed-ecr.md)),
  the registry-side layer. Design of record:
  [ADR-035](docs/decisions/ADR-035-container-image-scanning.md) /
  [docs/container-image-scanning.md](docs/container-image-scanning.md).
- **SBOM & image provenance.** Every build emits a **CycloneDX SBOM** of the image
  (Trivy, in the `docker` CI job) and CI **asserts the git→image binding** — the built
  image's `org.opencontainers.image.revision` label must equal the commit SHA. At
  release, [`scripts/release-image.sh`](scripts/release-image.sh) captures the immutable
  ECR **sha256 digest** (cross-checked against `aws ecr describe-images`) and records the
  **git commit → image tag → digest** chain; the deploy can be **pinned by digest**
  (opt-in) and [`scripts/verify-deployed-digest.sh`](scripts/verify-deployed-digest.sh)
  confirms the **running** workload matches. **Image signing (cosign)** is available as
  an opt-in keyless step, not a mandatory gate (rationale documented). The SBOM is a CI
  artifact, never committed. Design of record:
  [ADR-036](docs/decisions/ADR-036-sbom-and-image-provenance.md) /
  [docs/supply-chain-provenance.md](docs/supply-chain-provenance.md).
- **Least privilege.** Use scoped tokens for the DagsHub/MLflow and DVC remotes
  rather than broad credentials.
- **Validate data sources.** Treat external datasets and artifacts as untrusted
  input.
- **Infrastructure credentials & IAM.** The Terraform ([`terraform/`](terraform/))
  AWS platform never stores static AWS credentials — no access keys or secret
  keys are committed; Terraform authenticates via the standard AWS credential
  chain. AWS IAM roles are least-privilege and dedicated to purpose (see
  [ADR-016](docs/decisions/ADR-016-aws-iam-foundation.md)): permissions come from
  the AWS-managed policies EKS requires, with no `AdministratorAccess` and no
  project-authored wildcard. **The Amazon VPC CNI runs under its own dedicated IAM
  role** — assumed only by the `aws-node` service account via **EKS Pod Identity**
  — rather than the worker-node instance profile (finding **M-01**): `AmazonEKS_CNI_Policy`
  is off the node role, so a pod on the node can no longer reach the CNI's
  ENI-manipulation permissions through IMDS. Pinned by an offline `terraform test`
  suite ([ADR-024](docs/decisions/ADR-024-vpc-cni-pod-identity.md)). **Cluster
  access uses explicit EKS access entries**,
  not automatic cluster-creator admin (finding **H-03**): the principal that runs
  `apply` receives **no** implicit cluster-admin, `authentication_mode` defaults to
  `API` (access entries only), and each identity is granted a **scoped** AWS-managed
  EKS access policy (default `AmazonEKSAdminPolicy`, never cluster-admin for
  convenience) via a `cluster_access_entries` map populated from a **git-ignored**
  `terraform.tfvars` — no personal ARNs or account IDs are committed. Re-enabling the
  old creator-admin bootstrap is **rejected** by validation and pinned by an offline
  `terraform test` suite ([ADR-023](docs/decisions/ADR-023-eks-access-control.md)).
  State files and kubeconfigs are git-ignored and
  never committed. **CI holds no AWS credentials or cloud identity**: the
  `terraform-validate` job validates the IaC *statically* (`fmt`/`init
  -backend=false`/`validate`, an offline `terraform test` contract suite, TFLint,
  Trivy) and never runs `terraform plan`/`apply` — real provisioning is a
  deliberate, operator-driven step against
  one's own account (see [ADR-019](docs/decisions/ADR-019-terraform-ci-validation.md)).
- **Kubernetes Secret encryption at rest (finding M-02).** EKS Kubernetes
  **Secrets are envelope-encrypted with a dedicated, customer-managed KMS key**
  (`terraform/kms.tf`) wired into the cluster's `encryption_config` — a
  customer-controlled layer on top of the AWS-owned etcd default, auditable in
  CloudTrail and revocable via its key policy. The key has **automatic rotation
  enabled** and a **least-privilege key policy** (the account-root administration
  statement plus an explicit use-grant to the EKS cluster role with a
  `GrantIsForAWSResource`-constrained `CreateGrant` — **no bare `"*"` principal and
  no project-authored `kms:*` use grant**); permissions are granted through the key
  policy, so no additional IAM policy widens the surface. Encryption is
  **unconditional — there is no toggle to disable it** — and the previously
  documented `AVD-AWS-0039` Trivy suppression has been **removed**. Pinned by an
  offline `terraform test` suite ([ADR-025](docs/decisions/ADR-025-eks-secrets-kms-encryption.md)).
- **Ephemeral cloud environment & verified teardown.** The AWS/EKS environment is
  **short-lived** — provisioned to capture evidence, then **destroyed and verified
  clean** (Terraform state empty + AWS API shows no cluster/NAT + ECR repo absent),
  never assumed clean from the `destroy` line alone. The EKS API server is **private
  by default** (finding **H-02**): public access is off out of the box, and enabling
  it is an explicit opt-in that must be scoped to the operator's own IP/CIDR — an
  unrestricted `0.0.0.0/0` is **rejected** by Terraform validation and cluster
  preconditions, and pinned by an offline `terraform test` contract suite
  ([ADR-022](docs/decisions/ADR-022-eks-secure-api-access.md)). No AWS account
  identifier is committed (the ECR image reference stays a `000000000000`
  placeholder). The full
  lifecycle, cost drivers, and teardown procedure are in
  [docs/cloud-operations.md](docs/cloud-operations.md)
  ([ADR-020](docs/decisions/ADR-020-cloud-lifecycle-cost-control.md)).

---

## Related Documentation

- [Support](SUPPORT.md)
- [Contributing](CONTRIBUTING.md)
- [Engineering Philosophy](docs/philosophy.md)
