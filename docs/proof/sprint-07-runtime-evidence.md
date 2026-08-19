# Sprint 7 — Runtime Evidence (PR 10, Cloud-Native MLOps E2E)

**Executed 2026-08-19** against a real, Terraform-provisioned **Amazon EKS**
cluster in the operator's **own** AWS account (`us-east-1`). This is the Sprint 7
runtime **proof gate**: the full hardened platform — Terraform-managed **ECR**,
**KMS**-encrypted **S3** dataset + artifact stores, **EKS Pod Identity** workload
identity, and the **in-cluster MLflow** tracking platform (PostgreSQL + S3) — was
provisioned from scratch, the MLOps `Job` run to completion, and every claim
verified on the **live** cluster, then the environment was destroyed.

> **Scope & honesty.** This is a short-lived **validation** run, not a production
> deployment. Account identifiers, the operator IP, KMS key IDs, and all
> credentials/secrets are deliberately redacted (`<ACCOUNT_ID>`, `<OPERATOR_IP>`,
> `<KEY_ID>`); no kubeconfig, token, password, or secret material is recorded here.
> Two failures and their fixes, plus the deliberate residual limitations, are
> documented in [§ Failures & fixes](#failures--fixes) and [§ Limitations](#limitations)
> — nothing is hidden. Unlike the Sprint 6 run (which used an offline MLflow file
> store), this run exercised the **real in-cluster MLflow tracking server** end to
> end, logging to PostgreSQL with artifacts in S3.

**Runbook:** [docs/cloud-operations.md](../cloud-operations.md) ·
**Design of record:** ADR-021 (ECR), ADR-022–025 (EKS API/access/CNI/KMS),
ADR-026 (in-cluster MLflow), ADR-027 (S3 dataset).

---

## Result summary

| Dimension | Result |
|---|---|
| **Prerequisites** | Sprint 7 PRs 1–9 merged to `main` (PRs #44–#52); clean-slate account (0 prior resources) |
| **Infrastructure (Terraform)** | `apply` **63 added, 0 changed, 0 destroyed**; +2 added / 1 changed for operator access; `state list` = 69 entries |
| **EKS** | cluster `mlops-pipeline-dev-eks` **ACTIVE**, control plane **v1.35.6-eks** (`eks.21`), 1 node **Ready** in a **private** subnet |
| **ECR** | 2 images pushed, immutable tags: `mlops-pipeline:1.3.1`, `mlflow-server:0.1.0` |
| **Workload identity** | 4 **EKS Pod Identity** associations live; **no static AWS keys** on the cluster |
| **S3 dataset** | object retrieved in-cluster via Pod Identity, **sha256 == pinned identity** |
| **In-cluster MLflow** | PostgreSQL-backed tracking server **Ready**, artifacts proxied to **SSE-KMS S3** |
| **MLOps pipeline** | Job **Complete**, pod **Succeeded**, **exit 0**, all 4 DVC stages ran |
| **MLflow run** | experiment `mlops-pipeline`, 2 runs `FINISHED`, metrics + artifacts persisted |
| **Security (live pod)** | non-root uid 10001, seccomp `RuntimeDefault`, caps drop ALL, no API token; EKS Secrets KMS-encrypted |
| **Teardown** | see [§ 15 Teardown](#15-teardown) — real `terraform destroy` result, verified clean |

---

## 0. Environment & prerequisites

- **Date:** 2026-08-19 · **Region:** `us-east-1` · **Account:** `<ACCOUNT_ID>` (operator's own; `aws sts get-caller-identity` → `user/terraform`, **not** any client account).
- **Tooling:** Terraform v1.15.8, AWS CLI v2.35, kubectl v1.34 (client) / v1.35 (server), Docker 29.3 (Linux engine). `eksctl`/`helm` not required (Terraform + Kustomize).
- **Build provenance:** repo at commit `4f85974` (HEAD of `main` with Sprint 7 PRs 1–9 merged: #44 ECR, #45 API-security, #46 access-control, #47 CNI-identity, #48 KMS, #49 in-cluster MLflow, #50 MLflow-pipeline, #51 S3-dataset, #52 DVC-dataflow).
- **Starting state:** `terraform state list` empty, `aws eks list-clusters` empty, `aws ecr describe-repositories` empty — provisioned entirely from scratch for this run.

---

## 1. Terraform validation & apply

```
$ terraform validate            -> Success! The configuration is valid.
$ terraform plan  -out=tfplan    -> Plan: 63 to add, 0 to change, 0 to destroy.
$ terraform apply tfplan         -> Apply complete! Resources: 63 added, 0 changed, 0 destroyed.
```

**State summary (no secrets)** — `terraform state list` = **69 entries** (65 managed
resources + data sources). Provisioned, by group:

| Group | Resources |
|---|---|
| **Network** | VPC `10.0.0.0/16`, 2 public + 2 private subnets across 2 AZs, IGW, **1 NAT** + 1 EIP, route tables |
| **IAM** | 6 roles (EKS cluster, node, VPC-CNI, EBS-CSI, MLflow-S3, dataset-reader) + AWS-managed policy attachments |
| **EKS** | cluster (K8s **1.35**), 1-node managed group, 5 addons (`coredns`, `kube-proxy`, `vpc-cni`, `eks-pod-identity-agent`, `aws-ebs-csi-driver`) |
| **ECR** | 2 repositories (`mlops-pipeline`, `mlflow-server`) + lifecycle policies, **immutable tags** |
| **KMS** | 3 customer-managed keys + aliases (`eks-secrets`, `datasets`, `mlflow-artifacts`) |
| **S3** | 2 buckets (`datasets`, `mlflow-artifacts`) — versioned, SSE-KMS, all public access blocked |
| **Pod Identity** | 4 associations (VPC-CNI, EBS-CSI, MLflow-S3, dataset-reader) |

A second `apply` (**2 added, 1 changed**) enabled the documented operator access
path — see [§ Failures & fixes](#failures--fixes). Terraform state is the local
default backend (`terraform/terraform.tfstate`, git-ignored, never committed).

---

## 2. EKS verification

```
$ aws eks describe-cluster --name mlops-pipeline-dev-eks --query cluster.status  -> ACTIVE
$ ... --query 'cluster.{version:version,platform:platformVersion}'              -> 1.35 / eks.21
$ kubectl version --short | grep Server                                          -> v1.35.6-eks-bca9cf6
$ kubectl get nodes -o wide
NAME                        STATUS   VERSION               INTERNAL-IP   EXTERNAL-IP   OS-IMAGE
ip-10-0-9-93.ec2.internal   Ready    v1.35.6-eks-254016e   10.0.9.93     <none>        Amazon Linux 2023
```

- Control plane **ACTIVE**, Kubernetes **1.35** (platform `eks.21`).
- **1 node Ready**, `t3.medium`, `ON_DEMAND`, `AL2023_x86_64_STANDARD`, containerd 2.2.5.
- Node is in a **private subnet** (`10.0.9.93`, **no external IP**) — worker capacity is not internet-exposed.
- Access model: `authentication_mode = API` (access entries only), `bootstrap_cluster_creator_admin_permissions = false` (no implicit creator-admin — H-03).

**kube-system addons** (all `Running`): `aws-node` (VPC CNI) 2/2, `coredns` ×2,
`kube-proxy`, `eks-pod-identity-agent`, `ebs-csi-controller` 6/6 ×2, `ebs-csi-node`.

---

## 3. ECR image verification

Both images built `--platform linux/amd64` (matching the AL2023 x86_64 node),
`--provenance=false --sbom=false`, tagged with immutable versions (never `:latest`),
and pushed to the Terraform-managed repositories:

| Repository | Tag | Digest |
|---|---|---|
| `mlops-pipeline` | `1.3.1` | `sha256:fe4e5404b736f0f7f6f1e140def739a39e0b998061620d6ea3833b53c76f4c22` |
| `mlflow-server` | `0.1.0` | `sha256:91e9f5b99ab01ed468610365acb0f35eb0c577a00148336f3684d1f8f3812106` |

The `mlflow-server` image is built `FROM` the pipeline image (client/server MLflow
parity, ADR-026). Both repositories enforce **immutable tags**, so a pushed version
can never be repointed. Kubelet pulls via the node instance role
(`AmazonEC2ContainerRegistryReadOnly`) — no pod-level registry credential.

---

## 4. Workload identity (EKS Pod Identity)

`aws eks list-pod-identity-associations` — **4 live associations**, each binding a
service account to a dedicated least-privilege IAM role. **No static AWS keys exist
anywhere on the cluster**; every AWS call uses short-lived, pod-scoped credentials.

| Namespace / ServiceAccount | Role purpose |
|---|---|
| `kube-system` / `aws-node` | VPC CNI (isolated from node role — M-01) |
| `kube-system` / `ebs-csi-controller-sa` | EBS volume provisioning (Postgres PVC) |
| `mlops` / `mlflow-server` | MLflow → S3 artifact read/write |
| `mlops` / `mlops-pipeline` | **read-only** dataset access (M-04) |

Both application identities were **proven at runtime**: the `fetch-dataset` init
container downloaded the dataset (dataset-reader role), and the MLflow server wrote
artifacts to S3 (mlflow-server role) — see §§ 6, 13.

---

## 5. S3 dataset (runtime retrieval, integrity-pinned)

Uploaded out-of-band to the Terraform-provisioned, private, **versioned**,
**SSE-KMS** dataset bucket:

```
s3://<ACCOUNT_ID redacted bucket>/pima-indians-diabetes/v1/data.csv
  VersionId       : lE73sfq4KdjUANCbP_yAWjzJfIQS7nyi
  ContentLength   : 23872 bytes
  ServerSideEncryption : aws:kms  (alias/mlops-pipeline-dev-datasets)
  ChecksumSHA256  : 7lsMktWtRh6GFRxUSzt2vWJpxgUsXrYoxLBhigjP/Ik=
```

The S3 object's SHA-256 **matches the pinned dataset identity** in the base
ConfigMap (`DATASET_SHA256 = ee5b0c92…a08cffc89`; the hex pin base64-encodes to the
S3 checksum above — verified equal). The pipeline's `fetch-dataset` init container
independently re-verified this in-cluster (§ 6). Bucket posture: versioning
`Enabled`; `BlockPublicAcls / IgnorePublicAcls / BlockPublicPolicy /
RestrictPublicBuckets` all `true`.

---

## 6. In-cluster MLflow platform

Deployed via `kubectl apply -k k8s/overlays/aws` (Namespace, ServiceAccounts,
ConfigMaps, DVC no-SCM config, MLflow **StatefulSet** Postgres + **Deployment**
server + Services, and the pipeline **Job**). The DB credential Secret
(`mlflow-db-credentials`) was created **out-of-band** (never committed, password
generated at deploy time, not logged).

- **PostgreSQL** (`postgres:16.4`, StatefulSet) — `1/1 Running`; PVC
  `data-mlflow-postgres-0` **Bound** to a **1Gi gp2 EBS** volume (see the storage
  fix in [§ Failures & fixes](#failures--fixes)).
- **MLflow Tracking Server** — `1/1 Ready`; backend PostgreSQL, artifact store the
  real S3 bucket via `--artifacts-destination` + `--serve-artifacts` (clients upload
  through the server over `mlflow-artifacts:`; only the server holds S3 access).
- **Service** — `mlflow.mlops.svc.cluster.local:5000`, **ClusterIP** (internal only;
  never a public LoadBalancer/NodePort). The pipeline reads it via
  `MLFLOW_TRACKING_URI` from the base ConfigMap — no MLflow credentials needed.

Init-container proof (successful pod):

```
fetch-dataset  : Dataset checksum verified (sha256=ee5b0c92…a08cffc89); 23872 bytes
wait-for-mlflow: MLflow ready at http://mlflow.mlops.svc.cluster.local:5000/health
```

---

## 7–8. Job submission & DVC stage execution

`kubectl apply -k k8s/overlays/aws` created `job/mlops-pipeline`. The Job reached
**`Complete`** (`succeeded: 1, failed: 1` — the first pod's `wait-for-mlflow` init
timed out during MLflow's first-boot DB migration; the `backoffLimit` retry pod
`mlops-pipeline-g5fpx` ran to completion — the documented transient-fault semantics,
see [§ Failures & fixes](#failures--fixes)).

Every DVC stage was observed in the successful pod's logs
(`preprocess → split → train → evaluate`):

```
Running stage 'preprocess':  768 rows written to data/processed/data.csv
Running stage 'split':       614 train rows, 154 held-out rows
Running stage 'train':       GridSearchCV — 3 folds × 4 candidates (12 fits)
                             best params: {min_samples_leaf: 1, min_samples_split: 5}
                             Best model accuracy: 0.7398 ; model saved to models/model.pkl
                             Registered model 'Best Random Forest Classifier' version 1
Running stage 'evaluate':    model accuracy (held-out test): 0.7078
```

The held-out boundary held: `train` consumed only `train.csv`; `evaluate` scored
only the disjoint `test.csv`.

---

## 9–12. Training/evaluation, MLflow run, metrics

Queried from the live MLflow server (`/api/2.0/mlflow`, PostgreSQL-backed):

- **Experiment:** `mlops-pipeline` (id `1`), `artifact_location = mlflow-artifacts:/1`.
- **Runs (both `FINISHED`):**

| Run name | Run ID | Stage | Metric | Params |
|---|---|---|---|---|
| `crawling-lark-191` | `1fa7c4f3fb274864b663813c166c0472` | train | `accuracy = 0.7398` | `best_n_estimators=100, best_max_depth=5, best_samples_split=5, best_samples_leaf=1` |
| `skillful-gnu-461` | `268fddf4fc804ee7a414dcd01ce02222` | evaluate | `accuracy = 0.7078` (held-out) | — |

Metrics are persisted in the PostgreSQL backend (survive server pod recreation by
design — the server is stateless, all state in Postgres + S3).

---

## 13. Artifact evidence (S3, written via Pod Identity)

`aws s3 ls` on the SSE-KMS MLflow artifact bucket — **7 objects**, written by the
MLflow server using its Pod Identity role (no static keys):

```
artifacts/1/1fa7c4f3…/artifacts/classification_report.txt      326 B
artifacts/1/1fa7c4f3…/artifacts/confusion_matrix.txt            18 B
artifacts/1/models/m-a576e136…/artifacts/MLmodel             1347 B
artifacts/1/models/m-a576e136…/artifacts/conda.yaml           242 B
artifacts/1/models/m-a576e136…/artifacts/model.skops      2058950 B
artifacts/1/models/m-a576e136…/artifacts/python_env.yaml      115 B
artifacts/1/models/m-a576e136…/artifacts/requirements.txt     116 B
```

The registered model (`model.skops`, ~2 MB) plus its environment descriptors and the
train run's text artifacts are all present in the CMK-encrypted bucket.

---

## 14. Job completion & 15. pod security context

```
$ kubectl get job mlops-pipeline -o jsonpath conditions -> Complete=True (CompletionsReached)
$ kubectl get pod mlops-pipeline-g5fpx -o jsonpath      -> phase=Succeeded exit=0 reason=Completed
```

Security context enforced **live** by the EKS API server on the successful pod
(inherited verbatim from the hardened base — the AWS overlay weakens nothing):

| Scope | Field | Value |
|---|---|---|
| Pod | `runAsNonRoot` / `runAsUser` / `runAsGroup` / `fsGroup` | `true` / `10001` / `10001` / `10001` |
| Pod | `seccompProfile.type` | `RuntimeDefault` |
| Pod | `automountServiceAccountToken` | `false` (no Kubernetes API access) |
| Container | `allowPrivilegeEscalation` | `false` |
| Container | `capabilities.drop` | `[ALL]` |
| Container | `readOnlyRootFilesystem` | `false` (deliberate — DVC writes in-tree; ADR-010, see [§ Limitations](#limitations)) |
| Cluster | EKS Secrets encryption | **KMS envelope** (`resources: [secrets]`, customer-managed key) |
| Cluster | API endpoint | **private** + scoped public `<OPERATOR_IP>/32` (never `0.0.0.0/0`) |

---

## Failures & fixes

Nothing here is hidden; the run required two fixes to reach a green end-to-end
state, and one transient fault self-healed by design.

1. **API endpoint unreachable from the workstation (fixed — operator config).**
   `kubectl` timed out because the cluster's secure default is a **private-only**
   API endpoint (H-02) and `bootstrap_cluster_creator_admin_permissions = false`
   grants the provisioning principal no access (H-03). The git-ignored
   `terraform.tfvars` carried the scoped `/32` CIDR but not
   `cluster_endpoint_public_access = true`, and defined no access entry. **Fix:**
   enabled scoped public access (locked to `<OPERATOR_IP>/32`) and added an explicit
   operator access entry (`AmazonEKSClusterAdminPolicy`) in `terraform.tfvars`
   (git-ignored — no personal ARN or account ID reaches git), then re-applied
   (**2 added, 1 changed**). This is the documented operator opt-in (runbook § 3.3),
   not an architecture change.

2. **Postgres PVC never bound → MLflow stalled (fixed — committed code).**
   The base Postgres `volumeClaimTemplate` intentionally leaves `storageClassName`
   unset to use the **cluster default** StorageClass (portable to Docker Desktop /
   kind). **EKS 1.35 ships a `gp2` class (served by the EBS CSI driver) but marks no
   default**, so the claim failed with *"no persistent volumes available for this
   claim and no storage class is set"* — Postgres never started, blocking the whole
   MLflow platform. **Root cause belongs to the Sprint 7 EKS architecture**, and the
   base manifest documents the remedy ("make it explicit per environment"). **Fix:**
   a surgical JSON6902 patch in the **AWS overlay** pinning
   `volumeClaimTemplates/0/spec/storageClassName = gp2` (env-specific, so it lives in
   the overlay, not the shared base; a strategic merge was rejected because it
   replaces the whole claim spec and drops `accessModes`/`resources`). Committed as
   part of this PR. After the fix the PVC **Bound** to a gp2 EBS volume and Postgres
   went `1/1 Running`.

3. **First Job pod `Init:Error` — self-healed (no fix needed).** On a cold cluster
   MLflow's first-boot DB migration ran longer than the `wait-for-mlflow` init
   budget (~5 min), so the first pod failed its init and the Job's `backoffLimit: 2`
   created a retry pod that completed cleanly once MLflow was `Ready`. This is the
   intended transient-fault behaviour (ADR-011), recorded as `failed: 1` alongside
   `succeeded: 1`. The MLflow server's brief `CrashLoopBackOff` before Postgres
   existed (DNS `mlflow-postgres` unresolvable) likewise self-healed once Postgres
   came up. A minor tuning opportunity (a larger first-boot init budget) is noted in
   [§ Limitations](#limitations); it was not changed here as the retry semantics
   already guarantee completion.

---

## 15. Teardown

Teardown was **not delayed** — the environment was destroyed and verified clean the
same session. The Kubernetes workload was removed first (`kubectl delete -k
k8s/overlays/aws` → `mlops` namespace cascade, releasing the EBS PVC), then all
infrastructure:

```
$ terraform destroy -auto-approve   -> Destroy complete! Resources: 65 destroyed.
```

Verified from **three** independent angles (runbook § 5.2):

| Check | Command | Result |
|---|---|---|
| Terraform state empty | `terraform show` / `terraform state list` | *"The state file is empty."* / **0** entries |
| No EKS cluster | `aws eks list-clusters` | **empty** |
| No NAT gateway (available) | `aws ec2 describe-nat-gateways --filter state=available` | **empty** |
| No unattached Elastic IP | `aws ec2 describe-addresses` (no `AssociationId`) | **empty** |
| ECR gone | `aws ecr describe-repositories --repository-names mlops-pipeline` | **RepositoryNotFoundException** |
| S3 buckets gone | `aws s3api head-bucket` (datasets + artifacts) | **404 Not Found** (0 project buckets remain) |
| Repo working tree | `git status` | account-ID/run-time overlay edits reverted; `terraform.tfvars` git-ignored |

The 3 customer-managed **KMS keys** are **scheduled for deletion** (their bucket/
cluster consumers are already gone) — the standard AWS 7–30 day pending-deletion
window, not an active resource; they incur no meaningful ongoing charge. **Nothing
billable remains.**

---

## Limitations

Stated plainly so nothing is over-read. This is a **validation** environment; it is
**not** production. Beyond the general bounds in
[docs/cloud-operations.md § 7](../cloud-operations.md), specific to this run:

- **`readOnlyRootFilesystem: false`** — deferred by design (ADR-010); `dvc repro`
  writes state in-tree.
- **Single node, single NAT, 2 AZs** — cost-minimized; no HA/scale is provisioned or
  claimed.
- **Transient first-pod failure** — the `wait-for-mlflow` init budget is slightly
  tight versus MLflow's first-boot DB migration on a cold cluster; the `backoffLimit`
  retry covers it, but the first pod burns a retry (`failed: 1`). A larger init
  budget would avoid the spurious attempt; not changed here (out of scope, and the
  retry already guarantees completion).
- **Operator access is a scoped public opt-in** — reaching the private-by-default API
  from a workstation needs the `/32` public opt-in; a standing private-only posture
  would require in-VPC access (bastion/VPN/SSM), which this project does not
  provision (ADR-022).
- **Restricted Pod Security Standard not admission-ratified** — the controls are
  applied and a live API server admitted the pod, but no admission-policy engine
  enforces the profile.

The credible, evidenced claim: **the full Sprint 7 hardened platform — ECR images,
KMS-encrypted S3 dataset and MLflow artifact stores, EKS Pod Identity, and the
in-cluster PostgreSQL+S3 MLflow tracking server — runs the MLOps pipeline to
completion on a real EKS 1.35 cluster, with the Sprint 5 security controls intact and
all AWS access on workload identity (no static keys), and the environment is
destroyed and verified clean afterward.**

---

## Related documentation

- [Cloud Operations runbook](../cloud-operations.md) · [Sprint 6 Runtime Evidence](sprint-06-runtime-evidence.md)
- [MLflow platform](../mlflow-platform.md) · [dataset](../dataset.md)
- ADR-021 · ADR-022 · ADR-023 · ADR-024 · ADR-025 · ADR-026 · ADR-027
