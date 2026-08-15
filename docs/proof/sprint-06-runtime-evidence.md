# Sprint 6 — Runtime Evidence (PR 7, Real Cloud Integration Test)

**Executed 2026-08-15** against a real, Terraform-provisioned **Amazon EKS**
cluster in the operator's **own** AWS account (`us-east-1`). This is the runtime
**proof** milestone: the same MLOps pipeline validated locally in Sprint 5 (PR 8)
was provisioned onto EKS and run to completion as a Kubernetes `Job`, and the
Sprint 5 security controls were re-verified on the **live** pod.

> **Scope & honesty.** This is a short-lived **validation** run, not a production
> deployment. Account identifiers, the operator IP, and all credentials/secrets
> are deliberately redacted (`<ACCOUNT_ID>`, `<OPERATOR_IP>`); no kubeconfig, token,
> or secret material is recorded here. One deliberate deviation from the AWS
> overlay's default — the offline MLflow file store — is documented in
> [§ Limitations](#limitations); it is **not** hidden.

---

## Result summary

| Dimension | Result |
|---|---|
| **Infrastructure (Terraform)** | `apply` of **29 resources**, `0 changed, 0 destroyed` — VPC, IAM, EKS, 1-node group |
| **EKS** | cluster `mlops-pipeline-dev-eks` **ACTIVE**, control plane **v1.35.6-eks**, 1 node **Ready** |
| **Kubernetes** | `kubectl apply -k k8s/overlays/aws` → Job created; image pulled from ECR |
| **MLOps pipeline** | Job **Complete**, pod **Succeeded**, **exit 0**, all 4 stages ran (preprocess→split→train→evaluate) |
| **Security (live pod)** | all 6 Sprint 5 controls verified on the running workload |
| **Teardown** | see [§ Teardown](#teardown) — records the real `terraform destroy` result |

---

## 0. Pre-flight inspection (offline, no cloud)

Verified before any provisioning — account-agnostic, no AWS calls:

- **Terraform valid** — `terraform fmt -check -recursive` clean; `terraform validate` → **"Success! The configuration is valid."**
- **EKS config valid** — covered by `validate` (eks.tf / iam.tf / network.tf parse and typecheck).
- **AWS overlay valid** — `kubectl kustomize k8s/overlays/aws` renders 5 objects (Namespace, ServiceAccount, 2 ConfigMaps, Job) with no error.
- **Sprint 5 controls present (static)** — the rendered Job carries `runAsNonRoot`, `runAsUser/Group 10001`, seccomp `RuntimeDefault`, `allowPrivilegeEscalation:false`, `capabilities.drop:[ALL]`, and CPU/memory requests+limits.

---

## 1. Infrastructure result (Terraform)

`terraform plan` ran against the operator's own account (`data.aws_caller_identity`
resolved to `<ACCOUNT_ID>`, **not** any other account) and `terraform apply`
created the plan exactly:

```
Plan:  29 to add, 0 to change, 0 to destroy.
Apply: Apply complete! Resources: 29 added, 0 changed, 0 destroyed.
```

**What was provisioned** (cost-conscious, no unrelated infrastructure):

| Component | Configuration |
|---|---|
| **VPC** | `10.0.0.0/16`, DNS hostnames + support enabled |
| **Subnets** | 2 public (`10.0.0.0/24`, `10.0.1.0/24`) + 2 private (`10.0.8.0/24`, `10.0.9.0/24`) across `us-east-1a`/`us-east-1b`; public tagged `kubernetes.io/role/elb`, private `internal-elb` |
| **Edge** | 1 Internet Gateway, **1 NAT Gateway** + 1 EIP (single-NAT, cost-optimized) |
| **IAM** | EKS cluster role (`AmazonEKSClusterPolicy`) + node role (`AmazonEKSWorkerNodePolicy`, `AmazonEKS_CNI_Policy`, `AmazonEC2ContainerRegistryReadOnly`) |
| **EKS cluster** | `mlops-pipeline-dev-eks`, Kubernetes **1.35**, logs `api`/`audit`/`authenticator`, public+private endpoint, **public API restricted to `<OPERATOR_IP>/32`**, `API_AND_CONFIG_MAP` auth + creator-admin bootstrap |
| **Node group** | **1 node**, `ON_DEMAND`, `t3.medium`, 20 GiB, `AL2023_x86_64_STANDARD`, `desired=min=max=1` (no autoscaler) |
| **Add-ons** | `coredns`, `kube-proxy`, `vpc-cni` |

---

## 2. EKS result

```
$ aws eks describe-cluster --name mlops-pipeline-dev-eks --query cluster.status
ACTIVE

$ kubectl get nodes -o wide
NAME                         STATUS   ROLES    AGE   VERSION               OS-IMAGE                       CONTAINER-RUNTIME
ip-10-0-9-159.ec2.internal   Ready    <none>   30m   v1.35.6-eks-254016e   Amazon Linux 2023.12.20260803  containerd://2.2.5+unknown

$ kubectl version   # (abridged)
Server Version: v1.35.6-eks-bca9cf6
```

- Cluster **ACTIVE**; control-plane **v1.35.6-eks**.
- **1 node Ready** in a private subnet (`10.0.9.159`), Amazon Linux 2023, containerd 2.2.5.
- `aws eks update-kubeconfig` wrote a context for `…:<ACCOUNT_ID>:cluster/mlops-pipeline-dev-eks`; kubectl connectivity confirmed (server version returned, API reachable from the allow-listed operator IP).

---

## 3. Kubernetes result

```
$ kubectl apply -k k8s/overlays/aws
namespace/mlops configured
serviceaccount/mlops-pipeline created
configmap/mlops-pipeline-config created
configmap/mlops-pipeline-dvc-config created
job.batch/mlops-pipeline created
```

- Workload rendered from the **committed base**, specialized by `k8s/overlays/aws`
  (image → ECR, `imagePullPolicy: Always`, read-only dataset mount).
- Runtime dataset supplied out-of-band as ConfigMap `mlops-pipeline-dataset`
  (`data.csv`, ~23 KiB), mounted read-only at `/app/data/raw`.
- Image pulled from ECR `<ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/mlops-pipeline:1.3.1`
  (built `--platform linux/amd64`, single manifest; the node role's
  `AmazonEC2ContainerRegistryReadOnly` authorized the pull — no pod credential).

---

## 4. MLOps result (the pipeline ran to completion)

```
$ kubectl -n mlops describe job/mlops-pipeline   # (abridged)
Completions:  1
Duration:     52s
Start Time:   Sat, 15 Aug 2026 14:49:44 +0500
Completed At: Sat, 15 Aug 2026 14:50:36 +0500
Pods Statuses: 0 Active / 1 Succeeded / 0 Failed

$ kubectl -n mlops get pod mlops-pipeline-6wb66 -o jsonpath=...
phase=Succeeded | exitCode=0 | reason=Completed | restarts=0 | qos=Burstable | automount=false
```

- **Job `Complete`**, `1/1`, **duration 52s** (first attempt, no back-off).
- **Pod `Succeeded`**, container **exit code 0**, `RESTARTS 0`, pod IP `10.0.9.85` on `ip-10-0-9-159`.
- Command `dvc repro`. **All four stages ran in order:**

| Stage | Evidence (from pod logs) |
|---|---|
| **preprocess** | `Preprocess stage completed: 768 rows written` |
| **split** | `614 train rows -> train.csv, 154 held-out rows -> test.csv` |
| **train** | `GridSearchCV` 12 fits; best params `{min_samples_leaf:1, min_samples_split:5}`; **best accuracy 0.7398**; model saved to `models/model.pkl` |
| **evaluate** | `Evaluate stage completed; model accuracy: 0.7078` |

The train/evaluate stages logged to MLflow's **in-pod file store** (`file:///app/mlruns`)
— see [§ Limitations](#limitations). Metrics match the Sprint 5 local run (train 0.7398 /
evaluate 0.7078), confirming behavior parity between the local and EKS runs.

**Runtime duration:** Job wall time **52s** (image already cached after first pull;
individual stages ~2–17s each per the timestamped logs).

---

## 5. Security result — Sprint 5 controls verified on the LIVE pod

Read directly from the running pod's spec (`kubectl get pod … -o jsonpath`):

```json
// .spec.securityContext (pod)
{"runAsGroup":10001,"runAsNonRoot":true,"runAsUser":10001,"seccompProfile":{"type":"RuntimeDefault"}}
// .spec.containers[0].securityContext
{"allowPrivilegeEscalation":false,"capabilities":{"drop":["ALL"]},"readOnlyRootFilesystem":false}
// .spec.containers[0].resources
{"limits":{"cpu":"1","memory":"512Mi"},"requests":{"cpu":"250m","memory":"256Mi"}}
```

| Control | Required | Live pod | Verdict |
|---|---|---|---|
| Non-root | `runAsNonRoot: true` | `true` | ✅ |
| Fixed UID/GID | `10001` / `10001` | `runAsUser 10001`, `runAsGroup 10001` | ✅ |
| seccomp | `RuntimeDefault` | `RuntimeDefault` | ✅ |
| No privilege escalation | `false` | `allowPrivilegeEscalation: false` | ✅ |
| Dropped capabilities | `drop: [ALL]` | `["ALL"]` | ✅ |
| Resource requests/limits | present | `250m/256Mi` req, `1/512Mi` lim | ✅ |

**Additional confirmations:** `automountServiceAccountToken: false` (no API token
mounted), QoS **Burstable** (requests ≠ limits), dedicated ServiceAccount
`mlops-pipeline`. `readOnlyRootFilesystem` is `false` — a **deliberate, pre-existing
deferral** ([ADR-010](../decisions/ADR-010-kubernetes-security-hardening.md); DVC
writes state in-tree), not a regression in this PR.

All controls are **inherited verbatim from the committed base** — the AWS overlay
changed no security field, and the transient MLflow patch (below) touches none either.

---

## Limitations

1. **Real MLflow/DagsHub tracking path NOT exercised this run.** DagsHub
   credentials (`.env` / `mlops-pipeline-secret`) were not supplied, so — rather
   than let the `train`/`evaluate` stages fail on a 401 (`TrackingError`, see
   [`src/tracking.py`](../../src/tracking.py)) — the run used a **transient**
   offline override mirroring the local overlay: `MLFLOW_TRACKING_URI=file:///app/mlruns`
   + `MLFLOW_ALLOW_FILE_STORE=true` (patch `k8s/overlays/aws/job-mlflow-filestore.yaml`,
   added to the overlay's `patches:` for this run only, **not committed**, reverted
   at teardown). This proves the pipeline **executes end-to-end on EKS**; it does
   **not** connectivity-test DagsHub from the cluster. To exercise real tracking:
   drop the patch and create `mlops-pipeline-secret` from DagsHub creds.
2. **`readOnlyRootFilesystem: false`** — deferred by design (ADR-010).
3. **Not production-grade topology** — 1 on-demand node, single NAT gateway,
   no Cluster Autoscaler; cost-tuned for a short validation run.
4. **Dataset via ConfigMap** — a ~23 KiB validation mechanism, **not** production
   storage (production would use S3 / a PVC / `dvc pull`; ADR-018).
5. **Resource requests/limits are not production-certified** — tuned for the small
   bundled dataset on a single node (ADR-011).
6. **Restricted Pod Security Standard compliance is not claimed** — the manifest
   carries the fields and a real API server admitted it, but no PSA label/policy
   engine validated it, and read-only root is not met.

---

## Teardown

Executed **2026-08-15**, immediately after evidence capture. The environment was
fully torn down:

```
$ kubectl delete -k k8s/overlays/aws        # workload + mlops namespace (cascades to dataset ConfigMap + pod)
$ aws ecr delete-repository --repository-name mlops-pipeline --force --region us-east-1
$ git checkout -- k8s/overlays/aws/kustomization.yaml   # revert run-time image edit + patch line
$ del k8s\overlays\aws\job-mlflow-filestore.yaml        # remove transient offline-MLflow patch
$ terraform destroy
Destroy complete! Resources: 29 destroyed.
```

**Verified clean:**

| Check | Result |
|---|---|
| `terraform destroy` | **`Destroy complete! Resources: 29 destroyed.`** (all 29 created resources removed) |
| Local Terraform state | **0 resources** |
| ECR repository | deleted (`--force`, image included) |
| AWS overlay working tree | **clean** — committed placeholder image (`000000000000`) restored, transient `job-mlflow-filestore.yaml` removed |

No resources remain from this run: EKS cluster, node group, NAT gateway, EIP, VPC,
subnets, and IAM roles were all destroyed by Terraform, and the ECR repository was
deleted separately. There is **no ongoing cost** and **no leftover repository diff**.
The teardown is not delayed — cleanup is complete and verified.
