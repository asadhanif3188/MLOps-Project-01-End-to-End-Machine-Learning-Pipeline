# Kubernetes Manifests (`k8s/`)

Kubernetes deployment surface for the End-to-End ML Pipeline. This directory
holds the **architectural foundation** for running the pipeline as a
Kubernetes-native **batch workload** (a `Job`), not a long-running service.

> **Status — operations & proof (Sprint 5, PR 7).** PR 1 established
> the structure and namespace; PR 2 made it a **runnable** workload (real
> `ml-pipeline:local` image, real `dvc repro`, finite-run lifecycle —
> `restartPolicy: Never`, `backoffLimit: 2`, `activeDeadlineSeconds: 1800`); PR 3
> externalized **configuration** (a `ConfigMap`), a **Secret** template (created
> out-of-band, never committed), and a least-privilege **ServiceAccount** with the
> API-token automount off (see
> [§ Configuration, secrets & identity](#configuration-secrets--identity-pr-3));
> PR 4 added a **hardened `securityContext`** — non-root with an explicit uid/gid
> `10001`, `allowPrivilegeEscalation: false`, all Linux capabilities dropped, and
> seccomp `RuntimeDefault` (see [§ Security hardening](#security-hardening-pr-4);
> read-only root is deliberately deferred —
> [ADR-010](../docs/decisions/ADR-010-kubernetes-security-hardening.md)); PR 5
> added **resource requests/limits chosen from measured usage** and documented the
> lifecycle, the **deliberate absence of health probes**, and the **failure modes**
> (see [§ Resource & lifecycle management](#resource--lifecycle-management-pr-5)
> and [ADR-011](../docs/decisions/ADR-011-kubernetes-resource-lifecycle.md)); PR 6
> added **automated CI validation** of these manifests — YAML syntax,
> upstream **schema** (`kubeconform`), **Kustomize** rendering, and the PR 1–5
> **security/resource contract** (`k8s/validate.py`), plus an opt-in ephemeral-cluster
> admission dry-run (see [§ CI manifest validation](#ci-manifest-validation-pr-6)
> and [ADR-012](../docs/decisions/ADR-012-kubernetes-manifest-validation.md)); PR 7
> (this change) completes the sprint with the **operations & proof** documentation —
> a full deployment guide (below), an [operations runbook](../docs/kubernetes-operations.md)
> with a [troubleshooting matrix](#troubleshooting), a
> [security document](../docs/kubernetes-security.md), and a
> [Sprint 5 Proof-Impact Assessment](../docs/proof/sprint-05-proof-impact.md) — and
> re-executes the local deployment path from a clean state as evidence.
>
> **Update — Kubernetes Runtime Execution (Sprint 5, PR 8).** The last proof gap is
> now closed: the **complete pipeline runs to completion inside the Job**. On a
> local Docker Desktop cluster (2026-08-14) the Job reached **`Complete`**, the pod
> **`Succeeded`** with **exit code 0** on the first attempt, and all four stages ran
> (preprocess 768 rows → split 614/154 → train acc 0.7398 → evaluate acc 0.7078).
> The three PR 1–7 blockers were resolved as a minimal runtime contract (design of
> record: [ADR-013](../docs/decisions/ADR-013-kubernetes-runtime-execution.md)):
> DVC no-SCM mode (`core.no_scm = true`, mounted `config.local`) replaces the
> `/app is not a git repository` abort; the dataset is mounted read-only at
> `/app/data/raw` from an out-of-band ConfigMap; and the local overlay points MLflow
> at an in-pod file store so a local run needs no external MLflow or credentials. The
> earlier lifecycle/security/resource evidence (below) remains valid.
> **CI validation is still static** (plus opt-in admission) — it does **not** deploy
> or run the workload. Nothing here has been applied to a production cluster; the
> dataset ConfigMap and MLflow file store are **local-validation mechanisms, not
> production storage**; the resource values are **not production-certified** and
> **restricted Pod Security Standard compliance is not claimed** (read-only root is
> still deferred — [ADR-010](../docs/decisions/ADR-010-kubernetes-security-hardening.md)).

For the full rationale — why Kubernetes, why a `Job` and not a `Deployment`, the
workload lifecycle, and the local-vs-production boundary — see
[docs/kubernetes-architecture.md](../docs/kubernetes-architecture.md),
[ADR-009](../docs/decisions/ADR-009-kubernetes-workload-model.md), and
[ADR-010](../docs/decisions/ADR-010-kubernetes-security-hardening.md). For **day-2
operations**, the **security posture**, and the **evidence-based claims**, see the
[Operations runbook](../docs/kubernetes-operations.md),
[Security document](../docs/kubernetes-security.md), and the
[Sprint 5 Proof-Impact Assessment](../docs/proof/sprint-05-proof-impact.md).

## Layout

```text
k8s/
├── base/                     # environment-independent definition
│   ├── namespace.yaml        # the `mlops` namespace (environment boundary)
│   ├── serviceaccount.yaml   # least-privilege identity, token automount off
│   ├── configmap.yaml        # non-secret runtime config (LOG_LEVEL, MLFLOW_TRACKING_URI)
│   ├── dvc-config.yaml       # DVC no-SCM runtime config (config.local: core.no_scm=true) — PR 8
│   ├── secret.example.yaml   # Secret TEMPLATE (placeholders; excluded from kustomize)
│   ├── job.yaml              # the pipeline as a run-to-completion batch Job
│   └── kustomization.yaml    # aggregates the base, applies common labels
├── overlays/
│   └── local/                # specialization for a local cluster (kind/minikube)
│       ├── job-runtime.yaml   # PR 8: dataset mount + offline MLflow file-store env
│       └── kustomization.yaml # pins the image to ml-pipeline:local; applies the patch
└── validate.py              # static validation (security + required + runtime) — PR 6/8
```

Why Kustomize (and not raw `kubectl apply -f`): a single base is specialized per
environment through overlays with no duplicated YAML. Today the local overlay
only remaps the image; later PRs add environment-specific resources, config, and
security to the *same* structure.

## Render the manifests

Kustomize is the source of truth — always render through it rather than reading
the raw files as the final output:

```bash
# Base (image name only: ml-pipeline — the tag is pinned by an overlay)
kustomize build k8s/base

# Local overlay (image tag pinned to the locally built ml-pipeline:local)
kustomize build k8s/overlays/local
```

`kubectl` can render the same way with `kubectl kustomize k8s/overlays/local`.

## Prerequisites

| Tool | Purpose | Notes |
|---|---|---|
| **Docker** | Build the `ml-pipeline:local` image | Already required by the project ([root README § Running with Docker](../README.md#running-with-docker)). |
| **A local Kubernetes cluster** | Run the workload | **Docker Desktop Kubernetes**, **kind**, or **minikube** — any single-node local cluster. |
| **`kubectl`** | Apply/inspect/log/delete | Point it at the local context: `kubectl config current-context` should be `docker-desktop`, `kind-*`, or `minikube`. |
| **`kustomize`** (optional) | Render the manifests standalone | `kubectl` has it built in (`kubectl kustomize …` / `kubectl apply -k …`), so a separate binary is optional. |

No registry, no cloud account, and no credentials are required to render, apply,
inspect, **or run** the workload: a local green run uses the mounted dataset
(step 3) and the local overlay's in-pod MLflow **file store**, so no MLflow/DagsHub
credentials are needed. Credentials are only needed to exercise the real DagsHub
tracking path (see [§ Runtime execution record](#runtime-execution-record-pr-8) and
[ADR-013](../docs/decisions/ADR-013-kubernetes-runtime-execution.md)).

### Local cluster setup

Enable **one** local cluster:

```bash
# Docker Desktop: Settings → Kubernetes → "Enable Kubernetes" (shares the Docker daemon)
kubectl config use-context docker-desktop

# …or kind:
kind create cluster --name mlops
kubectl config use-context kind-mlops

# …or minikube:
minikube start
kubectl config use-context minikube
```

Confirm it is reachable before deploying: `kubectl get nodes` should list a `Ready`
node.

## Run it locally (runbook)

The steps below are the operational flow for a local cluster (kind, minikube, or
Docker Desktop Kubernetes). For **day-2 operations** — re-running, rotating the
Secret, updating config, the full failure-mode playbook, and the complete
troubleshooting matrix — see the
[Kubernetes Operations runbook](../docs/kubernetes-operations.md).

> **Executed green on 2026-08-14** against Docker Desktop Kubernetes (v1.34.3): the
> Job reached **`Complete`**, the pod **`Succeeded`** with **exit 0** on the first
> attempt, and the full pipeline ran (preprocess → split → train → evaluate) — see
> [§ Runtime execution record (PR 8)](#runtime-execution-record-pr-8). This uses the
> PR 8 runtime contract ([ADR-013](../docs/decisions/ADR-013-kubernetes-runtime-execution.md)):
> DVC no-SCM (mounted `config.local`), a dataset mounted out-of-band at
> `/app/data/raw`, and an in-pod MLflow file store. The earlier PR 2 run (2026-08-12)
> verified the Job *lifecycle* and terminated at the then-open SCM blocker; it is
> kept below as the historical baseline that PR 8 closes.

### 1. Build the image locally

The workload uses the locally built production image — there is no registry to
pull from yet (see the root README § "Running with Docker").

```bash
docker build \
  --build-arg VCS_REF="$(git rev-parse --short HEAD)" \
  --build-arg BUILD_VERSION="1.3.1" \
  -t ml-pipeline:local .
```

### 2. Make the image available to the cluster

**kind** and **minikube** run their own container runtime and must be side-loaded:

```bash
kind load docker-image ml-pipeline:local          # kind
# or:
minikube image load ml-pipeline:local             # minikube
```

**Docker Desktop.** Older Docker Desktop shared the Docker daemon with its
Kubernetes, so no load step was needed. On newer Docker Desktop the Kubernetes node
(`desktop-control-plane`) runs on **containerd**, whose image store is **separate**
from the `docker build` daemon — so a freshly built image can be *invisible* to the
kubelet, and pods silently run a **stale** cached image (symptom: the pod behaves
like old code even though `docker images` shows your new build). Build without an
attestation manifest and import into the node's `k8s.io` containerd namespace:

```bash
docker build --provenance=false --sbom=false -t ml-pipeline:local .
docker save ml-pipeline:local \
  | docker exec -i desktop-control-plane ctr -n k8s.io images import -
# verify the node now has YOUR digest:
docker exec desktop-control-plane ctr -n k8s.io images ls | grep ml-pipeline:local
```

> If a run misbehaves, confirm the pod is on the fresh image:
> `kubectl -n mlops exec <pod> -- md5sum dvc.yaml` should match
> `docker run --rm --entrypoint md5sum ml-pipeline:local dvc.yaml`.

### 3. Provide the runtime dataset (required for a green run)

The runtime image ships **no** dataset by design (`.dockerignore`; the raw file is
DVC-tracked). The local overlay mounts it read-only at `/app/data/raw` from a
ConfigMap you create **out-of-band** from the local, git-ignored `data/raw/data.csv`
— the same out-of-band pattern as the Secret, so the dataset never enters git or a
rendered manifest. Fetch the dataset first if you don't have it (`dvc pull`, or the
public DagsHub raw URL), then:

```bash
kubectl create namespace mlops --dry-run=client -o yaml | kubectl apply -f -   # if not yet applied
kubectl create configmap mlops-pipeline-dataset --namespace mlops \
  --from-file=data.csv=data/raw/data.csv
```

> **Local validation only, not production storage.** A ConfigMap caps at 1 MiB; a
> real dataset would come from a PVC / object store / `dvc pull`
> ([ADR-013](../docs/decisions/ADR-013-kubernetes-runtime-execution.md)). The volume
> is `optional: true`: if you skip this step the Job still starts, then fails fast at
> preprocess with `Dataset not found: 'data/raw/data.csv'` (the intended missing-input
> failure).

### 4. (Optional) create the credential Secret

**Not needed for a local green run:** the local overlay overrides
`MLFLOW_TRACKING_URI` to an in-pod file store, so tracking works offline with no
credentials. The Secret is only for exercising the **real DagsHub** path locally
(create it, then drop the overlay's `MLFLOW_*` override). It is created
**out-of-band** so nothing is committed — full rationale in
[§ Secrets](#secrets--creation-lifecycle-and-why-nothing-is-committed):

```bash
kubectl create namespace mlops --dry-run=client -o yaml | kubectl apply -f -   # if not yet applied
kubectl create secret generic mlops-pipeline-secret --namespace mlops --from-env-file=.env
```

### 5. Apply the workload

```bash
kubectl apply -k k8s/overlays/local
```

This creates the `mlops` namespace and the `mlops-pipeline` Job (plus the
ServiceAccount and ConfigMaps). With the dataset ConfigMap from step 3 in place, the
Job runs the full pipeline to completion (exit 0).

### 6. Inspect it

```bash
kubectl -n mlops get jobs,pods                    # high-level status
kubectl -n mlops describe job/mlops-pipeline      # events, completions, backoff
kubectl -n mlops wait --for=condition=complete job/mlops-pipeline --timeout=300s
# exit code of the (first) pod — expect 0:
pod=$(kubectl -n mlops get pods -l app.kubernetes.io/name=mlops-pipeline -o jsonpath='{.items[0].metadata.name}')
kubectl -n mlops get pod "$pod" -o jsonpath='{.status.containerStatuses[0].state.terminated.exitCode}{"\n"}'
```

### 7. Retrieve logs

```bash
kubectl -n mlops logs job/mlops-pipeline          # logs from the Job's pod
kubectl -n mlops logs -f job/mlops-pipeline       # follow while it runs
# Expect the four stages: preprocess -> split -> train -> evaluate.
```

### 8. Delete / re-run

A Job's pod template is immutable, so re-running means delete-then-apply (this is
expected for batch Jobs — you are starting a fresh run, not mutating a live one):

```bash
kubectl delete -k k8s/overlays/local              # remove Job + namespace
kubectl apply  -k k8s/overlays/local              # re-create for a fresh run
```

To re-run while keeping the namespace, delete just the Job first:

```bash
kubectl -n mlops delete job/mlops-pipeline
kubectl apply -k k8s/overlays/local
```

## Troubleshooting

Deployment-time symptoms and fixes. The **full** operational matrix (stalls, OOM,
secret/config, admission, RBAC) is in the
[Kubernetes Operations runbook § Troubleshooting matrix](../docs/kubernetes-operations.md#3-troubleshooting-matrix).

| Symptom | Likely cause | Investigation | Remediation |
|---|---|---|---|
| `ImagePullBackOff` / `ErrImagePull` | Cluster can't see `ml-pipeline:local` — not built, or not side-loaded (kind/minikube). | `kubectl -n mlops describe pod <pod>` → `Events`. | Build it, then side-load for kind/minikube (`kind load docker-image …` / `minikube image load …`); on Docker Desktop import into the node's containerd — see [§ Step 2](#2-make-the-image-available-to-the-cluster). |
| Pod runs **stale code** (e.g. an old `dvc.yaml`; DVC errors about stages/params that don't match the repo) | Docker Desktop k8s uses a containerd store separate from `docker build`; the kubelet is running an **old cached** `ml-pipeline:local`. | `kubectl -n mlops exec <pod> -- md5sum dvc.yaml` vs `docker run --rm --entrypoint md5sum ml-pipeline:local dvc.yaml` (differ ⇒ stale). | Import the fresh image into the node's `k8s.io` containerd namespace ([§ Step 2](#2-make-the-image-available-to-the-cluster)); then delete + re-apply the Job. |
| Pod `Pending`, never schedules | Node lacks CPU/memory for the `requests` (250m/256Mi). | `describe pod` → `FailedScheduling`. | Raise the local cluster's resources (Docker Desktop → Settings → Resources). |
| `CreateContainerConfigError` — "runAsNonRoot and image has non-numeric user" | The explicit numeric `runAsUser: 10001` was removed (the image's `USER` is a name). | `describe pod` → container state. | Keep `runAsUser: 10001` — it is **required** ([ADR-010](../docs/decisions/ADR-010-kubernetes-security-hardening.md)). |
| Pod `Error`, log `/app is not a git repository` | The DVC no-SCM config isn't mounted — `config.local` (`core.no_scm=true`) missing at `/app/.dvc/config.local`. | `kubectl -n mlops exec <pod> -- cat .dvc/config.local`. | Ensure `dvc-config.yaml` is in the base and the Job mounts it (subPath) — [ADR-013](../docs/decisions/ADR-013-kubernetes-runtime-execution.md); `python k8s/validate.py` asserts this. |
| Pod `Error`, log `Dataset not found: 'data/raw/data.csv'` | The dataset ConfigMap wasn't created (volume is `optional: true`, so the pod still starts). | `kubectl -n mlops get configmap mlops-pipeline-dataset`. | Create it out-of-band — see [§ Step 3](#3-provide-the-runtime-dataset-required-for-a-green-run). This is the intended graceful missing-input failure. |
| Pod `Error`, MLflow "filesystem tracking backend … maintenance mode" | Newer MLflow gates its file store; `MLFLOW_ALLOW_FILE_STORE` not set for the local file-store path. | `kubectl -n mlops logs job/mlops-pipeline` at the tracking boundary. | The local overlay sets `MLFLOW_ALLOW_FILE_STORE=true`; keep it, or point at a real MLflow endpoint + Secret ([ADR-013](../docs/decisions/ADR-013-kubernetes-runtime-execution.md)). |
| Job `Failed`, `BackoffLimitExceeded` | Pod failed all `backoffLimit + 1 = 3` attempts. | `describe job` events, then the **pod logs** for the real cause. | Fix the underlying pod error above; fail-fast is intentional for a deterministic pipeline. |
| `kubectl apply -k` render/schema error | Malformed or schema-invalid manifest. | `kustomize build k8s/overlays/local` and `python k8s/validate.py`; `kubeconform -strict`. | Fix the manifest; CI's `k8s-validate` catches this class before merge. |

## Execution record (PR 2)

Executed **2026-08-12** against **Docker Desktop Kubernetes v1.34.3**
(`containerd://2.2.0`), image `ml-pipeline:local` built from this repo.

**What was proven — the Job lifecycle on a real cluster:**

- `kubectl apply -k k8s/overlays/local` created the `mlops` namespace and the
  `mlops-pipeline` Job. The local image resolved with **no registry pull**.
- The Job ran its designed retry lifecycle — **3 attempts** (initial pod +
  `backoffLimit: 2`), each a *fresh* pod with `RESTARTS: 0` (confirming
  `restartPolicy: Never`) — then emitted `BackoffLimitExceeded` and settled into a
  terminal **`Failed`** state (`status.failed: 3`).

```text
SuccessfulCreate  pod: mlops-pipeline-z9tsv     (attempt 1)
SuccessfulCreate  pod: mlops-pipeline-lxtq7     (attempt 2)
SuccessfulCreate  pod: mlops-pipeline-d5szf     (attempt 3)
BackoffLimitExceeded  Job has reached the specified backoff limit  ->  Job Failed
```

**What did NOT happen — a green pipeline run (honest result).** All three attempts
failed identically:

```text
$ kubectl -n mlops logs job/mlops-pipeline
ERROR: /app is not a git repository
```

`dvc repro` requires an SCM; the runtime image neither runs `git init` nor sets
`core.no_scm`, so DVC aborts before evaluating any stage — earlier than the
missing-data failure one might expect. Making the pipeline **green** in-cluster is
the PR 3 "make it runnable" scope and needs three things, in order:

1. an SCM in the image — `git init` at build time, or `core.no_scm = true` in
   `.dvc/config`;
2. a mounted dataset — the runtime image ships **no** data by design
   (`.dockerignore`), and `data/raw/data.csv` is itself DVC-tracked;
3. MLflow/DagsHub credentials for the tracking calls.

The workload *mechanism* is validated here; its *green execution* is not claimed.
**(Superseded by the PR 8 record below, which achieves the green run.)**

## Runtime execution record (PR 8)

Executed **2026-08-14** against **Docker Desktop Kubernetes v1.34.3**, image
`ml-pipeline:local` (imported into the node's containerd — see [§ Step 2](#2-make-the-image-available-to-the-cluster)),
with the dataset ConfigMap created out-of-band and the local overlay's MLflow
file-store override. Design of record:
[ADR-013](../docs/decisions/ADR-013-kubernetes-runtime-execution.md).

**What was proven — the complete pipeline runs to completion in-cluster:**

```text
$ kubectl -n mlops get job/mlops-pipeline
NAME             STATUS     COMPLETIONS   DURATION
mlops-pipeline   Complete   1/1           41s

$ kubectl -n mlops get pod <pod> -o jsonpath='{.status.phase} {...exitCode}'
Succeeded  exitCode=0   restarts=0        # first attempt, no retries

$ kubectl -n mlops logs job/mlops-pipeline        # (abridged)
Running stage 'preprocess':  … Preprocess stage completed: 768 rows written
Running stage 'split':       … 614 train rows, 154 held-out rows
Running stage 'train':       … Best model accuracy: 0.7398 … Model saved
Running stage 'evaluate':    … model accuracy: 0.7078
```

- **Job** `Complete=True (CompletionsReached)`, `succeeded: 1`.
- **Pod** `Succeeded`, container **exit code 0**, `RESTARTS: 0` (green on the first
  attempt — no back-off needed).
- **All four stages** ran in order: preprocess → split → train → evaluate.
- **Security unchanged and re-verified on the live pod:** QoS `Burstable`,
  `runAsUser 10001`, `automountServiceAccountToken: false`; the two added mounts are
  read-only ConfigMaps (`/app/.dvc/config.local` subPath, `/app/data/raw`).

**Failure test (verifies fail-fast + back-off).** With the dataset ConfigMap
removed and the Job re-applied, every attempt failed fast at preprocess
(`ERROR: failed to reproduce 'preprocess': [Errno 2] No such file or directory:
'/app/data/raw/data.csv'`); the Job ran its **3 fresh-pod attempts** (`RESTARTS: 0`
each), then settled into terminal **`Failed: BackoffLimitExceeded`** (`failed: 3`).
Restoring the ConfigMap returned the Job to green. This confirms the intended
missing-input path: application fails → non-zero exit → Job fails → back-off →
terminal `Failed` (ADR-011).

## Configuration, secrets & identity (PR 3)

The image is immutable; everything environment-specific is injected at run time,
split by sensitivity. All names below are the **actual** variables the code reads
(`src/pipeline_io.py::require_env`, `src/logging_config.py`, and the MLflow
client) — none are invented.

| Value | Sensitive? | Carrier | Committed? |
|---|---|---|---|
| `LOG_LEVEL` (default `INFO`) | No | `ConfigMap` `mlops-pipeline-config` | Yes |
| `MLFLOW_TRACKING_URI` (DagsHub endpoint) | No — an endpoint, not a credential | `ConfigMap` `mlops-pipeline-config` | Yes |
| `MLFLOW_TRACKING_USERNAME` | **Yes** — auth | `Secret` `mlops-pipeline-secret` | **No — created out-of-band** |
| `MLFLOW_TRACKING_PASSWORD` | **Yes** — auth token | `Secret` `mlops-pipeline-secret` | **No — created out-of-band** |

Both carriers are wired into the Job with `envFrom` (see
[`base/job.yaml`](base/job.yaml)): the ConfigMap is always present (it is part of
the Kustomize base); the Secret reference is `optional: true` so
`kubectl apply -k` succeeds *before* the Secret exists.

### ConfigMap usage

`kustomize build k8s/overlays/local` renders the ConfigMap and the Job's
`envFrom`. The `MLFLOW_TRACKING_URI` is the project's public DagsHub MLflow
endpoint — the same host already committed as the DVC S3 remote in
[`.dvc/config`](../.dvc/config), so committing it leaks nothing. A future
staging/prod overlay overrides it with a patch or `configMapGenerator` without
touching the base.

### Secrets — creation, lifecycle, and why nothing is committed

**Why credentials are never committed.** A `Secret`'s `data` is only base64, not
encryption; committing it (even the example) would leak the DagsHub token into git
history forever. So the repo ships **only a template** —
[`base/secret.example.yaml`](base/secret.example.yaml), with placeholder values —
and it is deliberately **excluded from `base/kustomization.yaml`**, so no render or
apply can ever emit it. The real Secret is created straight from your local,
git-ignored `.env` and never passes through git or a rendered manifest.

**Create it** (once per cluster/namespace, after the namespace exists):

```bash
# From your local .env (see .env.example for the three variables):
kubectl create secret generic mlops-pipeline-secret \
  --namespace mlops \
  --from-env-file=.env

# …or supply just the two credential keys explicitly:
kubectl create secret generic mlops-pipeline-secret \
  --namespace mlops \
  --from-literal=MLFLOW_TRACKING_USERNAME='<dagshub-username>' \
  --from-literal=MLFLOW_TRACKING_PASSWORD='<dagshub-token>'
```

**Lifecycle.** Rotate by replacing it (the next Job run picks up new values):

```bash
kubectl create secret generic mlops-pipeline-secret --namespace mlops \
  --from-env-file=.env --dry-run=client -o yaml | kubectl apply -f -
```

Remove it with `kubectl -n mlops delete secret mlops-pipeline-secret`. Because the
Job's `secretRef` is `optional: true`, the pipeline still *starts* without it —
the MLflow calls then fail with a clear auth error rather than the pod refusing to
schedule.

### Does the workload need Kubernetes API access? No.

The pipeline runs `dvc repro` (preprocess → split → train → evaluate) and talks
only to MLflow/DagsHub over HTTPS — it never creates, reads, or watches cluster
objects. So it gets a dedicated **ServiceAccount** (`mlops-pipeline`) purely as a
named identity, with **`automountServiceAccountToken: false`** (set on both the
ServiceAccount and the Job's pod template): no API token is mounted, so there is
no unused, exfiltratable credential in the pod. For the same reason **no
`Role`/`RoleBinding` is defined** — granting permissions the workload never uses
would violate least privilege.

This was verified on the live cluster: the applied pod carried
`serviceAccountName: mlops-pipeline` with an **empty `spec.volumes`** and an empty
container `volumeMounts` — i.e. no `kube-api-access-*` projected-token volume and
no `/var/run/secrets/kubernetes.io/serviceaccount` mount. The pod still *started*
(container Created → Started) with the optional Secret absent, then hit the same
known SCM blocker as PR 2 (`/app is not a git repository`) — confirming this PR
changed configuration/identity only, not pipeline behavior.

## Security hardening (PR 4)

The `Job` runs under a `securityContext` that *enforces* a restricted posture at
the platform layer, rather than trusting the image alone. Design of record:
[ADR-010](../docs/decisions/ADR-010-kubernetes-security-hardening.md).

**Pod level** (applies to every container in the pod):

| Field | Value | Why |
|---|---|---|
| `runAsNonRoot` | `true` | Refuse to start as uid 0 — enforced, not assumed. |
| `runAsUser` / `runAsGroup` | `10001` | Matches the Dockerfile's `useradd/groupadd --uid/--gid 10001`. **Required**, not cosmetic: the image's `USER` is the *name* `appuser`, which the kubelet cannot verify as non-root, so `runAsNonRoot` alone would fail with `CreateContainerConfigError`. |
| `seccompProfile.type` | `RuntimeDefault` | Apply the runtime's default syscall filter instead of running unconfined. |

**Container level** (these fields cannot live at the pod level):

| Field | Value | Why |
|---|---|---|
| `allowPrivilegeEscalation` | `false` | No setuid/setgid escalation (verified: `NoNewPrivs: 1`). |
| `capabilities.drop` | `[ALL]` | The pipeline runs Python + `dvc` over HTTPS; it needs **no** Linux capabilities. |
| `readOnlyRootFilesystem` | `false` | Deliberately deferred — see below. |

### Why the root filesystem is **not** read-only (yet)

`readOnlyRootFilesystem: true` is evaluated and intentionally *not* enabled,
because `dvc repro` **writes DVC state in-tree at the `/app` repo root**:

```bash
# Proof — read-only root, only /tmp writable:
docker run --rm --user 10001:10001 --cap-drop ALL --security-opt no-new-privileges \
  --read-only --tmpfs /tmp ml-pipeline:local dvc repro
# → ERROR: unexpected error - [Errno 30] Read-only file system: '/app/.dvc/tmp'
```

DVC further writes `/app/.dvc/cache`, rewrites `/app/dvc.lock`, and needs a
writable `/app/.git` for its SCM — all at the repo root, alongside the read-only
baked-in code and `.dvc/config`. Those paths cannot be carved out with `emptyDir`
without shadowing the image's own files or making the code tree writable (which
defeats the control). Enabling it now would make the container fail *earlier* than
the pre-existing SCM blocker — weakening a working workload to pass a checkbox. It
is deferred to the same work that makes the pipeline green in-cluster (relocating
DVC cache/tmp/lock + SCM onto declared writable volumes), tracked in
[ADR-010](../docs/decisions/ADR-010-kubernetes-security-hardening.md).

### How this was validated

- **`docker run` probes** against the real image: the core stack imports cleanly
  under `--cap-drop ALL --security-opt no-new-privileges` as uid `10001`; `dvc
  repro` reaches the *same* pre-existing `/app is not a git repository` blocker
  (behaviour-neutral); `NoNewPrivs: 1` confirmed; and the read-only-root failure
  above is reproduced.
- **Live cluster (Docker Desktop v1.34.3):** `kubectl apply -k k8s/overlays/local`
  was **admitted** (the explicit numeric uid satisfies `runAsNonRoot`). The pod's
  enforced `spec.securityContext` and container `securityContext` reported exactly
  the values above; `spec.volumes` was still empty (PR 3 token automount-off
  intact); the container **ran** and terminated (exit 255) at the same SCM blocker
  — no regression. Resources were deleted afterward.
- **21 rendered-manifest assertions** pass (fields present, at the correct pod-vs-
  container scope, no `privileged`/`hostNetwork`/`hostPID`/`hostIPC`). No local
  static scanner (`kubesec`/`kube-score`/`kube-linter`/`checkov`/`trivy`) is
  installed; the cluster admission + kubelet enforcement served as the
  authoritative check, and the manifest was not sent to any external service.

> **Not claimed:** restricted **Pod Security Standard compliance**. The manifest
> carries the fields the profile requires and a live cluster admitted it, but no
> Pod Security admission label or policy engine has validated it, and read-only
> root is not met.

## Resource & lifecycle management (PR 5)

The Job declares **resource requests and limits chosen from measured usage of the
real image**, and its finite-run lifecycle is documented and justified. Design of
record: [ADR-011](../docs/decisions/ADR-011-kubernetes-resource-lifecycle.md).

### Resources

```yaml
resources:
  requests: { cpu: 250m, memory: 256Mi }
  limits:   { cpu: "1",  memory: 512Mi }
```

| Field | Value | Why (measured) |
|---|---|---|
| `requests.cpu` | `250m` | Scheduling reservation for a short deterministic batch; each grid fit ≈ 1 s on one core. |
| `requests.memory` | `256Mi` | Above the measured **~133 MiB** import+train floor — a genuine reservation. |
| `limits.cpu` | `"1"` | Caps joblib's `n_jobs=-1` worker fan-out to one worker → **bounds memory** (loky reads the cgroup quota, not the node's core count). |
| `limits.memory` | `512Mi` | ~3.9× the measured 1-CPU peak; a runaway is `OOMKilled`, not left to eat the node. |

Requests ≠ limits ⇒ **Burstable** QoS (confirmed on a live cluster).

**Why the CPU limit is load-bearing here.** `train` runs
`GridSearchCV(n_jobs=-1)`; joblib/loky sizes its pool from the **CPU limit**, and
each worker forks the ~130 MiB interpreter. So the CPU limit is also the
memory-safety control. Measured peak (whole cgroup, incl. workers) on the real
image:

| Granted CPU | Peak memory | `train` wall time |
|---|---|---|
| 1 CPU | ~133 MiB | ~2.5 s |
| 2 CPU | ~419 MiB | ~5.4 s |
| unlimited (20 cores) | ~1785 MiB | ~20 s |

At this data size more cores *hurt* (fork/dispatch overhead dwarfs the sub-second
fits), so capping at 1 CPU is leaner **and** faster.

### Lifecycle

- **`restartPolicy: Never`** — each retry is a fresh, independently-inspectable
  pod (a Job requires `Never` or `OnFailure`).
- **`backoffLimit: 2`** — the pipeline is deterministic, so retries only absorb a
  *transient* MLflow/DagsHub blip; the controller back-offs exponentially between
  attempts, so a fast-failing pod cannot hot-loop.
- **`activeDeadlineSeconds: 1800`** — an outer stall-guard (e.g. a hung network
  call), not a performance SLO; deliberately generous.

### No health probes — by design

No liveness/readiness/startup probes are defined. Probes model a long-running
server (readiness gates Service traffic; liveness restarts a wedged daemon); this
is a finite batch Job with **no socket, no Service, and no traffic**. Its health
is *terminal* — exit `0` = success, non-zero = failure — which the Job controller
already observes from the exit status. A liveness probe would need an HTTP endpoint
the app should not expose, or would fire during normal quiet compute and kill a
healthy run. The real health signal is the exit code plus the structured logs.

### Failure modes

| Failure | Surfaces as | Where to look |
|---|---|---|
| Image pull | `ErrImagePull` / `ImagePullBackOff` | `describe pod` events; image not side-loaded into the cluster. |
| Configuration | early non-zero exit (`ConfigError`) | logs; ConfigMap absent/misnamed. |
| Secret | starts (Secret is `optional`), then MLflow auth error | logs at the tracking boundary; Secret not created out-of-band. |
| Application | non-zero exit (today: `/app is not a git repository`) | identical across all 3 attempts ⇒ deterministic, not transient. |
| Resource exhaustion | `OOMKilled` (exit 137) over `limits.memory`; throttling over `limits.cpu` | pod `terminated.reason: OOMKilled` (validated at 64Mi). |

### How this was validated

- **Resource probe** on the real image (`run_training`, synthetic Pima-shaped
  data): import floor ~132 MiB; peak-vs-CPU as tabled; `joblib.cpu_count()`
  confirmed to track the cgroup quota, not `os.cpu_count()`.
- **Success at the chosen limits** — `docker run --cpus=1 --memory=512m
  --memory-swap=512m …` completes, ~133 MiB peak, exit 0.
- **Resource-exhaustion failure** — the same run under `--memory=64m` is
  `OOMKilled` (exit 137): the limit is kernel-enforced.
- **Live cluster (Docker Desktop v1.34.3)** — enforced `resources` matched exactly,
  **QoS `Burstable`**, **no probes**, `restartPolicy: Never`; the Job ran its 3-attempt
  back-off lifecycle and every attempt hit the *same* pre-existing SCM blocker
  (exit 255) — **none `OOMKilled`**, i.e. no new failure mode. Deleted afterward.

> **Not claimed:** production-certified capacity. These values are tuned for a
> *local* single-node run on the small bundled dataset; a larger dataset, wider
> grid, or real cluster would require re-measuring.

## CI manifest validation (PR 6)

Every push and pull request runs **static** validation of these manifests, so a
future edit cannot silently regress the PR 1–5 contract. Design of record:
[ADR-012](../docs/decisions/ADR-012-kubernetes-manifest-validation.md);
the job is `k8s-validate` in [ci.yml](../.github/workflows/ci.yml).

**What runs (deterministic, no cluster, workload never executed):**

| Check | Tool (pinned) | Proves |
|---|---|---|
| YAML syntax + Kustomize rendering | `kustomize build` (v5.4.3) | `base/` and `overlays/local/` render; the YAML parses. |
| Kubernetes schema | `kubeconform -strict` (v0.6.7, schema v1.31.0) | every field is a real API field of the right type; unknown fields are rejected. |
| Security + required fields | `k8s/validate.py` (stdlib + PyYAML) | the workload contract below. |

`k8s/validate.py` asserts, with a PASS/FAIL line per check: `runAsNonRoot` + non-root
`runAsUser`; `allowPrivilegeEscalation: false`; `seccompProfile: RuntimeDefault`;
`capabilities: drop [ALL]`; an explicit non-default ServiceAccount that exists;
`automountServiceAccountToken: false` (pod + SA); CPU/memory **requests and limits**;
`restartPolicy` `Never`/`OnFailure`; an explicit **pinned** image (no `:latest`);
namespace pinning; and **secret hygiene** (no rendered `Secret`, no inline
credentials, no secret fingerprints, template holds only placeholders).

**Run it locally** (uses your local `kustomize`/`kubectl`; matches CI):

```bash
python k8s/validate.py                      # security + required-field checks
kustomize build k8s/overlays/local | \
  kubeconform -strict -summary -kubernetes-version 1.31.0 -schema-location default -
```

> **This is static validation, not deployment validation.** It proves the manifests
> are well-formed, schema-valid, and hardened — **not** that the workload deploys or
> runs. `kubeconform` needs network to fetch the pinned upstream schema; everything
> else is offline.

### Cluster integration (opt-in, separate)

Full cluster integration is kept **off** the per-PR path (bootstrap cost + flake
surface). A separate, manual job — `k8s-cluster-dry-run` (`workflow_dispatch` only) —
stands up an **ephemeral kind cluster** and does a **server-side dry run**
(`kubectl apply -k k8s/overlays/local --dry-run=server`): every object passes through
a real API server's validation, defaulting, and admission (incl. Pod Security), but
nothing is persisted and the Job never runs. Reproduce it against any local cluster:

```bash
kubectl create namespace mlops --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -k k8s/overlays/local --dry-run=server     # admits; runs nothing
```

This validates **admissibility**, not execution — a green dry-run still does not
mean the pipeline completes in-cluster (it needs an SCM in the image + mounted data,
see [§ Execution record](#execution-record-pr-2)).

## What is deliberately absent (and where it lands)

### Roadmap within Sprint 5

| Concern | Status | Target PR |
|---|---|---|
| Namespace + workload model (`Job`) | ✅ done | PR 1 |
| Kustomize base/overlay structure | ✅ done | PR 1 |
| Runnable workload (real image, command, lifecycle) | ✅ this PR | PR 2 |
| Local run **runbook** (build/load/apply/inspect/logs/re-run) | ✅ this PR | PR 2 |
| Demonstrated local cluster run (Job lifecycle) | ✅ executed 2026-08-12 (see [§ Execution record](#execution-record-pr-2)) | PR 2 |
| ConfigMap / Secret template / ServiceAccount + token automount off | ✅ this PR | PR 3 |
| Green in-cluster `dvc repro` (no-SCM + mounted data + MLflow) | ✅ done ([ADR-013](../docs/decisions/ADR-013-kubernetes-runtime-execution.md); see [§ Runtime execution record](#runtime-execution-record-pr-8)) | PR 8 |
| Runtime-contract static checks (no-SCM config, dataset mount, MLflow) | ✅ done (`k8s/validate.py` § Runtime execution contract) | PR 8 |
| Security hardening (securityContext, seccomp, dropped caps) | ✅ done (read-only root deferred, [ADR-010](../docs/decisions/ADR-010-kubernetes-security-hardening.md)) | PR 4 |
| CPU/memory resource requests/limits, lifecycle & probe decision, failure modes | ✅ done (measured values, [ADR-011](../docs/decisions/ADR-011-kubernetes-resource-lifecycle.md)) | PR 5 |
| CI manifest validation (syntax, schema, kustomize, security) | ✅ done (static + opt-in dry-run, [ADR-012](../docs/decisions/ADR-012-kubernetes-manifest-validation.md)) | PR 6 |
| Operations runbook, security doc & proof (this PR) | ✅ this PR ([operations](../docs/kubernetes-operations.md), [security](../docs/kubernetes-security.md), [proof](../docs/proof/sprint-05-proof-impact.md)) | PR 7 |

No credentials are committed anywhere in this directory, and none will be — the
Secret strategy (PR 3) uses a committed **template without values**.
