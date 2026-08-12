# Kubernetes Manifests (`k8s/`)

Kubernetes deployment surface for the End-to-End ML Pipeline. This directory
holds the **architectural foundation** for running the pipeline as a
Kubernetes-native **batch workload** (a `Job`), not a long-running service.

> **Status — configuration & identity (Sprint 5, PR 3).** PR 1 established the
> structure and namespace; PR 2 made it a **runnable** workload (real
> `ml-pipeline:local` image, real `dvc repro`, finite-run lifecycle —
> `restartPolicy: Never`, `backoffLimit: 2`, `activeDeadlineSeconds: 1800`); PR 3
> (this change) externalizes **configuration** (a `ConfigMap`), a **Secret**
> template (created out-of-band, never committed), and a least-privilege
> **ServiceAccount** with the API-token automount turned off (see
> [§ Configuration, secrets & identity](#configuration-secrets--identity-pr-3)).
> It intentionally does **not** yet include security-context hardening or
> CPU/memory resource limits — those land in later, focused PRs (see
> [§ Roadmap within Sprint 5](#roadmap-within-sprint-5)).
> The Job was **executed on a local Docker Desktop cluster** (2026-08-12) and its
> lifecycle verified end to end (see [§ Execution record](#execution-record-pr-2)),
> but the pipeline does **not** complete yet: `dvc repro` aborts with `/app is not
> a git repository`, and a *green* run additionally needs the PR 3 data/credential
> wiring. Nothing here has been applied to a production cluster.

For the full rationale — why Kubernetes, why a `Job` and not a `Deployment`, the
workload lifecycle, and the local-vs-production boundary — see
[docs/kubernetes-architecture.md](../docs/kubernetes-architecture.md) and
[ADR-009](../docs/decisions/ADR-009-kubernetes-workload-model.md).

## Layout

```text
k8s/
├── base/                     # environment-independent definition
│   ├── namespace.yaml        # the `mlops` namespace (environment boundary)
│   ├── job.yaml              # the pipeline as a run-to-completion batch Job
│   └── kustomization.yaml    # aggregates the base, applies common labels
└── overlays/
    └── local/                # specialization for a local cluster (kind/minikube)
        └── kustomization.yaml # pins the image to the locally built ml-pipeline:local
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

## Run it locally (runbook)

The steps below are the operational flow for a local cluster (kind, minikube, or
Docker Desktop Kubernetes).

> **Executed on 2026-08-12** against Docker Desktop Kubernetes (v1.34.3). The Job
> lifecycle was verified end to end (see [§ Execution record](#execution-record-pr-2)):
> namespace + Job created, image resolved with no registry pull, 3 attempts (the
> initial pod + `backoffLimit: 2`), then a terminal `Failed`. The pipeline itself
> does **not** complete yet — `dvc repro` aborts with `/app is not a git
> repository`. A *green* run needs the PR 3 "make it runnable" work: an SCM in the
> image (`git init` or `core.no_scm=true`), a mounted dataset (the runtime image
> ships **no** data by design), and MLflow/DagsHub credentials.

### 1. Build the image locally

The workload uses the locally built production image — there is no registry to
pull from yet (see the root README § "Running with Docker").

```bash
docker build \
  --build-arg VCS_REF="$(git rev-parse --short HEAD)" \
  --build-arg BUILD_VERSION="1.3.1" \
  -t ml-pipeline:local .
```

### 2. Load the image into the cluster

A local cluster cannot see your host's Docker images until they are side-loaded:

```bash
kind load docker-image ml-pipeline:local          # kind
# or:
minikube image load ml-pipeline:local             # minikube
```

### 3. Apply the workload

```bash
kubectl apply -k k8s/overlays/local
```

This creates the `mlops` namespace and the `mlops-pipeline` Job.

### 4. Inspect it

```bash
kubectl -n mlops get jobs,pods                    # high-level status
kubectl -n mlops describe job/mlops-pipeline      # events, completions, backoff
kubectl -n mlops get job/mlops-pipeline -o wide
```

### 5. Retrieve logs

```bash
kubectl -n mlops logs job/mlops-pipeline          # logs from the Job's pod
kubectl -n mlops logs -f job/mlops-pipeline       # follow while it runs
```

### 6. Delete / re-run

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
| Green in-cluster `dvc repro` (SCM + data + credentials) | ⬜ deferred | PR 3+ |
| Security hardening (securityContext, seccomp, dropped caps) | ⬜ deferred | PR 4 |
| CPU/memory resource requests/limits, lifecycle tuning | ⬜ deferred | PR 5 |
| CI manifest validation | ⬜ deferred | PR 6 |
| Operations runbook & proof | ⬜ deferred | PR 7 |

No credentials are committed anywhere in this directory, and none will be — the
Secret strategy (PR 3) uses a committed **template without values**.
