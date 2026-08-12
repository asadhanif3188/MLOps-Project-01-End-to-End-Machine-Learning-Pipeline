# Kubernetes Manifests (`k8s/`)

Kubernetes deployment surface for the End-to-End ML Pipeline. This directory
holds the **architectural foundation** for running the pipeline as a
Kubernetes-native **batch workload** (a `Job`), not a long-running service.

> **Status — runnable batch workload (Sprint 5, PR 2).** PR 1 established the
> structure, namespace boundary, and workload *model*; PR 2 makes it the actual
> **runnable** workload — the real `ml-pipeline:local` image, the real `dvc repro`
> command, and a finite-run lifecycle (`restartPolicy: Never`, `backoffLimit: 2`,
> `activeDeadlineSeconds: 1800`). It intentionally does **not** yet include
> configuration/secrets, security hardening, or CPU/memory resource limits — those
> land in later, focused PRs (see [§ Roadmap within Sprint 5](#roadmap-within-sprint-5)).
> A cluster run has **not** been demonstrated in this repo's dev environment (no
> local cluster/Docker was available), and a *green* `dvc repro` additionally
> needs the PR 3 data/credential wiring. Nothing here has been applied to a
> production cluster.

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

The steps below are the operational flow for a local cluster (kind or minikube).

> **Not yet demonstrated here.** This runbook was authored and its manifests were
> validated offline (render + parse + field assertions), but it was **not
> executed** in this repo's dev environment because no local cluster or running
> Docker daemon was available. It is written to be run as-is. Also note a *green*
> `dvc repro` needs the data/credential wiring from PR 3 (the runtime image ships
> **no** data by design); until then the Job runs but fails fast at `preprocess`
> for lack of a mounted dataset — which still exercises the Job lifecycle
> (scheduling, retry, terminal state).

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

## What is deliberately absent (and where it lands)

### Roadmap within Sprint 5

| Concern | Status | Target PR |
|---|---|---|
| Namespace + workload model (`Job`) | ✅ done | PR 1 |
| Kustomize base/overlay structure | ✅ done | PR 1 |
| Runnable workload (real image, command, lifecycle) | ✅ this PR | PR 2 |
| Local run **runbook** (build/load/apply/inspect/logs/re-run) | ✅ this PR | PR 2 |
| Demonstrated local cluster run (executed) | 🟡 runbook ready, not executed (no cluster in dev env) | PR 2 |
| ConfigMap / Secret / ServiceAccount | ⬜ deferred | PR 3 |
| Security hardening (securityContext, seccomp, dropped caps) | ⬜ deferred | PR 4 |
| CPU/memory resource requests/limits, lifecycle tuning | ⬜ deferred | PR 5 |
| CI manifest validation | ⬜ deferred | PR 6 |
| Operations runbook & proof | ⬜ deferred | PR 7 |

No credentials are committed anywhere in this directory, and none will be — the
Secret strategy (PR 3) uses a committed **template without values**.
