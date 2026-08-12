# Kubernetes Manifests (`k8s/`)

Kubernetes deployment surface for the End-to-End ML Pipeline. This directory
holds the **architectural foundation** for running the pipeline as a
Kubernetes-native **batch workload** (a `Job`), not a long-running service.

> **Status — foundation (Sprint 5, PR 1).** This PR establishes the structure,
> the namespace boundary, and the workload *model*. It intentionally does **not**
> yet include configuration/secrets, security hardening, resource limits, CI
> validation, or a demonstrated cluster run — those land in later, focused PRs
> (see [§ Roadmap within Sprint 5](#roadmap-within-sprint-5)). Nothing here has
> been applied to a production cluster.

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
# Base (registry-neutral image reference)
kustomize build k8s/base

# Local overlay (image mapped to the locally built ml-pipeline:local)
kustomize build k8s/overlays/local
```

`kubectl` can render the same way with `kubectl kustomize k8s/overlays/local`.

## Apply locally (forward-looking)

A demonstrated local run belongs to the **batch-workload PR (PR 2)**; the steps
below are the intended flow and require the data/credential wiring added in PR 3
before `dvc repro` can succeed end to end.

```bash
# 1. Build the image (see the root README § "Running with Docker")
docker build -t ml-pipeline:local .

# 2. Make it available to the local cluster (kind example)
kind load docker-image ml-pipeline:local

# 3. Apply the local overlay
kubectl apply -k k8s/overlays/local

# 4. Observe the Job
kubectl -n mlops get jobs,pods
kubectl -n mlops logs job/mlops-pipeline
```

Remove the workload with `kubectl delete -k k8s/overlays/local`.

## What is deliberately absent (and where it lands)

### Roadmap within Sprint 5

| Concern | Status | Target PR |
|---|---|---|
| Namespace + workload model (`Job`) | ✅ this PR | PR 1 |
| Kustomize base/overlay structure | ✅ this PR | PR 1 |
| Demonstrated local cluster run | ⬜ deferred | PR 2 |
| ConfigMap / Secret / ServiceAccount | ⬜ deferred | PR 3 |
| Security hardening (securityContext, seccomp, dropped caps) | ⬜ deferred | PR 4 |
| Resource requests/limits, lifecycle tuning | ⬜ deferred | PR 5 |
| CI manifest validation | ⬜ deferred | PR 6 |
| Operations runbook & proof | ⬜ deferred | PR 7 |

No credentials are committed anywhere in this directory, and none will be — the
Secret strategy (PR 3) uses a committed **template without values**.
