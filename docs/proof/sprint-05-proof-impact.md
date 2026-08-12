# Sprint 5 — Proof-Impact Assessment (Kubernetes Platform Engineering)

- **Date:** 2026-08-12
- **Status:** Unreleased (Sprint 5). **No release tag is cut in this PR** — this
  assessment covers the work staged under `[Unreleased]` in the
  [CHANGELOG](../../CHANGELOG.md); a version tag is a separate release step.
- **Related:** [Kubernetes Architecture](../kubernetes-architecture.md),
  [Kubernetes Operations](../kubernetes-operations.md),
  [Kubernetes Security](../kubernetes-security.md), [`k8s/README.md`](../../k8s/README.md),
  [ADR-009](../decisions/ADR-009-kubernetes-workload-model.md),
  [ADR-010](../decisions/ADR-010-kubernetes-security-hardening.md),
  [ADR-011](../decisions/ADR-011-kubernetes-resource-lifecycle.md),
  [ADR-012](../decisions/ADR-012-kubernetes-manifest-validation.md)

> This document answers one question with evidence:
> **What can this project credibly claim after Sprint 5 that it could not after
> Sprint 4?** "Credibly" is the operative word — every claim below points to a
> manifest, a command, an ADR, or an executed run a reviewer can check.
> [§5](#5-what-still-cannot-be-claimed) lists what the repository does **not**
> support, so the credible claims are not diluted by overreach.

---

## 1. The shift, in one line

| | Claim licensed by the repository |
|-|----------------------------------|
| **After Sprint 4** | "I built a correct, reproducible, containerized ML pipeline validated in CI (lint, types, tests, DVC integrity, a fixture `dvc repro`, and an image build) — offline, with no credentials." |
| **After Sprint 5** | "I additionally expressed that containerized pipeline as a **Kubernetes-native batch workload** with an honest workload model (a `Job`, not a fake service), a least-privilege identity, an **enforced** hardened security context, **measured** resource requests/limits, and **automated static manifest validation in CI** — executed on a local cluster to verify the lifecycle and enforcement end to end, while documenting exactly what is not yet green." |

The Sprint 4 claim is about the **correctness of the pipeline**. The Sprint 5 claim
is about **platform-engineering judgment** applied to that pipeline: modelling the
workload correctly, scoping identity and privilege, governing resources from
measurement, and gating the manifests in CI — each with its trade-offs recorded and
its limits stated.

---

## 2. New credible claims, with evidence

Each row was **not** defensible after Sprint 4 and **is** after Sprint 5.

### 2.1 "I modelled the pipeline with the correct Kubernetes workload primitive."
- A `batch/v1` **`Job`** in a dedicated `mlops` namespace — not a `Deployment` (which
  would read the pipeline's normal exit `0` as a crash-loop) and with **no** invented
  HTTP API to justify a `Service`/`Ingress`.
- **Evidence:** [`k8s/base/job.yaml`](../../k8s/base/job.yaml),
  [`k8s/base/namespace.yaml`](../../k8s/base/namespace.yaml),
  [ADR-009](../decisions/ADR-009-kubernetes-workload-model.md); renders via
  `kustomize build k8s/overlays/local`.

### 2.2 "I gave the workload a least-privilege identity with no cluster API access."
- A dedicated `ServiceAccount` with `automountServiceAccountToken: false` (SA **and**
  pod), and **no** `Role`/`RoleBinding` — because the pipeline never calls the
  Kubernetes API.
- **Evidence:** [`k8s/base/serviceaccount.yaml`](../../k8s/base/serviceaccount.yaml);
  live pod had empty `spec.volumes`/`volumeMounts` (no token mounted).

### 2.3 "I externalized configuration and secrets without committing credentials."
- A `ConfigMap` for non-secret config; a `Secret` **template** (placeholders only),
  excluded from the Kustomize base and created out-of-band from a git-ignored `.env`.
- **Evidence:** [`k8s/base/configmap.yaml`](../../k8s/base/configmap.yaml),
  [`k8s/base/secret.example.yaml`](../../k8s/base/secret.example.yaml),
  [kubernetes-security.md §5](../kubernetes-security.md#5-secret--data-handling);
  `k8s/validate.py` secret-hygiene checks pass.

### 2.4 "I enforced a hardened security context at the platform layer."
- `runAsNonRoot` + numeric `runAsUser/Group 10001`, seccomp `RuntimeDefault`,
  `allowPrivilegeEscalation: false`, `capabilities.drop: [ALL]`; read-only root
  **evaluated and deferred with proof**.
- **Evidence:** [`k8s/base/job.yaml`](../../k8s/base/job.yaml),
  [ADR-010](../decisions/ADR-010-kubernetes-security-hardening.md); live pod
  reported the exact enforced context.

### 2.5 "I set resource requests/limits from measurement, not guesswork."
- `requests {cpu 250m, mem 256Mi}` / `limits {cpu 1, mem 512Mi}` → **Burstable** QoS,
  derived from `docker run` probes (1 CPU ≈ 133 MiB/2.5 s; the CPU limit bounds
  joblib's fan-out and therefore memory).
- **Evidence:** [ADR-011](../decisions/ADR-011-kubernetes-resource-lifecycle.md);
  live pod reported QoS `Burstable` and the exact `resources`; `--memory=64m` run
  `OOMKilled` (limit kernel-enforced).

### 2.6 "I gate the manifests in CI so the contract can't silently regress."
- A static, deterministic `k8s-validate` job: pinned `kustomize` render +
  `kubeconform -strict` schema + `k8s/validate.py` (34 checks), plus an opt-in
  ephemeral-kind server-side dry-run.
- **Evidence:** [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml),
  [ADR-012](../decisions/ADR-012-kubernetes-manifest-validation.md),
  [ci-cd.md](../ci-cd.md); negative tests caught a flipped `allowPrivilegeEscalation`
  and a string `activeDeadlineSeconds`.

### 2.7 "I documented and executed the operational path, and proved it."
- A deployment guide, an operations runbook with a troubleshooting matrix, and a
  security document — plus an **executed clean-state run** (deploy → observe → logs →
  cleanup) verifying the lifecycle and enforcement.
- **Evidence:** [`k8s/README.md`](../../k8s/README.md),
  [kubernetes-operations.md](../kubernetes-operations.md),
  [kubernetes-security.md](../kubernetes-security.md), and
  [§3](#3-proof--evidence) below.

---

## 3. Proof / evidence

The six dimensions the sprint set out to establish, each mapped to its enforcing
artifact and its executed/verified evidence. All cluster evidence is from a
clean-state run on **Docker Desktop Kubernetes v1.34.3, 2026-08-12**, image
`ml-pipeline:local`.

| Dimension | Enforcing artifact | Verified evidence |
|---|---|---|
| **Workload model** | `Job` in `mlops` ns, Kustomize base/overlay ([ADR-009](../decisions/ADR-009-kubernetes-workload-model.md)) | Renders to 4 objects; applied and ran its **3-attempt** back-off lifecycle (`RESTARTS: 0`, `restartPolicy: Never`) → Job `Failed` with `BackoffLimitExceeded`. |
| **Security controls** | pod/container `securityContext`, SA + token automount off ([ADR-010](../decisions/ADR-010-kubernetes-security-hardening.md)) | Live pod enforced `{runAsNonRoot, 10001, seccomp RuntimeDefault}` + `{allowPrivilegeEscalation:false, drop[ALL], readOnlyRootFilesystem:false}`; **empty** volumes (no API token). |
| **Resource management** | measured `requests`/`limits` ([ADR-011](../decisions/ADR-011-kubernetes-resource-lifecycle.md)) | Live pod: QoS `Burstable`, `resources` exactly `{250m/256Mi, 1/512Mi}`; `--memory=64m` → `OOMKilled` (137). |
| **Configuration management** | `ConfigMap` + out-of-band `Secret` template ([kubernetes-security.md](../kubernetes-security.md)) | ConfigMap holds only `LOG_LEVEL`/`MLFLOW_TRACKING_URI`; Secret template **not** rendered; pod started with the optional Secret absent. |
| **Validation** | `k8s-validate` CI job ([ADR-012](../decisions/ADR-012-kubernetes-manifest-validation.md)) | `python k8s/validate.py` → **34/34**; `kubeconform -strict` clean (base + overlay); negative tests caught by both tools. |
| **Operational procedures** | deployment guide + ops runbook + troubleshooting matrix | Executed deploy → observe (`get`/`describe`) → logs (`/app is not a git repository`) → cleanup (`delete -k`, ns `NotFound`). |

**The one honest negative that runs through all of it:** every attempt terminated at
the pre-existing SCM blocker (`ERROR: /app is not a git repository`, exit 255). The
Job **mechanism** and the **enforcement** of every control are proven on a real
cluster; a **green pipeline run** is **not** claimed — it needs an SCM in the image,
a mounted dataset, and credentials.

---

## 4. Before / After (conservative)

Capability status, stated conservatively — "✅" only where the repository has the
artifact **and** the evidence; "⬜" where deferred; "❌" where not attempted/claimed.

| Capability | After Sprint 4 | After Sprint 5 |
|---|---|---|
| Containerized pipeline | ✅ | ✅ |
| CI: lint / types / tests / DVC integrity / fixture `dvc repro` / image build | ✅ | ✅ (unchanged) |
| Kubernetes workload model (`Job`, namespace, Kustomize) | ❌ | ✅ |
| Externalized config + out-of-band secrets + least-privilege identity | ❌ | ✅ |
| Enforced hardened `securityContext` | ❌ | ✅ (read-only root ⬜ deferred) |
| Measured resource requests/limits + lifecycle/probe decisions | ❌ | ✅ (not production-certified) |
| Automated **static** manifest validation in CI | ❌ | ✅ |
| Executed local-cluster run (lifecycle + enforcement verified) | ❌ | ✅ |
| **Green** in-cluster `dvc repro` | ❌ | ⬜ deferred (SCM + data + creds) |
| Restricted PSS **certification** | ❌ | ❌ not claimed |
| Production cloud deployment / GitOps / HA / serving / prod observability | ❌ | ❌ not claimed (roadmap v5–v6) |

---

## 5. What still **cannot** be claimed

Documented so none is accidentally implied:

- **❌ A green in-cluster pipeline run.** `dvc repro` aborts at the SCM check; the
  mechanism is proven, the completed run is not.
- **❌ Restricted Pod Security Standard compliance.** Controls applied and
  cluster-admitted, but no admission label/policy engine ratifies the profile, and
  read-only root is deferred.
- **❌ Production Kubernetes deployment.** Everything is validated on a **local**
  single-node cluster. No managed cluster (EKS/AKS/GKE), no cloud IaC.
- **❌ GitOps / continuous delivery to a cluster.** CI **validates** manifests
  statically; it does not deploy them (see [ci-cd.md](../ci-cd.md)).
- **❌ High availability / autoscaling / model serving.** The workload is a single
  finite batch Job; none of these are present (roadmap v5–v6).
- **❌ Production observability.** No metrics/tracing/alerting stack; diagnosis is
  `kubectl` + structured logs.
- **❌ Production-certified capacity.** Resource values are measured for a *local*
  run on the small bundled dataset; a larger dataset/grid/cluster needs re-measuring.
- **❌ Deployment validation from CI.** CI performs **static** checks (schema,
  render, security/required fields) plus an opt-in admission dry-run — not a deploy.

---

## 6. Known limitations (explicit)

- **Local cluster only** — Docker Desktop / kind / minikube; no managed cloud cluster.
- **No production cloud deployment.**
- **No GitOps** (Argo CD / Flux).
- **No production observability stack** (Prometheus/Grafana, tracing, alerting).
- **No production HA claims** — single finite Job, single node.
- **No production serving claims** — batch workload, no inference endpoint.
- **Read-only root filesystem deferred** — with recorded evidence (ADR-010).
- **No `NetworkPolicy`, no admission-policy engine, no secrets manager** — plain
  Kubernetes `Secret`, unrestricted cluster-layer egress.
- **No supply-chain provenance** — name-pinned image/base (not digest-pinned/signed).

---

## 7. The honest one-paragraph statement

> Building on a correct, reproducible, CI-validated containerized ML pipeline, I
> expressed it as a Kubernetes-native **batch workload** and made platform-
> engineering decisions I can defend: I modelled it as a `Job` (not a `Deployment`,
> and without inventing a service to justify one), gave it a least-privilege
> `ServiceAccount` with no API token and no RBAC, externalized configuration and
> kept credentials out of git via an out-of-band `Secret`, **enforced** a hardened
> pod/container security context at the platform layer, set resource requests and
> limits from **measured** usage, and **gated the manifests in CI** with static
> schema/Kustomize/security validation plus an opt-in cluster admission dry-run. I
> executed the whole path on a local cluster and verified the lifecycle and the
> enforcement end to end. Deliberately, I do **not** claim a green in-cluster run
> (the image lacks an SCM), restricted-PSS certification, or any production/cloud
> deployment — each is documented as deferred or out of scope. The claim is about
> **platform-engineering judgment on a local cluster**, evidenced and bounded, not
> about production Kubernetes operation.

That paragraph is fully supported by the repository. Its restraint — naming the
non-green run and the local-only boundary out loud — is part of what makes the rest
of it credible.
