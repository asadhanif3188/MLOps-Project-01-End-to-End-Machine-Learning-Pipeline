# Kubernetes Operations

Day-2 operations for the ML pipeline running as a Kubernetes **batch `Job`** on a
**local** cluster. This is the operator's runbook: how to deploy, observe, retrieve
logs, re-run, rotate configuration/secrets, tear down, and diagnose failures with a
symptom → cause → investigation → remediation matrix.

> **Scope — local operations, honestly bounded.** Everything here targets a
> **local** single-node cluster (Docker Desktop Kubernetes, kind, or minikube). It
> is deliberately **not** a production operations manual: there is no managed
> cluster, no high-availability topology, no GitOps reconciler, no production
> observability stack (metrics/traces/alerting), and no model-serving endpoint —
> those are future roadmap items ([roadmap.md](roadmap.md) v5–v6), not capabilities
> this repository has. Observability here is `kubectl` + the pipeline's structured
> logs, which is appropriate for a finite batch Job on a laptop and is not
> presented as more than that. No production Kubernetes operational expertise is
> claimed.

For the manifests and the first-run deployment walkthrough see
[`k8s/README.md`](../k8s/README.md); for the architecture and the workload model
see [kubernetes-architecture.md](kubernetes-architecture.md); for the security
posture see [kubernetes-security.md](kubernetes-security.md). Design of record:
[ADR-009](decisions/ADR-009-kubernetes-workload-model.md),
[ADR-010](decisions/ADR-010-kubernetes-security-hardening.md),
[ADR-011](decisions/ADR-011-kubernetes-resource-lifecycle.md),
[ADR-012](decisions/ADR-012-kubernetes-manifest-validation.md),
[ADR-013](decisions/ADR-013-kubernetes-runtime-execution.md).

---

## 1. Operational model

The workload is a **finite batch `Job`** (`mlops-pipeline`) in the **`mlops`**
namespace. It is not a service: it has no port, no `Service`/`Ingress`, no replicas,
and no steady state to keep alive. An operator's job is therefore not "keep it up"
but "start a run, watch it complete or fail, read the result, clean up, re-run if
needed." Its health is **terminal**, not continuous:

| Signal | Meaning |
|---|---|
| Pod exits `0` → Job `Complete` | Success — the run finished and wrote its artifacts/metrics. |
| Pod exits non-zero → retried up to `backoffLimit`, then Job `Failed` | Failure — read the pod logs; a deterministic pipeline fails identically on each attempt. |
| Job still `Running` past `activeDeadlineSeconds` (1800s) | Stall — the Job controller terminates it with `DeadlineExceeded`. |

There are **no liveness/readiness probes** by design (a finite Job has nothing to
poll — see [ADR-011](decisions/ADR-011-kubernetes-resource-lifecycle.md)), so the
authoritative status comes from the Job controller (`kubectl get job`) and the
container exit code, and the diagnosis comes from the structured logs.

> **Current state (PR 8 — green).** On a local cluster the **complete pipeline runs
> to completion**: the Job reaches `Complete`, the pod `Succeeds` with exit 0, and
> all four stages run (preprocess → split → train → evaluate). This is achieved by
> the runtime contract in
> [ADR-013](decisions/ADR-013-kubernetes-runtime-execution.md): DVC no-SCM
> (`core.no_scm=true` via a mounted `config.local`) replaces the earlier
> `/app is not a git repository` abort; the dataset is mounted read-only at
> `/app/data/raw` from an out-of-band ConfigMap; and the local overlay points MLflow
> at an in-pod file store (offline, no credentials). See
> [§6](#6-operational-evidence). The dataset ConfigMap and MLflow file store are
> **local-validation mechanisms, not production storage**.

---

## 2. Routine operations

All commands assume `kubectl` is pointed at a **local** context
(`kubectl config current-context` → `docker-desktop`, `kind-*`, or `minikube`) and
the image `ml-pipeline:local` is available to the cluster (see
[k8s/README §Build & load](../k8s/README.md#2-make-the-image-available-to-the-cluster)).

### Deploy a run

For a **green** run, create the out-of-band dataset ConfigMap first (the image ships
no data; the volume is `optional: true` so a missing dataset fails fast at
preprocess rather than blocking the pod). Then apply:

```bash
kubectl create namespace mlops --dry-run=client -o yaml | kubectl apply -f -
kubectl create configmap mlops-pipeline-dataset -n mlops \
  --from-file=data.csv=data/raw/data.csv      # local-validation dataset (ADR-013)
kubectl apply -k k8s/overlays/local           # creates namespace, SA, ConfigMaps, Job
```

On Docker Desktop, also ensure the fresh image is in the node's containerd store
(see [k8s/README §2](../k8s/README.md#2-make-the-image-available-to-the-cluster)) —
otherwise the pod may run a stale cached image.

### Observe it

```bash
kubectl -n mlops get jobs,pods -o wide        # high-level status + node/IP
kubectl -n mlops describe job/mlops-pipeline  # events: SuccessfulCreate, BackoffLimitExceeded
kubectl -n mlops get job/mlops-pipeline -o jsonpath='{.status}{"\n"}'   # completions/failed
```

Confirm the *enforced* runtime contract on the live pod (useful when verifying a
change did what you intended):

```bash
pod=$(kubectl -n mlops get pods -o jsonpath='{.items[0].metadata.name}')
kubectl -n mlops get pod "$pod" -o jsonpath=\
'QoS={.status.qosClass}{"\n"}sc={.spec.securityContext}{"\n"}res={.spec.containers[0].resources}{"\n"}vol={.spec.volumes}{"\n"}'
# Expect: QoS=Burstable; runAsNonRoot/10001 + seccomp RuntimeDefault; the 250m/256Mi–1/512Mi
# block; and exactly TWO read-only ConfigMap volumes (dvc-runtime-config, dataset) —
# and NO `kube-api-access-*` projected token (automountServiceAccountToken:false). The
# no-API-token check is now "no projected serviceaccount-token volume", not "empty volumes"
# (PR 8 added the two read-only runtime mounts; see ADR-013).
```

### Retrieve logs

```bash
kubectl -n mlops logs job/mlops-pipeline      # logs from the Job's (first) pod
kubectl -n mlops logs -f job/mlops-pipeline   # follow while running
# A specific attempt (each retry is a fresh pod, independently inspectable):
kubectl -n mlops logs mlops-pipeline-<suffix>
```

### Re-run

A Job's pod template is immutable, so a re-run is delete-then-apply — you are
starting a *fresh* run, not mutating a live one:

```bash
kubectl -n mlops delete job/mlops-pipeline    # keep the namespace/config/secret
kubectl apply -k k8s/overlays/local           # re-create the Job for a new run
```

### Update configuration

`LOG_LEVEL` and `MLFLOW_TRACKING_URI` live in the `mlops-pipeline-config`
ConfigMap. Edit [`k8s/base/configmap.yaml`](../k8s/base/configmap.yaml) and
re-apply; the value is read at container start, so the change takes effect on the
**next** Job run (a running pod does not see it):

```bash
kubectl apply -k k8s/overlays/local           # applies the updated ConfigMap
kubectl -n mlops delete job/mlops-pipeline && kubectl apply -k k8s/overlays/local   # re-run to pick it up
```

### Rotate the credential Secret

The Secret is created **out-of-band** (never committed — see
[kubernetes-security.md](kubernetes-security.md)). Rotate by replacing it; the next
run picks up the new values:

```bash
kubectl create secret generic mlops-pipeline-secret --namespace mlops \
  --from-env-file=.env --dry-run=client -o yaml | kubectl apply -f -
```

### Tear down

```bash
kubectl delete -k k8s/overlays/local          # removes the Job, ConfigMap, SA, and namespace
# Deleting the namespace also garbage-collects the out-of-band Secret it contained.
kubectl get ns mlops                           # -> NotFound
```

---

## 3. Troubleshooting matrix

Authoritative operational diagnosis table. Symptoms are what `kubectl get`/`describe`
or the logs actually show; remediation is the concrete fix. The failure-mode
rationale is in [ADR-011 §Failure modes](decisions/ADR-011-kubernetes-resource-lifecycle.md).

| Symptom | Likely cause | Investigation | Remediation |
|---|---|---|---|
| Pod stuck `Pending` | Node lacks schedulable CPU/memory for the `requests` (250m / 256Mi), or the local cluster is under-provisioned. | `kubectl -n mlops describe pod <pod>` → `Events` (`FailedScheduling`, "Insufficient cpu/memory"). | Free resources or raise the local cluster's node size (Docker Desktop → Settings → Resources); the requests themselves are modest and measured (ADR-011). |
| Pod `ImagePullBackOff` / `ErrImagePull` | The cluster can't see `ml-pipeline:local` — it was not built, or not side-loaded into a kind/minikube node. | `kubectl -n mlops describe pod <pod>` → `Events` (`Failed to pull image`). | Build it (`docker build -t ml-pipeline:local .`) and, on kind/minikube, side-load it (`kind load docker-image ml-pipeline:local` / `minikube image load ml-pipeline:local`). Docker Desktop shares the daemon — no load needed. See [k8s/README §2](../k8s/README.md#2-make-the-image-available-to-the-cluster). |
| Pod `CreateContainerConfigError`, message "container has runAsNonRoot and image has non-numeric user" | `runAsNonRoot: true` with only a *name* (`appuser`) resolvable — the numeric `runAsUser` is missing/removed. | `kubectl -n mlops describe pod <pod>` → container state. | Keep the explicit `runAsUser: 10001` in the pod `securityContext` (it is **required**, not cosmetic — [ADR-010](decisions/ADR-010-kubernetes-security-hardening.md)). |
| Pod `Error`, log `ERROR: /app is not a git repository` | The DVC no-SCM config isn't mounted (`core.no_scm=true` in `/app/.dvc/config.local`). Resolved by default in PR 8. | `kubectl -n mlops exec <pod> -- cat .dvc/config.local` (should show `no_scm = true`). | Ensure `k8s/base/dvc-config.yaml` is in the base and the Job mounts it (subPath) — [ADR-013](decisions/ADR-013-kubernetes-runtime-execution.md); `python k8s/validate.py` asserts it. |
| Pod `Error`, log `No such file or directory: '/app/data/raw/data.csv'` | Dataset ConfigMap not created (volume is `optional: true`, so the pod starts then fails fast). | `kubectl -n mlops get configmap mlops-pipeline-dataset`. | Create it out-of-band (see [§2 Deploy a run](#deploy-a-run)); this is the intended missing-input failure. |
| Pod runs **stale code** (DVC errors about stages/params not matching the repo; unexpected old behaviour) | Docker Desktop k8s uses a containerd image store separate from `docker build`; the kubelet has an **old cached** `ml-pipeline:local`. | `kubectl -n mlops exec <pod> -- md5sum dvc.yaml` vs `docker run --rm --entrypoint md5sum ml-pipeline:local dvc.yaml` (differ ⇒ stale). | Import the fresh image into the node's `k8s.io` containerd namespace ([k8s/README §2](../k8s/README.md#2-make-the-image-available-to-the-cluster)); delete + re-apply the Job. |
| Pod `OOMKilled`, exit `137` | Container exceeded `limits.memory` (512Mi) — e.g. a much larger dataset or a wider grid than the measured envelope. | `kubectl -n mlops get pod <pod> -o jsonpath='{.status.containerStatuses[0].lastState.terminated.reason}'` → `OOMKilled`. | Re-measure the workload and raise `limits.memory` deliberately (ADR-011 documents the measured 1-CPU ≈133 MiB peak); do not raise blindly. The limit is kernel-enforced by design. |
| Job `Failed` with `BackoffLimitExceeded` | The pod failed `backoffLimit + 1 = 3` times. | `kubectl -n mlops describe job/mlops-pipeline` → `Events`; then the **pod logs** for the real cause. | Fix the underlying pod error (see the exit-code rows above); `backoffLimit: 2` is intentional fail-fast for a deterministic pipeline. |
| Job terminated `DeadlineExceeded` | The run exceeded `activeDeadlineSeconds` (1800s) — a stall (e.g. a hung MLflow/DagsHub call), not normal compute (which is sub-minute). | `kubectl -n mlops describe job/mlops-pipeline` → condition `DeadlineExceeded`. | Investigate the stall (network to the tracking endpoint); the deadline is an outer stall-guard, not an SLO (ADR-011). |
| Container starts, then MLflow **auth** error in logs | The optional `mlops-pipeline-secret` is absent or holds wrong credentials (`secretRef.optional: true`, so the pod still starts). | `kubectl -n mlops get secret mlops-pipeline-secret`; read the logs at the tracking boundary. | Create/rotate the Secret out-of-band from `.env` (see [§2 Rotate](#rotate-the-credential-secret)). |
| `kubectl apply -k` fails: "may not add resource with an already registered id" or a Kustomize render error | A malformed manifest or a duplicate resource in `base`/overlay. | `kustomize build k8s/overlays/local` locally to see the render error; run `python k8s/validate.py`. | Fix the manifest; the CI `k8s-validate` job catches this class before merge ([ADR-012](decisions/ADR-012-kubernetes-manifest-validation.md)). |
| `kubectl apply -k` rejected by the API server (`--dry-run=server` or real apply) | A schema-invalid field (wrong type, unknown key) that renders but the API server rejects. | Reproduce offline: `kustomize build …` piped to `kubeconform -strict …`; or `kubectl apply -k … --dry-run=server`. | Fix the field; `kubeconform -strict` in CI rejects unknown/mistyped fields ([ci-cd.md](ci-cd.md)). |
| Pod carries a `/var/run/secrets/kubernetes.io/serviceaccount` mount | `automountServiceAccountToken` was flipped/removed — the pipeline needs **no** API token. | `kubectl -n mlops get pod <pod> -o jsonpath='{.spec.volumes[*].name}'` — expect only `dvc-runtime-config` and `dataset` (both read-only ConfigMaps), and **no** `kube-api-access-*` projected-token volume. | Restore `automountServiceAccountToken: false` on both the ServiceAccount and the pod template ([kubernetes-security.md](kubernetes-security.md)). |

---

## 4. Failure handling playbook

1. **Read the Job condition first** — `kubectl -n mlops get job/mlops-pipeline -o jsonpath='{.status.conditions}'`.
   It distinguishes `BackoffLimitExceeded` (pod kept failing) from `DeadlineExceeded`
   (stall).
2. **Then read the pod logs** — the Job condition tells you *that* it failed; the
   pod logs tell you *why*. For a deterministic pipeline all attempts fail the same
   way, so the first pod's logs are representative.
3. **Check the container exit code / reason** for the failure *class*:
   `137 = OOMKilled` (resource), `255/Error` (application — today the SCM blocker),
   config/secret errors surface in the logs at the relevant boundary.
4. **Reproduce statically before re-applying** — `python k8s/validate.py` and
   `kustomize build … | kubeconform -strict …` catch manifest regressions without a
   cluster; use them before blaming the cluster.
5. **Re-run cleanly** — delete the Job (or the whole overlay) before re-applying; a
   Job's template is immutable, so "edit and re-apply" on an existing Job is
   rejected, not merged.

---

## 5. Observability posture (honest)

What exists, and what does not:

| Capability | Status | How |
|---|---|---|
| Run status / completion | ✅ | `kubectl get job` + container exit code (the Job controller is the source of truth). |
| Per-attempt diagnosis | ✅ | `kubectl logs` per pod — each retry is a fresh, independently inspectable pod. |
| Structured application logs | ✅ | The pipeline logs through `src/logging_config.py` (level via `LOG_LEVEL`); logs go to stdout and are captured by the container runtime. |
| Events / scheduling / lifecycle | ✅ | `kubectl describe job/pod` → `Events`. |
| Resource enforcement visibility | ✅ | `kubectl get pod -o jsonpath='{.status.qosClass}'` / `{.spec.containers[0].resources}` — verified `Burstable`. |
| Metrics server / `kubectl top` | ⚠️ optional | Not part of this repo; available if the local cluster ships metrics-server, but not required or assumed. |
| Production observability stack (Prometheus/Grafana, tracing, alerting, log aggregation) | ❌ not implemented | Roadmap v6 ([roadmap.md](roadmap.md)). **Not claimed.** |

This is deliberately minimal: a finite batch Job on a local cluster is fully
diagnosable with `kubectl` + structured logs, and standing up an observability stack
for a laptop run would be architecture inflation. A production deployment would add
one — that is future work, not a present capability.

---

## 6. Operational evidence

The full local operational path was **executed from a clean state** (2026-08-12,
Docker Desktop Kubernetes v1.34.3, image `ml-pipeline:local`) and each step
produced the expected result:

| Step | Command | Observed result |
|---|---|---|
| Render | `kubectl kustomize k8s/overlays/local` | 4 objects: `Namespace`, `ServiceAccount`, `ConfigMap`, `Job`. |
| Static validation | `python k8s/validate.py` | **34/34** checks pass. |
| Deploy | `kubectl apply -k k8s/overlays/local` | `namespace/mlops`, `serviceaccount/mlops-pipeline`, `configmap/mlops-pipeline-config`, `job.batch/mlops-pipeline` created; image resolved with **no registry pull**. |
| Lifecycle | `kubectl -n mlops get jobs,pods` | **3 attempts** (initial + `backoffLimit: 2`), each a fresh pod with `RESTARTS: 0` (`restartPolicy: Never`), then Job `Failed` (`status.failed: 3`). |
| Events | `kubectl -n mlops describe job/mlops-pipeline` | 3× `SuccessfulCreate`, then `BackoffLimitExceeded`. |
| Enforced runtime | `kubectl -n mlops get pod <pod> -o jsonpath=…` | QoS `Burstable`; pod `{runAsNonRoot:true, runAsUser:10001, runAsGroup:10001, seccompProfile:RuntimeDefault}`; container `{allowPrivilegeEscalation:false, capabilities.drop:[ALL], readOnlyRootFilesystem:false}`; `resources` exactly `{requests: cpu 250m/mem 256Mi, limits: cpu 1/mem 512Mi}`; `serviceAccountName: mlops-pipeline`, `automountServiceAccountToken:false`, **empty** `volumes`/`volumeMounts`. |
| Logs | `kubectl -n mlops logs job/mlops-pipeline` | `ERROR: /app is not a git repository` (identical across attempts). |
| Cleanup | `kubectl delete -k k8s/overlays/local` | all four objects deleted; `kubectl get ns mlops` → `NotFound`. |

**What the 2026-08-12 run proved:** the deployment, observation, log-retrieval, and
cleanup operations work as documented, and the platform *enforces* the declared
security and resource contract. It did **not** prove a green pipeline run — every
attempt terminated at the then-open SCM blocker (exit 255).

### PR 8 addendum — green run (2026-08-14)

With the PR 8 runtime contract ([ADR-013](decisions/ADR-013-kubernetes-runtime-execution.md))
the same operational path was re-executed and the **pipeline completed**:

| Step | Command | Observed result |
|---|---|---|
| Static validation | `python k8s/validate.py` | **43/43** checks pass (now incl. the runtime-contract section). |
| Prepare inputs | `kubectl create configmap mlops-pipeline-dataset … --from-file=data.csv=…` | Out-of-band dataset ConfigMap created (local-validation). |
| Deploy | `kubectl apply -k k8s/overlays/local` | Namespace, SA, ConfigMaps, Job created. |
| Complete | `kubectl -n mlops wait --for=condition=complete job/mlops-pipeline` | Job `Complete` (`succeeded: 1`) in ~41s. |
| Pod result | `kubectl -n mlops get pod <pod> -o jsonpath=…` | `Succeeded`, container **exit 0**, `RESTARTS: 0` (first attempt). |
| Stages | `kubectl -n mlops logs job/mlops-pipeline` | preprocess (768 rows) → split (614/154) → train (acc 0.7398) → evaluate (acc 0.7078). |
| Failure test | remove dataset ConfigMap, re-apply | fail-fast at preprocess → 3 fresh-pod attempts → `Failed: BackoffLimitExceeded`; restoring the ConfigMap returns it to green. |

**What this proves:** the containerized MLOps pipeline runs to completion as a
secured Kubernetes Job, and its runtime-integration failures (SCM, dataset, MLflow,
stale image) are diagnosable and fixable. **Still not claimed:** production storage,
production MLflow connectivity (the local run uses a file store), or any
production/cloud deployment ([§7](#7-known-limitations)).

---

## 7. Known limitations

Explicitly, so nothing here is over-read:

- **Local cluster only.** Validated on Docker Desktop Kubernetes; kind/minikube are
  supported by the same runbook. No managed cluster (EKS/AKS/GKE) is used or claimed.
- **Green run is local-only.** The complete pipeline runs to completion in-cluster
  (PR 8, [ADR-013](decisions/ADR-013-kubernetes-runtime-execution.md)), but with a
  **local-validation** dataset (out-of-band ConfigMap, ≤1 MiB) and an **in-pod
  MLflow file store** — not production dataset storage or a real MLflow server.
  DagsHub connectivity is configuration-validated, not connectivity-tested locally.
- **No production cloud deployment**, **no GitOps** (Argo CD/Flux), **no HA**, and
  **no model serving** — all roadmap v5–v6, none present.
- **No production observability stack** — diagnosis is `kubectl` + structured logs
  (see [§5](#5-observability-posture-honest)).
- **Resource values are not production-certified** — tuned for a local single-node
  run on the small bundled dataset ([ADR-011](decisions/ADR-011-kubernetes-resource-lifecycle.md)).
- **Restricted Pod Security Standard compliance is not claimed** — the controls are
  applied and cluster-admitted, but no admission label/policy engine ratifies the
  profile, and read-only root is deferred ([kubernetes-security.md](kubernetes-security.md),
  [ADR-010](decisions/ADR-010-kubernetes-security-hardening.md)).

---

## Related documentation

- [`k8s/README.md`](../k8s/README.md) — manifests + first-run deployment guide
- [Kubernetes Architecture](kubernetes-architecture.md) — workload model and boundaries
- [Kubernetes Security](kubernetes-security.md) — the security posture in depth
- [Sprint 5 — Proof Impact](proof/sprint-05-proof-impact.md) — evidence-based claims
- [ADR-009](decisions/ADR-009-kubernetes-workload-model.md) ·
  [ADR-010](decisions/ADR-010-kubernetes-security-hardening.md) ·
  [ADR-011](decisions/ADR-011-kubernetes-resource-lifecycle.md) ·
  [ADR-012](decisions/ADR-012-kubernetes-manifest-validation.md)
- [CI/CD](ci-cd.md) — how the manifests are validated in CI
