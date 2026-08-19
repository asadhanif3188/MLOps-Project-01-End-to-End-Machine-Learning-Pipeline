# Kubernetes Architecture

How the End-to-End ML Pipeline runs as a **Kubernetes-native batch workload**.
This document describes the architecture established across Sprint 5: the workload
model and namespace boundary (PR 1), the **runnable batch Job** with its real
image and lifecycle (PR 2), the configuration boundary (PR 3), and the
**enforced security context** (PR 4), and — critically — exactly which parts are
locally validatable today versus deferred to a production cluster.

> **Scope note.** Through PR 7 the manifests in [`k8s/`](../k8s/) define the
> namespace, a **runnable** batch `Job` — the real `ml-pipeline:local` image, the
> real `dvc repro` command, and a finite-run lifecycle — plus externalized
> **configuration** (`ConfigMap`), a **Secret** template (created out-of-band,
> never committed), a least-privilege **ServiceAccount** with the API-token
> automount off, a **hardened `securityContext`** (non-root with an explicit
> uid/gid, no privilege escalation, all capabilities dropped, seccomp
> `RuntimeDefault`), and **resource requests/limits chosen from measured usage**
> (see [ADR-011](decisions/ADR-011-kubernetes-resource-lifecycle.md)); all render
> cleanly through Kustomize. PR 6 added **automated CI validation** of these
> manifests — static YAML-syntax, upstream **schema** (`kubeconform`), **Kustomize**
> rendering, and the security/resource contract (`k8s/validate.py`), plus an opt-in
> ephemeral-cluster admission dry-run
> ([ADR-012](decisions/ADR-012-kubernetes-manifest-validation.md)); this validation
> is **static** — it does not deploy or run the workload (see
> [§6](#6-identity--security-boundary) and
> [§7](#7-local-validation-vs-production-deferred)). PR 7 completes the sprint with
> **operations & proof** documentation — a full deployment guide
> ([`k8s/README.md`](../k8s/README.md)), an
> [operations runbook](kubernetes-operations.md) with a troubleshooting matrix, a
> [security document](kubernetes-security.md), and a
> [Sprint 5 Proof-Impact Assessment](proof/sprint-05-proof-impact.md) — and
> re-executes the local deployment path from a clean state as evidence. The Job was
> **executed on a
> local Docker Desktop cluster** (2026-08-12) and its lifecycle verified end to
> end, but the pipeline does **not** complete yet: `dvc repro` aborts because the
> image has no SCM (`/app is not a git repository`), and a *green* run also needs
> the PR 3 data/credential wiring. One control — **read-only root filesystem — is
> deliberately deferred** (DVC writes state in-tree; see [§6](#6-identity--security-boundary)
> and [ADR-010](decisions/ADR-010-kubernetes-security-hardening.md)). Nothing here
> has been deployed to a production cluster, and **restricted Pod Security Standard
> compliance is not claimed**. Claims are kept to what the repository can currently
> prove.

Design of record: [ADR-009 — Kubernetes Workload Model](decisions/ADR-009-kubernetes-workload-model.md).
Related: [Architecture](architecture.md), [Containerization](containerization.md),
[ADR-005](decisions/ADR-005-containerization-strategy.md), [Roadmap](roadmap.md).

---

## 1. Why Kubernetes

The pipeline is already containerized ([ADR-005](decisions/ADR-005-containerization-strategy.md)):
a non-root, multi-stage image runs `dvc repro` to completion. Containerization
gives environment parity; it does not give a **scheduler, an environment
boundary, a declarative configuration/identity/security contract, or a
reproducible way to hand the workload to a platform** instead of a laptop.

Kubernetes is introduced to demonstrate that the containerized workload can be
expressed as a first-class platform citizen with explicit, reviewable contracts:

- a **namespace** as the environment/RBAC/quota boundary,
- a **workload primitive** that matches the workload's real lifecycle,
- **configuration and secrets** separated from the image (design contract; PR 3),
- a **security context** and **least-privilege identity** (design contract; PR 4),
- **resource requests/limits** for reproducible scheduling (design contract; PR 5).

This is deliberately *not* "add Kubernetes because it is on the résumé." The test
for every manifest is whether it lets the repository prove platform-engineering
judgment — workload modelling, boundaries, and trade-offs — not merely
familiarity with Kubernetes resource kinds.

---

## 2. Workload Architecture

The workload is the existing four-stage DVC pipeline, unchanged:
`preprocess → split → train → evaluate` (`dvc repro`). It is a **finite, batch,
file-based** computation: it starts, does work, writes artifacts and metrics, and
**exits**. There is no request/response surface and nothing to keep alive.

```mermaid
flowchart TD
    dev["Developer<br/><i>kubectl apply -k k8s/overlays/local</i>"]
    subgraph cluster["Kubernetes cluster (local: kind / minikube)"]
        subgraph ns["Namespace: mlops"]
            job["Job: mlops-pipeline<br/><i>batch/v1 · restartPolicy: Never · backoffLimit: 2 · activeDeadlineSeconds: 1800</i>"]
            pod["Pod (one attempt)"]
            container["ML container<br/><i>ml-pipeline:local · CMD: dvc repro</i>"]
            job --> pod --> container
            subgraph pipeline["DVC pipeline (dvc repro)"]
                direction TB
                pre["preprocess"] --> split["split"] --> train["train"] --> evaluate["evaluate"]
            end
            container --> pipeline
            artifacts["Artifacts + metrics<br/><i>models/model.pkl · metrics/metrics.json</i>"]
            evaluate --> artifacts
        end
        completion(["Job completion<br/><i>pod exits 0 → Job Complete</i>"])
        artifacts --> completion
    end
    dev --> job

    classDef boundary fill:#eef,stroke:#557,stroke-width:1px;
    class cluster,ns boundary;
```

The same flow as ASCII (source also kept in
[`diagrams/kubernetes-architecture/`](diagrams/kubernetes-architecture/)):

```text
Developer ── kubectl apply ─▶ Kubernetes cluster
                                    │
                             Namespace: mlops
                                    │
                              Job: mlops-pipeline
                                    │
                                   Pod
                                    │
                              ML container (dvc repro)
                                    │
              preprocess ─▶ split ─▶ train ─▶ evaluate
                                                  │
                                     artifacts / metrics
                                                  │
                                          Job completion (exit 0)
```

The Kubernetes objects and their responsibilities:

| Object | Kind | Responsibility | This PR |
|---|---|---|---|
| `mlops` | `Namespace` | Environment / RBAC / quota / Pod Security boundary | ✅ |
| `mlops-pipeline` | `Job` (`batch/v1`) | Run the pipeline once to completion, with bounded retries | ✅ (runnable, PR 2) |
| `mlops-pipeline-config` | `ConfigMap` | Non-secret runtime config (`LOG_LEVEL`, `MLFLOW_TRACKING_URI`) | ✅ PR 3 |
| `mlops-pipeline-secret` | `Secret` | MLflow/DagsHub credentials — template only, real Secret created out-of-band | ✅ PR 3 (template) |
| `mlops-pipeline` | `ServiceAccount` | Least-privilege identity, **no API access** (`automountServiceAccountToken: false`) | ✅ PR 3 |

Note the absence of a `Service`, `Ingress`, or `Deployment`. That absence is a
design decision, not an omission: a batch Job exposes nothing and serves nothing,
so those primitives would be architecture inflation. See
[ADR-009](decisions/ADR-009-kubernetes-workload-model.md).

---

## 3. Why a `Job` (not a `Deployment`)

Summarized here; the full decision, alternatives, and lifecycle analysis are in
[ADR-009](decisions/ADR-009-kubernetes-workload-model.md).

| | Batch `Job` (chosen) | `Deployment` (rejected) |
|---|---|---|
| Intended workload | Run to completion, then stop | Keep a process alive indefinitely |
| Success = | Pod exits `0` → Job `Complete` | Process never exits |
| Pipeline exit `0` | Correct terminal state | Read as a crash → **restart loop** |
| Retry model | `backoffLimit` (bounded, then fail) | `restartPolicy: Always` (unbounded) |
| Health probes | Not applicable (no endpoint) | Expected (liveness/readiness) |
| Matches `dvc repro`? | ✅ finite computation | ❌ forces a fake long-running shape |

A `Deployment` would treat the pipeline's natural, successful exit as a failure
and restart it forever. The `Job` primitive is the one Kubernetes provides for
exactly this finite shape.

---

## 4. Lifecycle

A batch Job has a completion-oriented lifecycle rather than a
"stay-healthy-forever" one:

```text
Pending ─▶ Running ─▶ (pipeline executes) ─▶ Succeeded  ─▶ Job: Complete
                                          └▶ Failed ─▶ retry (≤ backoffLimit) ─▶ Job: Failed
```

- **restartPolicy: `Never`** — a failed attempt is not restarted in place; the
  Job controller schedules a fresh pod for each retry, so every attempt is a
  clean run of the deterministic pipeline.
- **backoffLimit: `2`** — the pipeline is deterministic, so a genuine failure (bad
  input, config error) fails identically on every attempt; the two retries exist
  only to absorb a *transient* MLflow/DagsHub blip. The controller back-offs
  exponentially between attempts (~10 s, 20 s, …), so a fast-failing pod cannot
  hot-loop.
- **activeDeadlineSeconds: `1800`** — a wall-clock ceiling for the whole Job (all
  retries). This is a completion-semantics stall-guard, **not** a performance SLO:
  compute finishes in well under a minute locally (`train`, the heaviest stage,
  measured at ~2.5 s under a 1-CPU limit), so a Job still running after 30 minutes
  is stuck (e.g. a stalled MLflow/DagsHub call) and should fail with
  `DeadlineExceeded` rather than hold a pod indefinitely. Deliberately generous;
  `ttlSecondsAfterFinished` (finished-Job cleanup) remains a future item.
- **Resource requests/limits (PR 5)** — `requests: {cpu: 250m, memory: 256Mi}`,
  `limits: {cpu: "1", memory: 512Mi}`, giving **Burstable** QoS. Chosen from
  measured usage of the real image, not guessed; the CPU limit doubles as the
  memory-safety control because `GridSearchCV(n_jobs=-1)` sizes joblib's worker
  fan-out from the cgroup CPU quota. Full method and numbers in
  [ADR-011](decisions/ADR-011-kubernetes-resource-lifecycle.md) and
  [§7](#7-local-validation-vs-production-deferred).
- **No liveness/readiness probes** — there is no live endpoint or long-running
  process to probe; a finite Job's health is *terminal* (exit `0` = success), which
  the Job controller reads directly from the container's exit status. A liveness
  probe would need an HTTP endpoint the app should not expose, or would fire during
  normal quiet compute and kill a healthy run. This mirrors the container image,
  which ships **no `HEALTHCHECK`** by the same reasoning (see the `Dockerfile` and
  [ADR-005](decisions/ADR-005-containerization-strategy.md)); the build-time import
  smoke test plays the environment-validity role a probe otherwise would. Decision
  recorded in [ADR-011](decisions/ADR-011-kubernetes-resource-lifecycle.md).

---

## 5. Configuration Boundary (implemented in PR 3)

Configuration is layered so that the image stays immutable and environment-
specific values live in the cluster, consistent with the twelve-factor approach
already used for the container ([ADR-005](decisions/ADR-005-containerization-strategy.md)).
Each variable below is one the code actually reads
(`src/pipeline_io.py::require_env`, `src/logging_config.py`, the MLflow client) —
classification is by sensitivity, not convenience:

| Layer | Holds | Kubernetes carrier | Committed? |
|---|---|---|---|
| Image | Application code + dependencies | Container image | Code only |
| Non-secret config | `LOG_LEVEL`, `MLFLOW_TRACKING_URI` (an endpoint, not a credential) | `ConfigMap` `mlops-pipeline-config` | Yes (no secrets) |
| Secrets | `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` | `Secret` `mlops-pipeline-secret` | **Template only, never values** |

Both are wired into the Job with `envFrom` — the ConfigMap unconditionally (it is
in the Kustomize base) and the Secret with `optional: true`, so `kubectl apply -k`
succeeds before the operator creates the real Secret out-of-band from a git-ignored
`.env`. The classifying rule: `MLFLOW_TRACKING_URI` is a public endpoint (the same
host already committed as the DVC S3 remote in `.dvc/config`) and carries no
authority, so it is config; the username/token authenticate to it, so they are
secret. `LOG_LEVEL` is the pipeline's own knob.

Contract, held: **no real credentials are ever committed.** The repo ships only
[`k8s/base/secret.example.yaml`](../k8s/base/secret.example.yaml) with placeholder
values, and it is excluded from `base/kustomization.yaml` so no render or apply can
emit it. A *green* run still needs the remaining "make it runnable" work (an SCM in
the image + a mounted dataset); credentials are now injectable.

---

## 6. Identity & Security Boundary

**Identity (implemented in PR 3).** The workload runs under a dedicated
`ServiceAccount` (`mlops-pipeline`) rather than the namespace `default`, giving it
a named identity to scope any future policy to. Because the pipeline never calls
the Kubernetes API — it runs `dvc repro` and reaches only MLflow/DagsHub over
HTTPS — the account sets **`automountServiceAccountToken: false`** (on both the
ServiceAccount and the Job's pod template), so no API token is projected into the
pod. Verified on a live cluster: the applied pod had an empty `spec.volumes` and
empty container `volumeMounts` (no `kube-api-access-*` token volume, no
`/var/run/secrets/kubernetes.io/serviceaccount` mount). For the same reason **no
`Role`/`RoleBinding`** is defined — the workload needs no permissions, and granting
unused ones would violate least privilege.

**Security context (implemented in PR 4; [ADR-010](decisions/ADR-010-kubernetes-security-hardening.md)).**
The container image is already built to satisfy a restricted posture: a dedicated
non-root UID/GID (`10001`), no baked-in secrets or data, and a minimal surface
([ADR-005](decisions/ADR-005-containerization-strategy.md) §9). PR 4 *enforces*
that at the platform layer with a `securityContext` split across the two levels
Kubernetes defines:

- **Pod level** — `runAsNonRoot: true`, explicit `runAsUser: 10001` /
  `runAsGroup: 10001`, and `seccompProfile.type: RuntimeDefault`. The explicit
  numeric uid is **required**, not cosmetic: the image's `USER` is the *name*
  `appuser`, which the kubelet cannot verify as non-root, so `runAsNonRoot` alone
  would reject the pod with `CreateContainerConfigError`.
- **Container level** — `allowPrivilegeEscalation: false` (verified:
  `NoNewPrivs: 1`), `capabilities.drop: [ALL]` (the pipeline needs no Linux
  capabilities), and an explicit `readOnlyRootFilesystem: false`.

**Why read-only root filesystem is deferred (not skipped).** `dvc repro` mutates
DVC state **in-tree at the `/app` repo root** — under a read-only root FS its
first action fails with `[Errno 30] Read-only file system: '/app/.dvc/tmp'`, and
it further writes `/app/.dvc/cache`, rewrites `/app/dvc.lock`, and needs a writable
`/app/.git` for the SCM. Those paths sit at the repo root alongside the read-only
baked-in code and `.dvc/config`, so they cannot be carved out with `emptyDir`
without shadowing image files or making the code tree writable. Enabling it now
would make the container fail *earlier* than the pre-existing SCM blocker — i.e.
weaken a working workload to pass a checkbox. It is deferred to the same work that
makes the pipeline green in-cluster (relocating DVC cache/tmp/lock + SCM onto
declared writable volumes). Full rationale and evidence:
[ADR-010](decisions/ADR-010-kubernetes-security-hardening.md).

The hardening is **behaviour-neutral**: with it applied the Job reaches the *same*
pre-existing SCM blocker it hits without it (proven on the live cluster, below), so
no new failure mode was introduced. Restricted Pod Security Standard *compliance*
is **not** claimed — the fields are present and cluster-admitted, but no Pod
Security admission label or policy engine has validated the profile, and read-only
root is not met.

---

## 7. Local Validation vs Production-Deferred

A reviewer should be able to tell exactly what is proven today.

**Validated now (no cluster required) — actually performed in PR 2:**

- The manifests parse as YAML (round-tripped through PyYAML).
- `kustomize build k8s/base` and `kustomize build k8s/overlays/local` render
  successfully; the rendered container image is `ml-pipeline:local`, the command
  is `["dvc","repro"]`, and the Job carries `backoffLimit: 2`,
  `activeDeadlineSeconds: 1800`, `restartPolicy: Never`, and `namespace: mlops`.
  The rendered container also carries the PR 5 `resources` block —
  `requests: {cpu: 250m, memory: 256Mi}`, `limits: {cpu: "1", memory: 512Mi}` —
  and **no** health probes.
- Config/identity wiring is asserted (PR 3): the rendered Job carries
  `serviceAccountName: mlops-pipeline`, `automountServiceAccountToken: false`, and
  an `envFrom` pulling the `mlops-pipeline-config` ConfigMap plus the optional
  `mlops-pipeline-secret`; the `ConfigMap` holds exactly `LOG_LEVEL` and
  `MLFLOW_TRACKING_URI` (no credential keys), and the **Secret template is not
  emitted** by the build.
- Security context is asserted (PR 4): 21 field/scope checks pass — the pod
  carries `runAsNonRoot`, `runAsUser`/`runAsGroup: 10001`, and
  `seccompProfile: RuntimeDefault`; the container carries
  `allowPrivilegeEscalation: false`, `capabilities.drop: [ALL]`, and an explicit
  `readOnlyRootFilesystem: false`; container-only fields are not at the pod level
  (and vice-versa); and no privilege/escape footguns (`privileged`, `hostNetwork`,
  `hostPID`, `hostIPC`) are present.
- Resource & lifecycle are asserted (PR 5): the rendered container carries
  `resources.requests {cpu: 250m, memory: 256Mi}` and `resources.limits
  {cpu: "1", memory: 512Mi}`, and no `livenessProbe`/`readinessProbe`/
  `startupProbe` is present.
- **All of the above is now enforced by CI (PR 6):** the `k8s-validate` job renders
  base + overlay with `kustomize`, validates every object against the pinned
  upstream Kubernetes **schema** with `kubeconform -strict`, and runs the committed
  `k8s/validate.py` (the productionized successor to the earlier per-PR scratch
  assertions) which re-checks all the security/required-field/secret-hygiene items
  above with a PASS/FAIL line each. Verified by negative test: a flipped
  `allowPrivilegeEscalation`/`runAsNonRoot` fails `k8s/validate.py`, and a string
  `activeDeadlineSeconds` fails `kubeconform`. This validation is **static** — it
  does not deploy or run the workload
  ([ADR-012](decisions/ADR-012-kubernetes-manifest-validation.md)).
- The workload model, boundaries, and trade-offs are documented and recorded in
  an ADR.

**Performed — an executed local cluster run (2026-08-12, Docker Desktop Kubernetes
v1.34.3):** the image was built and `kubectl apply -k k8s/overlays/local` was run
against a live cluster. The **Job lifecycle was verified end to end**:

- The `mlops` namespace and `mlops-pipeline` Job were created; the local image was
  resolved (no registry pull) and the container started (`Pending →
  ContainerCreating → Running`).
- The Job ran its designed retry lifecycle: **3 attempts total** — the initial pod
  plus `backoffLimit: 2` — each a *fresh* pod with `RESTARTS: 0` (confirming
  `restartPolicy: Never`), followed by a `BackoffLimitExceeded` event and a
  terminal **Job `Failed`** state.
- The pipeline itself did **not** complete: every attempt failed immediately with
  `ERROR: /app is not a git repository`. `dvc repro` requires an SCM, and the
  runtime image neither runs `git init` nor sets `core.no_scm`, so DVC aborts
  before evaluating any stage. This is an honest finding, not a success: the Job
  *mechanism* is proven on a real cluster, but a **green** end-to-end run needs the
  PR 3 "make it runnable" work — an SCM in the image (`git init` or
  `core.no_scm=true`), a mounted dataset (the image ships none by design), and
  MLflow/DagsHub credentials. Full OpenAPI schema validation via `kubectl apply`
  succeeds against this live cluster; since PR 6, offline schema validation also
  runs in CI via `kubeconform -strict`, and a server-side `kubectl apply -k
  --dry-run=server` admission check is available as an opt-in ephemeral-kind job
  ([ADR-012](decisions/ADR-012-kubernetes-manifest-validation.md)).

**PR 3 config/identity — also verified on the live cluster (Docker Desktop
Kubernetes v1.34.3).** `kubectl apply -k k8s/overlays/local` created the
`ServiceAccount`, `ConfigMap`, and Job; the applied pod carried
`serviceAccountName: mlops-pipeline` with an **empty `spec.volumes`** and empty
container `volumeMounts` — proving `automountServiceAccountToken: false` (no
`kube-api-access-*` token volume mounted). The container **started** with the
optional Secret absent (confirming `secretRef.optional: true`), then failed at the
same known SCM blocker as PR 2 — i.e. this PR altered configuration and identity
only, not pipeline behavior. The existing test suite is likewise unaffected
(100 passed, 1 pre-existing skip). Resources were deleted afterward.

**PR 4 security context — verified against the real image and the live cluster.**
The controls were validated empirically, not asserted:

- **`docker run` probes** established the read-only-root incompatibility and the
  behaviour-neutrality directly. Under the full hardened runtime *minus* read-only
  root (`--user 10001:10001 --cap-drop ALL --security-opt no-new-privileges`), the
  core stack imports cleanly and `dvc repro` reaches the *same* pre-existing
  `/app is not a git repository` blocker; `NoNewPrivs: 1` is confirmed. Adding
  `--read-only --tmpfs /tmp` makes DVC fail *earlier* with
  `[Errno 30] Read-only file system: '/app/.dvc/tmp'` — the evidence behind
  deferring `readOnlyRootFilesystem` ([ADR-010](decisions/ADR-010-kubernetes-security-hardening.md)).
- **Live cluster (Docker Desktop v1.34.3).** `kubectl apply -k k8s/overlays/local`
  was **admitted** (proving the explicit numeric `runAsUser` satisfies
  `runAsNonRoot`; a name-only USER would have been rejected). The applied pod's
  enforced `spec.securityContext` reported
  `{runAsNonRoot:true, runAsUser:10001, runAsGroup:10001, seccompProfile:RuntimeDefault}`
  and the container's `{allowPrivilegeEscalation:false, capabilities:{drop:[ALL]},
  readOnlyRootFilesystem:false}`. `spec.volumes` was still empty (PR 3's token
  automount-off intact). The container **ran** and terminated at the same
  `/app is not a git repository` blocker (exit 255) — no security-induced
  regression. Resources were deleted afterward.
- No local static manifest scanner (`kubesec`, `kube-score`, `kube-linter`,
  `checkov`, `trivy`) is installed; the live-cluster admission + kubelet
  enforcement above served as the authoritative validation, and the manifest was
  **not** sent to any external scanning service.

**PR 5 resource & lifecycle — measured on the real image and verified on the live
cluster (Docker Desktop v1.34.3).** The resource values were derived from
`docker run` probes of the real `ml-pipeline:local` image running the actual
`run_training` computation, not guessed: the import floor is ~132 MiB and peak
memory scales with granted CPU (1 CPU → ~133 MiB/~2.5 s; 2 → ~419 MiB; unlimited →
~1785 MiB/~20 s) because `GridSearchCV(n_jobs=-1)` sizes joblib's worker pool from
the cgroup CPU quota (confirmed: `joblib.cpu_count()` returns `2` under `--cpus=2`
while `os.cpu_count()` returns `20`). The chosen limits were validated directly —
`--cpus=1 --memory=512m` completes (exit 0, ~133 MiB peak) and `--memory=64m` is
`OOMKilled` (exit 137), proving the memory limit is kernel-enforced. On the live
cluster the applied pod reported the enforced `resources` exactly, **QoS
`Burstable`**, **no** probes, and `restartPolicy: Never`; the Job ran its 3-attempt
back-off lifecycle and every attempt terminated at the *same* pre-existing SCM
blocker (exit 255) — **none `OOMKilled`**, i.e. the resource constraints added no
new failure mode. Full method and the failure-mode table:
[ADR-011](decisions/ADR-011-kubernetes-resource-lifecycle.md).

**Production-deferred (explicitly out of scope for Sprint 5):**

- Any managed cluster (EKS/AKS/GKE), high availability, autoscaling, GitOps,
  service mesh, production observability, or model serving. These are future
  roadmap milestones ([roadmap.md](roadmap.md) v5–v6), not claims this sprint
  makes.

```text
Local render (kustomize)  ─▶  Local cluster run (Docker Desktop)  ─▶  Production deployment
   ✅ PR 1–6 (+CI static)  ✅ executed — lifecycle + config/identity     ⬜ future roadmap
                              + hardened securityContext + measured
                              resources verified;
                              pipeline not green yet (SCM + data)
```

> **Update — Sprint 6 (Cloud Platform Foundation).** The "managed cluster
> (EKS/…)" deferral above is being closed by Sprint 6. Terraform now provisions a
> real **Amazon EKS** platform ([ADR-017](decisions/ADR-017-eks-platform.md)), and
> a new **`k8s/overlays/aws`** overlay integrates *this same base workload* with it
> by reusing the base unchanged and layering only cloud-specific config — ECR image
> source, `imagePullPolicy`, and the dataset mount — while inheriting every Sprint 5
> security control verbatim ([ADR-018](decisions/ADR-018-aws-eks-deployment-overlay.md)).
> The AWS overlay is **statically validated and CI-gated** alongside the local
> overlay (render + `kubeconform` + `k8s/validate.py`, 45/45); the **real green run
> on EKS** is the Sprint 6 PR 7 integration test, run on the operator's own AWS
> account. Autoscaling, GitOps, mesh, and production observability remain out of
> scope.

---

## 8. What This Does *Not* Claim

- A **green** in-cluster pipeline run is now achieved (Sprint 5 runtime PR,
  [ADR-013](decisions/ADR-013-kubernetes-runtime-execution.md)): `dvc repro` runs the
  full pipeline to **exit 0** as a secured Job, via a minimal runtime contract (DVC
  no-SCM, a runtime-retrieved dataset, and — since Sprint 7 — the in-cluster MLflow
  platform). The dataset now comes from a private S3 bucket, retrieved at runtime by
  the `fetch-dataset` init container via EKS Pod Identity and checksum-verified
  (Sprint 7 PR 8, closes M-04 — [ADR-027](decisions/ADR-027-s3-dataset-runtime-retrieval.md)),
  and tracking runs on the in-cluster MLflow platform
  ([ADR-026](decisions/ADR-026-in-cluster-mlflow-platform.md)); locally these use MinIO
  and the same platform. A live **EKS** run remains operator-gated; read-only-root
  remains future work.
- It does not claim **restricted Pod Security Standard compliance**. PR 4 applies
  the individual `securityContext` controls that are compatible (non-root, no
  privilege escalation, all capabilities dropped, seccomp `RuntimeDefault`) and
  names the one that is not (read-only root filesystem, deferred with evidence in
  [ADR-010](decisions/ADR-010-kubernetes-security-hardening.md)); no admission
  controller has validated the profile.
- It does not claim **production-certified capacity**. PR 5 sets resource
  requests/limits from *measured* local usage on the small bundled dataset
  ([ADR-011](decisions/ADR-011-kubernetes-resource-lifecycle.md)); a larger
  dataset, wider grid, or real cluster would require re-measuring.
- It does not introduce an HTTP API, `Service`, or `Ingress`; the workload is
  batch and is modelled as such.

---

## Related Documentation

- [ADR-009 — Kubernetes Workload Model](decisions/ADR-009-kubernetes-workload-model.md)
- [ADR-010 — Security Hardening](decisions/ADR-010-kubernetes-security-hardening.md) ·
  [ADR-011 — Resource & Lifecycle](decisions/ADR-011-kubernetes-resource-lifecycle.md) ·
  [ADR-012 — Manifest Validation](decisions/ADR-012-kubernetes-manifest-validation.md)
- [Kubernetes Operations](kubernetes-operations.md) ·
  [Kubernetes Security](kubernetes-security.md) ·
  [Sprint 5 Proof-Impact](proof/sprint-05-proof-impact.md)
- [Architecture](architecture.md)
- [Containerization Strategy](containerization.md)
- [ADR-005 — Containerization Strategy](decisions/ADR-005-containerization-strategy.md)
- [Roadmap](roadmap.md)
- [`k8s/README.md`](../k8s/README.md)
