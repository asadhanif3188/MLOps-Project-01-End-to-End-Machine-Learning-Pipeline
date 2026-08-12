# ADR-011: Kubernetes Resource & Lifecycle Management (Requests/Limits, Backoff, No Probes)

- **Status:** Accepted
- **Date:** 2026-08-12
- **Deciders:** Asad Hanif
- **Related:** [Kubernetes Architecture](../kubernetes-architecture.md) §"Operational lifecycle",
  [`k8s/base/job.yaml`](../../k8s/base/job.yaml),
  [ADR-009 (Kubernetes Workload Model)](ADR-009-kubernetes-workload-model.md),
  [ADR-010 (Security Hardening)](ADR-010-kubernetes-security-hardening.md),
  [Roadmap](../roadmap.md) (v4 Kubernetes)

> **Scope note.** [ADR-009](ADR-009-kubernetes-workload-model.md) ratified the
> *workload model* (a run-to-completion `Job`) and [ADR-010](ADR-010-kubernetes-security-hardening.md)
> the *security posture*. This record covers Sprint 5 PR 5: **resource requests
> and limits chosen from measured usage**, the **finite-run lifecycle** already
> present on the Job (`restartPolicy`, `backoffLimit`, `activeDeadlineSeconds`)
> now reviewed and justified, the **deliberate absence of health probes**, and the
> **documented failure modes**. Values here are tuned for a *local* batch run and
> are **not presented as production-certified capacity figures**.

## Context

The pipeline runs as a `batch/v1` `Job` from the `ml-pipeline:local` image,
executing `dvc repro` (preprocess → split → train → evaluate) on the small Pima
Indians Diabetes dataset. Before PR 5 the Job had a correct *lifecycle*
(`restartPolicy: Never`, `backoffLimit: 2`, `activeDeadlineSeconds: 1800`) but
**no resource requests or limits** — meaning the kube-scheduler had no signal for
bin-packing it and the kubelet had no ceiling to protect the node from a runaway.

The repo's design principle (see ADR-010) is to derive each control from the
**real image and measured behaviour**, not a copy-pasted block. For resources
that is essential, because a specific application fact dominates the numbers.

### The load-bearing fact: `GridSearchCV(n_jobs=-1)` + joblib reads the cgroup CPU quota

The heaviest stage, `train` ([`src/train.py`](../../src/train.py)), tunes a
`RandomForestClassifier(n_estimators=100, max_depth=5)` with
`GridSearchCV(cv=3, n_jobs=-1)`. `n_jobs=-1` tells joblib/loky to parallelise
across "all CPUs". Critically, **loky sizes its worker pool from the cgroup CPU
quota (the Kubernetes CPU _limit_), not the node's physical core count** —
verified in-container: under `--cpus=2`, `os.cpu_count()` returns `20` but
`joblib.cpu_count()` returns `2`.

Each forked worker copies the ~130 MiB interpreter, so **memory scales with the
number of workers, which the CPU limit controls.** Measured peak of the whole
cgroup (parent + joblib workers), running the real `run_training` on a
schema-faithful 768×9 synthetic dataset in the real image under the hardened
securityContext (`--user 10001:10001 --cap-drop ALL --security-opt no-new-privileges`):

| Granted CPU | Peak memory | `train` wall time |
|---|---|---|
| 1 CPU | **~133 MiB** | **~2.5 s** |
| 2 CPU | ~419 MiB | ~5.4 s |
| unlimited (20 host cores) | ~1785 MiB | ~20 s |

At this data scale more cores **hurt**: the per-fit work is sub-second, so
fork/serialize/dispatch overhead dominates and both memory and wall-clock balloon.
The import floor alone (interpreter + numpy + pandas + sklearn) is ~132 MiB; the
tiny dataset and shallow forest add almost nothing on a single worker.

## Decision

### 1. Resource requests and limits (container-scoped)

```yaml
resources:
  requests:
    cpu: 250m
    memory: 256Mi
  limits:
    cpu: "1"
    memory: 512Mi
```

| Field | Value | Rationale |
|---|---|---|
| `requests.cpu` | `250m` | A **scheduling reservation**, not a hard need. A short, deterministic batch makes steady progress on a quarter-core (each grid fit ≈ 1 s on one core); small enough to schedule on a busy node. |
| `requests.memory` | `256Mi` | Above the measured ~133 MiB import+train floor, so the pod is *guaranteed* enough memory (the kernel cannot reclaim an app's live pages, so memory requests are a real reservation). |
| `limits.cpu` | `"1"` | Caps joblib's worker fan-out to a single worker (loky reads this quota) — **this is the memory-safety control**, keeping peak ~133 MiB and predictable. Also the fastest operating point at this data size. |
| `limits.memory` | `512Mi` | ~3.9× the measured 1-CPU peak. Headroom for log buffers and minor future grid growth, but tight enough that a genuine runaway is OOMKilled rather than silently consuming a node. |

Requests ≠ limits by design → **Burstable** QoS class (confirmed on a live
cluster): reserve a modest floor, cap the ceiling.

**Why a CPU limit at all** (the common guidance is to avoid CPU limits because
CFS throttling can harm latency-sensitive services): this is a batch Job with no
latency SLO. Throttling only makes it marginally slower, and any slowdown is
bounded by `activeDeadlineSeconds`. Here the CPU limit earns its place as the
control that bounds the `n_jobs=-1` memory balloon — a workload-specific reason,
not cargo-cult.

### 2. Lifecycle (reviewed, retained)

- **`restartPolicy: Never`** — a Job must use `Never` or `OnFailure`; `Never` is
  chosen so each retry is a fresh, independently-inspectable pod rather than an
  in-place container restart that overwrites the prior attempt's logs/events.
- **`backoffLimit: 2`** — the pipeline is deterministic, so a real code/data/config
  error fails identically every attempt; the two retries exist only to absorb a
  *transient* fault (a blip reaching MLflow/DagsHub). The Job controller applies
  exponential back-off (~10 s, 20 s, …) between attempts, so a fast-failing pod
  cannot hot-loop.
- **`activeDeadlineSeconds: 1800`** — an outer stall-guard, **not** a performance
  SLO. Local compute finishes in well under a minute; a Job still alive after
  30 min is stuck (e.g. a hung network call) and should fail with
  `DeadlineExceeded`. Kept deliberately generous — tightening it toward the real
  runtime would risk killing a legitimately slow-but-progressing run on a
  constrained node.

### 3. No health probes (liveness/readiness/startup) — by design

Probes model a **long-running server**: readiness gates Service traffic, liveness
restarts a wedged daemon, startup delays the other two. This workload has **no
listening socket, no Service, and no request traffic**. Its notion of "healthy"
is not a steady state to poll but a **terminal** one — exit `0` = success,
non-zero = failure — which the Job controller already observes directly from the
container's exit status. A liveness probe would either require an HTTP endpoint
the app does not (and should not) expose, or would fire during the pipeline's
normal quiet compute and **kill a healthy run**. The real health signal is the
exit code plus the structured logs; probes would add machinery without a health
question to answer.

### 4. Documented failure modes

| Failure | Surfaces as | Operator signal |
|---|---|---|
| **Image pull failure** | Pod `Waiting` → `ErrImagePull`/`ImagePullBackOff` | `kubectl -n mlops describe pod` events; local image not side-loaded into the cluster. |
| **Configuration failure** | Container exits non-zero early (e.g. missing `MLFLOW_TRACKING_URI` → `ConfigError`) | Pod `Error`; logs show the config error; ConfigMap absent/misnamed. |
| **Secret failure** | Pipeline starts (Secret is `optional: true`) but MLflow calls fail with an auth error | Non-zero exit at the tracking boundary; logs show the auth failure; Secret not created out-of-band. |
| **Application failure** | Container exits non-zero (today: `dvc repro` → `/app is not a git repository`) | `status.failed` increments per attempt; identical across all 3 attempts ⇒ deterministic, not transient. |
| **Resource exhaustion** | Container `OOMKilled` (exit 137) if it exceeds `limits.memory`; CPU throttling if it exceeds `limits.cpu` (slower, not killed) | Pod `terminated.reason: OOMKilled`. Validated at 64Mi (see below). |

## Validation

All measurements from the **real** `ml-pipeline:local` image under the hardened
securityContext; the k8s enforcement checked on a live Docker Desktop cluster
(v1.34.3):

- **Resource probe** — `run_training` on synthetic Pima-shaped data: import floor
  ~132 MiB; peak vs granted CPU as tabled above (1→133 MiB/2.5 s, 2→419 MiB,
  unlimited→1785 MiB/20 s); `joblib.cpu_count()` confirmed to track the cgroup
  quota (2 under `--cpus=2`), not `os.cpu_count()` (20).
- **Success at the chosen limits** — `docker run --cpus=1 --memory=512m
  --memory-swap=512m …` (mirroring the k8s limits with swap off) completes with a
  ~133 MiB peak, exit 0.
- **Resource-exhaustion failure mode** — the same run under `--memory=64m` is
  **OOMKilled (exit 137)**, confirming the memory limit is kernel-enforced.
- **Live cluster** — `kubectl apply -k k8s/overlays/local`; the pod's enforced
  `resources` reported exactly `{requests: cpu 250m/mem 256Mi, limits: cpu 1/mem
  512Mi}`, **QoS class `Burstable`**, **no** liveness/readiness/startup probes,
  `restartPolicy: Never`. The Job ran its designed lifecycle — 3 attempts (initial
  + `backoffLimit: 2`), exponential back-off, then `BackoffLimitExceeded` →
  `Failed`. Every attempt terminated with exit 255 (`Error`) at the **same**
  pre-existing SCM blocker (`/app is not a git repository`); **none** were
  `OOMKilled` — i.e. the resource constraints introduced **no new failure mode**.
  Resources were deleted afterward.

## Alternatives Considered

1. **No resource governance (status quo).** Rejected: leaves the scheduler blind
   and the node unprotected; the measured 1785 MiB unbounded peak shows the
   `n_jobs=-1` balloon is real, not hypothetical.
2. **Requests only, no limits.** Reasonable for many services, but here the CPU
   limit is precisely what bounds the joblib memory fan-out, and a memory limit is
   what turns a runaway into a contained `OOMKilled` instead of node pressure.
   Rejected for this workload.
3. **Guaranteed QoS (requests == limits).** Would pin a full core and 512Mi as a
   reservation for a job that idles most of a second between sub-second fits —
   wasteful on a shared node for no reliability gain at this scale. Burstable fits
   a bursty short batch better.
4. **Fix `n_jobs=-1` in the application instead.** The app-level over-subscription
   is real (more cores hurt here), but changing training code is out of scope for a
   Kubernetes platform PR and would couple the two concerns. The CPU limit bounds
   the blast radius today; an app-level `n_jobs` change is noted as a future
   refinement, not a prerequisite.
5. **Add a liveness/readiness probe "for completeness".** Rejected: see Decision §3
   — there is no health endpoint and no traffic to gate; a probe would be
   theatre at best and a healthy-run killer at worst.
6. **Tighten `activeDeadlineSeconds` to near the real runtime.** Rejected: it is a
   stall-guard, not an SLO; a tight value risks killing a legitimately slow run on
   a constrained node for no safety benefit.

## Consequences

- The scheduler can bin-pack the Job (it declares what it needs), and the kubelet
  caps it (`OOMKilled`/throttle) if it misbehaves — both from measured, not
  guessed, numbers.
- The CPU limit doubles as memory safety: joblib cannot fork 20 workers, so peak
  memory stays ~133 MiB and predictable.
- QoS is Burstable; under node memory pressure this Job is evicted before
  Guaranteed pods — acceptable for a re-runnable batch job.
- Failure modes are enumerated for operators, and the "why no probes" question is
  answered once, in writing.

## What This Decision Does **Not** Imply

- **Not production-certified capacity.** These values are tuned for a *local*
  single-node run on the small bundled dataset. A larger dataset, a wider grid, or
  a real multi-node cluster would require re-measuring; the numbers here are a
  measured starting point, not a guarantee.
- **Not a claim that the pipeline runs green in-cluster.** The pre-existing SCM
  blocker (`/app is not a git repository`) is unchanged; resource governance is
  orthogonal to making the run succeed (see ADR-010 and `k8s/README.md`).
- **Not a fix for the application's `n_jobs=-1` over-subscription** — it is
  contained at the platform layer, not removed.
