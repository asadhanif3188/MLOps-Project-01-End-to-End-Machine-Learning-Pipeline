# ADR-009: Kubernetes Workload Model — `Job`, not `Deployment`

- **Status:** Accepted (design)
- **Date:** 2026-08-12
- **Deciders:** Asad Hanif
- **Related:** [Kubernetes Architecture](../kubernetes-architecture.md),
  [`k8s/`](../../k8s/), [Architecture](../architecture.md),
  [Roadmap](../roadmap.md) (v4 Kubernetes),
  [ADR-005 (Containerization Strategy)](ADR-005-containerization-strategy.md)

> **Scope note.** This ADR ratifies the *workload model* and the directory
> foundation delivered in Sprint 5, PR 1: the `mlops` namespace and a `batch/v1`
> `Job`, structured with Kustomize. Configuration/secrets, security hardening,
> resource limits, and CI validation are **separate later PRs** and, where they
> constitute an architectural decision, their own ADRs — the Kubernetes security
> baseline is reserved for **ADR-010**. This record intentionally covers only the
> choice of workload primitive and its lifecycle contract.

## Context

The pipeline is a **finite, batch, file-based ML workflow** —
`preprocess → split → train → evaluate`, orchestrated by DVC and run as
`dvc repro` (see [architecture.md](../architecture.md) and
[ADR-005](ADR-005-containerization-strategy.md)). It starts, does work, writes
`models/model.pkl` and `metrics/metrics.json`, and **exits**. It has no
request/response surface, no port to listen on, and no long-running process to
keep alive.

The [roadmap](../roadmap.md) (v4) calls for running the pipeline as an
orchestrated Kubernetes workload. The first architectural decision is therefore:
**which Kubernetes workload primitive expresses this pipeline correctly?**

Kubernetes offers several: `Deployment`/`ReplicaSet` (long-running, always-on
replicas), `StatefulSet` (stateful, identity-stable replicas), `DaemonSet`
(one-per-node agents), `Job` (run-to-completion), and `CronJob` (scheduled
Jobs). Picking the wrong one is not cosmetic — the primitive dictates the
lifecycle, the success condition, the restart/retry semantics, and whether health
probes even make sense.

An anti-pattern we explicitly want to avoid: manufacturing an HTTP API or a fake
"online service" wrapper around a batch pipeline merely to justify a
`Deployment` + `Service` + `Ingress`. That would inflate the architecture and
misrepresent what the workload is.

## Decision

Deploy the pipeline as a Kubernetes **`Job` (`batch/v1`)** inside a dedicated
**`mlops` namespace**, structured with **Kustomize** (a `base/` plus an
`overlays/local/`).

**Batch vs service semantics.** A `Job` models a computation that runs to
completion; a `Deployment` models a service that must stay available. The
pipeline is the former. Its correct terminal state is "finished successfully and
stopped," which is precisely a Job's success condition (a pod exits `0` → the Job
is `Complete`).

**Lifecycle and completion.** The Job's lifecycle is
`Pending → Running → Succeeded/Failed → (Job) Complete/Failed`. Completion is
meaningful and observable: `kubectl get job` reports `Complete`, and the exit
code is the source of truth. There is nothing to "keep alive" after the metrics
are written.

**Retry / restart semantics.** The Job uses **`restartPolicy: Never`** with
**`backoffLimit: 2`**: a failed attempt is not restarted in place; the Job
controller schedules a *fresh* pod per retry (each a clean run of the
deterministic pipeline), and after the bounded retries a persistent failure
surfaces as `Job: Failed` rather than looping forever. For a deterministic
pipeline, unbounded retries would just repeat the same failure — bounded,
fail-fast retries are the correct contract. (`activeDeadlineSeconds` and
`ttlSecondsAfterFinished` are reliability *tuning*, deferred to PR 5.)

**No health probes.** Liveness/readiness probes assume a live endpoint or
long-running process to interrogate; a batch Job has neither. Adding probes would
be cargo-culting. This mirrors the container image, which ships **no
`HEALTHCHECK`** for the same reason (a healthcheck would flip to UNHEALTHY the
moment the pipeline finished); the image's build-time import smoke test validates
environment health instead ([ADR-005](ADR-005-containerization-strategy.md)).

**Namespace boundary.** The workload runs in its own `mlops` namespace so RBAC,
quotas, and Pod Security expectations are scoped to this project rather than
`default`.

## Alternatives Considered

1. **`Deployment` (+ `Service`/`Ingress`).**
   - *Decision:* rejected — a Deployment keeps a process running forever and
     treats exit `0` as a failure to be restarted, turning the pipeline's normal,
     successful completion into a crash-loop. `Service`/`Ingress` would require a
     server that does not exist. This is the architecture-inflation anti-pattern
     the design principle forbids.
2. **Wrap the pipeline in an HTTP API to justify a long-running service.**
   - *Decision:* rejected — manufacturing an online surface to fit a batch
     workload into a service abstraction misrepresents the workload. A serving
     component may be justified *later* as its own milestone (roadmap v6) if it
     creates real proof, and would then legitimately use a Deployment.
3. **`CronJob`.**
   - *Decision:* deferred — a `CronJob` is a *scheduled* `Job`. Scheduling is a
     separate concern layered on the same primitive; the initial decision is
     manual, on-demand execution (`kubectl apply`). A `CronJob` becomes a natural,
     non-breaking extension once periodic retraining is a goal.
4. **`StatefulSet` / `DaemonSet`.**
   - *Decision:* rejected — the pipeline has no stable network identity or
     per-node placement requirement. Neither primitive matches a single finite
     computation.
5. **Raw `kubectl apply -f` without Kustomize.**
   - *Decision:* rejected — the workload will be specialized per environment
     (image tag now; resources, config, and security later). Kustomize's
     base/overlay split delivers that without duplicated YAML, consistent with the
     "one source per concern" principle in
     [ADR-004](ADR-004-python-quality-toolchain.md) and
     [ADR-005](ADR-005-containerization-strategy.md). For a single small base this
     is a modest cost that pays off immediately in the next PRs.

## Consequences

**Positive**

- The Kubernetes model matches the real workload: completion is meaningful,
  success is `exit 0`, and failures fail fast with bounded retries.
- No fake service, `Service`, or `Ingress` is introduced; the architecture stays
  honest and minimal.
- The `mlops` namespace gives a clean boundary for the identity, config, and
  security controls added in later PRs.
- The Kustomize base/overlay structure is ready to absorb PR 3–5 changes
  (config/secrets, security context, resources) without restructuring.

**Trade-offs and follow-ups**

- **Foundation only.** The base `Job` establishes the *model*; it deliberately
  omits config/secret wiring, security context, resources, and CI validation.
  As written it renders and is schema-valid but is not yet a complete,
  cluster-runnable workload — a demonstrated local run is PR 2, and the
  data/credential wiring it needs is PR 3.
- The **Kubernetes security baseline** (non-root enforcement, dropped
  capabilities, seccomp, read-only filesystem, least-privilege ServiceAccount)
  is **not** covered here — it is reserved for **ADR-010** and PR 4.
- Retry/deadline values (`backoffLimit`, and future `activeDeadlineSeconds`) are
  initial workload-model choices, not tuned production values; the reliability PR
  (PR 5) revisits them with justification.

## What This Decision Does *Not* Imply

- It does **not** imply the pipeline can never be served online. A future
  inference/serving component would be a *separate* workload that legitimately
  uses a `Deployment` + `Service`; choosing a `Job` here says nothing against it.
- It does **not** imply a production deployment. The decision is validated locally
  (render now; kind/minikube run in PR 2); managed clusters, HA, autoscaling, and
  GitOps remain future roadmap items and are not claimed.
- It does **not** imply scheduled/periodic execution. That is a `CronJob`
  extension to be decided if and when periodic retraining is a goal.
- It does **not** imply any security posture on its own — the absence of a
  security context in the foundation manifest is intentional and addressed by
  ADR-010, not an oversight.
