# ADR-028: Observability Architecture — Prometheus + Grafana over a four-layer model

- **Status:** Accepted (design; no components deployed in this PR)
- **Date:** 2026-08-20
- **Deciders:** Asad Hanif
- **Related:** [`docs/observability.md`](../observability.md),
  [`docs/diagrams/observability-architecture/README.md`](../diagrams/observability-architecture/README.md),
  [`k8s/base/job.yaml`](../../k8s/base/job.yaml),
  [`k8s/base/mlflow/deployment.yaml`](../../k8s/base/mlflow/deployment.yaml),
  [`k8s/base/mlflow/postgres.yaml`](../../k8s/base/mlflow/postgres.yaml),
  [`terraform/eks.tf`](../../terraform/eks.tf),
  [ADR-009 (Workload model — a Job, not a Deployment)](ADR-009-kubernetes-workload-model.md),
  [ADR-011 (Resources/lifecycle — measured limits, no probes)](ADR-011-kubernetes-resource-lifecycle.md),
  [ADR-020 (Ephemeral cloud lifecycle & cost control)](ADR-020-cloud-lifecycle-cost-control.md),
  [ADR-026 (In-cluster MLflow platform)](ADR-026-in-cluster-mlflow-platform.md),
  [Logging Strategy](../logging.md),
  [Roadmap v5 → v6](../roadmap.md)

> **Scope.** This ADR ratifies the *observability architecture* for the platform —
> the monitoring stack, the four workload layers it must cover, the operational
> questions it must answer, and, critically, **how a short-lived batch `Job`'s
> metrics stay queryable after its pod has exited**. It is **architecture and
> documentation only**: it installs **nothing**. No Prometheus, no Grafana, no
> exporters, and no alerts are deployed here; no application code and no existing
> manifest is changed. Those land in the later Sprint 8 PRs whose acceptance
> criteria this ADR (with [`docs/observability.md`](../observability.md)) defines.
> This is the design of record for *why* Prometheus/Grafana, *what* is measured
> (and deliberately not), *where* each signal comes from, and *why* full tracing
> and log aggregation are deferred.

## Context

Through Sprint 7 the platform became genuinely cloud-native — a hardened EKS
cluster, workload identity, an in-cluster MLflow tracking server, PostgreSQL
metadata, and an S3 dataset path, all proven on real EKS
([Sprint 7 runtime evidence](../proof/sprint-07-runtime-evidence.md)). But its
**operational visibility is exactly what it was in Sprint 5**: `kubectl get`,
`kubectl describe`, `kubectl logs`, and the pipeline's structured logs. The
roadmap has always named this gap honestly — monitoring, ⬜ through Sprint 7 and
🚧 as of this ADR, in
[v5](../roadmap.md#version-5--production-cloud-platform) ("diagnosis is `kubectl`
+ structured logs") and a headline objective of
[v6](../roadmap.md#version-6--enterprise-mlops). Answering questions like *"did
last night's pipeline run succeed?"*, *"is MLflow approaching its memory limit?"*,
or *"is the Postgres PVC filling up?"* today requires a human at a terminal at the
right moment. That is not operable.

The platform has four distinct things worth observing, each with a different
shape:

1. **The Kubernetes platform** — nodes, pods, scheduling, resource pressure.
2. **The MLOps pipeline** — a `batch/v1` **Job** that runs `dvc repro`
   (preprocess → split → train → evaluate) and **exits**, typically in well under
   a minute of compute (ADR-011). It has **no listening socket and no Service**
   (ADR-009), and by deliberate design **no health probes** (ADR-011).
3. **MLflow** — a long-running `Deployment` with a `/health` endpoint (already
   used by its probes) but **no native Prometheus `/metrics` endpoint**.
4. **PostgreSQL** — a single-writer `StatefulSet` on a fixed **1 Gi** PVC, whose
   durability is the whole point of the metadata store (ADR-026).

Two hard constraints shape the design, both inherited from earlier ADRs and not
up for renegotiation here:

- **The batch-workload constraint.** Prometheus is a **pull** system: it scrapes a
  live target on an interval. The pipeline pod is **gone** seconds after it
  finishes. A naïve `/metrics` endpoint on the Job's container is unscrapable —
  there is nothing to scrape between runs, and a run can complete *between two
  scrapes*. This is the central design problem, called out explicitly in the
  sprint brief, and it must be reasoned about, not assumed away.
- **The project's posture.** Self-hosted over SaaS in every runtime path
  (ADR-026); ephemeral, cost-controlled, single-operator validation environment,
  not production (ADR-020); every control derived from the **real** workload and
  **measured**, never cargo-culted (ADR-010/011); and rigorous honesty about what
  is validated versus designed.

## Decision

### 1. Stack — Prometheus for metrics, Grafana for dashboards

Adopt **Prometheus** (metrics collection + storage + alerting rules) and
**Grafana** (dashboards), the CNCF-standard, open-source Kubernetes observability
pair, deployed **in-cluster** and **internal-only** (ClusterIP; reached via
`kubectl port-forward`, never a public endpoint — the same posture as the MLflow
UI, ADR-026 § Security).

Why this stack:

- **It is the Kubernetes-native standard.** The ecosystem this platform needs
  already exists as first-class Prometheus citizens: **kube-state-metrics** (KSM)
  for API-object state, **node-exporter** for node health, the kubelet's built-in
  **cAdvisor** for per-container resource use, **blackbox-exporter** for endpoint
  probing, and **postgres-exporter** for database internals. We assemble
  well-understood parts rather than build a bespoke pipeline.
- **PromQL fits the questions.** The operational questions in
  [`docs/observability.md`](../observability.md) map directly onto PromQL over
  these exporters' series — no custom query layer.
- **It matches the project's self-hosted, portable posture.** Prometheus/Grafana
  run identically on local Docker Desktop and on EKS, so the *same* stack is
  exercised locally and in the cloud — exactly the local/cloud parity principle
  ADR-026 and ADR-027 established for MLflow and the dataset path. No cloud lock-in
  and no per-metric SaaS billing on a short-lived cluster.
- **Cost and footprint are controllable.** It can run with modest, measured
  requests/limits and short local retention, honouring ADR-020.

Alternatives are recorded in [§ Alternatives Considered](#alternatives-considered)
(CloudWatch Container Insights, Datadog/New Relic, a bare OpenTelemetry Collector).

### 2. The four-layer coverage model

Observability is defined **per layer**, and for each layer the design fixes the
*operational questions*, the *signals*, the *likely Prometheus source*, the
*dashboard use*, the *alerting use*, and the *limitations*. The full tables live
in [`docs/observability.md`](../observability.md); the shape is:

| Layer | Primary source(s) | The questions it answers |
|---|---|---|
| **1. Kubernetes platform** | node-exporter, kube-state-metrics, cAdvisor | Are nodes Ready? Any node under CPU/mem/disk pressure? Are pods CrashLooping? Is any workload approaching its memory limit? Are pods Pending / PVCs unbound? |
| **2. MLOps pipeline (Job)** | **kube-state-metrics** (primary); Pushgateway *(deferred)*; MLflow for ML metrics | Did the **last** Job succeed? When did it last run? How long did the run take? Did it retry / get OOMKilled / hit the deadline? |
| **3. MLflow** | blackbox-exporter (`/health`), kube-state-metrics, cAdvisor | Is MLflow **available**? Is it restarting? Is it approaching its measured ~1.7 GiB / 2 Gi memory ceiling? |
| **4. PostgreSQL** | postgres-exporter, kubelet volume stats, kube-state-metrics, cAdvisor | Is the DB up and accepting connections? Is the **1 Gi PVC filling up**? Connections near the limit? Is it restarting? |

The design **deliberately does not invent metrics because it can**. Each signal is
justified by an operational question an operator would actually ask about *this*
workload. Vanity series (e.g. per-request MLflow histograms nobody will act on at
this scale) are excluded, matching the ADR-011 discipline of measuring what the
workload needs rather than what is possible.

### 3. Batch-workload metric strategy — kube-state-metrics is the answer, not Pushgateway

> **Superseded in part by [ADR-030](ADR-030-pipeline-operational-metrics.md)
> (Sprint 8, PR 3).** KSM remains the **primary**, app-change-free source for
> run-level Job signals exactly as decided here. What changed: the Pushgateway,
> *deferred* below for per-stage granularity, was **adopted (scoped)** in PR 3 to add
> `mlops_pipeline_stage_*` (per-stage duration/success) — the one gap KSM cannot
> fill — with the sticky-metric and ownership trade-offs addressed there.

This is the load-bearing decision. The pipeline Job's pod exits; a pull-scraped
`/metrics` endpoint on it cannot work. The strategy:

**Primary — kube-state-metrics reflects the persistent `Job` (and its finished
`Pod`) object, not the ephemeral process.** KSM is a long-running Deployment that
watches the Kubernetes API and exposes the *state of API objects* as metrics. The
**`Job` object survives the pod's process** (until it is deleted or its
`ttlSecondsAfterFinished` elapses), so the Job's terminal state is a **stable,
scrapable series on an always-up target** — no live pod required. This makes the
ephemeral Job's *operational* signals queryable after the fact, entirely without
touching the application:

- `kube_job_status_succeeded` / `kube_job_status_failed` → **did the last run
  succeed?**
- `kube_job_status_start_time` and `kube_job_status_completion_time` → **when did
  it run, and how long did the whole run take** (completion − start)?
- `kube_job_status_failed` climbing across attempts → **did it burn `backoffLimit`
  retries** (the ADR-011 transient-fault path)?
- `kube_pod_container_status_last_terminated_reason{reason="OOMKilled"}` → **was it
  killed at its memory limit** (the exact ADR-011 failure mode)? *(This one is a
  **Pod**-object series; it stays scrapable because the Job's finished pod is
  retained by owner-reference for as long as the Job — the same TTL cascade removes
  both.)*
- `kube_job_status_active` / a `DeadlineExceeded` condition → **did it stall into
  `activeDeadlineSeconds`?**

This is chosen because it is **robust (API-server-backed), requires no application
change, and answers the operational questions directly**. Its one real limitation
is **granularity**: KSM sees the *whole* Job, so it gives run-level duration and
outcome but **not per-stage** timing ("which stage took the longest?").

**The queryability contract (must be honoured by the runtime PRs).** KSM exposes
the terminal state only while the `Job` object exists, and Prometheus must scrape
it at least once during that window. Two things make this reliable and both are
design requirements, not afterthoughts:

- Set a **generous `ttlSecondsAfterFinished`** on the Job (or do not auto-delete
  it) so the finished object — and thus its `succeeded/failed/completion_time`
  series — lingers for many scrape intervals. A tiny TTL would let history vanish
  before Prometheus sees it.
- Once Prometheus **has** scraped the terminal state, those samples live in its
  TSDB for the whole retention window even after the Job object is gone. So the
  requirement is narrow: *the finished Job (and its pod) must outlive one scrape*, which a
  sensible TTL guarantees.

**Deferred — Pushgateway, for per-stage granularity only, and only if justified.**
The Prometheus-sanctioned way to get metrics *out of* a batch job is the
**Pushgateway**: the job pushes metrics to a persistent gateway before exiting and
Prometheus scrapes the gateway. It is the natural home for the one genuinely
operational signal KSM cannot provide — **per-stage duration** ("which stage took
the longest?"). But it is **not adopted now**, deliberately, because:

- **It is a known anti-pattern when misused.** Pushed metrics are **sticky** —
  they persist in the gateway until explicitly overwritten or deleted, with no
  `up`/staleness semantics — so the gateway accumulates stale series and becomes a
  single point of failure and a stale-data source if not carefully lifecycle-managed.
- **It requires an application change** (a push step in the pipeline), which this
  architecture-only PR forbids and which should be justified on its own merits.
- **It overlaps a store we already run.** MLflow **already** records per-run
  timing, parameters, and metrics in PostgreSQL (ADR-026, and the Sprint 7 run
  logged `accuracy`, best params, etc.). ML-*semantic* metrics (accuracy, model
  params) belong in **MLflow**, which is built for run-indexed experiment data, not
  in a Prometheus time series keyed by scrape time.

**The division of ownership** follows from that last point and is itself part of
the decision: **Prometheus/KSM own operational health** (did it run, did it
succeed, how long, was it killed); **MLflow owns ML semantics** (accuracy, params,
per-run artifacts). We do **not** duplicate model quality into Prometheus. If
per-stage *operational* duration is later deemed worth its cost, a scoped
Pushgateway (or structured-log-derived metric) is the mechanism — evaluated as its
own PR against these trade-offs, not adopted blindly.

**Rejected — a scrape sidecar / keep-alive.** Keeping a metrics endpoint alive
alongside the Job (a long-running sidecar) would prevent the Job from ever
**completing** — the pod would never reach `Succeeded`, defeating the entire
run-to-completion model (ADR-009) and breaking the KSM success signal it is
supposed to feed. Rejected outright.

### 4. Component ownership and placement

- A dedicated **`monitoring` namespace** owns the stack (Prometheus, Grafana,
  kube-state-metrics, node-exporter, blackbox-exporter),
  separating platform-observability lifecycle from the `mlops` workload namespace.
  > **Refined ([ADR-031](ADR-031-mlflow-postgres-monitoring.md), Sprint 8 PR 4).**
  > **postgres-exporter is the exception**: PR 4 places it in the **`mlops`**
  > namespace, beside the database, so its dedicated DB-credential Secret never has
  > to be copied into `monitoring` (Prometheus scrapes it cross-namespace). This ADR
  > listed it under `monitoring`; ADR-031 supersedes that placement for the
  > credential-locality reason, and rejects the `monitoring` placement explicitly.
- Every monitoring workload **inherits the same hardened baseline** the rest of the
  fleet already enforces (ADR-010): non-root numeric uid, `allowPrivilegeEscalation:
  false`, drop `ALL`, seccomp `RuntimeDefault`, no token automount unless a
  component genuinely needs the API (KSM and Prometheus's Kubernetes SD **do** need
  scoped read-only RBAC — the documented, least-privilege exception). The extended
  `k8s/validate.py` contract (ADR-012) must cover the monitoring namespace too.
- **Cross-namespace scraping** (Prometheus in `monitoring` scraping targets in
  `mlops`) is via Kubernetes service discovery with least-privilege RBAC —
  explicitly owned here so it is not improvised later.
- **postgres-exporter needs a database credential**: a dedicated **read-only
  monitoring role** in PostgreSQL, delivered as an out-of-band Secret (the same
  never-committed pattern as `mlflow-db-credentials`). This new identity is why
  Postgres internals land in a later PR, not for free.

### 5. Retention and resource trade-offs

- **Local TSDB, short retention.** Prometheus keeps a **short local retention**
  window (target **7–15 days**), sized for a short-lived validation cluster. **No
  long-term/remote store** (Thanos, Cortex, Mimir, AMP) — that is production
  capacity this project does not need and ADR-020 says not to pay for.
- **Storage is a deliberate trade-off**, resolved per environment: a small **PVC**
  (durable across a Prometheus pod restart, but a standing EBS cost and a teardown
  step) **versus** an **`emptyDir`** (free and truly ephemeral, but metrics vanish
  on pod restart/teardown). For the ephemeral cloud validation run, `emptyDir` with
  short retention is acceptable **provided the runtime evidence is captured while
  the stack is live**; a modest PVC is the option where survive-a-restart matters.
  The choice is documented, not hidden.
- **Measured, not guessed, resources.** Every monitoring component gets
  requests/limits derived from **measured** usage in the runtime PRs, exactly as
  ADR-011 did for the pipeline and ADR-026 for MLflow. This ADR does not invent
  numbers ahead of measurement.

### 6. What is deferred, and why

- **Distributed tracing (Jaeger / Tempo / OpenTelemetry traces) — deferred.** The
  pipeline is **four sequential stages in one process**, not a fan-out of network
  services; there is no distributed request path to trace. Per-run/per-stage timing
  is already available (MLflow run data; KSM run duration). Tracing would add a
  collector, storage, and app instrumentation for a question the workload's shape
  does not pose. Revisit only if a request-serving surface (e.g. model serving,
  roadmap v6) appears.
- **Centralized log aggregation (Loki / ELK / CloudWatch Logs) — deferred.** The
  pipeline already emits **structured logs** ([Logging Strategy](../logging.md))
  that are inspectable with `kubectl logs`. Metrics answer *"is it healthy / did it
  succeed?"* first; logs answer *"why did it fail?"* and remain reachable via
  `kubectl` for this single-operator, short-lived cluster. Aggregation is real
  value with its own storage cost and is a clean, separately-justified follow-on —
  not a prerequisite for the metric layer.
- **Alertmanager routing to real channels (email/Slack/PagerDuty) — deferred to
  the alerting PR** and even then kept minimal: alert *rules* are defined against
  the objectives below; wiring them to external notifiers is an operator step, not
  an architecture claim.

### 7. Operational objectives (SLO-*style*, not production SLOs)

A **small** set of defensible, early-warning expectations — explicitly **not**
production SLOs (no error budgets, no long-term measurement, single-operator
ephemeral cluster):

- **Pipeline success indicator** — the last submitted pipeline Job reaches
  `Complete` (exit 0). Tracked as an indicator; **not** a "% of runs" SLO, which
  would require run history this project does not yet accumulate.
- **MLflow availability** — `/health` returns 200 (blackbox `up == 1`) throughout a
  pipeline execution window. Reported as measured availability; **no** 99.x% claim
  (single replica, ephemeral cluster).
- **Resource-headroom expectation** — no workload sustains **> 90%** of its memory
  limit (an early OOM warning, grounded in the ADR-011/ADR-026 measured limits).
- **Storage-headroom expectation** — the Postgres PVC stays **< 85%** full (the
  fixed 1 Gi makes this the single highest-value alert).

These are stated as expectations with thresholds an operator can act on, and the
document is explicit that **production SLO compliance is not claimed** without
long-term data.

## Alternatives Considered

- **AWS CloudWatch Container Insights.** Rejected as the primary stack: it is
  AWS-locked (no local parity — it cannot run on the Docker Desktop cluster the
  project validates on), bills per metric/log, and contradicts the self-hosted,
  portable posture ADR-026/ADR-027 established. It remains a reasonable *future*
  option for a genuinely production AWS deployment; not now.
- **Datadog / New Relic / Grafana Cloud (SaaS).** Rejected: external SaaS in the
  runtime path (the exact thing ADR-026 removed for tracking), per-host/per-metric
  cost on a short-lived cluster, and data leaving the operator's account. Overkill
  for a single-operator validation environment.
- **A bare OpenTelemetry Collector + backend.** Reasonable and more future-proof,
  but heavier to stand up and operate for a **metrics-first** need that Prometheus
  + the standard exporters already meet directly. The OTel path is the natural
  door to tracing later; adopting it now would front-load complexity for deferred
  requirements.
- **Pushgateway as the primary batch mechanism.** Rejected as *primary* (see
  Decision § 3): it needs an app change, has sticky/stale-metric hazards, and
  overlaps MLflow for ML metrics; KSM answers the operational questions with none
  of those costs. Retained as the *deferred, scoped* option for per-stage
  operational duration only.
- **A custom Kubernetes/MLflow exporter.** Rejected as unnecessary: it would
  duplicate KSM (Job state) and MLflow (run metrics) — build-and-maintain cost for
  signals two systems we already run expose for free.
- **Do nothing / keep `kubectl` + logs.** Rejected: it is precisely the ⬜ gap the
  roadmap names; "a human at a terminal at the right moment" is not operability.

## Consequences

**Positive**

- A concrete, reviewed observability **design of record** exists before a single
  component is installed — later PRs implement against fixed acceptance criteria
  rather than improvising.
- The **hard problem is solved on paper**: an ephemeral Job's operational metrics
  are queryable via KSM's view of the persistent Job object, with a stated
  TTL/scrape queryability contract — no app change, no anti-pattern.
- Clear **ownership boundary**: Prometheus/KSM = operational health; MLflow = ML
  semantics; logs = root-cause. No duplicated model-quality metrics.
- The stack matches the project's **self-hosted, portable, cost-controlled**
  posture and reuses the established hardening/validation/evidence machinery.

**Negative / trade-offs**

- **No per-stage pipeline timing** until (and unless) a Pushgateway or
  log-derived metric is justified in a later PR — run-level duration only for now.
- **postgres-exporter adds a new identity** (a read-only DB monitoring role +
  out-of-band Secret) and blackbox/exporters add moving parts and measured resource
  cost to a deliberately lean cluster.
- **Short local retention, no long-term store** — historical analysis beyond the
  retention window (and true "% of runs succeeded" SLOs) is out of scope; evidence
  must be captured while the stack is live.
- **Tracing and log aggregation remain deferred**, so deep "why did it fail across
  components" investigation still leans on `kubectl logs`.

## What This Decision Does **Not** Imply

- **Nothing is deployed by this ADR/PR.** It is architecture only; no Prometheus,
  Grafana, exporter, alert, or manifest change ships here.
- **Not production monitoring.** These are validation-environment objectives and
  thresholds, not certified production SLOs; no error budget or long-term
  availability is claimed.
- **Not a commitment to Pushgateway, tracing, or log aggregation** — those are
  explicitly deferred and each must clear its own justification before adoption.

## Later-PR acceptance criteria (Sprint 8 PRs 2–6)

Ratified here, detailed in
[`docs/observability.md` § Sprint 8 delivery plan](../observability.md#9-sprint-8-delivery-plan-prs-26):

- **PR 2 — Metrics core.** Prometheus + kube-state-metrics + node-exporter in a
  hardened `monitoring` namespace; cAdvisor scrape; least-privilege scrape RBAC.
  Layer 1 signals and the **Layer 2 batch-Job** signals (via KSM) are collectable.
  Passes an extended `k8s/validate.py`; static/dry-run CI only — no deploy claim.
- **PR 3 — Dashboards.** Grafana (internal-only) with the four-layer dashboards
  rendering the documented signals, provisioned as code.
  > **Resequenced ([ADR-030](ADR-030-pipeline-operational-metrics.md)).** The actual
  > PR 3 delivered the pipeline's **operational metrics** (per-stage Pushgateway) —
  > the deferred item from § 3 — and **Grafana dashboards moved to a later PR**. The
  > dashboards then have the per-stage series to render in addition to Layers 1–4.
- **PR 4 — MLflow & PostgreSQL depth.** blackbox-exporter (`/health`) +
  postgres-exporter with a least-privilege read-only monitoring role; Layer 3/4
  availability, memory-headroom, and **PVC-fill** signals present.
  > **Delivered ([ADR-031](ADR-031-mlflow-postgres-monitoring.md)).** blackbox-exporter
  > (Layer 3 MLflow `/health`), postgres-exporter with a dedicated `pg_monitor`-only
  > role (Layer 4 up/connections/size), and a scoped kubelet volume-stats scrape
  > (Layer 4 PVC-fill) shipped as manifests + scrape config + validation. The
  > run-level replica/readiness/restart/CPU/memory signals were already collectable
  > from PR 2's KSM + cAdvisor; PR 4 documents those queries and adds the depth
  > exporters. **Not deployed** — runtime proof is PR 6.
- **PR 5 — Alerting.** Prometheus alert rules encoding *exactly* the § 7 objectives
  (pipeline failure/OOM, MLflow down/memory, node not ready, PVC almost full) —
  no arbitrary alerts.
- **PR 6 — Runtime evidence & operations.** Provision → run the pipeline → prove
  the four-layer signals (including a post-completion Job success/duration read and
  an induced OOM/`MLflowDown`/PVC-fill alert) → tear down and verify clean
  (ADR-020); an observability operations runbook, matching the Sprint 6/7 evidence
  conventions.
