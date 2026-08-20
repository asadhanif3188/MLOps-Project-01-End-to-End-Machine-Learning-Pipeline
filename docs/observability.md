# Observability & Operations Architecture

> **Status:** 🚧 **Architecture / design of record — nothing is deployed yet.**
> This document defines the Sprint 8 observability model *before* any monitoring
> component is installed. The decision rationale is
> [ADR-028](decisions/ADR-028-observability-architecture.md); the target picture is
> the [observability architecture diagram](diagrams/observability-architecture/README.md).
> No Prometheus, Grafana, exporter, or alert exists in the repository at the time
> of writing — the [Sprint 8 delivery plan](#9-sprint-8-delivery-plan-prs-26) below
> describes how each is added, and the [runtime-evidence expectations](#runtime-evidence-what-later-sprint-8-prs-must-prove)
> describe what must be *proven* before any of it is claimed to work.

This is the operational counterpart to the runtime and platform documents
([Kubernetes Operations](kubernetes-operations.md),
[Cloud Operations](cloud-operations.md), [MLflow Platform](mlflow-platform.md)):
it says **what we watch, why, where each signal comes from, and what we
deliberately do not watch**.

---

## 1. Why observability, why now

After Sprint 7 the platform is cloud-native and proven on real EKS, but its
operational visibility is still `kubectl get / describe / logs` plus the
pipeline's [structured logs](logging.md). Answering an ordinary operational
question — *"did the overnight pipeline run succeed?"*, *"is MLflow about to hit
its memory limit?"*, *"is the Postgres volume filling up?"* — needs a human at a
terminal at the right moment. The [roadmap](roadmap.md) names this gap plainly:
monitoring is now 🚧 in v5 (this PR defines the architecture) and a headline
objective of v6.

Sprint 8 closes the *design* of that gap. The stack is **Prometheus + Grafana**,
self-hosted and internal-only, for the reasons in
[ADR-028 § 1](decisions/ADR-028-observability-architecture.md#1-stack--prometheus-for-metrics-grafana-for-dashboards):
it is the Kubernetes-native standard, its exporter ecosystem already covers every
layer here, it runs identically on local Docker Desktop and EKS (local/cloud
parity), and it fits the project's self-hosted, cost-controlled, ephemeral posture
([ADR-020](decisions/ADR-020-cloud-lifecycle-cost-control.md),
[ADR-026](decisions/ADR-026-in-cluster-mlflow-platform.md)).

**Guiding principle — measure what the workload needs, not what is possible.**
Every signal below is tied to a question an operator would actually act on for
*this* workload. We do not add a metric merely because an exporter can emit it —
the same discipline ADR-011 applied to resource limits.

---

## 2. What is being observed (the four layers)

The platform has four things worth observing, each with a different shape:

| # | Layer | Kind | Key property for observability |
|---|-------|------|--------------------------------|
| 1 | Kubernetes platform | EKS control plane + a small managed node group (2 nodes by default; [`terraform/eks.tf`](../terraform/eks.tf)) | Long-running; standard node/pod signals |
| 2 | **MLOps pipeline** | `batch/v1` **Job** ([`k8s/base/job.yaml`](../k8s/base/job.yaml)) | **Ephemeral** — pod exits in < 1 min; no Service, no probes (ADR-009/011) |
| 3 | MLflow tracking server | `Deployment` ([`k8s/base/mlflow/deployment.yaml`](../k8s/base/mlflow/deployment.yaml)) | Long-running; `/health` exists, **no native `/metrics`** |
| 4 | PostgreSQL | `StatefulSet` ([`k8s/base/mlflow/postgres.yaml`](../k8s/base/mlflow/postgres.yaml)) | Single-writer; **fixed 1 Gi PVC**; durability is the point |

Layer 2 is the hard one and is treated in depth in [§ 4](#4-the-batch-job-problem-keeping-an-ephemeral-jobs-metrics-queryable).

---

## 3. Signal catalogue — per layer

For each layer: the **operational questions**, the **signals** that answer them,
the **likely Prometheus source**, the **dashboard** use, the **alerting** use, and
the **limitations**. Sources: **KSM** = kube-state-metrics, **node-exporter** =
node daemon, **cAdvisor** = the kubelet's built-in per-container metrics,
**blackbox** = blackbox-exporter, **pg-exporter** = postgres-exporter, **kubelet**
= kubelet volume stats.

### Layer 1 — Kubernetes platform

| Operational question | Signal(s) | Source | Dashboard | Alert |
|---|---|---|---|---|
| Are nodes Ready? | `kube_node_status_condition{condition="Ready",status="true"}` | KSM | Node status tile | **NodeNotReady** (≠ Ready for N min) |
| Is a node under memory/CPU/disk pressure? | `node_memory_MemAvailable_bytes`, `node_cpu_seconds_total`, `node_filesystem_avail_bytes`; `kube_node_status_condition{condition=~"MemoryPressure|DiskPressure"}` | node-exporter, KSM | Node resource gauges | **NodeUnderPressure** |
| Are pods repeatedly restarting? | `rate(kube_pod_container_status_restarts_total[15m])`; `kube_pod_container_status_waiting_reason{reason="CrashLoopBackOff"}` | KSM | Restart heatmap | **KubePodCrashLooping** |
| Is a workload approaching its memory limit? | `container_memory_working_set_bytes / kube_pod_container_resource_limits{resource="memory"}` | cAdvisor + KSM | Usage-vs-limit % per workload | **WorkloadMemoryHigh** (> 90%, § 6) |
| Is CPU being throttled? | `rate(container_cpu_cfs_throttled_periods_total[5m])` | cAdvisor | Throttle panel | *(info only — expected on the CPU-capped Job, ADR-011)* |
| Are pods stuck Pending / PVCs unbound? | `kube_pod_status_phase{phase="Pending"}`, `kube_persistentvolumeclaim_status_phase{phase!="Bound"}` | KSM | Scheduling panel | **PodPending / PVCUnbound** |

**Limitations.** The validation cluster is **small** (a 2-node group by default,
ADR-017; the Sprint 7 validation run used 1), so multi-node/HA signals have limited
value; CPU throttling on the pipeline is
**expected** (the CPU limit is the memory-safety control, ADR-011) and is
information, not an alert.

### Layer 2 — MLOps pipeline (the ephemeral Job)

| Operational question | Signal(s) | Source | Dashboard | Alert |
|---|---|---|---|---|
| Did the **last** pipeline Job succeed? | `kube_job_status_succeeded`, `kube_job_status_failed` | **KSM** | Last-run status tile | **PipelineJobFailed** |
| When did it last run? | `kube_job_status_start_time` / `_completion_time` | KSM | Last-run timestamp | *(info)* |
| How long did the run take? | `kube_job_status_completion_time − kube_job_status_start_time` | KSM | Run-duration trend | **PipelineJobSlow** *(optional; vs `activeDeadlineSeconds`)* |
| Did it retry (burn `backoffLimit`)? | `kube_job_status_failed` > 0 with an eventual success | KSM | Attempts panel | *(info — transient-fault path, ADR-011)* |
| Was it OOMKilled? | `kube_pod_container_status_last_terminated_reason{reason="OOMKilled"}` | KSM | Termination-reason panel | **PipelineJobOOMKilled** |
| Did it hit the deadline? | Job `DeadlineExceeded` condition / `kube_job_status_active` stuck | KSM | Stall indicator | **PipelineJobDeadlineExceeded** |
| **Which stage took the longest?** | *(per-stage duration — NOT available from KSM)* | *Pushgateway or log-derived — **deferred**, § 4* | *(future)* | — |
| Model accuracy / best params for a run | `accuracy`, `best_*` params | **MLflow** (not Prometheus) — [MLflow Platform](mlflow-platform.md) | MLflow UI | — |

**Limitations.** KSM gives **run-level** outcome and duration but **not per-stage**
granularity; ML-*semantic* metrics live in **MLflow**, not Prometheus (see the
ownership split in [§ 5](#5-ownership--who-owns-which-signal)). Why KSM works at
all for an exited pod is [§ 4](#4-the-batch-job-problem-keeping-an-ephemeral-jobs-metrics-queryable).

### Layer 3 — MLflow tracking server

| Operational question | Signal(s) | Source | Dashboard | Alert |
|---|---|---|---|---|
| Is MLflow **available**? | `probe_success` against `/health`; `probe_http_status_code` | **blackbox** | Availability tile | **MLflowDown** (`probe_success == 0` for N min) |
| Is the Deployment serving its replica? | `kube_deployment_status_replicas_available{deployment="mlflow"}` | KSM | Replica panel | **MLflowNoReplicas** |
| Is it restarting? | `kube_pod_container_status_restarts_total{pod=~"mlflow-.*"}` | KSM | Restart panel | **MLflowCrashLooping** |
| Is it approaching its memory ceiling? | `container_memory_working_set_bytes` vs the **2 Gi** limit (measured ~1.7 GiB, ADR-026) | cAdvisor + KSM | Memory-headroom gauge | **MLflowMemoryHigh** (> 90%, § 6) |
| Is `/health` latency degrading? | `probe_duration_seconds` | blackbox | Latency panel | *(info)* |

**Limitations.** MLflow ships **no native Prometheus `/metrics`**, so there are
**no request-level RED metrics** (rate/errors/duration per API call) without an
app-side exporter — **deferred** (ADR-028 § 6). `/health` is a **shallow** check
(server up), not a deep DB-connectivity assertion.

### Layer 4 — PostgreSQL

| Operational question | Signal(s) | Source | Dashboard | Alert |
|---|---|---|---|---|
| Is the DB up / accepting connections? | `pg_up`; also `pg_isready` (existing probe, ADR-026) | pg-exporter | DB-up tile | **PostgresDown** |
| **Is the 1 Gi PVC filling up?** | `kubelet_volume_stats_used_bytes / kubelet_volume_stats_capacity_bytes` | kubelet | PVC-usage gauge | **PostgresPVCAlmostFull** (> 85%, § 6) |
| Connections near the limit? | `pg_stat_activity_count` vs `pg_settings_max_connections` | pg-exporter | Connections panel | **PostgresConnectionsHigh** |
| Is it restarting / StatefulSet ready? | `kube_statefulset_status_replicas_ready{statefulset="mlflow-postgres"}`; restarts | KSM | Readiness panel | **PostgresNotReady** |
| Is it approaching its memory limit? | `container_memory_working_set_bytes` vs the **512 Mi** limit (ADR-026) | cAdvisor + KSM | Memory gauge | **PostgresMemoryHigh** (> 90%) |
| DB size / growth | `pg_database_size_bytes` | pg-exporter | DB-size trend | *(feeds the PVC alert)* |

**Limitations.** pg-exporter needs a **dedicated read-only monitoring role** and an
out-of-band Secret (the never-committed pattern of `mlflow-db-credentials`) — a new
identity, which is why Layer 4 depth lands in a later PR. Single instance → no
replication/lag signals (none exist by design, ADR-026). The **PVC-fill** alert is
the single highest-value signal here given the fixed 1 Gi.

---

## 4. The batch-Job problem: keeping an ephemeral Job's metrics queryable

**The problem.** Prometheus **pulls**: it scrapes a live target on an interval. The
pipeline pod **exits** seconds after it finishes (ADR-009/011). A `/metrics`
endpoint on the Job's container is therefore unscrapable — there is nothing to
scrape between runs, and a fast run can finish *between two scrapes*. A naïve
long-running `/metrics` endpoint is **not** automatically the right design here.

**The chosen answer — kube-state-metrics reflects the persistent `Job` (and its
finished `Pod`) object, not a live scrape target.** KSM is a long-running
Deployment that watches the Kubernetes API and exposes API-object *state* as
metrics. The **`Job` object outlives the pod's process** (until deleted or
`ttlSecondsAfterFinished` elapses), so the Job's terminal state is a **stable
series on an always-up target** — no live pod required. Most Layer 2 signals come
from the **Job** object (`kube_job_status_*`); the one exception is **OOMKilled**,
which is a **Pod**-object series (`kube_pod_container_status_last_terminated_reason`)
— it stays scrapable because the Job's finished pod is retained by owner-reference
for as long as the Job itself (the same `ttlSecondsAfterFinished` cascade removes
both). Either way KSM is scraping a persistent API object, not the exited process,
so every Layer 2 operational question except per-stage timing is answered with
**zero application change**.

### Trade-off analysis of the candidate approaches

| Approach | What it gives | Pros | Cons | Verdict |
|---|---|---|---|---|
| **kube-state-metrics** (Job/Pod objects) | success/fail, start & completion time → **run duration**, retries, OOMKilled, deadline, phase | No app change; API-server-backed & robust; **persists after the pod exits**; already needed for Layers 1/3/4 | **Run-level only** (no per-stage); depends on the finished Job object outliving one scrape | **Primary — adopted** |
| **Pushgateway** | per-**stage** duration; any custom job metric | The Prometheus-sanctioned way to get metrics *out of* a batch job; stage granularity | **Sticky/stale metrics** (persist until overwritten/deleted, no `up` semantics); a **single point of failure**; needs an **app push step**; **overlaps MLflow** for ML metrics | **Deferred** — reconsider for per-stage *operational* duration only |
| **Custom exporter / state metric** | anything (query the K8s API or MLflow) | Fully flexible | Duplicates KSM (Job state) **and** MLflow (run metrics); build-and-maintain cost | **Rejected as unnecessary** |
| **MLflow as the metric store** | accuracy, params, per-run artifacts, run timing | Already running; built for **run-indexed** experiment data | Not an operational/alerting system; not Prometheus-queryable | **Keep for ML semantics** (not operational health) |
| **Long-running scrape sidecar / keep-alive** | a live `/metrics` during & after the run | — | **Keeps the pod alive → the Job never `Completes`**, breaking the run-to-completion model **and** the KSM success signal | **Rejected outright** |

### The queryability contract (a design requirement for the runtime PRs)

KSM exposes the Job's terminal state **only while the `Job` object exists**, and
Prometheus must scrape it **at least once** in that window. Two requirements make
this reliable — both are design obligations, not afterthoughts:

1. **Retain the finished Job long enough.** Set a **generous
   `ttlSecondsAfterFinished`** on the Job (or do not auto-delete it) so
   `kube_job_status_succeeded/_failed/_completion_time` linger for **many** scrape
   intervals. A tiny TTL could let the terminal state vanish before Prometheus
   sees it.
2. **Scrape once, keep forever (within retention).** Once Prometheus has scraped
   the terminal state, those samples live in its TSDB for the **whole retention
   window** even after the Job object is deleted. So the narrow requirement is:
   *the finished Job (and its pod) must outlive one scrape interval* — which (1)
   guarantees.

> **Re-run tension (a PR-2 design note).** The base Job has a fixed
> `metadata.name: mlops-pipeline` ([`k8s/base/job.yaml`](../k8s/base/job.yaml)), and
> a finished Job of that name must be deleted before the same-named Job can be
> re-submitted. So "retain the finished Job for many scrapes" and "re-run the
> pipeline" pull against each other: deleting the old Job to re-run drops its **live**
> KSM gauges (the already-scraped samples still persist in the TSDB per (2), but the
> "last-run" tile then tracks the newest Job). PR 2 must choose the TTL — and whether
> to move to `generateName` / delete-before-recreate — with this trade-off explicit,
> not silently.

**Deferred, and only if justified — Pushgateway for per-stage duration.** The one
genuinely operational signal KSM cannot give is *"which stage took the longest?"*.
If that is later deemed worth its cost, the mechanism is a **scoped Pushgateway**
(the pipeline pushes per-stage timings before exit) **or** a **log-derived metric**
(the stages already emit structured logs, [logging.md](logging.md)). It is **not**
adopted now: it needs an application change (out of scope for an architecture-only
PR) and carries the sticky-metric hazards above. It will be evaluated as its own PR
against this table — **not adopted blindly.**

---

## 5. Ownership — who owns which signal

A clean split, ratified in ADR-028 § 3, prevents duplicated and confusing data:

- **Prometheus / kube-state-metrics own operational health** — did it run, did it
  succeed, how long, was it killed, is it up, is it full. Everything alertable.
- **MLflow owns ML semantics** — model accuracy, hyper-parameters, per-run
  artifacts and lineage ([MLflow Platform](mlflow-platform.md)). Run-indexed
  experiment data, **not** duplicated into Prometheus.
- **Structured logs own root cause** — *why* a run failed
  ([Logging Strategy](logging.md)), reachable via `kubectl logs`.

Component ownership/placement: a dedicated **`monitoring` namespace** holds the
stack; every monitoring workload inherits the hardened baseline (ADR-010) and is
covered by the extended `k8s/validate.py` contract (ADR-012); cross-namespace
scraping uses least-privilege RBAC; pg-exporter uses a dedicated **read-only** DB
role. Details: [ADR-028 § 4](decisions/ADR-028-observability-architecture.md#4-component-ownership-and-placement).

---

## 6. Operational objectives (SLO-*style*, not production SLOs)

A **small**, defensible set of early-warning expectations. These are **not**
production SLOs — there is no error budget, no long-term measurement, and this is a
single-operator ephemeral cluster (ADR-020). **Production SLO compliance is not
claimed** without long-term data.

| Objective | Expectation | Basis | Encoded as |
|---|---|---|---|
| **Pipeline success indicator** | The last submitted pipeline Job reaches `Complete` (exit 0) | ADR-009/011 completion semantics | `PipelineJobFailed` alert; last-run tile |
| **MLflow availability** | `/health` returns 200 during a pipeline execution window | Existing `/health` probe (ADR-026) | `MLflowDown` alert (blackbox `probe_success`) |
| **Memory headroom** | No **long-running** workload sustains **> 90%** of its memory limit | Measured limits (ADR-026: 2 Gi MLflow, 512 Mi PG) | `*MemoryHigh` alerts |
| **Storage headroom** | Postgres PVC stays **< 85%** full | Fixed 1 Gi PVC (ADR-026) | `PostgresPVCAlmostFull` alert |

> **Memory headroom applies to the long-running workloads (MLflow, PostgreSQL).**
> The pipeline **Job** runs for under a minute, so a "sustained > 90% for N min"
> expectation cannot meaningfully fire for it — its memory-safety control is the
> **`OOMKilled`** signal (`PipelineJobOOMKilled`, [Layer 2](#layer-2--mlops-pipeline-the-ephemeral-job)),
> not a sustained-usage threshold. The pipeline's 512 Mi limit is still enforced by
> the kernel (ADR-011); it just surfaces as a kill event, not a headroom gauge.

> **Not claimed:** a "% of runs succeeded" SLO (needs run history the project does
> not yet accumulate), a 99.x% availability figure (single replica, ephemeral
> cluster), or any latency SLO for the batch pipeline (it has no latency contract,
> ADR-011).

---

## 7. Deferred observability areas (and why)

| Area | Status | Why deferred |
|---|---|---|
| **Distributed tracing** (Jaeger/Tempo/OTel traces) | Deferred | The pipeline is four sequential stages in one process — no distributed request path to trace; per-stage/run timing already in MLflow + KSM. Revisit if model *serving* appears (roadmap v6). |
| **Centralized log aggregation** (Loki/ELK/CloudWatch Logs) | Deferred | Structured logs already exist and are `kubectl`-reachable ([logging.md](logging.md)); metrics answer "healthy/succeeded?" first, logs answer "why?". Aggregation is a separately-justified follow-on with its own storage cost. |
| **MLflow request-level RED metrics** | Deferred | MLflow has no native `/metrics`; needs an app-side exporter. Availability (blackbox `/health`) + resource (cAdvisor) cover the operational question now. |
| **Per-stage pipeline duration** | Deferred | Needs Pushgateway or a log-derived metric (app change) — see [§ 4](#4-the-batch-job-problem-keeping-an-ephemeral-jobs-metrics-queryable). |
| **Long-term / remote metric store** (Thanos/Cortex/Mimir/AMP) | Out of scope | Short local retention (7–15 d) suits an ephemeral validation cluster; long-term capacity is a production concern (ADR-020). |
| **Alertmanager routing to real channels** (email/Slack/PagerDuty) | Deferred to PR 5, minimal | Alert *rules* are defined against § 6; wiring external notifiers is an operator step, not an architecture claim. |

---

## 8. Retention & resource trade-offs

- **Local TSDB, short retention (target 7–15 days).** No long-term/remote store —
  that is production capacity this project does not need (ADR-020). The window is
  meaningful mainly for a **persistent local cluster** (e.g. Docker Desktop); on the
  **same-day provision→prove→destroy** cloud run (ADR-020) the effective history is
  only as long as the stack is live, so cloud evidence is captured in-session (see
  [§ runtime evidence](#runtime-evidence-what-later-sprint-8-prs-must-prove)).
- **Storage: PVC vs `emptyDir`, resolved per environment.** A small **PVC**
  survives a Prometheus pod restart but is a standing EBS cost and a teardown step;
  an **`emptyDir`** is free and truly ephemeral but loses metrics on restart/teardown.
  For the ephemeral cloud validation run, `emptyDir` + short retention is
  acceptable **provided evidence is captured while the stack is live**; a modest PVC
  is the option where survive-a-restart matters. The choice is documented per PR,
  never hidden.
- **Measured, not guessed, resources.** Every monitoring component's
  requests/limits are derived from **measured** usage in the runtime PRs — the same
  discipline as ADR-011 (pipeline) and ADR-026 (MLflow). No invented numbers here.

---

## 9. Sprint 8 delivery plan (PRs 2–6)

This PR (**PR 1**) is architecture/documentation only. The remaining PRs implement
the design against these acceptance criteria (ratified in ADR-028; all Kubernetes
work stays static/dry-run in CI per ADR-012 until the runtime-evidence PR).

| PR | Scope | Acceptance criteria |
|----|-------|---------------------|
| **PR 2 — Metrics core** ✅ *(manifests; not deployed)* | Prometheus + kube-state-metrics + node-exporter in a hardened `monitoring` namespace; cAdvisor scrape; least-privilege scrape RBAC | **Delivered:** [`k8s/monitoring/`](../k8s/monitoring/) renders Layer 1 signals **and** the Layer 2 batch-Job signals (via KSM); Job `ttlSecondsAfterFinished` set to honour the [queryability contract](#the-queryability-contract-a-design-requirement-for-the-runtime-prs); extended `k8s/validate.py` monitoring pass green over `k8s/monitoring`; `kustomize build` + `kubeconform` (CI) green. Minimal hand-written Kustomize (no Helm), ephemeral `emptyDir` TSDB, read-only RBAC, one documented node-exporter Pod Security exception — [ADR-029](decisions/ADR-029-monitoring-foundation.md), [Monitoring Operations](monitoring-operations.md). **No deploy claim** (no live cluster; runtime proof is PR 6). |
| **PR 3 — Dashboards** | Grafana (internal-only ClusterIP) with four-layer dashboards provisioned as code | Dashboards render every documented signal for Layers 1–4; no public exposure (matches ADR-026 UI posture) |
| **PR 4 — MLflow & PostgreSQL depth** | blackbox-exporter (`/health`) + postgres-exporter with a dedicated **read-only** monitoring role (out-of-band Secret) | Layer 3 availability/memory + Layer 4 up/**PVC-fill**/connections signals present; new DB role is least-privilege; secret hygiene checks pass |
| **PR 5 — Alerting** | Prometheus alert rules encoding **exactly** the [§ 6](#6-operational-objectives-slo-style-not-production-slos) objectives | The defined alert set exists and no others (no arbitrary alerts); rules unit-testable (e.g. `promtool test rules`) |
| **PR 6 — Runtime evidence & operations** | Provision → run pipeline → prove signals → tear down; observability operations runbook | The [runtime-evidence expectations](#runtime-evidence-what-later-sprint-8-prs-must-prove) are all met and recorded in a redacted proof doc matching the Sprint 6/7 conventions; environment destroyed & verified clean (ADR-020) |

---

## Runtime evidence: what later Sprint 8 PRs must prove

The runtime-evidence PR (PR 6) is the **proof gate**. It must demonstrate, on a
live cluster and recorded with the [Sprint 7 evidence](proof/sprint-07-runtime-evidence.md)
conventions (redacted, honest about failures, torn down and verified clean):

1. **All targets Up.** Prometheus `up == 1` for node-exporter, kube-state-metrics,
   cAdvisor, blackbox-exporter, and postgres-exporter.
2. **The ephemeral Job is observable *after* it exits.** After the pipeline pod is
   `Succeeded` **and gone**, the dashboard still shows
   `kube_job_status_succeeded == 1`, the run's **start/completion timestamps**, and
   the **computed run duration** — proving the [queryability contract](#the-queryability-contract-a-design-requirement-for-the-runtime-prs)
   holds (the finished Job outlived a scrape).
3. **A failure is surfaced and alerts.** An induced failure — e.g. the ADR-011
   memory-limit OOM (run at 64 Mi) — shows
   `last_terminated_reason="OOMKilled"` / `kube_job_status_failed` and **fires**
   `PipelineJobOOMKilled` / `PipelineJobFailed`.
4. **MLflow-down is detected.** Scaling MLflow to 0 drives blackbox
   `probe_success == 0` and fires **MLflowDown**; recovery clears it.
5. **PVC-fill is detected.** Postgres PVC usage crossing 85% fires
   **PostgresPVCAlmostFull** (induced or clearly documented if simulated).
6. **Dashboards render all four layers** with real data from the live run.
7. **Clean teardown.** The monitoring stack and any PVC are removed; the
   environment is destroyed and **verified clean** (ADR-020).

Until these are proven, this document and ADR-028 make a **design** claim only —
consistent with the project's rule that *structurally valid ≠ runtime-complete*.

---

## Related documentation

- [ADR-028 — Observability Architecture](decisions/ADR-028-observability-architecture.md) (design of record)
- [Observability architecture diagram](diagrams/observability-architecture/README.md)
- [Kubernetes Operations](kubernetes-operations.md) · [Cloud Operations](cloud-operations.md) · [MLflow Platform](mlflow-platform.md)
- [Logging Strategy](logging.md) · [Roadmap](roadmap.md)
- ADR-009 (workload model) · ADR-011 (resources/lifecycle) · ADR-020 (cost control) · ADR-026 (MLflow platform)
