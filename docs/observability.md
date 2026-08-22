# Observability & Operations Architecture

> **Status:** 🚧 **Architecture / design of record — nothing is deployed yet.**
> This document defines the Sprint 8 observability model *before* any monitoring
> component is installed. The decision rationale is
> [ADR-028](decisions/ADR-028-observability-architecture.md); the target picture is
> the [observability architecture diagram](diagrams/observability-architecture/README.md).
> No Prometheus, Grafana, exporter, or alert exists in the repository at the time
> of writing — the [Sprint 8 delivery plan](#9-sprint-8-delivery-plan-prs-27) below
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
| Is a workload approaching its memory limit? | `container_memory_working_set_bytes / kube_pod_container_resource_limits{resource="memory"}` | cAdvisor + KSM | Usage-vs-limit % per workload | **`MLflowMemoryHigh`** / **`PostgresMemoryHigh`** (> 90%, § 6) — workload-specific, one per long-running workload against its measured limit (PR 6, [ADR-033](decisions/ADR-033-alerting.md)); the pipeline Job's memory-safety signal is **`PipelineJobOOMKilled`**, not a headroom gauge (§ 6 note) |
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
| **Which stage took the longest?** | `mlops_pipeline_stage_duration_seconds{stage}` | **Pushgateway** (pipeline push, PR 3) | Per-stage duration bars | *(info)* |
| **Which stage failed?** | `mlops_pipeline_stage_success{stage} == 0` | **Pushgateway** (pipeline push, PR 3) | Stage status tiles | *(feeds `PipelineJobFailed`)* |
| How long did the dataset fetch take? | `mlops_pipeline_stage_duration_seconds{stage="fetch_dataset"}` | **Pushgateway** (pipeline push, PR 3) | Fetch-duration trend | *(info)* |
| Model accuracy / best params for a run | `accuracy`, `best_*` params | **MLflow** (not Prometheus) — [MLflow Platform](mlflow-platform.md) | MLflow UI | — |

**Limitations.** KSM gives **run-level** outcome and duration; **per-stage** duration
and per-stage failure attribution now come from the **Pushgateway** the pipeline
pushes to before exiting (PR 3, [ADR-030](decisions/ADR-030-pipeline-operational-metrics.md);
see [§ 4](#4-the-batch-job-problem-keeping-an-ephemeral-jobs-metrics-queryable)).
ML-*semantic* metrics still live in **MLflow**, not Prometheus — the pipeline's
metric emitter has no code path to accuracy/params (the ownership split,
[§ 5](#5-ownership--who-owns-which-signal)). Why KSM works at all for an exited pod
is [§ 4](#4-the-batch-job-problem-keeping-an-ephemeral-jobs-metrics-queryable).

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
identity, **delivered in PR 4** ([ADR-031](decisions/ADR-031-mlflow-postgres-monitoring.md)):
a `pg_monitor`-only `mlflow_exporter` role, the exporter co-located with the DB in
the `mlops` namespace so the credential never enters `monitoring`. Single instance →
no replication/lag signals (none exist by design, ADR-026). The **PVC-fill** signal
(the single highest-value one here given the fixed 1 Gi) comes from the **kubelet's**
volume stats, scraped in PR 4 scoped to `kubelet_volume_stats_*` only.

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
| **Pushgateway** | per-**stage** duration + success; any custom job metric | The Prometheus-sanctioned way to get metrics *out of* a batch job; stage granularity KSM cannot give | **Sticky/stale metrics** (persist until overwritten/deleted, no `up` semantics); a **single point of failure**; needs an **app push step**; would **overlap MLflow** if misused for ML metrics | **Adopted — scoped (PR 3, [ADR-030](decisions/ADR-030-pipeline-operational-metrics.md))**: operational per-stage duration/success only; stickiness controlled by a per-run reset; ownership boundary keeps ML metrics in MLflow |
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

**Adopted, scoped — Pushgateway for per-stage operational metrics (PR 3).** The one
genuinely operational signal KSM cannot give is *"which stage took the longest?"*
(and its sibling, *"which stage failed?"*). PR 3 adopts a **scoped Pushgateway** for
exactly that: the pipeline pushes `mlops_pipeline_stage_duration_seconds{stage}` and
`mlops_pipeline_stage_success{stage}` before each stage's process exits, and
Prometheus scrapes the always-up gateway (a 5th scrape job with `honor_labels`). The
sticky-metric hazard above is neutralised **at the producer**: one Pushgateway group
per stage (PUT-replaced), and a **per-run reset** (the `fetch-dataset` init container
DELETEs every stage group at the start of a run) so a shorter/failed run never leaves
a previous run's later-stage series behind. Cardinality is bounded (`stage` from a
fixed set — no run id / path / filename), emission is best-effort (a gateway outage
never fails the run) and disabled unless `PUSHGATEWAY_URL` is set. The **ownership
boundary is preserved**: only operational signals are pushed — accuracy/params stay
in MLflow. Full rationale, including why the brief's `*_total` counters are modelled
as last-run gauges, is in
[ADR-030](decisions/ADR-030-pipeline-operational-metrics.md). The structured logs
([logging.md](logging.md)) remain the root-cause layer.

---

## 5. Ownership — who owns which signal

A clean split, ratified in ADR-028 § 3, prevents duplicated and confusing data:

- **Prometheus / kube-state-metrics own operational health** — did it run, did it
  succeed, how long, was it killed, is it up, is it full. Everything alertable. The
  pipeline's own **per-stage** operational metrics (duration, success — pushed via
  the Pushgateway, PR 3 / [ADR-030](decisions/ADR-030-pipeline-operational-metrics.md))
  live on this side of the line too: they describe *execution*, never model quality.
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
| **Per-stage pipeline duration + failure attribution** | ✅ Delivered (PR 3) | Pushgateway push from the pipeline (`mlops_pipeline_stage_*`), bounded cardinality, per-run reset — [ADR-030](decisions/ADR-030-pipeline-operational-metrics.md), [§ 4](#4-the-batch-job-problem-keeping-an-ephemeral-jobs-metrics-queryable). |
| **Long-term / remote metric store** (Thanos/Cortex/Mimir/AMP) | Out of scope | Short local retention (7–15 d) suits an ephemeral validation cluster; long-term capacity is a production concern (ADR-020). |
| **Alert *rules*** (Prometheus) | ✅ Delivered (PR 6) | Eight high-signal rules encoding § 6 + the § 3 catalogue, promtool-unit-tested — [ADR-033](decisions/ADR-033-alerting.md), [alerting.md](alerting.md). |
| **Alertmanager routing to real channels** (email/Slack/PagerDuty) | Deferred (rules delivered PR 6) | Rules evaluate and show on Prometheus's own `/alerts`; wiring external notifiers is an operator step needing channel secrets, not an architecture claim. |

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

## 9. Sprint 8 delivery plan (PRs 2–7)

This PR (**PR 1**) is architecture/documentation only. The remaining PRs implement
the design against these acceptance criteria (ratified in ADR-028; all Kubernetes
work stays static/dry-run in CI per ADR-012 until the runtime-evidence PR).

| PR | Scope | Acceptance criteria |
|----|-------|---------------------|
| **PR 2 — Metrics core** ✅ *(manifests; not deployed)* | Prometheus + kube-state-metrics + node-exporter in a hardened `monitoring` namespace; cAdvisor scrape; least-privilege scrape RBAC | **Delivered:** [`k8s/monitoring/`](../k8s/monitoring/) renders Layer 1 signals **and** the Layer 2 batch-Job signals (via KSM); Job `ttlSecondsAfterFinished` set to honour the [queryability contract](#the-queryability-contract-a-design-requirement-for-the-runtime-prs); extended `k8s/validate.py` monitoring pass green over `k8s/monitoring`; `kustomize build` + `kubeconform` (CI) green. Minimal hand-written Kustomize (no Helm), ephemeral `emptyDir` TSDB, read-only RBAC, one documented node-exporter Pod Security exception — [ADR-029](decisions/ADR-029-monitoring-foundation.md), [Monitoring Operations](monitoring-operations.md). **No deploy claim** (no live cluster; runtime proof is PR 6). |
| **PR 3 — Pipeline operational metrics** ✅ *(manifests + instrumentation; not deployed)* | Per-stage duration + success/failure pushed by the pipeline to a scoped **Pushgateway**; 5th Prometheus scrape job | **Delivered:** `src/pipeline_metrics.py` (best-effort, bounded-cardinality, per-run reset) wired into `stage_runner` + the `fetch-dataset` init container; [`k8s/monitoring/base/pushgateway.yaml`](../k8s/monitoring/base/pushgateway.yaml) (hardened, internal-only) + `honor_labels` scrape; `PUSHGATEWAY_URL` in the base ConfigMap; unit tests for emission/timing/failure/reset; extended `k8s/validate.py`. Operational-vs-MLflow boundary preserved. [ADR-030](decisions/ADR-030-pipeline-operational-metrics.md), [Monitoring Operations](monitoring-operations.md). **No deploy claim** (runtime proof is PR 6). |
| **PR 4 — MLflow & PostgreSQL depth** ✅ *(manifests; not deployed)* | blackbox-exporter (`/health`) + postgres-exporter with a dedicated **read-only** monitoring role (out-of-band Secret) | **Delivered:** [`blackbox-exporter.yaml`](../k8s/monitoring/base/blackbox-exporter.yaml) (Layer 3 MLflow `/health` — a stable, load-free probe) + [`postgres-exporter.yaml`](../k8s/base/mlflow/postgres-exporter.yaml) (Layer 4 `pg_up`/connections/size via a `pg_monitor`-only role, credential in `mlops` only) + a scoped **kubelet** volume-stats scrape (Layer 4 **PVC-fill**); three new scrape jobs (→ eight); extended `k8s/validate.py` (password from Secret, no credential in the DSN). Run-level replica/readiness/restart/CPU/memory signals already collectable from PR 2 (KSM + cAdvisor). Secret hygiene green. [ADR-031](decisions/ADR-031-mlflow-postgres-monitoring.md), [Monitoring Operations](monitoring-operations.md). **No deploy claim** (runtime proof is PR 6). |
| **PR 5 — Dashboards** ✅ *(manifests + dashboard JSON; not deployed)* | Grafana (internal-only) with three purpose-built, version-controlled dashboards over the four-layer signal set | **Delivered:** [`k8s/monitoring/base/grafana/`](../k8s/monitoring/base/grafana/) — a hardened, internal-only Grafana (non-root, drop ALL, read-only root FS, no API token, ClusterIP) with a **file-provisioned** Prometheus datasource and dashboard provider, and three hand-authored dashboards — **EKS / Platform Health**, **MLOps Pipeline Operations**, **MLflow Platform Health** — each panel mapped to a [§ 3](#3-signal-catalogue--per-layer) operational question. Model quality stays in MLflow (explicit help panel). Stable PromQL, bounded windows; no secrets / account IDs on screen; admin password out-of-band. Extended `k8s/validate.py` + `kubeconform` + JSON parse-validation green. [ADR-032](decisions/ADR-032-grafana-dashboards.md), [Monitoring Operations](monitoring-operations.md). **No deploy claim** (runtime proof — panels populating — is the runtime-evidence PR). |
| **PR 6 — Alerting** ✅ *(rules + unit tests; not deployed / not fired live)* | Prometheus alert rules encoding **exactly** the [§ 6](#6-operational-objectives-slo-style-not-production-slos) objectives + the § 3 catalogue | **Delivered:** eight high-signal alerts in [`k8s/monitoring/base/prometheus/alerts.yml`](../k8s/monitoring/base/prometheus/alerts.yml) — `PipelineJobFailed` (terminal Failed condition, not "not Running" — batch-correct), `PipelineJobOOMKilled`, `MLflowDown`, `MLflowMemoryHigh`, `PostgresDown`, `PostgresPVCAlmostFull`, `PostgresMemoryHigh`, `KubePodCrashLooping`. Each carries severity + human summary/description + a `runbook_url`; thresholds traced to measured limits / § 6 (no invented numbers). `promtool check rules` + `promtool test rules` (Pending→Firing→Resolved, incl. a batch-semantics negative) wired into CI; `k8s/validate.py` M11 pins the exact set (no arbitrary alerts). No Alertmanager routing (deferred). [ADR-033](decisions/ADR-033-alerting.md), [Alerting](alerting.md). **No deploy claim** (live firing is PR 7). |
| **PR 7 — Runtime evidence & operations** | Provision → run pipeline → prove signals + dashboards populate → tear down; observability operations runbook | The [runtime-evidence expectations](#runtime-evidence-what-later-sprint-8-prs-must-prove) are all met and recorded in a redacted proof doc matching the Sprint 6/7 conventions; environment destroyed & verified clean (ADR-020) |

> **Resequencing note.** ADR-028 originally pencilled **PR 3 = Grafana dashboards**.
> The sprint reordered the work: **PR 3 delivered the pipeline's operational metrics**
> (the per-stage Pushgateway signal ADR-028 § 3 had deferred — now justified in
> [ADR-030](decisions/ADR-030-pipeline-operational-metrics.md)), and **Grafana
> dashboards landed as PR 5** ([ADR-032](decisions/ADR-032-grafana-dashboards.md)),
> after the Layer 3/4 depth (PR 4). The dashboards therefore render the *full* signal
> set (per-stage series in addition to KSM/node-exporter/cAdvisor/blackbox/pg-exporter),
> so the reorder is additive, not a scope cut. Alerting and runtime evidence follow as
> PR 6 and PR 7.

---

## Runtime evidence: what later Sprint 8 PRs must prove

The runtime-evidence PR (PR 7) is the **proof gate**. It must demonstrate, on a
live cluster and recorded with the [Sprint 7 evidence](proof/sprint-07-runtime-evidence.md)
conventions (redacted, honest about failures, torn down and verified clean):

1. **All targets Up.** Prometheus `up == 1` for node-exporter, kube-state-metrics,
   cAdvisor, **pushgateway**, blackbox-exporter, and postgres-exporter.
2. **The ephemeral Job is observable *after* it exits.** After the pipeline pod is
   `Succeeded` **and gone**, the dashboard still shows
   `kube_job_status_succeeded == 1`, the run's **start/completion timestamps**, and
   the **computed run duration** — proving the [queryability contract](#the-queryability-contract-a-design-requirement-for-the-runtime-prs)
   holds (the finished Job outlived a scrape).
2b. **Per-stage metrics are present and correct (PR 3, [ADR-030](decisions/ADR-030-pipeline-operational-metrics.md)).**
   After a run, `mlops_pipeline_stage_duration_seconds{stage}` exists for all five
   stages with `mlops_pipeline_stage_success == 1`; on an induced mid-pipeline
   failure only the stages that ran are present, with the failing stage's
   `success == 0` and later stages **absent** (proving the per-run reset works, not
   leftover stale series).
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
