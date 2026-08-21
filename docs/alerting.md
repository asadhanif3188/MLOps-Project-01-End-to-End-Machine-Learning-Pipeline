# Alerting — actionable alerts for critical platform failures

> **Status:** 🚧 **Design + unit-tested, not yet fired on a live cluster.**
> The alert rules exist, are structurally valid (`promtool check rules`) and are
> unit-tested (`promtool test rules`) — each alert is proven to transition
> Pending → Firing → Resolved against synthetic series in CI. They have **not** yet
> been exercised by the live failure-injection campaign; that is the runtime-evidence
> PR (**PR 7**, [observability.md § runtime evidence](observability.md#runtime-evidence-what-later-sprint-8-prs-must-prove)).
> Decision of record: [ADR-033](decisions/ADR-033-alerting.md).

This is the alerting counterpart to the [observability architecture](observability.md):
it says **what pages an operator, why, at what threshold, and what to do about it**.
Every rule maps to a real operator action and to a documented objective in
[observability.md § 6](observability.md#6-operational-objectives-slo-style-not-production-slos)
(the operational objectives) or the [§ 3 signal catalogue](observability.md#3-signal-catalogue--per-layer).
There are **no arbitrary alerts** — the deliberately-deferred candidates are listed
in [§ Known limitations](#known-limitations) rather than silently dropped.

- **Rules:** [`k8s/monitoring/base/prometheus/alerts.yml`](../k8s/monitoring/base/prometheus/alerts.yml)
- **Unit tests:** [`k8s/monitoring/base/prometheus/alerts_test.yml`](../k8s/monitoring/base/prometheus/alerts_test.yml)
- **Wiring:** `rule_files` in [`prometheus-config.yaml`](../k8s/monitoring/base/prometheus-config.yaml) → the `prometheus-alerts` ConfigMap mounted at `/etc/prometheus/rules`.

---

## 1. Design principles

**High-signal only.** Eight alerts. Each corresponds to a failure an operator would
act on for *this* workload — a failed pipeline run, a down component, a filling disk,
memory near a limit, or a crash loop. We do **not** alert on everything an exporter
can emit (the same discipline the rest of the observability stack applies).

**Batch semantics — the trap this PR is careful about.** A finished batch Job is
**not "running"**, and that is *normal*. So **nothing here alerts on "the pipeline pod
is not Running"**:

- `PipelineJobFailed` keys on the Job's terminal **Failed condition**
  (`kube_job_failed{condition="true"}`), which the Job controller sets **only after
  `backoffLimit` is exhausted** — never on the transient `kube_job_status_failed` pod
  counter (which is `> 0` during a run that retries and then succeeds).
- `PipelineJobOOMKilled` keys on the **retained finished pod's last terminated
  reason**.

Both read persistent kube-state-metrics API-object series, so they remain evaluable
**after the pod exits** (the [queryability contract](observability.md#the-queryability-contract-a-design-requirement-for-the-runtime-prs);
the Job's `ttlSecondsAfterFinished=3600` keeps the object alive for many scrapes).

**Absent ≠ Down.** The availability rules use `max(<metric>) == 0`, so a target that is
simply *absent* (before deploy, or an exporter not yet scraped) yields no series and
does **not** fire — only a real `0` does. This keeps pre-deploy noise at zero;
target-up alerting via `up`/`absent()` is a documented future addition
([§ Known limitations](#known-limitations)).

**Severity taxonomy.**

| Severity | Meaning | Alerts |
|---|---|---|
| **critical** | The platform cannot do its job now, or data-loss is imminent | `PipelineJobFailed`, `PipelineJobOOMKilled`, `PostgresDown`, `PostgresPVCAlmostFull` |
| **warning** | Degraded / early-warning / headroom — act soon, not now | `MLflowDown`, `MLflowMemoryHigh`, `PostgresMemoryHigh`, `KubePodCrashLooping` |

> **Why `MLflowDown` is a warning, not critical:** MLflow being down blocks *new*
> pipeline runs but loses **no data** (experiment history lives in PostgreSQL). On this
> single-operator ephemeral cluster that is "fix it soon", not "wake someone". In a
> production setup with an availability SLO it would be critical — a deliberate,
> documented choice, not an oversight.

---

## 2. Threshold rationale

Thresholds are **not invented** — each traces to a measured limit or a § 6 objective.

| Alert | Threshold | `for` | Rationale |
|---|---|---|---|
| `PipelineJobFailed` | Failed condition == true | 2m | Terminal condition (set only after `backoffLimit=2` exhausted, [ADR-011](decisions/ADR-011-kubernetes-resource-lifecycle.md)); `for` only debounces a scrape flap since the state won't recover on its own. |
| `PipelineJobOOMKilled` | last reason == OOMKilled | 2m | The pipeline's memory-safety signal is a **kill event**, not sustained headroom — a < 1 min Job cannot sustain "> 90% for N min" ([§ 6 note](observability.md#6-operational-objectives-slo-style-not-production-slos)). |
| `MLflowDown` | `probe_success == 0` | 5m | 5m > a rolling restart (~30–60s) + one 30s scrape, short enough to catch a real outage well within a pipeline-run window. **"How long down before alerting?" → 5m.** |
| `MLflowMemoryHigh` | > 90% of 2Gi | 15m | § 6 memory-headroom objective vs the **measured** 2Gi limit (baseline ~1.7GiB / 85%, [ADR-026](decisions/ADR-026-in-cluster-mlflow-platform.md)). 90% is a genuine "approaching ceiling" band above idle; 15m rides out a checkpoint/GC spike. **"What memory % is meaningful?" → 90% of the measured limit.** |
| `PostgresDown` | `pg_up == 0` | 5m | The exporter stays up when the DB is down, so `0` is a true "DB unreachable". 5m debounces a restart. |
| `PostgresPVCAlmostFull` | > 85% of 1Gi | 10m | § 6 storage-headroom objective vs the **fixed 1Gi** PVC (no autogrow, [ADR-026](decisions/ADR-026-in-cluster-mlflow-platform.md)). A filling volume is a slow trend, so 10m still leaves lead time before it is full. |
| `PostgresMemoryHigh` | > 90% of 512Mi | 15m | § 6 memory-headroom objective vs the **measured** 512Mi limit ([ADR-026](decisions/ADR-026-in-cluster-mlflow-platform.md)). |
| `KubePodCrashLooping` | in `CrashLoopBackOff` | 15m | **"How many restarts = persistent instability?" →** the kubelet's own CrashLoopBackOff determination (it backs a container off only after repeated crash-then-restart), held for 15m. This is the canonical kube-prometheus pattern; it filters a one-off restart. |

The memory and PVC expressions are **byte-for-byte the same** as the corresponding
Grafana dashboard gauges ([mlflow-platform-health.json](../k8s/monitoring/base/grafana/dashboards/mlflow-platform-health.json)),
so an alert and its dashboard panel can never disagree.

---

## 3. Runbook — per alert

Each subsection is the target of that alert's `runbook_url` annotation.

### PipelineJobFailed

- **Summary:** MLOps pipeline Job failed (last run did not Complete).
- **Severity:** critical · **Fires after:** 2m
- **Means:** the `mlops-pipeline` Job reached its terminal Failed condition — every
  attempt (up to `backoffLimit`) failed.
- **Do:**
  1. `kubectl -n mlops logs job/mlops-pipeline --all-containers` — read the failing
     stage's structured logs (the root-cause layer).
  2. Check the per-stage `mlops_pipeline_stage_success` series (Pushgateway,
     [ADR-030](decisions/ADR-030-pipeline-operational-metrics.md)) — which stage was
     `0`? Was `fetch_dataset` the failure (dataset retrieval)?
  3. Correlate with `PipelineJobOOMKilled` (memory) and `MLflowDown` / `PostgresDown`
     (dependencies) — a failed run is often a *symptom* of one of those.
  4. Fix the cause, delete the failed Job, re-submit.
- **Dataset retrieval failures (`fetch_dataset`).** When step 2 points at the
  `fetch_dataset` init stage (`stage_success{stage="fetch_dataset"}=0`, every later
  stage absent), the `fetch-dataset` init-container logs distinguish the two modes:
  - `Failed to download s3://…` → the object is **unavailable** (missing key, denied
    access, unreachable endpoint, or no credentials). Verify `DATASET_S3_URI` points at
    an existing object, the Pod Identity role grants read, and the endpoint is
    reachable. Transient? the Job's `backoffLimit` already retried it.
  - `Dataset integrity check failed: expected …, got …` → the object was **retrieved**
    but its SHA-256 does not match the pinned `DATASET_SHA256`
    ([ADR-027](decisions/ADR-027-s3-dataset-runtime-retrieval.md)). This is
    **deterministic** — retrying cannot fix it. Either the object in S3 was
    swapped/corrupted (restore the correct bytes) or the pin is stale (update
    `DATASET_SHA256` deliberately, with the new dataset's provenance). Do **not** add
    retries for a checksum mismatch — a fail-fast integrity gate is the intended
    behaviour. Runtime evidence for both modes:
    [docs/proof/sprint-08-dataset-failure-tests-evidence.md](proof/sprint-08-dataset-failure-tests-evidence.md).

### PipelineJobOOMKilled

- **Summary:** MLOps pipeline pod was OOMKilled.
- **Severity:** critical · **Fires after:** 2m
- **Means:** a pipeline pod hit its 512Mi memory limit and the kernel killed it
  ([ADR-011](decisions/ADR-011-kubernetes-resource-lifecycle.md)).
- **Do:** inspect which stage was running; reduce its working set (batch size, data
  loaded at once) **or** raise the limit **deliberately with a measured
  justification** — do not silently grow it. This is the pipeline's designed
  memory-safety backstop, so a single event on an intentionally-tight limit may be
  expected (e.g. the ADR-011 64Mi induced-OOM test).

### MLflowDown

- **Summary:** MLflow tracking server is unavailable (`/health` down 5m).
- **Severity:** warning · **Fires after:** 5m
- **Means:** the blackbox probe of MLflow `/health` has returned `0` for 5m. New runs
  cannot log to MLflow. **Experiment data is not lost** (it lives in PostgreSQL).
- **Do:** `kubectl -n mlops get deploy/mlflow` and `kubectl -n mlops logs deploy/mlflow`.
  Common causes: pod OOM/restart, image pull, or the DB being down (check
  `PostgresDown` first — MLflow depends on it).
- **Effect on pipeline runs (verified by Sprint 8 PR 11 —
  [proof](proof/sprint-08-mlflow-failure-tests-evidence.md)):** a run that **starts**
  while MLflow is down blocks at the `wait-for-mlflow` init gate (`MLflow not ready
  after …`) and fails **before any computation** — no wasted work. A run **already in
  progress** fails in the `train` stage (`src/tracking.py` raises `TrackingError`),
  which *does* discard the completed compute. Either way the Job's terminal failure
  raises `PipelineJobFailed`, so **`MLflowDown` + `PipelineJobFailed` firing together**
  is the signature of "the pipeline is blocked because MLflow is down" — restore MLflow
  (scale `deploy/mlflow` back up), then re-drive the run. (A bounded mid-run retry is a
  recorded PR 13 candidate; runs are **not** retried forever.)

### MLflowMemoryHigh

- **Summary:** MLflow memory > 90% of its 2Gi limit for 15m.
- **Severity:** warning · **Fires after:** 15m
- **Means:** working-set memory has held above 90% of the 2Gi limit. Baseline is
  ~1.7GiB / 85% ([ADR-026](decisions/ADR-026-in-cluster-mlflow-platform.md)), so this
  is genuinely elevated. Risk: an OOMKill of the tracking server.
- **Do:** check load (concurrent clients, large artifact proxying). Raise the limit
  only with a measured justification. See [§ Known limitations](#known-limitations) on
  the tight baseline → limited lead time.

### PostgresDown

- **Summary:** PostgreSQL metadata database is unreachable (`pg_up 0` for 5m).
- **Severity:** critical · **Fires after:** 5m
- **Means:** `postgres-exporter` cannot connect to the DB. This is the durable store
  the whole platform depends on — MLflow and the pipeline cannot persist runs.
- **Do:** `kubectl -n mlops get statefulset/mlflow-postgres` and its pod logs. Check
  the PVC is bound and not full (`PostgresPVCAlmostFull`) and the pod is not OOMing.

### PostgresPVCAlmostFull

- **Summary:** PostgreSQL PVC over 85% full.
- **Severity:** critical · **Fires after:** 10m
- **Means:** the fixed **1Gi** Postgres PVC (no autogrow,
  [ADR-026](decisions/ADR-026-in-cluster-mlflow-platform.md)) has held above 85% full.
  A full volume stops the DB accepting writes (outage / data-loss risk).
- **Do:** check `pg_database_size_bytes` growth; reclaim space (vacuum, prune old
  experiments) **or** grow the volume deliberately. On the ephemeral cloud run this may
  be induced/simulated — see the runtime-evidence PR.

### PostgresMemoryHigh

- **Summary:** PostgreSQL memory > 90% of its 512Mi limit for 15m.
- **Severity:** warning · **Fires after:** 15m
- **Means:** working-set memory has held above 90% of the 512Mi limit
  ([ADR-026](decisions/ADR-026-in-cluster-mlflow-platform.md)). Risk: an OOMKill of the
  single-writer metadata DB.
- **Do:** check load / connection count; raise the limit only with a measured
  justification.

### KubePodCrashLooping

- **Summary:** Pod `<namespace>/<pod>` is crash-looping.
- **Severity:** warning · **Fires after:** 15m
- **Means:** a container in the `mlops` or `monitoring` namespace has been in
  `CrashLoopBackOff` for 15m — persistent instability, not a one-off restart. Covers
  the pipeline, MLflow and Postgres in one rule.
- **Do:** `kubectl -n <ns> describe pod <pod>` and `kubectl -n <ns> logs <pod>
  --previous` — read the last terminated reason and the crash logs.

---

## 4. Runbook mapping

| Alert | Layer | Category (PR 6 brief) | § 6 / § 3 basis | Runbook |
|---|---|---|---|---|
| `PipelineJobFailed` | Pipeline | 1 — Pipeline Job failure | § 6 pipeline-success | [↑](#pipelinejobfailed) |
| `PipelineJobOOMKilled` | Pipeline | 4 — OOM risk | § 3 Layer 2 | [↑](#pipelinejoboomkilled) |
| `MLflowDown` | MLflow | 2 — MLflow unavailable / 5 — component unavailable | § 6 MLflow availability | [↑](#mlflowdown) |
| `MLflowMemoryHigh` | MLflow | 4 — high memory | § 6 memory headroom | [↑](#mlflowmemoryhigh) |
| `PostgresDown` | Postgres | 5 — critical component unavailable | § 3 Layer 4 | [↑](#postgresdown) |
| `PostgresPVCAlmostFull` | Postgres | (storage) | § 6 storage headroom | [↑](#postgrespvcalmostfull) |
| `PostgresMemoryHigh` | Postgres | 4 — high memory | § 6 memory headroom | [↑](#postgresmemoryhigh) |
| `KubePodCrashLooping` | Platform | 3 — persistent restart/crash | § 3 Layer 1 | [↑](#kubepodcrashlooping) |

---

## 5. Validation

- **`promtool check rules`** — the rule file parses and every rule is well-formed.
- **`promtool test rules`** — the unit-test suite
  ([`alerts_test.yml`](../k8s/monitoring/base/prometheus/alerts_test.yml)) drives
  synthetic series through the engine and asserts each alert's Pending → Firing
  transition (and MLflowDown's Resolved-on-recovery), with the exact expected labels
  **and rendered summary/description/runbook_url**. It includes a **batch-semantics
  negative**: a retried-but-successful Job (pod counter > 0, condition false) must
  **not** fire `PipelineJobFailed`.
- **CI** — both run on every PR in the `prometheus-rules` job
  ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)), promtool pinned to the
  same version the Prometheus Deployment runs.
- **`k8s/validate.py` M11** — asserts the rule file is packaged into the
  `prometheus-alerts` ConfigMap, that Prometheus mounts it at `/etc/prometheus/rules`,
  that `rule_files` is wired, that every rule carries severity + summary + description +
  a `docs/alerting.md` runbook_url, and — the *no arbitrary alerts* contract — that the
  alert set is **exactly** the eight documented here.

## 6. Preliminary runtime evidence

`promtool test rules` output (the safe preliminary test the brief asks for — proves
Pending → Firing → Resolved without a cluster):

```
Unit Testing:  alerts_test.yml
  SUCCESS
```

All eight alerts fire at their configured `for`-duration with the expected labels and
annotations; `MLflowDown` clears when the probe recovers; the batch-semantics negative
does not fire. The **live** failure-injection campaign (OOM the pipeline, scale MLflow
to 0, fill the PVC) is [PR 7](observability.md#runtime-evidence-what-later-sprint-8-prs-must-prove).

## Known limitations

- **No Alertmanager / no routing.** Rules evaluate and firing alerts show on
  Prometheus's own `/alerts` and `/api/v1/alerts`, but they are **not** routed to a
  real notifier (email/Slack/PagerDuty). Wiring a notifier is an operator step, not an
  architecture claim ([ADR-028 § 6](decisions/ADR-028-observability-architecture.md)),
  deliberately deferred.
- **Absent ≠ Down.** An availability alert (`== 0`) will not fire if the target is
  *missing* (no series) rather than reporting `0`. A `up == 0` / `absent()` "target
  gone" alert is a reasonable future addition; omitted here to keep pre-deploy noise at
  zero.
- **Tight MLflow memory baseline.** MLflow idles at ~85% of its 2Gi limit, so
  `MLflowMemoryHigh` (90%) has limited lead time before the kernel OOM backstop. The
  real safety net is the OOMKill; the alert is an early-warning, not a guarantee.
- **Deliberately deferred alerts (no arbitrary alerts).** The § 3 catalogue names more
  candidates than are encoded here; each is deferred with reason:
  `NodeNotReady` / `NodeUnderPressure` (a 1–2 node validation cluster, [ADR-017](decisions/ADR-017-eks-platform.md) — limited HA signal),
  `PodPending` / `PVCUnbound` (scheduling — surfaced on the dashboard, rarely a page
  here), `MLflowNoReplicas` / `PostgresNotReady` (overlap `MLflowDown` / `PostgresDown`
  which are the sharper signals), `PostgresConnectionsHigh` (a single-operator pipeline
  is far from `max_connections`), `PipelineJobSlow` / `PipelineJobDeadlineExceeded`
  (the batch pipeline has no latency contract, [ADR-011](decisions/ADR-011-kubernetes-resource-lifecycle.md)).
  They can be added later, each with its own rationale — not pre-emptively, to keep the
  set high-signal.

---

## Related documentation

- [Observability & Operations Architecture](observability.md) (§ 3 signals, § 6 objectives)
- [Monitoring Operations](monitoring-operations.md) (deploy / port-forward / view)
- [ADR-033 — Alerting](decisions/ADR-033-alerting.md) · [ADR-028 — Observability Architecture](decisions/ADR-028-observability-architecture.md)
- [ADR-011 (resources/lifecycle)](decisions/ADR-011-kubernetes-resource-lifecycle.md) · [ADR-026 (MLflow platform)](decisions/ADR-026-in-cluster-mlflow-platform.md) · [ADR-030 (pipeline metrics)](decisions/ADR-030-pipeline-operational-metrics.md) · [ADR-031 (MLflow/Postgres monitoring)](decisions/ADR-031-mlflow-postgres-monitoring.md)
