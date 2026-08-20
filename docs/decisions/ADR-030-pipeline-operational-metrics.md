# ADR-030: Pipeline Operational Metrics via Pushgateway (Sprint 8, PR 3)

- **Status:** Accepted (design; manifests + instrumentation added, no components deployed in this PR)
- **Date:** 2026-08-20
- **Deciders:** Asad Hanif
- **Related:**
  [ADR-028 (Observability architecture — batch metrics via KSM; Pushgateway *deferred*)](ADR-028-observability-architecture.md),
  [ADR-029 (Monitoring foundation — Prometheus + KSM + node-exporter)](ADR-029-monitoring-foundation.md),
  [ADR-026 (In-cluster MLflow platform)](ADR-026-in-cluster-mlflow-platform.md),
  [ADR-020 (Ephemeral cloud lifecycle & cost control)](ADR-020-cloud-lifecycle-cost-control.md),
  [ADR-011 (Resources/lifecycle — measured limits, no probes)](ADR-011-kubernetes-resource-lifecycle.md),
  [ADR-009 (Workload model — a Job, not a Deployment)](ADR-009-kubernetes-workload-model.md),
  [`docs/observability.md`](../observability.md),
  [`docs/monitoring-operations.md`](../monitoring-operations.md),
  [`src/pipeline_metrics.py`](../../src/pipeline_metrics.py),
  [`k8s/monitoring/base/pushgateway.yaml`](../../k8s/monitoring/base/pushgateway.yaml)

> **Scope.** This ADR ratifies adding **operational** metrics to the ephemeral
> MLOps pipeline Job — per-stage **duration** and **success/failure** — and the
> **Pushgateway** that makes them survive the Job's exit. It **supersedes the
> "Pushgateway deferred" position of [ADR-028 § 3](ADR-028-observability-architecture.md#3-batch-workload-metric-strategy--kube-state-metrics-is-the-answer-not-pushgateway)**
> for this one, scoped use, and explains why. It is **implementation + design**:
> the instrumentation (`src/pipeline_metrics.py`, wired into `stage_runner` and the
> `fetch-dataset` init container) and the Pushgateway manifest ship here and pass
> static validation, but **nothing is deployed** — runtime proof is Sprint 8 PR 6.

## Context

[ADR-028](ADR-028-observability-architecture.md) established the observability
architecture and made kube-state-metrics (KSM) the **primary** batch-Job metric
source: KSM reflects the *persistent* `Job`/`Pod` API object, so `did it succeed`,
`when`, `how long`, `OOMKilled?` stay queryable after the pod exits — with **zero
application change** ([ADR-029](ADR-029-monitoring-foundation.md) deployed that
foundation in PR 2). ADR-028 deliberately **deferred** the Pushgateway, naming the
**one** operational signal KSM cannot give — **per-stage** granularity ("which
stage took the longest?") — and required that, *if* it were ever adopted, it be
justified as its own PR against the sticky-metric and ownership-overlap trade-offs.

This PR is that justification. Two things changed the calculus:

1. **The sprint resequenced the work.** ADR-028 § 9's delivery plan pencilled "PR 3
   = Grafana dashboards"; the actual PR 3 brief is **"expose operational metrics for
   MLOps pipeline execution"** — precisely the deferred per-stage work, pulled
   forward (Grafana moves later). So the deferred decision is now *on the table by
   design*, not smuggled in.
2. **The gap is real and small.** KSM gives run-level outcome/duration but cannot
   say *which* of `fetch_dataset → preprocess → split → train → evaluate` was slow
   or *which* one failed. For a five-stage pipeline that is the difference between
   "the Job failed" and "`train` failed after 4 s" — genuinely operational, and not
   available anywhere else (MLflow only logs a run when `train`/`evaluate` reach
   their tracking call; a stage that fails earlier logs nothing).

The **hard constraint** ADR-028 identified still holds and dictates the mechanism:
Prometheus **pulls**, but the pipeline is a `batch/v1` Job whose pod exits seconds
after finishing, and each `dvc repro` stage is its **own short-lived process**
(`python src/<stage>.py`). There is nothing to pull-scrape. The Prometheus-
sanctioned way to get metrics *out of* a batch job is the **Pushgateway**.

## Decision

### 1. Adopt a Pushgateway, scoped to per-stage operational metrics only

Deploy a single **Prometheus Pushgateway** in the `monitoring` namespace
([`k8s/monitoring/base/pushgateway.yaml`](../../k8s/monitoring/base/pushgateway.yaml)).
Each pipeline stage **pushes** its operational metrics before its process exits;
Prometheus scrapes the always-up gateway (a 5th scrape job in
[`prometheus-config.yaml`](../../k8s/monitoring/base/prometheus-config.yaml), with
`honor_labels: true`). KSM **remains primary** for run-level signals — this **adds**
per-stage depth, it does not replace the ADR-028 design.

### 2. The metrics — and the operational-vs-MLflow ownership boundary

Two gauges, one label (`stage`), pushed by
[`src/pipeline_metrics.py`](../../src/pipeline_metrics.py):

| Metric | Type | Labels | Why it exists |
|---|---|---|---|
| `mlops_pipeline_stage_duration_seconds` | gauge | `stage` | Per-stage wall-clock — the KSM gap ("which stage took the longest?"). Includes `stage="fetch_dataset"`, i.e. the brief's `dataset_fetch_duration_seconds`. |
| `mlops_pipeline_stage_success` | gauge | `stage` | Per-stage outcome (1/0) — **failure attribution**: KSM knows the *Job* failed; this says *which stage*. |

`stage` is drawn from a **fixed set** of five values. The ownership split from
[ADR-028 § 3](ADR-028-observability-architecture.md#3-batch-workload-metric-strategy--kube-state-metrics-is-the-answer-not-pushgateway)
is **preserved and enforced by construction**: this module emits **only**
operational signals. Model accuracy, hyper-parameters, and per-run artifacts stay
in **MLflow** ([`src/tracking.py`](../../src/tracking.py)). Prometheus is **not** a
second experiment database, and this module has no code path that could log one —
`accuracy` never appears here.

### 3. Gauges, not counters — why the brief's `*_total` names are last-run gauges

The brief lists *potential* metrics including `pipeline_runs_total{status}` and
`pipeline_failures_total{stage}`. We deliberately **do not** emit those as
Prometheus counters, because under this architecture they would be a **lie**:

- A counter must monotonically accumulate, but **each stage is a fresh process** with
  no memory of prior runs, and **the Pushgateway replaces (never increments)** a
  pushed value. Pushing `pipeline_runs_total 1` every run yields a series that is
  *always 1* — `rate()`/`increase()` over it are meaningless.
- Making a counter accumulate would require a **per-run grouping key** (a run id in
  the Pushgateway path), which is exactly the **unbounded cardinality** the brief
  forbids and the Pushgateway's top documented foot-gun.

So run outcome and count are modelled as **last-run gauges** (the documented
Prometheus *batch-job* pattern) and derived in PromQL:

- **Did the last run succeed?** `min by (job) (mlops_pipeline_stage_success)` (== 1).
- **Which stage failed?** `mlops_pipeline_stage_success == 0`.
- **Slowest stage?** `topk(1, mlops_pipeline_stage_duration_seconds)`.
- **Approx. run count / cadence?** `changes(push_time_seconds{job="mlops_pipeline"}[7d])`
  (the Pushgateway's built-in per-group `push_time_seconds`).
- **Whole-run duration?** authoritative from **KSM**
  (`kube_job_status_completion_time − kube_job_status_start_time`, ADR-028); a
  compute-only approximation is `sum by (job)(mlops_pipeline_stage_duration_seconds)`.
  We do **not** re-emit a `pipeline_duration_seconds` — KSM already owns it, and
  synthesising it would need a "which stage is last" hack the design avoids.

### 4. Bounded label cardinality

The only label is `stage`, from the fixed tuple `PIPELINE_STAGES` =
(`fetch_dataset`, `preprocess`, `split`, `train`, `evaluate`). **No** run UUID,
model path, dataset filename, or timestamp label — those grow the series set
without bound. An out-of-set stage name is **refused**, not emitted, so a future
typo cannot silently balloon cardinality.

### 5. Pushgateway lifecycle — no stale metrics

This is the trade-off ADR-028 warned about ("pushed metrics are sticky; the gateway
accumulates stale series"), and the design neutralises it **at the producer**:

- **One group per stage.** Each stage pushes under grouping key
  `job=mlops_pipeline` + `stage=<name>` using **PUT** (`push_to_gateway`), which
  replaces *that stage's* whole group. Stages never clobber each other, and a
  re-run of a stage cleanly overwrites its own prior series.
- **Reset at run start.** The `fetch-dataset` init container — the first thing that
  runs — calls `reset_pipeline_metrics`, which **DELETEs every stage group** before
  the run. A shorter or failed run therefore cannot leave a *previous* run's
  later-stage series behind as stale data: a stage that did not run this time is
  simply **absent** (which reads correctly as "the pipeline never got there").
- **In-memory gateway.** The Pushgateway runs **without** `--persistence.file`, so
  it holds no state across its own restart. Combined with the per-run reset, the
  gateway never becomes a growing graveyard of stale series. Persistence via a PVC
  is the documented option if surviving a gateway restart ever matters (it does not
  on this ephemeral cluster, ADR-020).

### 6. Best-effort, and disabled unless configured

- **Never fatal.** Metric emission is observability, not a pipeline output. Every
  push/delete is wrapped so **any** failure is logged at WARNING and **swallowed** —
  a monitoring hiccup can never fail a real run. The catch is deliberately **broad**
  (`except Exception`, always logged — never a silent `pass`): `prometheus_client`'s
  push runs over `urllib`/`http.client`, which raises not only `OSError`
  (connect/timeout/DNS/TLS/4xx-5xx) and `ImportError` (client absent) but also
  `http.client.HTTPException` subclasses like `BadStatusLine` (e.g. the gateway pod
  rolled mid-response) that are **not** `OSError`. A narrow catch would let exactly
  that hiccup fail a stage whose real work already succeeded — precisely the outcome
  this guarantee forbids (a PR-3 review finding; regression-tested).
- **Opt-in by config.** Emission is a **no-op** unless `PUSHGATEWAY_URL` is set.
  In-cluster the base ConfigMap injects it; local `dvc repro`, the CI fixture run,
  and unit tests leave it unset and do **zero** network I/O. `prometheus_client` is
  imported **lazily** at push time, so importing the pipeline never requires the
  package — the same boundary discipline `fetch_dataset` uses for boto3 and `train`
  for MLflow.

### 7. Smallest useful instrumentation point

All timing lives in **one** place — `pipeline_metrics.time_stage`, a context
manager — wired into `stage_runner.run_stage` (which every `dvc repro` stage already
funnels through) and the `fetch-dataset` init container's `main`. No stage body was
touched; the ML computation stays MLflow-free and metrics-free. `KeyboardInterrupt`
/ `SystemExit` are `BaseException`, not `Exception`, so they propagate without
recording a spurious failure — matching `run_stage`'s existing contract.

### 8. Hardening, exposure, RBAC

The Pushgateway inherits the fleet baseline (ADR-010/029): non-root uid 65534, drop
`ALL`, no privilege escalation, seccomp `RuntimeDefault`, **read-only root FS**
(in-memory storage writes nothing), **no API token** (it never calls the K8s API),
and a **ClusterIP** Service (internal-only — it accepts unauthenticated pushes, so
it must never be exposed publicly, same rule as Prometheus/MLflow). The extended
`k8s/validate.py` monitoring pass covers it, including that the scrape sets
`honor_labels`.

## Alternatives Considered

- **KSM only (the ADR-028 primary), no Pushgateway.** Rejected *for this PR's goal*:
  KSM cannot give per-stage duration or per-stage failure attribution — the exact
  signal the brief asks for. KSM stays primary for everything it *can* answer.
- **A true `*_total` counter.** Rejected: impossible to accumulate honestly across
  fresh per-stage processes with a replace-only gateway, and only "fixable" with an
  unbounded per-run grouping key. Last-run gauges + PromQL (`changes()`) give the
  same operational answers without the lie or the cardinality (§ 3).
- **Per-run grouping key (run id / timestamp in the Pushgateway path).** Rejected:
  unbounded cardinality and a permanently growing gateway — the documented
  anti-pattern (§ 4/§ 5).
- **A long-running scrape sidecar / keep-alive on the Job.** Rejected outright (as in
  ADR-028): it would stop the Job from ever `Completing`, breaking the
  run-to-completion model (ADR-009) *and* the KSM success signal.
- **Log-derived metrics (parse the structured logs into series).** Reasonable and
  app-change-free in principle, but needs a log pipeline (Loki/agent) this project
  has deferred (ADR-028 § 6); a direct push is simpler and lands the signal now. The
  structured logs remain the root-cause layer.
- **Emit ML metrics (accuracy) to Prometheus too.** Rejected: it duplicates MLflow
  and violates the ownership boundary (§ 2). Model quality is run-indexed experiment
  data, not an operational time series.

## Consequences

**Positive**

- **Per-stage operational visibility** an ephemeral batch Job otherwise cannot
  offer: duration per stage and which stage failed, queryable after the pod is gone.
- **Honest data model** — gauges that mean what they say, bounded cardinality, and
  a producer-side reset that keeps the gateway free of stale series (the ADR-028
  hazard, mitigated).
- **Zero blast radius on the pipeline** — best-effort, opt-in, lazily imported;
  local/CI runs and the ML computation are entirely unaffected.
- **Clean ownership** preserved: Prometheus/KSM = operational health; MLflow = ML
  semantics; logs = root cause. No duplicated model-quality metrics.

**Negative / trade-offs**

- **A new moving part** (the Pushgateway) and a new runtime dependency
  (`prometheus-client`) on a deliberately lean stack.
- **Single-replica, in-memory gateway** — not HA, and pushed series are lost if the
  gateway pod restarts (accepted: the pipeline re-pushes; evidence is captured while
  live, ADR-020). Per-run reset means only the *latest* run's stages are live at any
  time — by design.
- **Unauthenticated push surface — a monitoring-*integrity* risk, not just DoS.**
  Any workload that can reach the ClusterIP (from **any** namespace — the cluster has
  no NetworkPolicy anywhere yet) can push **or delete**. Because the scrape uses
  `honor_labels`, a forged push (`stage=train, success=1`) is stored
  indistinguishably from a real one, and a `DELETE` can erase a genuine failure
  group before the next scrape — i.e. an attacker with a foothold pod could make the
  dashboards **lie** about pipeline health or **erase failure evidence**, and could
  flood distinct grouping keys to OOM the 64Mi gateway. Note the bounded-cardinality
  guard (§ 4) is **client-side** in `pipeline_metrics.py` — it does not protect the
  gateway's raw HTTP endpoint, and the per-run reset (§ 5) only clears the app's own
  five known groups, so externally-injected groups persist until the pod restarts.
  Exploitation requires an already-running in-cluster pod (the Service is correctly
  ClusterIP, not externally reachable). Acceptable on a single-operator internal
  validation cluster; a **NetworkPolicy** restricting who may POST is the right added
  control on a shared cluster and is **deferred** with the rest of the stack's
  NetworkPolicy work (ADR-029 § Consequences).
- **Per-stage failure attribution misses hard kills.** `mlops_pipeline_stage_success
  == 0` is set only for **Python-level** failures caught by `time_stage`'s
  `except Exception`. A `SIGKILL` — notably the ADR-011 **OOMKill** — terminates the
  process before the `except` runs, so that stage is **absent** rather than
  `success=0` (and a naïve `min()` over surviving stages could read 1). This is why
  KSM remains primary: run-level `kube_job_status_failed` /
  `kube_pod_container_status_last_terminated_reason="OOMKilled"` (ADR-028) catch the
  kill that the per-stage push cannot. The two sources are complementary by design.
- **No per-stage CPU/memory.** All five stages share one Job pod, so cAdvisor
  (PR 2) gives pod-level resource use but cannot decompose it per stage; only
  duration/success are per-stage. Per-stage resource accounting would need a
  different execution model and is not pursued.
- **Deviation from ADR-028's delivery plan** — Pushgateway lands in PR 3 and Grafana
  moves later. Recorded here explicitly rather than left implicit.

## What This Decision Does **Not** Imply

- **Nothing is deployed or proven by this PR.** The instrumentation and the
  Pushgateway manifest ship and pass **static** validation; a live push→scrape→query
  cycle is **runtime evidence for PR 6** (the project rule: *structurally valid ≠
  runtime-complete*).
- **Not a replacement for KSM.** KSM remains the primary, app-change-free source for
  run-level Job signals; this is additive per-stage depth.
- **Not a general-purpose application-monitoring bus.** The Pushgateway is scoped to
  this pipeline's batch operational metrics; it is not a place for arbitrary app
  metrics, and it is not treated as long-term storage.
- **Not production monitoring / not an SLO.** Same validation-environment posture as
  ADR-028 § 7 — no error budgets, no long-term run history.
