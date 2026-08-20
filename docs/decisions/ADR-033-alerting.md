# ADR-033: Actionable alerts for critical platform failures (Sprint 8, PR 6)

- **Status:** Accepted (design; alert rules + unit tests + CI added, no components deployed / no live firing in this PR)
- **Date:** 2026-08-20
- **Deciders:** Asad Hanif
- **Related:**
  [ADR-028 (Observability architecture — the four-layer model; § 6 objectives; Alertmanager deferred)](ADR-028-observability-architecture.md),
  [ADR-029 (Monitoring foundation — Prometheus + KSM + node-exporter)](ADR-029-monitoring-foundation.md),
  [ADR-030 (Pipeline operational metrics via Pushgateway)](ADR-030-pipeline-operational-metrics.md),
  [ADR-031 (MLflow & PostgreSQL monitoring — blackbox + postgres-exporter)](ADR-031-mlflow-postgres-monitoring.md),
  [ADR-032 (Grafana dashboards)](ADR-032-grafana-dashboards.md),
  [ADR-011 (Kubernetes resource lifecycle — limits, backoffLimit, OOM)](ADR-011-kubernetes-resource-lifecycle.md),
  [ADR-026 (In-cluster MLflow platform — measured limits)](ADR-026-in-cluster-mlflow-platform.md),
  [ADR-012 (Kubernetes manifest validation)](ADR-012-kubernetes-manifest-validation.md),
  [`docs/alerting.md`](../alerting.md),
  [`docs/observability.md`](../observability.md),
  [`k8s/monitoring/base/prometheus/`](../../k8s/monitoring/base/prometheus/)

> **Scope.** This ADR ratifies the **alerting** layer: a small, high-signal set of
> Prometheus alert rules encoding the [ADR-028 § 6](ADR-028-observability-architecture.md)
> operational objectives and the § 3 signal catalogue, plus their promtool unit tests
> and CI wiring. It **adds rules + tests + validation + docs**; **nothing is deployed**
> and **no alert has fired on a live cluster** beyond the offline promtool unit test —
> the live failure-injection campaign is the runtime-evidence PR. It does **not** add
> Alertmanager or notifier routing (deliberately deferred), and does **not** re-decide
> the architecture.

## Context

After PRs 2–5 the platform emits a complete four-layer metric set and renders it on
three dashboards, but the dashboards are *pull* — they need a human looking. The
[observability gap](observability.md#1-why-observability-why-now) is only fully closed
when a defined failure **pushes** a signal at an operator without one watching. The
[§ 6 operational objectives](observability.md#6-operational-objectives-slo-style-not-production-slos)
name exactly what to watch for; this PR encodes them as alert rules.

The brief sets firm constraints: **only high-signal alerts** that map to a real
operator action; cover at least (1) pipeline Job failure, (2) MLflow unavailable,
(3) persistent restart/crash, (4) high memory / OOM risk, (5) a critical component
unavailable; **be careful with batch semantics** (do **not** alert because a completed
Job is no longer Running); every alert must carry a summary, description, severity,
sensible `for`, and a runbook reference; **design low-noise thresholds** and **do not
invent thresholds without documenting rationale**; validate with proper tooling and add
CI; and perform **one safe preliminary test** (Pending → Firing → Resolved) without the
full failure-injection campaign (that is PRs 7 / 10–12).

## Decision

### 1. Eight high-signal alerts — the documented set, and no others

Grouped by layer in [`alerts.yml`](../../k8s/monitoring/base/prometheus/alerts.yml);
full runbook + per-alert rationale in [`docs/alerting.md`](../alerting.md):

| Alert | Severity | Brief category | Basis |
|---|---|---|---|
| `PipelineJobFailed` | critical | 1 — pipeline failure | § 6 pipeline-success |
| `PipelineJobOOMKilled` | critical | 4 — OOM risk | § 3 Layer 2 |
| `MLflowDown` | warning | 2 / 5 — component unavailable | § 6 MLflow availability |
| `MLflowMemoryHigh` | warning | 4 — high memory | § 6 memory headroom |
| `PostgresDown` | critical | 5 — component unavailable | § 3 Layer 4 |
| `PostgresPVCAlmostFull` | critical | (storage) | § 6 storage headroom |
| `PostgresMemoryHigh` | warning | 4 — high memory | § 6 memory headroom |
| `KubePodCrashLooping` | warning | 3 — persistent restart/crash | § 3 Layer 1 |

**No arbitrary alerts.** The set is *exactly* these eight — enforced by `k8s/validate.py`
M11 (the alert-name set is pinned, like the dashboard count). The § 3 catalogue names
more candidates; each deferred one is listed with a reason in
[`docs/alerting.md` § Known limitations](../alerting.md#known-limitations) rather than
silently dropped.

### 2. Batch semantics — never alert on "not Running"

A finished batch Job is *not* Running, and that is normal, so no rule keys on it:

- **`PipelineJobFailed`** keys on the Job's terminal **Failed condition**
  (`kube_job_failed{condition="true"}`), set by the controller **only after
  `backoffLimit=2` is exhausted** ([ADR-011](ADR-011-kubernetes-resource-lifecycle.md))
  — not the transient `kube_job_status_failed` pod counter, which is `> 0` during a
  retried-but-eventually-successful run. A promtool **negative test** proves a
  retried-then-succeeded Job does not fire it.
- **`PipelineJobOOMKilled`** keys on the retained finished pod's last terminated reason.

Both read persistent kube-state-metrics API-object series, so they stay evaluable after
the pod exits (the [queryability contract](observability.md#the-queryability-contract-a-design-requirement-for-the-runtime-prs);
`ttlSecondsAfterFinished=3600`).

### 3. Documented, low-noise thresholds

Every threshold traces to a **measured** limit or a § 6 objective — none invented
(full table in [`docs/alerting.md` § 2](../alerting.md#2-threshold-rationale)): memory
**90 %** of the measured 2Gi / 512Mi limits ([ADR-026](ADR-026-in-cluster-mlflow-platform.md)),
PVC **85 %** of the fixed 1Gi. `for`-durations sit above the relevant transient — 5m for
availability (> a rolling restart + a scrape), 15m for memory headroom (rides out a GC
spike), 10m for the slow PVC-fill trend, 2m as a mere debounce on the already-terminal
pipeline conditions. `KubePodCrashLooping` reuses the canonical kube-prometheus
`CrashLoopBackOff`-for-15m pattern so "how many restarts = instability?" is answered by
the kubelet's own determination, not a hand-picked count. The memory/PVC expressions are
**identical** to the Grafana dashboard gauges, so alert and panel can never disagree.

### 4. Full operator metadata + a runbook per alert

Every rule carries `summary`, `description`, `severity` + `layer` labels, a `for`, and a
`runbook_url` annotation pointing at that alert's section in
[`docs/alerting.md`](../alerting.md) — enforced by `k8s/validate.py` M11. The two-tier
**severity taxonomy** (critical = cannot-work-now / imminent data-loss; warning =
degraded / headroom) is documented, including the deliberate call that `MLflowDown` is a
warning (no data loss — history is in Postgres) not critical.

### 5. Rules wired as a first-class, promtool-tested file

The rule file lives at [`prometheus/alerts.yml`](../../k8s/monitoring/base/prometheus/alerts.yml)
— a raw, individually-validatable artifact (same pattern as the dashboard JSON), packaged
into the `prometheus-alerts` ConfigMap by a kustomize `configMapGenerator`, mounted
read-only at `/etc/prometheus/rules`, and loaded via `rule_files` in
[`prometheus-config.yaml`](../../k8s/monitoring/base/prometheus-config.yaml). Its
[`alerts_test.yml`](../../k8s/monitoring/base/prometheus/alerts_test.yml) unit tests run
in CI. No Alertmanager: firing alerts are exposed on Prometheus's `/alerts` +
`/api/v1/alerts`, but routing to a real notifier is deferred (see below).

## Alternatives considered

| Option | Why not |
|---|---|
| **Import a community alert bundle** (kube-prometheus's full rule set) | Hundreds of rules for signals this workload does not emit or care about — the exact noise the brief forbids. We encode the eight the § 6 objectives justify, borrowing only the one well-proven *pattern* (`CrashLoopBackOff`) where it fits. |
| **Alert on `kube_job_status_failed > 0`** | Fires on a run that failed a pod attempt but then **succeeded** within `backoffLimit` — a false alarm. The terminal Failed *condition* is the batch-correct signal. |
| **Alert on the pipeline pod not Running** | A completed batch Job is *supposed* to stop running — this would fire on every successful run. Explicitly rejected by the brief. |
| **Add Alertmanager + notifier routing now** | Routing to email/Slack/PagerDuty is an operator wiring step, not an architecture claim ([ADR-028 § 6](ADR-028-observability-architecture.md)); it needs real channel secrets and adds a component to harden. Deferred — rules are useful and testable on Prometheus's own `/alerts` without it. |
| **`up == 0` / `absent()` "target gone" alerts** | Useful, but they fire before the stack is deployed (no series yet), adding pre-deploy noise. Deferred to the runtime PRs where the baseline is live; `== 0` (absent ≠ down) is the low-noise choice now. |
| **Tighter/looser thresholds or shorter `for`** | Shorter `for` pages on transient restarts/GC spikes; looser thresholds miss the headroom window. The chosen values are justified against measured baselines in `docs/alerting.md § 2`. |

## Consequences

**Positive**
- Each § 6 objective and the five brief categories are now enforced by a rule that
  pushes at an operator, with a summary, threshold rationale, and a runbook.
- Batch semantics are handled correctly — no false alarm on a normal completed or
  retried-then-succeeded Job (proven by a negative unit test).
- The rules are **unit-tested deterministically in CI** (promtool, pinned to the running
  Prometheus version): every alert's Pending → Firing (→ Resolved) transition, labels and
  rendered annotations are asserted before merge.
- `k8s/validate.py` M11 guarantees the file stays wired end to end and the alert set
  cannot silently grow.

**Negative / limitations**
- **Nothing is deployed and no alert has fired live** — the promtool test proves the
  rule *logic*; that a real OOM / MLflow-down / PVC-fill actually fires it on a cluster
  is the [runtime-evidence PR](observability.md#runtime-evidence-what-later-sprint-8-prs-must-prove).
- **No routing.** Without Alertmanager, an operator must look at Prometheus `/alerts`;
  paging a channel is deferred.
- **Absent ≠ Down.** A missing target (no series) does not fire an availability alert; a
  `up`-based "target gone" alert is future work.
- **Tight MLflow memory baseline** (~85 % at idle) leaves `MLflowMemoryHigh` limited lead
  time before the kernel OOM backstop — documented, with the OOMKill as the real safety
  net.

## Validation

- `promtool check rules k8s/monitoring/base/prometheus/alerts.yml` — 8 rules, structurally valid.
- `promtool test rules …/alerts_test.yml` — **SUCCESS**; every alert fires at its `for`,
  `MLflowDown` resolves on recovery, and the batch-semantics negative does not fire.
- Both run in CI (`.github/workflows/ci.yml`, the `prometheus-rules` job), promtool pinned
  to Prometheus 2.53.2 (the Deployment's version).
- `python k8s/validate.py` — M11 asserts the ConfigMap packaging, the `/etc/prometheus/rules`
  mount, the `rule_files` wiring, the required per-rule metadata, and the exact eight-alert
  set (185/185 checks pass).
- `kustomize build` + `kubeconform -strict` render and schema-validate the added ConfigMap.
