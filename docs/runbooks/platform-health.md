# Runbook 1 — Platform health (first-response triage)

> **Part of the [Operational Runbooks](README.md).** Related: every other runbook in
> this directory. Evidence: [live-EKS §2 Results summary](../proof/sprint-08-live-eks-evidence.md#2-results-summary)
> (the green baseline + all four alerts firing under injected failures).

## Purpose

The **entry point** for "is the platform healthy?" — a fast, read-only triage that
confirms the four layers (platform → pipeline → MLflow → PostgreSQL) are up, and routes
you to the specific runbook when one is not. Run this after a deploy, at the start of an
incident, and as the last step of any recovery.

## Symptoms

- A dashboard tile is red, or you were paged by one of the eight alerts.
- A pipeline run did not Complete, or panels show unexpected "No data".
- General "something is wrong" with no single obvious cause yet.

## Detection

The health of the whole platform is answered by **eight scrape targets UP** and **zero
alerts firing**. The live baseline was **8/8 targets UP, `pg_up=1`, 8 alert rules
loaded, nothing firing** ([live-EKS §2](../proof/sprint-08-live-eks-evidence.md#2-results-summary)).

```bash
kubectl -n monitoring port-forward svc/prometheus 9090:9090 &

# Is everything being scraped? (value 1 == up; expect all 8 jobs = 1)
curl -s 'http://localhost:9090/api/v1/query?query=up' \
  | jq '.data.result[] | {job:.metric.job, up:.value[1]}'

# What is firing right now? (an empty list is SUCCESS, not a gap)
curl -s --data-urlencode 'query=ALERTS{alertstate="firing"}' \
  http://localhost:9090/api/v1/query | jq '.data.result[].metric.alertname'
```

The eight expected scrape jobs: `prometheus`, `kube-state-metrics`, `node-exporter`,
`kubernetes-cadvisor`, `pushgateway`, `kubelet`, `blackbox-mlflow-health`,
`postgres-exporter`.

## Initial checks

Read-only, top-down across the four layers:

```bash
# Layer 1 — nodes Ready? (expect every node Ready)
kubectl get nodes -o wide
curl -s --data-urlencode 'query=kube_node_status_condition{condition="Ready",status="true"}' \
  http://localhost:9090/api/v1/query | jq '.data.result[].value[1]'

# Layer 1 — any pod not Running / restarting across the project namespaces?
kubectl -n mlops get pods -o wide
kubectl -n monitoring get pods -o wide

# Layer 2 — did the last pipeline Job succeed? (survives pod exit — queryability contract)
kubectl -n mlops get jobs
curl -s --data-urlencode 'query=kube_job_status_succeeded{job_name="mlops-pipeline"}' \
  http://localhost:9090/api/v1/query | jq '.data.result[].value[1]'

# Layer 3 — is MLflow serving? (1 == /health returned 2xx)
curl -s --data-urlencode 'query=probe_success{job="blackbox-mlflow-health"}' \
  http://localhost:9090/api/v1/query | jq '.data.result[].value[1]'

# Layer 4 — is PostgreSQL up? (1 == a client connected and queried)
curl -s --data-urlencode 'query=pg_up' \
  http://localhost:9090/api/v1/query | jq '.data.result[].value[1]'
```

The visual equivalent is the **EKS / Platform Health** dashboard
(`kubectl -n monitoring port-forward svc/grafana 3000:3000` →
<http://localhost:3000/d/mlops-eks-platform-health>).

## Diagnosis

Map each failing signal to its runbook — do **not** try to fix everything from here:

| Failing signal | Alert | Go to |
|---|---|---|
| `up{job="…"} == 0` for a target | *(target-down not alerted — see limitations)* | [Monitoring Operations §5](../monitoring-operations.md#5-troubleshooting) |
| `kube_job_failed{…,condition="true"} == 1` | `PipelineJobFailed` | [Pipeline failure](pipeline-failure.md) |
| `…stage_success{stage="fetch_dataset"}==0` | `PipelineJobFailed` | [Dataset retrieval](dataset-retrieval-failure.md) / [integrity](dataset-integrity-failure.md) |
| `terminated_reason{reason="OOMKilled"}==1` | `PipelineJobOOMKilled` | [OOMKilled](oomkilled.md) |
| `probe_success{…}==0` | `MLflowDown` | [MLflow unavailable](mlflow-unavailable.md) |
| `pg_up == 0` | `PostgresDown` | [PostgreSQL failure](postgresql-failure.md) |
| PVC / memory gauges high | `PostgresPVCAlmostFull`, `*MemoryHigh` | [PostgreSQL](postgresql-failure.md) / [OOMKilled](oomkilled.md) |
| Pod in `CrashLoopBackOff` | `KubePodCrashLooping` | [Crash / restart](crash-restart.md) |

**Correlation is the diagnosis.** Multiple alerts firing together usually means one
root cause with downstream symptoms — e.g. `MLflowDown` **+** `PipelineJobFailed` is
"the pipeline is blocked because MLflow is down"; `PostgresDown` typically drags
`MLflowDown` with it (MLflow depends on the DB). Fix the **upstream** layer first
(PostgreSQL → MLflow → pipeline).

## Likely causes

1. **Nothing is wrong** — an empty `/alerts` and 8/8 targets UP is the correct healthy
   state; "No data" on a cold cluster is expected before the first run.
2. **A single layer down** — one of the layer runbooks above applies.
3. **Monitoring stack itself unhealthy** — a scrape target DOWN (RBAC, a missing
   Secret, an exporter not deployed). The most common live gotcha: `postgres-exporter`
   needs the out-of-band `mlflow-postgres-exporter-credentials` Secret + `mlflow_exporter`
   DB role, without which it reports `pg_up 0` ([live-EKS Finding 2 / deploy-runbook
   gap](../proof/sprint-08-live-eks-evidence.md#3-findings--4-real-defects-the-live-run-surfaced-all-fixed)).
4. **Fresh deploy not settled** — pods still pulling/starting; re-check in a minute.

## Remediation

Triage-only — the fix lives in the layer-specific runbook. The two safe actions here:

```bash
# Re-read the scrape config + alert rules without deleting the pod:
curl -s -X POST http://localhost:9090/-/reload

# A target DOWN? confirm the pod and its Service endpoints:
kubectl -n monitoring get pods
kubectl -n monitoring get endpoints
```

Then follow the matching runbook. If the **monitoring stack** is the problem, its own
day-2 procedures are in [Monitoring Operations](../monitoring-operations.md).

## Recovery verification

The platform is healthy again only when **all** of these hold — this is the checklist
every other runbook's recovery hands back to:

```bash
# 1. All eight scrape targets UP.
curl -s 'http://localhost:9090/api/v1/query?query=up' \
  | jq '[.data.result[].value[1]] | map(tonumber) | add as $sum | {targets_up:$sum}'   # expect 8

# 2. Nothing firing.
curl -s --data-urlencode 'query=ALERTS{alertstate="firing"}' \
  http://localhost:9090/api/v1/query | jq '.data.result | length'                        # expect 0

# 3. The last pipeline run Completed (exit 0), all five stages succeeded.
kubectl -n mlops get job mlops-pipeline    # STATUS Complete, COMPLETIONS 1/1
curl -s --data-urlencode 'query=min by (job) (mlops_pipeline_stage_success)' \
  http://localhost:9090/api/v1/query | jq '.data.result[].value[1]'                      # expect 1

# 4. Dependencies healthy.
curl -s --data-urlencode 'query=probe_success{job="blackbox-mlflow-health"}' \
  http://localhost:9090/api/v1/query | jq '.data.result[].value[1]'                      # expect 1
curl -s --data-urlencode 'query=pg_up' \
  http://localhost:9090/api/v1/query | jq '.data.result[].value[1]'                      # expect 1
```

Visually: the three dashboards green — **EKS/Platform Health** (nodes Ready, 0 failed
Jobs, 0 CrashLoopBackOff), **MLOps Pipeline Operations** (last run success, all stages),
**MLflow Platform Health** (availability gauge green, PostgreSQL panels green).

## Escalation / known limitations

- **No Alertmanager / no routing.** Alerts are visible only on Prometheus's `/alerts`;
  wiring a notifier is a deliberate deferral ([alerting.md § Known limitations](../alerting.md#known-limitations)).
- **Absent ≠ Down.** Availability alerts use `== 0`, so a *missing* target (no series)
  does not fire — a `up == 0` / `absent()` "target-gone" alert is a documented future
  addition. Always eyeball `up` on `/targets`, not just `/alerts`.
- **Ephemeral TSDB.** Prometheus stores to an `emptyDir`; metrics vanish on a Prometheus
  restart. Capture evidence while the stack is live.
- **Small cluster.** 1–2 nodes; `NodeNotReady`/`NodeUnderPressure` are deliberately not
  alerted (limited HA signal, [ADR-017](../decisions/ADR-017-eks-platform.md)).
- **Design of record:** [ADR-028 (observability architecture)](../decisions/ADR-028-observability-architecture.md),
  [ADR-033 (alerting)](../decisions/ADR-033-alerting.md).
