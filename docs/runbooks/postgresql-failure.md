# Runbook 6 — PostgreSQL failure

> **Part of the [Operational Runbooks](README.md).** Alerts:
> [`PostgresDown`](../alerting.md#postgresdown) (critical, 5m),
> [`PostgresPVCAlmostFull`](../alerting.md#postgrespvcalmostfull) (critical, 10m),
> [`PostgresMemoryHigh`](../alerting.md#postgresmemoryhigh) (warning, 15m). Related:
> [MLflow unavailable](mlflow-unavailable.md) (depends on this DB). Evidence:
> [PR 11 persistence](../proof/sprint-08-mlflow-failure-tests-evidence.md)
> (`pg_up=1` throughout the MLflow outage), [live-EKS Finding 2](../proof/sprint-08-live-eks-evidence.md#3-findings--4-real-defects-the-live-run-surfaced-all-fixed)
> (postgres-exporter arg defect, fixed).

## Purpose

Diagnose and recover the **PostgreSQL metadata database** (`mlflow-postgres` StatefulSet)
— the durable store the whole platform depends on. Covers three failure shapes:
unreachable (`PostgresDown`), filling volume (`PostgresPVCAlmostFull`), and memory
pressure (`PostgresMemoryHigh`). **This is the most data-sensitive component** — treat
every remediation with the destructive-action warnings below.

## Symptoms

| Alert | Symptom |
|---|---|
| `PostgresDown` (critical) | `pg_up == 0` for 5m; MLflow errors / `MLflowDown` follows; runs cannot persist. |
| `PostgresPVCAlmostFull` (critical) | PVC > 85% full for 10m; DB near write-stop (the 1Gi PVC has **no autogrow**). |
| `PostgresMemoryHigh` (warning) | working set > 90% of the 512Mi limit for 15m; OOMKill risk for the single-writer DB. |

## Detection

```bash
# Availability (1 == a client connected and queried):
curl -s --data-urlencode 'query=pg_up' \
  http://localhost:9090/api/v1/query | jq '.data.result[].value[1]'

# PVC fill fraction (0..1) — same expr the dashboard gauge uses:
curl -s --data-urlencode \
  'query=kubelet_volume_stats_used_bytes / kubelet_volume_stats_capacity_bytes' \
  http://localhost:9090/api/v1/query | jq '.data.result[].value[1]'

# Memory vs the 512Mi limit (%):
curl -s --data-urlencode 'query=100 * sum(container_memory_working_set_bytes{namespace="mlops",pod=~"mlflow-postgres.*",container!="",container!="POD"}) / sum(kube_pod_container_resource_limits{namespace="mlops",pod=~"mlflow-postgres.*",resource="memory"})' \
  http://localhost:9090/api/v1/query | jq '.data.result[].value[1]'
```

> **`pg_up` comes from `postgres-exporter`, which runs in the `mlops` namespace** beside
> the DB (not `monitoring`), so its dedicated read-only DB credential never enters the
> monitoring namespace. The exporter stays up when the DB is down, so `pg_up == 0` is a
> *true* "DB unreachable", not "exporter gone".

## Initial checks

```bash
# 1. StatefulSet + pod state.
kubectl -n mlops get statefulset/mlflow-postgres      # READY 1/1?
kubectl -n mlops get pods -l app.kubernetes.io/name=mlflow-postgres -o wide
kubectl -n mlops describe statefulset/mlflow-postgres
kubectl -n mlops logs statefulset/mlflow-postgres --tail=100

# 2. Is the PVC bound, and how full? (from the volumeClaimTemplate: data-mlflow-postgres-0)
kubectl -n mlops get pvc | grep mlflow-postgres
curl -s --data-urlencode 'query=pg_database_size_bytes{datname="mlflow"}' \
  http://localhost:9090/api/v1/query | jq '.data.result[].value[1]'

# 3. Is the exporter itself healthy? (a common false-'down')
kubectl -n mlops get deploy postgres-exporter
kubectl -n mlops logs deploy/postgres-exporter --tail=50
```

## Diagnosis

| Observation | Root cause | Route |
|---|---|---|
| `pg_up 0` **but** exporter target UP, DB pod Running | **auth/connect** — missing/wrong `mlflow_exporter` role or `mlflow-postgres-exporter-credentials` Secret | fix the role/Secret (below) |
| `pg_up 0`, DB pod **not** Running / CrashLoop | DB pod down — OOM, PVC full, or scheduling | check `describe`/logs; PVC & memory rows below |
| exporter pod **CrashLoopBackOff**, `error: unexpected false, try --help` | **exporter arg defect** (live Finding 2) | already fixed to `--no-auto-discover-databases` in the committed manifest |
| PVC > 85% | 1Gi volume filling (no autogrow) | reclaim / grow (below) |
| memory > 90% | connection/load pressure on the 512Mi limit | investigate load; raise limit deliberately |

## Likely causes

1. **Exporter credential/role missing** — the most common live gotcha: `pg_up 0` with
   the target UP means auth failed. The DB itself is usually fine; the *monitoring* is
   what is broken. Create the `mlflow_exporter` `pg_monitor` role + the
   `mlflow-postgres-exporter-credentials` Secret
   ([template](../../k8s/base/mlflow/postgres-exporter-secret.example.yaml)).
2. **DB pod down** — OOMKilled (memory), unschedulable, or crashing on a full/corrupt
   volume.
3. **PVC filling** — a slow trend; the fixed 1Gi PVC stops accepting writes when full.
4. **Memory pressure** — sustained load near the 512Mi limit risks an OOMKill of the
   single-writer DB.

## Remediation

**Exporter can't authenticate** (DB is fine, `pg_up 0`):

```bash
# Recreate the dedicated read-only role + Secret out-of-band (never committed):
#   see k8s/base/mlflow/postgres-exporter-secret.example.yaml for the exact SQL + kubectl create secret.
kubectl -n mlops rollout restart deploy/postgres-exporter
```

**DB pod down** — inspect first, then let the StatefulSet reschedule; the PVC persists
across pod restarts:

```bash
kubectl -n mlops describe pod -l app.kubernetes.io/name=mlflow-postgres    # why did it stop?
# A stuck pod can be safely recreated — the StatefulSet reattaches the SAME PVC:
kubectl -n mlops delete pod -l app.kubernetes.io/name=mlflow-postgres      # ⚠️ pod only — the PVC (data) is retained
```

**PVC almost full** — reclaim space (prune old experiments) or grow the volume
deliberately:

```bash
# Inspect growth first:
curl -s --data-urlencode 'query=pg_database_size_bytes{datname="mlflow"}' \
  http://localhost:9090/api/v1/query | jq '.data.result[].value[1]'
# Growing the PVC requires an allowVolumeExpansion StorageClass; do it deliberately, not reflexively.
```

> **⚠️ Data-loss warnings.** **Never** `kubectl delete pvc` for `mlflow-postgres`,
> **never** `kubectl delete statefulset --cascade`, and **never** `terraform destroy`
> while you need the data — any of these destroys experiment history. `terraform destroy`
> is the *intentional end-of-session teardown* only ([Cloud Operations §5](../cloud-operations.md#5-safe-teardown)),
> not an incident tool. Deleting the *pod* is safe (PVC retained); deleting the *PVC* is not.

**Memory high** — check connection count / load; raise the 512Mi limit only with a
measured justification ([ADR-026](../decisions/ADR-026-in-cluster-mlflow-platform.md)).

## Recovery verification

```bash
# 1. DB reachable again.
curl -s --data-urlencode 'query=pg_up' \
  http://localhost:9090/api/v1/query | jq '.data.result[].value[1]'      # expect 1
kubectl -n mlops get statefulset/mlflow-postgres        # READY 1/1

# 2. Data intact — the metadata is still there (previous MLflow runs survive).
kubectl -n mlops port-forward svc/mlflow 5000:5000 &    # MLflow UI: prior runs + registered model present

# 3. Headroom back to normal.
curl -s --data-urlencode 'query=kubelet_volume_stats_used_bytes / kubelet_volume_stats_capacity_bytes' \
  http://localhost:9090/api/v1/query | jq '.data.result[].value[1]'      # well below 0.85

# 4. Alerts cleared.
curl -s --data-urlencode 'query=ALERTS{alertstate="firing",layer="postgres"}' \
  http://localhost:9090/api/v1/query | jq '.data.result[].metric.alertname'   # empty

# 5. MLflow recovered on top of the DB, and a fresh pipeline run Completes.
curl -s --data-urlencode 'query=probe_success{job="blackbox-mlflow-health"}' \
  http://localhost:9090/api/v1/query | jq '.data.result[].value[1]'      # expect 1
kubectl -n mlops delete job mlops-pipeline           # ⚠️ discards the failed Job object
kubectl apply -k k8s/overlays/<aws|local>            # on EKS: scripts/render-cloud-manifests.sh --apply
kubectl -n mlops wait --for=condition=complete job/mlops-pipeline --timeout=600s
```

## Escalation / known limitations

- **`PostgresDown` is critical** — it is the durable store for the whole platform; both
  MLflow and the pipeline stall without it.
- **Deploy-runbook gap (documented):** the monitoring stack needs the out-of-band
  `mlflow-postgres-exporter-credentials` Secret + `mlflow_exporter` role that the
  [cloud-operations §3.8 runbook](../cloud-operations.md) does not yet list — tracked in
  [monitoring-operations.md](../monitoring-operations.md) and [live-EKS §3](../proof/sprint-08-live-eks-evidence.md#3-findings--4-real-defects-the-live-run-surfaced-all-fixed).
- **No PVC autogrow, no backup/restore, no RTO/RPO** — this is an ephemeral validation
  platform; DR is out of scope ([Cloud Operations §7](../cloud-operations.md#7-limitations)).
- **Design of record:** [ADR-026 (MLflow platform: server + PostgreSQL + S3)](../decisions/ADR-026-in-cluster-mlflow-platform.md),
  [ADR-031 (MLflow/Postgres monitoring)](../decisions/ADR-031-mlflow-postgres-monitoring.md).
