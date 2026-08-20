# Monitoring Operations

Day-2 operations for the **metrics + dashboards stack** — Prometheus,
kube-state-metrics, node-exporter, the **Pushgateway**, the **MLflow/PostgreSQL depth
exporters**, and **Grafana** — deployed from [`k8s/monitoring/`](../k8s/monitoring/)
(plus the postgres-exporter in the mlops workload). This is the operator's runbook for
the Sprint 8 PR 2 + PR 3 + PR 4 + PR 5 stack: how to deploy it, reach Prometheus and
Grafana, run a query (across all four layers), troubleshoot a missing target, and tear
it down cleanly.

> **Scope — the metrics core + pipeline + platform depth, not yet runtime-proven.**
> This covers Prometheus + KSM + node-exporter + the cAdvisor scrape (Layer 1
> platform signals and the Layer 2 batch-Job signals via KSM, PR 2), **the
> Pushgateway the pipeline pushes per-stage duration/success to** (PR 3,
> [ADR-030](decisions/ADR-030-pipeline-operational-metrics.md)), **and the Layer 3/4
> depth exporters** — blackbox (MLflow `/health`), postgres-exporter (DB backend
> health), and the kubelet volume-stats scrape (Postgres PVC-fill) — (PR 4,
> [ADR-031](decisions/ADR-031-mlflow-postgres-monitoring.md)), **and Grafana with the
> three purpose-built dashboards** — EKS/Platform Health, MLOps Pipeline Operations,
> MLflow Platform Health — (PR 5,
> [ADR-032](decisions/ADR-032-grafana-dashboards.md)). **Alerts** (PR 6) are not here.
> As of these PRs the stack is **defined and statically validated but not deployed** —
> no live cluster was available. The commands below are the runbook for when it *is*
> deployed; the live four-layer evidence (targets Up, dashboards populating) is the job
> of the runtime-evidence PR
> (PR 7, per [`docs/observability.md`](observability.md#runtime-evidence-what-later-sprint-8-prs-must-prove)).
> Design of record: [ADR-028](decisions/ADR-028-observability-architecture.md),
> [ADR-029](decisions/ADR-029-monitoring-foundation.md),
> [ADR-030](decisions/ADR-030-pipeline-operational-metrics.md),
> [ADR-031](decisions/ADR-031-mlflow-postgres-monitoring.md), and
> [ADR-032](decisions/ADR-032-grafana-dashboards.md).

For the architecture (why these components, the four-layer model, the batch-Job
strategy) see [`docs/observability.md`](observability.md); for the mlops workload's
own runbook see [Kubernetes Operations](kubernetes-operations.md).

---

## 1. What gets deployed

`kustomize build k8s/monitoring/base` renders, all in the **`monitoring`**
namespace unless cluster-scoped:

| Component | Kind | Purpose | Exposure |
|---|---|---|---|
| `prometheus` | Deployment + ClusterIP Service (`:9090`) | Scrape + local TSDB (7d, emptyDir) + PromQL/UI | Internal only |
| `kube-state-metrics` | Deployment + ClusterIP Service (`:8080`) | Kubernetes API-object state (Job/Pod/Node/…) | Internal only |
| `node-exporter` | DaemonSet + headless Service (`:9100`) | Per-node CPU/memory/filesystem | Internal only |
| `pushgateway` | Deployment + ClusterIP Service (`:9091`) | Sink for the ephemeral pipeline's **per-stage** operational metrics (duration + success), pushed before each stage exits (PR 3) | Internal only |
| `blackbox-exporter` | Deployment + ClusterIP Service (`:9115`) | Layer 3 — probes MLflow `/health` for availability (PR 4) | Internal only |
| `grafana` | Deployment + ClusterIP Service (`:3000`) | Dashboards over Prometheus — three provisioned dashboards (PR 5); datasource + dashboards loaded from ConfigMaps at start-up | Internal only |
| `prometheus` / `kube-state-metrics` | ServiceAccount + **read-only** ClusterRole + binding | Least-privilege API access | — |

The kubelet's built-in **cAdvisor** and the kubelet's own **volume stats** (Postgres
PVC-fill) are scraped through the API server proxy (no separate component). The
**postgres-exporter** (Layer 4 DB backend health) runs in the **`mlops`** namespace
beside the database — deployed with the mlops workload (`kubectl apply -k
k8s/overlays/…`), not this stack — so its dedicated read-only DB credential Secret
never enters `monitoring`; Prometheus scrapes it cross-namespace (PR 4,
[ADR-031](decisions/ADR-031-mlflow-postgres-monitoring.md)). Everything is
`ClusterIP`/headless — nothing is exposed outside the cluster; you reach Prometheus
with `kubectl port-forward` (§ 3).

> **postgres-exporter prerequisite (PR 4).** Before it can report, create its
> dedicated read-only role and Secret out-of-band (the exact SQL + `kubectl create
> secret` are in
> [`k8s/base/mlflow/postgres-exporter-secret.example.yaml`](../k8s/base/mlflow/postgres-exporter-secret.example.yaml)).
> Without them the exporter runs but reports `pg_up 0` and logs auth failures.

> **Grafana admin-credential prerequisite (PR 5).** Before deploying Grafana, create
> its admin Secret out-of-band (the exact `kubectl create secret` is in
> [`k8s/monitoring/base/grafana/grafana-admin-secret.example.yaml`](../k8s/monitoring/base/grafana/grafana-admin-secret.example.yaml)).
> Without it the Grafana pod stays `CreateContainerConfigError` (the `secretKeyRef`
> cannot resolve). Dashboards themselves need no login — anonymous Viewer access is
> enabled — but the pod still requires the admin Secret to start.

---

## 2. Deploy

The stack is a **separate** kustomize root from the mlops workload, so it deploys
independently:

```bash
# Local (Docker Desktop / kind / minikube):
kubectl apply -k k8s/monitoring/overlays/local

# EKS:
kubectl apply -k k8s/monitoring/overlays/aws
```

> **Pod Security note.** The `monitoring` namespace is created with
> `pod-security.kubernetes.io/enforce: privileged` — required because node-exporter
> mounts the node's `/proc`, `/sys`, and root filesystem (read-only) to report node
> metrics. This is the single documented exception
> ([ADR-029 § 5](decisions/ADR-029-monitoring-foundation.md)); Prometheus and KSM
> remain restricted-equivalent. Apply the manifest (which carries the namespace and
> its labels) rather than creating the namespace by hand, so the labels are set.

Watch it come up:

```bash
kubectl -n monitoring get pods -w
# Expect: prometheus-* Running (1/1), kube-state-metrics-* Running (1/1),
#         pushgateway-* Running (1/1), blackbox-exporter-* Running (1/1),
#         grafana-* Running (1/1), node-exporter-* Running (1/1) on every node.
```

---

## 3. Check Prometheus

Prometheus is internal-only. Port-forward it to your workstation (the same posture
as the MLflow UI — never a public endpoint):

```bash
kubectl -n monitoring port-forward svc/prometheus 9090:9090
```

Then:

- **UI / health:** open <http://localhost:9090> — or `curl -s localhost:9090/-/healthy`
  and `curl -s localhost:9090/-/ready` (both return `200`/`Prometheus ... is Healthy`).
- **Targets:** <http://localhost:9090/targets> — every scrape job
  (`prometheus`, `kube-state-metrics`, `node-exporter`, `kubernetes-cadvisor`,
  `pushgateway`, `kubelet`, `blackbox-mlflow-health`, `postgres-exporter`) should
  show its targets **UP**. This is the first thing to check after a deploy. (The
  `pushgateway` target is UP as soon as the gateway is running, even before any
  pipeline has pushed — it just has no `mlops_pipeline_*` series yet. The
  `postgres-exporter` target is only UP once that exporter is deployed with the mlops
  workload and its DB role/Secret exist — see § 1.)

---

## 3.1 View the Grafana dashboards

Grafana is internal-only too. Port-forward it and open the browser (anonymous Viewer
access is enabled, so no login is needed to read the dashboards):

```bash
kubectl -n monitoring port-forward svc/grafana 3000:3000
```

Then open <http://localhost:3000> → **Dashboards → MLOps Platform**. Three dashboards
are provisioned automatically (no manual import) from
[`k8s/monitoring/base/grafana/dashboards/`](../k8s/monitoring/base/grafana/dashboards/):

| Dashboard | uid | Answers |
|---|---|---|
| **EKS / Platform Health** | `mlops-eks-platform-health` | Nodes Ready? Workloads healthy? Pods restarting? CPU/memory/disk pressure? Jobs failing? |
| **MLOps Pipeline Operations** | `mlops-pipeline-operations` | Did the last pipeline succeed? How long? Which stage dominates? Did dataset retrieval fail? Recent failures? |
| **MLflow Platform Health** | `mlops-mlflow-platform-health` | Is MLflow available? MLflow pods stable? PostgreSQL up? Either under memory / PVC / connection pressure? |

Deep-link a specific dashboard by uid, e.g.
<http://localhost:3000/d/mlops-pipeline-operations>. Each panel carries a description
(hover the ℹ️ in its header) explaining the query and what "healthy" looks like.

> **Panels show "No data" on a cold cluster.** Per-stage series appear only after a
> pipeline has run (`mlops_pipeline_*` via the Pushgateway); MLflow/Postgres panels
> need those workloads up and, for `pg_*`, the postgres-exporter role/Secret (§ 1).
> This is expected — it is the runtime-evidence PR that proves panels populate.
>
> **Editing a dashboard vs. editing provisioning.** A change to a **dashboard JSON**
> (`grafana/dashboards/*.json`) hot-reloads — the provider re-reads the directory
> every 30 s, so `kubectl apply -k …` is enough. A change to the **datasource** or
> **dashboard-provider** ConfigMap is read by Grafana **only at start-up**, so after
> `apply` you must also restart it:
> `kubectl -n monitoring rollout restart deploy/grafana`.
>
> **Model accuracy is not in Grafana.** The pipeline dashboard says so explicitly:
> accuracy / best params / per-run artifacts live in **MLflow** (the ownership
> boundary, [ADR-030](decisions/ADR-030-pipeline-operational-metrics.md) /
> [ADR-032](decisions/ADR-032-grafana-dashboards.md)). Port-forward `svc/mlflow` in
> the `mlops` namespace for those.

---

## 4. Execute a query

Use the UI (<http://localhost:9090/graph>) or the HTTP API. Instant query via
`curl` (jq optional, for readability):

```bash
# Is Prometheus scraping everything? (one row per target; value 1 == up)
curl -s 'http://localhost:9090/api/v1/query?query=up' | jq '.data.result[] | {job:.metric.job, instance:.metric.instance, up:.value[1]}'
```

Signals worth knowing (from [`docs/observability.md`](observability.md) § 3):

```bash
# Layer 1 — are all nodes Ready?
curl -s --data-urlencode 'query=kube_node_status_condition{condition="Ready",status="true"}' http://localhost:9090/api/v1/query

# Layer 1 — node memory available (bytes), per node:
curl -s --data-urlencode 'query=node_memory_MemAvailable_bytes' http://localhost:9090/api/v1/query

# Layer 2 — did the LAST pipeline Job succeed? (survives the pod exiting — the
# queryability contract, ADR-028 § 3)
curl -s --data-urlencode 'query=kube_job_status_succeeded{job_name="mlops-pipeline"}' http://localhost:9090/api/v1/query

# Layer 2 — run duration (seconds) of the pipeline Job:
curl -s --data-urlencode 'query=kube_job_status_completion_time{job_name="mlops-pipeline"} - kube_job_status_start_time{job_name="mlops-pipeline"}' http://localhost:9090/api/v1/query

# Layer 2 — was a pipeline pod OOMKilled? (a Pod-object series)
curl -s --data-urlencode 'query=kube_pod_container_status_last_terminated_reason{reason="OOMKilled",namespace="mlops"}' http://localhost:9090/api/v1/query

# Layer 1 — per-container memory working set (cAdvisor), mlops namespace:
curl -s --data-urlencode 'query=container_memory_working_set_bytes{namespace="mlops"}' http://localhost:9090/api/v1/query
```

**Pipeline per-stage operational metrics (PR 3, via the Pushgateway).** These are
pushed by the pipeline itself (`src/pipeline_metrics.py`) and answer the per-stage
questions KSM cannot. They appear after the pipeline has run at least once with
`PUSHGATEWAY_URL` set (it is, in-cluster, from the base ConfigMap):

```bash
# Which stage took the longest in the last run?
curl -s --data-urlencode 'query=topk(1, mlops_pipeline_stage_duration_seconds)' http://localhost:9090/api/v1/query

# Per-stage duration, all stages (fetch_dataset, preprocess, split, train, evaluate):
curl -s --data-urlencode 'query=mlops_pipeline_stage_duration_seconds' http://localhost:9090/api/v1/query

# Dataset fetch duration specifically:
curl -s --data-urlencode 'query=mlops_pipeline_stage_duration_seconds{stage="fetch_dataset"}' http://localhost:9090/api/v1/query

# Did the last run succeed end to end? (1 == every stage that ran succeeded)
# NOTE: catches Python-level stage failures; a hard OOMKill leaves the stage ABSENT,
# so pair this with the KSM run-level OOM query above (§ Layer 2, ADR-030).
curl -s --data-urlencode 'query=min by (job) (mlops_pipeline_stage_success)' http://localhost:9090/api/v1/query

# Which stage failed (if any)?
curl -s --data-urlencode 'query=mlops_pipeline_stage_success == 0' http://localhost:9090/api/v1/query

# Approximate run count over the last day. push_time_seconds is per-group (one
# series per stage, each ~= the run count), so count() over one stage gives a single
# number; using all stages would return five identical-ish series.
curl -s --data-urlencode 'query=changes(push_time_seconds{job="mlops_pipeline",stage="preprocess"}[1d])' http://localhost:9090/api/v1/query
```

> **Operational only.** These series describe *execution* (how long, did it
> succeed). Model **accuracy** and hyper-parameters are **not** here — they live in
> MLflow (the ownership boundary, [ADR-030](decisions/ADR-030-pipeline-operational-metrics.md)).
> Whole-run duration is authoritative from KSM (`completion_time − start_time`
> above); the per-stage series is the *decomposition* of it.

**Layer 3 — MLflow availability (PR 4, via blackbox-exporter).** MLflow has no native
`/metrics`; blackbox probes its `/health` and Prometheus records the result. Run-level
signals (replicas, restarts, memory) come from KSM + cAdvisor as in Layers 1–2.

```bash
# Is MLflow serving? (1 == /health returned 2xx)
curl -s --data-urlencode 'query=probe_success{job="blackbox-mlflow-health"}' http://localhost:9090/api/v1/query

# /health latency (seconds) and the HTTP status code it returned:
curl -s --data-urlencode 'query=probe_duration_seconds{job="blackbox-mlflow-health"}' http://localhost:9090/api/v1/query
curl -s --data-urlencode 'query=probe_http_status_code{job="blackbox-mlflow-health"}' http://localhost:9090/api/v1/query

# Is the MLflow Deployment serving its replica? (run-level, KSM — since PR 2)
curl -s --data-urlencode 'query=kube_deployment_status_replicas_available{namespace="mlops",deployment="mlflow"}' http://localhost:9090/api/v1/query
```

**Layer 4 — PostgreSQL backend health + PVC fill (PR 4).** `pg_*` come from
postgres-exporter; the PVC-fill ratio comes from the kubelet volume stats.

```bash
# Is the DB up and accepting connections? (1 == a client connected and queried)
curl -s --data-urlencode 'query=pg_up' http://localhost:9090/api/v1/query

# The highest-value signal: how full is the Postgres 1 Gi PVC? (fraction 0..1)
curl -s --data-urlencode 'query=kubelet_volume_stats_used_bytes / kubelet_volume_stats_capacity_bytes' http://localhost:9090/api/v1/query

# Active backends vs the connection limit:
curl -s --data-urlencode 'query=sum(pg_stat_activity_count) / on() pg_settings_max_connections' http://localhost:9090/api/v1/query

# Database size (bytes) — feeds the PVC-fill picture:
curl -s --data-urlencode 'query=pg_database_size_bytes{datname="mlflow"}' http://localhost:9090/api/v1/query

# Postgres StatefulSet ready? (run-level, KSM — since PR 2)
curl -s --data-urlencode 'query=kube_statefulset_status_replicas_ready{namespace="mlops",statefulset="mlflow-postgres"}' http://localhost:9090/api/v1/query
```

> **What PR 4 added vs what was already there.** The blackbox `probe_*`, the `pg_*`
> series, and `kubelet_volume_stats_*` are new in PR 4. The **replica/readiness/
> restart/CPU/memory** signals for MLflow and Postgres were collectable from **KSM +
> cAdvisor since PR 2** — PR 4 documents their queries here; it did not add a
> component for them ([ADR-031](decisions/ADR-031-mlflow-postgres-monitoring.md)).

---

## 5. Troubleshooting

| Symptom | Likely cause | Investigate / remediate |
|---|---|---|
| A target is **DOWN** on `/targets` | Pod not Ready, or wrong port/label | `kubectl -n monitoring get pods`; `kubectl -n monitoring describe pod <p>`; check the Service selector matches the pod labels. |
| `node-exporter` pod won't schedule / admission denied | The `monitoring` namespace lost its `privileged` PSA label (e.g. namespace created by hand) | Re-apply the kustomization so the namespace labels are set (`kubectl get ns monitoring -o jsonpath='{.metadata.labels}'` should show `enforce=privileged`). See [ADR-029 § 5](decisions/ADR-029-monitoring-foundation.md). |
| `kubernetes-cadvisor` target is **DOWN** (403/401) | Prometheus RBAC missing `nodes/proxy` | Confirm the `prometheus` ClusterRole + binding applied: `kubectl get clusterrole prometheus -o yaml`. |
| **No `kube_job_*` series** for the pipeline | No Job has run yet, **or** the finished Job was reaped | Run the pipeline ([Kubernetes Operations](kubernetes-operations.md)); the finished Job persists for `ttlSecondsAfterFinished` (1h) — read the gauges within that window (already-scraped samples persist in the TSDB regardless). |
| **No `mlops_pipeline_stage_*` series** | No pipeline has run since the gateway came up, **or** `PUSHGATEWAY_URL` is unset/blanked in the pipeline's env | Confirm the `pushgateway` target is UP; run the pipeline; check `kubectl -n mlops get cm mlops-pipeline-config -o jsonpath='{.data.PUSHGATEWAY_URL}'` is the gateway Service FQDN. Emission is best-effort, so a bad URL fails silently (check the pipeline pod logs for a `Could not push metrics` WARNING). |
| **Stale/old stage series linger** across runs | The per-run reset did not run (e.g. the `fetch-dataset` init container was skipped) | The reset (`reset_pipeline_metrics`) runs first in `fetch-dataset`; confirm that init container executed. You can also clear a group by hand: `curl -X DELETE http://localhost:9091/metrics/job/mlops_pipeline/stage/<stage>` via a `port-forward svc/pushgateway 9091:9091`. |
| **`pushgateway` target UP but series carry `job="pushgateway"`** | The scrape lost the pushed labels | The scrape job must set `honor_labels: true` (it does — [prometheus-config.yaml](../k8s/monitoring/base/prometheus-config.yaml)); without it the pushed `job`/`stage` labels are overwritten. `k8s/validate.py` enforces this. |
| `kube-state-metrics` target UP but **no metrics** for a kind | That kind is not in KSM's read-only ClusterRole (deliberately scoped) | Expected for kinds outside the four-layer model; add the kind to the ClusterRole only if a documented signal needs it. |
| `blackbox-mlflow-health` shows `probe_success 0` **but MLflow looks up** | Wrong target/path, DNS, or MLflow genuinely not serving `/health` | Confirm the scrape target is `…/health` (not `/`); `kubectl -n mlops get pods` for MLflow; the probe is exempt from MLflow's host allow-list, so a 0 means the server did not return 2xx. `probe_http_status_code` shows what it got. |
| `postgres-exporter` target **DOWN** or `pg_up 0` | Exporter not deployed with the mlops workload, or the DB role/Secret is missing/wrong | The exporter ships in the **mlops** namespace (`kubectl -n mlops get deploy postgres-exporter`), not `monitoring`. `pg_up 0` with the target UP means auth/connect failed — create the `mlflow_exporter` role + `mlflow-postgres-exporter-credentials` Secret (see [the template](../k8s/base/mlflow/postgres-exporter-secret.example.yaml)); check `kubectl -n mlops logs deploy/postgres-exporter`. |
| **No `kubelet_volume_stats_*`** series | The `kubelet` scrape is down (RBAC) or the node has no PVC mounted | Confirm the `kubelet` target is UP on `/targets` (needs the same `nodes/proxy` RBAC as cAdvisor); volume stats appear only for nodes actually mounting a PVC (the Postgres one). |
| Prometheus pod **restarts / OOMs** | TSDB/scrape load above the starting limits | `kubectl -n monitoring top pod` (if metrics-server present) or check `container_memory_working_set_bytes`; raise the `prometheus` limits (they are conservative starting values, tightened from measurement in PR 6). |
| Metrics **disappeared after a restart** | Expected — the TSDB is an `emptyDir` (ephemeral by design, [ADR-029 § 3](decisions/ADR-029-monitoring-foundation.md)) | Capture evidence while the stack is live; use a PVC only if survive-a-restart is required. |

Useful commands:

```bash
kubectl -n monitoring logs deploy/prometheus            # scrape/config errors
kubectl -n monitoring logs deploy/kube-state-metrics    # RBAC/watch errors
kubectl -n monitoring get endpoints                     # do Services have endpoints?
# Reload the scrape config without deleting the pod (lifecycle API is enabled):
kubectl -n monitoring port-forward svc/prometheus 9090:9090 &
curl -s -X POST http://localhost:9090/-/reload
```

---

## 6. Cleanup

The stack is self-contained in the `monitoring` namespace and its cluster-scoped
RBAC. Remove it with the same kustomization used to deploy — this also deletes the
namespace (and, being an `emptyDir`, the TSDB simply vanishes with the pod; there
is **no PVC to clean up**):

```bash
# Local:
kubectl delete -k k8s/monitoring/overlays/local
# EKS:
kubectl delete -k k8s/monitoring/overlays/aws
```

Verify nothing is left — including the cluster-scoped RBAC (which a namespace
delete alone would **not** remove, so confirm the kustomization took it):

```bash
kubectl get ns monitoring                       # -> NotFound
kubectl get clusterrole,clusterrolebinding | grep -E 'prometheus|kube-state-metrics'   # -> no rows
```

On the ephemeral EKS platform, the subsequent `terraform destroy`
([Cloud Operations](cloud-operations.md), ADR-020) tears down the cluster itself;
removing the stack first keeps the destroy clean and leaves no orphaned
cluster-scoped RBAC.

---

## Related documentation

- [Observability & Operations](observability.md) — the architecture and signal catalogue
- [ADR-028](decisions/ADR-028-observability-architecture.md) · [ADR-029](decisions/ADR-029-monitoring-foundation.md) · [ADR-030](decisions/ADR-030-pipeline-operational-metrics.md) · [ADR-031](decisions/ADR-031-mlflow-postgres-monitoring.md)
- [Kubernetes Operations](kubernetes-operations.md) · [Cloud Operations](cloud-operations.md)
- [`k8s/monitoring/`](../k8s/monitoring/) — the manifests · [`src/pipeline_metrics.py`](../src/pipeline_metrics.py) — the pipeline's metric emitter
