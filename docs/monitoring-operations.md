# Monitoring Operations

Day-2 operations for the **metrics foundation** — Prometheus, kube-state-metrics,
node-exporter, and the **Pushgateway** — deployed from
[`k8s/monitoring/`](../k8s/monitoring/). This is the operator's runbook for the
Sprint 8 PR 2 + PR 3 stack: how to deploy it, reach Prometheus, run a query
(including the pipeline's per-stage metrics), troubleshoot a missing target, and
tear it down cleanly.

> **Scope — the metrics core + pipeline operational metrics, not yet runtime-proven.**
> This covers Prometheus + KSM + node-exporter + the cAdvisor scrape (Layer 1
> platform signals and the Layer 2 batch-Job signals via KSM, PR 2) **and the
> Pushgateway the pipeline pushes per-stage duration/success to** (PR 3,
> [ADR-030](decisions/ADR-030-pipeline-operational-metrics.md)). **Grafana**,
> **MLflow/PostgreSQL exporters** (PR 4), and **alerts** (PR 5) are not here. As of
> these PRs the stack is **defined and statically validated but not deployed** — no
> live cluster was available. The commands below are the runbook for when it *is*
> deployed; the live four-layer evidence is the job of the runtime-evidence PR
> (PR 6, per [`docs/observability.md`](observability.md#runtime-evidence-what-later-sprint-8-prs-must-prove)).
> Design of record: [ADR-028](decisions/ADR-028-observability-architecture.md),
> [ADR-029](decisions/ADR-029-monitoring-foundation.md), and
> [ADR-030](decisions/ADR-030-pipeline-operational-metrics.md).

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
| `prometheus` / `kube-state-metrics` | ServiceAccount + **read-only** ClusterRole + binding | Least-privilege API access | — |

The kubelet's built-in **cAdvisor** is scraped through the API server proxy (no
separate component). Everything is `ClusterIP`/headless — nothing is exposed
outside the cluster; you reach Prometheus with `kubectl port-forward` (§ 3).

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
#         pushgateway-* Running (1/1), node-exporter-* Running (1/1) on every node.
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
  `pushgateway`) should show its targets **UP**. This is the first thing to check
  after a deploy. (The `pushgateway` target is UP as soon as the gateway is running,
  even before any pipeline has pushed — it just has no `mlops_pipeline_*` series yet.)

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
curl -s --data-urlencode 'query=min(mlops_pipeline_stage_success)' http://localhost:9090/api/v1/query

# Which stage failed (if any)?
curl -s --data-urlencode 'query=mlops_pipeline_stage_success == 0' http://localhost:9090/api/v1/query

# Approximate run cadence over the last day (Pushgateway's built-in push_time):
curl -s --data-urlencode 'query=changes(push_time_seconds{job="mlops_pipeline"}[1d])' http://localhost:9090/api/v1/query
```

> **Operational only.** These series describe *execution* (how long, did it
> succeed). Model **accuracy** and hyper-parameters are **not** here — they live in
> MLflow (the ownership boundary, [ADR-030](decisions/ADR-030-pipeline-operational-metrics.md)).
> Whole-run duration is authoritative from KSM (`completion_time − start_time`
> above); the per-stage series is the *decomposition* of it.

> The Layer 1 + Layer 2 signals above are all this stack serves today. Layer 3
> (MLflow `/health`) and Layer 4 (Postgres internals / PVC fill) need the PR 4
> exporters and will return no data until then.

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
- [ADR-028](decisions/ADR-028-observability-architecture.md) · [ADR-029](decisions/ADR-029-monitoring-foundation.md) · [ADR-030](decisions/ADR-030-pipeline-operational-metrics.md)
- [Kubernetes Operations](kubernetes-operations.md) · [Cloud Operations](cloud-operations.md)
- [`k8s/monitoring/`](../k8s/monitoring/) — the manifests · [`src/pipeline_metrics.py`](../src/pipeline_metrics.py) — the pipeline's metric emitter
