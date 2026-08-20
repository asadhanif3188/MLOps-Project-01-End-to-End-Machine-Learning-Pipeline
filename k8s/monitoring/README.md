# Monitoring manifests (`k8s/monitoring/`)

The platform **observability foundation** (Sprint 8, PR 2): a minimal,
hand-written Prometheus stack that observes the EKS/local platform. It is a
**separate Kustomize root** from the mlops workload (`k8s/base`), with its own
`monitoring` namespace and lifecycle, so the two are deployed and torn down
independently ([ADR-028 § 4](../../docs/decisions/ADR-028-observability-architecture.md)).

> **Status — manifests only, not deployed.** This PR adds version-controlled,
> hardened, statically-validated manifests. **Nothing is deployed** (no live
> cluster was available); full platform observability is **not** claimed — runtime
> proof is Sprint 8 PR 6. Design of record:
> [ADR-028](../../docs/decisions/ADR-028-observability-architecture.md) (architecture)
> and [ADR-029](../../docs/decisions/ADR-029-monitoring-foundation.md) (this
> implementation). Operator runbook:
> [docs/monitoring-operations.md](../../docs/monitoring-operations.md).

## What's here

```
base/
  namespace.yaml           # monitoring ns + Pod Security labels (privileged — see below)
  kube-state-metrics.yaml  # KSM: SA + read-only ClusterRole + binding + Deployment + Service
  node-exporter.yaml       # node-exporter: SA + DaemonSet + headless Service
  pushgateway.yaml         # Pushgateway (PR 3): SA + Deployment + Service — sink for the
                           #   ephemeral pipeline's per-stage operational metrics
  blackbox-exporter.yaml   # blackbox (PR 4): SA + module ConfigMap + Deployment + Service —
                           #   Layer 3 MLflow /health availability probing
  prometheus-config.yaml   # Prometheus scrape config (ConfigMap): 8 scrape jobs
  prometheus.yaml          # Prometheus: SA + read-only ClusterRole + binding + Deployment + Service
  kustomization.yaml
overlays/
  local/                   # Docker Desktop / kind / minikube  (currently == base)
  aws/                     # EKS                                 (currently == base)
```

> **Layer 4 postgres-exporter lives elsewhere.** The PostgreSQL exporter (PR 4) is in
> the **mlops** workload ([`k8s/base/mlflow/postgres-exporter.yaml`](../base/mlflow/postgres-exporter.yaml)),
> not this stack — it is co-located with the DB so its dedicated read-only credential
> Secret stays in the `mlops` namespace. Prometheus scrapes it cross-namespace
> ([ADR-031](../../docs/decisions/ADR-031-mlflow-postgres-monitoring.md)).

Covers **Layer 1** (Kubernetes platform — node-exporter + cAdvisor + KSM), the
**Layer 2** batch-Job run-level signals (via KSM, PR 2), the **Layer 2 per-stage**
operational metrics the pipeline pushes to the **Pushgateway** (PR 3 —
[ADR-030](../../docs/decisions/ADR-030-pipeline-operational-metrics.md)), and the
**Layer 3/4 platform depth** (PR 4 — [ADR-031](../../docs/decisions/ADR-031-mlflow-postgres-monitoring.md)):
MLflow `/health` (blackbox), PostgreSQL backend health (postgres-exporter), and the
Postgres PVC-fill signal (a scoped kubelet volume-stats scrape). Grafana dashboards
and alerts (PR 5) are **not** here.

## Build & validate

```bash
kustomize build k8s/monitoring/base                 # render
python k8s/validate.py k8s/overlays/local           # runs the monitoring pass too
kustomize build k8s/monitoring/base | kubeconform -strict -   # schema (CI-pinned)
```

## Deploy / operate / clean up

See **[docs/monitoring-operations.md](../../docs/monitoring-operations.md)** —
deploy, port-forward Prometheus, run a PromQL query, troubleshoot, and tear down.

## Notes worth knowing

- **Pod Security exception:** the `monitoring` namespace is `enforce: privileged`
  because node-exporter needs read-only `hostPath` access to the node's `/proc`,
  `/sys`, and rootfs — the sole documented exception
  ([ADR-029 § 5](../../docs/decisions/ADR-029-monitoring-foundation.md)). Prometheus
  and KSM stay restricted-equivalent.
- **Least privilege:** Prometheus/KSM ClusterRoles are **read-only**
  (`get`/`list`/`watch`); their tokens are mounted only because they genuinely use
  the API; node-exporter, the Pushgateway, and the blackbox/postgres exporters mount
  no token (none calls the API).
- **Layer 3/4 depth (PR 4):** blackbox probes MLflow's **stable, load-free `/health`**
  (MLflow has no native `/metrics`); postgres-exporter reports `pg_up`/connections/
  size via a **dedicated `pg_monitor`-only DB role** (credentials in a `mlops` Secret,
  never in config or metrics); the Postgres **PVC-fill** signal comes from a **scoped
  kubelet** volume-stats scrape (`kubelet_volume_stats_*` only). Run-level replica/
  readiness/restart/CPU/memory signals came from KSM + cAdvisor in PR 2.
- **Pushgateway (PR 3):** the ephemeral pipeline Job cannot be pull-scraped, so each
  stage **pushes** its duration + success here before exiting; Prometheus scrapes the
  gateway with `honor_labels: true`. In-memory (no persistence), reset once per run
  by the pipeline to avoid stale series, internal-only. Operational metrics only —
  model accuracy/params stay in MLflow ([ADR-030](../../docs/decisions/ADR-030-pipeline-operational-metrics.md)).
- **Ephemeral storage:** the Prometheus TSDB is an `emptyDir` with 7d/1GB retention
  — no PVC, no long-term store (cost discipline, ADR-020/ADR-028 § 5).
