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
  prometheus-config.yaml   # Prometheus scrape config (ConfigMap): 5 scrape jobs
  prometheus.yaml          # Prometheus: SA + read-only ClusterRole + binding + Deployment + Service
  kustomization.yaml
overlays/
  local/                   # Docker Desktop / kind / minikube  (currently == base)
  aws/                     # EKS                                 (currently == base)
```

Covers **Layer 1** (Kubernetes platform — node-exporter + cAdvisor + KSM), the
**Layer 2** batch-Job run-level signals (via KSM, PR 2), and the **Layer 2
per-stage** operational metrics the pipeline pushes to the **Pushgateway** (PR 3 —
[ADR-030](../../docs/decisions/ADR-030-pipeline-operational-metrics.md)). Grafana
dashboards, MLflow/Postgres exporters (PR 4), and alerts (PR 5) are **not** here.

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
  the API; node-exporter and the Pushgateway mount no token (neither calls the API).
- **Pushgateway (PR 3):** the ephemeral pipeline Job cannot be pull-scraped, so each
  stage **pushes** its duration + success here before exiting; Prometheus scrapes the
  gateway with `honor_labels: true`. In-memory (no persistence), reset once per run
  by the pipeline to avoid stale series, internal-only. Operational metrics only —
  model accuracy/params stay in MLflow ([ADR-030](../../docs/decisions/ADR-030-pipeline-operational-metrics.md)).
- **Ephemeral storage:** the Prometheus TSDB is an `emptyDir` with 7d/1GB retention
  — no PVC, no long-term store (cost discipline, ADR-020/ADR-028 § 5).
