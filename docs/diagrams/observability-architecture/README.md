# Observability Architecture Diagram

Source for the Sprint 8 observability architecture. Discussed in
[docs/observability.md](../../observability.md); the design of record is
[ADR-028](../../decisions/ADR-028-observability-architecture.md).

> **Status.** ✅ **Validated on live EKS (Sprint 8, v1.7.0).** This diagram defined
> the target at PR 1; the Sprint 8 runtime PRs then built and exercised it on real
> EKS (11 Prometheus targets UP, three Grafana dashboards, eight alert rules —
> [Sprint 8 evidence](../../proof/README.md)). The one edge that solves the
> ephemeral-Job problem — **KSM reads the persistent `Job` object, not the exited
> pod** — is drawn deliberately. A reviewer-facing **operational data-flow** view
> (which signals reach Prometheus, and which deliberately do **not**) follows below.

## Target architecture

```mermaid
flowchart TB
    operator["Operator<br/><i>kubectl port-forward (internal only)</i>"]

    subgraph cluster["Kubernetes cluster (local Docker Desktop / EKS)"]
        subgraph mon["Namespace: monitoring (planned)"]
            prom["Prometheus<br/><i>pull scrape · local TSDB · short retention</i>"]
            graf["Grafana<br/><i>four-layer dashboards · ClusterIP</i>"]
            ksm["kube-state-metrics<br/><i>API-object state</i>"]
            nodeexp["node-exporter<br/><i>DaemonSet · node health</i>"]
            black["blackbox-exporter<br/><i>probes MLflow /health</i>"]
            alerts["Alert rules<br/><i>(PR 5)</i>"]
        end

        subgraph mlops["Namespace: mlops"]
            job["Job: mlops-pipeline<br/><i>batch/v1 · pod EXITS &lt; 1 min</i>"]
            jobobj[["Job OBJECT<br/><i>persists after pod (ttlSecondsAfterFinished)</i>"]]
            mlflow["Deployment: mlflow<br/><i>/health · no native /metrics</i>"]
            pg["StatefulSet: mlflow-postgres<br/><i>1Gi PVC</i>"]
            pgexp["postgres-exporter<br/><i>read-only DB role · beside the DB (PR 4)</i>"]
        end

        kubelet["kubelet / cAdvisor<br/><i>per-container mem/cpu</i>"]

        job -. "creates / updates" .-> jobobj

        %% scrape edges (pull)
        ksm -- "watches Job/Pod/Deploy/STS objects" --> jobobj
        ksm --> mlflow
        ksm --> pg
        prom -- scrape --> ksm
        prom -- scrape --> nodeexp
        prom -- scrape --> kubelet
        prom -- scrape --> pgexp
        prom -- scrape --> black
        black -- "HTTP /health" --> mlflow
        pgexp -- "SQL (read-only)" --> pg

        prom --> alerts
        graf -- PromQL --> prom
    end

    operator --> graf

    classDef planned fill:#f7f7f7,stroke:#aaa,stroke-dasharray:4 3,color:#555;
    classDef boundary fill:#eef,stroke:#557,stroke-width:1px;
    classDef obj fill:#fef9e7,stroke:#b7950b,color:#7d6608;
    class cluster,mon,mlops boundary;
    class prom,graf,ksm,nodeexp,black,pgexp,alerts planned;
    class jobobj obj;
```

## Operational data flow (reviewer view)

Which signals Prometheus actually collects — and the boundary that matters:
**experiment metrics stay in MLflow/PostgreSQL and are NOT duplicated in
Prometheus.** Prometheus carries only *operational* signals.

```mermaid
flowchart TB
    subgraph sources["Operational signal sources"]
        pgw["Pushgateway<br/><i>last-run per-stage mlops_pipeline_stage_success</i>"]
        ksm["kube-state-metrics<br/><i>Job / Pod / Deploy / STS objects</i>"]
        node["node-exporter + kubelet/cAdvisor<br/><i>node + container health</i>"]
        black["blackbox-exporter<br/><i>MLflow /health probe</i>"]
        pgexp["postgres-exporter<br/><i>pg_up · size · memory</i>"]
    end

    pipe["Pipeline Job<br/><i>batch/v1</i>"]
    prom["Prometheus<br/><i>pull scrape · 8 alert rules</i>"]
    graf["Grafana<br/><i>4-layer dashboards</i>"]
    op["Operator<br/><i>runbooks/ → diagnose → remediate</i>"]
    mldb[("MLflow + PostgreSQL<br/><i>EXPERIMENT metrics live here</i>")]

    pipe -- "push (9091)" --> pgw
    pipe -. "params / metrics / artifacts — NOT scraped" .-> mldb
    pgw --> prom
    ksm --> prom
    node --> prom
    black --> prom
    pgexp --> prom
    prom --> graf
    prom -- "alert Firing" --> op
    graf --> op
    op -. "runbook recovery" .-> pipe

    classDef boundary fill:#eef,stroke:#557,stroke-width:1px;
    classDef store fill:#eefaf0,stroke:#2e7d5b;
    class sources boundary;
    class mldb store;
```

**What it proves / helps explain.**

- **The operational vs. experiment boundary.** The pipeline's *operational* health
  (did each stage succeed?) is pushed to the **Pushgateway** and scraped; the
  pipeline's *experiment* results (parameters, accuracy, artifacts) go to
  **MLflow → PostgreSQL** and are **not** mirrored into Prometheus. Two stores, one
  purpose each.
- Every operational signal is a **pull** scrape into Prometheus; Grafana reads it by
  PromQL; alerts route an operator to a runbook and back to a verified recovery.

**Limitations.** No Alertmanager routing (alerts show on Prometheus `/alerts`
only); short local TSDB retention; internal-only access via `kubectl port-forward`.

## The four layers (what Prometheus/Grafana surface)

```mermaid
flowchart LR
    subgraph L1["Layer 1 · K8s platform"]
        l1["nodes Ready? · pressure?<br/>pods CrashLooping?<br/>near memory limit?"]
    end
    subgraph L2["Layer 2 · Pipeline Job (ephemeral)"]
        l2["last run succeeded?<br/>run duration? OOMKilled?<br/><b>via KSM Job object</b>"]
    end
    subgraph L3["Layer 3 · MLflow"]
        l3["available (/health)?<br/>restarting?<br/>near 2Gi limit?"]
    end
    subgraph L4["Layer 4 · PostgreSQL"]
        l4["up? PVC filling (1Gi)?<br/>connections?<br/>ready?"]
    end
    L1 --- L2 --- L3 --- L4
```

## ASCII fallback

```text
Operator ── kubectl port-forward (internal only) ─▶ Grafana ──PromQL──▶ Prometheus
                                                                            │ pull scrape
        ┌───────────────────────────────────────────────────────────────┬─┴───────────────┐
        ▼                    ▼                 ▼                ▼          ▼                 ▼
  kube-state-metrics   node-exporter     kubelet/cAdvisor  blackbox-exp  postgres-exp   (alert rules)
        │                                                     │             │
        │ watches API OBJECTS                        HTTP /health      SQL (read-only)
        ▼                                                     ▼             ▼
  ┌───────────────┐                                       MLflow        PostgreSQL
  │  Job OBJECT   │◀── created by ── Job pod (EXITS <1min)  (Deployment)  (StatefulSet, 1Gi PVC)
  │ persists via  │
  │ ttlSeconds... │   ← the pod is gone, but its Job object (success/duration/OOM) is still scrapable
  └───────────────┘

Layers surfaced:  1) K8s platform   2) Pipeline Job (ephemeral, via KSM)   3) MLflow   4) PostgreSQL
```

## Why the highlighted edge matters

The pipeline **pod exits in under a minute** and has no Service to scrape
(ADR-009/011). The dashed **Job OBJECT** node is the key: **kube-state-metrics
watches the persistent `Job` API object, not the ephemeral process**, so
`succeeded / failed / start_time / completion_time` remain scrapable after the pod
is gone. The one Pod-object signal, **OOMKilled**
(`kube_pod_container_status_last_terminated_reason`), stays scrapable too because the
Job's finished pod is retained by owner-reference for as long as the Job — all
provided the finished Job (and its pod) outlives one scrape
(`ttlSecondsAfterFinished`). Full reasoning and the rejected alternatives
(Pushgateway, custom exporter, keep-alive sidecar) are in
[docs/observability.md § 4](../../observability.md#4-the-batch-job-problem-keeping-an-ephemeral-jobs-metrics-queryable)
and [ADR-028 § 3](../../decisions/ADR-028-observability-architecture.md#3-batch-workload-metric-strategy--kube-state-metrics-is-the-answer-not-pushgateway).
