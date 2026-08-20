# ADR-032: Grafana dashboards for platform & pipeline operations (Sprint 8, PR 5)

- **Status:** Accepted (design; manifests + dashboard JSON added, no components deployed in this PR)
- **Date:** 2026-08-20
- **Deciders:** Asad Hanif
- **Related:**
  [ADR-028 (Observability architecture — Prometheus for metrics, Grafana for dashboards; the four-layer model)](ADR-028-observability-architecture.md),
  [ADR-029 (Monitoring foundation — Prometheus + KSM + node-exporter)](ADR-029-monitoring-foundation.md),
  [ADR-030 (Pipeline operational metrics via Pushgateway)](ADR-030-pipeline-operational-metrics.md),
  [ADR-031 (MLflow & PostgreSQL monitoring — blackbox + postgres-exporter)](ADR-031-mlflow-postgres-monitoring.md),
  [ADR-026 (In-cluster MLflow platform)](ADR-026-in-cluster-mlflow-platform.md),
  [ADR-010 (Kubernetes security hardening)](ADR-010-kubernetes-security-hardening.md),
  [ADR-012 (Kubernetes manifest validation)](ADR-012-kubernetes-manifest-validation.md),
  [ADR-020 (Ephemeral cloud lifecycle & cost control)](ADR-020-cloud-lifecycle-cost-control.md),
  [`docs/observability.md`](../observability.md),
  [`docs/monitoring-operations.md`](../monitoring-operations.md),
  [`k8s/monitoring/base/grafana/`](../../k8s/monitoring/base/grafana/)

> **Scope.** This ADR ratifies the **Grafana dashboards** layer that
> [ADR-028 § 1](ADR-028-observability-architecture.md) chose as the visualisation
> tool and that the sprint **resequenced to land after** the metrics PRs (2–4), so
> the dashboards have the full signal set — Layers 1–4 plus the per-stage Pushgateway
> series — to render. It **adds Grafana manifests, provisioning, and three
> version-controlled dashboard JSON files, plus validation and docs**; **nothing is
> deployed** (no live cluster — runtime proof is the runtime-evidence PR). It does
> **not** add alert rules (the alerting PR) and does **not** re-decide the
> architecture.

## Context

After PRs 2–4 the platform emits a complete, four-layer metric set into Prometheus:

- **Layer 1 (Kubernetes platform):** node-exporter (`node_*`), cAdvisor
  (`container_*`), and kube-state-metrics (`kube_node_*`, `kube_pod_*`,
  `kube_deployment_*`, `kube_statefulset_*`, `kube_job_*`).
- **Layer 2 (MLOps pipeline):** run-level state from KSM (`kube_job_status_*`,
  `kube_pod_container_status_last_terminated_reason`) and **per-stage** duration /
  success pushed by the pipeline to the Pushgateway
  (`mlops_pipeline_stage_duration_seconds`, `mlops_pipeline_stage_success`;
  [ADR-030](ADR-030-pipeline-operational-metrics.md)).
- **Layer 3 (MLflow):** `probe_success` / `probe_http_status_code` /
  `probe_duration_seconds` from blackbox-exporter against `/health`
  ([ADR-031](ADR-031-mlflow-postgres-monitoring.md)).
- **Layer 4 (PostgreSQL):** `pg_up`, `pg_stat_activity_count`,
  `pg_settings_max_connections`, `pg_database_size_bytes` from postgres-exporter, and
  `kubelet_volume_stats_*` for PVC fill.

Those signals are only reachable today via ad-hoc PromQL in the Prometheus
expression browser. The operational gap ADR-028 § 1 named — *"answering an ordinary
operational question needs a human at a terminal at the right moment"* — is closed by
**purpose-built dashboards** that map each signal to a question an operator actually
asks.

The brief adds firm constraints: **do not** import arbitrary community dashboards and
call it done; build **three purposeful** dashboards, each answering a named set of
questions; keep **model accuracy / model comparison in MLflow**; **version-control**
the JSON and provisioning; use **stable PromQL and bounded time windows**; add panel
help text; and **never expose** AWS account IDs, secrets, or credential values.

## Decision

### 1. Three purpose-built dashboards, each mapped to operational questions

Not a community bundle — three hand-authored dashboards
([`k8s/monitoring/base/grafana/dashboards/`](../../k8s/monitoring/base/grafana/dashboards/)),
each panel tied to a question from [`docs/observability.md` § 3](../observability.md):

| Dashboard (uid) | Answers |
|---|---|
| **EKS / Platform Health** (`mlops-eks-platform-health`) | Are nodes Ready? Are workloads healthy (Deployment replicas available vs desired, pods by phase)? Are pods restarting (CrashLoopBackOff, 1h restart increase)? Is CPU/memory/disk pressure visible (node utilisation, pressure conditions)? Are Jobs failing? |
| **MLOps Pipeline Operations** (`mlops-pipeline-operations`) | Did the most recent pipeline succeed (`kube_job_status_succeeded`)? How long did it take (completion − start)? Which stage dominates runtime (per-stage duration bar gauge)? Did dataset retrieval fail (`fetch_dataset` stage success)? Are recent failures visible (per-stage success table, Job outcomes, OOMKilled)? |
| **MLflow Platform Health** (`mlops-mlflow-platform-health`) | Is MLflow available (`probe_success`)? Are MLflow pods stable (replicas, 1h restarts)? Is PostgreSQL available (`pg_up`, StatefulSet ready)? Are either under resource pressure (memory-vs-limit gauges, PVC-fill gauge, connections vs max)? |

**Model quality stays in MLflow.** The pipeline dashboard carries an explicit help
panel stating accuracy / best params / per-run artifacts are owned by MLflow, not
Prometheus/Grafana — preserving the ownership boundary of ADR-028 § 5 / ADR-030. No
dashboard queries model-semantic series (the pipeline never emits them).

### 2. Everything provisioned from version-controlled files

Grafana loads its datasource and dashboards from files on disk at start-up — no
manual UI import, no click-ops, nothing stored only in Grafana's database:

- **Datasource** ([`grafana-datasource.yaml`](../../k8s/monitoring/base/grafana/grafana-datasource.yaml)):
  one Prometheus datasource, `access: proxy` (queries run server-side so Prometheus
  stays internal-only), a fixed `uid: prometheus` that every dashboard references, and
  `editable: false` so the file is the single source of truth.
- **Dashboard provider** ([`grafana-dashboard-provider.yaml`](../../k8s/monitoring/base/grafana/grafana-dashboard-provider.yaml)):
  a `file` provider pointing at `/var/lib/grafana/dashboards`, `allowUiUpdates: false`
  so a UI edit can never diverge from git, re-reading the directory every 30 s.
- **Dashboard JSON**: three first-class `.json` files, packaged into ONE ConfigMap by
  a kustomize `configMapGenerator` (so the JSON is never embedded as a YAML string and
  can be validated on its own), mounted where the provider looks.

### 3. Grafana hardened to the monitoring contract, internal-only

Grafana runs in the `monitoring` namespace as a single stateless replica
([`grafana.yaml`](../../k8s/monitoring/base/grafana/grafana.yaml)): non-root uid 472,
drop ALL, no privilege escalation, seccomp RuntimeDefault, **read-only root
filesystem** (writable `/var/lib/grafana` and `/tmp` are dedicated emptyDir volumes),
no ServiceAccount token (Grafana calls only Prometheus, never the K8s API). It is a
**ClusterIP** Service reached by `kubectl port-forward` — never a NodePort /
LoadBalancer, the same internal-only posture as Prometheus and the MLflow UI. All
outbound calls (update checks, analytics, gravatar, news feed) are disabled so a
render never reaches outside the cluster. State is an emptyDir (dashboards are
re-provisioned from disk each start), so there is nothing durable to persist — no PVC,
no standing cost, matching ADR-020.

### 4. No secrets or account IDs exposed

The admin credential comes **only** from an out-of-band Secret
([`grafana-admin-secret.example.yaml`](../../k8s/monitoring/base/grafana/grafana-admin-secret.example.yaml)
is a placeholder-only template, never applied by kustomize), delivered as env from a
`secretKeyRef` — never inline, never in a ConfigMap, never in the dashboards.
Anonymous access is enabled at the **Viewer** role so dashboards are readable over a
port-forward without a login, while editing needs the admin credential. Dashboard JSON
references resources by their in-cluster names/labels only (`namespace="mlops"`,
`deployment="mlflow"`, …) — **no AWS account IDs**, ARNs, or endpoints appear anywhere.

### 5. Stable PromQL, bounded windows

Every panel uses stable metric names and **bounded** range windows (`[5m]` for node
CPU rate, `[1h]` for restart increase) — no unbounded `[$__range]` scans. Dashboards
default to bounded time ranges (`now-6h`, `now-24h` for the pipeline history), refresh
at 1 m to match the 30 s scrape interval, and resource-pressure panels are expressed
as percent-of-limit gauges with thresholds at the ADR-028 § 6 objectives (memory 90 %,
PVC 85 %).

## Alternatives considered

| Option | Why not |
|---|---|
| **Import a community dashboard bundle** (e.g. the "Kubernetes / Node Exporter Full" pack) | Explicitly forbidden by the brief, and rightly: hundreds of panels for signals this workload does not emit, no mapping to *our* operational questions, and a maintenance liability. We build exactly the three dashboards the four-layer model needs. |
| **Store dashboards only in Grafana's DB (UI-authored)** | Not version-controlled, drifts from git, lost on the emptyDir teardown. File provisioning + `allowUiUpdates: false` makes the committed JSON authoritative. |
| **Embed dashboard JSON inline in a ConfigMap YAML** | Unreadable, unvalidatable as JSON, and merge-hostile. Keeping `.json` files + a `configMapGenerator` gives first-class, individually-lintable artifacts. |
| **Expose Grafana via LoadBalancer / Ingress** | Would put an auth-optional dashboards server on the public internet. Internal-only ClusterIP + port-forward matches the Prometheus/MLflow posture (ADR-026/028). |
| **Put model accuracy on the pipeline dashboard** | Violates the ownership boundary (ADR-028 § 5): Prometheus is not an experiment store; accuracy/params live in MLflow. A help panel points there instead. |
| **A PVC for Grafana state** | Dashboards are provisioned from disk every start, so there is nothing durable worth a standing EBS cost + teardown step (ADR-020). emptyDir is the right fit for an ephemeral validation cluster. |

## Consequences

**Positive**
- Every operational question in [`observability.md` § 3](../observability.md) now has
  a panel; an operator answers "did the overnight run succeed / is MLflow up / is the
  PVC filling?" at a glance instead of hand-writing PromQL.
- Dashboards are version-controlled, code-reviewed, and provisioned automatically:
  `kubectl apply -k` brings them up with zero manual steps.
- The security posture is unchanged from the rest of the stack — hardened,
  internal-only, no secrets or account IDs on screen.
- `k8s/validate.py` (monitoring pass) and `kubeconform` both cover the Grafana
  manifests; the dashboard JSON is parse-validated in CI.

**Negative / limitations**
- **Nothing is deployed here** — this PR proves the manifests render, validate, and
  the JSON parses; that panels *populate with real data* is the runtime-evidence PR's
  job (provision → run a healthy pipeline → confirm panels populate).
- Grafana state is ephemeral: UI-side changes (annotations, ad-hoc panels) do not
  survive a restart. That is intentional — the committed JSON is the source of truth.
- No alerting here (alert rules are the alerting PR); the dashboards' thresholds are
  visual early-warning only, not notifications.
- Panels that depend on a running workload (per-stage series, MLflow/PG memory) show
  "no data" until a pipeline has run and the platform is up — expected on a cold
  cluster.

## Validation

- `kustomize build k8s/monitoring/base` renders Grafana + provisioning + the
  dashboards ConfigMap.
- `python k8s/validate.py` — the monitoring pass asserts Grafana's hardening
  (non-root, drop ALL, seccomp, read-only root FS), pinned image, resources,
  internal-only Service, and token discipline.
- `kubeconform -strict` schema-validates every rendered object.
- The three dashboard JSON files parse as JSON with unique panel ids and required
  panel fields (CI check).
