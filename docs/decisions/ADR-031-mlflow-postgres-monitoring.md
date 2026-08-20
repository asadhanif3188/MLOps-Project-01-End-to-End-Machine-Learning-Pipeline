# ADR-031: MLflow & PostgreSQL platform monitoring (Sprint 8, PR 4)

- **Status:** Accepted (design; manifests added, no components deployed in this PR)
- **Date:** 2026-08-20
- **Deciders:** Asad Hanif
- **Related:**
  [ADR-028 (Observability architecture — the four-layer model; Layer 3/4 depth deferred to PR 4)](ADR-028-observability-architecture.md),
  [ADR-029 (Monitoring foundation — Prometheus + KSM + node-exporter)](ADR-029-monitoring-foundation.md),
  [ADR-030 (Pipeline operational metrics via Pushgateway)](ADR-030-pipeline-operational-metrics.md),
  [ADR-026 (In-cluster MLflow platform — MLflow Deployment + Postgres StatefulSet)](ADR-026-in-cluster-mlflow-platform.md),
  [ADR-010 (Kubernetes security hardening)](ADR-010-kubernetes-security-hardening.md),
  [ADR-012 (Kubernetes manifest validation)](ADR-012-kubernetes-manifest-validation.md),
  [ADR-020 (Ephemeral cloud lifecycle & cost control)](ADR-020-cloud-lifecycle-cost-control.md),
  [`docs/observability.md`](../observability.md),
  [`docs/monitoring-operations.md`](../monitoring-operations.md),
  [`k8s/monitoring/base/blackbox-exporter.yaml`](../../k8s/monitoring/base/blackbox-exporter.yaml),
  [`k8s/base/mlflow/postgres-exporter.yaml`](../../k8s/base/mlflow/postgres-exporter.yaml)

> **Scope.** This ADR ratifies the **Layer 3 (MLflow)** and **Layer 4
> (PostgreSQL)** monitoring depth that [ADR-028 § 6](ADR-028-observability-architecture.md)
> and [ADR-029 § 2](ADR-029-monitoring-foundation.md) explicitly **deferred to
> PR 4**. It implements ADR-028's PR-4 acceptance criteria — it does **not**
> re-decide the architecture. It **adds manifests, scrape config, validation, and
> docs only**; **nothing is deployed** (no live cluster — runtime proof is PR 6).

## Context

After PR 2 (foundation) and PR 3 (pipeline per-stage metrics), the observable
gaps are the two **long-running platform** workloads (ADR-026):

- **MLflow** (a `Deployment`) — its *run-level* signals (available replicas, pod
  readiness, restarts, CPU, memory) are **already collectable** from
  kube-state-metrics + cAdvisor deployed in PR 2. What is missing is **"is it
  actually serving?"** — MLflow ships **no native Prometheus `/metrics`**, so
  scraping it directly is impossible.
- **PostgreSQL** (a `StatefulSet`) — likewise its readiness/restarts/CPU/memory
  come from KSM + cAdvisor already. Missing are the DB's **own backend health**
  (can a client connect and query — `pg_up`; connections vs the limit; database
  size) and, the single highest-value signal given the **fixed 1 Gi PVC**, whether
  that **volume is filling up**.

The brief adds firm constraints: prefer native/exporter metrics where justified;
**do not overcomplicate** Postgres monitoring; **do not** add a full
database-monitoring product; **do not expose DB credentials** in metrics/config;
**review exporter security carefully**; and if health-probing MLflow over HTTP, use
a **stable endpoint** and **avoid unnecessary load**.

The important framing: **PR 4 is mostly the two depth exporters plus documenting
the queries for the run-level signals PR 2 already makes collectable.** It is not a
from-scratch monitoring build.

## Decision

### 1. MLflow availability — blackbox-exporter probing `/health`

Deploy a **blackbox-exporter** in the `monitoring` namespace
([`blackbox-exporter.yaml`](../../k8s/monitoring/base/blackbox-exporter.yaml)).
Prometheus hands it MLflow's URL via a `/probe` scrape (a 7th scrape job); the
exporter performs a single HTTP GET and returns `probe_success`,
`probe_http_status_code`, and `probe_duration_seconds`.

- **Why a generic prober, not an MLflow-specific exporter.** The operational
  question is shallow and universal — *"does `/health` return 2xx, and how fast?"*.
  blackbox answers exactly that with **no MLflow coupling and no credentials**, and
  can later probe any other internal endpoint. Deep request-level RED metrics
  (per-API rate/errors/latency) need an app-side exporter inside MLflow and stay
  **deferred** (ADR-028 § 6) — availability + resource (cAdvisor) cover the
  operational question now.
- **A stable, load-free endpoint.** MLflow's `/health` (and `/version`) are
  **exempt** from the server's DNS-rebinding host allow-list (ADR-026), so the probe
  passes regardless of Host header, and `/health` is a trivial "server up" handler —
  **no DB round-trip**. Prometheus probes it **once per 30 s scrape**: one small GET
  per interval, negligible load (the brief's "avoid unnecessary load").
- **Honest limitation:** `/health` is **shallow** — it proves the server process is
  up, not that its DB connection is healthy. Deep MLflow health is out of scope; the
  Postgres exporter (below) covers the DB side independently.

### 2. PostgreSQL backend health — a minimal postgres-exporter with a dedicated role

Deploy a **postgres-exporter**
([`postgres-exporter.yaml`](../../k8s/base/mlflow/postgres-exporter.yaml)) that
connects to PostgreSQL and exposes `pg_up`, `pg_stat_activity`/`pg_settings`
(connections vs limit), and `pg_database_size_bytes`.

- **Separate `Deployment`, not a StatefulSet sidecar.** A standalone exporter is
  **independently observable**: if the DB is down it reports `pg_up 0` (a positive
  signal), where a sidecar would go down *with* the DB pod and surface only as a
  scrape `up == 0`. It also keeps the exporter's restarts out of the stateful DB
  pod. The only cost — an extra intra-cluster hop to the DB — is nil (it is the same
  network path the MLflow server already uses). *Sidecar is a reasonable
  alternative and is recorded below.*
- **Runs in the `mlops` namespace, beside the DB.** The exporter needs DB
  credentials, and a `Secret` is namespace-scoped. Co-locating keeps that credential
  in `mlops` — it is **never copied into the `monitoring` namespace**. Prometheus
  scrapes it cross-namespace (an 8th scrape job discovering `mlops` endpoints) — which
  its **existing** cluster-wide read-only endpoints RBAC already permits — **no new
  grant**.
- **Minimal, not a DB-monitoring product.** Only the exporter's built-in
  cluster-level collectors run (`--auto-discover-databases=false`; **no** custom
  per-table/index query files). Enough for up/connections/size — deliberately not a
  per-table/query-plan monitor (the brief: "do not overcomplicate").

### 3. PVC-fill — the kubelet's volume stats (a scoped 6th scrape job)

The Postgres PVC-fill signal (`kubelet_volume_stats_used_bytes /
kubelet_volume_stats_capacity_bytes`) comes from the **kubelet's own `/metrics`
endpoint**, not from postgres-exporter or cAdvisor. PR 2 scrapes only the kubelet's
`/metrics/cadvisor` subresource, so PR 4 adds a **kubelet** scrape of `/metrics`
through the same API-server proxy, with the **same `nodes/proxy` RBAC** (no new
grant). It is **scoped at ingestion** with a `metric_relabel_configs` **keep** on
`kubelet_volume_stats_.*` — the kubelet endpoint is large, and this project needs
exactly that one family, so everything else is dropped (the measure-what-you-need
discipline; avoids duplicating cAdvisor).

### 4. Security — reviewed carefully (the brief's explicit ask)

- **Dedicated, least-privilege DB role.** The exporter authenticates as a
  purpose-made `mlflow_exporter` role granted **only** the built-in **`pg_monitor`**
  role — read access to Postgres' statistics views (`pg_stat_*`, database size) and
  **nothing else**: no table data, no writes. It is **not** the `mlflow` application
  role. It is created **out-of-band** (the never-committed pattern of
  `mlflow-db-credentials`), with the exact SQL in
  [`postgres-exporter-secret.example.yaml`](../../k8s/base/mlflow/postgres-exporter-secret.example.yaml),
  and carries a `statement_timeout` so a monitoring query can never wedge the DB.
- **No credentials in config or metrics.** The password is delivered **only** as an
  env var from a `Secret` (`DATA_SOURCE_PASS` ← `secretKeyRef`) — never in argv,
  never in a ConfigMap, never in the exposed `/metrics`. The **split**
  `DATA_SOURCE_URI` form carries host/port/db only, so the password is never part of
  a DSN string; the **single-target** env form (not the multi-target `?target=` API)
  means no DSN is ever passed as a scrape query parameter. `k8s/validate.py` asserts
  the password is a `secretKeyRef` and the URI has no embedded `user:pass@`.
- **blackbox carries no credentials at all** (it only makes an outbound GET).
- **Fleet hardening on both exporters.** Non-root uid 65534, drop `ALL`, no
  privilege escalation, seccomp `RuntimeDefault`, **read-only root filesystem**, and
  **no API token** (neither calls the Kubernetes API). blackbox satisfies the
  monitoring stack's contract; postgres-exporter satisfies the `mlops` namespace's
  `restricted` Pod Security Standard. Both are covered by the extended
  `k8s/validate.py` (§ 6).
- **In-cluster traffic is not TLS-terminated** (exporter⇄DB, blackbox⇄MLflow) — the
  same posture as the existing MLflow⇄Postgres hop. Adding mTLS would require a
  service mesh and is out of scope; recorded, not hidden.

### 5. What is *not* new — the run-level signals PR 2 already covers

MLflow **available replicas / readiness / restarts** and Postgres **StatefulSet
ready / restarts** come from **kube-state-metrics**; both workloads' **CPU / memory**
come from **cAdvisor** — all deployed in PR 2 and scraped cluster-wide. PR 4 adds
**no** component for these; it **documents the exact queries**
([`docs/observability.md`](../observability.md) § 3 Layers 3–4,
[`docs/monitoring-operations.md`](../monitoring-operations.md) § 4). This keeps the
PR honest: the genuinely new components are the two depth exporters and the scoped
kubelet scrape, nothing more.

### 6. Validation

The extended `k8s/validate.py` asserts the PR-4 contract: the scrape config wires
`blackbox-mlflow-health`, `postgres-exporter`, and the `kubelet_volume_stats_*`
keep; the blackbox job targets MLflow **`/health`**; blackbox-exporter's
Deployment/Service/http-module render (monitoring pass); and postgres-exporter's
Deployment/Service render with its **password from a Secret** and **no credential in
`DATA_SOURCE_URI`** (mlops pass). `kustomize build` + `kubeconform` (CI) cover render
+ schema; this script covers the contract. **Static only** — it proves the manifests
declare the contract, not that the exporters scrape (runtime proof is PR 6).

## Alternatives Considered

- **postgres-exporter as a StatefulSet sidecar.** Reasonable and slightly fewer
  objects, but loses the independent `pg_up 0` signal when the DB pod is down and
  couples the exporter's lifecycle to the DB. Chose a separate Deployment for
  independent observability (§ 2); the sidecar remains a documented option.
- **postgres-exporter in the `monitoring` namespace.** Rejected: it would require
  copying the DB credential Secret into `monitoring`, widening the credential's blast
  radius. Co-locating with the DB keeps the secret in `mlops` (§ 2).
- **Reuse the `mlflow` DB role for the exporter.** Rejected: violates least
  privilege — the exporter would hold the application role's full table access. A
  dedicated `pg_monitor`-only role is the security-correct choice (§ 4).
- **A full DB-monitoring product / per-table & query-plan collectors.** Rejected
  (the brief): overkill for a single-writer metadata DB; the built-in cluster-level
  collectors answer up/connections/size (§ 2).
- **An app-side MLflow `/metrics` exporter for RED metrics.** Deferred (ADR-028
  § 6): blackbox `/health` + cAdvisor answer the operational question now; deep
  request metrics are a separately justified follow-on.
- **Probing a heavier MLflow path (e.g. an experiments API) for a "deeper" check.**
  Rejected: it would add real load and couple the probe to app internals; `/health`
  is the stable, load-free contract (§ 1). The DB side is covered independently by
  postgres-exporter.

## Consequences

**Positive**

- **Layer 3 and Layer 4 depth exist**, hardened and statically validated: MLflow
  availability/latency (blackbox), Postgres up/connections/size (exporter), and the
  high-value **PVC-fill** signal (kubelet), on top of the run-level signals PR 2
  already provides — with **no live component** and therefore **no cost or attack
  surface** until deliberately deployed.
- **Least-privilege by construction:** a dedicated read-only DB role, credentials
  only ever in a Secret, no new Prometheus RBAC, and both exporters fully hardened.
- **Minimal and legible** — two small exporters + one scoped kubelet scrape, matching
  the hand-written, measure-what-you-need posture of ADR-029.

**Negative / trade-offs**

- **`/health` is a shallow MLflow check** — server-up, not deep DB connectivity
  (the Postgres exporter covers the DB independently).
- **A new out-of-band credential** (the `mlflow_exporter` role + its Secret) is a new
  operator step and a new identity to rotate — the cost of doing DB monitoring
  securely rather than reusing the app role.
- **Two more moving parts** (blackbox, postgres-exporter) and **two more scrape
  jobs** (kubelet, the two exporters → eight total) on a deliberately lean stack.
- **postgres-exporter deploys with the mlops workload**, not the monitoring stack
  (it lives in the mlops base for credential locality). If Prometheus is absent the
  exporter is a tiny idle pod — its metrics are simply unscraped, harmless.
- **No NetworkPolicy yet** — the exporters serve unauthenticated `/metrics` reachable
  by any in-cluster pod (same posture as the rest of the stack, ADR-029 §
  Consequences); a scoped policy is the right follow-on once a NetworkPolicy-enforcing
  posture exists. postgres-exporter's `/metrics` exposes **DB statistics, never
  credentials or row data** (the `pg_monitor` scope), so the exposure is metrics, not
  secrets.
- **`pg_monitor` residual — the role can read live query text, though `/metrics`
  does not.** `pg_monitor` includes `pg_read_all_stats`, which lets the role read
  other sessions' `query` column in `pg_stat_activity`. The exporter's **built-in
  collectors do not scrape or expose that** (no custom query file, no
  `pg_stat_statements` — the extension is not loaded), so **nothing sensitive reaches
  `/metrics`**. The residual is second-order: if the `mlflow-postgres-exporter-credentials`
  Secret were exfiltrated, the holder could `psql` in and read live SQL text (not
  table data). This is the documented minimum role for the exporter and the accepted
  cost of least-privilege DB monitoring; recorded, not hidden.
- **In-cluster hops are not TLS-encrypted** (§ 4) — accepted, mesh is out of scope.

## What This Decision Does **Not** Imply

- **Nothing is deployed or proven by this PR.** Manifests + scrape config + validation
  + docs ship and pass **static** checks; a live scrape→query cycle (all targets Up,
  a forced MLflow-down driving `probe_success 0`, an induced PVC-fill firing an alert)
  is **runtime evidence for PR 6** (the project rule: *structurally valid ≠
  runtime-complete*).
- **Not deep MLflow request-level metrics** — availability only; RED metrics remain
  deferred (ADR-028 § 6).
- **Not a full database monitoring product** — up/connections/size via the built-in
  collectors, deliberately scoped.
- **Not production monitoring / not an SLO** — same validation-environment posture as
  ADR-028 § 7 (no HA, no long-term store, no SLO compliance claim).
- **Not alerting** — the alert *rules* that consume these signals (`MLflowDown`,
  `PostgresDown`, `PostgresPVCAlmostFull`, …) are **PR 5**; this PR makes the signals
  collectable.
