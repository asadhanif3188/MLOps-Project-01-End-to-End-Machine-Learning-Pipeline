# ADR-029: Monitoring foundation — a minimal, hand-written Prometheus stack

- **Status:** Accepted (design; manifests added, no components deployed in this PR)
- **Date:** 2026-08-20
- **Deciders:** Asad Hanif
- **Related:** [ADR-028 (Observability architecture — the design of record)](ADR-028-observability-architecture.md),
  [`docs/observability.md`](../observability.md),
  [`docs/monitoring-operations.md`](../monitoring-operations.md),
  [`k8s/monitoring/`](../../k8s/monitoring/),
  [`k8s/base/job.yaml`](../../k8s/base/job.yaml),
  [ADR-009 (Workload model — a Job, not a Deployment)](ADR-009-kubernetes-workload-model.md),
  [ADR-010 (Kubernetes security hardening)](ADR-010-kubernetes-security-hardening.md),
  [ADR-012 (Kubernetes manifest validation)](ADR-012-kubernetes-manifest-validation.md),
  [ADR-020 (Ephemeral cloud lifecycle & cost control)](ADR-020-cloud-lifecycle-cost-control.md),
  [ADR-026 (In-cluster MLflow platform)](ADR-026-in-cluster-mlflow-platform.md)

> **Scope.** ADR-028 ratified the observability *architecture* (why Prometheus/
> Grafana, the four-layer model, and how an ephemeral `Job`'s metrics stay
> queryable). This ADR records the *implementation* decisions for **Sprint 8 PR 2
> — the metrics foundation**: how the stack is packaged, what is deployed now
> versus deferred, the storage/retention choice, the RBAC model, and the one
> Pod Security exception node-exporter forces. It **adds version-controlled
> manifests only** — it deploys nothing (no live cluster; runtime proof is PR 6).
> It implements ADR-028's PR-2 acceptance criteria; it does not re-decide the
> architecture.

## Context

ADR-028 fixed *what* to observe and *why*. PR 2 must stand up the smallest
Prometheus-compatible foundation that makes the **Layer 1** (Kubernetes platform)
and **Layer 2** (the batch pipeline Job, via kube-state-metrics) signals in
[`docs/observability.md`](../observability.md) collectable — **without** Grafana
(PR 3), **without** MLflow/PostgreSQL depth (PR 4), and **without** alerts (PR 5).

Several implementation choices were open and are settled here: how to package the
stack in a repo that uses **hand-written Kustomize and no Helm**; where the
Prometheus TSDB lives on an **ephemeral, cost-controlled** cluster (ADR-020); how
to grant Prometheus and kube-state-metrics the API access they need while keeping
the fleet's least-privilege posture (ADR-010); and how to admit node-exporter,
which cannot satisfy the `restricted` Pod Security Standard the rest of the fleet
enforces.

## Decision

### 1. Package the stack as minimal, hand-written Kustomize manifests — not Helm

The stack is authored as plain Kubernetes manifests under
[`k8s/monitoring/`](../../k8s/monitoring/) (a **separate** kustomize root from the
`mlops` workload, with base + `local`/`aws` overlays), **not** installed from
`kube-prometheus-stack`, the Prometheus Operator, or any Helm chart.

Why:

- **It matches the repository's deployment model.** Every workload here is
  hand-written, hardened, and statically validated Kustomize (ADR-009/010/012);
  the project uses **no Helm** and no operators. Introducing a chart would add a
  whole toolchain and the CRD-heavy Prometheus Operator for a four-target scrape.
- **It honours "explicitly configure only what the project needs"** (the sprint
  brief's explicit instruction not to blindly import a packaged stack).
  `kube-prometheus-stack` ships dozens of components, hundreds of alert rules, and
  a large default scrape config — the opposite of the measured, minimal posture
  ADR-011/ADR-026 established. Hand-writing four scrape jobs is legible and
  reviewable line by line.
- **It keeps full control of hardening, RBAC, and resources** — the exact things
  this project validates in CI. A chart's rendered security context and RBAC would
  have to be audited and overridden anyway.

The cost is that component upgrades are manual (bump a pinned image tag) rather
than a chart version bump — acceptable for a handful of pinned images and
consistent with how the rest of the fleet is maintained.

### 2. Components now, and what is deferred

**Deployed by this PR (manifests):** a dedicated **`monitoring` namespace**;
**Prometheus** (scrape + local TSDB + PromQL/UI, ClusterIP); **kube-state-metrics**
(KSM — API-object state, the source of the Layer 2 batch-Job signals);
**node-exporter** (DaemonSet — Layer 1 node metrics); and a Prometheus scrape of
the kubelet's built-in **cAdvisor** (Layer 1 per-container usage). Four scrape
jobs, nothing more.

**Deferred to later Sprint 8 PRs, deliberately:** Grafana (PR 3);
blackbox-exporter + postgres-exporter for Layer 3/4 depth (PR 4); alert rules and
Alertmanager (PR 5); runtime evidence + operations runbook proof (PR 6). No
pipeline custom metrics and no Pushgateway (ADR-028 § 3 keeps ML semantics in
MLflow and per-stage timing deferred).

### 3. Ephemeral storage — `emptyDir` TSDB, short retention, no long-term store

Prometheus stores its TSDB on an **`emptyDir`** (not a PVC), with
`--storage.tsdb.retention.time=7d`, a `--storage.tsdb.retention.size=1GB` cap, and
an `emptyDir.sizeLimit` of 2Gi. There is **no remote_write / no Thanos/Cortex/
Mimir/AMP**.

Why (requirements 9–11; ADR-028 § 5, ADR-020):

- **Cost discipline on an ephemeral cluster.** A PVC is a standing EBS cost and an
  extra teardown step; `emptyDir` is free and vanishes with the pod. Long-term/
  remote storage is production capacity this project does not need.
- **Bounded footprint.** Time *and* size retention plus the volume `sizeLimit`
  mean the TSDB can never pressure a node.
- **Stated trade-off:** metrics are lost on pod restart/teardown. That is
  acceptable because runtime **evidence is captured while the stack is live**
  (PR 6); a modest PVC is the documented option (the `aws` overlay is the place to
  add a gp3 claim) if survive-a-restart is ever wanted.

### 4. RBAC and token discipline — read-only, token mounted only where the API is used

- Prometheus and KSM genuinely need the Kubernetes API (service discovery + the
  cAdvisor kubelet proxy; API-object reflection), so — **unlike the mlops
  workloads** — their ServiceAccount tokens **are** mounted. This is the scoped
  exception ADR-028 § 4 anticipated.
- Each is granted a **read-only `ClusterRole`** (`get`/`list`/`watch` only; no
  create/update/patch/delete, no `*`), scoped to exactly the discovery/scrape
  surface (Prometheus) or the object kinds behind the documented signals (KSM —
  and **not** secrets, HPAs, etc.).
- **node-exporter needs no API access**, so its token is **not** mounted (the
  fleet default). The invariant — *token mounted **iff** the SA is bound to a
  ClusterRole* — is enforced statically (see § 6).

### 5. The Pod Security exception — node-exporter forces a `privileged` namespace

node-exporter must read the node's `/proc`, `/sys`, and root filesystem to report
Layer 1 node metrics, which requires **read-only `hostPath`** volumes. `hostPath`
is forbidden by **both** the `restricted` and `baseline` Pod Security Standards,
and PSA is namespace-scoped (it cannot exempt one workload), so the `monitoring`
namespace is set to **`enforce: privileged`**. This is the **sole** reviewed
exception (requirements 7–8). It is kept narrow and compensated:

- The only relaxation is node-exporter's three **read-only** hostPath mounts.
  node-exporter is otherwise fully hardened — non-root (uid 65534), drop `ALL`,
  no privilege escalation, **not** privileged, **no** host namespaces
  (`hostNetwork`/`hostPID`/`hostIPC`), seccomp `RuntimeDefault`, read-only root FS.
  It is scraped on its **pod IP**, not the host network.
- Prometheus and KSM use **no** hostPath and are **restricted-equivalent** by
  explicit securityContext (they would pass `restricted` on their own); both even
  run with a read-only root filesystem, exceeding the pipeline Job's baseline
  (which must keep root writable for `dvc repro`, ADR-010).
- `warn`/`audit` stay at **`baseline`** so any regression *other* than
  node-exporter's known hostPath (a privileged container, a host namespace, a root
  user on any component) is still surfaced at admission.
- The real, CI-enforced control is the **extended `k8s/validate.py` monitoring
  contract** (§ 6): PSA is the cluster backstop; validate.py is the gate.

A residual, deliberately accepted, is that `privileged` PSA gives this namespace
**no live admission protection** — any pod applied here (not just node-exporter)
would be admitted regardless of its security posture; only `warn`/`audit` at
`baseline` would log it. On this single-operator, ephemeral validation cluster the
compensating controls above are proportionate. If this ever became a shared or
production cluster, the correct control is a **label-scoped admission policy**
(Kyverno / OPA Gatekeeper: "hostPath permitted only for pods labelled
`app.kubernetes.io/name: node-exporter`") rather than PSA `privileged` plus a
CI-time lint. That is out of scope here and recorded as future work.

Trade-off of avoiding `hostNetwork`/`hostPID`: node-exporter's network-device
metrics reflect the pod's network namespace rather than the host's. The node CPU/
memory/filesystem signals this project actually alerts on come from the read-only
`/proc`+`/sys`+rootfs mounts and are unaffected. Adopting host namespaces is a
deferred, separately justified change — not taken "because the upstream chart does".

### 6. Validation — a monitoring-aware `k8s/validate.py` pass

The mlops security contract (automount off everywhere, no hostPath, `restricted`
PSA) would wrongly fail this stack, so the monitoring render gets its **own**
static pass (`validate_monitoring()`), asserting the monitoring-appropriate
invariants: PSA labels present + version-pinned; every component hardened; hostPath
**only** on node-exporter and **only** read-only; **read-only RBAC**; the
token-iff-API invariant; ClusterIP-only exposure; and that the scrape config wires
KSM, node-exporter, and cAdvisor. The mlops pass additionally asserts the pipeline
Job now sets a positive **`ttlSecondsAfterFinished`** (§ 7). `kustomize build` and
`kubeconform` (CI) cover render + schema; this script covers the contract.

### 7. Job retention — `ttlSecondsAfterFinished` honours the queryability contract

The pipeline Job ([`k8s/base/job.yaml`](../../k8s/base/job.yaml)) now sets
**`ttlSecondsAfterFinished: 3600`**. KSM reflects the persistent Job/Pod object, so
the last-run success/duration/OOM series stay scrapable **only while the finished
Job lingers** (ADR-028 § 3). One hour comfortably outlives many 30s scrape
intervals — the gauges remain live for an ample post-completion evidence read —
and then the Job auto-reaps so finished Jobs do not accumulate. Already-scraped
samples persist in the TSDB for the whole retention window regardless.

**Re-run tension (recorded, not hidden):** the Job has a fixed `metadata.name`, so
re-submitting within the hour still needs the finished Job deleted first
(`kubectl delete job mlops-pipeline`). Moving to `generateName` / delete-before-
recreate would remove that friction but changes the run UX and the "last-run" gauge
semantics; it is a **deferred option**, to be decided with real re-run experience
in PR 6, not adopted blindly now.

## Alternatives Considered

- **kube-prometheus-stack / Prometheus Operator (Helm).** Rejected for PR 2 (see
  § 1): introduces Helm + CRDs + an operator and a large default configuration for
  a four-target need, contradicting the minimal, hand-written, explicitly-configured
  posture the brief and ADR-011/026 require. Reasonable to revisit only if the
  monitoring surface grows to justify an operator.
- **A PVC-backed Prometheus TSDB.** Rejected as the default (§ 3): a standing EBS
  cost and teardown step on an ephemeral cluster (ADR-020). Retained as the
  documented `aws`-overlay option where survive-a-restart matters.
- **Amazon Managed Prometheus / remote_write.** Rejected: a long-term/remote store
  is out of scope on an ephemeral validation cluster (ADR-028 § 5) and adds cloud
  cost + lock-in against the self-hosted, portable posture (ADR-026/027).
- **node-exporter with `hostNetwork`/`hostPID` (the common chart default).**
  Rejected (§ 5): broadens the Pod Security exception beyond what the required
  signals need; pod-IP scraping plus read-only hostPath is sufficient and tighter.
- **Dropping node-exporter to keep the namespace `restricted`.** Rejected: cAdvisor
  gives per-container metrics but **not** node-level memory/filesystem/CPU (Layer 1),
  which ADR-028 and requirement 3 name explicitly. The narrow, documented exception
  is the right trade, not omitting the signal.

## Consequences

**Positive**

- A **version-controlled, hardened, statically-validated** metrics foundation
  exists, collectable for Layer 1 and the Layer 2 batch-Job signals, with **no
  live component** and therefore **no cost or attack surface** until deliberately
  deployed.
- The stack is **minimal and legible** — four scrape jobs, read-only RBAC, ephemeral
  storage — and matches the repo's existing hand-written, cost-controlled model.
- The one security relaxation (node-exporter hostPath) is **explicit, narrow, and
  compensated**, and the token-iff-API and read-only-RBAC invariants are enforced
  in CI.

**Negative / trade-offs**

- The `monitoring` namespace is **`privileged`**, not `restricted` — a real (if
  narrow and documented) departure from the fleet baseline, mitigated by per-pod
  hardening and the validate.py contract.
- **Manual component upgrades** (pinned image bumps) rather than a chart version.
- **Ephemeral metrics** (emptyDir) — history is lost on restart/teardown; evidence
  must be captured live (PR 6).
- **No node network metrics from the host netns** (pod-IP scraping trade-off, § 5).
- **No NetworkPolicy yet.** The three components serve unauthenticated HTTP
  (Prometheus `:9090`, KSM `:8080`, node-exporter `:9100`), reachable by any pod
  that can route to them. This matches the repo's current posture (no workload
  defines a NetworkPolicy, and NetworkPolicy enforcement is CNI-dependent — the EKS
  VPC CNI and local CNIs do not all enforce it by default), so adding one now could
  give false assurance. A default-deny + scoped-allow policy in `monitoring` is the
  right follow-on hardening once a NetworkPolicy-enforcing posture exists; recorded
  as future work, not silently omitted.

## What This Decision Does **Not** Imply

- **Nothing is deployed by this PR.** It adds manifests, validation, and docs; no
  Prometheus, KSM, or node-exporter runs, and no cluster is contacted. Full
  platform observability is **not** claimed — runtime proof is PR 6.
- **Not Grafana, exporters, or alerts** — those are PRs 3–5; this is metrics core.
- **Not production monitoring** — validation-environment sizing and retention, no
  HA, no long-term store, no SLO compliance claim (ADR-028 § 7).
- **Not a commitment to `generateName`** for the Job — the re-run trade-off is
  recorded and deferred (§ 7).
