# Operational Runbooks

Failure-diagnosis and recovery procedures for the MLOps platform, distilled from the
**Sprint 8 live-EKS failure-injection campaign** (2026-08-21) — every symptom, command,
and recovery check below was observed on a real cluster, not asserted. The canonical
runtime record is
[proof/sprint-08-live-eks-evidence.md](../proof/sprint-08-live-eks-evidence.md).

Where [alerting.md](../alerting.md) is organised **by alert** (the target of each
alert's `runbook_url`), these runbooks are organised **by failure mode** — "I see *this*,
what do I do?" The two layers cross-link: each alert points here for the deep procedure,
and each runbook points back to the alert that pages for it.

> **Scope — an ephemeral, single-operator validation platform.** This is not a
> production operations manual. The platform is `provision → prove → destroy`
> ([ADR-020](../decisions/ADR-020-cloud-lifecycle-cost-control.md)); there is **no
> Alertmanager routing** (firing alerts surface on Prometheus's own `/alerts`, not a
> pager), **no on-call rotation**, and **no HA**. "Escalation" here means *where the
> deeper design decision lives*, not a human hand-off. See each runbook's **Escalation /
> known limitations** section.

---

## The runbooks

| # | Runbook | Covers | Primary alert | Evidence |
|---|---------|--------|---------------|----------|
| 1 | [Platform health](platform-health.md) | Is the platform healthy right now? The first-response triage that routes to the others. | *(all)* | [live-EKS §2](../proof/sprint-08-live-eks-evidence.md#2-results-summary) |
| 2 | [Pipeline failure](pipeline-failure.md) | The `mlops-pipeline` Job reached its terminal Failed condition. | `PipelineJobFailed` | [PR 10](../proof/sprint-08-dataset-failure-tests-evidence.md), [live-EKS §6](../proof/sprint-08-live-eks-evidence.md#6-pr-10--12--failure-paths--alerts) |
| 3 | [Dataset retrieval failure](dataset-retrieval-failure.md) | `fetch-dataset` cannot download the object (missing key, denied, unreachable). | `PipelineJobFailed` | [PR 10 Scenario A](../proof/sprint-08-dataset-failure-tests-evidence.md) |
| 4 | [Dataset checksum / integrity failure](dataset-integrity-failure.md) | The object is retrieved but its SHA-256 does not match the pin. | `PipelineJobFailed` | [PR 10 Scenario B](../proof/sprint-08-dataset-failure-tests-evidence.md) |
| 5 | [MLflow unavailable](mlflow-unavailable.md) | The tracking server is down; runs cannot log. | `MLflowDown` | [PR 11](../proof/sprint-08-mlflow-failure-tests-evidence.md), [live-EKS §7](../proof/sprint-08-live-eks-evidence.md#7-pr-11--mlflow-outage-detection--recovery-8-items) |
| 6 | [PostgreSQL failure](postgresql-failure.md) | The metadata DB is unreachable, near-full, or under memory pressure. | `PostgresDown`, `PostgresPVCAlmostFull`, `PostgresMemoryHigh` | [PR 11 persistence](../proof/sprint-08-mlflow-failure-tests-evidence.md), [live-EKS Finding 2](../proof/sprint-08-live-eks-evidence.md#3-findings--4-real-defects-the-live-run-surfaced-all-fixed) |
| 7 | [OOMKilled](oomkilled.md) | A pipeline pod hit its memory limit (exit 137). | `PipelineJobOOMKilled`, `MLflowMemoryHigh`, `PostgresMemoryHigh` | [PR 12 Scenario A](../proof/sprint-08-resource-failure-tests-evidence.md), [live-EKS Finding 4](../proof/sprint-08-live-eks-evidence.md#3-findings--4-real-defects-the-live-run-surfaced-all-fixed) |
| 8 | [Crash / restart behaviour](crash-restart.md) | Deterministic crash-loop, backoff exhaustion, and `CrashLoopBackOff`. | `KubePodCrashLooping` | [PR 12 Scenario B](../proof/sprint-08-resource-failure-tests-evidence.md), [PR 13 retry](../proof/sprint-08-reliability-hardening-evidence.md) |

---

## The template

Every runbook follows the same nine sections, so an operator always knows where to look:

| Section | Answers |
|---------|---------|
| **Purpose** | What this runbook is for, in one line. |
| **Symptoms** | What you actually see (`kubectl`/dashboard/alert). |
| **Detection** | The alert and/or query that surfaces it. |
| **Initial checks** | The fast, safe, read-only commands to run first. |
| **Diagnosis** | How to narrow the symptom to a root cause. |
| **Likely causes** | The ranked candidate causes, with how to tell them apart. |
| **Remediation** | The fix — destructive steps carry a ⚠️ warning. |
| **Recovery verification** | **How to prove the platform is healthy again** — never ends at "restart the pod". |
| **Escalation / known limitations** | Where the design decision lives; what this platform deliberately does not do. |

---

## Conventions used throughout

- **Namespaces.** The workload (`mlops-pipeline` Job, MLflow, PostgreSQL,
  `postgres-exporter`) lives in **`mlops`**; the monitoring stack (Prometheus, Grafana,
  KSM, exporters) lives in **`monitoring`**.
- **Overlay placeholder.** `<aws|local>` means *pick the overlay for your cluster* —
  `k8s/overlays/aws` on EKS, `k8s/overlays/local` on Docker Desktop / kind / minikube.
  The monitoring stack has its own roots: `k8s/monitoring/overlays/<aws|local>`.
- **Redaction.** Account IDs, bucket names, and operator IPs are templated (`<ACCOUNT>`,
  `<YOUR_IP>`) per the Sprint 7 evidence convention — never commit the real values.
- **Reaching Prometheus / Grafana.** Both are internal-only; port-forward them:
  ```bash
  kubectl -n monitoring port-forward svc/prometheus 9090:9090   # UI: http://localhost:9090
  kubectl -n monitoring port-forward svc/grafana    3000:3000   # UI: http://localhost:3000
  ```
- **No Alertmanager.** "The alert fires" means it appears on Prometheus's own
  `/alerts` and in `ALERTS{alertstate="firing"}` — there is no external notifier
  ([alerting.md § Known limitations](../alerting.md#known-limitations)).
- **The universal recovery pattern.** A finished Job is immutable, so recovery is
  *delete-then-reapply*, then **prove** completion:
  ```bash
  kubectl -n mlops delete job mlops-pipeline           # ⚠️ discards the failed Job object
  kubectl apply -k k8s/overlays/<aws|local>            # (on EKS: scripts/render-cloud-manifests.sh --apply)
  kubectl -n mlops wait --for=condition=complete job/mlops-pipeline --timeout=600s
  ```

---

## Related documentation

- [Alerting](../alerting.md) — the eight alert rules, thresholds, and the per-alert `runbook_url` targets
- [Monitoring Operations](../monitoring-operations.md) — deploy the stack, reach Prometheus/Grafana, run a query
- [Kubernetes Operations](../kubernetes-operations.md) — local day-2 ops, the troubleshooting matrix
- [Cloud Operations](../cloud-operations.md) — the EKS provision → prove → destroy lifecycle and safe teardown
- [Observability & Operations](../observability.md) — the four-layer signal architecture
- [Sprint 8 — Live-EKS Evidence](../proof/sprint-08-live-eks-evidence.md) — the campaign these runbooks are built from
