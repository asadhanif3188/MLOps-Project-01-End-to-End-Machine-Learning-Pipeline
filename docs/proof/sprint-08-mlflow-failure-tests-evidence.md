# Sprint 8 PR 11 — MLflow Outage Detection & Recovery Evidence (PENDING)

> **STATUS: NOT YET EXECUTED.** The MLflow outage-detection / recovery paths are
> covered by a **runtime harness**
> ([`k8s/tests/mlflow-failure/run.sh`](../../k8s/tests/mlflow-failure/run.sh)) that
> creates a safe, reversible outage on a real cluster, but it has **not yet been run
> against a live EKS cluster**. This file is the **checklist to complete that
> capture** on the next enforcing cluster. Deferring is safe: **no reliability
> behaviour changed in this PR** — it only *observes* the current behaviour and
> records recommendations for PR 13 (§ 8).

> **⏳ Capture this TOGETHER with the other pending Sprint 8 live-EKS captures.** Four
> Sprint 8 proofs are now `PENDING` on the next live cluster:
> - [sprint-08-network-policy-runtime-evidence.md](sprint-08-network-policy-runtime-evidence.md) — PR 7, allowed/denied network paths;
> - [sprint-08-sbom-provenance-evidence.md § 4b](sprint-08-sbom-provenance-evidence.md#4b-operator-checklist-run-on-the-next-enforcing-cluster-session) — PR 9, push → immutable digest → verify;
> - [sprint-08-dataset-failure-tests-evidence.md](sprint-08-dataset-failure-tests-evidence.md) — PR 10, dataset unavailable + checksum mismatch;
> - **this doc** — PR 11, MLflow outage detection & recovery.
>
> Standing up EKS is the billable part (`provision → prove → destroy`,
> [ADR-020](../decisions/ADR-020-cloud-lifecycle-cost-control.md)); each check costs
> only minutes and all four need the **same deployed workload**. So on the **next
> cluster session, do all FOUR in one run** to amortise the cluster cost — deploy the
> workload once, then run this MLflow-outage harness *and* the netpol harness *and* the
> PR 9 digest verification *and* the PR 10 dataset-failure harness against the same live
> cluster before teardown.

**Design of record:** [ADR-026](../decisions/ADR-026-in-cluster-mlflow-platform.md)
(in-cluster MLflow platform: server + PostgreSQL + S3) ·
[ADR-031](../decisions/ADR-031-mlflow-postgres-monitoring.md) (MLflow/Postgres
monitoring — the blackbox `/health` probe) ·
[ADR-033](../decisions/ADR-033-alerting.md) (alerting) ·
**Tracking boundary:** [`src/tracking.py`](../../src/tracking.py) ·
**Gate:** `wait-for-mlflow` init container ([`k8s/base/job.yaml`](../../k8s/base/job.yaml)) ·
**Harness:** [`k8s/tests/mlflow-failure/run.sh`](../../k8s/tests/mlflow-failure/run.sh) ·
**Runbook:** [docs/alerting.md#mlflowdown](../alerting.md#mlflowdown)

---

## What is being proven

That an MLflow tracking-server outage is **visible** (an availability signal drops and
an alert fires), **diagnosable** (a runbook maps the signal to the cause), and
**recoverable** (restoring MLflow clears the alert with **no data loss**) — and that
the pipeline's behaviour during the outage is understood and appropriate.

**The outage is created safely.** The harness scales the **stateless** `mlflow`
Deployment to zero — a fully reversible mutation that **never touches** the
`mlflow-postgres` StatefulSet, its PVC, or the S3 artifact bucket. PostgreSQL metadata
and S3 artifacts therefore persist *by construction*. An `EXIT`/`INT`/`TERM` trap
restores the original replica count on **any** exit path, so the harness can never
leave MLflow down.

> **No reliability behaviour is changed in this PR.** Where the current behaviour is a
> candidate for improvement, it is **documented** as a PR 13 recommendation (§ 8), not
> fixed here. In particular **no retry is added** — per the task, "retry forever" is
> explicitly not introduced.

## Prerequisites

- [ ] A live cluster (EKS via `terraform apply`) with the mlops workload **and** the
      in-cluster MLflow platform deployed:
      `kubectl apply -k k8s/overlays/<aws|local>`.
- [ ] The monitoring stack deployed (Prometheus + blackbox-exporter + Grafana) for the
      `probe_success` / `MLflowDown` / dashboard evidence:
      `kubectl apply -k k8s/monitoring/overlays/<aws|local>`.
- [ ] A healthy baseline first — a **previous** pipeline run visible in MLflow, so
      "runs persist across the outage" is provable:
      `kubectl -n mlops wait --for=condition=complete job/mlops-pipeline --timeout=600s`.

## Steps

- [ ] **1. Baseline** — run `k8s/tests/mlflow-failure/run.sh`; confirm its baseline
      section is green (MLflow Ready, `mlflow-postgres` Ready, Service Endpoints
      present). Capture `probe_success{job="blackbox-mlflow-health"}` == `1`,
      `MLflowDown` **not** firing, and a screenshot of a previous run in the MLflow UI
      (`kubectl -n mlops port-forward svc/mlflow 5000:5000`).
- [ ] **2. Outage** — the harness scales `deploy/mlflow` → 0. Capture
      `kubectl -n mlops get deploy mlflow` (0/0) and
      `kubectl -n mlops get endpoints mlflow` (no addresses).
- [ ] **3. Detection** — after ≥5m capture `probe_success` == `0` and `MLflowDown`
      **firing** (Prometheus `/alerts`); screenshot the Grafana "MLflow Platform
      Health" availability gauge (red) with the **PostgreSQL panels still green**.
- [ ] **4. Pipeline during outage** — the harness submits a throwaway Job. Capture the
      pod: `fetch-dataset` Terminated **exit 0**; `wait-for-mlflow` Terminated
      **non-zero** with `MLflow not ready after …` in its logs; the `pipeline`
      container **never Running** (`kubectl -n mlops describe pod <pod>`).
- [ ] **5. Run-level signal** — apply the workload while MLflow is down so a **real**
      Job exhausts `backoffLimit`; confirm `PipelineJobFailed` fires **alongside**
      `MLflowDown`. Capture both from `/alerts` — the correlation is the diagnosis.
- [ ] **6. Diagnose via the runbook** — walk
      [docs/alerting.md#mlflowdown](../alerting.md#mlflowdown): `MLflowDown` firing →
      `kubectl -n mlops get deploy/mlflow` shows 0 replicas → scale back up.
- [ ] **7. Restore & recover** — the harness scales MLflow back and verifies Endpoints
      return and `mlflow-postgres` stayed Ready. Then capture recovery:
      `probe_success` back to `1`, `MLflowDown` **Resolved**, the **previous run still
      visible** in the MLflow UI (PostgreSQL persisted), S3 artifacts still listed, and
      a **new** unmodified pipeline Job Completes.
- [ ] **8. Teardown** (EKS) per [ADR-020](../decisions/ADR-020-cloud-lifecycle-cost-control.md).

## Record results here (fill in on execution)

Structured as the eight items the PR asks to return.

| # | Item | Result |
|---|---|---|
| 1 | **Outage method** | _pending_ — `kubectl scale deploy/mlflow --replicas=0` (stateless server only; PostgreSQL/S3 untouched; auto-restored by an EXIT trap) |
| 2 | **Detection** | _pending_ — `probe_success{job="blackbox-mlflow-health"}` `1 → 0`; Service Endpoints drain to empty |
| 3 | **Alert behaviour** | _pending_ — `MLflowDown` Pending → Firing at its 5m `for:` (warning); Resolves on restore |
| 4 | **Pipeline behaviour** | _pending_ — `fetch-dataset` **succeeds**; `wait-for-mlflow` gate **fails** (`MLflow not ready after …`); `pipeline` **never starts** → **no wasted computation**; the Job fails at the gate |
| 5 | **Diagnosis** | _pending_ — `MLflowDown` (+ `PipelineJobFailed` on a real Job) → runbook → `deploy/mlflow` at 0 replicas |
| 6 | **Recovery** | _pending_ — scale back → Ready + Endpoints return; alert Resolves; a new run Completes |
| 7 | **Persistence verification** | _pending_ — `mlflow-postgres` Ready throughout; previous runs still visible; S3 artifacts persist |
| 8 | **Candidate reliability improvements** | see § 8 below (mid-run tracking-failure handling; **not** implemented here) |

> Redact account IDs / bucket names / operator IPs / any secret material, per the
> Sprint 7 evidence convention. Paste the harness output and `kubectl`/PromQL captures
> below the table, add the Grafana / MLflow-UI screenshots under
> [`docs/screenshots/`](../screenshots/), and flip the STATUS banner to
> **EXECUTED &lt;date&gt;**.

## Where MLflow failure is caught (code walk)

Two independent guards make an MLflow outage fail *cleanly* rather than silently:

1. **Start-of-run gate** — the `wait-for-mlflow` init container
   ([`k8s/base/job.yaml`](../../k8s/base/job.yaml)) polls `/health` 60×5s and exits
   non-zero if MLflow never answers. The pipeline container is *gated* behind it, so a
   start-time outage stops the run **before any computation**.
2. **In-run tracking boundary** — every MLflow call goes through
   [`src/tracking.py`](../../src/tracking.py), which wraps `log_training_run` /
   `log_evaluation` in `try/except MlflowException` and re-raises a typed
   `TrackingError`. A mid-run outage therefore surfaces as a clear, typed stage failure
   (not a stack-trace or a silent skip).

> **Why no offline unit test is added here.** `tracking.py` imports `mlflow`, which is
> **deliberately absent** from the unit environment (the `stub_tracking` design,
> ADR-006 dec. 4 — unit tests never import MLflow, touch the network, or need
> credentials). Pinning the `MlflowException → TrackingError` contract offline would
> mean adding MLflow to the fast unit suite, contradicting that design. The contract is
> instead proven **at runtime** by this harness (the appropriate layer for an
> availability/outage test).

## Reliability analysis: is the current behaviour appropriate?

| Timing | Current behaviour | Appropriate? |
|---|---|---|
| **MLflow down at START** | `wait-for-mlflow` gate fails the Job after ~300s; **no compute runs** | **Yes** — deterministic, well-signalled fail-fast; nothing is wasted |
| **MLflow down MID-RUN** | `train`'s tracking call → `TrackingError` → whole Job fails; **the completed preprocess/split/train compute is wasted** (the model is persisted *by* the MLflow log) | **Defensible, but improvable** — see § 8 |

The **start-time** behaviour is good as-is and should not change. The **mid-run**
behaviour is the only real gap: a *transient* MLflow blip (e.g. a ~30–60s rolling
restart) mid-run currently fails the whole Job and discards work, even though the
Job's own `backoffLimit=2` comment already frames such blips as the thing retries
exist to absorb.

## § 8 — Candidate reliability improvements (for PR 13, NOT implemented here)

Recorded for PR 13; each is deliberately **out of scope** for this observation-only PR:

1. **Bounded retry around the in-run tracking calls** (not "retry forever"): a small,
   capped exponential back-off around `log_training_run` / `log_evaluation` in
   `src/tracking.py`, sized to ride out a rolling restart (a few attempts over
   ~60–90s), then fail as today. Keeps the fail-fast guarantee while absorbing a
   transient blip — directly addresses the wasted-compute case. **Explicitly bounded**,
   per the task's "do not implement retry forever."
2. **Durable model fallback / decouple persistence from tracking:** on a tracking
   failure, persist the fitted model + metrics to a durable location (the S3 artifact
   store or a PVC) and emit a warning, so a tracking outage never *loses the model*.
   Needs a reconcile-later story to restore MLflow lineage; a design trade-off, not a
   drop-in.
3. **Make the tracking policy an explicit decision (ADR):** is experiment tracking a
   *hard* requirement (fail the run if it can't be recorded — current behaviour,
   defensible for auditability/reproducibility) or *best-effort* (the model matters
   more than its metadata)? Record the choice so it is deliberate rather than emergent.

Recommendation: **(1)** is the smallest, highest-value change and preserves current
semantics; **(3)** should precede **(2)**.

## Observability sufficiency (assessed for this PR)

The current signals are **sufficient** to detect and diagnose an MLflow outage
end-to-end — **no instrumentation change was required or made**:

- **Availability:** the blackbox `probe_success{job="blackbox-mlflow-health"}` gauge
  (ADR-031) drops `1 → 0`; the `MLflowDown` alert fires after 5m.
- **Pipeline effect:** on a start-time outage, `mlops_pipeline_stage_success{stage=
  "fetch_dataset"}` = `1` while every later stage is **absent**, and the
  `wait-for-mlflow` gate logs `MLflow not ready after …` — together these localise the
  stall to the tracking gate. The run-level `PipelineJobFailed` fires on a real Job.
- **Diagnosis:** `MLflowDown` + `PipelineJobFailed` firing **together** is the
  signature of "the pipeline is blocked because MLflow is down"; the runbook
  ([docs/alerting.md#mlflowdown](../alerting.md#mlflowdown)) maps it to
  `kubectl get deploy/mlflow`.
- **Persistence:** `mlflow-postgres` StatefulSet readiness (asserted by the harness)
  and the MLflow UI (previous runs still visible) prove the durable store survived.

## Honesty boundary

- **Nothing on a live cluster has been executed for this PR.** Every "Result" cell is
  `pending`. The harness's static/syntax correctness and this analysis are the only
  things verified to date.
- The **`MLflowDown` firing** (5m `for:`) and the **`PipelineJobFailed` correlation**
  (real Job exhausting `backoffLimit`) require the live cluster; they are steps 3 and 5,
  not claims made here.
- The **mid-run** wasted-compute case is derived from the `src/tracking.py` code path,
  not from an injected live failure (it is timing-dependent). It is presented as
  analysis and a PR 13 candidate, not as an observed run result.
- Teardown cost/lifecycle follows [ADR-020](../decisions/ADR-020-cloud-lifecycle-cost-control.md).
