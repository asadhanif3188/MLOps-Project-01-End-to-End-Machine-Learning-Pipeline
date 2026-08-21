# Sprint 8 PR 10 — Dataset Failure-Path Runtime Evidence (PENDING)

> **STATUS: NOT YET EXECUTED.** The dataset availability & integrity failure paths
> are covered by **unit** tests ([`tests/unit/test_fetch_dataset.py`](../../tests/unit/test_fetch_dataset.py))
> and a **runtime harness** ([`k8s/tests/dataset-failure/run.sh`](../../k8s/tests/dataset-failure/run.sh)),
> but the controlled failures have not yet been exercised on a **live EKS cluster**.
> This file is the **checklist to complete that capture** — do it the next time an
> enforcing cluster exists. Deferring is safe: no reliability behaviour changed in
> this PR (the only code change is an additive log line — see § 8).

> **⏳ Capture this TOGETHER with the Sprint 8 PR 7 + PR 9 runtime evidence.** All
> three are `PENDING` on the next live cluster:
> - [sprint-08-network-policy-runtime-evidence.md](sprint-08-network-policy-runtime-evidence.md) — allowed/denied network paths;
> - [sprint-08-sbom-provenance-evidence.md § 4b](sprint-08-sbom-provenance-evidence.md#4b-operator-checklist-run-on-the-next-enforcing-cluster-session) — push → immutable digest → verify;
> - **this doc** — dataset unavailable + checksum mismatch failure paths.
>
> Standing up EKS is the billable part (`provision → prove → destroy`,
> [ADR-020](../decisions/ADR-020-cloud-lifecycle-cost-control.md)); all three need only
> the same deployed workload and cost minutes each. So on the **next cluster session,
> do all three in one run** to amortise the cluster cost — deploy the workload once,
> then run this harness *and* the netpol harness *and* the digest verification against
> the same live cluster before teardown.

**Design of record:** [ADR-027](../decisions/ADR-027-s3-dataset-runtime-retrieval.md)
(runtime S3 retrieval + integrity pin) · [ADR-030](../decisions/ADR-030-pipeline-operational-metrics.md)
(per-stage operational metrics) · [ADR-033](../decisions/ADR-033-alerting.md)
(alerting) ·
**Retrieval code:** [`src/fetch_dataset.py`](../../src/fetch_dataset.py) ·
**Harness:** [`k8s/tests/dataset-failure/run.sh`](../../k8s/tests/dataset-failure/run.sh) ·
**Runbook:** [docs/alerting.md#pipelinejobfailed](../alerting.md#pipelinejobfailed)

---

## What is being proven

The reliability contract for the dataset boundary: a bad dataset must **stop the
pipeline before training**, with a **clear error** and an **operational signal** —
never a silent start against missing or corrupt data. Two controlled failures, on
the **same deployed workload**:

- **Scenario A — dataset UNAVAILABLE.** The `fetch-dataset` init container is pointed
  at a non-existent object key (a safe, fully reversible stand-in for a missing object
  or denied access). The S3 GET fails; the init container exits non-zero; the Job
  retries to `backoffLimit=2` and reaches its terminal **Failed** condition. The
  pipeline (`preprocess → split → train → evaluate`) never starts.
- **Scenario B — checksum MISMATCH.** The object is retrieved, but `DATASET_SHA256` is
  overridden to a wrong-but-well-formed digest. The integrity gate rejects it and the
  Job fails — again **before** training. This is a **deterministic** failure: a
  re-download yields the same bytes and the same mismatch, so it is **never retried
  away** and must fail fast. **No retries are added for checksum mismatch** (that would
  be a correctness regression; resilience/retry work is PR 13).

> **Method note — why the harness never touches the committed manifests.** For each
> scenario the harness renders the *deployed* Job, applies one narrow env override to
> the `fetch-dataset` init container, and submits it under a throwaway name; "restore"
> is simply deleting the throwaway Job — there is nothing to roll back. The
> `PipelineJobFailed` **alert** needs the override on the **real** Job (so it exhausts
> `backoffLimit`); that step is spelled out below and is the only one that touches the
> real Job, reverted by re-applying the overlay.

## Prerequisites

- [ ] A live cluster (EKS via `terraform apply`, or a local cluster) with the mlops
      workload deployed and the `fetch-dataset` init container present:
      `kubectl apply -k k8s/overlays/<aws|local>`.
- [ ] The monitoring stack deployed (Pushgateway + Prometheus + Grafana) for the
      metric / alert / dashboard evidence:
      `kubectl apply -k k8s/monitoring/overlays/<aws|local>`.
- [ ] A healthy baseline run first (so "before" is green):
      `kubectl -n mlops wait --for=condition=complete job/mlops-pipeline --timeout=600s`.

## Steps

- [ ] **1. Run the harness** (both scenarios, throwaway Jobs):
      `k8s/tests/dataset-failure/run.sh` → expect `RESULT: PASS`.
- [ ] **2. Scenario A observation** — capture, for the throwaway Job's pod:
      `kubectl -n mlops get pod <pod> -o wide`, the init-container status
      (`kubectl -n mlops describe pod <pod>` → `fetch-dataset` Terminated, non-zero),
      and `kubectl -n mlops logs <pod> -c fetch-dataset` (the `Failed to download …`
      error). Confirm the `pipeline` container is `Waiting/PodInitializing`, never
      `Running`.
- [ ] **3. Scenario B observation** — same capture; the log shows `Dataset retrieved:
      … (N bytes); verifying integrity` **then** `integrity check failed: expected …,
      got …` (retrieve-then-reject ordering). Confirm training never begins.
- [ ] **4. Metrics** — query Prometheus / the Pushgateway:
      `mlops_pipeline_stage_success{stage="fetch_dataset"}` == `0`, and every later
      stage (`preprocess`/`split`/`train`/`evaluate`) **ABSENT** (the per-run reset +
      early failure ⇒ the pipeline "never got there", ADR-030).
- [ ] **5. Alert** — apply the SAME override to the **real** Job so it exhausts
      `backoffLimit`; after the terminal Failed condition (+2m `for:`) confirm
      `PipelineJobFailed` is `firing` (Prometheus `/alerts`). Capture
      `kubectl -n mlops get job mlops-pipeline` (`FAILED`/`Failed=True`).
- [ ] **6. Grafana** — screenshot "MLOps Pipeline Operations": the **Dataset
      retrieval** panel = `0` (red), **Stage success — last run** shows only
      `fetch_dataset=0`, **Pipeline Job outcomes** reflects the failure.
- [ ] **7. Diagnose via the runbook** — walk
      [docs/alerting.md#pipelinejobfailed](../alerting.md#pipelinejobfailed): logs →
      which stage `mlops_pipeline_stage_success==0` → `fetch_dataset`; distinguish
      unavailable vs. checksum by the log message (§ *Dataset retrieval failures*).
- [ ] **8. Recover** — restore and prove a healthy run:
      `kubectl -n mlops delete job mlops-pipeline && kubectl apply -k k8s/overlays/<aws|local>`
      then `kubectl -n mlops wait --for=condition=complete job/mlops-pipeline --timeout=600s`.
      Confirm `mlops_pipeline_stage_success` is `1` for all five stages and the alert
      clears.
- [ ] **9. Teardown** (EKS) per [ADR-020](../decisions/ADR-020-cloud-lifecycle-cost-control.md).

## Record results here (fill in on execution)

Structured as the eight items the PR asks to return.

| # | Item | Result |
|---|---|---|
| 1 | **Exact failure method** | _pending_ (A: `DATASET_S3_URI` → non-existent key; B: `DATASET_SHA256` → `0…0`) |
| 2 | **Failure results** | _pending_ (init-container exit code + Job status, both scenarios) |
| 3 | **Metrics / alerts** | _pending_ (`stage_success{fetch_dataset}=0`, later stages absent; `PipelineJobFailed` firing) |
| 4 | **Diagnosis** | _pending_ (runbook walk: logs → failing stage → root cause) |
| 5 | **Recovery** | _pending_ (config/override restored; nothing to roll back for throwaway runs) |
| 6 | **Final healthy run** | _pending_ (Job Complete; all five stages `success=1`; alert cleared) |
| 7 | **Evidence files** | _pending_ (harness output, `kubectl` captures, Grafana screenshots — paste below) |
| 8 | **Reliability issue discovered** | _pending_ (none expected; record any) |

> Redact account IDs / bucket names / operator IPs / any secret material, per the
> Sprint 7 evidence convention. Paste the harness output and `kubectl`/PromQL captures
> below the table, add the Grafana screenshots under
> [`docs/screenshots/`](../screenshots/), and flip the STATUS banner at the top to
> **EXECUTED &lt;date&gt;**.

## Observability sufficiency (assessed for this PR)

The current signals are sufficient to diagnose **both** scenarios end-to-end:

- **Init-container status** distinguishes "ran and failed" from "never ran".
- **Structured logs** are the root-cause layer and carry **distinct** messages —
  `Failed to download …` (unavailable) vs `integrity check failed: expected …, got …`
  (mismatch) — so the two failure modes are unambiguous from the logs alone.
- **Metrics** (`mlops_pipeline_stage_success{stage="fetch_dataset"}=0`, later stages
  absent) and the **`PipelineJobFailed` alert** provide the run-level operational
  signal; the **Dataset retrieval** Grafana panel surfaces it visually.

One **smallest-possible** instrumentation improvement was made (§ 8 of the return, and
[`src/fetch_dataset.py`](../../src/fetch_dataset.py)): an additive INFO log
(`Dataset retrieved: … (N bytes); verifying integrity`) emitted **after** a successful
download and **before** the integrity gate. Rationale: Scenario B must prove
*retrieve-then-reject* ordering; before this line, a mismatch was only distinguishable
from a failed retrieval by the *absence* of a later log. It changes **no** retrieval or
failure behaviour, adds **no** retry, and adds **no** metric cardinality — pure
observability. No broader reliability changes were made here; those belong to PR 13.

## Honesty boundary

- **Nothing on a live cluster has been executed for this PR.** Every "Result" cell
  above is `pending`. The unit tests (offline, with an injected client) and the
  harness's static/syntax correctness are the only things verified in CI to date.
- The `PipelineJobFailed` **alert firing** requires the real Job to exhaust
  `backoffLimit`; it is step 5, not something claimed here.
- Teardown cost/lifecycle follows [ADR-020](../decisions/ADR-020-cloud-lifecycle-cost-control.md).
