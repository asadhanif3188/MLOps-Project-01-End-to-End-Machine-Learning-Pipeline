# Sprint 8 PR 13 — Reliability Hardening Evidence

**Objective:** Act on the PR 10–12 failure-test evidence — implement only the fixes it
justifies, decline the rest with reasons, and re-verify the healthy path and the
unchanged fail-fast behaviours.

**Status:** ✅ **COMPLETE** — implemented + verified on local Docker Desktop Kubernetes 2026-08-21

**Design of record:** [ADR-037](../decisions/ADR-037-pipeline-reliability-hardening.md) ·
**Code:** [`src/retry.py`](../../src/retry.py), [`src/tracking.py`](../../src/tracking.py) ·
**Tests:** [`tests/unit/test_retry.py`](../../tests/unit/test_retry.py)

---

## 1. Evidence-to-fix mapping

| # | Observed issue (evidence) | Current behaviour | Desired behaviour | Proposed fix | Decision |
|---|---|---|---|---|---|
| E1 | **Mid-run MLflow blip wastes compute** — PR 11 § "Reliability analysis" + § 8 #1: a transient outage (e.g. ~30-60s rolling restart of the stateless `mlflow` Deployment) during `train`/`evaluate` raises `TrackingError` and fails the whole Job, **discarding completed preprocess/split/train work**. | In-run tracking call fails immediately on the first `MlflowException`. | Ride out a *transient* blip; still fail fast on a *persistent* outage. | **Bounded** exponential-backoff retry around the two tracking ops (`src/tracking.py` via new `src/retry.py`): 5 attempts, 5s→10s→20s→30s ≈ 65s, then `TrackingError` as before. | ✅ **IMPLEMENTED** |
| E2 | **Dataset unavailable / checksum mismatch** — PR 10: both fail before training, with distinct log messages; checksum is deterministic. | Fail-fast, no retry on checksum; `botocore` handles transient S3 faults. | Unchanged (already correct). | (none) | ❌ **Declined** — retrying a deterministic checksum is forbidden (rule 1); S3 transients already bounded by botocore + `activeDeadlineSeconds`. |
| E3 | **OOM under-provisioned run** — PR 12 Scenario A: forcing 200Mi (< 256Mi baseline) OOM-fails; 512Mi limit has ~3.9× margin over measured ~133 MiB peak. | Kernel-enforced OOM at the real limit; correct. | Unchanged. | (none) | ❌ **Declined** — no evidence to raise limits (rule 5); envelope is correct. |
| E4 | **Deterministic crash-loop** — PR 12 Scenario B: `backoffLimit=2` stops after 3 attempts; `restartPolicy: Never` gives clean pods. | Terminates correctly; no infinite loop. | Unchanged. | (none) | ❌ **Declined** — behaved as designed. |
| E5 | **Stalled/wedged run** — PR 12 "hardening candidates": liveness probe for stalls. | `activeDeadlineSeconds=1800` kills a stuck Job. | Unchanged. | (none) | ❌ **Declined** — a liveness probe on a batch Job with no socket would kill healthy quiet compute (rule 4); stall-guard already exists (ADR-011). |
| E6 | **OOM reason differs by runtime** — PR 12 finding #3: Docker Desktop reports `Error`, EKS reports `OOMKilled`; `PipelineJobOOMKilled` alert keys on `OOMKilled`. | Alert would match on EKS, not on Docker Desktop. | Keep alert precise. | (none) | ❌ **Declined** — matching `Error` generically would false-positive on any failure; documented cross-runtime limitation to confirm on EKS. |
| E7 | **Durable model fallback / tracking-policy ADR** — PR 11 § 8 #2/#3. | Tracking failure loses the in-memory model. | Larger design change. | (none this PR) | ⏭️ **Deferred** — out of scope; E1 is the smallest, highest-value step and preserves semantics. |

---

## 2. Changes implemented

Only **E1**. See [ADR-037](../decisions/ADR-037-pipeline-reliability-hardening.md) for
the full rationale.

- **`src/retry.py` (new):** `retry_call[T]` — a dependency-free, unit-testable bounded
  retry. Runs a callable a **finite** number of times, retries only on
  caller-specified exception types, and **re-raises the last exception** on exhaustion
  (structural guarantees for "no infinite retries" and "do not hide a persistent
  failure").
- **`src/tracking.py`:** `log_training_run` and `log_evaluation` now run their MLflow
  work in a fresh-run closure wrapped by `retry_call(..., retry_on=(MlflowException,),
  attempts=5, base_delay=5s, max_delay=30s)`. On exhaustion the `MlflowException` is
  converted to `TrackingError` — the original contract is unchanged.

**Failure evidence:** PR 11 § "Reliability analysis" (mid-run row) + § 8 candidate #1.
**Rationale:** a transient rolling-restart blip should not discard expensive training.
**Expected effect:** transient blip absorbed in ≤ ~65s; persistent outage still fails
fast and loud. **Test demonstrating improvement:** `tests/unit/test_retry.py`
(`test_absorbs_transient_failures_then_succeeds`, plus the bound/re-raise/backoff tests).

## 3. Changes deliberately NOT implemented

E2–E7 above, each with its rule-based reason. In short: no checksum retry (rule 1),
no infinite retry (rule 2, structural), no hidden failure (rule 3, structural), no
destabilising probe (rule 4), no resource bump (rule 5), security controls and batch
semantics untouched (rules 6–7).

## 4. Regression / failure tests

**Unit suite (offline):**
```
python -m pytest -m "unit or smoke or contract"
→ 187 passed, 1 skipped, 3 deselected
  (skip = smoke test that needs MLflow, absent from the unit env by design)
python -m ruff check src/ tests/  → All checks passed!
```
Coverage of the fix: `tests/unit/test_retry.py` (the bounded-retry primitive) **and**
`tests/unit/test_tracking.py` (the real `tracking.py` wiring — fresh-run-per-attempt,
`TrackingError` conversion with chaining, bounded to 5 attempts, non-`MlflowException`
not retried — using a `sys.modules` MLflow stub so no MLflow is needed offline).

**MLflow-failure harness — start-gate regression (local, 2026-08-21):** re-run against
the rebuilt image (`k8s/tests/mlflow-failure/run.sh`):
```
Summary: 12 passed, 0 failed.
RESULT: PASS — the MLflow outage was detected, the pipeline failed at the gate
        with no wasted computation, and MLflow was restored with its data intact.
```
The `wait-for-mlflow` gate still exits after `MLflow not ready after 300s`; the fix
does not alter the start-of-run fail-fast path. MLflow was auto-restored to 1/1 and
`mlflow-postgres` stayed Ready throughout.

> The **mid-run transient** improvement (E1) is proven at the **unit** layer — both the
> retry *primitive* (`test_retry.py`) and the real `tracking.py` *wiring*
> (`test_tracking.py`, via a `sys.modules` MLflow stub, since MLflow is deliberately
> absent from the offline unit env — ADR-006 dec. 4). Injecting a precisely-timed
> mid-`train` MLflow blip on the *cluster* is timing-dependent and is not attempted
> here; the harness proves the healthy path and the unchanged start-gate.

## 5. Healthy-run evidence (local, 2026-08-21)

Rebuilt `ml-pipeline:local` (digest `sha256:e93bbafe…`), imported into the node's
containerd `k8s.io` store, re-applied `k8s/overlays/local`:
```
kubectl -n mlops wait --for=condition=complete job/mlops-pipeline --timeout=600s
→ job.batch/mlops-pipeline condition met   (EXIT=0)
NAME             STATUS     COMPLETIONS   DURATION
mlops-pipeline   Complete   1/1           118s
```
Pipeline pod log (abridged) — `train` and `evaluate` both logged to MLflow, model
registered, **no `tracking | … retrying` warnings** (the retry wrapper is transparent
on the happy path):
```
train    | Train stage completed
         | Created version '6' of model 'Best Random Forest Classifier'
evaluate | Evaluate stage completed; model accuracy: 0.7078
```

## 6. Remaining reliability limitations

- **Mid-run retry — now proven live on EKS (2026-08-21)**, not just at the unit layer: a
  timed ~90 s MLflow blip injected during the train stage was absorbed (the tracking call
  retried and succeeded once MLflow returned; train-stage duration extended to 117 s vs
  ~3 s compute), and the completed training was **not discarded** — Job **exit 0**.
  Canonical: [sprint-08-live-eks-evidence.md](sprint-08-live-eks-evidence.md#pr-13--bounded-retry-rides-out-a-transient-blip).
- **A transient failure may leave ≤ 1 orphan MLflow run** before a successful retry
  (accepted trade; ADR-037). For `log_training_run`, an ambiguous-write race can
  occasionally leave a **duplicate registered model version** (the earlier one tied to
  a FAILED run), not just an orphan experiment run — relevant to any consumer that
  selects "latest registered version". See ADR-037 § Consequences.
- **A persistent outage now fails ~65s slower** (bounded; « `activeDeadlineSeconds=1800`).
- **OOM alert cross-runtime discrepancy (E6) — ✅ confirmed on EKS (2026-08-21).** The
  real reason is `OOMKilled` (exit 137), not Docker Desktop's `Error`. The batched
  next-cluster session also **found and fixed** a related defect: `PipelineJobOOMKilled`
  keyed on `kube_pod_container_status_last_terminated_reason`, which is empty for a
  `restartPolicy:Never` Job, so it could never fire — fixed to `..._terminated_reason`,
  after which the alert **fired** (canonical §3, Finding 4).
- **Durable-model fallback / explicit tracking-policy ADR (E7)** — deferred to a
  future sprint.

---

**Document Status:** ✅ COMPLETE · **Sprint:** 8 · **PR:** 13
