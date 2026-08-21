# ADR-037: Pipeline reliability hardening from failure-test evidence (Sprint 8, PR 13)

- **Status:** Accepted (implemented — bounded in-run MLflow tracking retry; unit-tested; healthy run + start-gate regression re-verified on local Docker Desktop Kubernetes 2026-08-21)
- **Date:** 2026-08-21
- **Deciders:** Asad Hanif
- **Related:**
  [ADR-009 (Kubernetes workload model — Job, not Deployment)](ADR-009-kubernetes-workload-model.md),
  [ADR-011 (Kubernetes resource lifecycle — limits, backoffLimit, activeDeadlineSeconds, no probes)](ADR-011-kubernetes-resource-lifecycle.md),
  [ADR-026 (In-cluster MLflow platform)](ADR-026-in-cluster-mlflow-platform.md),
  [ADR-027 (S3 dataset runtime retrieval + integrity pin)](ADR-027-s3-dataset-runtime-retrieval.md),
  [ADR-033 (Alerting)](ADR-033-alerting.md),
  [`src/retry.py`](../../src/retry.py), [`src/tracking.py`](../../src/tracking.py),
  [`tests/unit/test_retry.py`](../../tests/unit/test_retry.py),
  [dataset-failure evidence (PR 10)](../proof/sprint-08-dataset-failure-tests-evidence.md),
  [mlflow-failure evidence (PR 11)](../proof/sprint-08-mlflow-failure-tests-evidence.md),
  [resource-failure evidence (PR 12)](../proof/sprint-08-resource-failure-tests-evidence.md)

> **Scope.** This ADR ratifies the reliability changes that Sprint 8's failure-test
> campaign (PRs 10–12) *justified by evidence*. The campaign's own PRs deliberately
> changed **no** runtime behaviour — they observed and recorded. This PR acts on that
> evidence, and only where it is warranted: it adds a **bounded, work-preserving retry
> around the in-run MLflow tracking calls** and **deliberately declines** the other
> candidate changes, recording why. It does not re-decide the workload model (ADR-009),
> the resource envelope (ADR-011), or the dataset integrity contract (ADR-027).

## Context

Sprint 8 built a failure-injection campaign against the MLOps pipeline and captured
what the platform actually does when its dependencies fail:

- **PR 10 (dataset):** an unavailable object or a checksum mismatch stops the pipeline
  **before training**, with distinct, actionable log messages. Checksum mismatch is a
  **deterministic** failure and is (correctly) **never retried**.
- **PR 11 (MLflow):** a start-of-run outage is caught by the `wait-for-mlflow` init
  gate (fail-fast, no compute wasted). A **mid-run** outage, however, surfaces as a
  `TrackingError` from [`src/tracking.py`](../../src/tracking.py) that fails the whole
  Job — **discarding the completed preprocess/split/train compute** — even for a
  *transient* blip such as a ~30–60s rolling restart of the stateless MLflow
  Deployment. The evidence doc flagged this as the one real gap and recommended a
  **bounded** retry (candidate #1), explicitly not "retry forever".
- **PR 12 (resource):** on a local cluster, the 512Mi limit was shown to correctly
  OOM-fail a run forced under baseline, `backoffLimit=2` correctly stops a
  deterministic crash-loop after 3 attempts, and `restartPolicy: Never` gives each
  retry a clean pod. No behaviour needed changing; the 512Mi limit has a ~3.9× margin
  over the measured ~133 MiB peak.

The task's rules bound what "hardening" may do: no retrying deterministic checksum
failures; no infinite retries; do not hide a persistent dependency failure; no probes
that add more instability than they prevent; no resource-limit increases without
evidence; preserve the security controls and the batch (run-to-completion) semantics.

## Decision

**Add a bounded, exponential-back-off retry around the two in-run MLflow tracking
operations** (`log_training_run`, `log_evaluation`), and nothing else.

- A new dependency-free primitive, [`src/retry.py`](../../src/retry.py)
  (`retry_call`), runs a callable a **finite** number of times, retrying only on
  caller-specified exception types, and **re-raises the last exception** once attempts
  are exhausted. The bound and the re-raise are structural, so the "no infinite
  retries" and "do not hide a persistent failure" rules cannot be violated by
  configuration.
- [`src/tracking.py`](../../src/tracking.py) wraps each tracking operation in a
  fresh-run closure and retries it on `MlflowException`: **5 attempts**, back-off
  **5s → 10s → 20s → 30s** (clamped) ≈ **65s** of waiting — long enough to ride out a
  rolling restart, a tiny fraction of the Job's `activeDeadlineSeconds=1800`. On
  exhaustion the `MlflowException` is converted to `TrackingError` exactly as before,
  so a *persistent* outage still fails the run fast and loud.

Each retry reruns the closure from the top, opening a **new** MLflow run, so a retried
attempt never resumes a half-written one. The accepted cost is that a transient
failure may leave at most one incomplete run behind before a later attempt succeeds —
a bounded, self-healing duplication that is far cheaper than discarding the model
training it protects.

### Why a hand-rolled stdlib helper, not `tenacity`

This module ships inside the hardened pipeline image whose dependencies are
inventoried (SBOM, PR 8) and whose provenance is verified (PR 9). A ~40-line,
dependency-free primitive with an explicit, reviewable bound is cheaper to audit than
a new transitive dependency for a single call site, and it makes the two rule
guarantees (finite; never-swallow) legible in the diff.

## Changes deliberately NOT made (and why)

- **No retry for dataset checksum mismatch** (PR 10). Deterministic by construction —
  a re-download yields the same bytes and the same mismatch. Retrying would waste time
  and violate the task's first rule. The existing fail-fast behaviour is correct.
- **No change to the dataset S3 retrieval retry/timeout** (PR 10). `botocore` already
  applies bounded connect/read timeouts and a bounded retry policy for transient S3
  faults, and a missing object fails fast (404, no retry). No evidence of an unbounded
  hang exists; `activeDeadlineSeconds=1800` is the outer stall-guard. Adding bespoke
  timeouts would be speculative, not evidence-driven.
- **No resource-limit increase** (PR 12). The 512Mi limit has a ~3.9× margin over the
  measured ~133 MiB peak and correctly OOM-fails only when forced below baseline. Rule
  5 forbids increases without evidence; the evidence says the envelope is right.
- **No `backoffLimit` / `activeDeadlineSeconds` change** (PR 12). `backoffLimit=2`
  demonstrably stops a deterministic crash-loop after 3 attempts; the 1800s deadline
  is an intentionally generous outer stall-guard. Both behaved as designed.
- **No liveness/readiness/startup probes** (PR 12 candidate). This is a finite batch
  Job with no listening socket or Service; "healthy" is a terminal state (exit code),
  which the Job controller already observes. A liveness probe would fire on the
  pipeline's normal quiet compute and kill healthy runs — rule 4. Stall detection is
  already covered by `activeDeadlineSeconds`. Rationale is unchanged from ADR-011.
- **No loosening of the `PipelineJobOOMKilled` alert to also match `Error`** (PR 12
  finding #3). Docker Desktop's containerd reports memory exhaustion as `Error`, not
  `OOMKilled`; EKS reports `OOMKilled`. Matching `Error` generically would fire on any
  non-OOM failure — a false-positive regression. Left as a documented cross-runtime
  limitation to verify on EKS, not a code change.
- **No mid-run durable-model fallback / tracking-policy ADR** (PR 11 candidates #2/#3).
  Larger design changes out of scope for an evidence-driven hardening PR; the bounded
  retry (#1) is the smallest, highest-value step and preserves current semantics.

## Consequences

**Positive.** A transient mid-run MLflow blip no longer discards completed training —
the dominant wasted-compute failure mode from PR 11 is closed while the fail-fast
guarantees (start-gate; persistent-outage failure) are preserved. The retry primitive
is generic, unit-tested, and reusable. The `Job.backoffLimit=2` remains a coarser,
whole-run retry layered above this fine-grained, work-preserving one.

**Negative / accepted.** At most one orphan MLflow run may be left by a transient
failure before a successful retry. For `log_training_run` specifically, the last call
in the retried closure is `mlflow.sklearn.log_model(..., registered_model_name=…)`;
if a transient fault strikes *after* the server has durably created a **registered
model version** but *before* the client sees success (an ambiguous-write race on a
non-idempotent call), the retry can register a **second** version — so the cost is
occasionally a duplicate *registry version* (the earlier one pointing at a run the
run-context marks FAILED), not merely a duplicate experiment run. A consumer that
selects the "latest registered version" or audits registry history should be aware of
this narrow edge case. A *persistent* outage now takes up to ~65s longer to fail
(bounded, « the 1800s deadline). A deterministic `MlflowException` (rare — e.g. a
malformed logging call) is retried up to the bound before failing; the added latency
is bounded and the failure is not hidden.

## Verification

- **Unit:** [`tests/unit/test_retry.py`](../../tests/unit/test_retry.py) pins the
  primitive: absorbs transient failures then succeeds; re-raises the last exception
  after exhausting attempts (never swallows); retrying is finite (exactly `attempts`);
  non-retryable exceptions propagate immediately; back-off is exponential and clamped.
  [`tests/unit/test_tracking.py`](../../tests/unit/test_tracking.py) pins the
  *integration* (via a `sys.modules` MLflow stub, since MLflow is absent from the unit
  env): the real `log_training_run`/`log_evaluation` absorb a transient
  `MlflowException` with a **fresh run per attempt**, convert a *persistent* failure to
  `TrackingError` with the original **chained**, stay **bounded** to the 5-attempt
  policy, and do **not** retry or wrap a non-`MlflowException`. Full suite:
  **187 passed, 1 skipped** (the skip is the smoke test that needs MLflow, by design).
- **Healthy run (local, 2026-08-21):** rebuilt image, `mlops-pipeline` **Complete** in
  118s; `train` and `evaluate` logged to MLflow (model version registered) with **no
  retry warnings** — the wrapper is transparent on the happy path.
- **Start-gate regression (local, 2026-08-21):** the MLflow-failure harness
  (`k8s/tests/mlflow-failure/run.sh`) was re-run against the new image; the
  start-of-run gate behaviour is unchanged (fail-fast, no compute wasted). Evidence in
  [the PR 11 doc](../proof/sprint-08-mlflow-failure-tests-evidence.md).

The mid-run transient improvement is proven at the unit layer (the tracking module
imports MLflow, deliberately absent from the offline unit env — ADR-006 dec. 4); the
runtime harness proves the healthy path and the unchanged start-gate.
