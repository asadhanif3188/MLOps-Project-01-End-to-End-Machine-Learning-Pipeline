# Dataset failure-path tests (Sprint 8, PR 10)

Runtime verification that the pipeline's **dataset availability and integrity**
failure paths behave correctly on a real cluster — the counterpart to the
**unit** contract in [`tests/unit/test_fetch_dataset.py`](../../../tests/unit/test_fetch_dataset.py).

- **Unit (CI, no cluster):** proves the retrieval logic — URI parsing, the integrity
  gate, and typed-error-on-failure — with an injected fake S3 client. Fast, offline.
- **Runtime (this suite, needs a cluster):** proves the deployed init container
  actually fails **fast**, **before training**, with a **clear error** and an
  **operational signal**, when the dataset is unavailable or its checksum mismatches.

## What it checks

Two controlled, reversible failures on the **same deployed workload**:

| Scenario | Injection (fetch-dataset init container) | Expected failure | Distinct error |
|---|---|---|---|
| **A — unavailable** | `DATASET_S3_URI` → a non-existent object key | S3 GET fails; Job retries to `backoffLimit` then Failed | `Failed to download …` |
| **B — checksum mismatch** | `DATASET_SHA256` → a wrong (but well-formed) digest | object **retrieved**, integrity gate rejects it | `integrity check failed …` |

For each scenario the harness asserts:

1. the `fetch-dataset` init container **terminates non-zero**;
2. its logs carry the **expected, distinct** error (the root-cause layer, ADR-030);
3. the `pipeline` container **never starts** — *training does not begin*, which is
   why the Job cannot Complete (implied by check 3, not measured separately — a
   backoff-retrying throwaway Job is trivially "not yet Complete" mid-run; the
   run-level Failed condition is proven on the **real** Job in the alert step).

**Scenario B is a deterministic failure and must fail fast.** A checksum mismatch is
not transient — re-downloading yields the same bytes and the same mismatch — so it is
**never** retried away. The harness asserts fail-fast; it must **not** be "fixed" by
adding retries (that is a correctness regression, explicitly out of scope for this PR;
resilience/retry work is PR 13).

## How it works

The committed manifests are **never mutated**. For each scenario the harness renders
the *deployed* Job to a clean spec, applies **one** narrow env override to the
`fetch-dataset` init container, submits it under a throwaway name
(`<job>-datafail-<A|B>`), observes the failure, captures the init-container logs, and
deletes the throwaway Job. The real Job / ConfigMap / overlays are untouched, so
"restore" is simply *stop using the override* — there is nothing to roll back.

Cluster-wide signals (Pushgateway series, Grafana panel, the `PipelineJobFailed`
alert) are captured by the operator per the
[proof-doc checklist](../../../docs/proof/sprint-08-dataset-failure-tests-evidence.md);
the script prints the exact queries. The alert requires the override on the **real**
Job (so it exhausts `backoffLimit`), not a throwaway.

## Usage

```bash
k8s/tests/dataset-failure/run.sh              # both scenarios
SCENARIO=A k8s/tests/dataset-failure/run.sh   # unavailable only
SCENARIO=B k8s/tests/dataset-failure/run.sh   # checksum only
NAMESPACE=mlops JOB=mlops-pipeline k8s/tests/dataset-failure/run.sh
```

Prerequisites: a reachable cluster with the mlops workload deployed
(`kubectl apply -k k8s/overlays/<aws|local>`); the monitoring stack deployed if
capturing the metric/alert evidence.

Exit codes: `0` = every requested scenario failed as expected; `1` = a scenario did
**not** fail as required (e.g. the pipeline started despite a bad dataset — a real
reliability regression); `2` = environment/precondition problem.

## Recovery

After the failure runs, prove healthy recovery with an **unmodified** Job:

```bash
kubectl -n mlops delete job mlops-pipeline
kubectl apply -k k8s/overlays/<aws|local>
kubectl -n mlops wait --for=condition=complete job/mlops-pipeline --timeout=600s
```
