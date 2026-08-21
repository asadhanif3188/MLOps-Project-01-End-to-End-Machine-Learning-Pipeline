# MLflow outage failure-path tests (Sprint 8, PR 11)

Runtime verification that an **MLflow tracking-server outage** is **visible**,
**diagnosable** and **recoverable** on a real cluster — and characterisation of how
the pipeline currently behaves while MLflow is down.

Unlike the [dataset-failure suite](../dataset-failure/README.md) (which injects a
fault into a *throwaway* Job and touches nothing shared), this test creates a real —
but fully reversible — **outage of the shared MLflow Service**. It is therefore built
around one hard safety guarantee.

## Safety: MLflow is always restored

- The **only** mutation is `kubectl scale deploy/mlflow --replicas=0`, against the
  **stateless** tracking-server Deployment.
- It **never** touches the `mlflow-postgres` StatefulSet, its PVC, or the S3 artifact
  bucket. PostgreSQL metadata and S3 artifacts persist **by construction** — the
  harness asserts the StatefulSet stays Ready throughout.
- An **`EXIT`/`INT`/`TERM` trap** is installed *before* the scale-down, so a failure,
  a timeout, or a Ctrl-C all restore the original replica count on the way out. The
  harness cannot leave MLflow down.

## What it does

| Step | Action | Assertion |
|---|---|---|
| **1. Baseline** | read platform health | MLflow Deployment Ready, `mlflow-postgres` Ready, Service has Endpoints |
| **2. Outage** | scale `deploy/mlflow` → 0 | (records original replicas; trap armed) |
| **3. Observe** | watch the Service | Endpoints drain to empty; PostgreSQL stays Ready |
| **4. Pipeline** | submit a throwaway Job | `fetch-dataset` **succeeds**; `wait-for-mlflow` **fails** (`MLflow not ready after …`); `pipeline` **never starts** |
| **5. Restore** | scale back to original | Deployment Ready again; Endpoints return; PostgreSQL still Ready |
| **6. Recover** | (operator) | previous runs still visible; a new unmodified run Completes |

The cluster-wide signals (`probe_success{job="blackbox-mlflow-health"}` 1→0→1, the
`MLflowDown` alert Pending→Firing→Resolved, Grafana "MLflow Platform Health") are
captured by the operator per the
[proof-doc checklist](../../../docs/proof/sprint-08-mlflow-failure-tests-evidence.md);
the script prints the exact queries.

## Current pipeline behaviour under an outage (characterised, not changed)

Two timings, two behaviours — this PR **observes** them and changes neither:

- **MLflow down at START** (what the harness reproduces): the `wait-for-mlflow` init
  container polls `/health` for ~300s, then exits non-zero. The Job fails **at the
  gate**; the pipeline container never runs, so **no model computation is wasted**.
  This is a *deterministic, well-signalled* fail-fast — appropriate as-is.
- **MLflow down MID-RUN** (analysed, not injected here — it is timing-dependent): a
  tracking call inside the `train` stage raises `MlflowException`, which
  [`src/tracking.py`](../../../src/tracking.py) re-raises as `TrackingError`. That
  fails the `train` stage → the whole Job fails, **and the completed compute
  (preprocess/split/train) is wasted** because the model is persisted *by* the MLflow
  log. This is the reliability question the proof doc records as a **PR 13 candidate**
  (e.g. bounded retry around the log call to ride out a rolling restart, or a durable
  model fallback) — **no retry is added here**.

## Usage

```bash
k8s/tests/mlflow-failure/run.sh                       # full outage → recover cycle
NAMESPACE=mlops MLFLOW_DEPLOY=mlflow JOB=mlops-pipeline k8s/tests/mlflow-failure/run.sh
MLFLOW_OUTAGE_WAIT=480 k8s/tests/mlflow-failure/run.sh # more slack for image pull
SKIP_PIPELINE=1 k8s/tests/mlflow-failure/run.sh        # outage/observe/restore only
```

The pipeline-during-outage step waits for the `wait-for-mlflow` gate, which polls for
~300s before failing — so a full run takes several minutes plus image-pull time.

Prerequisites: a reachable cluster with the mlops workload **and** the in-cluster
MLflow platform deployed (`kubectl apply -k k8s/overlays/<aws|local>`); the monitoring
stack deployed if capturing the probe/alert evidence.

Exit codes: `0` = outage detected, pipeline behaved as characterised, MLflow restored;
`1` = behaviour differed from the reliability contract (see `[FAIL]` lines); `2` =
environment/precondition problem. **MLflow is restored on every exit path.**

## Recovery

The harness restores MLflow itself (step 5 + the trap). To prove a healthy end-to-end
run afterwards:

```bash
kubectl -n mlops delete job mlops-pipeline
kubectl apply -k k8s/overlays/<aws|local>
kubectl -n mlops wait --for=condition=complete job/mlops-pipeline --timeout=600s
```
