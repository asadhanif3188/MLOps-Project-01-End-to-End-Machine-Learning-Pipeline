# Runbook 7 — OOMKilled

> **Part of the [Operational Runbooks](README.md).** Alert:
> [`PipelineJobOOMKilled`](../alerting.md#pipelinejoboomkilled) (critical, 2m). Related:
> [`MLflowMemoryHigh`](../alerting.md#mlflowmemoryhigh),
> [`PostgresMemoryHigh`](../alerting.md#postgresmemoryhigh) (component memory headroom),
> [Pipeline failure](pipeline-failure.md), [Crash / restart](crash-restart.md). Evidence:
> [PR 12 Scenario A](../proof/sprint-08-resource-failure-tests-evidence.md),
> [live-EKS Finding 4](../proof/sprint-08-live-eks-evidence.md#3-findings--4-real-defects-the-live-run-surfaced-all-fixed)
> (real `OOMKilled`, exit **137**; the alert had a metric-keying defect that was found
> and fixed live).

## Purpose

Diagnose and recover when a **pipeline pod** hits its memory limit and the kernel kills
it (`OOMKilled`, exit 137). Also covers the *component* memory-pressure early-warnings
(`MLflowMemoryHigh`, `PostgresMemoryHigh`). This is the pipeline's **designed
memory-safety backstop** — a single OOM on an intentionally tight limit can be expected,
not necessarily a defect.

## Symptoms

- `PipelineJobOOMKilled` firing (critical); usually `PipelineJobFailed` fires alongside
  (the Job also reaches its terminal Failed condition).
- Pod terminated reason **`OOMKilled`**, exit code **137**.
- **EKS reports `OOMKilled`; Docker Desktop only ever reports `Error`** for the same
  memory exhaustion — a documented cross-runtime discrepancy
  ([PR 12 finding #3 / ADR-037 E6](../proof/sprint-08-reliability-hardening-evidence.md)).
- For components: `MLflowMemoryHigh` / `PostgresMemoryHigh` firing (working set > 90% of
  the limit, sustained).

## Detection

```bash
# The alert expression — CURRENT-state terminated reason on the retained finished pod.
# NOTE: it keys on ..._terminated_reason, NOT ..._last_terminated_reason (see limitations).
curl -s --data-urlencode \
  'query=kube_pod_container_status_terminated_reason{namespace="mlops",pod=~"mlops-pipeline-.*",reason="OOMKilled"}' \
  http://localhost:9090/api/v1/query | jq '.data.result[].value[1]'      # 1 == OOMKilled

# Component memory headroom (%):
curl -s --data-urlencode 'query=100 * sum(container_memory_working_set_bytes{namespace="mlops",pod=~"mlflow-.*",pod!~"mlflow-postgres.*",container!="",container!="POD"}) / sum(kube_pod_container_resource_limits{namespace="mlops",pod=~"mlflow-.*",pod!~"mlflow-postgres.*",resource="memory"})' \
  http://localhost:9090/api/v1/query | jq '.data.result[].value[1]'
```

## Initial checks

```bash
pod=$(kubectl -n mlops get pods -l job-name=mlops-pipeline \
  --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}')

# 1. Confirm the kill reason + exit code directly on the pod.
kubectl -n mlops get pod "$pod" \
  -o jsonpath='reason={.status.containerStatuses[0].state.terminated.reason} exit={.status.containerStatuses[0].state.terminated.exitCode}{"\n"}'
  # expect reason=OOMKilled exit=137   (on Docker Desktop: reason=Error — same root cause)

# 2. Which stage was running when it died? (train peaks — GridSearchCV)
kubectl -n mlops logs "$pod" -c pipeline --tail=50

# 3. What is the configured limit? (pipeline = 512Mi)
kubectl -n mlops get job mlops-pipeline \
  -o jsonpath='{.spec.template.spec.containers[0].resources.limits.memory}{"\n"}'
```

## Diagnosis

- **Which container?** The pipeline pod's `pipeline` container (512Mi limit) is the usual
  subject. MLflow (2Gi) and PostgreSQL (512Mi) OOMs surface via their own memory-high
  alerts and the component runbooks ([MLflow](mlflow-unavailable.md) /
  [PostgreSQL](postgresql-failure.md)).
- **Which stage?** `train` (and its `GridSearchCV`) is the memory peak; the measured
  baseline peak is ~133 MiB, well under the 512Mi limit (≈3.9× margin). A genuine OOM at
  512Mi therefore means the working set grew beyond the measured envelope — a larger
  dataset or a wider grid — **not** a routine run.
- **Induced vs real.** The failure-test harness forces an OOM by setting a limit *below*
  baseline (200Mi locally, 128Mi on EKS) — an expected, deliberate OOM, not a platform
  fault ([PR 12](../proof/sprint-08-resource-failure-tests-evidence.md)).

## Likely causes

| Cause | Signal | Response |
|---|---|---|
| **Deliberate failure test** | limit set to 128–200Mi via the harness | none — this is the induced-OOM proof; recover with normal limits |
| **Data / grid growth** | real OOM at the real 512Mi limit; larger inputs | re-measure, then raise the limit *deliberately* |
| **Memory leak / runaway stage** | working set climbs across a run | investigate the stage; reduce working set before raising limits |
| **Component OOM** (MLflow/Postgres) | `*MemoryHigh` fired first | see the component runbook |

## Remediation

**Do not silently grow the limit.** Reduce the stage's working set (batch size, data
loaded at once, grid breadth) **or** raise `limits.memory` *with a measured
justification* ([ADR-011](../decisions/ADR-011-kubernetes-resource-lifecycle.md)):

```bash
# If raising: edit k8s/base/job.yaml → resources.limits.memory (record the measurement in the PR).
```

Then recover with normal limits (the induced-OOM case needs only a clean re-run):

```bash
kubectl -n mlops delete job mlops-pipeline           # ⚠️ discards the failed Job object
kubectl apply -k k8s/overlays/<aws|local>            # restores the normal 512Mi limit
kubectl -n mlops wait --for=condition=complete job/mlops-pipeline --timeout=600s
```

> **Safe reproduction.** `k8s/tests/resource-failure/run.sh` (Scenario A) submits a
> throwaway low-memory Job and cleans it up by deletion — it never mutates the real Job.

## Recovery verification

Proving "it didn't OOM this time" requires a **completed** run at the normal limit, not
just a restarted pod:

```bash
# 1. Job Complete, exit 0, no OOM this run.
kubectl -n mlops get job mlops-pipeline               # STATUS Complete, COMPLETIONS 1/1
pod=$(kubectl -n mlops get pods -l job-name=mlops-pipeline \
  --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}')
kubectl -n mlops get pod "$pod" \
  -o jsonpath='reason={.status.containerStatuses[0].state.terminated.reason} exit={.status.containerStatuses[0].state.terminated.exitCode}{"\n"}'
  # expect reason=Completed exit=0

# 2. OOM series no longer present for the latest pod, and all stages green.
curl -s --data-urlencode 'query=mlops_pipeline_stage_success' \
  http://localhost:9090/api/v1/query \
  | jq '.data.result[] | {stage:.metric.stage, success:.value[1]}'   # all == 1

# 3. Alerts cleared (after the OOMKilled Job object is gone / TTL expires).
curl -s --data-urlencode 'query=ALERTS{alertname=~"PipelineJobOOMKilled|PipelineJobFailed",alertstate="firing"}' \
  http://localhost:9090/api/v1/query | jq '.data.result | length'    # expect 0
```

Live proof: with normal limits/config a healthy Job **Completed exit 0** (baseline + PR 13
runs); the induced-OOM alert fired at 14:19:49Z **only after** the metric-keying fix below.

## Escalation / known limitations

- **The metric-keying defect (fixed) — worth knowing.** The alert originally keyed on
  `kube_pod_container_status_last_terminated_reason`, which KSM derives from a
  container's `lastState` (its *previous*, post-restart termination). The pipeline Job
  runs `restartPolicy: Never` → each container terminates **once, no restart** → the
  `last_*` metric emits **no series** → the alert was **unfireable for this workload**.
  Fixed to `kube_pod_container_status_terminated_reason` (current state)
  ([live-EKS Finding 4](../proof/sprint-08-live-eks-evidence.md#3-findings--4-real-defects-the-live-run-surfaced-all-fixed)).
  If you write a new pod-termination alert, key on the **current** reason for a
  `restartPolicy: Never` workload.
- **Cross-runtime reason** — Docker Desktop reports `Error`, EKS reports `OOMKilled`.
  The alert keys on `OOMKilled` deliberately (matching `Error` generically would
  false-positive on any failure) — so this alert **only fires on EKS**, by design
  ([ADR-037 E6](../proof/sprint-08-reliability-hardening-evidence.md)).
- **`MLflowMemoryHigh` has limited lead time** — MLflow idles ~85% of its 2Gi limit, so
  the 90% early-warning gives little runway; the real backstop is the kernel OOM.
- **Design of record:** [ADR-011 (resources/lifecycle)](../decisions/ADR-011-kubernetes-resource-lifecycle.md),
  [ADR-037 (reliability hardening)](../decisions/ADR-037-pipeline-reliability-hardening.md).
