# Runbook 2 — Pipeline failure

> **Part of the [Operational Runbooks](README.md).** Alert:
> [`PipelineJobFailed`](../alerting.md#pipelinejobfailed) (critical, 2m). Evidence:
> [PR 10](../proof/sprint-08-dataset-failure-tests-evidence.md),
> [live-EKS §6](../proof/sprint-08-live-eks-evidence.md#6-pr-10--12--failure-paths--alerts)
> (`PipelineJobFailed` FIRING 13:26:13Z on a real Job that exhausted `backoffLimit`).

## Purpose

Diagnose and recover the `mlops-pipeline` batch Job when it reaches its **terminal
Failed condition** — i.e. the last run did not Complete. This is the top-level pipeline
runbook; several specific causes (dataset, OOM, MLflow) have their own deeper runbooks
that this one routes to.

## Symptoms

- `PipelineJobFailed` firing (critical).
- `kubectl -n mlops get job mlops-pipeline` shows `Failed`/`Complete 0/1`.
- **MLOps Pipeline Operations** dashboard shows the last run failed / a stage red.
- Job event `BackoffLimitExceeded`; three failed pods (initial + `backoffLimit=2`).

> **Batch semantics.** A *finished* Job is not "Running", and that is normal — nothing
> alerts on "the pod is not Running". `PipelineJobFailed` keys on the Job controller's
> terminal **Failed condition** (`kube_job_failed{condition="true"}`), set **only after
> `backoffLimit` is exhausted** — never on the transient pod-failure counter. A run that
> retries once and then succeeds does **not** page.

## Detection

```bash
# The alert's own expression (terminal Failed condition == true):
curl -s --data-urlencode \
  'query=kube_job_failed{namespace="mlops",job_name="mlops-pipeline",condition="true"}' \
  http://localhost:9090/api/v1/query | jq '.data.result[].value[1]'      # 1 == failed
```

## Initial checks

```bash
# 1. Job condition — distinguishes BackoffLimitExceeded (pod kept failing) from
#    DeadlineExceeded (stall past activeDeadlineSeconds=1800).
kubectl -n mlops get job mlops-pipeline -o jsonpath='{.status.conditions}{"\n"}'
kubectl -n mlops describe job/mlops-pipeline        # Events: SuccessfulCreate ×3, BackoffLimitExceeded

# 2. The failing stage's structured logs — the root-cause layer. All attempts of a
#    deterministic pipeline fail identically, so the first pod is representative.
kubectl -n mlops logs job/mlops-pipeline --all-containers --prefix
```

## Diagnosis

Walk from *which stage* to *why*:

```bash
# Which stage was 0 in the last run? (Pushgateway per-stage success, ADR-030)
curl -s --data-urlencode 'query=mlops_pipeline_stage_success == 0' \
  http://localhost:9090/api/v1/query | jq '.data.result[].metric.stage'
```

The five stages run in order `fetch_dataset → preprocess → split → train → evaluate`.
On an **early** failure the failing stage is `0` and **every later stage is absent** (a
per-run reset + fail-fast means "the pipeline never got there"). A **hard OOM** leaves
the stage *absent* rather than `0`, so pair the stage query with the run-level OOM check
below.

Route by failing stage:

| Failing stage / signal | Root cause | Go to |
|---|---|---|
| `fetch_dataset` + log `Failed to download s3://…` | dataset unavailable | [Dataset retrieval failure](dataset-retrieval-failure.md) |
| `fetch_dataset` + log `integrity check failed: expected …, got …` | checksum mismatch | [Dataset integrity failure](dataset-integrity-failure.md) |
| any stage + `terminated_reason{reason="OOMKilled"}==1` | memory limit hit | [OOMKilled](oomkilled.md) |
| `train`/`evaluate` + `TrackingError` / MLflow refused | tracking outage | [MLflow unavailable](mlflow-unavailable.md) |
| pod never started; `wait-for-mlflow` timed out | MLflow down at start | [MLflow unavailable](mlflow-unavailable.md) |
| `DeadlineExceeded` condition | stall (hung dependency) | see *Likely causes* below |
| deterministic app error (bad param) | pipeline code/config | [Crash / restart](crash-restart.md) |

```bash
# Was a pipeline pod OOMKilled? (current-state reason on the retained finished pod)
curl -s --data-urlencode \
  'query=kube_pod_container_status_terminated_reason{namespace="mlops",pod=~"mlops-pipeline-.*",reason="OOMKilled"}' \
  http://localhost:9090/api/v1/query | jq '.data.result[].value[1]'
```

## Likely causes

1. **Dataset boundary** — missing/denied object or a checksum mismatch (fails in
   `fetch-dataset` before training). See runbooks 3 and 4. Proven live: Scenario A →
   `404 HeadObject`, Scenario B → integrity mismatch, both exit 1, pipeline never started.
2. **Dependency down** — MLflow (`wait-for-mlflow` gate blocks at start, or `train`
   raises `TrackingError` mid-run) or PostgreSQL underneath it. `PipelineJobFailed`
   firing *with* `MLflowDown`/`PostgresDown` is the tell.
3. **Memory** — a stage exceeded the 512Mi limit → `OOMKilled` (exit 137).
4. **Deterministic application failure** — a bad parameter or a code bug fails every
   attempt identically; `backoffLimit=2` correctly stops after 3 pods (not an infinite
   loop).
5. **Stall** — `DeadlineExceeded` at `activeDeadlineSeconds=1800` means the run hung
   (e.g. a wedged network call), not normal compute (which is sub-minute). Investigate
   the dependency it stalled on; the deadline is a stall-guard, not an SLO
   ([ADR-011](../decisions/ADR-011-kubernetes-resource-lifecycle.md)).

## Remediation

Fix the **root cause first** (via the routed runbook), then re-drive a clean run. A
Job's pod template is immutable, so recovery is delete-then-reapply:

```bash
kubectl -n mlops delete job mlops-pipeline           # ⚠️ discards the failed Job object (logs already read above)
kubectl apply -k k8s/overlays/<aws|local>            # on EKS: scripts/render-cloud-manifests.sh --apply
kubectl -n mlops wait --for=condition=complete job/mlops-pipeline --timeout=600s
```

Do **not** raise `backoffLimit` or add blanket retries to "make it pass" — a
deterministic failure must fail fast (see [Crash / restart](crash-restart.md)).

## Recovery verification

Not "the Job was resubmitted" — **prove the run completed and the platform is green**:

```bash
# 1. Job Complete, exit 0.
kubectl -n mlops get job mlops-pipeline               # STATUS Complete, COMPLETIONS 1/1
pod=$(kubectl -n mlops get pods -l job-name=mlops-pipeline \
  --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}')
kubectl -n mlops get pod "$pod" \
  -o jsonpath='phase={.status.phase} exit={.status.containerStatuses[0].state.terminated.exitCode}{"\n"}'
  # expect phase=Succeeded exit=0

# 2. All five stages succeeded in the last run.
curl -s --data-urlencode 'query=mlops_pipeline_stage_success' \
  http://localhost:9090/api/v1/query \
  | jq '.data.result[] | {stage:.metric.stage, success:.value[1]}'      # all == 1

# 3. The alert cleared (once the failed Job object is gone / a run Completes).
curl -s --data-urlencode 'query=ALERTS{alertname="PipelineJobFailed",alertstate="firing"}' \
  http://localhost:9090/api/v1/query | jq '.data.result | length'        # expect 0
```

Then run the [Platform health](platform-health.md) triage to confirm the whole platform
is green.

## Escalation / known limitations

- **`PipelineJobFailed` stays firing while the failed Job object exists** (up to
  `ttlSecondsAfterFinished=3600`, or until you delete it). Deleting the failed Job and
  running a fresh green one both clear it; a green run alone does not un-fail the old
  object.
- **No Alertmanager routing** — the alert is visible on Prometheus `/alerts` only.
- A failed run is *often a symptom* of a lower layer — always check the routed runbook
  rather than only re-running.
- **Design of record:** [ADR-011 (resources/lifecycle)](../decisions/ADR-011-kubernetes-resource-lifecycle.md),
  [ADR-030 (pipeline metrics)](../decisions/ADR-030-pipeline-operational-metrics.md),
  [ADR-033 (alerting)](../decisions/ADR-033-alerting.md).
