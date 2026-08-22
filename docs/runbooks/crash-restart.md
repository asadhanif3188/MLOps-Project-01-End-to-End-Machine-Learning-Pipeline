# Runbook 8 — Crash / restart behaviour

> **Part of the [Operational Runbooks](README.md).** Alert:
> [`KubePodCrashLooping`](../alerting.md#kubepodcrashlooping) (warning, 15m). Related:
> [Pipeline failure](pipeline-failure.md), [OOMKilled](oomkilled.md). Evidence:
> [PR 12 Scenario B](../proof/sprint-08-resource-failure-tests-evidence.md)
> (deterministic exit 42, `backoffLimit=2` → 3 pods → terminal Failed),
> [PR 13](../proof/sprint-08-reliability-hardening-evidence.md) (bounded retry),
> [live-EKS §6](../proof/sprint-08-live-eks-evidence.md#6-pr-10--12--failure-paths--alerts)
> (`KubePodCrashLooping` FIRING on a pod held >15 min).

## Purpose

Explain and act on **crash / restart** behaviour across the platform: the batch Job's
retry-then-fail semantics, genuine `CrashLoopBackOff` on long-lived pods, and the
distinction between a *transient* blip (rideable) and a *deterministic* failure (must fail
fast). Crucially, it explains **why the pipeline Job never shows `CrashLoopBackOff`** — so
you diagnose the right thing.

## Symptoms

- `KubePodCrashLooping` firing (warning) — a container in `mlops`/`monitoring` has been in
  `CrashLoopBackOff` for 15m.
- A pod's `RESTARTS` count climbing (long-lived pods: MLflow, Postgres, exporters).
- The pipeline Job failing after exactly **3 pods** with `RESTARTS: 0` each and a
  `BackoffLimitExceeded` event — this is **normal batch retry**, not a crash-loop.

> **Two different mechanisms — do not conflate them.**
> - **The pipeline Job** runs `restartPolicy: Never`: the kubelet **never restarts the
>   container in place**. On failure the Job controller creates a **new pod** for the next
>   retry (up to `backoffLimit=2` → 3 pods total), so each pod shows `RESTARTS: 0`. A
>   `restartPolicy: Never` Job **cannot** produce `CrashLoopBackOff` — so
>   `KubePodCrashLooping` **does not fire for the pipeline Job**.
> - **Long-lived pods** (`restartPolicy: Always`: MLflow, Postgres, monitoring) *do*
>   restart in place and *can* enter `CrashLoopBackOff` — that is what
>   `KubePodCrashLooping` catches.

## Detection

```bash
# The alert expression — kubelet CrashLoopBackOff waiting reason, scoped to the two namespaces:
curl -s --data-urlencode \
  'query=kube_pod_container_status_waiting_reason{namespace=~"mlops|monitoring",reason="CrashLoopBackOff"}' \
  http://localhost:9090/api/v1/query | jq '.data.result[] | {ns:.metric.namespace, pod:.metric.pod}'

# Restart counts across the namespaces:
kubectl -n mlops get pods -o wide
kubectl -n monitoring get pods -o wide
```

## Initial checks

```bash
# For a crash-looping LONG-LIVED pod:
kubectl -n <ns> describe pod <pod>                    # Events + last State/Reason
kubectl -n <ns> logs <pod> --previous                 # the crash logs from the prior restart

# For the pipeline JOB (retry-then-fail, NOT a crash-loop):
kubectl -n mlops describe job/mlops-pipeline          # Events: 3× SuccessfulCreate, then BackoffLimitExceeded
kubectl -n mlops get job mlops-pipeline -o jsonpath='{.status.conditions}{"\n"}'
kubectl -n mlops logs job/mlops-pipeline --all-containers   # each attempt fails identically if deterministic
```

## Diagnosis

1. **Is it the pipeline Job or a long-lived pod?**
   - Job: 3 pods, `RESTARTS: 0`, `BackoffLimitExceeded` → this is normal fail-fast. Go to
     [Pipeline failure](pipeline-failure.md) to find *why* it failed (read the pod logs).
   - Long-lived pod: `RESTARTS` climbing, `CrashLoopBackOff` waiting reason → a real
     crash-loop; read `--previous` logs and the last terminated reason (often OOM → see
     [OOMKilled](oomkilled.md), or a bad config/arg like the live postgres-exporter
     Finding 2 → [PostgreSQL failure](postgresql-failure.md)).
2. **Deterministic vs transient?**
   - **Deterministic** (same failure every attempt — a bad param, exit 42, a rejected
     flag): `backoffLimit=2` correctly stops after 3 attempts. Retrying will not help; fix
     the cause.
   - **Transient** (a rolling restart / brief dependency blip): the Job's `backoffLimit`
     absorbs it, and for the mid-run MLflow case the PR 13 bounded retry rides out a
     ~90s blip so completed training is not discarded ([PR 13](../proof/sprint-08-reliability-hardening-evidence.md)).

## Likely causes

| Cause | Where | Signal |
|---|---|---|
| Deterministic pipeline error (bad param/code) | pipeline Job | 3 pods, identical logs, `BackoffLimitExceeded` |
| OOM | any container | last reason `OOMKilled` / exit 137 → [OOMKilled](oomkilled.md) |
| Bad container arg/config | long-lived pod | `error: … try --help` in logs (live postgres-exporter Finding 2) |
| Dependency flapping | MLflow/pipeline | connection refused; recovers when the dependency stabilises |
| Image / startup failure | any | `CreateContainerConfigError` / `ImagePullBackOff` |

## Remediation

- **Deterministic pipeline failure** — fix the root cause (config, code, data), then
  re-drive a clean run. **Do not raise `backoffLimit` or add blanket retries** to force it
  green — a deterministic failure *must* fail fast ([ADR-011](../decisions/ADR-011-kubernetes-resource-lifecycle.md);
  PR 13 declined this, rule 2):
  ```bash
  kubectl -n mlops delete job mlops-pipeline           # ⚠️ discards the failed Job object
  kubectl apply -k k8s/overlays/<aws|local>
  kubectl -n mlops wait --for=condition=complete job/mlops-pipeline --timeout=600s
  ```
- **Long-lived pod crash-loop** — fix the underlying cause (the arg/config, the OOM, the
  dependency), then let the controller (Deployment/StatefulSet) reschedule:
  ```bash
  kubectl -n <ns> rollout restart deploy/<name>        # or delete the crashing pod (StatefulSet reattaches its PVC)
  ```
  ⚠️ For `mlflow-postgres`, delete the **pod** only — never the PVC/StatefulSet (see
  [PostgreSQL failure](postgresql-failure.md)).

> **Safe reproduction.** `k8s/tests/resource-failure/run.sh` (Scenario B) injects a
> deterministic `exit 42` into a throwaway Job to exercise the `backoffLimit` path
> (3 pods → terminal Failed), then cleans up by deletion — the real Job is untouched.

## Recovery verification

Prove stability, not just a single restart:

```bash
# 1. No pod is crash-looping any more.
curl -s --data-urlencode \
  'query=kube_pod_container_status_waiting_reason{namespace=~"mlops|monitoring",reason="CrashLoopBackOff"}' \
  http://localhost:9090/api/v1/query | jq '.data.result | length'    # expect 0

# 2. Long-lived pods Ready with stable restart counts.
kubectl -n mlops get pods                              # RESTARTS stops climbing; READY 1/1
kubectl -n monitoring get pods

# 3. For the pipeline: a fresh run Completes end-to-end.
kubectl -n mlops get job mlops-pipeline                # STATUS Complete, COMPLETIONS 1/1
curl -s --data-urlencode 'query=min by (job) (mlops_pipeline_stage_success)' \
  http://localhost:9090/api/v1/query | jq '.data.result[].value[1]'  # expect 1

# 4. Alert cleared.
curl -s --data-urlencode 'query=ALERTS{alertname="KubePodCrashLooping",alertstate="firing"}' \
  http://localhost:9090/api/v1/query | jq '.data.result | length'    # expect 0
```

## Escalation / known limitations

- **`KubePodCrashLooping` needs 15m of sustained `CrashLoopBackOff`** — a short failure
  test (seconds) will not trip it; that is deliberate (it filters one-off restarts, the
  canonical kube-prometheus pattern). It was proven live with a dedicated
  `restartPolicy: Always` pod held >15 min.
- **The pipeline Job's fail-fast is a feature, not a loop** — `backoffLimit=2` +
  `restartPolicy: Never` bounds retries to 3 clean pods; there is no infinite loop to
  break, by design.
- **`activeDeadlineSeconds=1800`** is the outer stall-guard for a *hung* (not crashing)
  run → `DeadlineExceeded` ([Pipeline failure](pipeline-failure.md)).
- **Design of record:** [ADR-011 (resources/lifecycle)](../decisions/ADR-011-kubernetes-resource-lifecycle.md),
  [ADR-037 (reliability hardening / bounded retry)](../decisions/ADR-037-pipeline-reliability-hardening.md).
