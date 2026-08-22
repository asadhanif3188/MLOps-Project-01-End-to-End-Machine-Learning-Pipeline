# Runbook 5 — MLflow unavailable

> **Part of the [Operational Runbooks](README.md).** Alert:
> [`MLflowDown`](../alerting.md#mlflowdown) (warning, 5m). Related:
> [`MLflowMemoryHigh`](../alerting.md#mlflowmemoryhigh),
> [PostgreSQL failure](postgresql-failure.md) (MLflow depends on it),
> [Pipeline failure](pipeline-failure.md). Evidence:
> [PR 11](../proof/sprint-08-mlflow-failure-tests-evidence.md),
> [live-EKS §7](../proof/sprint-08-live-eks-evidence.md#7-pr-11--mlflow-outage-detection--recovery-8-items)
> (outage → `probe_success` 1→0 → `MLflowDown` FIRING @ 14:00:06Z → recover → RESOLVED
> 14:04:04Z, **persistence proven**).

## Purpose

Diagnose and recover when the MLflow **tracking server** is unavailable — new pipeline
runs cannot log to it. Critically: **experiment data is not lost** during an MLflow
outage (it lives in PostgreSQL + S3), which is why this is a *warning*, not *critical*.

## Symptoms

- `MLflowDown` firing (warning, after 5m).
- `probe_success{job="blackbox-mlflow-health"}` is `0`; the `mlflow` Service has **no
  endpoints**.
- **MLflow Platform Health** dashboard availability gauge **red** — but the **PostgreSQL
  panels stay green** (the DB survived).
- A pipeline run that *starts* while MLflow is down blocks at the `wait-for-mlflow` init
  gate (`MLflow not ready after 300s` / `urlopen error timed out`) and fails **before any
  computation** — no wasted work. A run *already mid-`train`* rides out a *transient*
  blip via the PR 13 bounded retry, but fails on a *persistent* outage.

## Detection

```bash
# The alert expression — the blackbox /health probe:
curl -s --data-urlencode 'query=probe_success{job="blackbox-mlflow-health"}' \
  http://localhost:9090/api/v1/query | jq '.data.result[].value[1]'      # 0 == down

# What HTTP status did /health return?
curl -s --data-urlencode 'query=probe_http_status_code{job="blackbox-mlflow-health"}' \
  http://localhost:9090/api/v1/query | jq '.data.result[].value[1]'
```

MLflow has no native `/metrics`, so availability is proven by this external HTTP probe
([ADR-031](../decisions/ADR-031-mlflow-postgres-monitoring.md)).

## Initial checks

```bash
# 1. Is the Deployment serving a replica? Are there endpoints?
kubectl -n mlops get deploy mlflow                    # READY should be 1/1
kubectl -n mlops get endpoints mlflow                 # should list a pod IP:5000

# 2. Why is the pod not serving?
kubectl -n mlops describe deploy/mlflow
kubectl -n mlops logs deploy/mlflow --tail=100

# 3. CHECK POSTGRES FIRST — MLflow depends on it.
curl -s --data-urlencode 'query=pg_up' \
  http://localhost:9090/api/v1/query | jq '.data.result[].value[1]'      # if 0 → runbook 6 first
```

## Diagnosis

Work outside-in — is MLflow *itself* down, or is its dependency down?

- **`pg_up == 0`** → the DB is the root cause; MLflow is a *symptom*. Fix PostgreSQL
  first ([PostgreSQL failure](postgresql-failure.md)), then MLflow recovers.
- **`deploy/mlflow` at 0 replicas / no endpoints** → the server itself is down (the live
  outage method was `kubectl scale deploy/mlflow --replicas=0`). Scale it back up.
- **Pod present but `probe_success 0`** → the server is up but not answering `/health`:
  image pull, OOM/restart (check `MLflowMemoryHigh` — MLflow idles ~85% of its 2Gi
  limit), or a bad probe target.

The **signature** `MLflowDown` **+** `PipelineJobFailed` firing together = "the pipeline
is blocked because MLflow is down" ([Pipeline failure](pipeline-failure.md)).

## Likely causes

| Cause | How to tell | Fix |
|---|---|---|
| Server scaled to 0 / evicted | `deploy/mlflow` READY 0/1, no endpoints | scale back to 1 |
| PostgreSQL down | `pg_up == 0` | [runbook 6](postgresql-failure.md) first |
| MLflow OOMKilled | pod restarts; `MLflowMemoryHigh` fired; `lastState.terminated.reason=OOMKilled` | reduce load / raise 2Gi limit deliberately |
| Image pull failure | `ImagePullBackOff` in `describe` | fix the image ref / registry access |
| Rolling restart (transient) | recovers within ~30–60s; `for: 5m` may not even fire | none — the 5m `for` rides out normal restarts |

## Remediation

**Restore the server** (the common case — scale back up):

```bash
kubectl -n mlops scale deploy/mlflow --replicas=1
kubectl -n mlops rollout status deploy/mlflow --timeout=180s
```

If the DB was the cause, resolve [PostgreSQL failure](postgresql-failure.md) first — MLflow
comes back once its backend is reachable. For an OOM, see [OOMKilled](oomkilled.md) (the
MLflow-memory case).

> **Safe outage-recovery harness.** `k8s/tests/mlflow-failure/run.sh` reproduces an
> outage by scaling the *stateless* Deployment to 0 (never touching the
> `mlflow-postgres` StatefulSet, its PVC, or S3) and restores it via an
> `EXIT`/`INT`/`TERM` trap, so it can never leave MLflow down. Expect `RESULT: PASS`.

Then re-drive any pipeline run that failed at the gate (per the universal pattern in
[Pipeline failure](pipeline-failure.md)).

## Recovery verification

Prove availability **and** that no experiment data was lost:

```bash
# 1. Availability restored.
curl -s --data-urlencode 'query=probe_success{job="blackbox-mlflow-health"}' \
  http://localhost:9090/api/v1/query | jq '.data.result[].value[1]'      # expect 1
kubectl -n mlops get endpoints mlflow                 # endpoints return

# 2. Alert Resolved.
curl -s --data-urlencode 'query=ALERTS{alertname="MLflowDown",alertstate="firing"}' \
  http://localhost:9090/api/v1/query | jq '.data.result | length'        # expect 0

# 3. PERSISTENCE — previous runs survived (the whole point).
#    pg_up stayed 1 throughout; the run count did not drop.
curl -s --data-urlencode 'query=pg_up' \
  http://localhost:9090/api/v1/query | jq '.data.result[].value[1]'      # expect 1
kubectl -n mlops port-forward svc/mlflow 5000:5000 &   # MLflow UI: prior runs + registered model still present

# 4. A fresh, unmodified pipeline run Completes end-to-end (proves the path works again).
kubectl -n mlops delete job mlops-pipeline           # ⚠️ discards the failed Job object
kubectl apply -k k8s/overlays/<aws|local>            # on EKS: scripts/render-cloud-manifests.sh --apply
kubectl -n mlops wait --for=condition=complete job/mlops-pipeline --timeout=600s
```

Live proof: after restore, `probe_success` 0→1 (14:03:47Z), `MLflowDown` Resolved
(14:04:04Z), runs **2 → 5 → 10** across the campaign (previous runs survived), a fresh
Job Completed.

## Escalation / known limitations

- **Warning, not critical, by design** — an MLflow outage blocks *new* runs but loses
  **no data**; on this single-operator cluster that is "fix soon", not "wake someone". A
  production availability SLO would make it critical ([alerting.md](../alerting.md#mlflowdown)).
- **Tight memory baseline** — MLflow idles ~85% of its 2Gi limit, so `MLflowMemoryHigh`
  (90%) has limited lead time before the kernel OOM backstop.
- **Mid-run behaviour** — a *transient* ~90s blip is now absorbed by the PR 13 bounded
  retry (5 attempts, ≈65s) so completed training is not discarded; a *persistent* outage
  still fails fast (proven live, [reliability-hardening evidence](../proof/sprint-08-reliability-hardening-evidence.md)).
  A transient failure may leave ≤1 orphan MLflow run / duplicate registered version
  (accepted trade, [ADR-037](../decisions/ADR-037-pipeline-reliability-hardening.md)).
- **⚠️ Never delete the `mlflow-postgres` StatefulSet or its PVC to "restart" MLflow** —
  that destroys experiment history. Scaling the *stateless* `deploy/mlflow` is safe; the
  DB is not.
- **No obsolete DagsHub path** — tracking is fully self-hosted in-cluster (server +
  PostgreSQL + S3); there is **no DagsHub MLflow** and no external tracking credentials
  ([ADR-026](../decisions/ADR-026-in-cluster-mlflow-platform.md)).
