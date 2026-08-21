#!/usr/bin/env bash
# MLflow outage detection & recovery verification (Sprint 8, PR 11).
#
# Proves that an MLflow tracking-server outage is VISIBLE, DIAGNOSABLE and
# RECOVERABLE on a real cluster, and characterises how the pipeline behaves while
# MLflow is down — WITHOUT changing any reliability behaviour (that is PR 13).
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT IT DOES, in order:
#
#   1. BASELINE (must be green first): MLflow Deployment Available, PostgreSQL
#      StatefulSet Ready, the mlflow Service has Endpoints. Prints the availability
#      queries the operator captures (probe_success == 1, MLflowDown not firing,
#      a previous run visible in the MLflow UI).
#
#   2. OUTAGE (safe, reversible): scale the STATELESS `mlflow` Deployment to zero.
#      This is the ONLY mutation. It NEVER touches the `mlflow-postgres` StatefulSet,
#      its PVC, or the S3 artifact bucket — so PostgreSQL metadata and S3 artifacts
#      persist by construction. The original replica count is recorded and restored
#      on ANY exit (success, failure, or Ctrl-C) by an EXIT trap, so the harness can
#      never leave MLflow down.
#
#   3. OBSERVE: the mlflow Service Endpoints drain to empty (no backends), which is
#      what makes the blackbox /health probe fail. Prints the outage-window queries
#      (probe_success == 0, MLflowDown Pending→Firing at 5m).
#
#   4. PIPELINE DURING OUTAGE: submit a throwaway copy of the deployed Job (the
#      committed manifests are NEVER mutated) and observe the CURRENT behaviour:
#        * fetch-dataset init container SUCCEEDS (exit 0) — dataset retrieval does
#          not depend on MLflow;
#        * wait-for-mlflow init container FAILS (non-zero) after its ~300s /health
#          poll, logging "MLflow not ready after …";
#        * the pipeline (dvc repro: preprocess→split→train→evaluate) NEVER starts —
#          so NO model computation is wasted; the Job fails at the tracking GATE.
#      This is the "down at START" timing. The "down MID-RUN" timing (a tracking
#      call inside the train stage raises TrackingError → the whole Job fails AND the
#      completed compute is wasted) is analysed in the proof doc, not injected here
#      (it is timing-dependent; the code path is in src/tracking.py).
#
#   5. RESTORE: scale MLflow back to its original replicas; wait for Available +
#      Endpoints. Assert PostgreSQL stayed Ready throughout (the durable store was
#      untouched).
#
#   6. RECOVER: prints the healthy unmodified run + the persistence/visibility
#      checks (previous runs still visible, a new run succeeds).
#
# This harness does NOT add retries or otherwise change reliability behaviour. Any
# improvement it motivates is recorded as a PR 13 candidate in the proof doc.
#
# PREREQUISITES: a reachable cluster (kubectl configured) with the mlops workload
# and the in-cluster MLflow platform deployed (kubectl apply -k k8s/overlays/<aws|
# local>), and the monitoring stack if capturing the probe/alert evidence.
#
# USAGE:
#   k8s/tests/mlflow-failure/run.sh              # full outage → recover cycle
#   NAMESPACE=mlops MLFLOW_DEPLOY=mlflow JOB=mlops-pipeline k8s/tests/mlflow-failure/run.sh
#   MLFLOW_OUTAGE_WAIT=480 k8s/tests/mlflow-failure/run.sh   # allow for image pull
#   SKIP_PIPELINE=1 k8s/tests/mlflow-failure/run.sh          # outage/observe/restore only
# Exit 0 = the outage was detectable, the pipeline behaved as characterised, and
# MLflow was restored. Exit 1 = behaviour differed from the reliability contract
# (see [FAIL] lines). Exit 2 = environment/precondition problem. MLflow is restored
# on every exit path regardless.
set -uo pipefail

NS="${NAMESPACE:-mlops}"
MLFLOW_DEPLOY="${MLFLOW_DEPLOY:-mlflow}"
PG_STS="${PG_STS:-mlflow-postgres}"
SVC="${MLFLOW_SVC:-mlflow}"
JOB="${JOB:-mlops-pipeline}"
PROBE_JOB="${PROBE_JOB:-blackbox-mlflow-health}"
FETCH_INIT="fetch-dataset"
GATE_INIT="wait-for-mlflow"
PIPELINE="pipeline"
# How long to wait for the wait-for-mlflow gate to fail. That gate polls /health 60×
# with a 3s connect timeout + 5s sleep, so it exits non-zero after ~300s if each probe
# is refused fast, but up to ~480s if the connections instead hang to their timeout
# (a Service with zero Endpoints can behave either way depending on kube-proxy/CNI).
# Default comfortably above the 480s worst case + image-pull slack; operator-overridable.
OUTAGE_WAIT="${MLFLOW_OUTAGE_WAIT:-540}"
# Shorter wait for the scale-to-zero / scale-up transitions (endpoints drain/appear).
SCALE_WAIT="${MLFLOW_SCALE_WAIT:-120}"
# Per-invocation id → RFC1123-valid, collision-free throwaway Job name on a rerun.
RUN_ID="${MLFLOW_FAIL_RUN_ID:-$$-$RANDOM}"
THROWAWAY="${JOB}-mlflowout-${RUN_ID}"

pass=0
fail=0
# Outage bookkeeping: the trap uses these to know whether — and to what — to restore.
ORIG_REPLICAS=""
SCALED_DOWN=0

log() { printf '%s\n' "$*"; }
ok() {
  pass=$((pass + 1))
  log "  [PASS] $1"
}
bad() {
  fail=$((fail + 1))
  log "  [FAIL] $1${2:+ -> $2}"
}

# ─────────────────────────────────────────────────────────────────────────────
# SAFETY: restore MLflow (and clean up the throwaway Job) on ANY exit path. Installed
# BEFORE the first mutation so an early failure or Ctrl-C can never leave MLflow down.
restore_mlflow() {
  # Delete the throwaway Job first (best-effort; independent of the scale restore).
  kubectl -n "$NS" delete job "$THROWAWAY" --ignore-not-found --wait=false \
    >/dev/null 2>&1 || true
  if [ "$SCALED_DOWN" -eq 1 ]; then
    log ""
    log "Restoring MLflow to ${ORIG_REPLICAS:-1} replica(s)…"
    kubectl -n "$NS" scale deploy "$MLFLOW_DEPLOY" \
      --replicas="${ORIG_REPLICAS:-1}" >/dev/null 2>&1 || true
    # Best-effort wait so the operator leaves the cluster healthy; failure to become
    # Ready here is reported, not swallowed, but never blocks the trap.
    if kubectl -n "$NS" rollout status deploy "$MLFLOW_DEPLOY" \
      --timeout="${SCALE_WAIT}s" >/dev/null 2>&1; then
      log "MLflow restored (Deployment rolled out)."
    else
      log "WARNING: MLflow did not report Ready within ${SCALE_WAIT}s after restore."
      log "         Check: kubectl -n ${NS} get deploy ${MLFLOW_DEPLOY}"
    fi
    SCALED_DOWN=0
  fi
}
# EXIT covers every normal/`exit` path. INT/TERM additionally ABORT: a trapped signal
# does not stop the script on its own, so without an explicit exit the harness would
# restore MLflow and then keep running the remaining steps against a now-healthy server
# and emit misleading results. The exit re-fires EXIT, but restore_mlflow is idempotent
# (SCALED_DOWN is cleared after the first restore), so the second call is a no-op.
trap restore_mlflow EXIT
trap 'restore_mlflow; exit 130' INT TERM

# ─────────────────────────────────────────────────────────────────────────────
command -v kubectl >/dev/null 2>&1 || {
  log "ERROR: kubectl not found on PATH."
  exit 2
}
kubectl cluster-info >/dev/null 2>&1 || {
  log "ERROR: no reachable cluster (kubectl cluster-info failed)."
  exit 2
}
for obj in "deploy/$MLFLOW_DEPLOY" "job/$JOB"; do
  kubectl -n "$NS" get "$obj" >/dev/null 2>&1 || {
    log "ERROR: $NS/$obj not found. Deploy the workload first:"
    log "       kubectl apply -k k8s/overlays/<aws|local>"
    exit 2
  }
done

# Endpoints IPs for a Service (empty string ⇒ no ready backends).
svc_endpoint_ips() {
  kubectl -n "$NS" get endpoints "$1" \
    -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null
}

# Is the mlflow-postgres StatefulSet fully Ready? (durable-store health proxy)
pg_ready() {
  local ready replicas
  ready="$(kubectl -n "$NS" get statefulset "$PG_STS" \
    -o jsonpath='{.status.readyReplicas}' 2>/dev/null)"
  replicas="$(kubectl -n "$NS" get statefulset "$PG_STS" \
    -o jsonpath='{.spec.replicas}' 2>/dev/null)"
  [ -n "$ready" ] && [ "$ready" = "${replicas:-1}" ]
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. BASELINE — must be green before we break anything.
log "══════════════════════════════════════════════════════════════════════"
log "1. BASELINE (healthy MLflow platform)"

# Record the ORIGINAL desired replica count so restore targets exactly that, never a
# hardcoded default. A failed/blank read must NOT be silently coerced to 1 (that could
# under-restore a deployment legitimately running >1) — treat it as a precondition
# failure. This runs BEFORE any mutation, so exiting here leaves the cluster untouched.
ORIG_REPLICAS="$(kubectl -n "$NS" get deploy "$MLFLOW_DEPLOY" \
  -o jsonpath='{.spec.replicas}' 2>/dev/null)"
if ! { [ -n "$ORIG_REPLICAS" ] && [ "$ORIG_REPLICAS" -ge 1 ] 2>/dev/null; }; then
  bad "could not read a valid replica count for ${MLFLOW_DEPLOY} (got '${ORIG_REPLICAS}')" \
    "refusing to run without a known restore target"
  log "RESULT: FAIL (precondition). Nothing was changed."
  exit 2
fi

mlflow_ready="$(kubectl -n "$NS" get deploy "$MLFLOW_DEPLOY" \
  -o jsonpath='{.status.readyReplicas}' 2>/dev/null)"
if [ -n "$mlflow_ready" ] && [ "$mlflow_ready" -ge 1 ] 2>/dev/null; then
  ok "MLflow Deployment Ready (${mlflow_ready} replica(s), spec=${ORIG_REPLICAS})"
else
  bad "MLflow Deployment is not Ready at baseline — cannot run the outage test" \
    "fix the platform first"
  log "RESULT: FAIL (precondition). Nothing was changed."
  exit 2
fi

if pg_ready; then
  ok "PostgreSQL StatefulSet ${PG_STS} Ready (the durable store is healthy)"
else
  bad "PostgreSQL StatefulSet ${PG_STS} not Ready at baseline" "fix it first"
  log "RESULT: FAIL (precondition). Nothing was changed."
  exit 2
fi

base_eps="$(svc_endpoint_ips "$SVC")"
if [ -n "$base_eps" ]; then
  ok "Service ${SVC} has Endpoints (traffic has a backend)"
else
  bad "Service ${SVC} has no Endpoints at baseline" "MLflow is not actually serving"
  log "RESULT: FAIL (precondition). Nothing was changed."
  exit 2
fi

log ""
log "  Operator — capture these BASELINE signals for the report:"
log "    * Prometheus : max(probe_success{job=\"${PROBE_JOB}\"}) == 1"
log "    * Alerts     : MLflowDown NOT firing (Prometheus /alerts)"
log "    * MLflow UI  : a previous run is visible (kubectl -n ${NS} port-forward"
log "                   svc/${SVC} 5000:5000, then browse / curl the experiments)."

# ─────────────────────────────────────────────────────────────────────────────
# 2. OUTAGE — scale the stateless Deployment to zero (reversible; DB/S3 untouched).
log ""
log "══════════════════════════════════════════════════════════════════════"
log "2. OUTAGE — scaling deploy/${MLFLOW_DEPLOY} to 0 (PostgreSQL & S3 untouched)"
SCALED_DOWN=1
if ! kubectl -n "$NS" scale deploy "$MLFLOW_DEPLOY" --replicas=0 >/dev/null 2>&1; then
  bad "could not scale ${MLFLOW_DEPLOY} to 0" "check RBAC / cluster state"
  log "RESULT: FAIL. The EXIT trap will attempt to restore MLflow."
  exit 1
fi

# Wait for the Service Endpoints to drain — the observable outage signal. A transient
# `kubectl get` error also yields empty stdout, which would look identical to "drained";
# to avoid a false-positive we require the read to SUCCEED (rc 0) AND return no IPs, and
# additionally confirm the Deployment reports zero ready replicas (the authoritative
# scale signal), so a single flaky API call on the first poll cannot declare the outage.
drained=0
deadline=$((SECONDS + SCALE_WAIT))
while [ "$SECONDS" -lt "$deadline" ]; do
  eps="$(kubectl -n "$NS" get endpoints "$SVC" \
    -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null)"
  eps_rc=$?
  ready="$(kubectl -n "$NS" get deploy "$MLFLOW_DEPLOY" \
    -o jsonpath='{.status.readyReplicas}' 2>/dev/null)"
  if [ "$eps_rc" -eq 0 ] && [ -z "$eps" ] && [ -z "${ready:-}" ]; then
    drained=1
    break
  fi
  sleep 3
done

# ─────────────────────────────────────────────────────────────────────────────
# 3. OBSERVE the outage.
log ""
log "══════════════════════════════════════════════════════════════════════"
log "3. OBSERVE the outage"
if [ "$drained" -eq 1 ]; then
  ok "Service ${SVC} Endpoints drained to empty (no backend → /health unreachable)"
else
  bad "Service ${SVC} still has Endpoints ${SCALE_WAIT}s after scale-to-zero" \
    "outage not established"
fi
if pg_ready; then
  ok "PostgreSQL StatefulSet still Ready during the outage (durable store intact)"
else
  bad "PostgreSQL StatefulSet went unready during the MLflow outage" \
    "unexpected — the outage must not affect the DB"
fi
log ""
log "  Operator — capture these OUTAGE-WINDOW signals:"
log "    * Prometheus : max(probe_success{job=\"${PROBE_JOB}\"}) == 0"
log "    * Alerts     : MLflowDown Pending → Firing at its 5m 'for:' (warning)."
log "    * Grafana    : 'MLflow Platform Health' → availability gauge red; the"
log "                   PostgreSQL panels stay green (the DB never went down)."

# ─────────────────────────────────────────────────────────────────────────────
# 4. PIPELINE DURING OUTAGE — observe the CURRENT behaviour (no changes made).
if [ "${SKIP_PIPELINE:-0}" != "1" ]; then
  log ""
  log "══════════════════════════════════════════════════════════════════════"
  log "4. PIPELINE DURING OUTAGE — submitting a throwaway Job (real ones untouched)"
  log "   (the wait-for-mlflow gate polls /health for up to ~480s before failing; wait)"

  kubectl -n "$NS" delete job "$THROWAWAY" --ignore-not-found --wait=false \
    >/dev/null 2>&1 || true

  # Render the deployed Job to a clean, submittable spec under a throwaway name —
  # SAME image, init containers, volumes: the identical run, just renamed. No env
  # override: the outage itself is the fault under test.
  spec="$(kubectl -n "$NS" get job "$JOB" -o json | python -c '
import json, sys
j = json.load(sys.stdin)
name = sys.argv[1]
tpl = j["spec"]["template"]
j["spec"].pop("selector", None)
for labels in (j["metadata"].get("labels", {}), tpl.get("metadata", {}).get("labels", {})):
    for k in list(labels):
        if k in ("controller-uid", "batch.kubernetes.io/controller-uid", "job-name",
                 "batch.kubernetes.io/job-name"):
            labels.pop(k, None)
out = {
    "apiVersion": "batch/v1",
    "kind": "Job",
    "metadata": {"name": name, "namespace": j["metadata"]["namespace"],
                 "labels": {"mlflow-failure-test": "true"}},
    "spec": {k: j["spec"][k] for k in j["spec"]
             if k in ("activeDeadlineSeconds", "ttlSecondsAfterFinished", "template")},
}
# backoffLimit 0: this throwaway only needs ONE attempt to observe the gate fail. Not
# retrying keeps the pod set to a single, unambiguous pod (no newest-pod race with a
# retry pod) and avoids paying the ~480s gate wait more than once.
out["spec"]["backoffLimit"] = 0
json.dump(out, sys.stdout)
' "$THROWAWAY")"

  if [ -z "$spec" ]; then
    bad "could not render the deployed Job (kubectl/JSON error)"
  elif ! printf '%s' "$spec" | kubectl apply -f - >/dev/null 2>&1; then
    bad "could not submit the throwaway Job"
  else
    # Wait until the wait-for-mlflow gate reaches a terminated state (or timeout).
    pod=""
    gate_exit=""
    deadline=$((SECONDS + OUTAGE_WAIT))
    while [ "$SECONDS" -lt "$deadline" ]; do
      pod="$(kubectl -n "$NS" get pods -l job-name="$THROWAWAY" \
        --sort-by=.metadata.creationTimestamp \
        -o jsonpath='{.items[-1:].metadata.name}' 2>/dev/null)"
      if [ -n "$pod" ]; then
        gate_exit="$(kubectl -n "$NS" get pod "$pod" \
          -o jsonpath="{.status.initContainerStatuses[?(@.name=='$GATE_INIT')].state.terminated.exitCode}" \
          2>/dev/null)"
        [ -n "$gate_exit" ] && break
      fi
      sleep 5
    done

    if [ -z "$pod" ] || [ -z "$gate_exit" ]; then
      bad "the ${GATE_INIT} gate did not terminate within ${OUTAGE_WAIT}s" \
        "check image pull / cluster state"
    else
      # 4a. fetch-dataset must have SUCCEEDED — retrieval is independent of MLflow.
      fetch_exit="$(kubectl -n "$NS" get pod "$pod" \
        -o jsonpath="{.status.initContainerStatuses[?(@.name=='$FETCH_INIT')].state.terminated.exitCode}" \
        2>/dev/null)"
      if [ "${fetch_exit:-1}" = "0" ]; then
        ok "${FETCH_INIT} succeeded (exit 0) — dataset retrieval is unaffected by the outage"
      else
        bad "${FETCH_INIT} did not succeed (exit ${fetch_exit:-<none>})" \
          "expected the dataset step to be independent of MLflow"
      fi

      # 4b. wait-for-mlflow must have FAILED non-zero, with its distinct message.
      if [ "${gate_exit}" -ne 0 ] 2>/dev/null; then
        ok "${GATE_INIT} failed (exit ${gate_exit}) — the outage is detected at the gate"
      else
        bad "${GATE_INIT} exited 0 despite the MLflow outage" "gate did not detect it"
      fi
      gate_logs="$(kubectl -n "$NS" logs "$pod" -c "$GATE_INIT" 2>/dev/null)"
      if printf '%s' "$gate_logs" | grep -qi "MLflow not ready"; then
        ok "${GATE_INIT} logs carry the clear error (\"MLflow not ready after …\")"
      else
        bad "expected outage error not found in ${GATE_INIT} logs" \
          "wanted \"MLflow not ready\""
      fi

      # 4c. The pipeline container must NEVER have started — no wasted computation.
      # jsonpath a SCALAR sub-field, not the whole `.state` map: `kubectl -o jsonpath`
      # renders a map with Go's `%v` (`map[waiting:map[reason:PodInitializing]]`), which
      # has no quotes, so grepping it for '"running"' can never match and the check would
      # be dead (always PASS). `startedAt` exists on both running and terminated states
      # and on neither of a waiting (never-started) container, so a non-empty result here
      # means the pipeline container actually started.
      pstate="$(kubectl -n "$NS" get pod "$pod" \
        -o jsonpath="{.status.containerStatuses[?(@.name=='$PIPELINE')].state.running.startedAt}{.status.containerStatuses[?(@.name=='$PIPELINE')].state.terminated.startedAt}" \
        2>/dev/null)"
      if [ -n "$pstate" ]; then
        bad "${PIPELINE} container STARTED despite the MLflow outage" \
          "unexpected — the gate should block it before any computation"
      else
        ok "${PIPELINE} never started — no model computation was wasted (fail at the gate)"
      fi

      log ""
      log "  ── captured ${GATE_INIT} logs (during outage) ──"
      printf '%s\n' "$gate_logs" | sed 's/^/    /'
    fi
  fi

  kubectl -n "$NS" delete job "$THROWAWAY" --ignore-not-found --wait=false \
    >/dev/null 2>&1 || true

  log ""
  log "  Operator — the run-level signal (needs the REAL Job, not a throwaway):"
  log "    apply the workload while MLflow is down so a real Job exhausts backoffLimit"
  log "    → PipelineJobFailed fires ALONGSIDE MLflowDown. The correlation (both"
  log "    firing) is the diagnosis; see docs/alerting.md#mlflowdown."
else
  log ""
  log "4. PIPELINE DURING OUTAGE — skipped (SKIP_PIPELINE=1)."
fi

# ─────────────────────────────────────────────────────────────────────────────
# 5. RESTORE — scale MLflow back and verify recovery (also guaranteed by the trap).
log ""
log "══════════════════════════════════════════════════════════════════════"
log "5. RESTORE MLflow (scale back to ${ORIG_REPLICAS})"
# Disarm the trap ONLY on POSITIVE proof the scale-up took effect. `rollout status`
# alone is not proof: it returns success instantly for a Deployment sitting at 0/0
# (updated==available==spec==0), so if the scale command was rejected (quota, webhook,
# RBAC, API blip — all swallowed by `|| true`) it would falsely report "rolled out"
# while MLflow is still down, AND clear SCALED_DOWN — disarming the safety net. So we
# require: the scale command itself succeeded, rollout status is green, AND the live
# spec.replicas actually equals ORIG_REPLICAS (≥1). Only then is it safe to tell the
# trap there is nothing left to restore.
scaled_ok=0
if kubectl -n "$NS" scale deploy "$MLFLOW_DEPLOY" --replicas="$ORIG_REPLICAS" \
  >/dev/null 2>&1 && kubectl -n "$NS" rollout status deploy "$MLFLOW_DEPLOY" \
  --timeout="${SCALE_WAIT}s" >/dev/null 2>&1; then
  cur_spec="$(kubectl -n "$NS" get deploy "$MLFLOW_DEPLOY" \
    -o jsonpath='{.spec.replicas}' 2>/dev/null)"
  if [ "$cur_spec" = "$ORIG_REPLICAS" ] && [ "${cur_spec:-0}" -ge 1 ] 2>/dev/null; then
    scaled_ok=1
  fi
fi
if [ "$scaled_ok" -eq 1 ]; then
  SCALED_DOWN=0  # positively restored in-flow; the trap now has nothing to do.
  ok "MLflow Deployment rolled back out to ${ORIG_REPLICAS} (Ready)"
else
  bad "MLflow did not verifiably return to ${ORIG_REPLICAS} replica(s) after restore" \
    "leaving SCALED_DOWN=1 so the EXIT trap retries the scale-up"
fi

# Wait for Endpoints to reappear.
restored_eps=0
deadline=$((SECONDS + SCALE_WAIT))
while [ "$SECONDS" -lt "$deadline" ]; do
  if [ -n "$(svc_endpoint_ips "$SVC")" ]; then
    restored_eps=1
    break
  fi
  sleep 3
done
if [ "$restored_eps" -eq 1 ]; then
  ok "Service ${SVC} Endpoints restored (a backend is serving again)"
else
  bad "Service ${SVC} still has no Endpoints ${SCALE_WAIT}s after restore"
fi
if pg_ready; then
  ok "PostgreSQL StatefulSet still Ready — metadata persisted across the outage"
else
  bad "PostgreSQL StatefulSet not Ready after restore" "investigate"
fi

log ""
log "  Operator — verify RECOVERY & persistence for the report:"
log "    * Prometheus : probe_success{job=\"${PROBE_JOB}\"} back to 1; MLflowDown Resolved."
log "    * Persistence: previous runs STILL visible in the MLflow UI (PostgreSQL"
log "                   metadata persisted); S3 artifacts still listed."
log "    * New run    : an unmodified pipeline Job Completes:"
log "        kubectl -n ${NS} delete job ${JOB} && kubectl apply -k k8s/overlays/<aws|local>"
log "        kubectl -n ${NS} wait --for=condition=complete job/${JOB} --timeout=600s"

# ─────────────────────────────────────────────────────────────────────────────
log ""
log "Summary: ${pass} passed, ${fail} failed."
if [ "$fail" -ne 0 ]; then
  log "RESULT: FAIL — MLflow outage behaviour differed from the contract (see [FAIL])."
  exit 1
fi
log "RESULT: PASS — the MLflow outage was detected, the pipeline failed at the gate"
log "        with no wasted computation, and MLflow was restored with its data intact."
exit 0
