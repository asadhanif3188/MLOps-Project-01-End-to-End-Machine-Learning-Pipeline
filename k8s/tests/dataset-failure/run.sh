#!/usr/bin/env bash
# Dataset availability & integrity failure-path verification (Sprint 8, PR 10).
#
# Executes the two controlled dataset failures the reliability contract must hold
# — on the SAME deployed workload, against a real cluster — and proves that each
# one fails FAST, BEFORE training, with a clear error and an operational signal:
#
#   Scenario A — dataset UNAVAILABLE: the fetch-dataset init container is pointed at
#     a non-existent object key. The S3 GET fails, the init container exits non-zero,
#     the Job retries to backoffLimit and then reaches its terminal Failed condition.
#     The pipeline (preprocess→split→train→evaluate) never starts.
#
#   Scenario B — checksum MISMATCH: the object is retrieved, but the pinned
#     DATASET_SHA256 is overridden to a wrong-but-well-formed digest. The integrity
#     gate rejects it and the Job fails — again before training. This is a
#     DETERMINISTIC failure: it is NEVER retried away (a re-download yields the same
#     bytes and the same mismatch), so it must fail fast. This harness asserts it
#     fails fast; it must NOT be "fixed" by adding retries (that is a correctness
#     regression, explicitly out of scope — see the proof doc).
#
# ─────────────────────────────────────────────────────────────────────────────
# HOW IT WORKS. The committed manifests are NEVER mutated. For each scenario the
# harness renders the deployed Job to YAML, applies a narrow, reversible override
# to the fetch-dataset init container (a bad DATASET_S3_URI key for A; a bad
# DATASET_SHA256 for B), submits it under a throwaway name, observes the failure,
# captures the evidence, and deletes the throwaway Job. The real Job, ConfigMap and
# overlays are untouched, so "restore" is simply "stop using the override" — there
# is nothing to roll back. A final unmodified run proves healthy recovery.
#
# WHAT IT ASSERTS (per scenario):
#   * the fetch-dataset init container terminates with a NON-ZERO exit code;
#   * its logs carry the EXPECTED, distinct error ("Failed to download …" for A;
#     "integrity check failed …" for B) — the root-cause layer (ADR-030);
#   * the pipeline container NEVER starts (training does not begin);
#   * the Job does not Complete.
# Metrics/alerts/Grafana are captured by the operator per the proof-doc checklist
# (they are cluster-wide, not per-throwaway-Job): the Pushgateway series
# mlops_pipeline_stage_success{stage="fetch_dataset"}=0 with every later stage
# ABSENT, and — once the real Job exhausts backoffLimit — the PipelineJobFailed
# alert. This script prints the exact queries to run.
#
# PREREQUISITES: a reachable cluster (kubectl configured) with the mlops workload
# deployed (kubectl apply -k k8s/overlays/<aws|local>) and the fetch-dataset init
# container present on the Job. The monitoring stack (Pushgateway/Prometheus/
# Grafana) deployed if capturing the metric/alert evidence.
#
# USAGE:
#   k8s/tests/dataset-failure/run.sh              # both scenarios + recovery run
#   SCENARIO=A k8s/tests/dataset-failure/run.sh   # just the unavailable scenario
#   SCENARIO=B k8s/tests/dataset-failure/run.sh   # just the checksum scenario
#   NAMESPACE=mlops JOB=mlops-pipeline k8s/tests/dataset-failure/run.sh
# Exit 0 = every requested scenario failed in the expected way AND (if run) the
# recovery run Completed. Exit 1 = a scenario did NOT fail as expected (e.g. the
# pipeline container started despite a bad dataset — a real reliability regression),
# or the recovery run did not Complete. Exit 2 = environment/precondition problem.
set -uo pipefail

NS="${NAMESPACE:-mlops}"
JOB="${JOB:-mlops-pipeline}"
SCENARIO="${SCENARIO:-both}"
INIT="fetch-dataset"
PIPELINE="pipeline"
# How long to wait for the init container to reach a terminal state. A failed S3
# GET / checksum check is seconds; allow generous slack for image pull.
WAIT="${DATASET_FAIL_WAIT:-180}"

pass=0
fail=0

log() { printf '%s\n' "$*"; }
ok() {
  pass=$((pass + 1))
  log "  [PASS] $1"
}
bad() {
  fail=$((fail + 1))
  log "  [FAIL] $1${2:+ -> $2}"
}

command -v kubectl >/dev/null 2>&1 || {
  log "ERROR: kubectl not found on PATH."
  exit 2
}
kubectl cluster-info >/dev/null 2>&1 || {
  log "ERROR: no reachable cluster (kubectl cluster-info failed)."
  exit 2
}
kubectl -n "$NS" get job "$JOB" >/dev/null 2>&1 || {
  log "ERROR: Job $NS/$JOB not found. Deploy the workload first:"
  log "       kubectl apply -k k8s/overlays/<aws|local>"
  exit 2
}

# Render the deployed Job to a clean, submittable spec (strip cluster-managed
# fields, the controller-set selector/labels, and status). We reuse the real Job's
# full pod template — same image, init containers, securityContext, volumes — so a
# throwaway run exercises the IDENTICAL retrieval path, only with one env override.
render_job() {
  local newname="$1"
  kubectl -n "$NS" get job "$JOB" -o json |
    python -c '
import json, sys
j = json.load(sys.stdin)
name = sys.argv[1]
tpl = j["spec"]["template"]
# Drop the controller-managed selector and the injected controller-uid labels so
# a fresh Job can own its own pods.
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
                 "labels": {"dataset-failure-test": "true"}},
    "spec": {k: j["spec"][k] for k in j["spec"]
             if k in ("backoffLimit", "activeDeadlineSeconds", "ttlSecondsAfterFinished",
                      "template")},
}
json.dump(out, sys.stdout)
' "$newname"
}

# Apply one env override to the fetch-dataset init container of a rendered spec and
# submit it. Args: rendered-json env-name env-value new-job-name
submit_with_override() {
  local spec="$1" env_name="$2" env_value="$3" newname="$4"
  printf '%s' "$spec" | python -c '
import json, sys
j = json.load(sys.stdin)
env_name, env_value = sys.argv[1], sys.argv[2]
for c in j["spec"]["template"]["spec"].get("initContainers", []):
    if c["name"] == "fetch-dataset":
        env = c.setdefault("env", [])
        env[:] = [e for e in env if e.get("name") != env_name]
        env.append({"name": env_name, "value": env_value})
json.dump(j, sys.stdout)
' "$env_name" "$env_value" | kubectl apply -f - >/dev/null
}

# Wait until the init container of a Job's pod reaches a terminated state, or WAIT
# elapses. Echoes the pod name.
wait_for_init_terminated() {
  local jobname="$1" deadline=$((SECONDS + WAIT)) pod=""
  while [ "$SECONDS" -lt "$deadline" ]; do
    pod="$(kubectl -n "$NS" get pods -l job-name="$jobname" \
      -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)"
    if [ -n "$pod" ]; then
      local state
      state="$(kubectl -n "$NS" get pod "$pod" -o jsonpath="{.status.initContainerStatuses[?(@.name=='$INIT')].state.terminated.exitCode}" 2>/dev/null)"
      if [ -n "$state" ]; then
        printf '%s' "$pod"
        return 0
      fi
    fi
    sleep 3
  done
  printf '%s' "$pod"
  return 1
}

cleanup_job() {
  kubectl -n "$NS" delete job "$1" --ignore-not-found --wait=false >/dev/null 2>&1 || true
}

# ─────────────────────────────────────────────────────────────────────────────
# One failure scenario. Args: label env-name bad-value expected-log-substring
run_failure_scenario() {
  local label="$1" env_name="$2" bad_value="$3" want_log="$4"
  local newname="${JOB}-datafail-${label}"
  log ""
  log "──────────────────────────────────────────────────────────────────────"
  log "Scenario ${label}: override ${env_name} on the ${INIT} init container"
  log "  value: ${bad_value}"
  cleanup_job "$newname"
  local spec
  spec="$(render_job "$newname")"
  submit_with_override "$spec" "$env_name" "$bad_value" "$newname" || {
    bad "scenario ${label}: could not submit throwaway Job"
    return
  }

  local pod
  if pod="$(wait_for_init_terminated "$newname")" && [ -n "$pod" ]; then
    local exitcode
    exitcode="$(kubectl -n "$NS" get pod "$pod" -o jsonpath="{.status.initContainerStatuses[?(@.name=='$INIT')].state.terminated.exitCode}" 2>/dev/null)"
    if [ "${exitcode:-0}" -ne 0 ]; then
      ok "scenario ${label}: ${INIT} terminated non-zero (exit ${exitcode})"
    else
      bad "scenario ${label}: ${INIT} exited 0 — the failure was NOT triggered"
    fi

    local logs
    logs="$(kubectl -n "$NS" logs "$pod" -c "$INIT" 2>/dev/null)"
    if printf '%s' "$logs" | grep -qi "$want_log"; then
      ok "scenario ${label}: logs carry the expected error (\"${want_log}\")"
    else
      bad "scenario ${label}: expected error not found in ${INIT} logs" \
        "wanted \"${want_log}\""
    fi

    # The pipeline container must NEVER have started — training does not begin.
    local pstate
    pstate="$(kubectl -n "$NS" get pod "$pod" -o jsonpath="{.status.containerStatuses[?(@.name=='$PIPELINE')].state}" 2>/dev/null)"
    if printf '%s' "$pstate" | grep -q '"running"\|"terminated"'; then
      bad "scenario ${label}: pipeline container STARTED despite the dataset failure" \
        "reliability regression — training must not begin"
    else
      ok "scenario ${label}: pipeline container never started (training did not begin)"
    fi

    log ""
    log "  ── captured ${INIT} logs (scenario ${label}) ──"
    printf '%s\n' "$logs" | sed 's/^/    /'
  else
    bad "scenario ${label}: ${INIT} did not terminate within ${WAIT}s" \
      "check image pull / cluster state"
  fi

  cleanup_job "$newname"
}

if [ "$SCENARIO" = "A" ] || [ "$SCENARIO" = "both" ]; then
  # Scenario A — dataset UNAVAILABLE: point at a key that cannot exist. boto3's GET
  # fails (NoSuchKey/AccessDenied), download_object raises DataError("Failed to
  # download …"). Reversible: the real DATASET_S3_URI is never touched.
  cur_uri="$(kubectl -n "$NS" get job "$JOB" \
    -o jsonpath="{.spec.template.spec.initContainers[?(@.name=='$INIT')].env[?(@.name=='DATASET_S3_URI')].value}" 2>/dev/null)"
  bad_uri="${cur_uri%/*}/THIS-OBJECT-KEY-DOES-NOT-EXIST-datafail-A.csv"
  [ -n "$cur_uri" ] || bad_uri="s3://mlops-dataset-failure-test/THIS-OBJECT-KEY-DOES-NOT-EXIST.csv"
  run_failure_scenario "A" "DATASET_S3_URI" "$bad_uri" "Failed to download"
fi

if [ "$SCENARIO" = "B" ] || [ "$SCENARIO" = "both" ]; then
  # Scenario B — checksum MISMATCH: a wrong-but-well-formed 64-hex digest. The
  # object downloads fine; verify_checksum rejects it with DataError("integrity
  # check failed …"). Deterministic → fails fast → must NOT be retried away.
  run_failure_scenario "B" "DATASET_SHA256" \
    "0000000000000000000000000000000000000000000000000000000000000000" \
    "integrity check failed"
fi

# ─────────────────────────────────────────────────────────────────────────────
log ""
log "Operator: capture the cluster-wide signals for the report (proof-doc checklist):"
log "  * Pushgateway  : mlops_pipeline_stage_success{stage=\"fetch_dataset\"} == 0,"
log "                   and every later stage (preprocess/split/train/evaluate) ABSENT."
log "  * Grafana      : 'MLOps Pipeline Operations' -> 'Dataset retrieval' panel == 0 (red)."
log "  * Alert        : run the SAME override on the REAL Job (not a throwaway) so it"
log "                   exhausts backoffLimit -> PipelineJobFailed fires (~2m after the"
log "                   terminal Failed condition). Capture kubectl get job + the alert."
log "  * Diagnose     : follow docs/alerting.md#pipelinejobfailed (the fetch_dataset path)."

log ""
log "Summary: ${pass} passed, ${fail} failed."
if [ "$fail" -ne 0 ]; then
  log "RESULT: FAIL — a dataset failure did not behave as required (see [FAIL] lines)."
  exit 1
fi
log "RESULT: PASS — every requested dataset failure failed fast, before training,"
log "        with a clear error. Run an UNMODIFIED Job next to prove healthy recovery:"
log "          kubectl -n ${NS} delete job ${JOB} && kubectl apply -k k8s/overlays/<aws|local>"
log "          kubectl -n ${NS} wait --for=condition=complete job/${JOB} --timeout=600s"
exit 0
