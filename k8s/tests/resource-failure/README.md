# Resource Failure Tests (Sprint 8, PR 12)

## Overview

Validates the platform's operational response to resource exhaustion and pod instability:
- **Scenario A:** OOM (Out-of-Memory) kill and recovery
- **Scenario B:** Crash-loop retry behavior and recovery

These tests run **controlled, ephemeral failure scenarios** on throwaway Job instances against a real cluster, then verify recovery with the normal configuration.

## Prerequisites

- Reachable Kubernetes cluster: `kubectl cluster-info` succeeds
- MLOps workload deployed: `kubectl -n mlops get job/mlops-pipeline`
- Monitoring stack (Prometheus, Grafana) with alert rules configured

## Scenarios

### Scenario A: OOM Kill

**What it does:**
1. Submits a temporary Job with an intentionally **low memory limit (64Mi)** instead of the normal 512Mi
2. The pipeline runs the training stage (the heaviest, memory-intensive stage)
3. Workload exceeds the 64Mi limit, kernel OOMKills the container
4. Job retries up to `backoffLimit=2`, then reaches terminal `Failed` condition

**What it observes:**
- Pod termination reason: `OOMKilled`
- Kubernetes event: container kill
- Prometheus metric: `kube_pod_container_status_last_terminated_reason{reason="OOMKilled"}` = 1
- Alert: `PipelineJobOOMKilled` fires (if alert rules are deployed and Prometheus is scraping)
- Job status: `Failed` condition set

**Expected outcome:**
- Throwaway Job fails deterministically with OOMKilled reason
- Recovery run with normal 512Mi limit succeeds

**Why it matters (ADR-011 § memory-safety):**
- Validates the kernel enforces memory limits
- Proves the platform detects and alerts on OOM kills
- Confirms resource accounting is accurate for this workload

---

### Scenario B: Crash-Loop Retry

**What it does:**
1. Submits a temporary Job with broken configuration that causes the pipeline to fail
   - Injects `PREPROCESS_PARAM=broken-value-causes-failure` env var
   - The preprocess stage fails immediately (deterministic, not transient)
2. The Job retries per `backoffLimit=2`
3. Each pod exits non-zero; the Job controller creates a fresh pod for the next attempt
4. After backoffLimit exhaustion, Job reaches terminal `Failed` condition

**What it observes:**
- Pod state transitions: Running → Failed → waiting/pending → Running again (for each retry)
- Restart count increments on each retry
- Kubernetes events: pod creation, termination, backoff
- Prometheus metrics:
  - `kube_pod_container_status_restarts_total` increments per retry
  - `kube_job_status_failed` increments
- Alert: `KubePodCrashLooping` fires if the failure persists for 15m+ (test verifies the alert *can* fire; a short test run may not sustain 15m)
- Job status: `Failed` condition set

**Expected outcome:**
- Throwaway Job retries and fails as designed
- Recovery run with normal config succeeds

**Why it matters (ADR-011 § reliability, backoffLimit):**
- Proves retry logic works (Job doesn't give up on transient-*looking* failures immediately)
- Validates backoffLimit terminates after N retries (prevents infinite loops)
- Confirms the platform detects repeated instability (CrashLooping alert)
- Tests the `restartPolicy: Never` behavior (each retry = new pod, not in-place restart)

---

## Running the Tests

### Run both scenarios:
```bash
k8s/tests/resource-failure/run.sh
```

### Run only Scenario A (OOM):
```bash
SCENARIO=A k8s/tests/resource-failure/run.sh
```

### Run only Scenario B (Crash-loop):
```bash
SCENARIO=B k8s/tests/resource-failure/run.sh
```

### Use a custom Job and namespace:
```bash
NAMESPACE=custom-ns JOB=custom-job k8s/tests/resource-failure/run.sh
```

### Control wait timeout:
```bash
RESOURCE_FAIL_WAIT=300 k8s/tests/resource-failure/run.sh
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All scenarios passed; Job behaved as expected |
| 1 | A scenario failed (not enough retries, unexpected termination reason, recovery failed) |
| 2 | Environment issue (no cluster, Job not found, kubectl missing) |

## Interpreting Results

### Success Markers

**Scenario A (OOM):**
```
  [PASS] scenario A: pipeline container terminated with OOMKilled
  [PASS] scenario A: Job reached Failed condition (2 failed pods)
```

**Scenario B (Crash-loop):**
```
  [PASS] scenario B: pipeline container failed with deterministic error
  [PASS] scenario B: restart count incremented (2 restarts)
  [PASS] scenario B: Job reached Failed condition (2 failed pods)
```

### Failure Markers

```
  [FAIL] scenario A: expected OOMKilled, got: Completed
          → The Job succeeded despite the low memory limit (the limit didn't enforce, or the
            workload fit within 64Mi — recheck resource measurement)
```

```
  [FAIL] scenario B: restart count did not increment (0)
          → The pod was not restarted; check if restartPolicy is correct or if the failure
            was not actually triggered
```

## Manual Verification

After the script completes, verify the evidence manually:

### Check Pod Status
```bash
kubectl -n mlops get pods -l resource-failure-test=true -o wide
```

### View Pod Events
```bash
kubectl -n mlops describe pod <pod-name>
```

### Query Prometheus
```bash
# OOMKilled metric
curl http://prometheus:9090/api/v1/query?query=kube_pod_container_status_last_terminated_reason

# Restart count
curl http://prometheus:9090/api/v1/query?query=kube_pod_container_status_restarts_total

# Job failed condition
curl http://prometheus:9090/api/v1/query?query=kube_job_failed
```

### Check Alert Rules
```bash
# Verify alert rules are loaded
kubectl -n monitoring get cm prometheus-alerts -o yaml | grep -A 5 PipelineJobOOMKilled

# View Prometheus alert status (if Prometheus UI is exposed)
kubectl -n monitoring port-forward svc/prometheus 9090:9090
# then navigate to http://localhost:9090/alerts
```

### View Grafana Dashboard
```bash
# Port-forward Grafana
kubectl -n monitoring port-forward svc/grafana 3000:3000
# then navigate to http://localhost:3000
# Check "MLOps Pipeline Operations" dashboard for resource/retry panels
```

## Evidence Checklist

For the proof document (docs/proof/sprint-08-resource-failure-tests-evidence.md), capture:

- [ ] Test harness output (both scenarios, exit code 0)
- [ ] Pod termination reason for Scenario A (OOMKilled)
- [ ] Pod restart count for Scenario B (incremented after backoffLimit retries)
- [ ] Kubernetes events for both scenarios (pod creation, failure, backoff)
- [ ] Prometheus queries confirming metrics
- [ ] Alert firing evidence (PipelineJobOOMKilled, KubePodCrashLooping)
- [ ] Recovery run output (unmodified Job Completed successfully)
- [ ] Grafana dashboard screenshots showing resource/retry behavior

## Deferral: EKS Validation

This test harness runs against **local Kubernetes** (Docker Desktop, minikube, kind) to validate the mechanics are sound. When the project provisions EKS again (future sprint):
- Re-run the same harness against the EKS cluster
- Capture production-scale evidence (larger data volumes, real network delays)
- Verify alert firing under realistic load

## Design Decisions (ADR References)

- **ADR-009:** Why a Job (not a Deployment) — Job is finite, one-shot, no keep-alive
- **ADR-011:** Resource limits and retry semantics (backoffLimit, activeDeadlineSeconds)
- **ADR-028:** Queryability contract — Job retained 3600s post-completion for metric scraping
- **docs/alerting.md:** Alert thresholds and runbooks (PipelineJobOOMKilled, KubePodCrashLooping)

## Troubleshooting

### "Job did not reach terminal state within 240s"
- Check if the pod is actually running: `kubectl -n mlops get pods -l job-name=<job-name>`
- Check node resources: `kubectl describe nodes`
- Check pod events: `kubectl -n mlops describe pod <pod-name>`

### "expected OOMKilled, got: Completed"
- The memory limit was not enforced (the workload fit)
- Increase the memory pressure by running a larger dataset or lower the limit further

### "restart count did not increment (0)"
- The preprocess stage might not be failing as expected
- Check pod logs: `kubectl -n mlops logs <pod-name> -c pipeline`
- Verify the env override was applied: `kubectl -n mlops get pod <pod-name> -o yaml | grep -A 20 containers:`

### "no reachable cluster"
- Ensure kubeconfig is set: `kubectl cluster-info`
- Check cluster connectivity: `kubectl get nodes`
