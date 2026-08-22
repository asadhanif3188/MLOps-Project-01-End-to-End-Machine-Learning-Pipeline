# Runbook 4 — Dataset checksum / integrity failure

> **Part of the [Operational Runbooks](README.md).** Alert:
> [`PipelineJobFailed`](../alerting.md#pipelinejobfailed) (critical, 2m) — the
> integrity-gate sub-case. Parent: [Pipeline failure](pipeline-failure.md). Sibling:
> [Dataset retrieval failure](dataset-retrieval-failure.md). Evidence:
> [PR 10 Scenario B](../proof/sprint-08-dataset-failure-tests-evidence.md),
> [live-EKS §6](../proof/sprint-08-live-eks-evidence.md#6-pr-10--12--failure-paths--alerts)
> (`integrity check failed … expected <bad>, got ee5b0c92…` — the real digest — exit 1,
> pipeline never started).

## Purpose

Diagnose and recover when the dataset object **is retrieved** but its SHA-256 does **not
match** the pinned `DATASET_SHA256`. This is a **deterministic** failure — re-downloading
yields the same bytes and the same mismatch — so it must fail fast and is **never retried
away**.

## Symptoms

- `PipelineJobFailed` firing; Job Failed with `BackoffLimitExceeded`.
- `fetch-dataset` init container `Terminated` **non-zero (exit 1)**; `pipeline` container
  never Running.
- Pod log (`-c fetch-dataset`) shows the **retrieve-then-reject** ordering:
  ```
  Dataset retrieved: s3://<bucket>/<key> (N bytes); verifying integrity
  Dataset integrity check failed: expected <pinned>, got <actual>
  ```
- `mlops_pipeline_stage_success{stage="fetch_dataset"}` is `0`; later stages absent.

## Detection

Same alert and stage signal as [retrieval failure](dataset-retrieval-failure.md) — the
**log message** is what separates the two modes:

```bash
pod=$(kubectl -n mlops get pods -l job-name=mlops-pipeline \
  --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}')
kubectl -n mlops logs "$pod" -c fetch-dataset | grep -i integrity
```

The presence of `Dataset retrieved: …` **before** the failure is the proof the object
was reachable — the problem is the *bytes*, not the *access*.

## Initial checks

```bash
# 1. Confirm the two log lines (retrieved, then integrity failed) and read both digests.
kubectl -n mlops logs "$pod" -c fetch-dataset

# 2. What SHA-256 is pinned, and what URI is it verifying?
kubectl -n mlops get cm mlops-pipeline-config \
  -o jsonpath='SHA={.data.DATASET_SHA256}{"\n"}URI={.data.DATASET_S3_URI}{"\n"}'
```

## Diagnosis

A mismatch has exactly two legitimate root causes — decide which **deliberately**:

1. **The object in S3 was swapped or corrupted** — the pin is correct, the bytes are
   wrong. Recompute the object's digest and compare:
   ```bash
   aws s3 cp "$(terraform -chdir=terraform output -raw dataset_s3_uri)" - | sha256sum
   ```
   If this differs from the known-good `ee5b0c92…` digest, the object is the problem.
2. **The pin is stale** — the dataset was intentionally updated to a new version but
   `DATASET_SHA256` was not updated with it. The new object's digest is legitimate and
   provenance-tracked; the pin simply lags.

> **Never "fix" this by disabling the check or adding retries.** A deterministic checksum
> mismatch cannot be retried away — the fail-fast integrity gate is the intended
> behaviour ([ADR-027](../decisions/ADR-027-s3-dataset-runtime-retrieval.md)). PR 13
> explicitly **declined** to add a checksum retry (rule 1).

## Likely causes

| Cause | Correct response |
|---|---|
| Object swapped / corrupted / truncated upload | **Restore the correct bytes** to S3; keep the pin. |
| Dataset intentionally updated | **Update the pin** to the new digest, with provenance. |
| Wrong object at the URI (pointing at a different dataset) | Fix `DATASET_S3_URI`, or re-upload the right object. |

## Remediation

**If the object is wrong** — restore the known-good bytes:

```bash
aws s3 cp data/raw/data.csv "$(terraform -chdir=terraform output -raw dataset_s3_uri)" \
  --sse aws:kms --sse-kms-key-id "$(terraform -chdir=terraform output -raw dataset_kms_key_arn)"
# Confirm the restored object's digest matches the pin:
aws s3 cp "$(terraform -chdir=terraform output -raw dataset_s3_uri)" - | sha256sum
```

**If the dataset genuinely changed** — update the pin deliberately, recording the new
dataset's provenance ([dataset.md](../dataset.md)):

```bash
# Edit k8s/base/configmap.yaml → DATASET_SHA256: <new digest>, then re-apply.
```

Then re-drive a clean run:

```bash
kubectl -n mlops delete job mlops-pipeline           # ⚠️ discards the failed Job object
kubectl apply -k k8s/overlays/<aws|local>            # on EKS: scripts/render-cloud-manifests.sh --apply
kubectl -n mlops wait --for=condition=complete job/mlops-pipeline --timeout=600s
```

## Recovery verification

```bash
# 1. fetch_dataset now retrieves AND passes the integrity gate.
pod=$(kubectl -n mlops get pods -l job-name=mlops-pipeline \
  --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}')
kubectl -n mlops logs "$pod" -c fetch-dataset        # 'Dataset retrieved …' with NO 'integrity check failed'

# 2. All five stages green, Job Complete exit 0.
curl -s --data-urlencode 'query=mlops_pipeline_stage_success' \
  http://localhost:9090/api/v1/query \
  | jq '.data.result[] | {stage:.metric.stage, success:.value[1]}'   # all == 1
kubectl -n mlops get job mlops-pipeline               # STATUS Complete, COMPLETIONS 1/1

# 3. Alert cleared.
curl -s --data-urlencode 'query=ALERTS{alertname="PipelineJobFailed",alertstate="firing"}' \
  http://localhost:9090/api/v1/query | jq '.data.result | length'    # expect 0
```

## Escalation / known limitations

- **The gate is a correctness control, not a reliability gap** — it is *supposed* to
  stop the run. The only "fix" is correct bytes or a deliberate pin update.
- **Retries are forbidden for this mode** — declined in PR 13
  ([ADR-037 / reliability-hardening evidence](../proof/sprint-08-reliability-hardening-evidence.md)).
- **Design of record:** [ADR-027 (S3 dataset retrieval + integrity pin)](../decisions/ADR-027-s3-dataset-runtime-retrieval.md);
  the additive `Dataset retrieved: …` log that proves retrieve-then-reject ordering was
  added in PR 10 ([`src/fetch_dataset.py`](../../src/fetch_dataset.py)).
