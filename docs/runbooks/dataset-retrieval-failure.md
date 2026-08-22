# Runbook 3 — Dataset retrieval failure

> **Part of the [Operational Runbooks](README.md).** Alert:
> [`PipelineJobFailed`](../alerting.md#pipelinejobfailed) (critical, 2m) — the
> `fetch_dataset` sub-case. Parent: [Pipeline failure](pipeline-failure.md). Sibling:
> [Dataset checksum / integrity failure](dataset-integrity-failure.md). Evidence:
> [PR 10 Scenario A](../proof/sprint-08-dataset-failure-tests-evidence.md),
> [live-EKS §6](../proof/sprint-08-live-eks-evidence.md#6-pr-10--12--failure-paths--alerts)
> (missing object → **404 HeadObject**, `fetch-dataset` exit 1, pipeline never started).

## Purpose

Diagnose and recover when the `fetch-dataset` **init container** cannot download the
dataset object from S3 — a missing key, denied access, an unreachable endpoint, or no
credentials. This is the "dataset **unavailable**" mode; the "retrieved but corrupt"
mode is the [integrity runbook](dataset-integrity-failure.md).

## Symptoms

- `PipelineJobFailed` firing; the Job Failed with `BackoffLimitExceeded`.
- The `fetch-dataset` init container is `Terminated` **non-zero (exit 1)**; the
  `pipeline` container is `Waiting`/`PodInitializing` and **never Running**.
- Pod log (`-c fetch-dataset`): `Failed to download s3://…` — on EKS specifically a
  `404 HeadObject Not Found` (missing key) or a credentials/timeout error.
- `mlops_pipeline_stage_success{stage="fetch_dataset"}` is `0`; every later stage absent.

## Detection

```bash
# The failing stage is fetch_dataset and nothing after it ran:
curl -s --data-urlencode 'query=mlops_pipeline_stage_success{stage="fetch_dataset"}' \
  http://localhost:9090/api/v1/query | jq '.data.result[].value[1]'      # expect 0
```

The `fetch_dataset` stage is the discriminator that separates this from every other
pipeline failure.

## Initial checks

```bash
pod=$(kubectl -n mlops get pods -l job-name=mlops-pipeline \
  --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}')

# 1. Init-container status: fetch-dataset Terminated non-zero, pipeline never started.
kubectl -n mlops describe pod "$pod" | sed -n '/Init Containers/,/^Containers:/p'

# 2. The root-cause log line (distinguishes THIS mode from the integrity mode).
kubectl -n mlops logs "$pod" -c fetch-dataset
```

## Diagnosis

The `fetch-dataset` log is unambiguous — this runbook applies when it reads:

```
Failed to download s3://<bucket>/<key> …        # object UNAVAILABLE  → this runbook
```

(If instead it reads `Dataset retrieved: … (N bytes); verifying integrity` **then**
`integrity check failed: expected …, got …`, the object *was* retrieved — go to the
[integrity runbook](dataset-integrity-failure.md).)

Then narrow *why* the download failed:

```bash
# What object is the Job pointing at? (URI is in the workload config)
kubectl -n mlops get cm mlops-pipeline-config -o jsonpath='{.data.DATASET_S3_URI}{"\n"}'

# On EKS — does the object actually exist, and does the role have access?
# (Reads use the operator's own credentials; the pod uses EKS Pod Identity.)
aws s3 ls "$(terraform -chdir=terraform output -raw dataset_s3_uri)"
```

## Likely causes

| Cause | How to tell | Note |
|---|---|---|
| **Object missing** (wrong key, not uploaded) | `404 HeadObject Not Found`; `aws s3 ls` shows nothing | The Scenario A live case — intended fail-fast on a missing input. |
| **Access denied** | `403`/`AccessDenied` in the log | Pod Identity role lacks read on the bucket/key ([ADR-027](../decisions/ADR-027-s3-dataset-runtime-retrieval.md)). |
| **No credentials / endpoint blocked** | `CredentialRetrievalError: Connect timeout … 169.254.170.23` | The live **Finding 1** — an enforced NetworkPolicy blocked the Pod Identity agent; fixed by `allow-pod-identity-egress` ([live-EKS Finding 1](../proof/sprint-08-live-eks-evidence.md#3-findings--4-real-defects-the-live-run-surfaced-all-fixed)). |
| **Endpoint unreachable** (local MinIO not up) | connection refused to the S3 endpoint | Local overlay: the `minio` Service / `datasets` bucket not ready. |
| **Transient S3 fault** | intermittent; a later attempt succeeds | Already bounded by `botocore` retries + `backoffLimit`; usually self-heals. |

## Remediation

Address the specific cause, then re-drive a clean run:

- **Object missing** — upload it out-of-band (never bake it into the image):
  ```bash
  # EKS (SSE-KMS):
  aws s3 cp data/raw/data.csv "$(terraform -chdir=terraform output -raw dataset_s3_uri)" \
    --sse aws:kms --sse-kms-key-id "$(terraform -chdir=terraform output -raw dataset_kms_key_arn)"
  # Local (MinIO): see kubernetes-operations.md § Deploy a run.
  ```
- **Access denied** — confirm the `dataset-reader` Pod Identity association and its IAM
  read policy ([ADR-027](../decisions/ADR-027-s3-dataset-runtime-retrieval.md)).
- **Credentials blocked by NetworkPolicy** — ensure `allow-pod-identity-egress` is
  applied (`kubectl -n mlops get networkpolicy allow-pod-identity-egress`); this is the
  committed fix from the live campaign.
- **Wrong URI** — correct `DATASET_S3_URI` in `k8s/base/configmap.yaml`, re-apply.

Then:

```bash
kubectl -n mlops delete job mlops-pipeline           # ⚠️ discards the failed Job object
kubectl apply -k k8s/overlays/<aws|local>            # on EKS: scripts/render-cloud-manifests.sh --apply
kubectl -n mlops wait --for=condition=complete job/mlops-pipeline --timeout=600s
```

> **Regression harness (safe).** To re-prove the failure path without touching the real
> Job, `k8s/tests/dataset-failure/run.sh` renders throwaway Jobs with a narrow env
> override (bad key / bad digest) and cleans them up by deletion — expect `RESULT: PASS`.

## Recovery verification

Prove the dataset boundary is healthy *and* the run completed:

```bash
# 1. fetch_dataset now succeeds and the whole run reached green.
curl -s --data-urlencode 'query=mlops_pipeline_stage_success' \
  http://localhost:9090/api/v1/query \
  | jq '.data.result[] | {stage:.metric.stage, success:.value[1]}'   # fetch_dataset=1 AND all 5 stages=1

# 2. Job Complete, exit 0.
kubectl -n mlops get job mlops-pipeline               # STATUS Complete, COMPLETIONS 1/1

# 3. The init container succeeded this time.
pod=$(kubectl -n mlops get pods -l job-name=mlops-pipeline \
  --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}')
kubectl -n mlops logs "$pod" -c fetch-dataset         # 'Dataset retrieved: … ; verifying integrity' → passes

# 4. Alert cleared.
curl -s --data-urlencode 'query=ALERTS{alertname="PipelineJobFailed",alertstate="firing"}' \
  http://localhost:9090/api/v1/query | jq '.data.result | length'    # expect 0
```

Visually: the **MLOps Pipeline Operations** dashboard **Dataset retrieval** panel green.

## Escalation / known limitations

- **A missing dataset is intended fail-fast**, not a bug — the pipeline must never start
  training on absent data ([ADR-027](../decisions/ADR-027-s3-dataset-runtime-retrieval.md)).
- **S3 egress precision is delegated to IAM** (+ a recommended VPC S3 endpoint), not the
  NetworkPolicy, which only bounds egress to internet `:443` — a documented limitation
  ([ADR-034](../decisions/ADR-034-network-policies.md)).
- **Design of record:** [ADR-027 (S3 dataset retrieval)](../decisions/ADR-027-s3-dataset-runtime-retrieval.md);
  see also the [Dataset doc](../dataset.md).
