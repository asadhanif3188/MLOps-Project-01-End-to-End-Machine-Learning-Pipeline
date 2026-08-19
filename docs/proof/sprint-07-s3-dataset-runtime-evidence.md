# Sprint 7 — PR 8 evidence: S3-backed runtime dataset retrieval (closes M-04)

This document is the proof for **PR 8**: the pipeline no longer receives its dataset
through a Kubernetes ConfigMap. It is retrieved at runtime from S3 (real Amazon S3 on
EKS via Pod Identity; the S3-compatible MinIO locally) by an init container, verified
against a pinned checksum, and handed to the unchanged DVC pipeline.

- **Design of record:** [ADR-027](../decisions/ADR-027-s3-dataset-runtime-retrieval.md)
- **Dataset identity:** [docs/dataset.md](../dataset.md)
- **Local live run:** validated 2026-08-19 on Docker Desktop Kubernetes (node
  `desktop-control-plane`, k8s v1.34.3) against in-cluster MinIO.
- **AWS/EKS:** design-complete + offline-validated; a live cluster run is
  operator-gated (see the runbook at the end).

## Architecture

```
                 Terraform (datasets.tf)                 operator, out-of-band
   private + SSE-KMS + versioned S3 bucket  <----------  aws s3 cp data/raw/data.csv
                        │                                    s3://…/pima-…/v1/data.csv
      EKS Pod Identity  │  (mlops/mlops-pipeline SA → dataset-reader role, READ-ONLY)
                        ▼
   ┌────────────────────────────────────────── pipeline Pod ──────────────────────────┐
   │  initContainer fetch-dataset (src/fetch_dataset.py, boto3)                         │
   │     └─ download s3://…/v1/data.csv ─► verify SHA-256 ─► write /app/data/raw/data.csv│
   │                                          │  (shared emptyDir)                       │
   │  initContainer wait-for-mlflow           ▼                                          │
   │  container pipeline: dvc repro  ◄── reads /app/data/raw (read-only)                 │
   │     preprocess → split → train → evaluate                                           │
   └───────────────────────────────────────────────────────────────────────────────────┘
```

## Requirement checklist (§12 of the sprint brief)

| # | Requirement | Status |
| --- | --- | --- |
| 1 | Terraform-managed S3 dataset storage | ✅ `terraform/datasets.tf` (bucket + CMK + IAM + Pod Identity) |
| 2 | Dataset not committed to Git | ✅ `data/raw/.gitignore`; object uploaded out-of-band |
| 3 | Dataset not baked into the image | ✅ `.dockerignore` excludes `data/`; retrieved at runtime |
| 4 | No ConfigMap for dataset delivery | ✅ ConfigMap removed; volume is an emptyDir (validator enforces) |
| 5 | Least-privilege S3 **read** IAM policy | ✅ `GetObject`+`ListBucket` on one bucket; **no** Put/Delete/`s3:*` |
| 6 | Access via workload identity | ✅ EKS Pod Identity → `mlops/mlops-pipeline` SA; no static keys |
| 7 | Runtime dataset retrieval implemented | ✅ `src/fetch_dataset.py` in the `fetch-dataset` init container |
| 8 | Deliberate placement decision | ✅ init container (not app code / not entrypoint wrapper) — ADR-027 |
| 9 | DVC semantics preserved | ✅ `dvc.yaml`/`params.yaml` unchanged; all 4 stages ran live |
| 10 | Clear failure when retrieval fails | ✅ typed `DataError` + exit 1 (demonstrated below) |
| 11 | Dataset version/identity/checksum documented | ✅ [docs/dataset.md](../dataset.md) + `DATASET_SHA256` pin |
| 12 | Tests added | ✅ `tests/unit/test_fetch_dataset.py` (18), `terraform/tests/dataset_s3.tftest.hcl` (5), validator checks |
| 13 | Validated on a real cluster | ✅ locally (MinIO, identical code path); ⏳ EKS operator-gated |

## Live local run (Docker Desktop Kubernetes + MinIO)

### 1. Image rebuilt with the new code and imported into the cluster

```
$ docker build -t ml-pipeline:local .          # exit 0 (adds boto3 + src/fetch_dataset.py)
$ docker save ml-pipeline:local | docker exec -i desktop-control-plane \
      ctr -n k8s.io images import -            # import exit 0
```

### 2. Dataset seeded into MinIO (the operator upload step)

```
created bucket datasets
uploaded s3://datasets/pima-indians-diabetes/v1/data.csv  size=23872
objects: ['pima-indians-diabetes/v1/data.csv']
```

### 3. Old ConfigMap mechanism removed to prove independence

```
$ kubectl delete configmap mlops-pipeline-dataset -n mlops
configmap "mlops-pipeline-dataset" deleted from mlops namespace
$ kubectl delete job mlops-pipeline -n mlops        # Job spec is immutable
$ kustomize build k8s/overlays/local | kubectl apply -f -
configmap/mlops-pipeline-config configured
job.batch/mlops-pipeline created
$ kubectl logs job/minio-setup -n mlops | tail -3
Bucket created successfully `local/mlflow`.
Bucket created successfully `local/datasets`.
buckets local/mlflow and local/datasets ready
```

### 4. `fetch-dataset` init container — download + checksum verification

```
$ kubectl logs -n mlops mlops-pipeline-6ctht -c fetch-dataset
2026-08-19 04:56:03 | INFO | fetch_dataset | Fetching dataset s3://datasets/pima-indians-diabetes/v1/data.csv -> data/raw/data.csv via http://minio:9000
2026-08-19 04:56:05 | INFO | fetch_dataset | Dataset checksum verified (sha256=ee5b0c92d5ad461e86151c544b3b76bd6269c6052c5eb628c4b0618a08cffc89)
2026-08-19 04:56:05 | INFO | fetch_dataset | Dataset ready: data/raw/data.csv (23872 bytes)
```

The verified SHA-256 equals the `DATASET_SHA256` pinned in `k8s/base/configmap.yaml`
and recorded in [docs/dataset.md](../dataset.md).

### 5. Structural evidence — init containers, emptyDir (not ConfigMap), no hostPath

```
$ kubectl get pod -n mlops mlops-pipeline-6ctht -o jsonpath='{...initContainerStatuses...}'
fetch-dataset: Completed (exit 0)
wait-for-mlflow: Completed (exit 0)

$ # pod volumes  (name -> configMap.name / emptyDir)
dvc-runtime-config -> mlops-pipeline-dvc-config
dataset -> {}                        # emptyDir, NOT a ConfigMap

$ # pipeline container dataset mount
/app/data/raw ro=true                # read-only; the init container is the sole writer

$ kubectl get configmap mlops-pipeline-dataset -n mlops
Error from server (NotFound): configmaps "mlops-pipeline-dataset" not found
$ kubectl get pod ... -o yaml | grep mlops-pipeline-dataset
(no reference — confirmed)
```

### 6. Pipeline ran to completion over the retrieved data

```
$ kubectl get job mlops-pipeline -n mlops
NAME             STATUS     COMPLETIONS   DURATION   AGE
mlops-pipeline   Complete   1/1           52s        …

$ kubectl logs -n mlops mlops-pipeline-6ctht -c pipeline
Running stage 'preprocess':  … Preprocess stage completed: 768 rows written to data/processed/data.csv
Running stage 'split':       … 614 train rows -> train.csv, 154 held-out rows -> test.csv
Running stage 'train':       … best params: {'min_samples_leaf': 1, 'min_samples_split': 5}
                             … Best model accuracy: 0.7398 ; Model saved to models/model.pkl
Running stage 'evaluate':    … Evaluate stage completed; model accuracy: 0.7078
```

`preprocess` read `data/raw/data.csv` — the file the init container fetched from S3 — so
the whole DAG ran on the S3-delivered dataset, with no ConfigMap involved.

### 7. Clear-failure behaviour (requirement 10)

Pointing the same retrieval code at a non-existent object:

```
$ DATASET_S3_URI=s3://datasets/pima-indians-diabetes/v1/DOES-NOT-EXIST.csv python src/fetch_dataset.py
… | ERROR | fetch_dataset | Dataset retrieval failed: Failed to download the dataset
  s3://datasets/pima-indians-diabetes/v1/DOES-NOT-EXIST.csv: An error occurred (404) when
  calling the HeadObject operation: Not Found. Verify DATASET_S3_URI points at an existing
  object, the workload identity grants read access to it, and the S3 endpoint is reachable.
exit code: 1
```

In-cluster this exit 1 fails the init container, which fails the pod; the Job retries per
`backoffLimit: 2` and then fails cleanly with these logs — the pipeline never starts
against missing/corrupt data.

## Offline / static evidence (CI-equivalent)

```
$ cd terraform && terraform validate          # Success! The configuration is valid.
$ terraform fmt -check -recursive             # exit 0
$ terraform test                              # Success! 41 passed, 0 failed.
   tests/dataset_s3.tftest.hcl:
     dataset_bucket_blocks_all_public_access ............ pass
     dataset_bucket_is_kms_encrypted_and_versioned ...... pass
     dataset_cmk_is_rotated_and_read_only_least_privilege pass
     dataset_access_is_workload_identity ................ pass
     dataset_policy_is_read_only_and_scoped ............. pass

$ python k8s/validate.py k8s/overlays/local   # 88/88 checks passed
$ python k8s/validate.py k8s/overlays/aws     # 84/84 checks passed
$ python -m pytest -q                         # 152 passed, 1 skipped
$ python -m ruff check . && python -m mypy    # clean
```

## AWS / EKS operator runbook (gated)

A live EKS run needs the cluster provisioned (ADR-020; out of scope for the automated
environment). Once EKS + the platform are up:

```bash
cd terraform && terraform apply                         # creates the dataset bucket, CMK,
                                                        # dataset-reader role + Pod Identity assoc.

# Upload the dataset object (out-of-band; never committed):
aws s3 cp data/raw/data.csv \
    "$(terraform output -raw dataset_s3_uri)" \
    --sse aws:kms --sse-kms-key-id "$(terraform output -raw dataset_kms_key_arn)"

# Point the AWS overlay at the real bucket (no file edit needed):
cd k8s/overlays/aws
kustomize edit set image ml-pipeline="$(terraform -chdir=../../../terraform output -raw ecr_repository_url)":<tag>
# set DATASET_S3_URI in job-cloud.yaml to `terraform -chdir=../../../terraform output -raw dataset_s3_uri`

kubectl apply -k k8s/overlays/aws
kubectl logs -n mlops <pod> -c fetch-dataset            # expect the same download + checksum lines
kubectl wait --for=condition=complete job/mlops-pipeline -n mlops --timeout=30m
```

On AWS the init container obtains **short-lived, pod-scoped** credentials from EKS Pod
Identity (the `dataset-reader` role) — there is **no** `AWS_S3_ENDPOINT_URL`, **no**
credential Secret, and **no** static keys anywhere.

## M-04 — closed

The ConfigMap dataset mechanism is removed from both overlays and the base; the dataset
is delivered from a private, encrypted, versioned S3 bucket via a least-privilege
read-only workload identity, retrieved at runtime by an init container, integrity-checked
against a pinned checksum, with clear failure on error — proven end-to-end locally and
design-complete + offline-validated for EKS. **Finding M-04 is closed.**

## Limitations

- The **DVC data remote** is still DagsHub (a separate concern from this runtime dataset
  path); migrating it onto the project's own S3 is already on the roadmap.
- The live EKS exercise is operator-gated (the local MinIO run covers the identical
  retrieval code path).
- `readOnlyRootFilesystem: true` remains deferred (ADR-010), unaffected by this change.
