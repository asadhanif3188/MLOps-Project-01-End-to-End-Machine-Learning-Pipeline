# Dataset identity, version & integrity

This project trains on the **Pima Indians Diabetes** dataset. This page is the
single source of truth for **which** dataset the pipeline expects, its **version**,
and its **checksums** — the identity that the runtime retrieval verifies and that
makes a run reproducible.

## Identity

| Property | Value |
| --- | --- |
| Name | Pima Indians Diabetes |
| Target column | `Outcome` (binary) |
| Rows × columns | 768 × 9 (8 features + target) |
| File | `data/raw/data.csv` (CSV, header row) |
| Size | **23,872 bytes** |
| SHA-256 | `ee5b0c92d5ad461e86151c544b3b76bd6269c6052c5eb628c4b0618a08cffc89` |
| DVC md5 | `739f3c9177b13d1c15aa060046cfb023` (see [`data/raw/data.csv.dvc`](../data/raw/data.csv.dvc)) |
| Version | `v1` (encoded in the S3 object key path, below) |

The dataset is **not committed to Git** (`data/raw/.gitignore`) and **not baked into
the image** (`.dockerignore`). It is delivered at runtime from object storage.

## Where it lives

| Environment | Location |
| --- | --- |
| AWS (EKS) | `s3://<name_prefix>-datasets-<account-id>/pima-indians-diabetes/v1/data.csv` — a private, SSE-KMS-encrypted, versioned bucket ([`terraform/datasets.tf`](../terraform/datasets.tf)) |
| Local (kind / Docker Desktop) | `s3://datasets/pima-indians-diabetes/v1/data.csv` in in-cluster MinIO |
| Developer workstation | `data/raw/data.csv`, fetched with `dvc pull` (DVC data remote) |

## How the runtime verifies it

The `fetch-dataset` init container ([`src/fetch_dataset.py`](../src/fetch_dataset.py))
downloads the object into `/app/data/raw/data.csv` and compares its SHA-256 against
`DATASET_SHA256`, pinned in the base ConfigMap ([`k8s/base/configmap.yaml`](../k8s/base/configmap.yaml)).
A mismatch fails the Job before any training runs, so a swapped or corrupted object can
never silently enter a run. See [ADR-027](decisions/ADR-027-s3-dataset-runtime-retrieval.md).

## Bumping the dataset

Treat the dataset as immutable-per-version:

1. Produce the new file and record its size + SHA-256 (`sha256sum data/raw/data.csv`).
2. Upload it under a **new** version-pathed key (`.../v2/data.csv`) — never overwrite `v1`.
3. Update `DATASET_SHA256` in `k8s/base/configmap.yaml` and `DATASET_S3_URI` in the
   overlays (and this page + `data/raw/data.csv.dvc`).

Because the key path carries the version and the bucket is versioned, past runs remain
reproducible against the exact bytes they consumed.
