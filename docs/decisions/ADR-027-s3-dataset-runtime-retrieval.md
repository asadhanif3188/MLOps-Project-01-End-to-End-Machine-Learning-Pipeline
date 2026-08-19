# ADR-027: S3-backed runtime dataset retrieval (init container + workload identity)

- **Status:** Accepted (validated locally against MinIO; EKS path is design + operator-gated)
- **Date:** 2026-08-19
- **Deciders:** Asad Hanif
- **Related:** [`terraform/datasets.tf`](../../terraform/datasets.tf),
  [`src/fetch_dataset.py`](../../src/fetch_dataset.py),
  [`k8s/base/job.yaml`](../../k8s/base/job.yaml),
  [`k8s/overlays/local/job-runtime.yaml`](../../k8s/overlays/local/job-runtime.yaml),
  [`k8s/overlays/aws/job-cloud.yaml`](../../k8s/overlays/aws/job-cloud.yaml),
  [`docs/dataset.md`](../dataset.md),
  [ADR-013 (Kubernetes Runtime Execution)](ADR-013-kubernetes-runtime-execution.md),
  [ADR-024 (VPC CNI Pod Identity)](ADR-024-vpc-cni-pod-identity.md),
  [ADR-026 (In-cluster MLflow platform)](ADR-026-in-cluster-mlflow-platform.md)
- **Supersedes:** the ConfigMap dataset-delivery mechanism documented in
  [ADR-013](ADR-013-kubernetes-runtime-execution.md) and ADR-018 (the dataset half only).
- **Closes:** finding **M-04** (Dataset delivered through ConfigMap).

> **Scope.** Through Sprint 6 the runtime dataset reached the pipeline pod as a
> Kubernetes **ConfigMap** (`mlops-pipeline-dataset`), created out-of-band from the
> git-ignored CSV and mounted read-only at `/app/data/raw`. It was always labelled a
> *local-validation* mechanism, never production storage. This ADR replaces it with a
> professional cloud data path: the dataset lives in a private, encrypted, versioned
> **S3 bucket**; an **init container** retrieves it at runtime into a shared volume
> using **EKS Pod Identity** (no static keys); the DVC pipeline then runs unchanged.
> This is the design of record for dataset storage, the retrieval mechanism, the
> IAM/identity model, and the removal of the ConfigMap dataset.

## Context

The ConfigMap mechanism had structural problems that M-04 flagged:

1. **Wrong carrier.** A ConfigMap is etcd-backed and caps at ~1 MiB. It is meant for
   configuration, not data; a real dataset does not belong in the cluster's control-plane
   store. The bundled Pima dataset (~23 KiB) fit only because it is tiny.
2. **Not a cloud data path.** Production ML data lives in object storage. Delivering it
   through a ConfigMap modelled nothing about how a real dataset is versioned, secured,
   or retrieved, so the "cloud" story was incomplete.
3. **Operational awkwardness.** The object had to be hand-materialised into a ConfigMap
   (`kubectl create configmap --from-file`) on every environment, an out-of-band step
   with none of S3's versioning, encryption, or access control.

The constraints were firm: the dataset must **not** be committed to Git, **not** be
baked into the image, **not** travel in a ConfigMap, and access must use **workload
identity** with **no static AWS keys** and **no hostPath**. The platform already had
every building block: a private/encrypted/versioned S3 pattern (ADR-026), EKS Pod
Identity wiring (ADR-024/026), and an init-container precedent (`wait-for-mlflow`).

## Decision

### 1. Storage — a dedicated, secured S3 bucket ([`terraform/datasets.tf`](../../terraform/datasets.tf))

A Terraform-managed `${name_prefix}-datasets-${account_id}` bucket, mirroring the MLflow
artifact bucket's posture: **all public access blocked**, **BucketOwnerEnforced** (ACLs
off), **versioning enabled**, and **SSE-KMS** with a **dedicated customer-managed key**
(rotation on, CloudTrail-auditable). Terraform owns the *empty, secured bucket*; the
dataset **object** is uploaded out-of-band by the operator (like a credential), so the
data is never in Git and never in the image. The canonical key is **version-pathed**
(`pima-indians-diabetes/v1/data.csv`) so a dataset change is a new, reviewable key.

### 2. Identity — a least-privilege, READ-ONLY role via EKS Pod Identity

A dedicated `${name_prefix}-dataset-reader-role` trusted only by `pods.eks.amazonaws.com`
(`sts:AssumeRole` + `sts:TagSession`), bound by an `aws_eks_pod_identity_association` to
the pipeline's **existing** service account `mlops/mlops-pipeline`. Its inline policy
grants exactly `s3:GetObject` + `s3:ListBucket`/`GetBucketLocation` on **this one
bucket** — and **no `PutObject`/`DeleteObject`, no `s3:*`, no wildcard bucket**. The
dataset CMK grants the role only `kms:Decrypt`/`DescribeKey` (the SSE-KMS read path) —
tighter than the MLflow server, which also writes. There are **no static AWS keys** on
the cluster; boto3 resolves short-lived, pod-scoped credentials automatically.

### 3. Retrieval — an init container running first-party Python

The base Job gains a **`fetch-dataset` init container** (same pipeline image) that runs
[`src/fetch_dataset.py`](../../src/fetch_dataset.py). It downloads the object into a
shared **`emptyDir`** mounted at `/app/data/raw` (writable in the init container,
**read-only** in the pipeline container), verifies the bytes against a pinned
`DATASET_SHA256`, and fails the Job with a clear typed error on any missing object,
denied access, unreachable endpoint, or checksum mismatch. Pod-level `fsGroup: 10001`
makes the emptyDir writable by the non-root user. The retrieval **mechanism** is
environment-independent and lives in the **base**; only the **source** (bucket URI,
endpoint, credentials) is layered by the overlays — real Amazon S3 + Pod Identity on
AWS, in-cluster **MinIO** + an out-of-band Secret locally (so the identical code path is
testable without AWS).

**Why an init container (not application code, not an entrypoint wrapper):**

- **Not a DVC stage / application code.** The stages must stay pure computation over
  `data/raw` so `dvc repro` is reproducible and the DVC DAG represents the
  *computation graph*, not data acquisition. Embedding S3 in `preprocess` would also
  force the main container to hold S3 identity. Acquiring the input is a *pre-run*
  concern — exactly what an init container models.
- **Not an entrypoint wrapper.** The image `CMD` stays exactly `dvc repro`; wrapping it
  would blur that contract and muddy failure isolation. The pod already uses an init
  container (`wait-for-mlflow`) as its pre-run gate, so this is the established pattern.
- **First-party Python, not shell `aws s3 cp`.** Keeping the logic in `fetch_dataset.py`
  makes it unit-testable with an injected client, gives it the project's typed
  exceptions + structured logging, and lets it enforce the checksum.

### 4. DVC semantics preserved

`dvc.yaml`/`params.yaml` are unchanged: `data/raw/data.csv` is still the `preprocess`
stage input and DVC checksums it as a dependency. The init container simply delivers
that tracked file (an alternative to `dvc pull`) before the DAG runs. The live run below
executed all four stages in order over the retrieved file.

## Alternatives Considered

- **Keep the ConfigMap.** Rejected — it is the finding (M-04); wrong carrier, no cloud
  data model, 1 MiB cap.
- **`dvc pull` from a DVC remote in the container.** Deferred — the DVC *data* remote is
  still on DagsHub (a separate migration, already backlogged in the roadmap). It would
  also require SCM/remote credentials in the runtime pod. Reading the object directly
  from the project's own S3 with Pod Identity is simpler and credential-free now; moving
  the DVC remote onto this bucket is a clean follow-up.
- **PVC / CSI volume seeded out-of-band.** Heavier (a PV lifecycle) for a
  read-once batch input, and still needs a seeding path. An init-container pull into an
  emptyDir is the lightest correct option for a run-to-completion Job.
- **`aws s3 cp` in a shell init container.** Rejected — not unit-testable, no typed
  errors, no checksum enforcement, and it would add an external image or the AWS CLI.
- **Baking the dataset into the image.** Rejected outright by the constraints (and by
  ADR-005's `.dockerignore` policy): couples data to code, bloats the image, and leaks
  data into every registry copy.

## Consequences

**Positive**

- M-04 closed: the AWS runtime dataset path is professional object storage with
  versioning, encryption, and least-privilege read via workload identity.
- No static AWS credentials anywhere; the main container never needs S3 (the init
  container is the only S3 consumer, and even it holds only read-only, pod-scoped creds).
- Dataset **integrity + identity** are enforced at runtime (`DATASET_SHA256`) and
  documented ([`docs/dataset.md`](../dataset.md)); a swapped/corrupt object fails fast.
- One retrieval mechanism for both environments (S3 API), so local runs against MinIO
  exercise the exact production code path — the retrieval is genuinely tested, not stubbed.
- No ConfigMap dataset, no hostPath, all Sprint 5 security controls preserved
  (non-root, uid/gid 10001, seccomp, drop ALL, no-automount), plus `fsGroup`.

**Negative / trade-offs**

- The dataset object is still uploaded out-of-band (by design — it is not in Git). The
  operator runbook documents the one `aws s3 cp` step (and its MinIO equivalent).
- Pod Identity credentials are pod-scoped, so the (unused) S3 read capability is present
  in the main container too. The role grants only dataset read, so the blast radius is
  minimal; container-scoped identity is not something Pod Identity offers.
- `readOnlyRootFilesystem: true` remains deferred (ADR-010) — unrelated to this change
  (DVC still writes its cache/lock at the repo root); the dataset now arrives on a
  dedicated volume, which is a step toward it.

## Validation

- **Local, live (validated 2026-08-19)** on Docker Desktop Kubernetes against in-cluster
  MinIO: the `fetch-dataset` init container downloaded `s3://datasets/…/v1/data.csv`,
  verified `sha256=ee5b0c92…`, and the Job completed with all four DVC stages running
  over the retrieved file; the old `mlops-pipeline-dataset` ConfigMap was deleted first
  to prove independence; the missing-object failure path was exercised (clear error,
  exit 1). Full transcript: [proof](../proof/sprint-07-s3-dataset-runtime-evidence.md).
- **AWS/EKS** is design-complete (Terraform + AWS overlay) and validated offline by
  `terraform test` (mock provider) and the k8s static validator, but a live EKS run is
  **operator-gated** (provisioning EKS is out of this environment's scope; ADR-020).
  The proof doc gives the exact operator runbook.
