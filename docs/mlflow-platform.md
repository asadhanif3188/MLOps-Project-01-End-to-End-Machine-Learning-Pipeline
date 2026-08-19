# MLflow Tracking Platform — Operations Guide

The project's experiment tracking runs on a **persistent, self-hosted MLflow
platform inside Kubernetes** (Sprint 7, PR 6). This guide covers deploying it,
operating it, and the persistence test that proves it. Design rationale is in
[ADR-026](decisions/ADR-026-in-cluster-mlflow-platform.md); this document is the
runbook.

## Architecture at a glance

```
Pipeline Job ──HTTP──▶ Service "mlflow" (ClusterIP :5000)
                          │
                    MLflow Tracking Server  (Deployment, stateless, --serve-artifacts)
                          ├── PostgreSQL  (StatefulSet + PVC)      ← experiment/run METADATA
                          └── S3 artifact store                    ← models, plots, reports
                                • local: MinIO (StatefulSet + PVC),  creds via Secret
                                • aws:   Amazon S3 (Terraform),       creds via EKS Pod Identity
```

- The tracking server is **stateless** — all durable state is in PostgreSQL and
  S3, so it survives pod recreation.
- The server **proxies artifacts** (`--serve-artifacts`), so the pipeline logs over
  HTTP only and needs **no S3 credentials**.
- Every Service is **ClusterIP** (internal); the UI is reached via
  `kubectl port-forward`, never a public endpoint.

## Components

| Object | Kind | Purpose |
|---|---|---|
| `mlflow` | Deployment + Service | The tracking server; internal front door on `:5000` |
| `mlflow-postgres` | StatefulSet + headless Service | Metadata backend; PVC-backed persistence |
| `minio` (local only) | StatefulSet + Service | S3-compatible artifact store; PVC-backed |
| `minio-setup` (local only) | Job | Creates the `mlflow` bucket (idempotent) |
| `mlflow-server` | ServiceAccount | Server identity; AWS Pod Identity subject on EKS |
| `mlflow-config`, `mlflow-boto-config` | ConfigMap | Non-secret config (DB host/name, artifact dest, S3 addressing) |
| `mlflow-db-credentials` | Secret (out-of-band) | Postgres user/password |
| `mlflow-s3-credentials` | Secret (out-of-band, local) | MinIO access keys |

## Prerequisites (local)

1. A local Kubernetes cluster (Docker Desktop / kind / minikube) with a **default
   StorageClass** (Docker Desktop ships `standard`/local-path).
2. The two images built and available to the cluster's container runtime:
   ```bash
   # Pipeline image (unchanged)
   docker build -t ml-pipeline:local .
   # MLflow server image (new)
   docker build -f docker/mlflow/Dockerfile -t mlflow-server:local .
   ```
   On **Docker Desktop Kubernetes** (containerd), import both into the node's
   `k8s.io` namespace so pods see them (its image store is separate from
   `docker build` — see [Kubernetes Operations](kubernetes-operations.md)):
   ```bash
   docker save ml-pipeline:local   | nerdctl --namespace k8s.io load   # or ctr -n k8s.io images import
   docker save mlflow-server:local | nerdctl --namespace k8s.io load
   ```

## Deploy (local)

```bash
# 1) Namespace (created by the overlay, but needed first for out-of-band Secrets)
kubectl apply -f k8s/base/namespace.yaml

# 2) Create the out-of-band Secrets (NEVER committed — templates in the repo):
kubectl create secret generic mlflow-db-credentials -n mlops \
  --from-literal=POSTGRES_USER=mlflow \
  --from-literal=POSTGRES_PASSWORD="$(openssl rand -base64 24 | tr -d '/+=')"

kubectl create secret generic mlflow-s3-credentials -n mlops \
  --from-literal=AWS_ACCESS_KEY_ID="$(openssl rand -hex 12)" \
  --from-literal=AWS_SECRET_ACCESS_KEY="$(openssl rand -base64 24 | tr -d '/+=')"

# 3) Deploy the platform + pipeline workload
kubectl apply -k k8s/overlays/local

# 4) Wait for the platform to be ready
kubectl -n mlops rollout status statefulset/mlflow-postgres
kubectl -n mlops rollout status statefulset/minio
kubectl -n mlops wait --for=condition=complete job/minio-setup --timeout=180s
kubectl -n mlops rollout status deployment/mlflow
```

Verify health:

```bash
kubectl -n mlops get pods
kubectl -n mlops port-forward svc/mlflow 5000:5000 &
curl -fsS http://127.0.0.1:5000/health && echo OK
```

## Run the pipeline against the platform

Provide the dataset out-of-band by uploading it into MinIO (Sprint 7 PR 8; the
`fetch-dataset` init container retrieves it at runtime — [ADR-027](decisions/ADR-027-s3-dataset-runtime-retrieval.md)),
then run the Job:

```bash
kubectl apply -k k8s/overlays/local          # brings up MinIO + the Job
kubectl -n mlops port-forward svc/minio 9000:9000 &
export AWS_ACCESS_KEY_ID=$(kubectl -n mlops get secret mlflow-s3-credentials -o jsonpath='{.data.AWS_ACCESS_KEY_ID}' | base64 -d)
export AWS_SECRET_ACCESS_KEY=$(kubectl -n mlops get secret mlflow-s3-credentials -o jsonpath='{.data.AWS_SECRET_ACCESS_KEY}' | base64 -d)
aws --endpoint-url http://localhost:9000 s3 cp data/raw/data.csv s3://datasets/pima-indians-diabetes/v1/data.csv
kubectl -n mlops wait --for=condition=complete job/mlops-pipeline --timeout=300s
kubectl -n mlops logs job/mlops-pipeline | tail
```

The run's metrics/params land in PostgreSQL and its model/artifacts in S3 (MinIO).

The pipeline Job carries a `wait-for-mlflow` **init container** that polls the
server's `/health` and blocks the pipeline until it returns 200 — so the Job never
races a not-yet-Ready server (it waits deterministically instead of burning a
retry). If the server never comes up, the init container fails after ~5 min and the
Job fails cleanly (then retries per `backoffLimit`).

## Pipeline tracking configuration

The pipeline's tracking configuration is resolved by
[`src/mlflow_config.py`](../src/mlflow_config.py) **before** the (lazy) MLflow
import, so a misconfiguration fails fast as a typed `ConfigError` with no MLflow,
network, or credentials involved. In-cluster these come from the base
`ConfigMap` ([`k8s/base/configmap.yaml`](../k8s/base/configmap.yaml)); locally
from `.env` (see [`.env.example`](../.env.example)).

| Env var | Required | Default | Purpose |
|---|---|---|---|
| `MLFLOW_TRACKING_URI` | Yes | — | The tracking server. In-cluster it is the `mlflow` Service DNS name; the value lives in config, never hardcoded in Python. |
| `MLFLOW_EXPERIMENT_NAME` | No | `mlops-pipeline` | Experiment the train/evaluate runs are grouped under (instead of MLflow's catch-all `Default`). |
| `MLFLOW_ALLOW_FILE_STORE` | No | unset (off) | Opt-in escape hatch for a local `file:` store — see the guard below. |

**File-store guard.** A `file:` (or scheme-less) tracking URI records runs to the
local filesystem. In a cluster that filesystem is the pod's *ephemeral* storage,
so every run, metric, and artifact vanishes when the pod exits — the "transient
offline file store" failure mode. `resolve_tracking_uri()` therefore **rejects** a
file-store URI unless `MLFLOW_ALLOW_FILE_STORE` is truthy (`1`/`true`/`yes`/`on`),
turning a silent data-loss footgun into an explicit, offline-only choice. Normal
runs against this platform need a server URI and never set the flag.

**No credentials.** The server is internal-only and unauthenticated, so the
pipeline carries no MLflow username/password/token and no S3 access — artifacts are
proxied through the server (`mlflow-artifacts:`). The platform's *own* Secrets
(`mlflow-db-credentials`, `mlflow-s3-credentials`) are a separate concern, created
out-of-band per [Deploy (local)](#deploy-local).

## Persistence test — the real proof

A Deployment becoming Ready proves nothing about persistence. The real test:
**log a run, destroy the stateful pods, and confirm the run is still there.**

```bash
# Count runs BEFORE (via the tracking API, through the port-forward)
curl -s -X POST http://127.0.0.1:5000/api/2.0/mlflow/runs/search \
  -H 'Content-Type: application/json' \
  -d '{"experiment_ids":["0"],"max_results":1000}' | grep -o '"run_id"' | wc -l

# Recreate BOTH stateful pods (PVCs re-bind) and the stateless server
kubectl -n mlops delete pod mlflow-postgres-0 minio-0
kubectl -n mlops delete pod -l app.kubernetes.io/name=mlflow-server
kubectl -n mlops rollout status statefulset/mlflow-postgres
kubectl -n mlops rollout status statefulset/minio
kubectl -n mlops rollout status deployment/mlflow

# Count runs AFTER — must equal BEFORE (metadata survived), and an artifact must
# still be downloadable (artifact bytes survived in S3/MinIO).
curl -s -X POST http://127.0.0.1:5000/api/2.0/mlflow/runs/search \
  -H 'Content-Type: application/json' \
  -d '{"experiment_ids":["0"],"max_results":1000}' | grep -o '"run_id"' | wc -l
```

Equal counts across a full stateful-pod recreation is the persistence guarantee:
the data lived in the PVCs (Postgres) and the object store (MinIO/S3), not in any
pod. The executed evidence is recorded in
[ADR-026 § Runtime evidence](decisions/ADR-026-in-cluster-mlflow-platform.md#runtime-evidence).

## AWS (EKS) notes

The AWS path is authored in `k8s/overlays/aws` + `terraform/` and applied by the
operator against their own account (operator-driven apply, as elsewhere):

- **Images → ECR.** Terraform provisions TWO repositories: `mlops-pipeline` (the
  pipeline image) and `mlflow-server` (the tracking server image). Build, tag, and
  push both, then pin them on the overlay:
  ```bash
  cd k8s/overlays/aws
  kustomize edit set image \
    ml-pipeline="$(terraform -chdir=../../../terraform output -raw ecr_repository_url)":<ver> \
    mlflow-server="$(terraform -chdir=../../../terraform output -raw mlflow_server_ecr_repository_url)":<ver>
  ```
- **Artifacts → real S3.** `terraform/s3.tf` provisions a private, encrypted,
  versioned bucket. Set it on the overlay:
  ```bash
  cd k8s/overlays/aws
  # bucket: terraform -chdir=../../../terraform output -raw mlflow_artifact_bucket_name
  # -> edit MLFLOW_ARTIFACTS_DESTINATION in mlflow-cloud.yaml to s3://<bucket>/artifacts
  ```
- **No static AWS keys.** The `mlflow-server` ServiceAccount is bound to a scoped
  IAM role via **EKS Pod Identity** (`terraform/s3.tf`); boto3 resolves pod-scoped
  credentials automatically. Do **not** create `mlflow-s3-credentials` on AWS.
- **Postgres persistence → EBS.** `terraform/ebs-csi.tf` installs the EBS CSI driver
  (Pod Identity) so the Postgres PVC provisions an EBS volume from the default
  StorageClass.

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `mlflow-postgres-0` `Pending` | No default StorageClass / (EKS) EBS CSI not ready | `kubectl get sc`; ensure the EBS CSI addon is Ready on EKS |
| MLflow pod restarts, `Exit Code 137` (worker killed), `dmesg` shows a cgroup OOM | MLflow 3.x server needs ~1.7 GiB; a limit below ~2 GiB OOM-kills the worker on startup | The base sets `limits.memory: 2Gi` for this reason — do not lower it; scale it (and `--workers`) up together if you raise concurrency |
| MLflow pod `CrashLoopBackOff` on first boot | DB not ready yet / wrong `mlflow-db-credentials` | Check `kubectl -n mlops logs deploy/mlflow`; confirm the Secret exists and matches the Postgres role |
| Pipeline gets `403 "Invalid Host header - possible DNS rebinding attack detected"` | MLflow 3.x host-validation middleware rejects the Service DNS name | The base sets `MLFLOW_SERVER_ALLOWED_HOSTS` to the in-cluster names + localhost; add any extra client host there (`/health` and `/version` are exempt, so probes are unaffected) |
| Artifact upload fails against MinIO | Path-style addressing / bucket missing | Confirm `minio-setup` Job completed; the `mlflow-boto-config` ConfigMap forces path-style |
| Pipeline `TrackingError` (connection refused) | Server not reachable/ready (e.g. mid-rollout) | `curl .../health`; wait for the `mlflow` Deployment to be Ready before running the Job (the Job retries per `backoffLimit`) |
| Runs disappear after pod delete | (should not happen) writing to the wrong store | Confirm Postgres PVC is `Bound` and the server uses `postgresql+psycopg2`, not a file store |

## Related documentation

- [ADR-026 — In-cluster MLflow platform](decisions/ADR-026-in-cluster-mlflow-platform.md)
- [ADR-013 — Kubernetes runtime execution](decisions/ADR-013-kubernetes-runtime-execution.md)
- [ADR-024 — VPC CNI Pod Identity](decisions/ADR-024-vpc-cni-pod-identity.md)
- [Kubernetes Operations runbook](kubernetes-operations.md)
- [k8s/README](../k8s/README.md)
