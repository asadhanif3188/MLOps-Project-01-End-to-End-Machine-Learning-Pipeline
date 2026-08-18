# Sprint 7 · PR 7 — In-cluster MLflow integration: runtime evidence

**PR:** `feat(mlops): integrate pipeline with in-cluster MLflow`
**Branch:** `feature/sprint-07-mlflow-pipeline-integration`
**Date:** 2026-08-18
**Cluster:** Docker Desktop Kubernetes (kind-based, node `desktop-control-plane`, k8s v1.34)
**Overlay:** `k8s/overlays/local` (MLflow server + PostgreSQL + MinIO S3, per ADR-026)

This is the piece the Sprint 6 proof explicitly deferred. The earlier record
([sprint-06-runtime-evidence.md](sprint-06-runtime-evidence.md#limitations)) noted
that connectivity to a real MLflow backend was only *config-validated* — the Job
had logged against a transient offline file store, never a live server. This PR
closes that gap: a real pipeline run logging over HTTP to the **in-cluster MLflow
Tracking Server**, with the tracking metadata in **PostgreSQL** and the model +
artifacts in the **S3 (MinIO)** store — verified end to end below.

The image under test was rebuilt from this branch (so `src/mlflow_config.py` and
the experiment-naming change are exercised) and side-loaded into the node's
containerd; the Job then ran the image's default `dvc repro` (preprocess → split →
train → evaluate).

---

## 1. Configuration model (what the pipeline reads)

The pipeline's only tracking configuration is injected from the Kustomize
`ConfigMap` — no Secret, no credentials, no DagsHub. The service DNS name is
**config, not code** (`src/mlflow_config.py` reads the env var):

```
$ kubectl get configmap mlops-pipeline-config -n mlops -o jsonpath='{.data}'
{"LOG_LEVEL":"INFO",
 "MLFLOW_EXPERIMENT_NAME":"mlops-pipeline",
 "MLFLOW_TRACKING_URI":"http://mlflow.mlops.svc.cluster.local:5000"}
```

The pipeline pod mounts **only** that ConfigMap (`envFrom.configMapRef=mlops-pipeline-config`)
— there is no credential Secret in the runtime path, and neither the pipeline nor
the MLflow server carries any `dagshub` / `MLFLOW_TRACKING_USERNAME` /
`MLFLOW_TRACKING_PASSWORD` value.

## 2. Readiness gate — the init container reached the real Service

```
$ kubectl logs mlops-pipeline-9mq59 -n mlops -c wait-for-mlflow
MLflow ready at http://mlflow.mlops.svc.cluster.local:5000/health
```

The `wait-for-mlflow` init container polled the in-cluster Service's `/health` and
exited 0, so the pipeline started only once the server was actually serving.

## 3. Job status — green, exit 0

```
$ kubectl get job mlops-pipeline -n mlops -o jsonpath='{.status}'
{ "succeeded": 1,
  "startTime": "2026-08-18T13:57:19Z",
  "completionTime": "2026-08-18T13:57:57Z",
  "conditions": [ { "type": "SuccessCriteriaMet", "status": "True", ... },
                  { "type": "Complete",           "status": "True", ... } ] }

pod mlops-pipeline-9mq59:  image=ml-pipeline:local  initExit=0  mainExit=0
```

Whole pipeline (four stages) completed in **~38 s**; init container and pipeline
container both exited 0.

## 4. Pipeline log — stages logged to the in-cluster server

```
Running stage 'train':
train | Best model accuracy: 0.7398
train | Model saved to models/model.pkl
Registered model 'Best Random Forest Classifier' ... Created version '3'.
🏃 View run fun-eel-627 at: http://mlflow.mlops.svc.cluster.local:5000/#/experiments/1/runs/27723f52...
🧪 View experiment at:      http://mlflow.mlops.svc.cluster.local:5000/#/experiments/1
train | Train stage completed

Running stage 'evaluate':
🏃 View run resilient-skink-709 at: http://mlflow.mlops.svc.cluster.local:5000/#/experiments/1/runs/8e22039e...
evaluate | Evaluate stage completed; model accuracy: 0.7078
```

Both stages logged to **experiment `1`** — the named `mlops-pipeline` experiment,
not MLflow's catch-all `Default`. That is the proof the new
`resolve_experiment_name()` → `mlflow.set_experiment()` path executed in-cluster.

## 5. Tracking server (REST API) — experiment, runs, metrics, params, status

**Experiment** (`GET /api/2.0/mlflow/experiments/get?experiment_id=1`):

```json
{ "experiment_id": "1", "name": "mlops-pipeline",
  "artifact_location": "mlflow-artifacts:/1", "lifecycle_stage": "active" }
```

**Train run** `fun-eel-627` (`27723f5267344c1794610d14d8bb81a5`):

| Field | Value |
|---|---|
| status | `FINISHED` |
| experiment_id | `1` |
| metric `accuracy` | `0.7398373983739838` |
| params | `best_n_estimators=100`, `best_max_depth=5`, `best_samples_split=5`, `best_samples_leaf=1` |
| model output | `m-2ad50b90b6104ea081f67e7c0e1be227` |

**Evaluate run** `resilient-skink-709` (`8e22039e6f1f4630b251140003e0e7f5`):

| Field | Value |
|---|---|
| status | `FINISHED` |
| experiment_id | `1` |
| metric `accuracy` | `0.7077922077922078` |

## 6. Model & artifacts — registered and physically stored in S3

**Model Registry** (`GET /api/2.0/mlflow/model-versions/search`):

```
name="Best Random Forest Classifier"  version=3  status=READY
source=models:/m-2ad50b90b6104ea081f67e7c0e1be227
run_id=27723f5267344c1794610d14d8bb81a5
```

**Artifact read path** through the tracking server (`/get-artifact`):

```
$ curl .../get-artifact?run_id=27723f52...&path=classification_report.txt
              precision    recall  f1-score   support
           0       0.79      0.85      0.82        85
           1       0.59      0.50      0.54        38
    accuracy                           0.74       123
```

**Artifact bytes physically in the S3 (MinIO) store** — the model was proxied
through the server (`mlflow-artifacts:`) to the bucket, not written to pod-local
disk:

```
$ kubectl exec -n mlops minio-0 -- ls /data/mlflow/artifacts/1/models/m-2ad50b90.../artifacts
MLmodel  conda.yaml  model.skops  python_env.yaml  requirements.txt

$ kubectl exec -n mlops minio-0 -- ls /data/mlflow/artifacts/1/27723f52.../artifacts
classification_report.txt  confusion_matrix.txt
```

## 7. Requirement checklist

| # | Requirement | Evidence |
|---|---|---|
| 9 | Real pipeline run against in-cluster MLflow | §3 Job `Complete`, exit 0; §4 logs to the Service |
| 10 | experiment | §5 `mlops-pipeline` (id 1) |
| 10 | run | §5 train `fun-eel-627`, evaluate `resilient-skink-709` |
| 10 | metrics | §5 accuracy 0.7398 (train) / 0.7078 (eval) |
| 10 | model / artifact | §6 registry v3 `READY` + `model.skops` in S3 + text artifacts |
| 10 | run status | §5 both runs `FINISHED` |

---

## Limitations

- **Backend is local MinIO, not real S3.** This run used the `local` overlay, so
  the artifact store is in-cluster MinIO. The AWS overlay points the *same* server
  at a Terraform-provisioned Amazon S3 bucket; that path is config-identical from
  the pipeline's side (it only ever talks to the `mlflow` Service) but was not
  exercised here — the operator provisions/destroys EKS on their own account
  (see [cloud-operations.md](../cloud-operations.md)), and this environment has no
  Terraform access.
- **DVC data remote still references DagsHub.** `.dvc/config` uses DagsHub's
  S3-compatible endpoint for **dataset/model versioning** — a separate concern from
  experiment tracking and out of scope for this PR. The in-cluster run does not
  contact it: it uses `core.no_scm=true` and a mounted dataset. Removing DagsHub
  from the **experiment-tracking runtime path** (this PR's objective) is complete;
  migrating the DVC data remote is future work.
- **Single run, deterministic data.** The accuracy figures reflect the committed
  Pima dataset and seeded pipeline; they are integration evidence, not a model
  quality claim.
