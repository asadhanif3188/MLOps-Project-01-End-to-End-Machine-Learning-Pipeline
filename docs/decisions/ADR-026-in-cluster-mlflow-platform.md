# ADR-026: Persistent in-cluster MLflow tracking platform (PostgreSQL + S3)

- **Status:** Accepted
- **Date:** 2026-08-18
- **Deciders:** Asad Hanif
- **Related:** [`k8s/base/mlflow/`](../../k8s/base/mlflow/),
  [`k8s/overlays/local/minio.yaml`](../../k8s/overlays/local/minio.yaml),
  [`docker/mlflow/Dockerfile`](../../docker/mlflow/Dockerfile),
  [`terraform/s3.tf`](../../terraform/s3.tf),
  [`terraform/ebs-csi.tf`](../../terraform/ebs-csi.tf),
  [ADR-002 (Why MLflow)](ADR-002-why-mlflow.md),
  [ADR-013 (Kubernetes Runtime Execution)](ADR-013-kubernetes-runtime-execution.md),
  [ADR-024 (VPC CNI Pod Identity)](ADR-024-vpc-cni-pod-identity.md)
- **Supersedes:** the DagsHub-hosted MLflow *hosting* decision recorded in
  [ADR-002](ADR-002-why-mlflow.md) (MLflow itself is retained; only the backend
  changes).

> **Scope.** Through Sprint 6 the pipeline logged experiments to a **DagsHub-hosted
> MLflow** SaaS (a public endpoint + committed credentials pattern), and local/CI
> runs used an **in-pod file store** (ADR-013) as an offline stand-in. This ADR
> replaces both with a **self-hosted, persistent MLflow tracking platform running
> inside Kubernetes**: a stateless MLflow Tracking Server backed by a **PostgreSQL**
> metadata database and an **S3** artifact store, reached by the pipeline over an
> internal Service. It is the design of record for the platform's architecture,
> persistence, security, and the removal of DagsHub.

## Context

The tracking backend to date had three problems this platform resolves:

1. **External dependency + committed-secret shape.** DagsHub is a third-party SaaS;
   real tracking required a `MLFLOW_TRACKING_USERNAME/PASSWORD` Secret and the
   endpoint was a public URL committed in the ConfigMap. Experiment data — models,
   metrics, run history — lived on infrastructure the project neither controls nor
   can guarantee the lifecycle of.
2. **No persistent platform of our own.** Local/CI runs used an in-pod **file
   store** (`file:///app/mlruns`, ADR-013). That is ephemeral by construction: the
   moment the pipeline pod exits, the tracking data is gone. There was no durable,
   queryable tracking backend on the cluster.
3. **The architecture goal is a real backend.** The target is
   `Pipeline Job → Service → MLflow Tracking Server → { relational metadata, S3
   artifacts }` — a backend that survives pod recreation and has no single-writer
   persistence weakness.

The constraints (from the brief and the existing platform): deploy inside
Kubernetes; keep the service internal; use Kubernetes Secrets and commit none;
add probes, resource requests/limits, and non-root/security settings; make
persistence explicit and survive pod recreation; use **workload identity** for AWS
access rather than static keys; **no GitOps**; **no Terraform remote state**; and
**do not make the platform depend on a development-only hack**.

## Decision

Deploy MLflow as a **three-tier platform**, expressed in the existing
base/overlay Kustomize structure so every environment inherits the same hardened
definition:

```
Pipeline Job ──HTTP──▶ Service "mlflow" (ClusterIP :5000)
                          │
                    MLflow Tracking Server  (Deployment, stateless)
                     --serve-artifacts  (artifact proxy)
                          ├── postgresql+psycopg2 ─▶ Service "mlflow-postgres" ─▶ PostgreSQL (StatefulSet + PVC)
                          └── boto3/S3 ────────────▶ artifact store
                                                      • local: MinIO (StatefulSet + PVC)     — creds via Secret
                                                      • aws:   Amazon S3 (terraform/s3.tf)   — creds via EKS Pod Identity
```

### 1. Backend architecture — PostgreSQL + S3, and why not SQLite

- **Metadata → PostgreSQL** (`k8s/base/mlflow/postgres.yaml`). A `StatefulSet`
  with a `volumeClaimTemplate` gives the database a stable identity and a
  PersistentVolumeClaim that is re-bound to a replacement pod, so the metadata
  survives pod recreation. **SQLite was rejected as the final architecture**: it
  is a single-file, single-writer store with no real concurrency and known
  fragility on networked/re-mounted filesystems — exactly the avoidable
  single-writer/persistence weakness the brief warns against. PostgreSQL is the
  backend MLflow documents for a real deployment.
- **Artifacts → S3** (`--artifacts-destination`). The server stores artifacts in
  an S3 bucket. Locally that S3 API is provided by **MinIO** in-cluster
  (`k8s/overlays/local/minio.yaml`); on AWS it is a real, Terraform-provisioned
  bucket (`terraform/s3.tf`). MinIO is a faithful S3 implementation, not a mock, so
  the S3 path is genuinely exercised — including in the persistence test — rather
  than stubbed. **This is the key anti-"dev-only-hack" property:** the local
  artifact store speaks the same protocol as production; only the endpoint URL and
  credential *source* differ.
- **The MLflow Tracking Server is stateless** (`k8s/base/mlflow/deployment.yaml`),
  a `Deployment`, because all durable state is external (Postgres + S3). A
  replacement pod reconnects to the same data — which is exactly the property the
  persistence test exercises.

### 2. Artifact proxying — one identity holds S3 access, not every client

The server runs with `--serve-artifacts`, so it is the **single component that
talks to S3**. Clients (the pipeline Job) upload/download artifacts **through** the
tracking server over HTTP using the `mlflow-artifacts:` scheme. Consequences:

- The pipeline needs **no S3 endpoint and no S3 credentials** — and its image
  needs no `boto3`. It logs to `http://mlflow…:5000` and nothing else.
- Only **one** workload identity is ever granted S3 access (the server's), which is
  what makes the AWS workload-identity story tight (§4).

### 3. Persistence — explicit, and survives pod recreation

- **Metadata:** the Postgres `volumeClaimTemplate` (a 1Gi PVC from the default
  StorageClass). Deleting `mlflow-postgres-0` re-binds the *same* volume to the new
  pod, so runs/experiments persist.
- **Artifacts:** the MinIO `volumeClaimTemplate` locally; the versioned,
  CMK-encrypted (SSE-KMS) S3 bucket on AWS.
- **Tracking server:** deliberately holds **no** state, so deleting the MLflow pod
  loses nothing.
- **On EKS**, dynamic PVC provisioning requires the **EBS CSI driver**
  (`terraform/ebs-csi.tf`) — without it the Postgres PVC would stay `Pending`. It
  is installed as a managed addon whose controller draws AWS permissions from **Pod
  Identity**, consistent with §4. The default `gp2` StorageClass (CSI-migrated) then
  provisions the EBS-backed volume.

### 4. Security model

- **Internal-only exposure (requirement 13).** Every Service is `ClusterIP`
  (`mlflow`, `mlflow-postgres`, `minio`) — none is a `NodePort`/`LoadBalancer`. The
  tracking server runs with **no authentication**, which is acceptable *precisely
  because* it is unreachable outside the cluster; exposing a no-auth MLflow to the
  internet would be an unauthenticated data-exfiltration surface. UI access is done
  deliberately and temporarily via `kubectl port-forward`, not a standing endpoint.
  Adding an auth proxy (e.g. oauth2-proxy) is the documented prerequisite for any
  future external exposure. MLflow 3.x additionally ships a **host-validation
  (DNS-rebinding) middleware**; it is KEPT ON and scoped via
  `MLFLOW_SERVER_ALLOWED_HOSTS` to this project's in-cluster Service names plus
  localhost (rather than disabled), so an unexpected `Host` header is still
  rejected even though the server is already ClusterIP-internal.
- **No committed credentials.** The Postgres credentials and the local MinIO/S3
  keys live in **out-of-band Secrets** created by the operator (templates:
  `k8s/base/mlflow/secret.example.yaml`, `k8s/overlays/local/secret.example.yaml`),
  never rendered by Kustomize and never in git — the same discipline as the
  removed DagsHub Secret. The DB password is delivered to the server via libpq
  `PGUSER/PGPASSWORD` env, so it is **never in the process argv and never in the
  connection URI** (also sidestepping URL-encoding pitfalls).
- **AWS access via workload identity, not static keys.** On EKS the tracking
  server's `mlflow-server` ServiceAccount is bound to a dedicated IAM role via
  **EKS Pod Identity** (`terraform/s3.tf`), scoped by a least-privilege inline
  policy to exactly the one artifact bucket. There are **no static AWS keys** on
  the cluster — boto3 resolves short-lived, pod-scoped credentials automatically.
  This reuses the Pod Identity mechanism established for the VPC CNI (ADR-024) and
  the EBS CSI controller. (Locally, where there is no AWS, MinIO uses a Secret — the
  one place static keys exist, and only for a local S3 emulator.)
- **Workload hardening.** Every platform pod runs **non-root** with a
  `seccompProfile: RuntimeDefault`, `allowPrivilegeEscalation: false`, and all
  Linux capabilities dropped: the MLflow server and MinIO as uid `10001`/`1000`,
  PostgreSQL as its image's non-root `postgres` user (uid `999`) with an `fsGroup`
  so the PVC is writable. `automountServiceAccountToken: false` everywhere (no pod
  calls the Kubernetes API). `readOnlyRootFilesystem` is left `false` for the
  platform pods (as for the pipeline Job, ADR-010/013) — the servers write to
  in-tree paths (`/tmp`, PGDATA, the MinIO data dir); relocating those to enable a
  read-only root is the same tracked follow-up as for the pipeline.
- **Probes.** The MLflow server has startup/readiness/liveness probes on
  `/health`; PostgreSQL and MinIO have readiness/liveness probes (`pg_isready`,
  `/minio/health/*`). Readiness gates dependents (the pipeline waits for a ready
  server); liveness restarts a wedged process.

### 5. Why DagsHub was removed

DagsHub was the **experiment-tracking host** (ADR-002's hosting choice). It is
removed because:

- it is a third-party SaaS holding the project's experiment data, with a
  committed-endpoint + injected-credential shape;
- the project now has a **self-hosted, persistent platform under its own control**,
  which is the actual goal of an "in-cluster MLflow" milestone; and
- keeping DagsHub would leave the platform's default tracking path dependent on
  external infrastructure and secrets.

Concretely: `MLFLOW_TRACKING_URI` now points at the in-cluster Service; the
DagsHub credential Secret ref and the pipeline's file-store override are gone; the
`dagshub` Python dependency is dropped; and `k8s/validate.py` asserts **no DagsHub
reference** survives in any rendered manifest.

> **Out of scope (deliberately unchanged):** DVC's separate **S3-compatible data
> remote** in `.dvc/config` (data/model *versioning*, ADR-003) still points at
> DagsHub storage. That is a different concern from experiment *tracking* and is
> left untouched here; migrating it is a separate, future decision.

### 6. What is NOT introduced

- **No GitOps.** The platform is deployed with `kubectl apply -k`, exactly like the
  existing workload — no Argo/Flux, no reconciler.
- **No Terraform remote state.** `terraform/s3.tf` adds an *artifact* bucket; it is
  **not** a state backend. Local state remains (ADR-014).
- **No development-only hack as a load-bearing dependency.** MinIO and the
  bucket-bootstrap Job are real, scoped to the local overlay, and never rendered on
  AWS; the production S3 path is the same code path with a different endpoint and
  credential source.

## Alternatives Considered

1. **Keep DagsHub-hosted MLflow.** *Rejected* — the milestone's entire point is a
   self-hosted, persistent, in-cluster platform under the project's control; a SaaS
   with committed-credential shape is what we are moving away from.
2. **SQLite backend store.** *Rejected as the final architecture* — single-file,
   single-writer, and fragile on networked/re-mounted volumes; it is the avoidable
   persistence weakness the brief calls out. PostgreSQL removes it.
3. **File-store artifacts / no S3.** *Rejected* — a file store on a PVC does not
   match the target architecture, couples the server to a specific node's volume,
   and is not the production shape. S3 (MinIO locally, real S3 on AWS) is the
   documented artifact backend and keeps local and cloud identical bar the endpoint.
4. **Amazon RDS for the metadata DB.** *Deferred* — a managed database is a
   reasonable production hardening, but it enlarges the AWS surface (subnet groups,
   security groups, parameter groups) well beyond this PR and contradicts the
   "in-cluster MLflow platform" framing. In-cluster PostgreSQL on an EBS-backed PVC
   satisfies the persistence requirement now; RDS is a noted future option.
5. **Direct client → S3 artifact access (no `--serve-artifacts`).** *Rejected* — it
   would force the pipeline (and any future client) to hold S3 credentials/endpoint
   config and pull in `boto3`, and would grant S3 access to more identities than
   necessary. Proxying keeps S3 access to the single server identity.
6. **`pip install` the DB driver / boto3 at container start on a stock MLflow
   image.** *Rejected* — a runtime install makes every server start depend on a
   reachable package index (offline-hostile, non-reproducible) — precisely the
   development-only hack the brief forbids. A **purpose-built, pinned server image**
   (`docker/mlflow/Dockerfile`) bakes and import-checks the stack at build time.
7. **IRSA instead of EKS Pod Identity for S3 access.** *Rejected for consistency* —
   the project already standardised on Pod Identity (no OIDC/TLS-thumbprint
   bookkeeping) for the VPC CNI (ADR-024); the MLflow S3 role and the EBS CSI role
   reuse it.
8. **Expose MLflow via LoadBalancer/Ingress.** *Rejected (default)* — a no-auth
   server must stay internal (requirement 13). External access requires an auth
   layer first and is deferred.

## Consequences

**Positive**

- **A real, persistent tracking platform**, self-hosted and under project control:
  PostgreSQL metadata + S3 artifacts, reached over an internal Service.
- **Persistence proven, not assumed.** Verified on Docker Desktop Kubernetes: a
  logged run survives deletion of the MLflow server pod **and** the PostgreSQL pod
  (PVC re-bind) — see *Runtime evidence* below.
- **No secrets in git; AWS access on workload identity.** Credentials are
  out-of-band Secrets; on AWS the server uses Pod Identity, so no static keys exist.
- **Client simplification.** The pipeline logs over HTTP with no S3 config and no
  credentials; artifacts are proxied through the server.
- **Deterministic ordering.** The pipeline Job carries a `wait-for-mlflow` init
  container that polls the server's `/health` (exempt from host validation) and
  blocks the pipeline until it is Ready — turning a start-order race into a bounded
  wait. Verified: with the server scaled to 0 the init container blocks; the instant
  it is scaled back up the init passes and the pipeline completes.
- **AWS path fully wired.** Terraform provisions a dedicated `mlflow-server` ECR
  repository (alongside the pipeline repository) with the same hardened contract, so
  the AWS overlay's server-image reference resolves to a repository Terraform owns.
- **Same contract, statically checked.** `k8s/validate.py` gained an "MLflow
  tracking platform" section (server probes/resources/hardening, internal-only
  Service, explicit Postgres PVC, no-DagsHub, in-cluster tracking URI) that runs in
  CI over both overlays; `terraform test` pins the bucket + Pod Identity contract.

**Trade-offs and follow-ups**

- **Single replica each** (MLflow, PostgreSQL). Right-sized for a single-operator
  batch pipeline; HA (a Postgres operator/RDS, multiple stateless server replicas)
  is future work, not a hidden gap.
- **No tracking-server authentication.** Acceptable only because the server is
  ClusterIP-internal; an auth proxy is the prerequisite for any external exposure.
- **`readOnlyRootFilesystem: false`** on the platform pods — the same deferred item
  as the pipeline Job (ADR-010/013).
- **AWS path is authored, not applied here.** The Terraform (S3 + Pod Identity +
  EBS CSI) and the AWS overlay are provisioned by the operator against their own
  account (consistent with the project's operator-driven apply model); the
  *executed* evidence in this PR is on the local cluster.
- **DVC data remote still on DagsHub storage** (out of scope, above).

## Runtime evidence

Executed end to end on **Docker Desktop Kubernetes v1.34.3** (local cluster),
overlay `k8s/overlays/local`, images `ml-pipeline:local` + `mlflow-server:local`
(MLflow **3.15.1**, matching the pipeline client):

**1. Platform came up healthy.**

```
$ kubectl -n mlops get pods
NAME                     READY   STATUS      RESTARTS   AGE
minio-0                  1/1     Running     0          ...   # S3 artifact store (PVC-backed)
minio-setup-xxxxx        0/1     Completed   0          ...   # created bucket local/mlflow
mlflow-xxxxxxxxxx-xxxxx  1/1     Running     0          ...   # tracking server (2Gi)
mlflow-postgres-0        1/1     Running     0          ...   # metadata DB (PVC-backed)
$ curl -s .../health -> 200 OK
```

**2. The pipeline ran to completion against the in-cluster platform** (Job
`Complete`, exit 0, all four stages), logging to PostgreSQL + S3:

```
preprocess: 768 rows -> split: 614 train / 154 test
train: Best model accuracy: 0.7398 ; "Successfully registered model 'Best Random Forest Classifier'"
evaluate: model accuracy: 0.7078 ; Job Complete 1/1
```

Backends verified directly (MLflow client against the server):
- **Metadata in PostgreSQL:** 3 runs (train acc 0.7398, evaluate acc 0.7078) and
  Model Registry entry `Best Random Forest Classifier` v1 — the registry works
  because the backend is a real DB (a file store cannot register).
- **Artifacts in S3 (MinIO):** 7 objects under `artifacts/0/…`, including the
  registered model (`model.skops` 2.0 MiB, `MLmodel`, `conda.yaml`, …) and the
  run's text artifacts. The pipeline uploaded them via the `mlflow-artifacts:`
  proxy with **no S3 credentials of its own**.

**3. Persistence test — the real proof (not "Deployment Ready").** All three pods
were **deleted** and left to be recreated:

```
$ kubectl -n mlops delete pod mlflow-postgres-0 minio-0 <mlflow-pod>
# StatefulSets recreate pods bound to the SAME PVCs:
data-mlflow-postgres-0  ->  pvc-389adfb0…   (identical before & after)
data-minio-0            ->  pvc-680c5fbb…   (identical before & after)
```

After every pod was recreated fresh (RESTARTS 0):

| Check | Before | After pod recreation |
|---|---|---|
| Run count (PostgreSQL) | 3 | **3** (same run IDs, same metrics) |
| Registered model | `Best Random Forest Classifier` v1 | **v1, intact** |
| Artifact `confusion_matrix.txt` (MinIO) | `[[72 13]\n [19 19]]` | **byte-identical, still downloadable** |

The metadata survived because it lives in the PostgreSQL PVC; the artifacts
survived because they live in the MinIO PVC; the stateless MLflow server
reconnected to both with zero data loss. **This is the persistence guarantee M-03
requires — proven by destroying the stateful pods and recovering the data, not by
a Ready probe.**

**4. Notable finding (recorded honestly).** MLflow 3.15.1's server has a MEASURED
steady footprint of **~1.7 GiB** (uvicorn + the full 3.15 stack) — far heavier
than a 2.x server. A 1 GiB limit OOM-kills the worker on startup (verified,
exit 137); the memory limit is therefore set to **2 GiB** from this measurement,
not guessed, and the server runs `--workers 1` so the worker count does not
multiply that footprint.

## What This Decision Does *Not* Imply

- It does **not** claim a production deployment: the executed run is on a **local**
  cluster with MinIO as the S3 backend; the real-S3/EKS path is authored and
  operator-applied.
- It does **not** claim HA, multi-replica, or disaster recovery for the metadata DB
  or the server.
- It does **not** claim the tracking server is safe to expose externally — it is
  internal-only and unauthenticated by design.
- It does **not** change the pipeline's reproducibility guarantee (`params.yaml` +
  pinned deps + seeded RNG + the DVC DAG, ADR-006), nor DVC's data-versioning remote.
