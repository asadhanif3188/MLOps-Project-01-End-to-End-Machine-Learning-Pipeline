# Architecture

The **End-to-End Machine Learning Pipeline** is architected around
reproducibility: a Random Forest training workflow orchestrated with
[DVC](https://dvc.org/) and instrumented with [MLflow](https://mlflow.org/),
tracked on the project's **in-cluster MLflow platform** (self-hosted server +
PostgreSQL + S3; Sprint 7, [ADR-026](decisions/ADR-026-in-cluster-mlflow-platform.md)).
This document explains how the pieces fit together.

> **Scope note.** This document reflects the repository as it exists today. Where
> a component is planned but not yet implemented, it is marked with an explicit
> `TODO`. Diagrams referenced here live under [`diagrams/`](diagrams/) and are
> placeholders until produced in a later sprint.

---

## 1. System Overview

The system is a **batch, file-based ML pipeline**. It transforms a raw CSV
dataset into a trained, serialized model while tracking experiments and
versioning data and artifacts.

There is no online serving component today; the pipeline runs on demand (locally
or via `dvc repro`) and produces:

- a preprocessed dataset (`data/processed/data.csv`),
- a train/held-out split of that dataset (`data/processed/train.csv` and
  `data/processed/test.csv`),
- a serialized model (`models/model.pkl`),
- a metrics artifact (`metrics/metrics.json`, DVC-tracked),
- experiment metadata (parameters, metrics, artifacts) logged to the in-cluster
  MLflow tracking server (PostgreSQL metadata backend, S3 artifact store).

The stage inputs, outputs, artifact ownership, evaluation boundary, and
reproducibility expectations are specified in the
[Pipeline Contract](pipeline-contract.md) and enforced by the `contract` test
suite and CI (see [§3](#3-pipeline-flow) and [CI/CD](ci-cd.md)).

The dataset is the **Pima Indians Diabetes** dataset; the task is binary
classification of the `Outcome` column.

**Key properties:**

| Property | Current state |
|----------|---------------|
| Execution model | Batch, on-demand |
| Orchestration | DVC pipeline (`dvc.yaml`) |
| Experiment tracking | In-cluster MLflow platform (self-hosted server + PostgreSQL + S3) — [MLflow Platform](mlflow-platform.md), [ADR-026](decisions/ADR-026-in-cluster-mlflow-platform.md) |
| Data/artifact versioning | DVC with S3-compatible remote (DagsHub — **data/model remote only**) |
| Model type | `RandomForestClassifier` (scikit-learn) |
| Packaging | Multi-stage Docker image (non-root `runtime`) — [Containerization](containerization.md) |
| Local dev environment | Docker Compose (`dev` service) — [Docker Development](docker-development.md) |
| Continuous integration | GitHub Actions: lint, test, image build/validate — [CI/CD](ci-cd.md) |
| Kubernetes orchestration | ✅ `mlops` namespace + hardened `batch/v1` **Job** (Kustomize, [`k8s/`](../k8s/)); run green to completion on a **local** cluster (Sprint 5) **and on real Amazon EKS** (Sprint 6, and the full hardened platform in Sprint 7) — [Kubernetes Architecture](kubernetes-architecture.md), [ADR-009](decisions/ADR-009-kubernetes-workload-model.md) |
| Cloud platform (IaC) | ✅ AWS VPC + least-privilege IAM + managed **EKS** + Terraform-managed **ECR**, **KMS**-encrypted **S3** (dataset + MLflow artifacts), and **EKS Pod Identity** workload identity, all as **Terraform** ([`terraform/`](../terraform/)); provisioned, run, and torn down as a short-lived validation environment — [§8](#8-cloud-platform-terraform--eks), [terraform/README.md](../terraform/README.md) |
| Serving | ❌ Not implemented — see [roadmap](roadmap.md) |

---

## 2. Component Diagram

<!-- TODO: Add the rendered component diagram under
     diagrams/system-architecture/ and embed it here once produced. -->

> 📌 **Diagram placeholder:** [`diagrams/system-architecture/`](diagrams/system-architecture/)

The major components and their relationships:

```text
        params.yaml ───────────────────────────────────────────┐ (parameters)
                                                                ▼
data/raw/data.csv ─▶ [ preprocess ] ─▶ data/processed/data.csv ─▶ [ split ]
                                                                       │
                              ┌────────────────────────────────────────┤
                              ▼                                         ▼
                   data/processed/train.csv               data/processed/test.csv
                              │                                         │
                              ▼                                         ▼
                        [ train ] ─▶ models/model.pkl ─▶ [ evaluate ] ─▶ metrics/metrics.json
                              │                               │
                              ▼                               ▼
                 In-cluster MLflow (server + PostgreSQL + S3) ◀── metrics, params, artifacts
```

**Components:**

- **`src/preprocess.py`** — reads the raw dataset and writes the processed CSV
  (with header) that `split` consumes.
- **`src/split.py`** — partitions the processed dataset into a training set
  (`data/processed/train.csv`) and a **held-out** evaluation set
  (`data/processed/test.csv`) with a stratified, seeded `train_test_split`, and
  asserts the two partitions are disjoint and exhaustive. This stage is what makes
  held-out evaluation a property of the DAG (see
  [ADR-007](decisions/ADR-007-held-out-evaluation.md)).
- **`src/train.py`** — trains a seeded Random Forest on the **training** partition
  (`GridSearchCV` tunes leaf/split regularization; `n_estimators`/`max_depth`/
  `random_state` come from `params.yaml`), serializes the best estimator to
  `models/model.pkl`, and logs the run to MLflow. Never reads the held-out set.
- **`src/evaluate.py`** — loads the model, scores it on the **held-out** partition,
  writes `metrics/metrics.json`, and logs the accuracy to MLflow.
- **`src/tracking.py`** — the single MLflow boundary: every experiment-tracking
  call goes through it, and the stages import it lazily so their ML computation
  stays free of MLflow and unit-testable without it.
  [`src/mlflow_config.py`](../src/mlflow_config.py) resolves the tracking URI and
  experiment name from config before the (lazy) MLflow import.
- **DVC** — defines stage dependencies and outputs (`dvc.yaml`) and versions
  data/model artifacts to the DagsHub S3-compatible remote (data/model versioning
  only; not the experiment-tracking path).
- **In-cluster MLflow platform** — self-hosted tracking server (Deployment,
  `--serve-artifacts`) backed by a PostgreSQL metadata store and an S3 artifact
  store (MinIO locally, Amazon S3 on EKS), plus the MLflow Model Registry
  ([MLflow Platform](mlflow-platform.md),
  [ADR-026](decisions/ADR-026-in-cluster-mlflow-platform.md)).

The stages share a small infrastructure layer (introduced in Sprint 2, extended
in Sprint 4 with the tracking boundary):

- **`src/logging_config.py`** — centralized logging configuration (console +
  rotating file); see [§7](#7-observability--logging).
- **`src/exceptions.py`** — the typed exception hierarchy rooted at
  `PipelineError`.
- **`src/pipeline_io.py`** — IO/config/serialization helpers that wrap every
  filesystem and config boundary with consistent, typed error handling.
- **`src/stage_runner.py`** — the uniform stage entry point: logs any failure
  once (with traceback) and exits non-zero.

---

## 3. Pipeline Flow

The pipeline is defined in [`dvc.yaml`](../dvc.yaml) as four stages executed in
dependency order. The `split` stage forks the flow into a training path and a
disjoint held-out evaluation path.

> 📌 **Diagram placeholder:** [`diagrams/pipeline-flow/`](diagrams/pipeline-flow/)

### Stage 1 — `preprocess`

- **Command:** `python src/preprocess.py`
- **Inputs:** `data/raw/data.csv`, `src/preprocess.py`,
  params `preprocess.input`, `preprocess.output`
- **Output:** `data/processed/data.csv` (written **with** its header row).

### Stage 2 — `split`

- **Command:** `python src/split.py`
- **Inputs:** `data/processed/data.csv`, `src/split.py`, and params `split.input`,
  `split.train_output`, `split.test_output`, `split.target`, `split.test_size`,
  `split.random_state`
- **Outputs:** `data/processed/train.csv` and `data/processed/test.csv` (both
  written **with** headers).
- **Behavior:** performs a **stratified** `train_test_split` seeded by
  `split.random_state`, holding out `split.test_size` (0.2) of the rows for
  evaluation. Asserts the two partitions are **disjoint** (no shared row → no
  leakage into evaluation) and **exhaustive** (their union is the input). Seeded →
  the exact held-out rows are reproducible. Crosses no MLflow boundary.

### Stage 3 — `train`

- **Command:** `python src/train.py`
- **Inputs:** `data/processed/train.csv`, `src/train.py`, and params
  `train.input`, `train.output`, `train.target`, `train.random_state`,
  `train.n_estimators`, `train.max_depth`
- **Output:** `models/model.pkl`
- **Behavior:** consumes **only** the training partition, takes an internal
  validation `train_test_split` **within it** (20%, seeded by
  `train.random_state`) for in-training reporting — never the held-out set — fits a
  `RandomForestClassifier` whose `n_estimators`/`max_depth`/`random_state` come
  from config, runs a small `GridSearchCV` (3-fold) tuning leaf/split
  regularization, logs parameters/metrics/artifacts to MLflow, conditionally
  registers the model, and pickles the best estimator. Seeded → deterministic
  given the same inputs and parameters.

### Stage 4 — `evaluate`

- **Command:** `python src/evaluate.py`
- **Inputs:** `data/processed/test.csv`, `models/model.pkl`, `src/evaluate.py`,
  and params `evaluate.data`, `evaluate.model`, `evaluate.target`,
  `evaluate.metrics`
- **Output:** `metrics/metrics.json` (DVC-tracked metrics artifact, `cache:
  false`); also logs the `accuracy` metric to MLflow.

> ✅ **Held-out evaluation.** `evaluate` scores the model on
> `data/processed/test.csv` — the partition `split` held out and `train` never
> reads — so the reported `accuracy` is a genuine **out-of-sample** figure. The
> disjointness is guaranteed on three independent layers: the DVC DAG topology, the
> `contract` disjointness test
> (`test_train_and_evaluate_consume_disjoint_datasets`), and `split`'s runtime
> assertions. This closes deviation **D5** in the
> [Pipeline Contract](pipeline-contract.md#8-evaluation-boundary); see
> [ADR-007](decisions/ADR-007-held-out-evaluation.md). Remaining caveats are quality
> refinements (single split, not cross-validated; small held-out set) rather than a
> correctness gap. Reproducibility is likewise proven by execution: a committed
> `dvc.lock` is reproduced by a real `dvc repro` in CI against a self-contained
> fixture pipeline (closing the `dvc.lock`/execution portion of D7;
> [ADR-008](decisions/ADR-008-fixture-reproducibility.md)). The remaining
> limitation is documented, not a gap — reproducing the *production* run end to end
> needs the remote-only dataset, live MLflow, and digest-pinned deps
> ([pipeline-contract §7](pipeline-contract.md#7-reproducibility-expectations),
> level 4).

---

## 4. Technology Choices

| Concern | Technology | Rationale (see ADR) |
|---------|-----------|---------------------|
| Pipeline orchestration & data versioning | DVC (+ `dvc-s3`) | [ADR-003](decisions/ADR-003-why-dvc.md) |
| Experiment tracking / model registry | In-cluster MLflow (server + PostgreSQL + S3) | [ADR-002](decisions/ADR-002-why-mlflow.md), [ADR-026](decisions/ADR-026-in-cluster-mlflow-platform.md) |
| Repository structure | Stage-per-script `src/` layout | [ADR-001](decisions/ADR-001-repository-structure.md) |
| Model | `RandomForestClassifier` (scikit-learn) | Baseline for tabular classification |
| Config | `params.yaml` (declarative parameters) | Separates config from code |
| Secrets | `python-dotenv` + `.env` | Keeps credentials out of source |
| Logging | Python `logging` (stdlib), centralized config | [Logging Strategy](logging.md) |
| Error handling | Typed exception hierarchy (`PipelineError`) | [Exception Strategy](exception-strategy.md) |
| Lint/format/type/test toolchain | Ruff + mypy + pytest + pre-commit | [ADR-004](decisions/ADR-004-python-quality-toolchain.md) |
| Containerization | Multi-stage Docker (`python:3.12-slim`) | [ADR-005](decisions/ADR-005-containerization-strategy.md) |
| Continuous integration | GitHub Actions | [CI/CD](ci-cd.md) |
| Kubernetes workload model | `batch/v1` Job + Kustomize (base + local/aws overlays) | [ADR-009](decisions/ADR-009-kubernetes-workload-model.md) |
| Cloud platform (IaC) | Terraform → AWS VPC/IAM + managed EKS | [ADR-014](decisions/ADR-014-terraform-architecture.md)…[ADR-020](decisions/ADR-020-cloud-lifecycle-cost-control.md) |

For the reasoning behind these choices, see [Design Principles](design-principles.md).
Full dependency lists: [`requirements.txt`](../requirements.txt) (runtime) and
[`requirements-dev.txt`](../requirements-dev.txt) (development tooling).

---

## 5. Data Flow

> 📌 **Diagram placeholder:** [`diagrams/pipeline-flow/`](diagrams/pipeline-flow/)

1. **Ingestion.** The raw dataset is tracked by DVC
   (`data/raw/data.csv.dvc`) and retrieved from the S3-compatible DagsHub remote
   via `dvc pull`.
2. **Preprocessing.** `preprocess` reads the raw CSV and writes
   `data/processed/data.csv` (a DVC-tracked output).
3. **Splitting.** `split` partitions the processed dataset into
   `data/processed/train.csv` (training) and `data/processed/test.csv` (held-out
   evaluation) with a stratified, seeded split — both DVC-tracked outputs, disjoint
   and exhaustive.
4. **Training.** `train` reads **only** the training partition, tunes and fits the
   model, and writes `models/model.pkl` (DVC-tracked, git-ignored on disk).
5. **Tracking.** During training and evaluation, parameters, metrics
   (`accuracy`), and artifacts (confusion matrix, classification report, model)
   are logged to the in-cluster MLflow tracking server through the `tracking`
   boundary — metadata to PostgreSQL, artifact bytes proxied to S3
   (`--serve-artifacts`), so the pipeline needs no S3 or MLflow credentials.
6. **Evaluation.** `evaluate` loads the pickled model, scores it on the
   **held-out** partition (`data/processed/test.csv`), writes
   `metrics/metrics.json`, and logs the accuracy metric.

**Data classification & storage:**

- Raw and processed data are **not** stored in Git; they are versioned by DVC and
  pushed to the remote.
- Models are git-ignored (`models/` in `.gitignore`) and versioned by DVC.
- Credentials live only in `.env` (see [`.env.example`](../.env.example)).

---

## 6. Containerization & Continuous Integration

Sprint 3 packaged the pipeline for reproducible execution and automated its
quality gates. Both are **implemented and in the repository today**.

### Containerization

- A single multi-stage [`Dockerfile`](../Dockerfile) builds three targets from
  one source of truth: `builder` (installs the dependency virtualenv),
  `development` (adds the Ruff/mypy/pytest/pre-commit toolchain), and `runtime`
  (the default — a lean, **non-root** production image on `python:3.12-slim`).
- A [`.dockerignore`](../.dockerignore) keeps the build context small and free of
  secrets, data, and history.
- State (`data/`, `models/`, `logs/`) is mounted as volumes and credentials are
  injected at run time — nothing sensitive is baked into the image.
- The full rationale is in the [Containerization Strategy](containerization.md)
  and [ADR-005](decisions/ADR-005-containerization-strategy.md).

### Local development environment

- A [`docker-compose.yml`](../docker-compose.yml) provides a `dev` service (the
  `development` image, working tree bind-mounted for live edits) and a
  profile-gated `pipeline` service that runs the production image's `dvc repro`.
  A contributor runs `docker compose up` and works entirely in-container. See the
  [Docker Development Workflow](docker-development.md).

### Continuous integration

- A [GitHub Actions workflow](../.github/workflows/ci.yml) validates every push to
  `main` and every pull request: checkout → Python 3.12 → install dependencies →
  Ruff (lint + format check) → pytest (including the `contract` tests) → **DVC
  pipeline integrity** (`dvc dag` + `dvc status`, offline, analytics disabled) →
  Docker build of the `runtime` image → build validation (non-root UID, core
  imports, `dvc` entrypoint). The DVC and contract checks (added in Sprint 4)
  fail a PR when the graph, parameter contract, or artifact lineage is broken —
  without contacting the DagsHub remote. It is validation-only — it does not
  deploy, push images, or use Kubernetes. See [CI/CD](ci-cd.md).

> 📌 **Diagram placeholders:**
> [`diagrams/cicd-flow/`](diagrams/cicd-flow/),
> [`diagrams/deployment-architecture/`](diagrams/deployment-architecture/)

### Kubernetes orchestration

Sprint 5 expressed the containerized pipeline as a Kubernetes workload, and it is
now **runnable and proven**. Under [`k8s/`](../k8s/): an `mlops` namespace and the
pipeline modelled as a run-to-completion **`batch/v1` Job** (not a Deployment — the
workload is finite batch, so it has no service to keep alive), structured with
Kustomize (`base/` + `overlays/local/` + `overlays/aws/`). The base carries the
enforced security context, externalized config, an out-of-band credential Secret, a
least-privilege ServiceAccount with no API token, and measured resource
requests/limits; the manifests are validated statically in CI. The complete pipeline
**runs green to completion (exit 0)** as a secured Job — on a **local** cluster
(Sprint 5, [ADR-013](decisions/ADR-013-kubernetes-runtime-execution.md)) and on
**real Amazon EKS** (Sprint 6, [runtime evidence](proof/sprint-06-runtime-evidence.md)).
The rationale is in [Kubernetes Architecture](kubernetes-architecture.md) and
[ADR-009](decisions/ADR-009-kubernetes-workload-model.md)…[ADR-013](decisions/ADR-013-kubernetes-runtime-execution.md),
with the batch-workload diagram under
[`diagrams/kubernetes-architecture/`](diagrams/kubernetes-architecture/).

---

## 7. Observability & Logging

All pipeline stages emit structured logs through a single, centralized
configuration in [`src/logging_config.py`](../src/logging_config.py). This
replaces the ad-hoc `print()` statements previously used for diagnostics
(engineering review finding **H-1**).

- **Configuration.** `configure_logging()` sets up console **and** rotating-file
  handlers on the root logger at each stage's entry point; `get_logger(name)`
  returns a per-stage logger.
- **Destinations.** Logs stream to the console and persist to
  `logs/pipeline.log` (git-ignored, rotating at 5 MB × 3 backups).
- **Levels.** Controlled by the `LOG_LEVEL` environment variable (default
  `INFO`): `INFO` for lifecycle, `WARNING` for recoverable issues, `ERROR` for
  failures, `DEBUG` for development only.
- **Lifecycle logs.** Each stage emits deliberate start/completion logs plus key
  outcomes (e.g. best model accuracy, model saved) — enough to trace a run
  without over-logging.

For the full policy — format, level guidance, and per-stage log inventory — see
[Logging Strategy](logging.md).

### Error handling

Failures are handled through a small typed exception hierarchy
([`src/exceptions.py`](../src/exceptions.py)) rooted at `PipelineError`, with
IO/config/model boundaries centralized in
[`src/pipeline_io.py`](../src/pipeline_io.py) and a uniform stage entry point
([`src/stage_runner.py`](../src/stage_runner.py)) that logs each failure once —
with the full traceback — and exits non-zero so `dvc repro` and CI stop on error
(engineering review finding **H-2**). See [Exception Strategy](exception-strategy.md)
for the hierarchy, propagation rules, and user-facing error contract.

---

## 8. Cloud Platform (Terraform + EKS)

Sprint 6 defines the managed cloud platform the pipeline runs on **entirely as
Infrastructure as Code** under [`terraform/`](../terraform/), and integrates the
existing workload with it — **no application logic changes**. The Terraform stops at
infrastructure; the Kubernetes workload stays in Kustomize
([ADR-014](decisions/ADR-014-terraform-architecture.md)).

Sprint 7 then hardened this platform and made it fully cloud-native: two
Terraform-managed ECR repositories, KMS-encrypted S3 stores for the dataset and
MLflow artifacts, EKS Pod Identity workload identity, and the in-cluster MLflow
tracking platform.

```text
   Terraform (terraform/)                          Kubernetes (k8s/overlays/aws)
   ─────────────────────                           ─────────────────────────────
   VPC 10.0.0.0/16                                 Namespace + ServiceAccounts
     ├─ 2 public + 2 private subnets (2 AZs)       ConfigMaps + out-of-band DB Secret
     ├─ IGW + 1 shared NAT + EIP                   hardened batch/v1 Job (base, verbatim)
     ├─ 6 IAM roles (cluster, node, VPC-CNI,         ├─ pipeline image ← ECR (node-role pull)
     │   EBS-CSI, MLflow-S3, dataset-reader)         ├─ dataset ← S3 (fetch-dataset init,
     ├─ 2 ECR repos (pipeline + mlflow-server)       │             Pod Identity, read-only)
     │   + lifecycle policies (immutable tags)       ├─ in-cluster MLflow (server + Postgres
     ├─ 3 KMS CMKs (+ aliases): EKS-secrets,         │             + S3 artifacts, ClusterIP)
     │   datasets, mlflow-artifacts                  └─ security context unchanged
     ├─ 2 S3 buckets (datasets, mlflow-artifacts) — versioned, SSE-KMS, public access blocked
     ├─ EBS CSI driver addon (Postgres PVC)
     └─ EKS control plane (K8s 1.35)  ◀── node ───▶
         │   API endpoint: PRIVATE by default (public = scoped opt-in, never 0.0.0.0/0)
         │   Access: explicit EKS access entries (authentication_mode=API, no creator-admin)
         │   Secrets: KMS-envelope-encrypted (customer-managed CMK)
         │   Workload identity: EKS Pod Identity (VPC-CNI, EBS-CSI, MLflow-S3, dataset-reader)
         └─ 1 managed node group (t3.medium, AL2023, private subnets)
```

- **What it provisions.** ~65 managed resources — a VPC with public/private subnets
  across two AZs and a single shared NAT gateway; six least-privilege IAM roles
  (cluster, node, VPC-CNI, EBS-CSI, MLflow-S3, dataset-reader); a managed EKS control
  plane + one small node group + the three core addons + the `eks-pod-identity-agent`
  and `aws-ebs-csi-driver` addons; **two** Terraform-managed **ECR** repositories
  (`mlops-pipeline` + `mlflow-server`, immutable tags, scan-on-push, lifecycle
  retention — Sprint 7 PR 1/PR 6, closing finding H-01); **three** customer-managed
  **KMS keys** (+ aliases) for EKS Secret envelope encryption, the dataset bucket, and
  the MLflow artifact bucket; and **two** private, versioned, SSE-KMS **S3 buckets**
  (dataset store + MLflow artifact store, all public access blocked). The security
  posture: the EKS API endpoint is **secure by default** — private-only, with public
  access only as a scoped opt-in that can never be `0.0.0.0/0` (Sprint 7 PR 2, closing
  finding H-02); **cluster access is via explicit, scoped EKS access entries**
  (`authentication_mode = API`), not automatic creator-admin (Sprint 7 PR 3, closing
  finding H-03); the **VPC CNI runs under its own role via EKS Pod Identity**, off the
  node instance profile (Sprint 7 PR 4, closing finding M-01); **Kubernetes Secrets
  are envelope-encrypted with a customer-managed KMS key** (Sprint 7 PR 5, closing
  finding M-02); and the **runtime dataset is retrieved from a private S3 bucket via
  Pod Identity**, not a ConfigMap (Sprint 7 PR 8, closing finding M-04). No GPUs, no
  autoscaler, no ingress/mesh/observability stack, no unrelated services
  ([ADR-015](decisions/ADR-015-aws-network-architecture.md),
  [ADR-016](decisions/ADR-016-aws-iam-foundation.md),
  [ADR-017](decisions/ADR-017-eks-platform.md),
  [ADR-021](decisions/ADR-021-terraform-managed-ecr.md),
  [ADR-022](decisions/ADR-022-eks-secure-api-access.md),
  [ADR-023](decisions/ADR-023-eks-access-control.md),
  [ADR-024](decisions/ADR-024-vpc-cni-pod-identity.md),
  [ADR-025](decisions/ADR-025-eks-secrets-kms-encryption.md),
  [ADR-026](decisions/ADR-026-in-cluster-mlflow-platform.md),
  [ADR-027](decisions/ADR-027-s3-dataset-runtime-retrieval.md)).
- **How the workload attaches.** A thin `k8s/overlays/aws` reuses the base unchanged
  and layers only the genuine cloud differences (ECR images, `imagePullPolicy:
  Always`, the S3 dataset source, the MLflow artifact bucket, and the gp2 StorageClass
  for the Postgres PVC); every security field is byte-identical to the local overlay
  ([ADR-018](decisions/ADR-018-aws-eks-deployment-overlay.md)).
- **Validated, not just written.** The IaC is gated in CI **statically and without
  AWS credentials** (`fmt`/`validate`/`test`/TFLint/Trivy — never `plan`/`apply`,
  [ADR-019](decisions/ADR-019-terraform-ci-validation.md)); `terraform test` runs an
  offline `mock_provider` contract suite that pins the ECR security/lifecycle
  properties **and the secure-by-default EKS API posture** (private default, no
  `0.0.0.0/0`, creator-admin off, explicit access entries, CNI/dataset/MLflow
  workload identity, KMS encryption, S3 dataset store). The real provisioning is an
  operator-driven, own-account step.
- **Proven, then destroyed.** The full hardened platform was provisioned in the
  operator's own account, the Job ran to completion on real EKS 1.35 (exit 0, all four
  DVC stages, all AWS access on EKS Pod Identity with no static keys, in-cluster MLflow
  logging to PostgreSQL + SSE-KMS S3, security controls verified live), and the
  environment was **destroyed and verified clean** — a short-lived
  `provision → prove → destroy` validation environment, not a production deployment
  ([Sprint 7 runtime evidence](proof/sprint-07-runtime-evidence.md),
  [Sprint 6 runtime evidence](proof/sprint-06-runtime-evidence.md),
  [Cloud Operations](cloud-operations.md),
  [ADR-020](decisions/ADR-020-cloud-lifecycle-cost-control.md)).

> **Honestly bounded.** This is a **validation environment**: single node, single
> NAT, two AZs, one region, local Terraform state. It is **not** production, **not**
> HA, **not** multi-region, has **no** GitOps, **no** Terraform remote state, **no**
> disaster-recovery, and **no** production observability — these remain deferred (see
> [roadmap](roadmap.md) v5–v6). See [Cloud Operations §7](cloud-operations.md#7-limitations)
> and the [Sprint 7 Proof-Impact](proof/sprint-07-proof-impact.md) for the full,
> explicit limits and the credible-claims boundary.

---

## Related Documentation

- [Project Structure](project-structure.md)
- [Roadmap](roadmap.md)
- [Engineering Philosophy](philosophy.md)
- [Architecture Decision Records](decisions/)
- [Logging Strategy](logging.md)
- [Exception Strategy](exception-strategy.md)
- [Type Safety](type-safety.md)
- [Testing Strategy](testing-strategy.md)
- [Developer Guide](developer-guide.md)
- [Containerization Strategy](containerization.md)
- [Docker Development Workflow](docker-development.md)
- [CI/CD](ci-cd.md)
- [Kubernetes Architecture](kubernetes-architecture.md) · [Kubernetes Operations](kubernetes-operations.md)
- [MLflow Platform](mlflow-platform.md) · [Dataset](dataset.md)
- [Cloud Operations (AWS/EKS runbook, cost, teardown)](cloud-operations.md) · [terraform/README.md](../terraform/README.md)
- [Sprint 7 — Proof Impact](proof/sprint-07-proof-impact.md) · [Sprint 7 — Runtime Evidence](proof/sprint-07-runtime-evidence.md)
- [Sprint 6 — Proof Impact](proof/sprint-06-proof-impact.md) · [Sprint 6 — Runtime Evidence](proof/sprint-06-runtime-evidence.md)
