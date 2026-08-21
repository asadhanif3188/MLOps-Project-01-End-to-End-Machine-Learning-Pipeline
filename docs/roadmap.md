# Roadmap

The roadmap organizes the project's evolution into versioned milestones. Each
version states its **objectives** and **expected outcome** rather than
prescribing implementation details. Concrete technical decisions are captured as
[Architecture Decision Records](decisions/) when they are made.

> **Legend:** ✅ Done · 🚧 In progress · ⬜ Planned

| Version | Theme | Status |
|---------|-------|--------|
| [v1](#version-1--foundation-release) | Foundation Release | ✅ |
| [v2](#version-2--engineering-improvements) | Engineering Improvements | 🚧 |
| [v3](#version-3--cicd) | CI/CD | 🚧 |
| [v4](#version-4--kubernetes) | Kubernetes | 🚧 |
| [v5](#version-5--production-cloud-platform) | Production Cloud Platform | 🚧 |
| [v6](#version-6--enterprise-mlops) | Enterprise MLOps | ⬜ |

---

## Version 1 — Foundation Release

**Objective:** Establish a working, reproducible ML pipeline as the baseline for
future engineering enhancements.

> The initial implementation is based on a guided educational project. It serves
> as the honest baseline the rest of this roadmap builds on — not as the
> project's long-term identity.

**Scope delivered:**

- Three-stage DVC pipeline: preprocess → train → evaluate.
- Random Forest classifier with `GridSearchCV` hyperparameter tuning.
- MLflow experiment tracking hosted on DagsHub.
- DVC data/model versioning with an S3-compatible DagsHub remote.

**Expected outcome:** A runnable pipeline that reproduces training results and
tracks experiments — the foundation everything else builds on.

---

## Version 2 — Engineering Improvements

**Objective:** Elevate the repository from a baseline implementation to a
professional, maintainable codebase.

**Delivered:**

- ✅ Professional documentation under `docs/`: architecture, roadmap, project
  structure, ADRs, engineering philosophy, and design principles.
- ✅ Repository hygiene and governance: LICENSE, CONTRIBUTING, CODE_OF_CONDUCT,
  CHANGELOG, `.editorconfig`, security and support policies, issue/PR templates.
- ✅ Principal-engineer production-readiness review
  ([reviews/sprint-02-engineering-review.md](reviews/sprint-02-engineering-review.md)),
  whose findings drove the rest of the sprint.
- ✅ Code organization and readability refactor of the pipeline stages.
- ✅ Centralized logging in place of `print`
  ([Logging Strategy](logging.md), review finding H-1).
- ✅ Standardized exception handling: typed hierarchy, wrapped IO/config
  boundaries, uniform stage entry point
  ([Exception Strategy](exception-strategy.md), review finding H-2).
- ✅ Complete type annotations with a strict mypy configuration
  ([Type Safety](type-safety.md)).
- ✅ Testing foundation: pytest smoke and unit suites with shared fixtures
  ([Testing Strategy](testing-strategy.md), review finding H-3).
- ✅ Developer experience: Ruff linting/formatting, pre-commit hooks, Makefile,
  VS Code workspace settings
  ([Developer Guide](developer-guide.md),
  [ADR-004](decisions/ADR-004-python-quality-toolchain.md)).

**Delivered in Sprint 4 (`v1.3.0` — Pipeline Correctness & Reproducibility):**

- ✅ Correctness fixes surfaced during documentation: reconciled
  `dvc.yaml`/`params.yaml` parameter names and fed the `preprocess` output into
  `train` (and evaluation source). Now enforced by the `contract` tests and CI
  (see the [Pipeline Contract](pipeline-contract.md)).
- ✅ Decoupled the stage bodies from MLflow/network via the `tracking` boundary,
  so `train` and `evaluate` logic is now unit-tested
  ([Testing Strategy](testing-strategy.md)).
- ✅ Corrected the stale root `README.md` claims (training input, hyperparameter
  tuning, evaluation output, DVC-stage snippets) to match the as-built pipeline.

**Still remaining:**

- ✅ Evaluate on a genuine **held-out split** — done. A dedicated `split` stage now
  partitions the processed dataset so `train` and `evaluate` consume disjoint data
  and the reported accuracy is out-of-sample (deviation **D5** resolved;
  [ADR-007](decisions/ADR-007-held-out-evaluation.md),
  [pipeline contract §8](pipeline-contract.md#8-evaluation-boundary)).
- ✅ Commit a `dvc.lock` and run `dvc repro` in CI against a fixture dataset —
  done. A self-contained fixture pipeline reproduces the same four stages and the
  same `src/` code offline (no remote, no MLflow, no credentials); CI runs a real
  `dvc repro`, asserts the committed lock is up to date, and requires byte-identical
  outputs on a forced re-run. This upgrades reproducibility from *logical* to
  *drift-gated* and closes the `dvc.lock`/execution portion of deviation **D7**
  ([ADR-008](decisions/ADR-008-fixture-reproducibility.md),
  [pipeline contract §7](pipeline-contract.md#7-reproducibility-expectations)).
  End-to-end reproduction of the *production* run (remote data + live MLflow +
  digest-pinned deps) remains a documented limitation, not a gap.

**Expected outcome:** A repository that reads as an actively maintained,
professionally engineered project and is safe to change with confidence.

---

## Version 3 — CI/CD

**Objective:** Automate quality gates and pipeline reproduction.

**Objectives:**

- ✅ Continuous integration: run linting, strict type checking (mypy), and tests
  on every pull request (GitHub Actions — see [CI/CD](ci-cd.md) and
  [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)).
- ✅ Build and validate the container image in CI (build only — **not** pushed).
- ✅ Automated pipeline validation in CI — **definition** validated offline in
  Sprint 4 (`dvc dag` + local `dvc status` + the `contract` tests), and a real
  **execution** check now runs a scoped `dvc repro` against a committed fixture
  dataset + `dvc.lock` (proof-hardening milestone;
  [ADR-008](decisions/ADR-008-fixture-reproducibility.md)). The production raw
  dataset stays remote-only, so the production run itself is still not executed in
  CI — a documented limitation, not a gap.
- ✅ Basic security/supply-chain scanning — image vulnerability scan (PR 8) + CycloneDX
  SBOM and git→tag→digest provenance (PR 9) delivered; cosign signing is opt-in, an
  enforced signing gate remains deferred.
- 🚧 Publish the container image on release
  ([Containerization Strategy](containerization.md), [ADR-005](decisions/ADR-005-containerization-strategy.md)).
- 🚧 Branch protection requiring green checks before merge.

**Expected outcome:** Every change is automatically validated before merge,
reducing regressions and manual effort.

> **Status:** CI (checkout → setup Python → install → Ruff → mypy → pytest → DVC
> pipeline integrity → fixture `dvc repro` → Docker build → build validation →
> **Kubernetes manifest validation**) is implemented on **GitHub Actions**. It
> validates only — no deploy, no image push, and its Kubernetes checks are
> **static** (schema/Kustomize/security) plus one opt-in cluster admission dry-run;
> the workload is never run on a cluster (Sprint 5, PR 6 —
> [ADR-012](decisions/ADR-012-kubernetes-manifest-validation.md)). Offline
> pipeline-definition validation landed in Sprint 4, and a real fixture-pipeline
> `dvc repro` execution check landed in the proof-hardening milestone
> ([ADR-008](decisions/ADR-008-fixture-reproducibility.md)); the
> remaining items above (publish, scan, branch protection) are the path to
> continuous *delivery*; see
> [CI/CD § Future CD roadmap](ci-cd.md#future-cd-roadmap).
>
> **TODO:** Ratify the CD approach (registry, signing, deploy target) as an ADR.

---

## Version 4 — Kubernetes

**Objective:** Make the pipeline portable and horizontally runnable.

**Objectives:**

- ✅ Containerize the pipeline for consistent execution environments —
  **implemented** in Sprint 3: multi-stage [`Dockerfile`](../Dockerfile),
  [`.dockerignore`](../.dockerignore), and a [`docker-compose.yml`](../docker-compose.yml)
  local dev workflow ([Containerization Strategy](containerization.md),
  [Docker Development](docker-development.md),
  [ADR-005](decisions/ADR-005-containerization-strategy.md)).
- 🚧 Run pipeline stages as an orchestrated Kubernetes workload — **runnable
  workload landed** (Sprint 5, PR 1–2): an `mlops` namespace and the pipeline
  modelled as a **runnable** `batch/v1` **Job** (not a Deployment) — the real
  `ml-pipeline:local` image, the real `dvc repro` command, and a finite-run
  lifecycle (`restartPolicy: Never`, `backoffLimit: 2`,
  `activeDeadlineSeconds: 1800`) — structured with Kustomize under
  [`k8s/`](../k8s/), with a local run runbook
  ([Kubernetes Architecture](kubernetes-architecture.md),
  [ADR-009](decisions/ADR-009-kubernetes-workload-model.md)). Executed on a local
  Docker Desktop cluster (2026-08-12): the Job lifecycle is verified end to end.
  The pipeline is now **green in-cluster** (Sprint 5, PR 8): `dvc repro` runs the
  full pipeline to **exit 0** via the runtime contract in
  [ADR-013](decisions/ADR-013-kubernetes-runtime-execution.md) (DVC no-SCM, mounted
  dataset, in-pod MLflow file store), on a local cluster. *(Sprint 7 superseded the
  Sprint 5 mechanism: the mounted-dataset ConfigMap became S3-via-Pod-Identity
  retrieval ([ADR-027](decisions/ADR-027-s3-dataset-runtime-retrieval.md)) and the
  in-pod file store became the in-cluster MLflow platform
  ([ADR-026](decisions/ADR-026-in-cluster-mlflow-platform.md)); see v5.)*
- ✅ Externalize configuration and secrets for a cluster — **implemented** (Sprint
  5, PR 3): a `ConfigMap` for non-secret runtime config (`LOG_LEVEL`,
  `MLFLOW_TRACKING_URI`), a **Secret template** for the MLflow/DagsHub credentials
  (`MLFLOW_TRACKING_USERNAME`/`_PASSWORD`, created out-of-band — never committed),
  and a least-privilege `ServiceAccount` with `automountServiceAccountToken: false`
  (the workload needs no Kubernetes API access, so no RBAC is granted). Wired into
  the Job via `envFrom` and verified on a local cluster.
- ✅ Harden the workload with a Kubernetes `securityContext` — **implemented**
  (Sprint 5, PR 4): non-root with an explicit uid/gid `10001` (required — the
  image's `USER` is a name), `allowPrivilegeEscalation: false`, all Linux
  capabilities dropped, and seccomp `RuntimeDefault`, applied at the correct
  pod/container scopes and verified enforced on a local cluster. Read-only root
  filesystem is **evaluated and deliberately deferred** (DVC writes state in-tree
  at the repo root); restricted Pod Security Standard compliance is **not** claimed
  ([ADR-010](decisions/ADR-010-kubernetes-security-hardening.md)).
- ✅ Define resource requests/limits for reproducible scheduling — **implemented**
  (Sprint 5, PR 5): `requests: cpu 250m / mem 256Mi`, `limits: cpu 1 / mem 512Mi`
  (Burstable QoS), chosen from *measured* usage of the real image — the CPU limit
  doubles as the memory-safety control because `GridSearchCV(n_jobs=-1)` sizes
  joblib's worker fan-out from the cgroup CPU quota. The finite-run lifecycle,
  the deliberate absence of health probes, and the failure modes are documented;
  values are **not** production-certified
  ([ADR-011](decisions/ADR-011-kubernetes-resource-lifecycle.md)).
- ✅ Validate the manifests automatically in CI — **implemented** (Sprint 5, PR 6):
  a static, deterministic `k8s-validate` job (pinned `kustomize` render +
  `kubeconform -strict` schema + a project `k8s/validate.py` security/required-field
  check), plus an opt-in ephemeral-kind **server-side dry-run** admission job.
  Static validation only — it does **not** deploy or run the workload
  ([ADR-012](decisions/ADR-012-kubernetes-manifest-validation.md)).
- ✅ Document the operations & prove the platform work — **implemented** (Sprint 5,
  PR 7): a complete deployment guide ([`k8s/README.md`](../k8s/README.md)), a
  [Kubernetes Operations runbook](kubernetes-operations.md) with a troubleshooting
  matrix, a [Kubernetes Security document](kubernetes-security.md), and an
  evidence-based [Sprint 5 Proof-Impact Assessment](proof/sprint-05-proof-impact.md),
  with the local deployment path re-executed from a clean state as proof. Local
  cluster only; **no** production/GitOps/HA/serving/observability claims.

**Expected outcome:** The pipeline runs reproducibly on any conformant cluster,
independent of a developer's local machine.

> **Status.** The workload model is ratified in
> [ADR-009](decisions/ADR-009-kubernetes-workload-model.md) and the `k8s/`
> manifests define a **runnable** Job (namespace + Job + Kustomize + local
> runbook), validated by offline rendering and field assertions **and by an
> executed run on a local Docker Desktop cluster** (2026-08-12) that verified the
> Job lifecycle end to end. PR 3 adds externalized **config/secrets + identity** (a
> `ConfigMap`, an out-of-band `Secret` template, and a least-privilege
> `ServiceAccount` with token automount off), and PR 4 added an **enforced
> `securityContext`** (non-root uid/gid `10001`, no privilege escalation, all
> capabilities dropped, seccomp `RuntimeDefault`; read-only root filesystem
> deferred with evidence — [ADR-010](decisions/ADR-010-kubernetes-security-hardening.md)),
> and PR 5 added **resource requests/limits chosen from measured usage** plus the
> documented lifecycle/probe/failure-mode decisions
> ([ADR-011](decisions/ADR-011-kubernetes-resource-lifecycle.md)) — all verified on
> the local cluster, and PR 6 added **automated CI manifest validation** (static
> syntax/schema/Kustomize/security checks + an opt-in cluster admission dry-run —
> [ADR-012](decisions/ADR-012-kubernetes-manifest-validation.md)), and PR 7 added the
> **operations & proof** documentation ([operations runbook](kubernetes-operations.md),
> [security document](kubernetes-security.md),
> [Sprint 5 Proof-Impact](proof/sprint-05-proof-impact.md)) with the local deployment
> path re-executed as evidence. **PR 8 then closed the runtime gap**: a **green**
> in-cluster `dvc repro` — the complete pipeline (preprocess → split → train →
> evaluate) runs to **exit 0** as a secured Job, via a minimal runtime contract
> (DVC no-SCM, a mounted dataset, an in-pod MLflow file store —
> [ADR-013](decisions/ADR-013-kubernetes-runtime-execution.md)), with a controlled
> failure test verifying retry/terminal-failure behaviour. Bounded to a **local**
> cluster (local-validation dataset, file-store MLflow); production storage/MLflow
> and read-only-root remain future work.
> The container **base image** is ratified in
> [ADR-005](decisions/ADR-005-containerization-strategy.md).

---

## Version 5 — Production Cloud Platform

**Objective:** Provision and operate the pipeline on managed cloud
infrastructure defined as code.

**Objectives:**

- ✅ **Infrastructure as Code** with Terraform (versioned, reviewable) — delivered
  in **Sprint 6** as a structured Terraform root module (VPC, IAM, EKS) with pinned
  providers and a committed lock, gated statically in CI
  ([`terraform/`](../terraform/), [ADR-014](decisions/ADR-014-terraform-architecture.md),
  [ADR-019](decisions/ADR-019-terraform-ci-validation.md)). Scoped to a **validation**
  environment, not yet a production module structure.
- ✅ **AWS** as the target cloud provider — VPC + IAM + managed **EKS** provisioned in
  Sprint 6 ([ADR-015](decisions/ADR-015-aws-network-architecture.md)…[ADR-017](decisions/ADR-017-eks-platform.md)).
- ⬜ **Remote state** management for Terraform (e.g., S3 backend with locking) —
  deliberately deferred; Sprint 6 uses **local state** for a single-operator,
  short-lived environment ([ADR-014](decisions/ADR-014-terraform-architecture.md)).
- ✅ **IAM** roles and least-privilege access — two dedicated, single-trust EKS roles
  with AWS-managed policies only, no `AdministratorAccess`
  ([ADR-016](decisions/ADR-016-aws-iam-foundation.md)).
- 🚧 **CI/CD** integration for infrastructure — CI **validates** the IaC statically
  with **no AWS credentials** and never runs `plan`/`apply`
  ([ADR-019](decisions/ADR-019-terraform-ci-validation.md)); credentialed
  plan/apply (e.g. via OIDC) is future work. Provisioning today is a deliberate,
  operator-driven, own-account step.
- 🚧 **Monitoring** and centralized logging for pipeline and infrastructure health —
  the observability **architecture is defined** (Sprint 8, PR 1): a self-hosted
  Prometheus + Grafana stack over a four-layer model (Kubernetes platform, the
  batch pipeline Job, MLflow, PostgreSQL), with the ephemeral-Job metric problem
  solved via kube-state-metrics and SLO-style operational objectives
  ([Observability & Operations](observability.md),
  [ADR-028](decisions/ADR-028-observability-architecture.md)). The **metrics
  foundation is now built** (Sprint 8, PR 2) — version-controlled, hardened,
  statically-validated manifests for **Prometheus + kube-state-metrics +
  node-exporter + a cAdvisor scrape** (Layer 1 + the Layer 2 batch-Job signals).
  Minimal, hand-written Kustomize, ephemeral TSDB, read-only RBAC, and a single
  documented node-exporter Pod Security exception
  ([`k8s/monitoring/`](../k8s/monitoring/),
  [Monitoring Operations](monitoring-operations.md),
  [ADR-029](decisions/ADR-029-monitoring-foundation.md)). The **pipeline's own
  operational metrics are now instrumented** (Sprint 8, PR 3): per-stage duration +
  success/failure pushed to a scoped, hardened **Pushgateway** (the per-stage
  granularity KSM cannot give), with bounded label cardinality, a per-run reset to
  avoid stale series, best-effort emission, and a strict operational-vs-MLflow
  ownership boundary — reversing ADR-028's "Pushgateway deferred" for this scoped use
  ([`src/pipeline_metrics.py`](../src/pipeline_metrics.py),
  [ADR-030](decisions/ADR-030-pipeline-operational-metrics.md)). **MLflow &
  PostgreSQL platform depth is now added** (Sprint 8, PR 4): a **blackbox-exporter**
  probing MLflow's stable `/health` (Layer 3 availability — MLflow has no native
  `/metrics`), a **postgres-exporter** reporting `pg_up`/connections/size via a
  dedicated read-only `pg_monitor` role (credentials confined to the `mlops`
  namespace), and a scoped **kubelet** volume-stats scrape for the Postgres
  **PVC-fill** signal — eight scrape jobs in all; the run-level replica/readiness/
  restart/CPU/memory signals were already collectable from PR 2's KSM + cAdvisor
  ([ADR-031](decisions/ADR-031-mlflow-postgres-monitoring.md)). **Grafana dashboards
  are now added** (Sprint 8, PR 5): three purpose-built, version-controlled dashboards
  — **EKS/Platform Health**, **MLOps Pipeline Operations**, **MLflow Platform Health** —
  each panel mapped to an operational question, provisioned from files into a hardened,
  internal-only Grafana, with model quality kept in MLflow
  ([ADR-032](decisions/ADR-032-grafana-dashboards.md)). **Alerting is now added**
  (Sprint 8, PR 6): eight high-signal Prometheus alert rules encoding the § 6
  objectives — pipeline failure (batch-correct: the terminal Failed condition, never
  "not Running"), OOMKill, MLflow/Postgres unavailable, PVC-fill, memory headroom, and
  crash-looping — each with a severity, human summary/description, documented
  threshold and a runbook, unit-tested with `promtool` in CI
  ([ADR-033](decisions/ADR-033-alerting.md)). **Least-privilege NetworkPolicies are
  now added** (Sprint 8, PR 7): an evidence-mapped communication matrix drives a
  **default-deny + explicit-allow** set across both namespaces — PostgreSQL is
  reachable by exactly two peers with zero egress, the pipeline provably cannot reach
  the DB directly, and DNS, Pod Identity, and the full scrape graph are preserved and
  asserted in CI (validate.py §8/M12) alongside a runtime harness with an enforcement
  canary; NetworkPolicy enforcement is switched on for EKS via the VPC CNI
  `enableNetworkPolicy` flag. The AWS **S3-egress limitation** is documented rather
  than faked: S3's dynamic public IPs cannot be pinned in a standard NetworkPolicy, so
  egress is bounded to internet-only:443 and the "which bucket/actions" precision is
  delegated to IAM (Pod Identity) + a recommended VPC S3 endpoint; **no service mesh**
  ([ADR-034](decisions/ADR-034-network-policies.md)). **Manifests + instrumentation +
  dashboards + alert rules + network policies only — not deployed/runtime-proven yet**
  (no live cluster; live firing, the failure-injection campaign, and the live
  allowed/denied-path capture are the runtime-evidence work — tracked as checklists to
  run in ONE batched next-cluster session:
  [network-policy](proof/sprint-08-network-policy-runtime-evidence.md) (PR 7),
  [SBOM/provenance digest](proof/sprint-08-sbom-provenance-evidence.md) (PR 9), and the
  [dataset availability + integrity failure paths](proof/sprint-08-dataset-failure-tests-evidence.md)
  (PR 10 — unavailable object and checksum mismatch both fail fast, before training,
  driven by [`k8s/tests/dataset-failure/run.sh`](../k8s/tests/dataset-failure/run.sh)));
  Alertmanager notifier routing remains deferred.
  Centralized **log aggregation** and **tracing** are deliberately deferred;
  today's diagnosis is `kubectl` + structured logs.
- ✅ **Container-image vulnerability scanning** (Sprint 8, PR 8): the `docker` CI job
  now scans **both** shipped images — the `mlops-pipeline` runtime image and the
  `mlflow-server` image layered on it — with **Trivy** over their OS + Python packages,
  on the locally-built images (never pulled from a registry, so PR CI stays
  credential-free/AWS-independent). The gate **fails on *fixable* HIGH/CRITICAL**
  (`--ignore-unfixed`, actionable via a base/dependency bump) while **reporting** the
  non-fixable ones (surfaced, not muted; auto-promoted to the gate when a fix ships) —
  **not** a blanket ignore of HIGH/CRITICAL. Justified, time-boxed exceptions (CVE id +
  rationale + `expired_at`, auto-expired) live in
  [`.trivyignore.yaml`](../.trivyignore.yaml); a table + JSON report is published as a
  build artifact. It **complements** ECR `scan_on_push`
  ([ADR-021](decisions/ADR-021-terraform-managed-ecr.md)), the registry-side layer.
  This delivers the **scanning** slice of the v3 supply-chain item
  ([ADR-035](decisions/ADR-035-container-image-scanning.md),
  [`docs/container-image-scanning.md`](container-image-scanning.md)).
- ✅ **SBOM + immutable image provenance** (Sprint 8, PR 9): the `docker` CI job now
  emits a **CycloneDX SBOM** (Trivy) for both images and **asserts the git→image
  binding** (the image's `org.opencontainers.image.revision` label must equal the commit
  SHA); the SBOM + a provenance record ship as the `sbom-and-provenance` artifact (never
  committed). The operator release ([`scripts/release-image.sh`](../scripts/release-image.sh))
  captures the immutable ECR **sha256 digest** (cross-checked against `aws ecr
  describe-images`) and records the full **git commit → image tag → digest** chain; the
  deploy can be **pinned by digest** (opt-in in the renderer) and
  [`scripts/verify-deployed-digest.sh`](../scripts/verify-deployed-digest.sh) confirms the
  **running** workload uses it. **Image signing** (cosign) ships as an **opt-in** keyless
  step; an *enforced* signing gate stays deferred
  ([ADR-036](decisions/ADR-036-sbom-and-image-provenance.md),
  [`docs/supply-chain-provenance.md`](supply-chain-provenance.md)).
- ⬜ **Migrate the DVC data remote off DagsHub** — Sprint 7 removed DagsHub from the
  **experiment-tracking** path (tracking now runs on the in-cluster MLflow platform,
  [ADR-026](decisions/ADR-026-in-cluster-mlflow-platform.md)), but the DVC **data/model
  remote** in [`.dvc/config`](../.dvc/config) still uses DagsHub's S3-compatible
  endpoint — a separate *versioning* concern. Point it at the project's own S3
  (the same Terraform-provisioned bucket family the MLflow artifact store uses),
  authorized by EKS Pod Identity rather than committed keys, so the whole platform
  is self-hosted with no external SaaS in any runtime path. Deliberately deferred:
  in-cluster runs already avoid the remote (`core.no_scm` + the S3-retrieved runtime
  dataset, below), so this is not blocking. Ratify as an ADR.
- ✅ **Runtime dataset from S3, not a ConfigMap** (Sprint 7, PR 8 — closes finding
  **M-04**). The dataset is delivered at runtime from a private, CMK-encrypted,
  versioned S3 bucket ([`terraform/datasets.tf`](../terraform/datasets.tf)) by a
  `fetch-dataset` init container, authorized by EKS Pod Identity (least-privilege
  read-only, no static keys) and integrity-checked against a pinned checksum — no
  ConfigMap, no baked-in data, no hostPath
  ([ADR-027](decisions/ADR-027-s3-dataset-runtime-retrieval.md)). Proven live on real
  EKS ([Sprint 7 runtime evidence](proof/sprint-07-runtime-evidence.md)).
- ✅ **Cloud platform security hardening** (Sprint 7, PRs 1–5) — the Sprint 6 review
  findings are closed: **Terraform-managed ECR** (H-01,
  [ADR-021](decisions/ADR-021-terraform-managed-ecr.md)), **secure-by-default EKS API**
  (private, never `0.0.0.0/0`; H-02, [ADR-022](decisions/ADR-022-eks-secure-api-access.md)),
  **explicit EKS access entries** with no creator-admin (H-03,
  [ADR-023](decisions/ADR-023-eks-access-control.md)), **VPC CNI via EKS Pod Identity**
  off the node role (M-01, [ADR-024](decisions/ADR-024-vpc-cni-pod-identity.md)), and
  **KMS-encrypted Kubernetes Secrets** (M-02,
  [ADR-025](decisions/ADR-025-eks-secrets-kms-encryption.md)).
- ✅ **In-cluster MLflow tracking platform** (Sprint 7, PR 6) — a self-hosted MLflow
  server + PostgreSQL metadata backend + S3 artifact store, replacing the external
  DagsHub SaaS in the experiment-tracking path
  ([ADR-026](decisions/ADR-026-in-cluster-mlflow-platform.md),
  [MLflow Platform](mlflow-platform.md)). All AWS access is via EKS Pod Identity (no
  static keys).

**Expected outcome:** Production-deployable infrastructure, provisioned
reproducibly from code with clear separation of environments.

> **Status (Sprint 6 → Sprint 7).** The IaC foundation and a **real, evidenced EKS
> run** landed in Sprint 6; **Sprint 7 hardened the platform and made it fully
> cloud-native**. The full platform — Terraform-managed ECR, secure-by-default EKS API
> with explicit access entries, VPC CNI + dataset + MLflow workload identity via EKS
> Pod Identity, KMS-encrypted Secrets and S3 stores, and the in-cluster PostgreSQL+S3
> MLflow tracking server — was provisioned onto managed EKS, ran the pipeline to
> completion (exit 0), had its Sprint 5 security controls verified on the live pod, and
> the environment was **destroyed and verified clean**
> ([Sprint 7 Proof-Impact](proof/sprint-07-proof-impact.md),
> [Sprint 7 runtime evidence](proof/sprint-07-runtime-evidence.md),
> [Cloud Operations](cloud-operations.md)). This remains a **short-lived,
> single-operator validation environment** — **not** production. Deliberately
> **deferred**: GitOps, **Terraform remote state**, credentialed CI/CD apply,
> multi-region, enterprise HA/DR, and production observability. Those are the path from
> "validated on cloud" to "production cloud platform" and are why v5 is 🚧, not ✅.
>
> **TODO:** Ratify remote-state, credentialed CI/CD (OIDC), a production module
> structure, and the serving mechanism as ADRs before a production deployment.
> *(Monitoring is now ratified — [ADR-028](decisions/ADR-028-observability-architecture.md),
> Sprint 8; implementation PRs 2–6 pending.)*

---

## Version 6 — Enterprise MLOps

**Objective:** Add the operational maturity expected of production ML systems.

**Objectives:**

- **Model serving** via a versioned inference endpoint with rollback.
- **Monitoring and alerting** for data quality, latency, and model performance —
  building on the Sprint 8 observability foundation
  ([ADR-028](decisions/ADR-028-observability-architecture.md),
  [Observability & Operations](observability.md)), which ratifies the
  Prometheus/Grafana stack and the four-layer platform-health model this extends.
- **Drift detection** (data and concept drift) with automated retraining
  triggers.
- **Governance:** model lineage, approval gates, and auditability.
- **Feature and artifact management** across environments.

**Expected outcome:** A production-grade MLOps platform demonstrating engineering
judgment well beyond model building.

> **TODO:** Ratify drift metrics and retraining triggers as ADRs. *(The monitoring
> stack itself is ratified in [ADR-028](decisions/ADR-028-observability-architecture.md),
> Sprint 8.)*

---

## Related Documentation

- [Architecture](architecture.md)
- [Design Principles](design-principles.md)
- [Engineering Philosophy](philosophy.md)
- [Architecture Decision Records](decisions/)
