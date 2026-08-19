# Sprint 7 — Release Gate (Cloud-Native MLOps Hardening)

- **Date:** 2026-08-19
- **Branch / PR:** `feature/sprint-07-release-gate` — *chore(release): prepare Sprint 7
  release evidence and gate*
- **Verdict:** **CONDITIONAL PASS** — no release blockers; release conditioned on a
  green CI run of the release commit (which executes the three linters this local gate
  could not run) and on accepting the captured 2026-08-19 runtime proof (see
  [§7](#7-final-verdict)).
- **Recommended release version:** **`v1.4.0`** (MINOR) — consolidates the
  `[Unreleased]` Sprint 5–7 work; the tag is **not** cut in this PR.
- **Related:** [Sprint 7 Proof-Impact](sprint-07-proof-impact.md),
  [Sprint 7 Runtime Evidence](sprint-07-runtime-evidence.md),
  [Sprint 7 Retrospective](../retrospectives/sprint-07-retrospective.md),
  [Cloud Operations](../cloud-operations.md),
  [CHANGELOG](../../CHANGELOG.md),
  ADR-021…ADR-027.

> **What this gate is.** A pre-release verification that the repository is *actually*
> ready for the Sprint 7 release: every check that can be run here was run and its raw
> result recorded; every Sprint 6 HIGH/MEDIUM finding was assessed individually against
> source **and** runtime evidence; and the two capabilities the sprint promised **not**
> to introduce (GitOps, Terraform remote state) were confirmed absent. Nothing is
> marked "passed" on source code alone where runtime evidence is required — those items
> are gated on the captured [runtime evidence](sprint-07-runtime-evidence.md).

> **Honesty boundary of this gate.** Two limits apply to *this execution* and are
> carried into the verdict, not hidden:
> 1. **No live cluster.** `aws eks list-clusters` (own account `419393719184`,
>    `us-east-1`) returns empty — the cost-controlled lifecycle (ADR-020) leaves no
>    standing EKS. Runtime claims are therefore verified against the **captured**
>    2026-08-19 evidence at commit `4f85974`, **not** re-executed live at this HEAD.
>    [§4](#4-runtime-proof-chain) records why that evidence still applies.
> 2. **Three tools not runnable here.** `tflint`, `trivy`, and `kubeconform` are not
>    executable in this environment (see [§2](#2-gate-execution-results)); their
>    coverage is delegated to CI and to the offline `terraform test` suite. This is the
>    reason the verdict is *conditional*, not a full PASS.

---

## 1. Executive summary

Sprint 7 closed **all seven** Sprint 6 HIGH/MEDIUM review findings (H-01…H-03,
M-01…M-04), each backed by Terraform/Kubernetes source, an ADR, an offline contract
test, and a redacted live-runtime record. Every static and test gate that can run in
this environment is **green**: `ruff`, `terraform fmt`/`validate`, `terraform test`
(**42/42** in a clean checkout), `kustomize build` (both overlays), `k8s/validate.py`
(**158/158**), and `pytest` (**152 passed, 1 skipped**). The full-platform pipeline run
(S3 → DVC → train → in-cluster MLflow → **Job exit 0**) is evidenced on real EKS 1.35
and torn down clean. Neither GitOps nor a remote Terraform state backend was
introduced.

No release blockers were found. The verdict is **CONDITIONAL PASS** only because three
IaC/manifest linters could not be executed in this gate environment and the runtime
proof is the captured (not re-run) evidence — both resolved by requiring a green CI run
on the release commit.

---

## 2. Gate execution results

### 2.1 Checks executed here (raw results)

| # | Check | Command | Result |
|---|---|---|---|
| 1 | Python lint | `ruff check .` | **PASS** — "All checks passed!" (exit 0) |
| 2 | Terraform format | `terraform fmt -check -recursive` | **PASS** — clean (exit 0) |
| 3 | Terraform init | `terraform init -backend=false` | **PASS** — initialized, no backend |
| 4 | Terraform validate | `terraform validate` | **PASS** — "Success! The configuration is valid." |
| 5 | Terraform contract tests | `terraform test` | **PASS — 42/42** in a clean checkout (see [§6](#6-the-terraform-test-observation-non-blocking)) |
| 6 | Kustomize render (local) | `kustomize build k8s/overlays/local` | **PASS** — renders (667 lines) |
| 7 | Kustomize render (aws) | `kustomize build k8s/overlays/aws` | **PASS** — renders (459 lines) |
| 8 | K8s security contract | `python k8s/validate.py` | **PASS — 158/158** (static; both overlays) |
| 9 | Python test suite | `pytest -q` | **PASS — 152 passed, 1 skipped** (32s) |

Notes:
- **(5)** In a clean checkout the suite is **42/42**. Run as-is on the operator's
  workstation it reports 41/1: the one failure is an artifact of the git-ignored local
  `terraform.tfvars` opt-in, not a code regression — fully diagnosed in
  [§6](#6-the-terraform-test-observation-non-blocking).
- **(9)** The single skip is `tests/smoke/test_smoke.py` — it self-skips because the
  optional `mlflow` runtime dependency is not installed locally; it is not a failure.
- **(8)** `k8s/validate.py` is explicit that it performs **static** checks only
  ("manifests are well-formed, hardened, and complete — STATIC checks only"). It proves
  the manifests *declare* the hardening, not kernel-level enforcement.

### 2.2 Checks NOT runnable in this environment (delegated to CI)

Recorded honestly — these are **gaps in this gate's execution**, not evidence of
passing:

| Tool | Purpose | State here | Where it is covered |
|---|---|---|---|
| **TFLint** | Terraform lint | **not installed** (`command -v tflint` → not found) | `terraform-validate` CI job (ADR-019); prior-sprint evidence |
| **Trivy** | IaC security scan (fails CRITICAL/HIGH) | **not installed** | `terraform-validate` CI job + `terraform/.trivyignore` triage record (ADR-019) |
| **kubeconform** | K8s schema validation | installed but **OS-blocked** (file read/exec "Access is denied") | `k8s-validate` CI job |

Because these three could not be executed here, **no HIGH/MEDIUM item is marked passed
on their basis**. Their offline equivalent — the `terraform test` contract suite and
`k8s/validate.py` — *was* run and is green, and `kustomize build` proves both overlays
render to schema-parseable YAML. Full linter coverage is a **release condition**: the
release commit's CI must be green.

---

## 3. Sprint 6 HIGH / MEDIUM findings — individual assessment

Each finding: **status**, **evidence** (source + offline contract + live runtime), and
**remaining limitation**. Runtime references are the captured 2026-08-19 evidence.

| ID | Finding | Status | Evidence | Remaining limitation |
|---|---|---|---|---|
| **H-01** | ECR created out-of-band (not IaC) | **CLOSED** | Source: [`terraform/ecr.tf`](../../terraform/ecr.tf) — 2 private repos, immutable tags, scan-on-push, retention, `force_delete`. Offline: `terraform test` `ecr_*` runs pass. Runtime: [evidence §3](sprint-07-runtime-evidence.md#3-ecr-image-verification) — 2 images pushed, immutable tags. ADR-021. | Registry KMS encryption uses AES256 (S3-managed), not a customer CMK — a scoped follow-up, out of H-01. |
| **H-02** | EKS API public / openable to world | **CLOSED** | Source: [`terraform/eks.tf`](../../terraform/eks.tf) — `endpoint_public_access` default **false**; `0.0.0.0/0`/any `/0` rejected by variable validation + `lifecycle` precondition. Offline: `terraform test` `eks_api_*` runs pass (rejection cases green; private-by-default green in clean checkout). Runtime: [evidence §2, §14](sprint-07-runtime-evidence.md#2-eks-verification) — endpoint private + scoped `/32`. ADR-022. | Operator reach from a workstation still needs the scoped **public** `/32` opt-in; a standing private-only posture requires in-VPC access (bastion/VPN/SSM), not provisioned. |
| **H-03** | Cluster access via implicit creator-admin | **CLOSED** | Source: [`terraform/eks.tf`](../../terraform/eks.tf) — `authentication_mode = API`, `bootstrap_cluster_creator_admin_permissions = false`. Offline: `terraform test` `eks_access_*` / `eks_rejects_*` runs pass. Runtime: [evidence §2](sprint-07-runtime-evidence.md#2-eks-verification) — access-entries-only, creator-admin off. ADR-023. | Operator access is granted by an explicit access entry recorded in git-ignored `terraform.tfvars`; the access model is not itself GitOps-managed. |
| **M-01** | VPC CNI uses node instance role | **CLOSED** | Source: [`terraform/iam.tf`](../../terraform/iam.tf) + `eks.tf` — dedicated CNI role via Pod Identity; `AmazonEKS_CNI_Policy` **not** on the node role. Offline: `terraform test` `cni_policy_is_not_on_the_node_role` + 5 CNI runs pass. Runtime: [evidence §4](sprint-07-runtime-evidence.md#4-workload-identity-eks-pod-identity) — 4 Pod Identity associations, no static keys. ADR-024. | Isolation is at the IAM-role boundary; no network-policy micro-segmentation of `aws-node` is claimed. |
| **M-02** | K8s Secrets not envelope-encrypted with a CMK | **CLOSED** | Source: [`terraform/kms.tf`](../../terraform/kms.tf) + `eks.tf` `encryption_config` (`resources = ["secrets"]`), rotation on. Offline: `terraform test` `cluster_encrypts_secrets_with_a_cmk` + 3 CMK runs pass. Runtime: [evidence §15](sprint-07-runtime-evidence.md#14-job-completion--15-pod-security-context) — EKS Secrets KMS-encrypted. ADR-025. | Envelope encryption covers `Secret` objects; application-layer secret hygiene (out-of-band DB password) is operational, not KMS-enforced. |
| **M-03** | Experiment tracking on external DagsHub SaaS | **CLOSED** | Source: [`k8s/base/mlflow/`](../../k8s/base/mlflow/) — stateless server + PostgreSQL StatefulSet + S3 artifacts, `--serve-artifacts`, ClusterIP-internal; [`terraform/s3.tf`](../../terraform/s3.tf). Runtime: [evidence §6, §9–13](sprint-07-runtime-evidence.md#6-in-cluster-mlflow-platform) — server Ready, **2 runs `FINISHED`**, metadata in PostgreSQL, 7 artifacts in SSE-KMS S3. ADR-026. | Single-writer PostgreSQL (no HA), no backup/restore drill; the DVC **data/model** remote in `.dvc/config` still uses DagsHub (separate versioning concern, not a runtime path). |
| **M-04** | Dataset delivered via ConfigMap | **CLOSED** | Source: [`terraform/datasets.tf`](../../terraform/datasets.tf) — private, all-public-blocked, versioned, SSE-KMS bucket; read-only dataset-reader role; [`k8s/overlays/aws/job-cloud.yaml`](../../k8s/overlays/aws/job-cloud.yaml) `fetch-dataset` init container. Offline: `terraform test` `dataset_*` runs pass. Runtime: [evidence §5–6](sprint-07-runtime-evidence.md#5-s3-dataset-runtime-retrieval-integrity-pinned) — sha256 == pinned identity, verified in-cluster. ADR-027. | Dataset is uploaded out-of-band (not a pipeline-managed ingestion); one region, one bucket. |

**Finding scorecard: 7 / 7 CLOSED** — 3 HIGH, 4 MEDIUM. No finding is source-only:
each pairs an offline contract (or manifest) with a live-runtime observation.

---

## 4. Runtime proof chain

The required chain, verified against the captured [Sprint 7 Runtime
Evidence](sprint-07-runtime-evidence.md) (real EKS 1.35, `us-east-1`, 2026-08-19,
commit `4f85974`; environment destroyed same session):

| Step | Observation | Evidence |
|---|---|---|
| S3 dataset | object retrieved in-cluster; sha256 == pinned `DATASET_SHA256` | §5–6 |
| ↓ EKS workload | Job scheduled on a **private-subnet** node; all AWS via Pod Identity | §2, §4 |
| ↓ DVC | `dvc repro` ran the declared DAG; declared == traced ([DVC correction evidence](sprint-07-dvc-dataflow-correction-evidence.md)) | §7–8 |
| ↓ preprocess | 768 rows written | §7–8 |
| ↓ split | 614 train / 154 held-out | §7–8 |
| ↓ train | GridSearchCV → accuracy **0.7398**; model registered | §7–9 |
| ↓ evaluate | held-out accuracy **0.7078** (disjoint test set) | §7–9 |
| ↓ in-cluster MLflow | 2 runs `FINISHED`; metadata in **PostgreSQL** | §9–12 |
| ↓ metrics/artifacts | 7 artifacts (incl. `model.skops`) in **SSE-KMS S3** via Pod Identity | §13 |
| ↓ Job exit 0 | Job `Complete`; successful pod `Succeeded`, **exit 0** | §14 |

**Does the captured evidence still apply to this HEAD?** Yes, with one caveat.
`git diff 4f85974..HEAD` touches **no** `terraform/`, `src/`, or `dvc.yaml` files. The
only platform-relevant changes since the proof run are: (a) the `gp2`
`storageClassName` patch in the AWS overlay — which was itself the committed fix that
*made* the proven run succeed ([evidence § Failures & fixes](sprint-07-runtime-evidence.md#failures--fixes)); and (b) additive PSA
`enforce: restricted` labels on the `mlops` Namespace plus expanded `k8s/validate.py`
checks (PR 11). See [§5](#5-non-blocking-findings) for the one caveat this creates.

---

## 5. Non-blocking findings

Real, but none blocks the release:

1. **PSA `enforce: restricted` not live-admission-tested.** The `restricted` Pod
   Security Admission labels added to the `mlops` Namespace in PR 11 postdate the
   live run, so no live API server was observed admitting the workloads *under
   `enforce`*. The pods statically satisfy `restricted` (`k8s/validate.py` 158/158)
   and the live pod ran with the exact security context, so the risk is low — but a
   fresh live run with PSA `enforce` on would upgrade this from static-plus-inference
   to directly observed.
2. **Three linters not run in this gate** (`tflint`, `trivy`, `kubeconform`) — see
   [§2.2](#22-checks-not-runnable-in-this-environment-delegated-to-ci). Covered by CI;
   made a release condition.
3. **`terraform test` is environment-sensitive on an operator workstation** — see
   [§6](#6-the-terraform-test-observation-non-blocking). Green in CI; a small
   test-hygiene note.
4. **`readOnlyRootFilesystem: false`** — deferred by design (DVC writes in-tree,
   ADR-010); unchanged from prior sprints.
5. **Runtime proof is captured, not re-run at HEAD** — inherent to the cost-controlled
   ephemeral lifecycle (ADR-020); mitigated by the no-platform-diff check in
   [§4](#4-runtime-proof-chain).

---

## 6. The `terraform test` observation (non-blocking)

Run verbatim on the operator's workstation, `terraform test` reports **41 passed, 1
failed** — `eks_api_is_private_by_default` (assertions at
`tests/eks_api_security.tftest.hcl:43` and `:53`). This is **not** a code regression:

- The source default is secure — `variable "cluster_endpoint_public_access" { default =
  false }` in `terraform/variables.tf`.
- `terraform test` **auto-loads** the git-ignored local `terraform.tfvars`, which the
  operator set to `cluster_endpoint_public_access = true` (scoped to their `/32`) during
  the live proof run to reach the API from their workstation
  ([evidence § Failures & fixes](sprint-07-runtime-evidence.md#failures--fixes)). That
  override flips the *effective* default the two assertions read.
- **Proof it is the tfvars, not the code:** moving `terraform.tfvars` aside and
  re-running yields **`Success! 42 passed, 0 failed.`** (the file was then restored,
  unchanged, still git-ignored). CI, which has no operator tfvars, sees 42/42.

The test's own header anticipates a tfvars *CIDR* being auto-loaded but did not
anticipate the *boolean* toggle also living there; hardening the two default-assertions
against a locally-present opt-in (e.g. asserting on an explicit `variables {}` block
rather than the ambient default) is a minor future tidy-up. It does not affect the
security posture, which the rejection tests (`0.0.0.0/0`, any `/0`, empty-list,
both-endpoints-off) all enforce and all pass.

---

## 7. Final verdict

### 7.1 — Verdict: **CONDITIONAL PASS**

The repository is ready for the Sprint 7 release. The verdict is *conditional* (not a
full PASS) on two release-time confirmations, each already substantially evidenced:

- **C1 — Green CI on the release commit.** The `terraform-validate` and `k8s-validate`
  jobs must pass, executing the three linters this local gate could not
  (`tflint`, `trivy`, `kubeconform`) plus `terraform test` (42/42 in the clean CI
  checkout).
- **C2 — Accept the captured runtime proof.** The runtime claims rest on the
  2026-08-19 evidence at `4f85974`; no `terraform/`/`src/`/`dvc.yaml` has changed since
  ([§4](#4-runtime-proof-chain)). Optionally, a fresh provision-prove-destroy run with
  PSA `enforce: restricted` would also clear non-blocking finding
  [§5.1](#5-non-blocking-findings).

### 7.2 — Release blockers

**None.** No broken check, no unclosed HIGH/MEDIUM finding, no accidental
architecture, no secret or state committed.

### 7.3 — Non-blocking findings

Five, listed in [§5](#5-non-blocking-findings): PSA-enforce not live-tested; three
linters not run locally (CI-covered); `terraform test` workstation-sensitivity;
`readOnlyRootFilesystem` deferred (ADR-010); runtime proof captured not re-run.

### 7.4 — Evidence for every HIGH/MEDIUM finding

In the [§3 table](#3-sprint-6-high--medium-findings--individual-assessment) — **7/7
CLOSED**, each with source + offline contract + live-runtime evidence and a stated
residual limitation.

### 7.5 — Final proof claims that ARE defensible

- All seven Sprint 6 HIGH/MEDIUM findings are **closed**, each evidenced in source, an
  offline contract test, and a redacted live-runtime record.
- The full hardened platform — Terraform-managed ECR, private-by-default EKS API with
  explicit access entries, EKS Pod Identity for the CNI and both app workloads (no
  static keys), KMS-encrypted Secrets, SSE-KMS S3 dataset retrieved and integrity-checked
  at runtime, and an in-cluster PostgreSQL+S3 MLflow platform — **ran the MLOps pipeline
  to completion on real EKS 1.35 (Job `Complete`, exit 0)** and was destroyed
  verified-clean.
- Offline, credential-free security contracts (`terraform test` **42/42**,
  `k8s/validate.py` **158/158**) and the application test suite (`pytest` **152
  passed**) are green.
- Neither **GitOps** nor a **Terraform remote-state backend** was introduced (verified:
  no Argo/Flux/Fleet controller anywhere; no `backend`/`cloud` stanza; no tracked
  `tfstate`/`tfvars`).

### 7.6 — Claims that must NOT be made

- **Not** production, **not** HA, **not** multi-region, **not** DR — single node, single
  NAT, single-writer PostgreSQL, one region, ephemeral.
- **No** GitOps / continuous delivery; **no** Terraform remote state; **no** production
  observability stack (Prometheus/Grafana/tracing/alerting).
- **No** model-serving / inference endpoint (roadmap v6).
- **Not** a re-run-at-HEAD live proof — runtime evidence is the captured 2026-08-19 run.
- **Not** Restricted-PSS *runtime-enforced-and-observed* — labels applied and statically
  satisfied, but not observed under live `enforce` admission.
- **No** customer-CMK encryption of the ECR registry (AES256 today).

### 7.7 — Recommended release version

**`v1.4.0`** — a MINOR bump over the last released tag `v1.3.1`, consolidating the
`[Unreleased]` Sprint 5–7 additions (all backward-compatible new capability, no
breaking change to the pipeline contract), per [docs/versioning.md](../versioning.md).
Per the task, **the git tag and GitHub release are NOT created in this PR** — cutting
`v1.4.0` is a separate, explicitly-requested step, gated on condition **C1** above.

---

## Related documentation

- [Sprint 7 Proof-Impact Assessment](sprint-07-proof-impact.md) ·
  [Sprint 7 Runtime Evidence](sprint-07-runtime-evidence.md) ·
  [Sprint 7 Retrospective](../retrospectives/sprint-07-retrospective.md)
- [Cloud Operations](../cloud-operations.md) · [MLflow Platform](../mlflow-platform.md) ·
  [Dataset](../dataset.md) · [Versioning](../versioning.md)
- ADR-021 · ADR-022 · ADR-023 · ADR-024 · ADR-025 · ADR-026 · ADR-027
