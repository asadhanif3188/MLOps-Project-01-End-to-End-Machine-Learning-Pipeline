# Evidence Index

A single reviewer-first map of what this repository **claims** and where each
claim can be **verified** — without reading the sprint history.

Every headline claim in the [repository README](../../README.md) traces to a
canonical document below. This index does **not** duplicate those documents; it
points to them and states how strong each proof is.

> **Scope.** This is a **portfolio-scoped platform-engineering proof**. Runtime
> evidence comes from **controlled, short-lived validation sessions** on real
> Amazon EKS — provisioned, proven, and destroyed the same session. It is
> deliberately **not** a claim of 24/7 production operation, formal SLA/SLO, or
> enterprise SRE maturity (see [§11 Known Limitations](#11--known-limitations)).

---

## Start here

If you only read three documents, read these — they are the authoritative,
integrated records:

| # | Document | What it is | Strength |
|---|----------|------------|----------|
| 1 | [Sprint 8 Release Gate — v1.7.0](sprint-08-release-gate.md) | The final release-readiness audit: 23/23 proof dimensions, runtime evidence §4, runbook validation matrix §5, and the explicit "claims safe / claims not safe" boundary §11. **Verdict: PASS.** | Live EKS validated |
| 2 | [PR 16 Release-Candidate Validation on Live EKS](sprint-08-pr16-release-validation-evidence.md) | The authoritative single `provision → prove → destroy` run of the complete merged system on real EKS (v1.35.6, 2× t3.large). Cost/resource observations §17, symmetric teardown. | Live EKS validated |
| 3 | [Batched Live-EKS Evidence](sprint-08-live-eks-evidence.md) | The earlier exploratory campaign that closed each Sprint 8 runtime item in one cluster session, and honestly records the **4 real defects the live run surfaced** (§3). | Live EKS validated |

**How to read proof strength** (used throughout this index):

| Label | Meaning |
|-------|---------|
| **Live EKS validated** | Observed on a real, Terraform-provisioned Amazon EKS cluster in the operator's own AWS account, then destroyed. |
| **Runtime validated** | Observed at runtime on a Kubernetes cluster (local Docker Desktop / kind) — real execution, not cloud. |
| **Static/CI validated** | Enforced by CI, static analysis, or build-time gates — not a live cluster observation. |
| **Historical implementation evidence** | A dated snapshot from an earlier sprint; the mechanism it describes was later integrated and re-proven. Preserved, not current. |
| **Explicitly deferred** | Deliberately out of scope; documented, not hidden. |

---

## Public proof matrix

The compact view: what was engineered, what proves it, and how strong that proof
is. Detail lives in the sectioned index below.

| Capability | What was engineered | What proves it | Proof strength |
|------------|--------------------|----------------|----------------|
| **Cloud infrastructure** | Terraform-provisioned EKS, VPC, IAM, KMS, ECR, S3 (65 resources) | [PR 16 §Environment](sprint-08-pr16-release-validation-evidence.md) · [Release gate §4.A](sprint-08-release-gate.md) | Live EKS validated |
| **ML pipeline on EKS** | DVC pipeline run to completion as a `batch/v1` Job (exit 0, 5/5 stages) | [PR 16 evidence](sprint-08-pr16-release-validation-evidence.md) · [Sprint 7 runtime](sprint-07-runtime-evidence.md) | Live EKS validated |
| **Experiment tracking** | In-cluster MLflow on PostgreSQL + SSE-KMS S3 artifacts; runs persisted | [MLflow integration](sprint-07-mlflow-integration-evidence.md) · [MLflow failure tests](sprint-08-mlflow-failure-tests-evidence.md) | Live EKS validated |
| **Dataset retrieval** | S3 runtime fetch by init container, checksum-verified before training | [S3 dataset evidence](sprint-07-s3-dataset-runtime-evidence.md) · [Dataset failure tests](sprint-08-dataset-failure-tests-evidence.md) | Live EKS validated |
| **Observability** | Prometheus (11 targets UP), 3 Grafana dashboards, pipeline metrics | [Release gate §4](sprint-08-release-gate.md) · [screenshots](../screenshots/) | Live EKS validated |
| **Alerting** | Alert rules that fire on real failures and resolve on recovery | [MLflow failure tests](sprint-08-mlflow-failure-tests-evidence.md) · [Dataset failure tests](sprint-08-dataset-failure-tests-evidence.md) | Live EKS validated |
| **Failure & recovery** | Dataset / MLflow / OOM / crash failures injected, detected, recovered via runbooks | [Release gate §4.B–4.D + §5](sprint-08-release-gate.md) · [Resource failure tests](sprint-08-resource-failure-tests-evidence.md) | Live EKS validated |
| **NetworkPolicy** | Default-deny + explicit-allow, both namespaces (6 allow / 3 deny) | [NetworkPolicy runtime](sprint-08-network-policy-runtime-evidence.md) · [live-EKS §5](sprint-08-live-eks-evidence.md) | Live EKS validated |
| **Workload identity** | EKS Pod Identity (VPC CNI) — no static credentials in the pipeline | [Sprint 7 runtime](sprint-07-runtime-evidence.md) · [ADR-024](../decisions/ADR-024-vpc-cni-pod-identity.md) | Live EKS validated |
| **Image scanning** | Trivy vulnerability scan enforced in CI; documented exceptions | [Image scan evidence](sprint-08-image-scan-evidence.md) | Static/CI validated |
| **Supply chain** | SBOM generation + immutable-digest provenance (git → digest → pod) | [SBOM/provenance evidence](sprint-08-sbom-provenance-evidence.md) · [Release gate §9](sprint-08-release-gate.md) | Runtime validated (mechanism) |
| **Cost & cleanup** | Lifecycle cost control; symmetric `terraform destroy`, verified clean | [PR 16 §17](sprint-08-pr16-release-validation-evidence.md) · [live-EKS §8](sprint-08-live-eks-evidence.md#8-teardown) | Live EKS validated |
| **Model serving / GitOps / remote state / multi-region** | — | [§11 Known Limitations](#11--known-limitations) | Explicitly deferred |

---

## 1 · Start Here

The repository is a cloud-native **MLOps platform on AWS EKS**. The interesting
engineering is the **platform around the ML workload** — infrastructure as code,
workload identity, in-cluster experiment tracking, observability, controlled
failure/recovery testing, and supply-chain controls — not the classifier itself.

- **Landing page:** [repository README](../../README.md)
- **What ran for real:** [PR 16 live-EKS validation](sprint-08-pr16-release-validation-evidence.md)
- **The go/no-go audit:** [Sprint 8 Release Gate — PASS](sprint-08-release-gate.md)

## 2 · Final Architecture

| Capability | Claim | Evidence Type | Canonical Link |
|------------|-------|---------------|----------------|
| System design | The final, integrated architecture and its key properties | Static | [docs/architecture.md](../architecture.md) |
| Design decisions | 37 Architecture Decision Records covering every major choice | Static | [docs/decisions/](../decisions/) |
| Diagrams | Architecture / dataflow diagrams | Static | [docs/diagrams/](../diagrams/) |

## 3 · Cloud Runtime

| Capability | Claim | Evidence Type | Canonical Link |
|------------|-------|---------------|----------------|
| EKS provisioning | Terraform applied 65 resources; EKS v1.35.6; 2× t3.large nodes `Ready` | Live EKS validated | [PR 16 evidence](sprint-08-pr16-release-validation-evidence.md) |
| Full-platform run | Complete merged system provisioned and exercised on real EKS | Live EKS validated | [PR 16 evidence](sprint-08-pr16-release-validation-evidence.md) · [Release gate §4.A](sprint-08-release-gate.md) |
| Workload identity | EKS Pod Identity (VPC CNI); no static AWS credentials in the pipeline | Live EKS validated | [Sprint 7 runtime](sprint-07-runtime-evidence.md) · [ADR-024](../decisions/ADR-024-vpc-cni-pod-identity.md) |
| Secrets encryption | EKS secrets envelope-encrypted with KMS | Live EKS validated | [Sprint 7 runtime](sprint-07-runtime-evidence.md) · [ADR-025](../decisions/ADR-025-eks-secrets-kms-encryption.md) |

## 4 · MLOps Runtime

| Capability | Claim | Evidence Type | Canonical Link |
|------------|-------|---------------|----------------|
| Pipeline execution | DVC pipeline ran to completion as a `batch/v1` Job — exit 0, 5/5 stages `success=1` | Live EKS validated | [PR 16 evidence](sprint-08-pr16-release-validation-evidence.md) |
| Dataset retrieval | Dataset fetched from S3 by an init container, checksum-verified before training | Live EKS validated | [S3 dataset evidence](sprint-07-s3-dataset-runtime-evidence.md) |
| DVC correctness | Declared DVC DAG matches actual Python data dependencies | Static/runtime | [DVC dataflow correction](sprint-07-dvc-dataflow-correction-evidence.md) |
| MLflow tracking | In-cluster MLflow logged runs; persisted to PostgreSQL + SSE-KMS S3 | Live EKS validated | [MLflow integration](sprint-07-mlflow-integration-evidence.md) |
| MLflow persistence | Runs survive an MLflow outage (`pg_up=1` throughout; run count monotonic) | Live EKS validated | [MLflow failure tests](sprint-08-mlflow-failure-tests-evidence.md) |

## 5 · Observability

| Capability | Claim | Evidence Type | Canonical Link |
|------------|-------|---------------|----------------|
| Metrics | Prometheus scraping 11 targets UP (platform + pipeline + exporters) | Live EKS validated | [Release gate §4](sprint-08-release-gate.md) |
| Dashboards | 3 Grafana dashboards (platform health, pipeline ops, MLflow health) | Live EKS validated | [screenshots/](../screenshots/) · [ADR-032](../decisions/ADR-032-grafana-dashboards.md) |
| Pipeline metrics | Per-stage success/duration metrics exported to Pushgateway | Live EKS validated | [Release gate §4](sprint-08-release-gate.md) · [ADR-030](../decisions/ADR-030-pipeline-operational-metrics.md) |
| Visual proof | Baseline-green and failure-red dashboard captures | Live EKS validated | [screenshots/](../screenshots/) |

## 6 · Failure & Recovery

Failures were **injected**, **detected** via metrics/alerts, and **recovered**
via documented [runbooks](../runbooks/) — then the healthy path was re-verified.

| Failure | Claim | Evidence Type | Canonical Link |
|---------|-------|---------------|----------------|
| Dataset unavailable (404) | S3 object missing → fail before training; `PipelineJobFailed` fires | Live EKS validated | [Dataset failure tests](sprint-08-dataset-failure-tests-evidence.md) |
| Checksum mismatch | Digest mismatch → `fetch-dataset` exit 1; pipeline never starts | Live EKS validated | [Dataset failure tests](sprint-08-dataset-failure-tests-evidence.md) |
| MLflow outage | Outage → `MLflowDown` FIRING → gate blocks compute → restore → RESOLVED | Live EKS validated | [MLflow failure tests](sprint-08-mlflow-failure-tests-evidence.md) |
| OOMKilled | Memory limit cut → real `OOMKilled` / exit 137 → restore → Completes | Live EKS validated | [Resource failure tests](sprint-08-resource-failure-tests-evidence.md) |
| Crash / restart | Crash-loop behaviour under `restartPolicy: Never` + `backoffLimit` | Runtime validated | [Resource failure tests](sprint-08-resource-failure-tests-evidence.md) |
| Runbook validation | Every critical runbook exercised against a live failure | Live EKS validated | [Release gate §5 runbook matrix](sprint-08-release-gate.md#5-runbook-validation-matrix) |
| Reliability hardening | Only the fixes the failure tests justified were implemented (others declined with reasons) | Runtime validated | [Reliability hardening](sprint-08-reliability-hardening-evidence.md) |

> **Honesty note.** The live run surfaced **4 real defects** (enforced
> NetworkPolicy blocked Pod Identity; a postgres-exporter arg rejected; a netpol
> harness trusting a curl exit code; an unfireable OOM alert metric) — all
> recorded and fixed. See [live-EKS evidence §3](sprint-08-live-eks-evidence.md#3-findings--4-real-defects-the-live-run-surfaced-all-fixed).

## 7 · Security

| Capability | Claim | Evidence Type | Canonical Link |
|------------|-------|---------------|----------------|
| Image scanning | Trivy vulnerability scan enforced in CI; FIXABLE HIGH bumped; residuals documented | Static/CI validated | [Image scan evidence](sprint-08-image-scan-evidence.md) · [ADR-035](../decisions/ADR-035-container-image-scanning.md) |
| Non-root runtime | Dedicated unprivileged UID/GID; restricted Pod Security Standard | Static | [ADR-010](../decisions/ADR-010-kubernetes-security-hardening.md) |
| Secrets at rest | KMS envelope encryption for EKS secrets | Live EKS validated | [ADR-025](../decisions/ADR-025-eks-secrets-kms-encryption.md) |
| Least-privilege IAM | Scoped IAM foundation; no static pipeline credentials | Live EKS validated | [ADR-016](../decisions/ADR-016-aws-iam-foundation.md) · [Release gate](sprint-08-release-gate.md) |

## 8 · NetworkPolicy

| Capability | Claim | Evidence Type | Canonical Link |
|------------|-------|---------------|----------------|
| Enforcement | Default-deny + explicit-allow in both namespaces; enforced on EKS via VPC CNI | Live EKS validated | [NetworkPolicy runtime](sprint-08-network-policy-runtime-evidence.md) |
| Verified matrix | Canary blocked; **6/6 allowed, 3/3 denied** | Live EKS validated | [live-EKS §5](sprint-08-live-eks-evidence.md#5-pr-7--networkpolicy-runtime) |
| Design | Evidence-mapped communication matrix; S3 egress limitation documented | Static | [ADR-034](../decisions/ADR-034-network-policies.md) · [network-policies.md](../network-policies.md) |

## 9 · Supply Chain

| Capability | Claim | Evidence Type | Canonical Link |
|------------|-------|---------------|----------------|
| SBOM | Software Bill of Materials generated for the image | Runtime validated (mechanism) | [SBOM/provenance evidence](sprint-08-sbom-provenance-evidence.md) |
| Image provenance | Immutable-digest chain: git commit → built image digest → running pod | Runtime validated (mechanism) | [SBOM/provenance evidence](sprint-08-sbom-provenance-evidence.md) · [Release gate §9](sprint-08-release-gate.md) |
| Digest deployment | Workloads reference images by digest, not mutable tags | Live EKS validated | [ADR-036](../decisions/ADR-036-sbom-and-image-provenance.md) |

> **Boundary.** The provenance **mechanism** is executed and verified; a fully
> signed and attested (e.g. cosign/SLSA) supply chain is **not** claimed — see
> [§11 Known Limitations](#11--known-limitations).

## 10 · Cost & Cleanup

| Capability | Claim | Evidence Type | Canonical Link |
|------------|-------|---------------|----------------|
| Cost control | Ephemeral lifecycle: provision → prove → destroy the same session | Live EKS validated | [ADR-020](../decisions/ADR-020-cloud-lifecycle-cost-control.md) |
| Cost observations | Recorded incremental cost drivers (EKS control plane, 2× t3.large, NAT/EBS) | Live EKS validated | [PR 16 §17](sprint-08-pr16-release-validation-evidence.md) |
| Verified teardown | `Destroy complete! Resources: 65 destroyed` — symmetric with 65 added, no orphans | Live EKS validated | [PR 16 teardown](sprint-08-pr16-release-validation-evidence.md) · [live-EKS §8](sprint-08-live-eks-evidence.md#8-teardown) |
| Clean-state proof | Verified clean 3 ways (Terraform state 0 resources; `aws eks` empty; KMS scheduled for deletion) | Live EKS validated | [live-EKS §8](sprint-08-live-eks-evidence.md#8-teardown) |

## 11 · Known Limitations

The repository is explicit about what it does **not** prove. This is a
deliberate honesty boundary, not an omission.

**Not claimed:** 24/7 production operation · formal SLA/SLO · enterprise SRE
maturity · multi-region / disaster recovery · model serving at scale · GitOps ·
Terraform remote state / state locking · service mesh · distributed tracing · a
fully signed & attested supply chain.

Canonical statements of these boundaries:

- [Sprint 8 Release Gate §11 — "Claims safe / Claims that must NOT be made"](sprint-08-release-gate.md)
- [README §14 — Known limitations](../../README.md)
- [Roadmap](../roadmap.md) — where deferred items would go next.

## 12 · Historical Evolution

These documents are **dated snapshots** from earlier sprints. They are preserved
for provenance and to show the engineering trajectory — but the current,
integrated proof is the Sprint 8 release gate and PR 16 above. Do not cite these
as the *current* state of the platform.

| Document | Snapshot | Type |
|----------|----------|------|
| [Sprint 4 — Proof Impact](sprint-04-proof-impact.md) | Pipeline correctness & reproducibility (v1.3.0) | Historical implementation evidence |
| [Sprint 5 — Proof Impact](sprint-05-proof-impact.md) | Kubernetes platform engineering | Historical implementation evidence |
| [Sprint 6 — Proof Impact](sprint-06-proof-impact.md) | Terraform cloud foundation (superseded by Sprint 7) | Historical implementation evidence |
| [Sprint 6 — Runtime Evidence](sprint-06-runtime-evidence.md) | First EKS run (29 resources, Job Complete) | Historical (live, superseded) |
| [Sprint 7 — Proof Impact](sprint-07-proof-impact.md) | Cloud-native MLOps hardening | Historical implementation evidence |
| [Sprint 7 — Runtime Evidence](sprint-07-runtime-evidence.md) | Full platform on EKS (63 resources) | Historical (live, superseded by PR 16) |
| [Sprint 7 — Release Gate](sprint-07-release-gate.md) | CONDITIONAL PASS, v1.6.0 | Historical gate |

---

_This index is the canonical entry point to the repository's evidence. If a
claim in the README is not reachable from here in one or two clicks, that is a
documentation bug — please open an issue._
