# Engineering Capability Matrix

**What does this repository prove this engineer can do?**

This is a **buyer-facing** translation of the repository's implementation into
professional engineering capability — for clients, technical recruiters, hiring
managers, and engineering leaders. It is deliberately **not** a technology list:
every capability below states the *problem addressed*, *what was engineered*, the
*evidence*, a *calibrated proof strength*, *why it matters*, and its *limitations*.

Each row points to canonical proof in the [Evidence Index](README.md). If a
capability cannot be traced to real repository evidence, it is not listed here.

> **Scope.** This is a **portfolio-scoped platform-engineering proof**. Runtime
> evidence comes from **controlled, short-lived validation sessions** on real
> Amazon EKS — provisioned, proven, and destroyed the same session. It is
> **not** a claim of 24/7 production operation, formal SLA/SLO, or enterprise SRE
> maturity. See [§ Known limitations](#known-limitations--what-this-does-not-prove).

---

## How to read "proof strength"

| Label | Meaning |
|-------|---------|
| **Live EKS validated** | Observed on a real, Terraform-provisioned Amazon EKS cluster in the operator's own AWS account, then destroyed. |
| **Runtime validated** | Observed at runtime on Kubernetes (local Docker Desktop / kind) — real execution, not cloud. |
| **CI/static validated** | Enforced by CI, static analysis, or build-time gates — not a live-cluster observation. |
| **Controlled capability demonstration** | A deliberately scoped exercise proving a mechanism works, not a production deployment of it. |
| **Documented engineering evidence** | A decision or design captured in an ADR / evidence doc; the reasoning is the artifact. |

Labels this document **does not** use: *production proven*, *enterprise proven*,
*battle-tested*, *hyperscale* — the evidence does not support them.

---

## Capability summary

The skim view. Detail and evidence links follow below.

| # | Capability | Proof strength | Why a buyer cares |
|---|------------|----------------|-------------------|
| 1 | Kubernetes platform engineering | Live EKS validated | Runs real workloads on Kubernetes the right way for the workload shape |
| 2 | AWS cloud infrastructure | Live EKS validated | Can stand up a coherent cloud footprint, not just consume one |
| 3 | Terraform / infrastructure as code | Live EKS validated · CI/static validated | Infrastructure is reproducible, reviewed, and destroyable |
| 4 | MLOps runtime engineering | Live EKS validated | Operates the ML pipeline as a first-class cloud workload |
| 5 | Workload identity / cloud IAM | Live EKS validated | No static credentials — the modern cloud-security baseline |
| 6 | Data runtime architecture | Live EKS validated | Data arrives correctly and fails safe before compute |
| 7 | MLflow / experiment-tracking platform | Live EKS validated | Owns the tracking platform, not just an SDK call |
| 8 | Kubernetes security | Live EKS validated · CI/static validated | Defense-in-depth from secrets-at-rest to the container |
| 9 | Network security | Live EKS validated | Least-privilege network posture, proven allow *and* deny |
| 10 | Observability | Live EKS validated | Can answer "is it healthy?" from real signals |
| 11 | Reliability engineering | Live EKS validated · Runtime validated | Breaks the system on purpose and proves recovery |
| 12 | Operational readiness / runbooks | Live EKS validated | Recovery is documented and exercised, not improvised |
| 13 | Supply-chain controls | CI/static validated · Runtime validated | Knows what shipped and pins exactly what runs |
| 14 | Cost / ephemeral-validation discipline | Live EKS validated | Proves on real cloud without burning budget |
| 15 | Architecture decision-making | Documented engineering evidence | Judgment and honest scope, captured in 37 ADRs |

---

## 1 · Kubernetes platform engineering

- **Problem addressed.** An ML pipeline is a finite, run-to-completion task, not a
  long-lived service. Running it as a service wastes resources and misreports
  health; running it as a Job needs the right lifecycle, backoff, and security.
- **What was engineered.** The pipeline runs as a `batch/v1` **Job** (not a
  Deployment) with `restartPolicy: Never` + bounded `backoffLimit`, requests/limits,
  a hardened `securityContext`, and an init-container runtime contract for
  dataset/MLflow. Ran to completion on EKS — exit 0, 5/5 stages `success=1`.
- **Evidence.** [PR 16 live-EKS validation](sprint-08-pr16-release-validation-evidence.md) ·
  [Sprint 7 runtime](sprint-07-runtime-evidence.md) ·
  [ADR-009 Job-not-Deployment](../decisions/ADR-009-kubernetes-workload-model.md) ·
  [ADR-013 runtime execution](../decisions/ADR-013-kubernetes-runtime-execution.md)
- **Proof strength.** Live EKS validated.
- **Why it matters.** Choosing the workload model that fits the *shape* of the work
  — and operating its lifecycle correctly — is core Kubernetes platform engineering.
- **Limitations.** Batch workload only; no long-running serving, HPA, or multi-tenant
  scheduling is claimed.

## 2 · AWS cloud infrastructure

- **Problem addressed.** A local pipeline can't demonstrate cloud engineering. It
  needs a real, coherent AWS footprint — network, compute, registry, storage, keys.
- **What was engineered.** A complete EKS-centred footprint provisioned as **65
  Terraform resources**: VPC (multi-AZ, NAT), EKS `v1.35.6` with a managed node
  group (2× t3.large across 2 AZs), IAM, KMS, ECR, and S3 — provisioned, exercised,
  and destroyed in one session.
- **Evidence.** [PR 16 §Environment](sprint-08-pr16-release-validation-evidence.md) ·
  [Release gate §4.A](sprint-08-release-gate.md) ·
  [ADR-015 network](../decisions/ADR-015-aws-network-architecture.md) ·
  [ADR-017 EKS platform](../decisions/ADR-017-eks-platform.md)
- **Proof strength.** Live EKS validated.
- **Why it matters.** Standing up a cloud platform end-to-end — and tearing it down
  cleanly — is the difference between *using* AWS and *engineering on* it.
- **Limitations.** Single-region, two AZs; no multi-region, DR, or HA topology.

## 3 · Terraform / infrastructure as code

- **Problem addressed.** Cloud infrastructure that isn't codified is neither
  reproducible nor reviewable, and drifts silently.
- **What was engineered.** The entire footprint is Terraform-owned with consistent
  tagging and versioning; a CI pipeline runs `fmt`/`init`/`validate`/lint/IaC-scan
  **without AWS credentials**; `apply` and `destroy` were **symmetric** (65 added → 65
  destroyed, verified clean three ways).
- **Evidence.** [PR 16 teardown](sprint-08-pr16-release-validation-evidence.md) ·
  [live-EKS §8 teardown](sprint-08-live-eks-evidence.md#8-teardown) ·
  [ADR-014 Terraform architecture](../decisions/ADR-014-terraform-architecture.md) ·
  [ADR-019 Terraform CI validation](../decisions/ADR-019-terraform-ci-validation.md)
- **Proof strength.** Live EKS validated (apply/destroy) · CI/static validated (pipeline).
- **Why it matters.** Reproducible, reviewed, destroyable infrastructure is the
  baseline expectation for any platform role.
- **Limitations.** **Local** state by deliberate choice; remote state / state locking
  is [explicitly deferred](../decisions/ADR-014-terraform-architecture.md) — suitable
  for solo ephemeral validation, not multi-operator concurrency.

## 4 · MLOps runtime engineering

- **Problem addressed.** Training that only runs on a laptop proves nothing about
  operating ML in the cloud. The pipeline must run reproducibly as a cloud workload.
- **What was engineered.** A DVC-defined pipeline (fetch → preprocess → split →
  evaluate → train) executed as a Kubernetes Job on EKS, with the declared DVC DAG
  corrected to match the actual Python data dependencies, per-stage operational
  metrics, and MLflow tracking wired into the run.
- **Evidence.** [PR 16 evidence](sprint-08-pr16-release-validation-evidence.md) ·
  [Sprint 7 runtime](sprint-07-runtime-evidence.md) ·
  [DVC dataflow correction](sprint-07-dvc-dataflow-correction-evidence.md) ·
  [ADR-006 reproducibility](../decisions/ADR-006-pipeline-reproducibility.md)
- **Proof strength.** Live EKS validated.
- **Why it matters.** The interesting engineering is the *platform around the ML
  workload* — running it reproducibly and observably on real infrastructure.
- **Limitations.** A single classifier pipeline; not a multi-pipeline orchestration
  platform (no Airflow/Kubeflow/Argo Workflows).

## 5 · Workload identity / cloud IAM

- **Problem addressed.** Static AWS keys in a pipeline are the classic cloud-security
  failure — leakable, long-lived, over-scoped.
- **What was engineered.** **EKS Pod Identity** (via the VPC CNI) gives the pipeline a
  dedicated, scoped role — **no static AWS credentials anywhere in the workload** — on
  a least-privilege IAM foundation. The live run surfaced (and fixed) a real defect
  where an enforced NetworkPolicy initially blocked Pod Identity.
- **Evidence.** [Sprint 7 runtime §4](sprint-07-runtime-evidence.md) ·
  [live-EKS §3 defects](sprint-08-live-eks-evidence.md#3-findings--4-real-defects-the-live-run-surfaced-all-fixed) ·
  [ADR-024 Pod Identity](../decisions/ADR-024-vpc-cni-pod-identity.md) ·
  [ADR-016 IAM foundation](../decisions/ADR-016-aws-iam-foundation.md)
- **Proof strength.** Live EKS validated.
- **Why it matters.** Credential-free workload identity is the modern cloud-security
  baseline; debugging it under enforced network policy shows real operational depth.
- **Limitations.** Single-account, single-cluster identity model; no cross-account or
  federated-identity topology.

## 6 · Data runtime architecture

- **Problem addressed.** Baking datasets into images or ConfigMaps doesn't scale and
  hides corruption; data must arrive at runtime and **fail safe** before training.
- **What was engineered.** The dataset is fetched from **S3 by an init container**
  using Pod Identity and **checksum-verified (`sha256 == pinned`) before training
  starts**. A wrong/missing object fails the fetch (`exit 1`) and the pipeline never
  runs — verified by injecting a 404 and a checksum mismatch.
- **Evidence.** [Sprint 7 runtime §5](sprint-07-runtime-evidence.md) ·
  [S3 mechanism](sprint-07-s3-dataset-runtime-evidence.md) ·
  [Dataset failure tests](sprint-08-dataset-failure-tests-evidence.md) ·
  [ADR-027 S3 dataset retrieval](../decisions/ADR-027-s3-dataset-runtime-retrieval.md)
- **Proof strength.** Live EKS validated.
- **Why it matters.** Fail-fast data integrity at the boundary is exactly the kind of
  correctness control that separates a demo from an operable pipeline.
- **Limitations.** Single dataset object; no feature store, streaming ingestion, or
  data-versioning-at-scale is claimed.

## 7 · MLflow / experiment-tracking platform

- **Problem addressed.** Depending on an external SaaS tracker (DagsHub) couples the
  platform to a third party and proves nothing about operating the tracking service.
- **What was engineered.** A **persistent in-cluster MLflow** backed by **PostgreSQL**
  (metadata) + **SSE-KMS S3** (artifacts), replacing DagsHub. Runs and a registered
  model persisted; runs **survived an MLflow outage** (`pg_up=1` throughout, run count
  monotonic) rather than being lost.
- **Evidence.** [Sprint 7 runtime §6](sprint-07-runtime-evidence.md) ·
  [MLflow integration](sprint-07-mlflow-integration-evidence.md) ·
  [MLflow failure tests](sprint-08-mlflow-failure-tests-evidence.md) ·
  [ADR-026 in-cluster MLflow](../decisions/ADR-026-in-cluster-mlflow-platform.md)
- **Proof strength.** Live EKS validated.
- **Why it matters.** Owning the tracking *platform* — its persistence, its failure
  behaviour — is platform engineering, not SDK usage.
- **Limitations.** Single-replica MLflow; availability-only fault tested (no
  data-loss/corruption scenarios), no multi-user auth/RBAC on the tracking server.

## 8 · Kubernetes security

- **Problem addressed.** A workload with root, broad capabilities, and unencrypted
  secrets is an open door regardless of the network around it.
- **What was engineered.** Defense-in-depth: dedicated **non-root** UID/GID, the
  **restricted** Pod Security Standard, seccomp, dropped Linux capabilities, and
  **KMS envelope encryption** of EKS secrets with customer-managed keys.
- **Evidence.** [Sprint 7 runtime §1](sprint-07-runtime-evidence.md) ·
  [ADR-010 security hardening](../decisions/ADR-010-kubernetes-security-hardening.md) ·
  [ADR-025 KMS secret encryption](../decisions/ADR-025-eks-secrets-kms-encryption.md)
- **Proof strength.** Live EKS validated (secrets-at-rest / identity) · CI/static
  validated (workload hardening enforced by manifest checks).
- **Why it matters.** Secrets-at-rest + least-privilege containers are table stakes
  reviewers actively check for.
- **Limitations.** No admission-controller policy engine (OPA/Kyverno), runtime threat
  detection, or image signing enforcement.

## 9 · Network security

- **Problem addressed.** A flat cluster network lets any compromised pod reach
  anything — the DB, the internet, other workloads.
- **What was engineered.** **Default-deny + explicit-allow** NetworkPolicies in both
  namespaces from an evidence-mapped communication matrix (PostgreSQL reachable only
  by its two peers, zero egress; the pipeline cannot reach the DB), enforced on EKS via
  the VPC CNI. A runtime harness with an enforcement canary verified **6/6 allowed,
  3/3 denied**.
- **Evidence.** [NetworkPolicy runtime](sprint-08-network-policy-runtime-evidence.md) ·
  [live-EKS §5](sprint-08-live-eks-evidence.md#5-pr-7--networkpolicy-runtime) ·
  [ADR-034 NetworkPolicies](../decisions/ADR-034-network-policies.md)
- **Proof strength.** Live EKS validated.
- **Why it matters.** Proving both the *allow* and the *deny* paths — with a canary —
  is far stronger than writing a policy and assuming it works.
- **Limitations.** S3 egress is bounded to `internet:443`; precise per-bucket egress is
  delegated to IAM + a recommended VPC S3 endpoint (documented). No service mesh / mTLS.

## 10 · Observability

- **Problem addressed.** A batch ML workload that exits in under a minute is invisible
  to naïve monitoring — you can't answer "did it run, and is the platform healthy?"
- **What was engineered.** Prometheus scraping **11 targets UP** across platform,
  pipeline, and exporters; **per-stage** pipeline metrics pushed to a **Pushgateway**
  so they survive the ephemeral Job; and **three purpose-built Grafana dashboards**
  (EKS/Platform Health, Pipeline Operations, MLflow Health) mapped to real operational
  questions — with model quality kept in MLflow, not duplicated into Prometheus.
- **Evidence.** [Release gate §4](sprint-08-release-gate.md) ·
  [visual evidence](visual-evidence.md) ·
  [ADR-030 pipeline metrics](../decisions/ADR-030-pipeline-operational-metrics.md) ·
  [ADR-032 Grafana dashboards](../decisions/ADR-032-grafana-dashboards.md)
- **Proof strength.** Live EKS validated.
- **Why it matters.** Knowing *what* to measure for a batch workload — and keeping the
  operational/experiment boundary clean — is a considered observability design.
- **Limitations.** Ephemeral Prometheus TSDB; no long-term metrics store, centralized
  logging beyond container logs, or distributed tracing.

## 11 · Reliability engineering

- **Problem addressed.** Reliability claims are worthless unless the system has
  actually been broken and observed to recover.
- **What was engineered.** A **controlled failure-injection campaign** on live EKS —
  dataset-unavailable (404), checksum mismatch, MLflow outage, **real OOMKilled
  (exit 137)**, and crash/restart — each **detected** via metric/alert and **recovered**
  to a re-verified healthy state. Reliability hardening was scoped to *only* the fixes
  the failures justified (others declined, with reasons).
- **Evidence.** [Release gate §4.B–4.D](sprint-08-release-gate.md) ·
  [Resource failure tests](sprint-08-resource-failure-tests-evidence.md) ·
  [Reliability hardening](sprint-08-reliability-hardening-evidence.md) ·
  [ADR-037 reliability hardening](../decisions/ADR-037-pipeline-reliability-hardening.md)
- **Proof strength.** Live EKS validated (dataset / MLflow / OOM) · Runtime validated
  (crash-loop behaviour).
- **Why it matters.** Failure-first engineering — and the discipline to *not* over-fix
  — is what senior reliability work actually looks like.
- **Limitations.** Manual, scenario-driven injection; not automated chaos engineering,
  and no SLO/error-budget framework.

## 12 · Operational readiness / runbooks

- **Problem addressed.** An alert with no documented response is just noise; recovery
  that lives only in one engineer's head isn't operable.
- **What was engineered.** Alert rules carry `runbook_url`s, and the runbooks for the
  **three injected failure classes — dataset retrieval, MLflow outage, and OOM — were
  each exercised against a live failure** (release-gate runbook validation matrix: 3/3
  PASS, no undocumented knowledge required), closing the detect → diagnose → remediate
  → verify loop.
- **Evidence.** [Release gate §5 runbook matrix](sprint-08-release-gate.md#5-runbook-validation-matrix) ·
  [runbooks/](../runbooks/) ·
  [ADR-033 alerting](../decisions/ADR-033-alerting.md)
- **Proof strength.** Live EKS validated.
- **Why it matters.** Runbooks tested *against real failures* — not written
  aspirationally — are a strong signal of operational maturity.
- **Limitations.** No on-call rotation, incident-management process, or paging
  integration (Alertmanager routing is deliberately deferred).

## 13 · Supply-chain controls

- **Problem addressed.** If you can't say what's in your image or prove exactly which
  artifact is running, you can't reason about vulnerabilities or provenance.
- **What was engineered.** **Trivy** vulnerability scanning enforced in CI (gate on
  *fixable* HIGH/CRITICAL; residuals documented with expiry, not muted); a **CycloneDX
  SBOM** for both images; and an **immutable-digest provenance chain** — git commit →
  built image `sha256` digest → running pod — with opt-in digest-pinned deploys and a
  runtime `imageID` verification step.
- **Evidence.** [Image scan evidence](sprint-08-image-scan-evidence.md) ·
  [SBOM/provenance evidence](sprint-08-sbom-provenance-evidence.md) ·
  [Release gate §9](sprint-08-release-gate.md) ·
  [ADR-036 SBOM & provenance](../decisions/ADR-036-sbom-and-image-provenance.md)
- **Proof strength.** CI/static validated (scanning) · Runtime validated (provenance
  mechanism).
- **Why it matters.** Knowing what shipped and pinning exactly what runs is the
  foundation of any credible supply-chain story.
- **Limitations.** A fully **signed & attested** (cosign/SLSA) supply chain is **not**
  claimed — signing is optional/opt-in and the enforced gate is deferred.

## 14 · Cost / ephemeral-validation discipline

- **Problem addressed.** Proving cloud engineering shouldn't require an always-on
  cluster quietly draining a budget.
- **What was engineered.** A deliberate **provision → prove → destroy** lifecycle:
  every validation stands up real infrastructure, exercises it, and tears it down the
  same session — with recorded cost drivers and a **symmetric, verified-clean teardown**
  (`65 destroyed`, confirmed three ways: Terraform state empty, `aws eks` empty, KMS
  scheduled for deletion).
- **Evidence.** [PR 16 §17 cost](sprint-08-pr16-release-validation-evidence.md) ·
  [live-EKS §8 teardown](sprint-08-live-eks-evidence.md#8-teardown) ·
  [ADR-020 lifecycle & cost control](../decisions/ADR-020-cloud-lifecycle-cost-control.md)
- **Proof strength.** Live EKS validated.
- **Why it matters.** Cost-aware, leave-no-orphans cloud discipline is exactly what a
  client paying the AWS bill wants to see.
- **Limitations.** Ephemeral by design — this is *not* evidence of steady-state cost
  optimisation of a long-running production system.

## 15 · Architecture decision-making

- **Problem addressed.** Tools alone don't demonstrate judgment; a reviewer needs to
  see *why* each choice was made and *what was deliberately not built*.
- **What was engineered.** **37 Architecture Decision Records** capturing context,
  alternatives, and consequences for every major choice — including explicit, reasoned
  **deferrals** (GitOps, Terraform remote state, service mesh, model serving) that keep
  the scope honest.
- **Evidence.** [37 ADRs](../decisions/) ·
  [architecture.md](../architecture.md) ·
  [Known limitations (Evidence Index §11)](README.md#11--known-limitations)
- **Proof strength.** Documented engineering evidence.
- **Why it matters.** Buyers hire *judgment*. Documented trade-offs and honest scope
  boundaries are a stronger signal than a longer tool list.
- **Limitations.** ADRs record design intent and rationale; they are not themselves
  runtime proof (the runtime proof is the evidence linked in the rows above).

---

## Engineering judgment — decisions, not just tools

The capabilities above rest on deliberate trade-offs. A reviewer can inspect the
reasoning directly:

| Decision | Choice & why | Trade-off accepted | Record |
|----------|--------------|--------------------|--------|
| **Workload model** | `Job`, not `Deployment` — the pipeline is finite | No always-on endpoint; needs backoff/lifecycle handling | [ADR-009](../decisions/ADR-009-kubernetes-workload-model.md) |
| **Dataset delivery** | S3 runtime fetch, not ConfigMap/baked-in | Runtime dependency on S3 + Pod Identity | [ADR-027](../decisions/ADR-027-s3-dataset-runtime-retrieval.md) |
| **Data integrity** | Fail-fast `sha256` checksum before training | A mismatch stops the run (intended) | [ADR-027](../decisions/ADR-027-s3-dataset-runtime-retrieval.md) · [dataset failure tests](sprint-08-dataset-failure-tests-evidence.md) |
| **Experiment tracking** | In-cluster MLflow (PostgreSQL + S3), not DagsHub | Own the tracking platform's persistence & failure modes | [ADR-026](../decisions/ADR-026-in-cluster-mlflow-platform.md) |
| **Workload identity** | Pod Identity, not static AWS keys | Depends on correct CNI/IAM/NetworkPolicy interplay | [ADR-024](../decisions/ADR-024-vpc-cni-pod-identity.md) |
| **Retry policy** | Bounded `backoffLimit`, `restartPolicy: Never` | No infinite retry masking real failures | [ADR-011](../decisions/ADR-011-kubernetes-resource-lifecycle.md) |
| **Failure-first** | Inject failures *before* hardening | Slower to "green"; fixes are evidence-justified | [ADR-037](../decisions/ADR-037-pipeline-reliability-hardening.md) |
| **GitOps** | Deliberately deferred | Manual apply for ephemeral validation | [Evidence Index §11](README.md#11--known-limitations) |
| **Terraform remote state** | Deliberately deferred (local state) | Not safe for multi-operator concurrency | [ADR-014](../decisions/ADR-014-terraform-architecture.md) |

---

## AI Platform Engineering alignment

This project is a strong, honest fit for **AI Platform Engineering / MLOps / senior
platform** positioning — precisely because it is specific about *which* platform
capabilities it proves.

**It strongly proves the platform capabilities around ML workloads:**

- cloud & Kubernetes foundations (Terraform-provisioned EKS);
- operational reliability (failure injection → detection → runbook recovery);
- observability (Prometheus + Grafana + actionable alerts);
- experiment-tracking platform ownership (in-cluster MLflow + PostgreSQL + S3);
- data/runtime integration (S3 runtime retrieval with fail-fast integrity);
- security (workload identity, KMS, NetworkPolicy, hardened containers);
- artifact / supply-chain controls (scan, SBOM, immutable-digest provenance).

**It does NOT prove — and does not claim — the model-serving/inference specialisation:**

- high-throughput LLM serving;
- GPU scheduling;
- KV-cache optimization;
- inference routing;
- large-scale RAG;
- model-serving autoscaling.

Stating that boundary **increases** credibility: the repository demonstrates the
*platform, reliability, and operations* half of AI Platform Engineering on real
cloud infrastructure, and is honest that the *high-scale inference-serving* half is
out of scope.

---

## Known limitations — what this does NOT prove

Consistent with the [Evidence Index §11](README.md#11--known-limitations):

**Not claimed:** 24/7 production operation · formal SLA/SLO · enterprise SRE maturity ·
multi-region / DR · HA topology · model serving at scale · GitOps · Terraform remote
state / locking · service mesh · distributed tracing · a fully signed & attested supply
chain · centralized logging beyond container logs · automated chaos engineering ·
production incident-response organisation · compliance certification.

These are deliberate scope boundaries, documented — not omissions.

---

_This matrix is a translation layer over the [Evidence Index](README.md); it adds no
new claims. Every capability here is verifiable from the canonical evidence linked in
its row, without trusting the author._
