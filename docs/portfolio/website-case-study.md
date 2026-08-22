<!--
Website-ready portfolio case study.

This is a concise public adaptation of the canonical flagship case study
(docs/case-study.md) — not a second, independent narrative. Every claim here is
already grounded in the repository's evidence. Links use absolute GitHub URLs so
the copy can be lifted into a personal website with minimal editing; swap the
base URL if the repository is renamed or mirrored.

Base URL: https://github.com/asadhanif3188/mlops-platform-on-eks
-->

# MLOps Platform on AWS EKS

**Cloud-native MLOps platform engineering — proven with live AWS runtime evidence, not assertions.**

I took a course-style, local ML pipeline and engineered the cloud platform around
it — Terraform-provisioned AWS EKS, secure workload identity, cloud-backed data and
experiment state, four-layer observability, and controlled failure/recovery testing —
then provisioned the whole thing on real Amazon EKS, broke it on purpose, recovered it
under runbook guidance, and tore it down clean in a single session.

---

## How to place this on a website

This file holds **three zoom levels** of the same project. Use the short forms in a
projects section; link them to the full case study (the rest of this file) on its own
page. Do **not** paste the full case study into a projects grid.

### Tier 1 — Project card (for a projects grid / list)

> **MLOps Platform on AWS EKS**
> Cloud-native MLOps platform engineering, proven on live AWS.
>
> A course-style ML pipeline re-engineered into a Terraform-provisioned EKS platform —
> then provisioned, failure-tested, recovered, and torn down clean on real Amazon EKS.
>
> `AWS EKS` · `Terraform` · `Kubernetes` · `MLflow` · `Prometheus / Grafana`
>
> **65 resources · 5/5 stages green · 3 failures recovered · verified teardown**
>
> [Case study →](#) · [GitHub →](https://github.com/asadhanif3188/mlops-platform-on-eks)

### Tier 2 — Featured summary (for a highlighted project block)

> **MLOps Platform on AWS EKS** — I took a course-style, local ML pipeline and
> engineered the cloud platform around it: Terraform-provisioned AWS EKS, EKS Pod
> Identity (no static keys), S3-backed data, an in-cluster MLflow stack on PostgreSQL,
> and four-layer Prometheus/Grafana observability. The whole system was provisioned on
> **real Amazon EKS** in one session — the pipeline ran to completion (exit 0, 5/5
> stages), three critical failures were injected and recovered under runbook guidance,
> and the environment was destroyed and verified clean. It is a portfolio-scoped
> platform-engineering proof, backed by runtime evidence rather than assertions.
>
> [Read the case study →](#) · [View on GitHub →](https://github.com/asadhanif3188/mlops-platform-on-eks)

> **Integration note.** Replace the `(#)` "Case study" targets with the URL of the
> detail page once it exists. The Tier-1 metrics and chips are the safe, calibrated
> headline set — keep them consistent with the [Results](#results) section below.

---

## Full case study (project detail page)

## Challenge

A pipeline that reproduces on a laptop demonstrates reproducibility. It says nothing
about *operating* machine learning on cloud infrastructure: how the environment is
provisioned and destroyed, how a workload obtains credentials without static keys,
where data and experiment state live durably, how you know the system is healthy, and
what happens when it breaks.

The starting point was a working local MLOps pipeline (DVC + MLflow) — a good learning
artifact, but the wrong proof for a platform-engineering portfolio. The problem I set
out to solve was the gap between *"it runs on my machine"* and *"it runs — observably
and recoverably — on infrastructure I provisioned and can prove I cleaned up."* The ML
model (a RandomForest classifier) is held deliberately small so the platform is the
subject.

## Architecture

A Terraform-owned AWS foundation hosts an EKS cluster that runs the ML pipeline as a
finite Kubernetes **Job**, wrapped in an in-cluster data/tracking plane and an
observability plane.

- **Infrastructure as code** — VPC and networking, least-privilege IAM, managed EKS,
  ECR, KMS keys, and S3 buckets, all provisioned by Terraform so the environment is
  reproducible and destroyable.
- **Workload** — the pipeline runs as a `batch/v1` Job (`restartPolicy: Never`), not a
  Deployment, because it is finite batch work; manifests are Kustomize base + overlays.
- **Identity** — EKS Pod Identity grants scoped AWS access with no long-lived keys in
  the cluster.
- **Data & tracking** — the dataset is fetched from S3 at runtime and checksum-verified
  before training; an in-cluster MLflow server persists run metadata to PostgreSQL and
  artifacts to SSE-KMS-encrypted S3.
- **Observability** — Prometheus scrapes four signal layers (Kubernetes platform, the
  ephemeral pipeline Job, MLflow, PostgreSQL); Grafana serves three dashboards; eight
  unit-tested alert rules each map to a concrete operator action.
- **Isolation & supply chain** — deny-by-default NetworkPolicy governs east-west
  traffic; images are non-root, scanned, carry a CycloneDX SBOM, and are deployed by
  immutable digest.

Every major choice is recorded as one of 37 Architecture Decision Records.

## What I Engineered

The ML model concept originates from the course-style pipeline; the platform around it
is my engineering. Concrete ownership:

- The **Terraform AWS/EKS foundation** — VPC, IAM, managed EKS, ECR, KMS, S3.
- The **Kubernetes runtime** — Job workload model, Kustomize overlays, hardened
  security context.
- **Workload identity** via EKS Pod Identity — removing all static AWS credentials.
- **Cloud-backed data delivery** — S3 runtime retrieval with checksum verification.
- The **in-cluster MLflow stack** on PostgreSQL + S3 for durable experiment state.
- The **observability plane** — Prometheus, Grafana dashboards, pipeline operational
  metrics, and actionable alert rules.
- **Security hardening** — NetworkPolicy, KMS encryption, least-privilege IAM,
  non-root restricted runtime.
- **Failure injection, runbooks, and reliability hardening** driven by observed
  failures.
- The **repository and quality gates** — Ruff, mypy (strict), 233 passing tests,
  pre-commit, CI, and supply-chain controls (SBOM + digest provenance).

## Key Decisions

Engineering judgment mattered more than tool selection. A few decisions, each with a
recorded trade-off:

- **Kubernetes Job, not Deployment** — the workload is finite batch training;
  inventing a long-running service would have misrepresented it. (This choice later
  exposed a real alerting bug — see below.)
- **Terraform as the single infrastructure owner** — one reproducible, destroyable
  source of truth; remote state/locking deferred for a single-operator proof.
- **Workload identity over static credentials** — no long-lived AWS keys in a public
  repository.
- **S3 runtime dataset retrieval, not a baked image or ConfigMap** — data stays
  cloud-backed, versionable, and checksum-verifiable.
- **In-cluster MLflow on PostgreSQL + S3** — self-owned experiment state instead of an
  external SaaS dependency.
- **Test failure *before* hardening** — reliability work was driven by injected,
  observed failures, and only the fixes those failures justified were implemented.
- **GitOps and remote state deferred, not omitted** — declaring the boundary is more
  honest than pretending it does not exist.

## Operational Proof

The strongest evidence is a single, authoritative **provision → prove → destroy**
session on live Amazon EKS:

- Terraform **applied 65 resources**; EKS `v1.35.6`, 2 × t3.large nodes Ready across 2
  availability zones.
- Dataset **retrieved from S3 via Pod Identity** (no static keys), checksum verified.
- Pipeline **Job completed, exit 0 — all 5 stages green**.
- MLflow **run persisted** — metadata in PostgreSQL, artifacts in S3.
- Prometheus reported **11 targets UP**; three Grafana dashboards served live data.
- Three failures **injected, detected, and recovered** under runbook guidance.
- Final healthy state restored — **0 alerts firing, 16 runs persisted**.
- `terraform destroy` **symmetric — 65 destroyed** and verified clean three independent
  ways; nothing left billing.

## Failure & Recovery

Reliability here means a closed loop: **inject → detect → alert → diagnose via runbook
→ recover → re-verify**. Three failures were run on live EKS and recovered using only
repository runbooks:

| Failure | Detection signal | Recovery |
|---|---|---|
| Dataset unavailable (S3 404) | per-stage `stage_success=0`; `PipelineJobFailed` firing | re-upload dataset; 5/5 stages green |
| MLflow outage | `probe_success=0` while `pg_up=1` | scale replicas to 1; runs preserved |
| OOMKilled | `terminated_reason="OOMKilled"`, exit 137 | restore memory limit; pod completes |

The most valuable evidence is the set of **four real defects that static validation
could not catch** — all passed CI and surfaced only under live enforcement. The
sharpest two: an enforced NetworkPolicy silently **blocked EKS Pod Identity** (two
independently-correct controls that were jointly wrong), and an OOM alert keyed on a
metric a `restartPolicy: Never` Job never emits — **unfireable for this workload**
until a real OOM disproved it. Both were root-caused, fixed, and re-validated.

## Security / Reliability

Defense-in-depth, scoped to what runtime evidence supports:

- **No static AWS credentials** — EKS Pod Identity issues scoped, short-lived
  credentials.
- **Encryption at rest** — SSE-KMS on S3, KMS envelope encryption for EKS secrets, via
  customer-managed keys.
- **Least-privilege IAM** and EKS access entries (API auth mode).
- **Hardened runtime** — non-root, `allowPrivilegeEscalation: false`, Pod Security
  Admission *restricted*, seccomp `RuntimeDefault`, all Linux capabilities dropped.
- **Deny-by-default NetworkPolicy** with explicit allow rules, verified on EKS.
- **Image scanning** (Trivy) enforced in CI; SBOM generated; workloads deployed by
  immutable digest.

Reliability hardening — bounded retry, checksum integrity, resource coupling — was
implemented only where the failure tests justified it; other candidate fixes were
declined with recorded reasons.

## Results

- A complete cloud-native MLOps platform **provisioned, exercised, and destroyed on
  real Amazon EKS** in one controlled session.
- **23 proof dimensions** audited by a release gate — verdict **PASS**.
- **Four static-invisible defects** found and fixed under live enforcement.
- **Three critical failures** injected and recovered via documented runbooks.
- **Symmetric, verified teardown** — provable cost discipline.

## What This Demonstrates

- **Platform-engineering judgment** — workload modeling, tool boundaries, and a
  deliberately bounded scope, each defended with a trade-off.
- **Kubernetes operations** and **cloud infrastructure as code** — reproducible,
  destroyable, with verified symmetric teardown.
- **MLOps runtime design** — cloud-backed data and durable experiment state.
- **Security and observability engineering** — workload identity over static keys, and
  signal design that localizes faults.
- **Reliability engineering and evidence-driven debugging** — break it first, fix what
  the failure justifies, and prove recovery.
- **Intellectual honesty** — every claim scoped to its evidence; every deferred
  capability named rather than hidden.

## Evidence / GitHub

Every claim above is traceable to a canonical runtime record:

- **Repository:** <https://github.com/asadhanif3188/mlops-platform-on-eks>
- **Evidence index:** <https://github.com/asadhanif3188/mlops-platform-on-eks/blob/main/docs/proof/README.md>
- **Flagship case study:** <https://github.com/asadhanif3188/mlops-platform-on-eks/blob/main/docs/case-study.md>
- **Architecture:** <https://github.com/asadhanif3188/mlops-platform-on-eks/blob/main/docs/architecture.md>
- **Live-EKS validation evidence:** <https://github.com/asadhanif3188/mlops-platform-on-eks/blob/main/docs/proof/sprint-08-pr16-release-validation-evidence.md>

> **View the architecture, runtime evidence, failure tests, and engineering decisions
> on GitHub.**

## Honest Limitations

This is a **portfolio-scoped platform-engineering proof**, not a production service.
The validation cluster was single-operator, two nodes, and short-lived — sufficient to
prove the operational model, not production scale. It does **not** claim: enterprise
SRE or formal SLA/SLO, 24/7 operations, multi-region high availability or disaster
recovery, service mesh, model serving at scale, GitOps or Terraform remote state
(intentionally deferred), distributed tracing, or a fully signed and attested supply
chain (SBOM and digest provenance yes; cosign/SLSA deferred). The claims stop precisely
where the evidence stops.
