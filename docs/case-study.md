# Case Study — Engineering a Cloud-Native MLOps Platform on AWS EKS

> A platform-engineering case study: how a course-style, local ML pipeline was
> re-engineered into a reproducible, secure, observable, and recoverable system
> running on real Amazon EKS — and proven with runtime evidence rather than
> assertions.

**Reading paths.** Reviewers and interviewers: read start to finish. Recruiters and
hiring managers: [Executive Summary](#1--executive-summary) and
[What This Demonstrates](#16--what-this-demonstrates). Technical verifiers: every
claim links to its canonical proof in the [Evidence Index](proof/README.md).

This document is self-contained. It does not require reading the sprint history to
follow, but every headline claim is traceable to a dated evidence record.

---

## 1 · Executive Summary

This repository takes a deliberately small ML workload — a `RandomForestClassifier`
on the Pima Indians Diabetes dataset — and builds the **cloud-native platform around
it** that turns "a training script" into an operable system. The model is the least
interesting part; the engineering subject is the platform: infrastructure as code,
secure workload identity, cloud-backed data and experiment state, four-layer
observability, controlled failure-and-recovery testing, and container supply-chain
traceability.

The distinguishing feature is **proof**. The complete, merged platform was
provisioned on real Amazon EKS in a single `provision → prove → destroy` session:
Terraform stood up 65 resources, the pipeline ran to completion as a Kubernetes Job
(exit 0, all five stages green), an in-cluster MLflow stack persisted the run to
PostgreSQL and KMS-encrypted S3, Prometheus scraped 11 targets, three critical
failures were injected and recovered under runbook guidance, every mandatory alert
fired and resolved, and the environment was destroyed and verified clean three
independent ways. The release gate audited 23 proof dimensions and returned **PASS**
([Sprint 8 Release Gate](proof/sprint-08-release-gate.md),
[live-EKS validation](proof/sprint-08-pr16-release-validation-evidence.md)).

It is explicitly a **portfolio-scoped platform-engineering proof**, not a production
service. Validation was controlled and ephemeral; the claims stop precisely where the
evidence stops (see [Limitations](#15--limitations)).

## 2 · Starting Point

The project originated from a course-style, local MLOps pipeline: DVC for
reproducibility and MLflow for experiment tracking, run on a laptop. As a *learning*
artifact that is a perfectly good starting point — it teaches the pipeline
vocabulary, the DAG mental model, and the discipline of tracking runs. It is not
being demeaned here; it did its job.

The problem is that it proves the wrong thing for a *professional* proof asset. A
pipeline that reproduces on one machine demonstrates reproducibility, but it says
nothing about **operating** ML on cloud infrastructure — how the environment is
provisioned and torn down, how the workload obtains credentials without static keys,
where data and experiment state live durably, how you know the system is healthy, and
what happens when it breaks. A reviewer evaluating platform-engineering capability
cannot learn any of that from a laptop notebook. The gap between "it runs on my
machine" and "it runs, observably and recoverably, on infrastructure I provisioned
and can prove I cleaned up" is exactly the gap this project set out to close. The
transformation itself — not the classifier — is the evidence of engineering maturity.

## 3 · Problem

The technical problem, stated plainly:

> How do you take a local, course-style ML pipeline and engineer the platform around
> it so that it can run **reproducibly, securely, observably, and recoverably** on
> real cloud infrastructure — and prove each of those properties with runtime
> evidence rather than assertions?

Every subsequent decision is downstream of that sentence. The ML model is held fixed
and simple on purpose, so that the platform engineering is the variable under study.

## 4 · Constraints

The work was done inside a deliberate constraint envelope, and the constraints shaped
the design as much as the goals did:

- **Public repository, zero secrets.** Everything is open; nothing sensitive is
  committed. Identity is handled by EKS Pod Identity, encryption keys by KMS — no
  static credentials anywhere in the repo or the cluster.
- **Cloud-cost discipline.** Real EKS is not free. Validation runs are ephemeral:
  provisioned, proven, and destroyed within a session, with teardown verified so
  nothing is left billing ([ADR-020](decisions/ADR-020-cloud-lifecycle-cost-control.md)).
- **Evidence-backed claims only.** If a capability cannot be shown with a runtime
  artifact — logs, metric values, alert states, `terraform destroy` output — it is
  either demoted to a design-only claim or dropped.
- **No unnecessary platform expansion.** The temptation in platform work is to add
  every fashionable component. Scope was held to what the workload actually needs;
  speculative surface area was declined and recorded as declined.
- **Controlled validation, not production claims.** The cluster was single-operator,
  two nodes, short-lived. It proves the operational *model*, not production scale, and
  the writing never blurs that line.
- **GitOps and Terraform remote state intentionally deferred.** Both are the obvious
  "next" steps; both were consciously left out of scope for a portfolio proof and
  documented as deferred rather than quietly omitted
  ([Roadmap](roadmap.md), [ADR-014](decisions/ADR-014-terraform-architecture.md)).

## 5 · Final Architecture

The system is a Terraform-owned AWS foundation hosting an EKS cluster that runs the
ML pipeline as a finite Kubernetes Job, surrounded by an in-cluster data/tracking
plane and an observability plane.

- **Infrastructure (Terraform).** VPC and networking, least-privilege IAM, a managed
  EKS cluster, ECR, KMS keys, and S3 buckets — provisioned as code so the entire
  environment is reproducible and destroyable
  ([ADR-014](decisions/ADR-014-terraform-architecture.md),
  [ADR-017](decisions/ADR-017-eks-platform.md)).
- **Workload.** The pipeline runs as a `batch/v1` **Job** (`restartPolicy: Never`),
  not a Deployment — it is finite batch work, and inventing a long-running HTTP
  service would have been dishonest about what the workload is
  ([ADR-009](decisions/ADR-009-kubernetes-workload-model.md)). Manifests are Kustomize
  base + `local`/`aws` overlays.
- **Identity.** EKS Pod Identity (via the VPC CNI) grants the workload scoped AWS
  access with no long-lived keys in the cluster
  ([ADR-024](decisions/ADR-024-vpc-cni-pod-identity.md)).
- **Data & tracking plane.** The dataset is fetched from S3 at runtime by an init
  container and checksum-verified before training
  ([ADR-027](decisions/ADR-027-s3-dataset-runtime-retrieval.md)); an in-cluster MLflow
  server persists run metadata to PostgreSQL and artifacts to SSE-KMS S3
  ([ADR-026](decisions/ADR-026-in-cluster-mlflow-platform.md)).
- **Observability plane.** Prometheus scrapes four signal layers (Kubernetes
  platform, the ephemeral pipeline Job, the MLflow server, PostgreSQL); Grafana serves
  three dashboards; eight unit-tested alert rules key to concrete operator actions
  ([ADR-028](decisions/ADR-028-observability-architecture.md),
  [ADR-033](decisions/ADR-033-alerting.md)).
- **Isolation & supply chain.** Deny-by-default NetworkPolicy governs east-west
  traffic ([ADR-034](decisions/ADR-034-network-policies.md)); images are non-root and
  scanned, carry a CycloneDX SBOM, and are deployed by immutable digest
  ([ADR-036](decisions/ADR-036-sbom-and-image-provenance.md)).

Full system design: [architecture.md](architecture.md). Every major choice is an ADR
([ADR-001 … ADR-037](decisions/README.md)).

## 6 · Engineering Evolution

The platform was not designed top-down and then built; it was grown in versioned
increments, each closing the review findings of the one before. The trajectory
matters because it shows the design responding to evidence rather than to a plan on
paper ([CHANGELOG](../CHANGELOG.md)):

- **Foundations (v1.0.0–v1.2.0).** Repository professionalization, an engineering
  quality baseline (Ruff, mypy strict, pytest, pre-commit, CI), and containerization —
  turning a script into a portable, gated artifact.
- **Pipeline correctness (v1.3.0, Sprint 4).** Made the pipeline genuinely
  reproducible and its evaluation honest: a disjoint held-out split, seeded fixtures,
  and a **contract** test suite that guards the DVC DAG against the code
  ([ADR-006](decisions/ADR-006-pipeline-reproducibility.md),
  [ADR-007](decisions/ADR-007-held-out-evaluation.md)).
- **Kubernetes (v1.4.0, Sprint 5).** Expressed the pipeline as a Kubernetes Job with
  Kustomize overlays and a hardened security context.
- **Cloud foundation (v1.5.0, Sprint 6).** Stood up the AWS platform as Terraform and
  ran the first live EKS execution.
- **Cloud-native hardening (v1.6.0, Sprint 7).** Closed every HIGH/MEDIUM finding
  from the Sprint 6 review: Terraform-managed ECR, private-by-default EKS API, Pod
  Identity, KMS-encrypted secrets, in-cluster MLflow replacing the external tracking
  path, and S3 runtime dataset retrieval — proven by a full live end-to-end run.
- **Observability & reliability (v1.7.0, Sprint 8).** Instrumented the platform,
  then deliberately broke it on real EKS to prove detection, alerting, and
  runbook-driven recovery — the subject of §8–§13 below.

## 7 · Key Engineering Decisions

Each decision below is recorded as an ADR; the point here is the *reasoning and
trade-off*, not the outcome alone.

**1. Kubernetes Job, not Deployment.** *Why:* the workload is finite batch training,
not a service. *Trade-off:* gives up always-on endpoints and rolling-update
semantics, and — as §8 shows — some alerting metrics assume restarts a `Never` Job
never produces. *Evidence:* [ADR-009](decisions/ADR-009-kubernetes-workload-model.md).

**2. Terraform as the single infrastructure owner.** *Why:* one reproducible,
destroyable source of truth for VPC/IAM/EKS/ECR/KMS/S3. *Trade-off:* slower iteration
than click-ops; remote state/locking deliberately deferred (single operator).
*Evidence:* [ADR-014](decisions/ADR-014-terraform-architecture.md).

**3. Terraform / Kustomize boundary.** *Why:* Terraform owns cloud resources;
Kustomize owns in-cluster manifests — each tool where it is strongest. *Trade-off:*
two toolchains and a handoff to manage rather than one. *Evidence:*
[ADR-018](decisions/ADR-018-aws-eks-deployment-overlay.md).

**4. Workload identity over static credentials.** *Why:* no long-lived AWS keys in a
public-repo project; EKS Pod Identity issues scoped, short-lived credentials.
*Trade-off:* more moving parts, and a subtle failure mode when NetworkPolicy meets the
Pod Identity agent (§8, Finding 1). *Evidence:*
[ADR-024](decisions/ADR-024-vpc-cni-pod-identity.md).

**5. S3 runtime dataset retrieval, not a baked image or ConfigMap.** *Why:* keeps data
cloud-backed, versionable, and checksum-verifiable, decoupled from the image.
*Trade-off:* adds a runtime dependency and an init-container fetch step that can fail
(and did, usefully — §8). *Evidence:*
[ADR-027](decisions/ADR-027-s3-dataset-runtime-retrieval.md).

**6. In-cluster MLflow on PostgreSQL + S3.** *Why:* a self-contained platform whose
experiment state is durable and owned, rather than an external SaaS dependency.
*Trade-off:* the team now operates MLflow and PostgreSQL, including their failure
modes. *Evidence:* [ADR-026](decisions/ADR-026-in-cluster-mlflow-platform.md).

**7. Prometheus for operational signals, not duplicated MLflow metrics.** *Why:*
Prometheus answers operator questions ("is it up, is it failing, why?"); MLflow owns
experiment metrics. Keeping them separate avoids conflating model quality with system
health. *Trade-off:* two metric systems to understand. *Evidence:*
[ADR-028](decisions/ADR-028-observability-architecture.md),
[ADR-030](decisions/ADR-030-pipeline-operational-metrics.md).

**8. Test failure *before* hardening.** *Why:* reliability work should be driven by
observed failures, not speculation. Failures were injected on live EKS first, and only
the fixes those failures justified were implemented — others were declined with
recorded reasons. *Trade-off:* slower, and it exposes defects publicly. *Evidence:*
[ADR-037](decisions/ADR-037-pipeline-reliability-hardening.md),
[reliability hardening evidence](proof/sprint-08-reliability-hardening-evidence.md).

**9. NetworkPolicy least privilege (deny-by-default).** *Why:* east-west isolation so
the two namespaces communicate only where required. *Trade-off:* default-deny is
unforgiving — it blocked Pod Identity until an explicit allow was added (§8).
*Evidence:* [ADR-034](decisions/ADR-034-network-policies.md).

**10. GitOps and remote state deferred, not omitted.** *Why:* both are the right next
step, but out of scope for a single-operator portfolio proof; declaring the boundary
is more honest than pretending it doesn't exist. *Evidence:* [Roadmap](roadmap.md).

## 8 · Failures That Changed the Design

The most valuable evidence in this project is the set of defects that **static
validation could not catch** — every one passed CI (`kubeconform`, Kustomize,
`validate.py`, `promtool`) and manifested only under live enforcement on EKS. Finding
and fixing them is the entire point of testing failure on real infrastructure
([findings §3](proof/sprint-08-live-eks-evidence.md#3-findings--4-real-defects-the-live-run-surfaced-all-fixed)).

**Enforced NetworkPolicy blocked EKS Pod Identity (CRITICAL).**
*Expectation:* Pod Identity (proven live in Sprint 7) and NetworkPolicy (validated
statically) would compose. → *What happened:* `fetch-dataset` failed with
`CredentialRetrievalError: Connect timeout` on `169.254.170.23`. → *Root cause:* the
S3-egress policy excepted the entire `169.254.0.0/16` link-local range to keep the
`:443` rule tight, and nothing re-allowed the Pod Identity agent at
`169.254.170.23:80`; default-deny did the rest. Sprint 7 had proven Pod Identity but
with **no enforced NetworkPolicy** — this was the first time the two ran together. →
*Fix:* a least-privilege `allow-pod-identity-egress` rule (one `/32`, TCP/80). →
*Revalidation:* `fetch-dataset` → exit 0, run Complete, artifacts in S3. → *Lesson:*
independently-correct controls can be jointly wrong; only running them together
proves composition.

**MLflow outage.** *Expectation:* the tracking server is a dependency; if it drops,
the operator should be told precisely and prior runs should survive. → *What
happened:* scaling the MLflow Deployment to zero produced `probe_success=0` while
`pg_up=1`. → *Root cause:* the server, not the database, was down. → *Fix:*
`kubectl scale --replicas=1` per [runbook](runbooks/mlflow-unavailable.md). →
*Revalidation:* probe recovered, `MLflowDown` RESOLVED, run count preserved (6→6). →
*Lesson:* a good signal *localizes* the fault — `pg_up=1` ruled out the database and
sent the runbook straight to the Deployment layer.

**OOMKilled, and an alert that could never fire.** *Expectation:* cutting the memory
limit to 128Mi would OOM the pod and fire `PipelineJobOOMKilled`. → *What happened:*
the pod was genuinely `OOMKilled` (exit 137, `terminated.reason=OOMKilled`) — but the
alert stayed silent. → *Root cause:* the alert keyed on
`kube_pod_container_status_last_terminated_reason`, which KSM derives from a
container's *previous* termination. A `restartPolicy: Never` Job terminates once and
never restarts, so `lastState` is empty and KSM emits no series — the alert was
**unfireable for this workload**. → *Fix:* key on the current-state metric
`kube_pod_container_status_terminated_reason`. → *Revalidation:* with the metric
corrected, `PipelineJobOOMKilled` fired on the exploratory live run (14:19:49Z), and
on the PR 16 session restoring the 512Mi limit returned the pod to exit 0 under the
[runbook](runbooks/oomkilled.md). → *Lesson:* an untested alert is a hypothesis;
this one was false until a real OOM disproved it.

**Two harness/config defects the live run exposed.** A pinned `postgres-exporter`
image rejected the `--auto-discover-databases=false` flag form and crash-looped (fixed
to `--no-auto-discover-databases`); and the NetworkPolicy test harness trusted curl's
exit code, which returns 23 on the response-write phase inside a hardened
`kubectl exec` even on an HTTP 200 — so it reported 6/6 *working* paths as failures
until it was changed to judge on TCP-connect success. Both are documented with fixes
and re-runs ([§3](proof/sprint-08-live-eks-evidence.md#3-findings--4-real-defects-the-live-run-surfaced-all-fixed)).

## 9 · Reliability & Recovery

Reliability here means a closed loop: **inject → detect → alert → diagnose via
runbook → recover → re-verify healthy**. Three mandatory failure scenarios were run
on live EKS, each recovered using only a repository runbook with no undocumented
tribal knowledge, and each returned to a verified-clean state
([Release gate §4–§5](proof/sprint-08-release-gate.md#5-runbook-validation-matrix)):

| Failure | Detection signal | Recovery | Runbook |
|---|---|---|---|
| Dataset unavailable (S3 404) | `stage_success{stage="fetch_dataset"}=0`; `PipelineJobFailed` FIRING | Re-upload (SSE-KMS); 5/5 stages green | [dataset-retrieval-failure](runbooks/dataset-retrieval-failure.md) |
| MLflow outage | `probe_success=0` while `pg_up=1`; `MLflowDown` FIRING | Scale replicas to 1; runs preserved | [mlflow-unavailable](runbooks/mlflow-unavailable.md) |
| OOMKilled | `...terminated_reason{reason="OOMKilled"}=1`, exit 137 | Restore 512Mi; pod Completes | [oomkilled](runbooks/oomkilled.md) |

Hardening followed the evidence: bounded retry, checksum integrity, and resource
coupling were implemented because the failure tests justified them; other candidate
fixes were declined with recorded reasons
([reliability hardening](proof/sprint-08-reliability-hardening-evidence.md)). The
runbook validation matrix records each critical runbook being exercised against a live
failure, not merely written.

## 10 · Security

Security is defense-in-depth across identity, encryption, isolation, and runtime
posture, and the claims are scoped to what runtime evidence supports:

- **No static AWS credentials.** EKS Pod Identity issues scoped, short-lived
  credentials for S3 access ([ADR-024](decisions/ADR-024-vpc-cni-pod-identity.md)) —
  verified live in the Sprint 7 runtime run.
- **Encryption at rest.** SSE-KMS on the S3 dataset/artifact buckets and KMS envelope
  encryption for EKS Secrets, backed by customer-managed keys
  ([ADR-025](decisions/ADR-025-eks-secrets-kms-encryption.md)).
- **Least-privilege IAM** and EKS access entries (API auth mode, no `aws-auth`
  configmap) ([ADR-016](decisions/ADR-016-aws-iam-foundation.md),
  [ADR-023](decisions/ADR-023-eks-access-control.md)).
- **Hardened runtime.** Non-root containers (fixed UID/GID 10001),
  `allowPrivilegeEscalation: false`, Pod Security Admission *restricted*, seccomp
  `RuntimeDefault`, all Linux capabilities dropped, ServiceAccount token automount off
  ([ADR-010](decisions/ADR-010-kubernetes-security-hardening.md),
  [kubernetes-security.md](kubernetes-security.md)).
- **Deny-by-default NetworkPolicy** with explicit allow rules, verified on EKS
  (6 allowed / 3 denied paths) ([network-policies.md](network-policies.md)).
- **Image scanning.** Trivy enforced in CI; fixable HIGH vulnerabilities bumped,
  residuals documented ([ADR-035](decisions/ADR-035-container-image-scanning.md)).

Full posture: [SECURITY.md](../SECURITY.md).

## 11 · Observability

Observability is structured as **four signal layers**, each answering a question an
operator would actually act on, and instrumented only where action is possible
([observability.md](observability.md)):

- **Kubernetes platform** — KSM, node-exporter, cAdvisor, kubelet (is the cluster and
  the Job healthy?).
- **Ephemeral pipeline Job** — per-stage `mlops_pipeline_stage_success` and
  `_duration_seconds` pushed to the Pushgateway so a finite Job's metrics survive its
  completion and answer "which stage failed?"
  ([ADR-030](decisions/ADR-030-pipeline-operational-metrics.md)).
- **MLflow server** — a blackbox HTTP probe (`probe_success`) for availability.
- **PostgreSQL** — postgres-exporter (`pg_up`, connections, PVC fill) to distinguish
  database faults from server faults
  ([ADR-031](decisions/ADR-031-mlflow-postgres-monitoring.md)).

On the PR 16 live run Prometheus reported **11 targets UP** and three Grafana
dashboards (EKS Platform Health, Pipeline Operations, MLflow Platform Health) served
live data; the exploratory live-EKS campaign captured those dashboards in
baseline-green and failure-red states ([screenshots](screenshots/)). Eight alert rules
are unit-tested with
`promtool test rules` and each is keyed to a specific operator action
([alerting.md](alerting.md), [ADR-033](decisions/ADR-033-alerting.md)).

## 12 · Supply Chain

The build produces a CycloneDX **SBOM**, and workloads are deployed by **immutable
digest** rather than mutable tag, establishing a traceable chain: git commit → built
image digest → running pod
([supply-chain-provenance.md](supply-chain-provenance.md),
[ADR-036](decisions/ADR-036-sbom-and-image-provenance.md)). The provenance *mechanism*
was executed and the digest chain verified live
([Release gate §9](proof/sprint-08-release-gate.md)). The boundary is stated
deliberately: a fully **signed and attested** supply chain (cosign / SLSA) is **not**
claimed — the SBOM and digest provenance are, the cryptographic attestation is
deferred.

## 13 · Real Runtime Proof

The strongest evidence is a single, authoritative `provision → prove → destroy`
session on live Amazon EKS
([PR 16 validation](proof/sprint-08-pr16-release-validation-evidence.md),
[batched live-EKS evidence](proof/sprint-08-live-eks-evidence.md)):

- **Terraform `Apply complete! Resources: 65 added`;** EKS ACTIVE, `v1.35.6-eks`,
  2 × t3.large nodes Ready across 2 AZs.
- **Dataset retrieved from S3** via Pod Identity (no static keys), checksum verified.
- **Pipeline Job Complete, exit 0 — 5/5 stages `success=1`.**
- **MLflow run persisted** — metadata in PostgreSQL, artifacts in S3; model registered.
- **Prometheus: 11 targets UP; 3 Grafana dashboards** served live data.
- **Three failures injected, detected, and recovered** under runbook guidance (§9).
- **Final healthy state restored** — 0 alerts firing, 16 runs persisted, `pg_up=1`.
- **`terraform destroy` symmetric — 65 destroyed;** verified clean three ways
  (Terraform state 0 resources, `aws eks` empty, KMS keys scheduled for deletion);
  nothing left billing.

The validation environments were **short-lived and ephemeral by design** — this is
stated openly, not hidden. The Sprint 8 release gate audited 23 proof dimensions
against this and prior runs and returned **PASS**
([release gate](proof/sprint-08-release-gate.md)). The complete claim-to-proof map,
with an honest strength label on each item, is the
[Evidence Index](proof/README.md).

## 14 · Cost / Cleanup Discipline

Cost discipline is a first-class design property, not an afterthought. The lifecycle
is ephemeral: **provision → prove → destroy the same session**
([ADR-020](decisions/ADR-020-cloud-lifecycle-cost-control.md)). The incremental cost
drivers (EKS control plane, 2 × t3.large, NAT, EBS) were recorded during the run
([PR 16 §17](proof/sprint-08-pr16-release-validation-evidence.md)), and teardown is
treated as part of the proof, not a formality: `Destroy complete! Resources: 65
destroyed` is symmetric with the 65 added, with no orphaned resources
([PR 16 teardown](proof/sprint-08-pr16-release-validation-evidence.md)); the
exploratory campaign additionally verified clean state three independent ways
([live-EKS §8](proof/sprint-08-live-eks-evidence.md#8-teardown)). A platform you cannot
prove you destroyed is a platform that is still costing money.

## 15 · Limitations

The project is explicit about what it does **not** prove; the honesty boundary is part
of the deliverable. It does **not** claim, and its evidence does not support:

- Enterprise SRE, formal SLA/SLO, or 24/7 on-call operations
- Multi-region high availability or disaster recovery
- Service mesh
- Model serving / online inference at scale
- GitOps or Terraform remote state / state locking (intentionally deferred)
- Distributed tracing
- A fully signed & attested supply chain (SBOM + digest yes; cosign/SLSA deferred)
- Client / production outcomes

The validation cluster was **single-operator, two nodes, and short-lived** —
sufficient to prove the operational model, not production scale. These boundaries are
tracked openly in the [Roadmap](roadmap.md), the [Evidence Index §11](proof/README.md)
and each sprint's release gate.

## 16 · What This Demonstrates

Translated from technical artifacts into professional capability, this repository is
evidence of:

- **Platform-engineering judgment** — choosing the Job model, the Terraform/Kustomize
  boundary, and a deliberately bounded scope, and defending each with a trade-off.
- **Kubernetes operations** — workload modeling, security context, NetworkPolicy, and
  diagnosing failures that only appear under live enforcement.
- **Cloud infrastructure as code** — a reproducible, destroyable AWS foundation with
  verified symmetric teardown.
- **MLOps runtime design** — cloud-backed data, in-cluster experiment tracking, and
  durable state on PostgreSQL + S3.
- **Security engineering** — workload identity over static keys, KMS encryption,
  least-privilege IAM, and a hardened runtime.
- **Observability** — signal design that localizes faults, keyed to operator action.
- **Reliability engineering** — evidence-driven hardening: break it first, fix what
  the failure justifies.
- **Evidence-driven debugging** — four real, static-invisible defects found and fixed
  under live enforcement.
- **Trade-off reasoning and intellectual honesty** — every claim scoped to its
  evidence, and every deferred capability named rather than hidden.

## 17 · Evidence / References

- **[Evidence Index](proof/README.md)** — every claim mapped to its canonical proof
  and proof strength (the reviewer's entry point).
- **[Sprint 8 Release Gate — PASS](proof/sprint-08-release-gate.md)** — 23-dimension
  go/no-go audit.
- **[Live-EKS Validation (PR 16)](proof/sprint-08-pr16-release-validation-evidence.md)**
  — the authoritative `provision → prove → destroy` record.
- **[Batched Live-EKS Evidence](proof/sprint-08-live-eks-evidence.md)** — the 4 real
  defects the live run surfaced, all fixed.
- **[Architecture](architecture.md)** · **[ADR-001 … ADR-037](decisions/README.md)**
  — system design and decision records.
- **[Runbooks](runbooks/README.md)** · **[Roadmap](roadmap.md)** ·
  **[CHANGELOG](../CHANGELOG.md)** · **[repository README](../README.md)**.

---

<sub>Portfolio-scoped platform-engineering case study. All runtime evidence comes from
controlled, ephemeral validation sessions on real Amazon EKS in the operator's own AWS
account — provisioned, proven, and destroyed the same session. Claims stop where the
evidence stops.</sub>
