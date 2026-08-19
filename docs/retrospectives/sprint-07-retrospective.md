# Sprint 7 — Retrospective (v1.4.0)

- **Date:** 2026-08-19
- **Release:** `v1.4.0` — Cloud-Native MLOps Hardening (pending; consolidates the
  `[Unreleased]` Sprint 5–7 work — the tag is a separate, explicitly-requested step
  gated by the [release gate](../proof/sprint-07-release-gate.md)).
- **Scope:** Take the Sprint 6 cloud platform — a Terraform-defined EKS environment the
  pipeline had run on **once** — and **close every HIGH/MEDIUM finding** from the Sprint
  6 review, hardening it into a defensible cloud-native MLOps platform: Terraform-managed
  ECR, a private-by-default EKS API, explicit access entries, EKS Pod Identity for the
  CNI and both app workloads, KMS-encrypted Secrets, an in-cluster MLflow platform, and
  an S3 runtime dataset path — then prove the whole thing on real EKS and gate the
  release honestly.
- **Companion:** [Sprint 5 — Retrospective](sprint-05-retrospective.md),
  [Sprint 7 Proof-Impact](../proof/sprint-07-proof-impact.md),
  [Sprint 7 Runtime Evidence](../proof/sprint-07-runtime-evidence.md),
  [Sprint 7 Release Gate](../proof/sprint-07-release-gate.md),
  [Cloud Operations](../cloud-operations.md),
  [ADR-021](../decisions/ADR-021-terraform-managed-ecr.md)–[ADR-027](../decisions/ADR-027-s3-dataset-runtime-retrieval.md)

This is a look-back on Sprint 7: what was planned, what shipped, what changed during
implementation, the problems hit, the decisions behind them, and what was deliberately
left for later.

---

## 1. Planned

Sprint 7 was scoped directly from the Sprint 6 review's HIGH/MEDIUM findings, plus the
data-architecture debt (ConfigMap dataset, external tracking):

- **H-01** — bring ECR under Terraform (it had been created out-of-band).
- **H-02** — make the EKS API private by default and structurally refuse `0.0.0.0/0`.
- **H-03** — replace implicit creator-admin with explicit, scoped access entries.
- **M-01** — move the VPC CNI off the node instance role onto its own identity.
- **M-02** — envelope-encrypt Kubernetes Secrets with a customer-managed KMS key.
- **M-03** — replace external DagsHub tracking with an in-cluster MLflow platform.
- **M-04** — deliver the dataset from private object storage, not a ConfigMap.
- **Prove** the hardened platform end-to-end on real EKS, and **gate** the release
  with honest evidence.

Non-goals (explicit): GitOps, Terraform remote state, multi-region, HA/DR, production
observability, model serving.

---

## 2. Delivered

Every planned item shipped. Mapped to its PR and design record:

| Area | Delivered | Design record |
|---|---|---|
| ECR (H-01) | 2 Terraform-managed repos, immutable tags, scan-on-push, retention, `force_delete` | ADR-021 |
| EKS API (H-02) | `endpoint_public_access` default false; `0.0.0.0/0`/any `/0` rejected by validation + precondition | ADR-022 |
| Access model (H-03) | `authentication_mode = API`, creator-admin off, explicit access entries | ADR-023 |
| CNI identity (M-01) | dedicated CNI role via Pod Identity; CNI policy off the node role | ADR-024 |
| Secrets (M-02) | dedicated CMK (rotation on) wired into `encryption_config = ["secrets"]` | ADR-025 |
| MLflow (M-03) | in-cluster server + PostgreSQL + S3 artifacts, `--serve-artifacts`, ClusterIP | ADR-026 |
| Dataset (M-04) | private/SSE-KMS/versioned S3 bucket; `fetch-dataset` init container with sha256 pin | ADR-027 |
| DVC correctness | declared DAG reconciled with traced execution; contract tests | — |
| Runtime proof | full-platform run on real EKS 1.35, **Job exit 0**, destroyed clean | [runtime evidence](../proof/sprint-07-runtime-evidence.md) |
| Offline contracts | `terraform test` **42/42**; `k8s/validate.py` extended to **158/158** (fleet-wide) | PR 11 |
| Docs reconciliation | ~14 docs reconciled to the shipped platform | PR 12 |
| Release gate | this sprint's verdict, blockers, defensible claims | [release gate](../proof/sprint-07-release-gate.md) |

**Result:** 7/7 HIGH/MEDIUM findings closed, each with source + offline contract +
live-runtime evidence.

---

## 3. What changed during implementation

- **Tracking replacement grew into a platform.** "Stop using DagsHub" became a full
  three-tier in-cluster MLflow (stateless server + PostgreSQL StatefulSet + S3 artifact
  store with `--serve-artifacts`), so clients stay credential-free and all S3 access sits
  behind the server's Pod Identity. That is more than the finding demanded but is what
  makes the tracking path actually self-hosted and persistent.
- **A second `apply` for operator access.** Because creator-admin is off (H-03) and the
  API is private by default (H-02), reaching the cluster from a workstation needed a
  deliberate, git-ignored opt-in (scoped `/32` + an explicit access entry). The proof
  run records this as `2 added, 1 changed` — a documented operator step, not an
  architecture change.
- **A storage-class fix surfaced only on EKS.** The portable base leaves
  `storageClassName` unset (cluster default); EKS 1.35 ships `gp2` but marks *no*
  default, so the Postgres PVC stalled until a surgical `gp2` patch was added to the AWS
  overlay. Caught only by actually running on EKS.
- **The security contract went fleet-wide.** `k8s/validate.py` was extended (PR 11) from
  per-workload spot-checks to **every** pod-bearing workload including init containers,
  plus namespace-level PSA `enforce: restricted` labels.

---

## 4. Problems encountered

- **Private-by-default is inconvenient by design.** The first `kubectl` timed out — the
  intended consequence of H-02 + H-03. Resolved by the documented scoped opt-in, not by
  weakening the default.
- **Cold-cluster first-boot race.** MLflow's first DB migration occasionally exceeds the
  `wait-for-mlflow` init budget on a cold cluster; the Job's `backoffLimit` retry
  absorbs it (`failed: 1, succeeded: 1`). Left as-is — the retry semantics already
  guarantee completion; a larger init budget is a noted tuning opportunity.
- **`terraform test` reads the operator's local tfvars.** On the workstation the suite
  shows 41/1 because the git-ignored opt-in tfvars flips the "private-by-default"
  assertion; it is **42/42** in a clean checkout. Diagnosed in the
  [release gate §6](../proof/sprint-07-release-gate.md#6-the-terraform-test-observation-non-blocking).
- **Local tooling gaps at gate time.** `tflint`, `trivy`, and `kubeconform` were not
  runnable in the gate environment; their coverage was delegated to CI and the offline
  contract suites, and made an explicit release condition rather than silently skipped.

---

## 5. Engineering decisions

- **Prove the contract offline, then once live.** Every security property is pinned by a
  credential-free `terraform test` (mock provider, `command = plan`) so CI enforces it on
  every push; the live run confirms it once, then the environment is destroyed (ADR-020).
- **Isolation at the IAM boundary via Pod Identity**, not IRSA/OIDC juggling — one addon,
  dedicated least-privilege roles, no static keys (ADR-024).
- **Server-mediated artifacts** (`--serve-artifacts`) so only the MLflow server holds S3
  access; pipeline clients need no cloud credentials (ADR-026).
- **Integrity-pinned dataset** — the init container verifies sha256 against a pinned
  identity before training, making the S3 path tamper-evident (ADR-027).
- **Honest, conditional release gating** — a CONDITIONAL PASS that names its two
  conditions (green CI, captured-proof acceptance) rather than a PASS that overstates
  what this environment could verify.

---

## 6. What went well

- **100% finding closure** with layered evidence — nothing rests on source code alone.
- **The offline contract suite paid off** — 42 `terraform test` assertions caught posture
  regressions with no AWS account, and cleanly explained the one workstation-only failure.
- **The live run found the real bug** (the `gp2` default gap) that no static check could.
- **Clean teardown** — `destroy` verified three ways, nothing billable left behind.
- **The reconciliation (PR 12) paid forward** — docs already matched the platform, so the
  gate had a trustworthy baseline to assess.

---

## 7. What was deliberately deferred

Named so none is mistaken for delivered:

- **GitOps / continuous delivery** — deployment is operator `kubectl apply`; CI validates,
  never deploys. (Verified absent — no Argo/Flux/Fleet anywhere.)
- **Terraform remote state** — local state by design (ADR-014); a remote S3 + lock backend
  is a documented migration path. (Verified absent — no `backend`/`cloud` stanza.)
- **Multi-region / DR** — one region, no backup/restore, no RTO/RPO.
- **Enterprise HA** — single node, single NAT, single-writer PostgreSQL.
- **Production observability** — `kubectl` + structured logs only; no Prometheus/Grafana/
  tracing/alerting.
- **Read-only root filesystem & Restricted-PSS runtime certification** — controls applied
  and statically satisfied; not observed under live `enforce` admission (ADR-010).
- **DVC data/model remote off DagsHub** — only the *tracking* path moved in-cluster; the
  `.dvc/config` data remote is a separate versioning concern (roadmap v5).
- **Registry customer-CMK encryption** — ECR uses AES256; a scoped follow-up.

---

## 8. Lessons learned

- **Secure defaults have an operational tax — budget for it.** Private-by-default plus
  no-creator-admin is correct *and* means access is a deliberate, documented step. Ship
  the runbook with the control.
- **Portability assumptions break at the managed-cluster boundary.** "Use the cluster
  default StorageClass" is portable right up until a cluster has a class but no default.
  Env-specific reality belongs in the overlay.
- **Tests that read ambient config can mislead locally.** A contract test that auto-loads
  an operator's tfvars can report a scary-looking failure that is pure local state; assert
  on explicit inputs, and diagnose before alarming.
- **A gate is only as honest as its stated limits.** Recording *what could not be run
  here* (three linters, no live cluster) and turning those into named release conditions
  is what makes the PASS credible — masking them would not.
- **Layer evidence: source → offline contract → live run.** Each layer catches what the
  others cannot; together they are what let this sprint claim closure defensibly.

---

## Related documentation

- [Sprint 7 Release Gate](../proof/sprint-07-release-gate.md) ·
  [Sprint 7 Proof-Impact](../proof/sprint-07-proof-impact.md) ·
  [Sprint 7 Runtime Evidence](../proof/sprint-07-runtime-evidence.md)
- [Cloud Operations](../cloud-operations.md) · [MLflow Platform](../mlflow-platform.md) ·
  [Dataset](../dataset.md) · [Versioning](../versioning.md)
- [Sprint 5 — Retrospective](sprint-05-retrospective.md) ·
  ADR-021 · ADR-022 · ADR-023 · ADR-024 · ADR-025 · ADR-026 · ADR-027
