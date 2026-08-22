# Architecture Visuals

A small, high-signal set of diagrams for a **cold technical reviewer** — enough to
understand the platform quickly without reading every implementation doc. Each
diagram is source-controlled **Mermaid** (renders inline on GitHub) with an ASCII
fallback, a caption stating what it proves, and its limitations.

Every diagram is checked against the actual implementation (`terraform/`, `k8s/`,
the security/observability/supply-chain docs, and the live-EKS runtime evidence). No
deferred system is drawn as if it were built — there is deliberately **no** GitOps /
ArgoCD, service mesh, remote Terraform state, model-serving stack, distributed
tracing, or HA/DR anywhere in this package.

## The package

| # | Diagram | One-line purpose | Status |
|---|---|---|---|
| A | [System / Final Platform Architecture](system-architecture/) | Commit → verified run on EKS, with CI / operator / Terraform / runtime ownership boundaries. | ✅ Validated (v1.7.0) |
| B | [Observability Architecture](observability-architecture/) | Which operational signals reach Prometheus/Grafana — and why experiment metrics stay in MLflow, not Prometheus. | ✅ Validated (v1.7.0) |
| C | [Security Architecture](security-architecture/) | Nested trust boundaries from the AWS account to the container syscall surface. | ✅ Implemented |
| D | [Failure / Recovery Loop](failure-recovery/) | Controlled failure → signal → alert → runbook → remediation → verified health, with three real examples. | ✅ Exercised on live EKS |
| E | [Supply-Chain Provenance](supply-chain-provenance/) | Git commit → build → scan → SBOM → ECR tag → sha256 digest → running pod imageID. | ✅ Implemented |
| F | [Platform Evolution](platform-evolution/) | The versioned story from a course-style local pipeline to a cloud-native, observable platform. | ✅ v1.0 → v1.7 |

## Other diagrams in the repo

- [`kubernetes-architecture/`](kubernetes-architecture/) — the batch-workload (`batch/v1` Job) flow, discussed in [kubernetes-architecture.md](../kubernetes-architecture.md).
- [`pipeline-flow/`](pipeline-flow/), [`deployment-architecture/`](deployment-architecture/), [`cicd-flow/`](cicd-flow/) — reserved; the flows they would show are covered by the package above and by [architecture.md](../architecture.md).

## Where these are embedded

To keep each document readable, only the **overview** (A) and the **failure loop**
(D) are embedded inline where they add the most value; the rest are linked from
here:

- [README.md](../../README.md) — embeds **A**.
- [case-study.md](../case-study.md) — embeds **A** (§ 5) and **D** (§ 9).
- [architecture.md](../architecture.md) — embeds **A** (§ 2).
