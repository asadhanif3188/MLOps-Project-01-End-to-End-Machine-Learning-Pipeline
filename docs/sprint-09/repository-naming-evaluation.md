# Repository Naming Evaluation (Sprint 9, PR 1)

This document evaluates candidate names for the repository's final public
identity and recommends one. It is the naming rationale behind
[Repository Metadata Recommendations](../repository-metadata.md) and the
[Repository Rename Checklist](repository-rename-checklist.md).

## Why rename

The repository began as a course-style exercise and kept its original slug:

```
MLOps-Project-01-End-to-End-Machine-Learning-Pipeline
```

Through Sprints 1–8 it grew into a **cloud-native MLOps platform-engineering
proof** on AWS EKS: Terraform-provisioned infrastructure, in-cluster MLflow
(server + PostgreSQL + S3), EKS Pod Identity / IRSA workload identity, S3-backed
data, Prometheus/Grafana observability, controlled failure-and-recovery testing,
operational runbooks, and supply-chain controls (SBOM + provenance).

The words "Project-01" and "End-to-End Machine Learning Pipeline" now understate
the scope and read as a tutorial rather than platform-engineering work. The name
should signal what the repository actually demonstrates — without overclaiming
production maturity.

## Constraints on the name

The name must **not** imply things the repository does not prove:

- enterprise production scale
- 24/7 operations or formal SRE maturity
- model serving at scale
- multi-region resilience

It **should** signal: MLOps, platform engineering, and AWS/EKS as the cloud
substrate. Prefer clarity over cleverness.

## Rubric

Each candidate is scored 1–10 on eight dimensions (higher is better):

| Dimension | Meaning |
|-----------|---------|
| Prof. credibility | Reads as serious engineering work to a knowledgeable reviewer |
| Platform signal | Communicates platform-engineering (infra + operations), not just a model |
| MLOps relevance | Communicates the MLOps discipline (lifecycle, tracking, reproducibility) |
| Cloud relevance | Communicates the cloud/AWS substrate |
| Memorability | Easy to recall and say |
| URL readability | Reads cleanly as a lowercase, hyphenated slug |
| Buzzword restraint | Avoids empty marketing words |
| Honesty of scope | Doesn't imply scale/maturity beyond the evidence |

## Candidate matrix

| Candidate | Prof. cred. | Platform | MLOps | Cloud | Memorable | URL | Buzzword restraint | Honesty | **Total /80** |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **mlops-platform-on-eks** | 9 | 8 | 9 | 8 | 8 | 9 | 9 | 9 | **69** |
| aws-eks-mlops-platform | 8 | 8 | 9 | 9 | 7 | 7 | 8 | 8 | 64 |
| mlops-on-eks | 8 | 6 | 9 | 8 | 9 | 9 | 9 | 9 | 67 |
| ml-platform-engineering | 9 | 9 | 6 | 3 | 8 | 8 | 9 | 7 | 59 |
| cloud-native-mlops-platform | 8 | 8 | 9 | 7 | 7 | 7 | 5 | 7 | 58 |
| cloud-ml-platform | 7 | 6 | 5 | 7 | 7 | 8 | 7 | 6 | 53 |

### Per-candidate notes

- **mlops-platform-on-eks** — Carries all three signals (MLOps + platform + the
  EKS substrate), reads as a natural phrase ("MLOps platform on EKS"), and "on
  eks" is specific and modest rather than grandiose. Best URL readability and the
  strongest honesty-of-scope score. **Recommended.**
- **aws-eks-mlops-platform** — Equally accurate and slightly more explicit about
  AWS, but the four stacked nouns read as keyword-stuffing and the slug is
  clunkier to say.
- **mlops-on-eks** — Cleanest and most memorable, but drops the "platform"
  signal, which is a core part of what the repo demonstrates. Strong runner-up if
  a shorter slug is preferred.
- **ml-platform-engineering** — Excellent platform/engineering signal but loses
  the *Ops* discipline (reads as generic ML infra) and carries **no** cloud
  signal, which undersells the EKS/Terraform work.
- **cloud-native-mlops-platform** — Accurate but "cloud-native" is the weakest
  buzzword-restraint term, and without "EKS" the cloud claim is vaguer.
- **cloud-ml-platform** — Too broad; reads like a product name and could imply a
  hosted service (dishonest for a portfolio proof).

## Recommendation

**`mlops-platform-on-eks`**

It is the clearest honest expression of the repository's scope: an MLOps
platform, built and operated on AWS EKS, demonstrated end-to-end. It signals
MLOps, platform engineering, and the cloud substrate without implying production
scale or SRE maturity.

Human-readable title: **MLOps Platform on AWS EKS**.

See [Repository Metadata Recommendations](../repository-metadata.md) for the full
final identity (slug, title, positioning, GitHub About, and topics) and the
[Repository Rename Checklist](repository-rename-checklist.md) for the manual
GitHub rename steps.
