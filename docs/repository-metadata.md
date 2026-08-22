# Repository Metadata Recommendations

This document recommends presentation and discoverability settings for the
repository. **It documents recommendations only** — no GitHub settings should be
changed as part of this document. Apply these manually in the repository
settings when ready (see the
[Repository Rename Checklist](sprint-09/repository-rename-checklist.md)).

---

## Canonical Identity

The repository's final identity, selected in the
[Repository Naming Evaluation](sprint-09/repository-naming-evaluation.md):

| Field | Value |
|-------|-------|
| **Canonical slug** | `mlops-platform-on-eks` |
| **Human-readable title** | MLOps Platform on AWS EKS |
| **Positioning (one line)** | Cloud-native MLOps platform engineering on AWS EKS — from Terraform-provisioned infrastructure to in-cluster MLflow, observability, and controlled failure/recovery testing. |

The original slug (`MLOps-Project-01-End-to-End-Machine-Learning-Pipeline`) is
retired; GitHub preserves a permanent redirect from it after the rename.

---

## Repository Description (GitHub "About")

The About text appears at the top of the repository and in search results.
Recommended:

> Cloud-native MLOps platform engineering on AWS EKS: Terraform IaC, in-cluster
> MLflow (PostgreSQL + S3), Pod Identity / IRSA workload identity, Prometheus &
> Grafana observability, controlled failure/recovery testing, and supply-chain
> controls. A portfolio-scoped platform-engineering proof — not a production
> service.

If a shorter form is needed where space is tight (under ~120 characters):

> Cloud-native MLOps platform-engineering proof on AWS EKS (Terraform, EKS,
> MLflow, observability).

The positioning deliberately does **not** imply enterprise production scale, 24/7
operations, formal SRE maturity, model serving at scale, or multi-region
resilience — the repository does not prove those.

## GitHub Topics

Topics improve discoverability. Recommended set (reflecting the platform scope):

`mlops`, `platform-engineering`, `aws`, `eks`, `kubernetes`, `terraform`,
`mlflow`, `dvc`, `observability`, `prometheus`, `grafana`, `workload-identity`,
`supply-chain-security`, `infrastructure-as-code`, `python`

> Add only topics that remain accurate as the project evolves; remove any that
> no longer apply.

## Social Preview Image Ideas

The social preview image is shown when the repository is shared on social media
and in link previews. Ideas:

- A clean title card: project name + one-line description on a simple background.
- A simplified pipeline diagram (preprocess → train → evaluate) with the DVC and
  MLflow logos.
- A high-level architecture snapshot once the
  [architecture diagram](diagrams/system-architecture/) exists.

> <!-- TODO: produce the social preview image (recommended 1280×640) and store
> the source under docs/diagrams/ or docs/screenshots/. -->

## Repository Website (Future)

A project website can host rendered documentation.

- Consider GitHub Pages built from the `docs/` directory.
- Candidate generators: MkDocs (Material) or Docusaurus.

> <!-- TODO: decide whether/when to publish docs as a site (see roadmap). No
> action required now. -->

## Pinned Repository Rationale

If pinned to the owner's GitHub profile, this repository demonstrates:

- **Cloud-native platform engineering** — Terraform-provisioned AWS/EKS
  infrastructure, workload identity, in-cluster MLflow, and observability.
- **MLOps and ML engineering** — reproducible pipelines, experiment tracking,
  data/model versioning.
- **Operational rigor** — controlled failure/recovery testing, runbooks, and
  supply-chain controls (SBOM + provenance).
- **Software engineering maturity** — documentation, ADRs, governance, and
  contribution standards.

This makes it a strong candidate for a pinned, flagship portfolio project.

---

## Related Documentation

- [Repository Naming Evaluation](sprint-09/repository-naming-evaluation.md)
- [Repository Rename Checklist](sprint-09/repository-rename-checklist.md)
- [Documentation Index](README.md)
- [Roadmap](roadmap.md)
- [Engineering Philosophy](philosophy.md)
