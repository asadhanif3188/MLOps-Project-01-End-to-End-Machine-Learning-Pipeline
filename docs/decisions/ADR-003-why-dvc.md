# ADR-003: Why DVC for Data & Pipeline Versioning

- **Status:** Accepted (first draft — describes the current implementation)
- **Date:** <!-- TODO: set the date this decision was formally recorded -->
- **Deciders:** Asad Hanif
- **Related:** [ADR-002 (MLflow)](ADR-002-why-mlflow.md), [architecture.md](../architecture.md)

## Context

Reproducibility is a core goal of the project. Git alone is a poor fit for
datasets and model binaries, and we also want a way to:

- version large data and model artifacts outside of Git,
- define the pipeline as an explicit dependency graph so stages re-run only when
  their inputs change, and
- store artifacts in a shared remote for collaboration.

## Decision

Use **DVC** to define and version the pipeline and its artifacts.

- The pipeline graph is declared in [`dvc.yaml`](../../dvc.yaml) with three
  stages (`preprocess`, `train`, `evaluate`), each specifying `deps`, `params`,
  and `outs`.
- Parameters are externalized in [`params.yaml`](../../params.yaml).
- Data and models are tracked by DVC (e.g., `data/raw/data.csv.dvc`) and kept
  out of Git (`models/` is git-ignored).
- The remote is an **S3-compatible endpoint on DagsHub** (configured in
  `.dvc/config`), installed via the `dvc-s3` extra.

This keeps data versioning and experiment tracking on the same platform
(DagsHub), consistent with [ADR-002](ADR-002-why-mlflow.md).

## Alternatives Considered

1. **Git LFS.**
   - *Pros:* simple, integrates with Git hosting.
   - *Cons:* no pipeline/stage orchestration or parameter tracking; weaker for ML
     reproducibility.
2. **Plain cloud storage + manual scripts (e.g., `aws s3 cp`).**
   - *Rejected:* no dependency graph, no reproducibility guarantees, error-prone.
3. **MLflow Projects / MLflow artifacts for everything.**
   - *Rejected:* MLflow is used for tracking; DVC is a better fit for
     data/pipeline versioning. The two are complementary here.
4. **Pachyderm or LakeFS.**
   - *Deferred:* heavier infrastructure than this project warrants today.

## Consequences

**Positive**

- Reproducible pipeline: `dvc repro` re-runs only stages whose dependencies
  changed.
- Large artifacts are versioned without bloating Git history.
- Shared remote enables collaboration; stack stays cohesive with MLflow on
  DagsHub.

**Negative / trade-offs**

- Additional tooling and mental model (`.dvc` files, remotes, `dvc repro`).
- Requires remote credentials/configuration to `pull`/`push` artifacts.
- **Current inconsistencies to resolve** (documented, not fixed in this
  documentation sprint):
  - `dvc.yaml` references params `train.data`/`train.model`, but `params.yaml`
    defines `train.input`/`train.output`.
  - The `train`/`evaluate` stages depend on `data/raw/data.csv`, so the
    `preprocess` output (`data/processed/data.csv`) is not consumed downstream.
  <!-- TODO: align dvc.yaml, params.yaml, and the stage scripts in Roadmap v2. -->
