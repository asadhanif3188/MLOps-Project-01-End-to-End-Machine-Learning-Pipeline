# ADR-001: Repository Structure

- **Status:** Accepted (first draft — describes the current structure)
- **Date:** <!-- TODO: set the date this decision was formally recorded -->
- **Deciders:** Asad Hanif
- **Related:** [ADR-002 (MLflow)](ADR-002-why-mlflow.md), [ADR-003 (DVC)](ADR-003-why-dvc.md), [project-structure.md](../project-structure.md)

## Context

The project is an end-to-end ML pipeline that must be **reproducible** and easy
for a new contributor to understand quickly. It needs a layout that:

- separates pipeline logic (code) from configuration and data,
- maps cleanly onto discrete, individually runnable pipeline stages
  (preprocess → train → evaluate),
- integrates with DVC (stage definitions and tracked outputs) and MLflow, and
- keeps large data/model artifacts out of Git.

## Decision

Adopt a **flat, stage-oriented layout** with one Python module per pipeline
stage under `src/`, declarative configuration at the repository root, and
DVC-managed data/model directories:

```text
.
├── src/            # one module per pipeline stage (preprocess, train, evaluate)
├── data/           # raw/ and processed/ datasets (DVC-tracked, not in Git)
├── models/         # serialized model artifacts (DVC-tracked, git-ignored)
├── docs/           # architecture, roadmap, ADRs, diagrams, screenshots
├── params.yaml     # declarative pipeline parameters
├── dvc.yaml        # DVC stage definitions (deps, params, outs)
└── requirements.txt
```

Rationale:

- **One module per stage** keeps each step small and independently runnable
  (`python src/train.py`) and aligns 1:1 with DVC stages in `dvc.yaml`.
- **Root-level config** (`params.yaml`, `dvc.yaml`) makes parameters and the
  pipeline graph easy to find.
- **`docs/` as a first-class directory** signals that documentation is
  maintained alongside code.

## Alternatives Considered

1. **Installable Python package** (`src/<package>/` with `pyproject.toml`,
   modules imported rather than executed as scripts).
   - *Pros:* better testability, reuse, packaging.
   - *Cons:* more ceremony than a script-per-stage pipeline needs today.
   - *Rejected for now;* revisit as an engineering-quality improvement.
   <!-- TODO: reconsider in Roadmap v2 (Engineering Improvements). -->
2. **Notebook-driven structure** (Jupyter notebooks as the primary artifact).
   - *Rejected:* poor reproducibility, hard to version and automate.
3. **Framework-imposed layout** (e.g., Kedro/Cookiecutter Data Science).
   - *Pros:* conventions out of the box.
   - *Cons:* additional dependency and learning curve for a small pipeline.
   - *Deferred.* <!-- TODO: evaluate a cookiecutter-style template if the project grows. -->

## Consequences

**Positive**

- Low cognitive overhead; a new contributor can map the repo to the pipeline in
  minutes.
- Clean DVC integration (stage ↔ module).
- Clear separation of code, config, data, and docs.

**Negative / trade-offs**

- Scripts are executed directly rather than imported, which makes unit testing
  and reuse harder. <!-- TODO: address when tests are introduced (Roadmap v2). -->
- A flat `src/` will not scale well if many stages/utilities are added later.
- No packaging metadata yet (`pyproject.toml`/`setup.cfg`).
