# ADR-001: Repository Structure

- **Status:** Accepted
- **Date:** 2026-08-01
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

The repository uses a **flat, stage-oriented layout** with one Python module per
pipeline stage under `src/`, declarative configuration at the repository root,
and DVC-managed data/model directories:

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

The design rests on three choices:

- **One module per stage** keeps each step small and independently runnable
  (`python src/train.py`) and aligns 1:1 with DVC stages in `dvc.yaml`.
- **Root-level configuration** (`params.yaml`, `dvc.yaml`) makes parameters and
  the pipeline graph easy to find.
- **`docs/` as a first-class directory** signals that documentation is maintained
  alongside code.

## Alternatives Considered

1. **Installable Python package** (`src/<package>/` with `pyproject.toml`,
   modules imported rather than executed as scripts).
   - *Pros:* better testability, reuse, packaging.
   - *Cons:* more ceremony than a script-per-stage pipeline needs today.
   - *Decision:* not adopted. The Roadmap v2 testing foundation was delivered
     without a package layout: the `pytest` suite imports the stages by their
     bare module names via `pythonpath = ["src"]` (see
     [testing-strategy.md](../testing-strategy.md)), so the script-per-stage
     structure was kept. A package layout remains a candidate only if import or
     reuse pressure appears later.
2. **Notebook-driven structure** (Jupyter notebooks as the primary artifact).
   - *Decision:* rejected — poor reproducibility, hard to version and automate.
3. **Framework-imposed layout** (e.g., Kedro or Cookiecutter Data Science).
   - *Pros:* conventions out of the box.
   - *Cons:* additional dependency and learning curve for a small pipeline.
   - *Decision:* deferred; worth revisiting if the project grows substantially.

## Consequences

**Positive**

- Low cognitive overhead; a new contributor can map the repository to the
  pipeline in minutes.
- Clean DVC integration (stage ↔ module).
- Clear separation of code, config, data, and docs.

**Trade-offs and follow-ups**

- Scripts are executed directly rather than imported. The Roadmap v2 testing
  foundation addressed the testability gap without a package layout, by putting
  `src/` on the path for the test runner (`pythonpath = ["src"]`) so tests
  import the stages the same way the interpreter does at runtime.
- A flat `src/` will not scale well if many stages or utilities are added later;
  it will be revisited when that pressure appears.
- A `pyproject.toml` now exists, but only as central configuration for the
  tooling (Ruff, mypy, pytest — see
  [ADR-004](ADR-004-python-quality-toolchain.md)); it declares no packaging
  metadata, so the pipeline remains a set of runnable scripts rather than an
  installable distribution.
