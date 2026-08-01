# Engineering Philosophy

This repository is built and maintained around a small set of engineering
principles. They explain the *why* behind the structure and decisions recorded
elsewhere ([architecture](architecture.md),
[design principles](design-principles.md), [ADRs](decisions/)) and set
expectations for future contributions.

These principles describe the direction of the project. Some are already
realized; others are aspirations being worked toward through the
[roadmap](roadmap.md). Where a principle is not yet fully met, it is stated
honestly rather than overclaimed.

---

## 1. Reproducibility First

The same data, parameters, and code should always produce the same result.

- The pipeline is defined declaratively with DVC so stages re-run only when their
  inputs change ([ADR-003](decisions/ADR-003-why-dvc.md)).
- Parameters are externalized in `params.yaml`, never hard-coded in logic.
- Data and models are versioned, not just the code.

## 2. Configuration Over Code

Behavior that varies between runs belongs in configuration, not in source.

- Pipeline inputs, outputs, and hyperparameters live in `params.yaml`.
- Credentials and environment-specific values come from the environment
  (`.env`), keeping code portable.

## 3. Separation of Concerns

Each piece of the system has one clear responsibility.

- One module per pipeline stage (`preprocess`, `train`, `evaluate`).
- Code lives in Git; data and models live in DVC; secrets live in the
  environment.
- Documentation is a first-class citizen under `docs/`.

## 4. Observability of Experiments

If an experiment isn't tracked, it didn't happen.

- Every training run records parameters, metrics, and artifacts to MLflow
  ([ADR-002](decisions/ADR-002-why-mlflow.md)).
- Results are stored on a shared, hosted tracker so they are comparable and
  reviewable, not trapped on one machine.

## 5. Security & Secret Hygiene

Secrets never enter version control.

- Credentials are provided via `.env`; only `.env.example` (a template) is
  committed.
- The repository ignores sensitive and generated artifacts by default.

## 6. Documented Decisions

Significant choices are recorded, with their alternatives and trade-offs.

- Architecture Decision Records ([`decisions/`](decisions/)) capture context,
  decision, alternatives, and consequences.
- Decisions are revisited as the project matures rather than treated as
  permanent.

## 7. Honest Over Impressive

Documentation reflects reality, including known gaps.

- Where the current implementation is incomplete or inconsistent, it is marked
  with an explicit `TODO` instead of being described as if finished (see the
  known gaps noted in [architecture.md](architecture.md)).
- We prefer accurate, plain descriptions over marketing language.

## 8. Incremental, Roadmapped Evolution

The project grows in deliberate stages.

- The [roadmap](roadmap.md) sequences improvements from engineering quality
  through CI/CD, Kubernetes, cloud, and enterprise MLOps.
- Each stage delivers a coherent increment rather than a large, risky rewrite.

## 9. Readability & Maintainability

Code and docs are written to be understood by the next person.

- A new contributor should grasp the repository in about ten minutes (see
  [project-structure.md](project-structure.md)).
- Consistent naming and formatting are enforced by shared conventions
  (`.editorconfig`; a formatter/linter toolchain is planned in Roadmap v2).

---

## Aspirations (Not Yet Fully Realized)

Stated explicitly so the philosophy isn't mistaken for current state:

- **Automated quality gates** (tests + linting in CI) — planned in Roadmap v3.
  <!-- TODO: revisit once CI is in place. -->
- **Test coverage** — no automated tests exist yet; targeted in Roadmap v2.
- **Production operability** (monitoring, drift detection, retraining) — planned
  in Roadmap v6.

---

## Related Documentation

- [Design Principles](design-principles.md)
- [Architecture](architecture.md)
- [Project Structure](project-structure.md)
- [Roadmap](roadmap.md)
- [Architecture Decision Records](decisions/)
