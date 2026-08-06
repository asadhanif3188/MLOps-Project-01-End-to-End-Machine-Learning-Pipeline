# ADR-006: Pipeline Reproducibility as an Engineering Requirement

- **Status:** Accepted — implemented in Sprint 4 (v1.3.0), except a held-out
  evaluation split and a committed `dvc.lock` (tracked as open deviations D5 / D7
  in the [pipeline contract](../pipeline-contract.md#11-deviation-status-sprint-4))
- **Date:** 2026-08-05 (design ratified); implementation landed 2026-08-06
- **Deciders:** Asad Hanif
- **Related:** [Pipeline Contract](../pipeline-contract.md),
  [ADR-003 (Why DVC)](ADR-003-why-dvc.md),
  [ADR-002 (Why MLflow)](ADR-002-why-mlflow.md),
  [ADR-001 (Repository Structure)](ADR-001-repository-structure.md),
  [architecture.md](../architecture.md)

> **Scope note.** This record ratifies a **design contract**, not a completed
> implementation. It is the decision layer behind
> [pipeline-contract.md](../pipeline-contract.md) and governs the remaining
> Sprint 4 PRs (correct DVC graph, config/data contracts, ML-stage refactor,
> stage tests, CI reproducibility). **No implementation code is changed in the PR
> that introduces this ADR.** Statements about current behavior describe the
> repository as it exists today; statements about required behavior describe the
> target the following PRs must meet.

> **Implementation status (updated 2026-08-06).** The Sprint 4 implementation PRs
> have since landed. The "Context" below describes the repository *before* those
> PRs and is retained as the historical record that motivated this decision; the
> decisions have been implemented, with the current state and the two remaining
> open items (in-sample evaluation, no `dvc.lock`) recorded in
> [pipeline-contract.md §11](../pipeline-contract.md#11-deviation-status-sprint-4).

## Context

The repository has strong engineering *infrastructure* around the ML system
(structure, quality toolchain, containerization, CI) but the ML *pipeline itself*
is not yet correctly modeled or reproducible. Verified today:

- The `preprocess` output (`data/processed/data.csv`) is **not consumed**
  downstream; `train` and `evaluate` both read `data/raw/data.csv` directly.
- `dvc.yaml` references `train.data`/`train.model` while `params.yaml` defines
  `train.input`/`train.output`, and `train.py` reads `input`/`output` — the
  parameter contract disagrees across three files.
- `train` and `evaluate` both call `require_env("MLFLOW_TRACKING_URI")` and connect
  to MLflow on DagsHub, so **neither stage can run — or be unit-tested — without an
  external service.**
- `evaluate` computes accuracy over the full training dataset (in-sample); there is
  no held-out evaluation set.
- Training is non-deterministic: `train_test_split` is called without a
  `random_state`, and the configured `train.random_state` is loaded but never
  applied. No `dvc.lock` is present, so pipeline state is not pinned.
- There is no metrics **artifact**; `accuracy` exists only in MLflow and the logs.

These are already recorded as known gaps in
[architecture.md](../architecture.md#3-pipeline-flow) and
[ADR-003](ADR-003-why-dvc.md#consequences). The consequence is that a technical
reviewer cannot answer "what depends on what, where does each artifact come from,
what configures it, and how is it validated?" without reading implementation
details — and cannot reproduce a run and get the same result.

We need a decision that (a) makes reproducibility a first-class, enforceable
property rather than an aspiration, and (b) fixes the ownership of lineage, the
explicitness of stage contracts, and the placement of external services — **before**
we start changing implementation.

## Decision

Treat **reproducibility and explicit stage contracts as engineering requirements**,
governed by [pipeline-contract.md](../pipeline-contract.md). Specifically:

1. **Reproducibility is a requirement, not a nicety.** A pipeline run must be
   reconstructable from declared inputs, parameters, and dependencies, and must be
   deterministic given fixed inputs, parameters, and seed. Seeds are read from
   `params.yaml` and actually applied. A committed `dvc.lock` pins resolved state.

2. **DVC owns pipeline and data lineage.** The DAG in `dvc.yaml` is the single
   source of truth for stage dependencies and artifacts and must model the real
   chain `raw → preprocess → processed → train → model → evaluate → metrics`.
   This continues the direction set in [ADR-003](ADR-003-why-dvc.md); ADR-006 is
   the correctness commitment ADR-003 anticipated.

3. **Stage contracts are explicit.** Every stage declares its inputs, outputs,
   configuration, and failure conditions (pipeline-contract §3). Each artifact has
   exactly one producing stage ("owner"); consumers never write it. Parameter names
   are authoritative in `params.yaml` and agree across `dvc.yaml` and the code, with
   no orphaned parameters.

4. **External services must not be required for ordinary unit tests.** MLflow/DagsHub
   is a **boundary**, isolated from core ML logic. Unit tests exercise preprocessing,
   training, evaluation, and configuration logic **without** live MLflow, network,
   or production credentials. Tests that genuinely require external services are
   **integration** tests, marked and segregated as such. Existing MLflow behavior is
   preserved, not removed, to make testing easier.

5. **The evaluation boundary is defined and honest.** The contract records what data
   trains, what data evaluates, how tuning is done, and which split is held out. No
   model-performance claim is made unless the methodology supports it; until a
   held-out split exists, reported accuracy is labeled in-sample.

6. **Reproducibility and contract integrity are enforced in CI.** Structural
   validation (valid DVC graph, consistent params, buildable pipeline) and
   stage-level tests run automatically, without production credentials or live
   external services.

This ADR is a **design contract**. The concrete edits to `dvc.yaml`, `params.yaml`,
the stage code, tests, and CI are delivered by the subsequent Sprint 4 PRs.

## Alternatives Considered

1. **Leave reproducibility implicit / "it runs on my machine."**
   - *Decision:* rejected — the project's stated purpose is a reproducible pipeline;
     non-deterministic training and an incorrect DAG directly undermine it, and the
     gaps are already visible to reviewers.

2. **Fix the code first, document the contract afterward (or never).**
   - *Decision:* rejected — the known issues (unused processed data, param mismatch,
     evaluation-on-training-data) are contract problems. Changing code without an
     agreed contract risks "over-refactoring" and re-introducing drift. Defining the
     contract first bounds the implementation PRs and gives them acceptance criteria.

3. **Use a different lineage/orchestration tool (MLflow Projects, a workflow engine,
   Pachyderm/LakeFS) instead of DVC.**
   - *Decision:* rejected/deferred — DVC is already adopted
     ([ADR-003](ADR-003-why-dvc.md)) and fits a batch, file-based pipeline. Swapping
     tools is heavier infrastructure than the correctness problem requires and is out
     of scope. MLflow remains for tracking ([ADR-002](ADR-002-why-mlflow.md)); the two
     stay complementary.

4. **Make MLflow/DagsHub a hard dependency of all tests (test against the real
   service).**
   - *Decision:* rejected as the default — coupling unit tests to a network service
     makes them slow, flaky, credential-dependent, and unrunnable in CI without
     secrets. Integration tests may exercise the real boundary explicitly; unit tests
     must not.

5. **Introduce a pipeline framework / abstraction layer to enforce contracts.**
   - *Decision:* rejected — "correctness over abstraction." The contract is enforced
     by explicit `dvc.yaml`/`params.yaml` wiring, typed errors, and tests, not by a
     new framework. This keeps the pipeline understandable, consistent with
     [ADR-001](ADR-001-repository-structure.md)'s stage-per-script simplicity.

6. **Enforce full end-to-end reproduction (including `dvc repro` against the remote)
   in CI.**
   - *Decision:* deferred — CI validates pipeline **structure and stage logic**
     without requiring the DagsHub remote or credentials. Full remote reproduction is
     a later-milestone concern; requiring it now would make CI depend on external
     infrastructure, which contradicts decision (4).

## Consequences

**Positive**

- A reviewer can trace lineage and configuration from the contract and `dvc.yaml`
  alone, and can reproduce a run deterministically.
- Stage contracts and single-owner artifacts remove ambiguity about what produces
  and consumes each file.
- ML logic becomes unit-testable without external services, so tests are fast,
  deterministic, and CI-friendly; MLflow behavior is preserved behind a boundary.
- CI can fail a PR that breaks the DAG, the parameter contract, or a stage contract —
  turning correctness into an automated guarantee.
- The claim the project can defensibly make strengthens from "built a pipeline with
  DVC/MLflow" to "engineered and validated a reproducible pipeline with explicit
  lineage, contracts, and CI enforcement."

**Trade-offs and follow-ups**

- **Design-only in this PR.** The current code still exhibits every gap listed in
  Context; this ADR and the contract describe the target, and the fixes land in
  Sprint 4 PRs 2–6. Until then, the repository intentionally documents-but-does-not-fix
  these issues (consistent with [architecture.md](../architecture.md)).
- **Accepted current limitations (explicitly):** in-sample evaluation, unused
  `data/processed/data.csv`, inert `train.*` hyperparameters, no `dvc.lock`, and
  MLflow-coupled stages remain true *today*; they are tracked as deviations D1–D8 in
  [pipeline-contract.md](../pipeline-contract.md#11-current-deviations-summary-current--target),
  each assigned to a later PR.
- **Determinism has a cost.** Fixing seeds and consuming the processed dataset may
  change the specific model and metric values relative to today's runs; this is
  expected and acceptable — correctness and reproducibility take priority over
  preserving a particular incidental result.
- **Out of scope for Sprint 4** (deferred to later milestones): Kubernetes, Helm,
  Terraform, cloud deployment, continuous delivery, model serving, Prometheus/Grafana,
  distributed training, production MLflow infrastructure, container registry, Trivy,
  SBOM, and image signing.
- **CI does not prove full reproduction** against the remote; it validates structure
  and stage logic. Full end-to-end remote reproduction is a future concern (see
  Alternative 6).
</content>
