# Sprint 4 — Retrospective (v1.3.0)

- **Date:** 2026-08-06
- **Release:** `v1.3.0` — Pipeline Correctness & Reproducibility
- **Scope:** Turn attention from the infrastructure *around* the ML pipeline to
  the pipeline itself — correct the DVC graph, make configuration consistent,
  decouple ML logic from MLflow, seed training, and enforce the pipeline contract
  in CI.
- **Companion:** [Sprint 3 — Retrospective](sprint-03-retrospective.md),
  [Sprint 4 — Final Validation](../reviews/sprint-04-final-review.md),
  [Pipeline Contract](../pipeline-contract.md),
  [ADR-006](../decisions/ADR-006-pipeline-reproducibility.md)

This is a look-back on Sprint 4: what was planned, what shipped, what changed
during implementation, the problems hit, the engineering decisions behind them,
and what was deliberately left for later. It records judgment and rationale — it
is not a validation gate.

---

## 1. Planned

Sprint 4 set out to close the highest-priority proof gap carried since Sprint 2:
the correctness and reproducibility of the ML pipeline itself. The plan was seven
PRs — a contract, a DVC correction, config/data contracts, an ML-stage refactor,
a test suite, CI enforcement, and a documentation/release wrap-up — driving toward
a defensible claim that the pipeline's lineage, contracts, and configuration are
correct, testable, and CI-validated.

Explicit guardrails from the plan: make the *smallest* change that establishes
correct contracts (no over-refactoring), **preserve** MLflow behavior rather than
delete it to ease testing, and keep CI free of production credentials and live
external services.

## 2. Delivered

Shipped as **`v1.3.0`** across six feature PRs plus this release PR, each
branch → PR → merge to `main`:

| PR | Branch | Delivered |
|----|--------|-----------|
| #17 | `feature/sprint-04-pipeline-contract` | `pipeline-contract.md` + ADR-006 (design contract) |
| #18 | `feature/sprint-04-dvc-pipeline` | Corrected `dvc.yaml` — `train`/`evaluate` consume processed data; metrics artifact declared |
| #19 | `feature/sprint-04-data-contracts` | Explicit stage input/output/config contracts; `params.yaml` reconciled (`evaluate` section) |
| #20 | `feature/sprint-04-ml-stage-refactor` | Separated ML compute / persistence / MLflow tracking (`tracking.py`; seeded, deterministic training) |
| #21 | `feature/sprint-04-pipeline-tests` | Stage-level unit tests, an end-to-end integration test, and the `contract` suite |
| #22 | `feature/sprint-04-reproducibility-ci` | Offline DVC integrity (`dvc dag` + `dvc status`) and contract checks in CI |
| #7 | `feature/sprint-04-release` | Documentation reconciliation + `v1.3.0` release (this PR) |

Concretely, the repository gained a correct, single-linear DVC lineage
(`raw → preprocess → processed → train → model → evaluate → metrics`), a consistent
parameter contract across `dvc.yaml`/`params.yaml`/code, a dedicated MLflow
boundary (`src/tracking.py`), a seeded and therefore deterministic training run, a
first-class metrics artifact (`metrics/metrics.json`), a four-tier test suite (84
tests), and a CI job that fails a PR on a broken graph, parameter, or lineage —
all offline.

## 3. What changed during implementation

The plan was a good map; a few things shifted once the code was in front of us:

- **`dvc repro --dry` was dropped from CI.** The plan floated `dvc dag`,
  `dvc status`, *and* `dvc repro --dry` as candidate CI checks. In practice
  `dvc repro --dry` reaches for the first stage's data dependency, which is
  remote-only, so it needs DagsHub credentials or fails nondeterministically —
  exactly the dependency the plan forbade. It was deliberately excluded, and the
  structural guarantees it would have given are enforced offline by the `contract`
  tests instead. (Recorded in `ci.yml` and [ADR-006](../decisions/ADR-006-pipeline-reproducibility.md)
  Alternative 6.)
- **The evaluation held-out split was not implemented.** The plan defined the
  evaluation boundary (§6.5) and required the methodology to be *documented* —
  which it is — but stopped short of mandating a held-out split this sprint.
  Evaluation remains in-sample; the split is deferred as deviation D5 because it is
  a configuration/graph change that is cleanest to make on its own, not bundled
  into the correctness sprint.
- **The header fix turned out to be the unlock for the whole chain.** Feeding the
  `preprocess` output into `train` (D1) was blocked by a smaller defect: the
  processed CSV was written headerless (D8), so downstream stages could not select
  `Outcome` by name. D8 had to be fixed first; only then could D1 land.
- **`GridSearchCV` had to be restructured, not just re-pointed.** The configured
  `n_estimators`/`max_depth` had been inert because a hardcoded grid shadowed them
  (C5). The fix set those on the base estimator and narrowed the grid to
  leaf/split regularization, so config genuinely governs the model — a slightly
  larger change than "pass the params through," but the minimum that makes the
  parameters real.

## 4. Problems encountered

- **The local `dvc` install is broken by a `pathspec` packaging anomaly**
  (`cannot import name '_DIR_MARK'`), unrelated to this repository. Graph
  validation had to be run in an isolated virtualenv with a clean resolve
  (`dvc 3.59.1`) — which is what a fresh CI runner produces anyway. Noted in the
  final review so it is not mistaken for a repository defect.
- **CI can validate the definition but not a locked execution.** Without a
  runnable dataset in CI there is no `dvc.lock` to commit and no `dvc repro` to
  run, so `dvc status` reports "changed" for the remote-only data. The honest
  resolution was to scope CI to *definition* validation and document the execution
  check as future work, rather than fake a green `dvc status`.
- **Keeping the refactor "smallest change" honest.** Separating compute from
  tracking is the kind of work that invites gold-plating. Holding to the plan's
  "correctness over abstraction" guardrail meant extracting exactly one boundary
  module and two pure functions — no framework, no plugin system.

## 5. Engineering decisions

- **A design contract before any code.** `pipeline-contract.md` + ADR-006 were
  written first (PR #17), enumerating the gaps as numbered deviations (D1–D8) and
  config inconsistencies (C1–C5). Every later PR closed specific, named items, and
  this release reconciled the contract to as-built — so the document never drifted
  from the code.
- **MLflow as a lazily-imported boundary.** All MLflow access goes through
  `src/tracking.py`, imported at the point a stage crosses the tracking boundary.
  The stages' ML computation imports no MLflow, so unit tests exercise it with no
  network — and, because the import is already lazy, the test stub needs *no*
  production-code change. MLflow behavior (metrics, params, artifacts, conditional
  registration) is preserved, not removed.
- **Determinism from config, not constants.** `random_state` seeds both the split
  and the estimator; `n_estimators`/`max_depth` are applied from `params.yaml`.
  Repeated runs on the same inputs and params yield the same model and metric.
- **The contract is executable.** Eight `contract` tests parse
  `dvc.yaml`/`params.yaml`/`src` and assert the lineage, parameter consistency,
  and single-owner artifacts — the checks a reviewer would otherwise redo by eye on
  every change. Paired with `dvc dag` in CI, a broken contract now fails a PR.
- **Honesty over a clean scorecard.** Rather than claim "reproducible pipeline"
  outright, the docs state precisely what is reproducible (logical: same inputs +
  params + seed → same result) and what is not yet (no `dvc.lock`; in-sample
  evaluation; name-pinned deps).

## 6. What went well

- **Deviation-driven PRs.** Numbering the gaps up front (D1–D8) made each PR's job
  unambiguous and made this release's reconciliation a checklist, not an
  investigation.
- **The lazy-import boundary paid off twice.** It made the ML compute testable
  *and* let the test stub work without touching production code — the same design
  decision serving correctness and testability.
- **Documentation stayed a correctness surface.** Writing the as-built contract
  re-verified every wiring claim against `dvc.yaml`/`params.yaml`/`src` and against
  `dvc dag`, catching any lingering drift.
- **CI got a real pipeline gate without new infrastructure.** `dvc` was already
  installed as a runtime dependency, so the integrity step added no install and no
  credentials.

## 7. What was deliberately deferred

Each is a conscious "not this sprint" decision, recorded so the deferral stays
honest:

- **Held-out evaluation (D5).** Evaluation is in-sample; a held-out split is the
  top item for the next sprint ([contract §8](../pipeline-contract.md#8-evaluation-boundary)).
- **Committed `dvc.lock` + in-CI `dvc repro` (part of D7).** Needs a runnable
  fixture dataset in CI; deferred with the execution check.
- **mypy in CI and branch protection.** Carried from Sprint 3; CI runs Ruff +
  pytest + DVC integrity, not yet mypy, and green checks are not yet required to
  merge.
- **Supply-chain hardening.** Digest pinning, image scanning (Trivy), SBOM,
  signing, and image publishing remain deferred (CD, not integration).
- **Kubernetes / cloud / serving / observability.** Roadmap v4–v6, explicitly out
  of scope.

## 8. Lessons learned

- **Fix the smallest blocking defect first.** The whole lineage correction (D1)
  waited on a one-line header change (D8). Sequencing by dependency, not by
  visibility, unblocked the sprint.
- **Make the contract executable, then reconcile it.** A contract enforced by
  tests and `dvc dag` cannot silently rot; reconciling prose to as-built at release
  is then verification, not archaeology.
- **Decouple the computation, preserve the capability.** Isolating MLflow behind a
  lazy boundary bought testability without dropping a single MLOps feature — the
  plan's "preserve MLflow" guardrail and its "unit-testable" goal were not in
  tension once the seam was in the right place.
- **Document what is *not* reproducible.** Stating the in-sample boundary and the
  missing `dvc.lock` plainly is what keeps "reproducible pipeline" a credible claim
  rather than an overclaim — and it hands the next sprint a precise backlog.

---

## Related documentation

- [Pipeline Contract](../pipeline-contract.md)
- [ADR-006 — Pipeline Reproducibility](../decisions/ADR-006-pipeline-reproducibility.md)
- [Sprint 4 — Final Validation](../reviews/sprint-04-final-review.md)
- [Sprint 4 — Proof Impact](../proof/sprint-04-proof-impact.md)
- [Sprint 3 — Retrospective](sprint-03-retrospective.md)
- [Roadmap](../roadmap.md) · [Changelog](../../CHANGELOG.md)
