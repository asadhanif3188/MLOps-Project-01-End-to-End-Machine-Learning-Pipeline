# Sprint 4 — Proof-Impact Assessment (v1.3.0)

- **Date:** 2026-08-06
- **Release:** `v1.3.0` — Pipeline Correctness & Reproducibility
- **Related:** [Sprint 4 Final Review](../reviews/sprint-04-final-review.md),
  [Sprint 4 Retrospective](../retrospectives/sprint-04-retrospective.md),
  [Pipeline Contract](../pipeline-contract.md),
  [ADR-006](../decisions/ADR-006-pipeline-reproducibility.md),
  [ADR-007](../decisions/ADR-007-held-out-evaluation.md)

> **Superseding update (2026-08-09).** This document is a snapshot as of the
> `v1.3.0` release. Its **#1 remaining limitation — in-sample evaluation (D5)** has
> since been **resolved** by a proof-hardening milestone that added a dedicated
> `split` stage, so `train` and `evaluate` now consume disjoint data and the
> reported accuracy is out-of-sample (see
> [ADR-007](../decisions/ADR-007-held-out-evaluation.md) and
> [contract §8](../pipeline-contract.md#8-evaluation-boundary)). The in-sample
> statements in §4 and §5 below are retained as the historical `v1.3.0` record;
> a committed `dvc.lock` (part of D7) is now the single remaining item.

## Purpose

This document answers one question, and answers it with evidence:

> **What can Asad credibly claim after Sprint 4 that he could not credibly claim
> after Sprint 3?**

"Credibly" is the operative word. Every claim below points to a file, a test, or a
command in this repository that a technical reviewer can check. Claims the
repository does **not** support are listed explicitly in
[§4](#4-what-still-cannot-be-claimed) so the credible ones are not diluted by
overreach.

---

## 1. The shift, in one line

| | Claim licensed by the repository |
|-|----------------------------------|
| **After Sprint 3** | "I built and containerized an MLOps project using DVC and MLflow, with a CI pipeline that lints, tests, and builds the image." |
| **After Sprint 4** | "I engineered the pipeline's dependency graph and stage contracts, corrected the DVC lineage so preprocessing actually feeds training, made configuration consistent across `dvc.yaml`/`params.yaml`/code, isolated MLflow behind a boundary so the ML logic is unit-testable, made training deterministic, and enforced pipeline integrity in CI — offline, with no credentials." |

The Sprint 3 claim was about the **infrastructure around** the pipeline. The
Sprint 4 claim is about the **correctness of the pipeline itself** — a materially
stronger engineering statement, because infrastructure quality cannot compensate
for an incorrect ML pipeline.

---

## 2. New credible claims, with evidence

Each row is a claim that was **not** defensible after Sprint 3 and **is** after
Sprint 4, with the artifact a reviewer can inspect to verify it.

### 2.1 "I corrected the DVC dependency graph to model the real pipeline lineage."

- **Before:** `train` and `evaluate` read `data/raw/data.csv` directly; the
  `preprocess` output was produced but never consumed (an orphaned artifact).
- **Now:** `train` depends on `data/processed/data.csv`; `evaluate` depends on the
  processed data **and** the model; the chain is
  `raw → preprocess → processed → train → model → evaluate → metrics`.
- **Evidence:** [`dvc.yaml`](../../dvc.yaml); `dvc dag` renders the acyclic graph;
  `tests/contract/test_pipeline_contract.py::test_lineage_matches_contract` and
  `::test_processed_data_is_consumed_not_orphaned` fail if this regresses.

### 2.2 "I made pipeline configuration consistent, with no orphaned parameters."

- **Before:** `dvc.yaml` referenced `train.data`/`train.model`, which did not exist
  in `params.yaml` (it defined `train.input`/`train.output`); the evaluate stage
  read a section named `test`; and `train.random_state`/`n_estimators`/`max_depth`
  were loaded but never applied.
- **Now:** one authoritative name per parameter across all three files; the
  `evaluate` section is aligned with the stage; configured hyperparameters govern
  the estimator.
- **Evidence:** [`params.yaml`](../../params.yaml);
  `test_every_dvc_param_key_exists_in_params_yaml`, `test_no_orphaned_params`,
  `test_declared_outputs_match_params`.

### 2.3 "I isolated MLflow behind a boundary so the ML logic is unit-testable without a live service."

- **Before:** `train` and `evaluate` imported MLflow at module load and required a
  live tracking server, so their logic could not be unit-tested offline.
- **Now:** all MLflow access is funneled through `src/tracking.py`, imported
  lazily; the pure computations `run_training` and `compute_metrics` import no
  MLflow and are tested directly; a `stub_tracking` fixture neutralizes the
  boundary end-to-end.
- **Evidence:** [`src/tracking.py`](../../src/tracking.py),
  [`src/train.py`](../../src/train.py), [`src/evaluate.py`](../../src/evaluate.py),
  [`tests/conftest.py`](../../tests/conftest.py); `python -m pytest` runs 84 tests
  offline (unit tests need no network, MLflow, or credentials).

### 2.4 "I made training deterministic."

- **Before:** `train_test_split` was called without a seed and the configured
  `random_state` was inert, so training was non-deterministic.
- **Now:** `random_state` seeds both the split and the `RandomForestClassifier`;
  repeated runs on the same inputs and params produce the same model and metric.
- **Evidence:** [`src/train.py`](../../src/train.py) (`run_training`);
  `tests/unit/test_train.py`.

### 2.5 "The pipeline emits a first-class, versioned metrics artifact."

- **Before:** `accuracy` existed only in MLflow and the log stream; nothing on disk
  or in DVC.
- **Now:** `evaluate` writes `metrics/metrics.json`, declared in `dvc.yaml` as an
  uncached DVC metric.
- **Evidence:** [`dvc.yaml`](../../dvc.yaml) (`evaluate.metrics`),
  [`src/evaluate.py`](../../src/evaluate.py); `test_declared_outputs_match_params`.

### 2.6 "I enforce pipeline integrity in CI without production credentials."

- **Before:** CI lint/tested and built the image, but nothing validated the DVC
  graph, the parameter contract, or the lineage.
- **Now:** every pull request runs `dvc dag` + local `dvc status`
  (`DVC_NO_ANALYTICS=true`, no remote access) and the `contract` tests, so a broken
  graph, an inconsistent parameter, or a mis-wired lineage fails the PR. Sprint 3's
  image build/validation and least-privilege `contents: read` are preserved.
- **Evidence:** [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml);
  `python -m pytest -m contract`.

### 2.7 "I wrote and enforce an engineering contract for the pipeline."

- **Before:** the pipeline's intended inputs/outputs/ownership existed only
  implicitly in code.
- **Now:** [`pipeline-contract.md`](../pipeline-contract.md) specifies every
  stage's inputs, outputs, configuration, artifact ownership, evaluation boundary,
  and external-service boundaries — and the `contract` test suite makes most
  divergences fail CI automatically. Rationale is ratified in
  [ADR-006](../decisions/ADR-006-pipeline-reproducibility.md).

---

## 3. The reviewer test

Sprint 4's success criterion was that a technical reviewer can answer, **without
inferring from implementation details**: *what depends on what, where does every
artifact come from, what configuration controls it, and how is it validated?*

That is now answerable from `dvc.yaml`, `params.yaml`, and
[pipeline-contract.md](../pipeline-contract.md) alone, and the answers are
machine-checked by `dvc dag` and the `contract` tests. That capability — not any
single code change — is the sprint's proof.

---

## 4. What still **cannot** be claimed

These are false or unsupported today. Claiming any of them would be an overreach,
and each is documented as a known limitation so it is not accidentally implied.

- **❌ "The model's accuracy reflects real-world / generalization performance."**
  Evaluation is **in-sample**: `evaluate` scores the model on the same processed
  dataset `train` fits on. The reported accuracy supports **no** generalization
  claim (deviation D5,
  [contract §8](../pipeline-contract.md#8-evaluation-boundary)). The credible claim
  is about *pipeline correctness and honesty*, not model quality.
- **❌ "Comprehensive ML correctness."** The pipeline is correctly *wired,
  configured, and tested*; it does not yet include held-out validation, data
  validation, or model-quality gates.
- **❌ "Complete / byte-for-byte reproducibility."** Training is *logically*
  reproducible (same inputs + params + seed → same result), but no `dvc.lock` is
  committed and dependencies/base image are name-pinned, not digest-pinned. Bit-for-
  bit reproducible rebuilds are not yet proven (deviation D7 part,
  [contract §7](../pipeline-contract.md#7-reproducibility-expectations)).
- **❌ "CI reproduces the pipeline end-to-end."** CI validates the pipeline
  **definition** (`dvc dag`, `dvc status`, contract tests). It does **not** run
  `dvc repro` — the raw dataset is remote-only — so it does not prove execution.
- **❌ Production deployment, model serving, Continuous Delivery, cloud
  infrastructure, or Kubernetes.** None of these exist in the repository; all are
  explicitly out of Sprint 4 scope (roadmap v4–v6). CI is validation-only: it does
  not deploy, publish images, or touch a cluster.

---

## 5. Remaining limitations (backlog for the next proof step)

In priority order — closing these is what would license the next round of claims:

1. ~~**Held-out evaluation (D5).** Turn the in-sample metric into a defensible
   generalization estimate. This is a configuration/graph change, not a rewrite.~~
   **✅ Resolved (2026-08-09)** by a dedicated `split` stage — `train` and
   `evaluate` now consume disjoint partitions, so the metric is out-of-sample
   ([ADR-007](../decisions/ADR-007-held-out-evaluation.md)). As predicted, it was a
   graph/config change, not a rewrite of the training logic.
2. **Committed `dvc.lock` + in-CI `dvc repro` against a fixture dataset (D7).**
   Upgrades the claim from "logically reproducible" to "reproduced and drift-gated
   in CI."
3. **mypy in CI + branch protection.** Makes the existing quality signal binding.
4. **Digest pinning + image scanning.** Prerequisites for "byte-for-byte
   reproducible" and for safely publishing the image.

---

## 6. The honest one-paragraph statement

> Building on a containerized, CI-validated MLOps repository, I re-engineered the
> ML pipeline for correctness and reproducibility: I corrected the DVC dependency
> graph so preprocessing output actually feeds training, reconciled the parameter
> contract across `dvc.yaml`/`params.yaml`/code with no orphaned parameters,
> isolated MLflow behind a lazily-imported boundary so the training and evaluation
> logic is unit-testable offline, seeded training for determinism, produced a
> first-class metrics artifact, and enforced the whole pipeline contract in CI
> (`dvc dag` + `dvc status` + static contract tests) with no production
> credentials. Evaluation is currently in-sample and reproducibility is logical
> rather than byte-for-byte — both documented as tracked limitations — so I make a
> claim about pipeline **engineering correctness**, not about model performance.

That paragraph is fully supported by the repository. Its restraint — naming the
in-sample boundary and the reproducibility caveat out loud — is part of what makes
the rest of it credible.
