# ADR-007: Held-Out Evaluation via a Dedicated `split` Stage

- **Status:** Accepted — implemented (proof hardening milestone). Resolves
  deviation **D5** (in-sample evaluation) tracked in the
  [pipeline contract](../pipeline-contract.md#11-deviation-status-sprint-4).
- **Date:** 2026-08-09
- **Deciders:** Asad Hanif
- **Related:** [ADR-006 (Pipeline Reproducibility)](ADR-006-pipeline-reproducibility.md),
  [ADR-003 (Why DVC)](ADR-003-why-dvc.md),
  [ADR-001 (Repository Structure)](ADR-001-repository-structure.md),
  [Pipeline Contract §8](../pipeline-contract.md#8-evaluation-boundary),
  [architecture.md](../architecture.md), [dvc.yaml](../../dvc.yaml),
  [params.yaml](../../params.yaml)

## Context

After the Sprint 4 PRs landed, the pipeline modeled a correct, reproducible DAG —
but **one correctness gap remained open** and was documented honestly rather than
hidden: evaluation was **in-sample**. Both `train` and `evaluate` depended on the
same file, `data/processed/data.csv`:

- `train` fit the model on the whole processed dataset.
- `evaluate` scored the model on that **same** dataset.
- The `accuracy` written to `metrics/metrics.json` and logged to MLflow therefore
  measured the model against data it had already seen, so it could not be cited as
  a generalization estimate.

This was recorded as deviation **D5** and named, in
[pipeline-contract.md §8](../pipeline-contract.md#8-evaluation-boundary), as *"the
single most valuable remaining pipeline-correctness improvement."* ADR-006
decision (5) committed the project to an honest evaluation boundary and
anticipated a held-out split; this ADR ratifies **how** that split is introduced.

The requirement is to make held-out evaluation a property that is **structurally
guaranteed and reproducible**, not merely a convention a future edit could quietly
break: training must never consume evaluation data, evaluation must consume
explicit held-out data, the partition must be deterministic, and the boundary must
be visible in the DAG.

## Decision

Introduce a **dedicated `split` DVC stage** between `preprocess` and
`train`/`evaluate` that partitions the processed dataset into a training set and a
held-out evaluation set. Specifically:

1. **The split is its own stage, not an implicit step.** `split`
   (`src/split.py`) reads `data/processed/data.csv` and produces two DVC-tracked
   outputs it exclusively owns: `data/processed/train.csv` and
   `data/processed/test.csv`. Making it a stage puts the train/evaluation fork in
   the DAG, where `dvc dag` and the `contract` tests can see and enforce it — the
   boundary is a topological fact, not a comment.

2. **`train` and `evaluate` consume disjoint files.** `train.input` becomes
   `data/processed/train.csv`; `evaluate.data` becomes `data/processed/test.csv`.
   Neither stage's code changes — both already read their dataset path from
   `params.yaml` — so the training algorithm is otherwise untouched. `train`'s
   pre-existing internal `train_test_split` is a *validation* split taken **within
   the training partition** and never touches the held-out file.

3. **The split is stratified and seeded.** `split_dataset` uses
   `train_test_split(test_size=split.test_size, random_state=split.random_state,
   stratify=<target>)`, so class balance is preserved in both partitions and the
   exact held-out rows are reproducible across runs.

4. **Leakage is asserted at runtime, not just assumed.** The pure computation
   asserts the two partitions are **disjoint** (`train.index ∩ test.index == ∅`)
   and **exhaustive** (`len(train) + len(test) == len(input)`); a violation aborts
   the run with a clear message. Invalid input (missing file, missing target
   column, a class too small to stratify, `test_size` out of range) fails as a
   typed `DataError`, consistent with the rest of the pipeline.

5. **The boundary is machine-checked offline.** A `contract` test
   (`test_train_and_evaluate_consume_disjoint_datasets`) parses `dvc.yaml`/
   `params.yaml` and asserts `train`'s dataset dependencies `== {train.csv}`,
   `evaluate`'s `== {test.csv}`, and that the two sets are disjoint — so a
   regression that repointed either stage at a shared dataset fails CI without any
   external service.

This keeps with **"correctness over abstraction"** (ADR-006): the guarantee is
expressed in explicit `dvc.yaml`/`params.yaml` wiring, typed errors, runtime
assertions, and tests — no new framework.

## Alternatives Considered

1. **Split inside `evaluate.py` (or `train.py`) at read time.**
   - *Decision:* rejected — this hides the boundary inside a stage where neither
     `dvc dag` nor the contract tests can see it, and it couples the split seed to
     a stage that should only score. A future edit could silently re-introduce
     in-sample evaluation. The boundary must be an explicit node in the DAG.

2. **Fold the split into `preprocess` (one stage emits three files).**
   - *Decision:* rejected — it overloads `preprocess` (a single-responsibility
     transform) with partitioning, and muddies artifact ownership: `preprocess`
     would own both the "clean data" concept and the "train/eval boundary"
     concept. A dedicated stage keeps each stage's responsibility and each
     artifact's owner singular (ADR-006 decision 3).

3. **A three-way split (train / validation / test) as separate persisted files.**
   - *Decision:* deferred — `train` already takes an internal validation split
     within its training data for tuning, which is sufficient for the current
     `GridSearchCV`. Persisting a separate validation file adds artifacts without
     current consumers. The `split` stage's two-way partition is the smallest
     change that closes D5; a persisted validation set can be added later if a
     stage needs it.

4. **Cross-validated / repeated-split evaluation instead of a single hold-out.**
   - *Decision:* deferred — a single stratified hold-out is the minimal, defensible
     step from in-sample to out-of-sample and matches the batch, file-based
     pipeline. Cross-validation would change the metrics contract
     (`metrics.json` shape) and cost, and is a quality refinement, not a
     correctness fix. Recorded as a remaining caveat in
     [pipeline-contract.md §8](../pipeline-contract.md#8-evaluation-boundary).

5. **Change `train`/`evaluate` code to accept the new paths.**
   - *Decision:* unnecessary — both stages read their dataset path from
     `params.yaml`, so repointing the config plus adding the `split` stage achieves
     held-out evaluation with **no change to training/scoring logic**. Smaller
     change, smaller risk (ADR-006's "correctness over abstraction").

## Consequences

**Positive**

- **D5 is resolved.** Reported `accuracy` is a genuine out-of-sample figure; the
  pipeline can now defensibly claim a held-out evaluation.
- **The boundary is guaranteed on three independent layers** — DAG topology, an
  offline contract disjointness test, and runtime disjoint/exhaustive assertions —
  so accidental leakage fails fast (CI or the run itself), not silently.
- **Reproducible.** `split.random_state` pins the exact held-out rows; stratifying
  preserves class balance so the metric is measured against a representative slice.
- **Minimal blast radius.** No change to the training algorithm or the metrics
  contract; `train.py`/`evaluate.py` logic is untouched (docstrings only). MLflow
  behavior is preserved.
- **Single-owner lineage preserved.** `split` exclusively owns both partitions;
  `preprocess` stays single-responsibility.

**Trade-offs and follow-ups**

- **Single split, not cross-validated.** The accuracy carries the variance of one
  20% partition rather than a confidence interval. Cross-validation is a future
  quality refinement (Alternative 4).
- **Small held-out set.** `test_size = 0.2` on a small dataset makes the estimate
  noisy; this is a data/quality caveat, documented in the contract, not a wiring
  defect.
- **`dvc.lock` still absent.** Held-out evaluation does not change the pipeline's
  other open item: no committed `dvc.lock` (part of D7), so CI validates the
  pipeline **definition** and lineage, not a locked **execution**. Full
  `dvc repro` against a committed fixture remains a later concern (ADR-006
  Alternative 6).
- **Metrics may shift.** Scoring on held-out data will generally report a lower,
  more honest accuracy than the previous in-sample figure; this is expected and is
  the point.
