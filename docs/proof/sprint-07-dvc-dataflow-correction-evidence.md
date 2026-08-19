# Sprint 7 · PR 9 — DVC data-flow correction: verification evidence

**PR:** `fix(dvc): align pipeline DAG with actual data dependencies`
**Branch:** `feature/sprint-07-dvc-dataflow-correction`
**Date:** 2026-08-19
**Scope:** Correctness verification of the declared DVC DAG against actual Python
execution — no cosmetic DVC changes.

## Summary

This PR set out to correct a historical mismatch: preprocess output orphaned
(not consumed downstream) and DVC stage parameter / input / output names not
matching the actual stage code. On inspection, **that mismatch no longer exists** —
it was already resolved across Sprint 4 (`dvc-pipeline`, `ml-stage-refactor`,
`pipeline-contract`, `pipeline-tests`) and the proof-hardening held-out-evaluation
work, and is guarded going forward by the offline **contract** test suite
(`tests/contract/`). The closed deviations are referenced by ID in the stage code
and contract tests: **D1** (train read raw directly), **D4** (metrics not a tracked
artifact), **D5** (in-sample evaluation), **D8** (headerless processed output). See
`docs/pipeline-contract.md` and ADR-006 / ADR-007 / ADR-008.

Rather than fabricate a diff or make cosmetic YAML edits (explicitly out of scope),
this PR records the verification that the **declared DVC graph now represents actual
execution**. The evidence is below.

---

## 1. Actual runtime graph (traced from `src/*.py`)

Each stage's `main()` loads its section from `params.yaml` and reads/writes exactly
these paths:

| Stage | Reads (`params.yaml`) | Writes | Params consumed |
|---|---|---|---|
| `preprocess.py` | `preprocess.input` = `data/raw/data.csv` | `data/processed/data.csv` (header preserved) | `input`, `output` |
| `split.py` | `split.input` = `data/processed/data.csv` | `data/processed/train.csv`, `data/processed/test.csv` (stratified, seeded, disjoint) | `input`, `train_output`, `test_output`, `target`, `test_size`, `random_state` |
| `train.py` | `train.input` = `data/processed/train.csv` | `models/model.pkl` | `input`, `output`, `target`, `random_state`, `n_estimators`, `max_depth` |
| `evaluate.py` | `evaluate.data` = `data/processed/test.csv`, `evaluate.model` = `models/model.pkl` | `metrics/metrics.json` | `data`, `model`, `target`, `metrics` |

## 2. Before ↔ After data-flow diagram

The declared DVC DAG (`dvc.yaml`) and the traced runtime graph are **identical** —
there is no correction to apply. The target lineage and the current state coincide:

```
BEFORE (as feared by the PR premise)      AFTER / CURRENT (verified)
──────────────────────────────────      ──────────────────────────────
raw ─┐                                    raw
     │ (processed orphaned; train           ↓  preprocess   reads preprocess.input
     │  reads raw directly — D1)         processed/data.csv  ✔ consumed by split
     ↓                                       ↓  split
   train (in-sample — D5)               train.csv ┐   test.csv (held-out)
     ↓                                       ↓     │        │
   evaluate (raw/no metrics — D4)        train ────┘        │
                                            ↓  → models/model.pkl
                                         evaluate  reads test.csv + model.pkl
                                            ↓  → metrics/metrics.json  (tracked — D4 closed)
```

`train` consumes **only** `data/processed/train.csv`; `evaluate` consumes **only**
`data/processed/test.csv` (the disjoint held-out partition) plus the model — the
held-out boundary is explicit in the graph and machine-checked (§3).

## 3. Validation performed

All commands run from the repository root on this branch.

**Lint:**

```
$ python -m ruff check .
All checks passed!
```

**Tests (full suite):**

```
$ python -m pytest -q
152 passed, 1 skipped in ~58s
# skip = tests/smoke/test_smoke.py (runtime dep 'mlflow' not installed)
```

The **13 contract tests** (`tests/contract/`) enforce precisely this PR's
requirements, offline (pure YAML parsing, no data/network):

- every `dvc.yaml` `params:` key exists in `params.yaml`; **no orphaned params**;
- declared `outs` equal their `params.yaml` values;
- each stage's **input dataset** in `params.yaml` equals its `dvc.yaml` dep
  (closes the params-only-drift gap that could reintroduce D5);
- graph is acyclic; lineage matches the contract chain; **each artifact has exactly
  one producer**; preprocess output **is consumed**; both split halves are consumed;
- `train` and `evaluate` consume **disjoint** dataset files (held-out guarantee);
- each stage command runs a tracked, existing `src/*.py`.

The **integration** test `tests/integration/test_fixture_reproducibility.py` runs a
real end-to-end reproduction of all four stages (the same `src/` code, offline
tracking) against the committed fixture dataset — proving
`declared pipeline + params + inputs + code = reproducible execution`.

**`dvc dag`** (root pipeline) confirms the linear chain:

```
preprocess ─▶ split ─▶ train ─▶ evaluate
                  └────────────▶ evaluate
```

**`dvc status`** notes (context, not defects):

- **Root pipeline** reports every stage as changed / uncached: there is no root
  `dvc.lock` and the production dataset is remote-only (DagsHub S3, `data/raw/data.csv.dvc`
  "not in cache"). This reflects an unlocked/unpulled working tree, **not** a
  DAG mismatch — the graph itself is well-formed (see `dvc dag`).
- **Fixture pipeline** reports `src/train.py` / `src/evaluate.py` as changed deps:
  the committed fixture `dvc.lock` is hash-stale relative to those files (they were
  edited in the earlier MLflow PRs). This is lock hygiene, **not** DAG alignment, and
  is left untouched here — the structural fixture-lock contract test
  (`tests/contract/test_fixture_lock_contract.py`) and the reproducibility integration
  test both pass, and DVC output hashes are intentionally platform-sensitive
  (regenerating on a different OS would re-drift; see ADR-008).

## 4. Path consistency: local Docker ↔ Kubernetes

Both execution environments drive the **same** entrypoint and paths:

- Docker: `Dockerfile` copies `dvc.yaml params.yaml` into `/app`; the pipeline runs
  `dvc repro` (`docker-compose.yml` mounts `./params.yaml:/app/params.yaml`).
- Kubernetes: `k8s/base/job.yaml` runs the image's default `dvc repro`
  (preprocess → split → train → evaluate); the dataset is provided at
  `/app/data/raw`, matching `params.yaml: preprocess.input = data/raw/data.csv`
  (relative to the `/app` WORKDIR). `k8s/validate.py` asserts the read-only
  `/app/data/raw` mount.

Because both environments use the identical `params.yaml` and relative paths, the
declared graph and its inputs/outputs are portable across local Docker and K8s.

## 5. Conclusion

The DVC pipeline definition (`dvc.yaml`), its configuration (`params.yaml`), the
stage code (`src/*.py`), the tests, the Docker/K8s runtime paths, and the
documentation (`README.md`, `docs/pipeline-contract.md`) are all mutually consistent.
**The DVC graph represents actual execution**, and the contract test suite pins it so
against future regression. No pipeline, parameter, or code change was required.
