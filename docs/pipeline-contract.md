# Pipeline Contract

- **Status:** Draft (design contract) — governs Sprint 4 (v1.3.0) implementation
- **Date:** 2026-08-05
- **Owner:** Asad Hanif
- **Related:** [ADR-006 (Pipeline Reproducibility)](decisions/ADR-006-pipeline-reproducibility.md),
  [ADR-003 (Why DVC)](decisions/ADR-003-why-dvc.md),
  [ADR-002 (Why MLflow)](decisions/ADR-002-why-mlflow.md),
  [architecture.md](architecture.md), [dvc.yaml](../dvc.yaml), [params.yaml](../params.yaml)

---

## 1. Purpose

This document defines the **engineering contract** for the ML pipeline: what each
stage consumes, what it produces, who owns each artifact, where external services
sit, and what counts as a valid, reproducible run.

It is a **design contract, not an implementation claim.** Where the current code
does not yet satisfy the contract, this document says so explicitly and marks the
gap. Nothing here asserts that a behavior exists unless it does.

Throughout, two states are distinguished:

| Marker | Meaning |
|--------|---------|
| **CURRENT** | What the repository does today, as verified in `dvc.yaml`, `params.yaml`, and `src/*.py` at the time of writing. |
| **TARGET** | What the contract requires the pipeline to do once Sprint 4 implementation lands. Not yet implemented in this PR. |

This PR (Sprint 4, PR 1) introduces **no implementation changes**. It records the
contract that later PRs (PR 2–PR 6) will implement and that CI (PR 6) will enforce.

The current state below is consistent with the "known gaps" already recorded in
[architecture.md](architecture.md#3-pipeline-flow); this document is the
authoritative, stage-by-stage elaboration of that summary.

---

## 2. Logical Pipeline

The intended logical pipeline is a linear artifact-lineage chain:

```text
raw data
   │
   ▼
preprocess          (stage)
   │
   ▼
processed data      (artifact)
   │
   ▼
train               (stage)
   │
   ▼
model               (artifact)
   │
   ▼
evaluate            (stage)
   │
   ▼
metrics             (artifact)
```

Experiment metadata (parameters, metrics, model registry) is logged to **MLflow
on DagsHub** as a side channel from `train` and `evaluate`; it is a boundary, not
a pipeline stage (see §9). Data and artifact lineage is owned by **DVC** (see
[ADR-006](decisions/ADR-006-pipeline-reproducibility.md)).

**CURRENT vs TARGET at a glance:**

```text
CURRENT (as wired today)                 TARGET (this contract)

raw data                                 raw data
   ├────────────┐                           │
   ▼            ▼                            ▼
preprocess    (raw fed directly)         preprocess
   │            │                            │
   ▼            ▼                            ▼
processed     train ──▶ model            processed data
 (orphaned)     │                            │
                ▼                            ▼
             evaluate (on raw)            train ──▶ model
                │                            │
                ▼                            ▼
             accuracy (MLflow only,       evaluate (on held-out)
              no metrics artifact)           │
                                             ▼
                                          metrics (tracked artifact)
```

In the current wiring, `train` and `evaluate` both read `data/raw/data.csv`
directly; the `processed data` artifact is produced but not consumed, and no
metrics **artifact** is written. The target restores the single linear chain.

---

## 3. Stage Contracts

Each stage below documents its **purpose**, **inputs**, **outputs**,
**configuration**, **artifact ownership**, and **failure conditions**, in both
CURRENT and TARGET form. Paths and parameter names are stated exactly as they
appear in the repository.

### 3.1 `preprocess`

**Purpose:** Produce the processed dataset that training consumes, from the raw
dataset.

| Aspect | CURRENT | TARGET |
|--------|---------|--------|
| Command | `python src/preprocess.py` | unchanged |
| Input (data) | `data/raw/data.csv` | `data/raw/data.csv` |
| Input (code) | `src/preprocess.py` | unchanged |
| Config keys | `preprocess.input`, `preprocess.output` (from `params.yaml`) | unchanged |
| Output | `data/processed/data.csv` | `data/processed/data.csv` (**consumed by `train`**) |
| DVC declaration | `deps: data/raw/data.csv, src/preprocess.py`; `params: preprocess.input, preprocess.output`; `outs: data/processed/data.csv` | unchanged, plus this output must be a real `train` dependency |
| Behavior | Reads the raw CSV and re-writes it **without header and without index** (`header=False, index=False`); performs no cleaning or transformation today. | Behavior is deliberately left open by this contract (Sprint 4 PR 3/PR 4). If the processed file is to be consumed by `train`, the **column contract** (see below) must be preserved. |

**Artifact ownership:** `preprocess` **owns** `data/processed/data.csv`. No other
stage may write it.

**Failure conditions (current, via typed exceptions):**

- `DataError` — raw input missing, empty, or not valid CSV.
- `DataError` — processed output directory/file cannot be written.
- `ConfigError` — `params.yaml` missing, unparseable, or missing
  `preprocess.input`/`preprocess.output`.

**Contract note — column header.** The processed file is currently written with
`header=False`. Downstream stages validate the presence of an `Outcome` column
(`ensure_columns(..., ["Outcome"])`). Therefore, **as written today the processed
file could not be consumed by `train`/`evaluate` unchanged** — its header row is
absent. This is one reason the current wiring routes `train`/`evaluate` at the raw
file instead. The TARGET requires this mismatch to be resolved (either the
processed file retains its header, or the downstream contract is redefined) before
`train` may consume the processed dataset. This contract does **not** prescribe
which; it only requires the input/output column contract to be explicit and
consistent (Sprint 4 PR 3).

### 3.2 `train`

**Purpose:** Fit and select a model from the training dataset and persist it as the
model artifact.

| Aspect | CURRENT | TARGET |
|--------|---------|--------|
| Command | `python src/train.py` | unchanged |
| Input (data) | `data/raw/data.csv` — read from `params['train']['input']` | `data/processed/data.csv` — the `preprocess` output |
| Input (code) | `src/train.py` | unchanged |
| Config keys read by code | `train.input`, `train.output`, `train.random_state`, `train.n_estimators`, `train.max_depth` | consistent set, all consumed (see §4) |
| Config keys declared in `dvc.yaml` | `train.data`, `train.model`, `train.max_depth`, `train.n_estimators`, `train.random_state` | must match the keys the code actually reads |
| Output | `models/model.pkl` | `models/model.pkl` |
| DVC declaration | `deps: data/raw/data.csv, src/train.py`; `outs: models/model.pkl` | `deps` must include `data/processed/data.csv`, not `data/raw/data.csv` |
| Behavior | Requires `MLFLOW_TRACKING_URI`; splits with `train_test_split(test_size=0.20)`; runs `GridSearchCV` (cv=3) over a **hardcoded** grid; logs params/metrics/artifacts to MLflow; pickles the best estimator. | Preserve MLflow behavior behind a testable boundary; make the data input the processed dataset; make configured parameters actually govern behavior. |

**Artifact ownership:** `train` **owns** `models/model.pkl`. No other stage writes
it. `train` also **owns** its MLflow run (params, metrics, model registry entry).

**Failure conditions (current, via typed exceptions):**

- `DataError` — dataset missing/empty/invalid, or missing the `Outcome` column.
- `ConfigError` — `MLFLOW_TRACKING_URI` unset/empty; or required `train.*` params
  absent.
- `TrackingError` — MLflow logging fails (URI, credentials, or network).
- `ModelError` — the fitted estimator cannot be pickled.

**Contract note — configured parameters not applied (CURRENT).** `train.py`
loads `random_state`, `n_estimators`, and `max_depth` and passes them into
`train()`, but the function body does **not** use them: the hyperparameter grid is
hardcoded, and `train_test_split` is called **without** a `random_state`. As a
result these parameters are currently inert and the split/fit are non-deterministic
(see §7). The TARGET requires configured parameters to actually govern the run, or
to be removed if genuinely unused — no orphaned parameters.

### 3.3 `evaluate`

**Purpose:** Measure the trained model against an evaluation dataset and emit
metrics.

| Aspect | CURRENT | TARGET |
|--------|---------|--------|
| Command | `python src/evaluate.py` | unchanged |
| Input (data) | `data/raw/data.csv` — read from `params['test']['data']` | An explicit **held-out** evaluation dataset (see §8) |
| Input (model) | `models/model.pkl` — read from `params['test']['model']` | `models/model.pkl` (the `train` output) |
| Input (code) | `src/evaluate.py` | unchanged |
| Config keys read by code | `test.data`, `test.model` (section literally named `test`) | An `evaluate.*` (or otherwise stage-aligned) section; naming consistent with the stage |
| Config keys declared in `dvc.yaml` | **none** | evaluation params declared |
| Output | **None tracked by DVC.** Logs an `accuracy` metric to MLflow and to the log stream only. | A declared **metrics artifact** (e.g. a metrics file), DVC-tracked. |
| DVC declaration | `deps: data/raw/data.csv, models/model.pkl, src/evaluate.py`; no `params`, no `outs`/`metrics` | `deps` include the evaluation data + model; a `metrics:` (or `outs:`) declaration for the metrics artifact |
| Behavior | Requires `MLFLOW_TRACKING_URI`; loads the model; predicts over the **entire** dataset; computes `accuracy_score`; logs it to MLflow. | Predict over the held-out set; write metrics as a first-class artifact; preserve MLflow logging where appropriate. |

**Artifact ownership:** In the TARGET, `evaluate` **owns** the metrics artifact. In
the CURRENT state, no metrics artifact exists on disk or in DVC — the only record of
`accuracy` is the MLflow run and the process logs.

**Failure conditions (current, via typed exceptions):**

- `DataError` — dataset missing/empty/invalid, or missing the `Outcome` column.
- `ConfigError` — `MLFLOW_TRACKING_URI` unset/empty; or required `test.*` params
  absent.
- `ModelError` — model file missing/corrupt, or the model fails to predict on the
  provided features.
- `TrackingError` — MLflow logging fails (URI, credentials, or network).

**Contract note — evaluation boundary (CURRENT).** `evaluate` scores the model on
the same full raw dataset the model was trained from. There is no held-out set, so
the reported `accuracy` is an **in-sample** figure and must not be presented as a
generalization estimate. See §8.

---

## 4. Configuration & Parameter Contract

**`params.yaml` is the single authoritative source of pipeline parameters.**
`dvc.yaml` references parameter *keys*; the stage code reads parameter *values*.
The contract requires all three to agree.

### 4.1 Current `params.yaml` sections

```yaml
preprocess:
  input:  data/raw/data.csv
  output: data/processed/data.csv

train:
  input:  data/raw/data.csv
  output: models/model.pkl
  random_state: 42
  n_estimators: 100
  max_depth: 5

test:
  data:  data/raw/data.csv
  model: models/model.pkl
```

### 4.2 Known configuration inconsistencies (CURRENT)

| # | Inconsistency | Evidence | TARGET resolution |
|---|---------------|----------|-------------------|
| C1 | `dvc.yaml` `train` stage declares params `train.data` and `train.model`, which **do not exist** in `params.yaml` (it defines `train.input`/`train.output`). | `dvc.yaml` `train.params`; `params.yaml` `train:` | One authoritative name per parameter; `dvc.yaml` params must match `params.yaml` keys and the keys the code reads. |
| C2 | `train.py` reads `train.input`/`train.output`; it does **not** read `train.data`/`train.model`. | `src/train.py:main` | Same as C1 — align names across code, `params.yaml`, and `dvc.yaml`. |
| C3 | `evaluate.py` reads a section named `test` (`test.data`, `test.model`), while the stage is named `evaluate`. | `src/evaluate.py:main` | Section name aligned with the stage (or explicitly justified). |
| C4 | `dvc.yaml` `evaluate` stage declares **no** params, so evaluation configuration is invisible to the DVC graph. | `dvc.yaml` `evaluate` | Declare evaluation params in `dvc.yaml`. |
| C5 | `train.random_state`, `train.n_estimators`, `train.max_depth` are loaded but **not applied** by `train()`. | `src/train.py:train` (hardcoded grid; `train_test_split` has no `random_state`) | Configured parameters must govern behavior, or be removed. |

These map directly to the "param name mismatch" and related gaps recorded in
[architecture.md](architecture.md#3-pipeline-flow) and
[ADR-003](decisions/ADR-003-why-dvc.md#consequences).

### 4.3 Parameter contract rules (TARGET)

1. Every parameter a stage reads is defined in `params.yaml`.
2. Every parameter key `dvc.yaml` references exists in `params.yaml`.
3. No parameter is declared or loaded that no stage uses (**no orphaned params**).
4. Parameter section names correspond to their stage names.
5. Parameters that affect reproducibility (e.g. seeds) are declared and applied.

---

## 5. Artifact Ownership & Lineage

Each artifact has exactly **one** producing stage ("owner"). Consumers may read it
but never write it.

| Artifact | Owner (writes) | Consumers (read) — CURRENT | Consumers (read) — TARGET | Tracking |
|----------|----------------|----------------------------|---------------------------|----------|
| `data/raw/data.csv` | External / ingestion (`data/raw/data.csv.dvc`) | `preprocess`, `train`, `evaluate` | `preprocess` only | DVC (`.dvc` pointer) |
| `data/processed/data.csv` | `preprocess` | *(none — orphaned)* | `train` (and evaluation source, per §8) | DVC stage output |
| `models/model.pkl` | `train` | `evaluate` | `evaluate` | DVC stage output |
| metrics artifact | `evaluate` (TARGET) | — | downstream reporting / CI | DVC metrics (TARGET) |
| MLflow run (params, metrics, registry) | `train`, `evaluate` | MLflow/DagsHub UI | unchanged | MLflow (external) |

**Rules:**

- A stage writes only the artifact(s) it owns.
- The raw dataset is owned upstream of the pipeline (DVC-tracked pointer) and must
  not be mutated by any stage.
- In the TARGET, the raw dataset has exactly one direct consumer (`preprocess`);
  training and evaluation consume derived artifacts, not the raw file.

---

## 6. Stage Input/Output Summary

The explicit contract (TARGET), for quick reference:

| Stage | Input | Output | Configuration |
|-------|-------|--------|---------------|
| `preprocess` | Raw dataset | Processed dataset | Preprocessing parameters |
| `train` | Processed dataset | Model artifact | Training parameters (incl. seed) |
| `evaluate` | Model + evaluation dataset | Metrics artifact | Evaluation parameters |

The CURRENT deviations from this table are enumerated per stage in §3 and
summarized in §11.

---

## 7. Reproducibility Expectations

Reproducibility is treated as an **engineering requirement**, not a nicety
(rationale in [ADR-006](decisions/ADR-006-pipeline-reproducibility.md)).

**TARGET expectations:**

1. **Declared lineage.** The pipeline is reconstructable from declared inputs,
   parameters, and dependencies. `dvc.yaml` accurately models the real DAG:
   `raw → preprocess → processed → train → model → evaluate → metrics`.
2. **Deterministic given fixed inputs + params + seed.** Random operations
   (data split, estimator construction) are seeded from `params.yaml`. Repeated
   runs on the same inputs and params yield the same model and metrics.
3. **Change detection.** `dvc status` / `dvc repro` re-run exactly the stages whose
   declared dependencies changed, and no others.
4. **Locked state.** A committed `dvc.lock` records the resolved dependency hashes
   for a run. *(CURRENT: no `dvc.lock` is present in the repository — the graph has
   not been locked. TARGET: locked and validated in CI.)*
5. **CI enforcement.** Pipeline integrity (valid DAG, consistent params, buildable
   graph) is validated automatically (Sprint 4 PR 6), without requiring production
   credentials or live external services.

**CURRENT reproducibility gaps:**

- `train_test_split` is called without `random_state`, and the estimator/grid do
  not fix a seed → training is **non-deterministic**.
- `train.random_state` exists but is not applied (see C5).
- No `dvc.lock` exists → the resolved pipeline state is not pinned.
- Reproducibility is not yet enforced in CI.

Each gap is scheduled for a later Sprint 4 PR; none is fixed in this PR.

---

## 8. Evaluation Boundary

The evaluation boundary defines **what data proves what claim.**

**CURRENT:**

- `train` performs an internal `train_test_split` (20% test) purely to report an
  in-training accuracy; that split is not persisted or exported.
- `evaluate` scores the model over the **entire** `data/raw/data.csv` — which
  includes the rows the model was trained on.
- Consequently the `accuracy` logged by `evaluate` is an **in-sample** metric.

**TARGET:**

- There is an explicit, documented held-out evaluation dataset that `evaluate`
  consumes and that `train` does **not** fit on.
- The contract records: what data trains, what data evaluates, how tuning is done,
  which split is held out, and which metrics are produced.
- **Honest-evaluation rule:** no model-performance claim is made unless the
  evaluation methodology supports it. Until a genuine held-out split exists, any
  reported accuracy is labeled in-sample.

This section is the "evaluation boundary" deliverable required by the Sprint 4
plan; the boundary is defined here **before** implementation changes.

---

## 9. External Service Boundaries

The pipeline touches two external systems. Both are **boundaries**, isolated from
core ML logic.

| Boundary | Used by | Purpose | Required to run? (CURRENT) | Contract (TARGET) |
|----------|---------|---------|----------------------------|-------------------|
| **MLflow on DagsHub** | `train`, `evaluate` | Experiment tracking, metric logging, model registry | **Yes** — both stages call `require_env("MLFLOW_TRACKING_URI")` and connect to MLflow; they cannot complete without it. | MLflow logging preserved, but isolated behind a boundary so **core ML logic is unit-testable without the service** (see [ADR-006](decisions/ADR-006-pipeline-reproducibility.md)). |
| **DVC remote (S3-compatible, DagsHub)** | data/model retrieval | Store/fetch DVC-tracked data & models | Needed to `dvc pull`/`push` artifacts; not needed for local reasoning about the graph | Unchanged; graph validation in CI must not require remote credentials. |

**Environment/config boundary:**

- `MLFLOW_TRACKING_URI` (and DagsHub credentials) are provided via `.env`
  (`python-dotenv`); see [`.env.example`](../.env.example). Secrets are never
  committed.
- **Contract rule:** ordinary **unit tests must not require** live MLflow, DagsHub,
  network access, or production credentials. Tests that genuinely exercise external
  behavior are **integration** tests and are marked/segregated as such. (Rationale:
  [ADR-006](decisions/ADR-006-pipeline-reproducibility.md).)

---

## 10. What Constitutes a Valid Pipeline Execution

A pipeline execution is **valid** when all of the following hold. Items already
true today are marked CURRENT; items requiring Sprint 4 work are marked TARGET.

**Structural validity (TARGET):**

1. `dvc.yaml` models the DAG `raw → preprocess → processed → train → model →
   evaluate → metrics`, with `train` depending on the processed data and
   `evaluate` depending on the model and evaluation data.
2. Every `dvc.yaml` parameter key exists in `params.yaml`; no orphaned params.
3. Each stage reads only its declared inputs and writes only the artifact(s) it
   owns.
4. A declared metrics artifact is produced by `evaluate`.

**Execution validity (partly CURRENT):**

5. Each stage exits non-zero on failure with a typed, actionable error, stopping
   `dvc repro`/CI. *(CURRENT — provided by `stage_runner` + typed exceptions.)*
6. Required configuration is present and validated before compute. *(CURRENT — via
   `load_params` / `require_env`.)*
7. Given identical inputs, params, and seed, the run is reproducible. *(TARGET.)*

**Evidence validity (TARGET):**

8. Metrics are emitted as a first-class artifact, not only to logs/MLflow.
9. Any performance claim is backed by a held-out evaluation (see §8).
10. `dvc status` reports "up to date" for an unchanged, committed pipeline
    (`dvc.lock` present and consistent).

An execution that logs an `accuracy` number but violates the evaluation boundary
(§8), or that cannot be reproduced (§7), is **not** a valid execution under this
contract even if the process exits 0.

---

## 11. Current Deviations Summary (Current → Target)

| ID | Deviation (CURRENT) | TARGET | Sprint 4 PR |
|----|---------------------|--------|-------------|
| D1 | `preprocess` output `data/processed/data.csv` is not consumed; `train` reads raw data. | `train` consumes the processed dataset. | PR 2 |
| D2 | `dvc.yaml` `train` params (`train.data`/`train.model`) don't match `params.yaml` (`train.input`/`train.output`) or the code. | One authoritative parameter naming. | PR 2 / PR 3 |
| D3 | `evaluate` reads a `test` section; `dvc.yaml` declares no evaluate params. | Stage-aligned, declared evaluation params. | PR 2 / PR 3 |
| D4 | No metrics artifact is produced (MLflow/log only). | Declared, DVC-tracked metrics artifact. | PR 2 / PR 4 |
| D5 | `evaluate` scores on the full training data (in-sample). | Held-out evaluation set. | PR 4 |
| D6 | `train`/`evaluate` require `MLFLOW_TRACKING_URI`; ML logic is coupled to the service, blocking isolated tests. | MLflow isolated behind a boundary; unit tests need no external service. | PR 4 / PR 5 |
| D7 | Training is non-deterministic; `random_state` unused; no `dvc.lock`. | Seeded, deterministic, locked, CI-validated. | PR 4 / PR 6 |
| D8 | Processed CSV written without header, incompatible with the downstream `Outcome` column contract. | Consistent column contract across stage boundaries. | PR 3 |

---

## 12. Out of Scope

This contract does **not** cover, and Sprint 4 does not deliver: Kubernetes, Helm,
Terraform, cloud deployment, continuous delivery, model serving, Prometheus/Grafana,
distributed training, a production MLflow deployment, container registry publishing,
Trivy/SBOM/image signing. These belong to later milestones (see
[roadmap.md](roadmap.md) and the Sprint 4 plan).

This document also does not prescribe the *internal algorithm* of preprocessing or
the *model family* — only the contracts (inputs, outputs, ownership, boundaries,
reproducibility) around them.

---

## 13. Change Control

- This contract is versioned with the repository. Material changes to any stage's
  input/output, ownership, evaluation boundary, or external-service posture must
  update this document in the same PR.
- Where implementation and this contract diverge, **the divergence is a defect** —
  either the implementation or the contract is corrected; they are not allowed to
  drift silently.
- The decision rationale behind this contract lives in
  [ADR-006](decisions/ADR-006-pipeline-reproducibility.md).
</content>
</invoke>
