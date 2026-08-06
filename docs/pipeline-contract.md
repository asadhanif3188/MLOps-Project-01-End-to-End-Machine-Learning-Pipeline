# Pipeline Contract

- **Status:** Active (as-built) — reconciled to the implementation delivered in
  Sprint 4 (v1.3.0)
- **Date:** 2026-08-06 (originally drafted 2026-08-05 as a design contract)
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

It began (Sprint 4, PR 1) as a **design contract** distinguishing what the
repository did then (CURRENT) from what the sprint would implement (TARGET). The
implementation PRs (PR 2–PR 6) have since landed, so this document now describes
the **as-built** pipeline. The wiring and configuration assertions below are
**enforced automatically** by the `contract` test suite
([`tests/contract/test_pipeline_contract.py`](../tests/contract/test_pipeline_contract.py))
and by CI (`dvc dag` + `dvc status`), so this is not a claim on trust — a broken
contract fails a pull request.

Two deviations from the original target **remain open and are documented
honestly** rather than hidden: evaluation is still **in-sample** (D5) and no
`dvc.lock` is committed (part of D7). See [§8](#8-evaluation-boundary) and
[§11](#11-deviation-status-sprint-4).

---

## 2. Logical Pipeline

The pipeline is a linear artifact-lineage chain:

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
a pipeline stage (see [§9](#9-external-service-boundaries)). Data and artifact
lineage is owned by **DVC** (see
[ADR-006](decisions/ADR-006-pipeline-reproducibility.md)).

**As wired today** (`dvc.yaml`), confirmed by `dvc dag`:

```text
data/raw/data.csv.dvc
        │
        ▼
   preprocess ──▶ data/processed/data.csv
                        │
              ┌─────────┴──────────┐
              ▼                    ▼
           train ──▶ model ──▶ evaluate ──▶ metrics/metrics.json
```

`train` and `evaluate` both consume the **processed** dataset; `preprocess` is
the only stage that reads the raw file; `evaluate` additionally consumes the
model and produces the metrics artifact. The single linear chain the architecture
always described is now the chain DVC actually builds.

---

## 3. Stage Contracts

Each stage documents its **purpose**, **inputs**, **outputs**, **configuration**,
**artifact ownership**, and **failure conditions**. Paths and parameter names are
stated exactly as they appear in the repository.

### 3.1 `preprocess`

**Purpose:** Produce the processed dataset that training consumes, from the raw
dataset.

| Aspect | As built |
|--------|----------|
| Command | `python src/preprocess.py` |
| Input (data) | `data/raw/data.csv` |
| Input (code) | `src/preprocess.py` |
| Config keys | `preprocess.input`, `preprocess.output` (from `params.yaml`) |
| Output | `data/processed/data.csv` — **consumed by `train`** |
| DVC declaration | `deps: data/raw/data.csv, src/preprocess.py`; `params: preprocess.input, preprocess.output`; `outs: data/processed/data.csv` |
| Behavior | Reads the raw CSV and re-writes it **with its header** (`header=True, index=False`); performs no cleaning or transformation today. The header is preserved so `train`/`evaluate` can select the `Outcome`/feature columns by name (resolving the former headerless-output mismatch, D8). |

**Artifact ownership:** `preprocess` **owns** `data/processed/data.csv`. No other
stage writes it.

**Failure conditions (via typed exceptions):**

- `DataError` — raw input missing, empty, or not valid CSV.
- `DataError` — processed output directory/file cannot be written.
- `ConfigError` — `params.yaml` missing, unparseable, or missing
  `preprocess.input`/`preprocess.output`.

### 3.2 `train`

**Purpose:** Fit and select a model from the training dataset and persist it as
the model artifact.

| Aspect | As built |
|--------|----------|
| Command | `python src/train.py` |
| Input (data) | `data/processed/data.csv` — the `preprocess` output, read from `params['train']['input']` |
| Input (code) | `src/train.py` |
| Config keys read by code | `train.input`, `train.output`, `train.target`, `train.random_state`, `train.n_estimators`, `train.max_depth` |
| Config keys declared in `dvc.yaml` | `train.input`, `train.output`, `train.target`, `train.random_state`, `train.n_estimators`, `train.max_depth` — the same set the code reads |
| Output | `models/model.pkl` |
| DVC declaration | `deps: data/processed/data.csv, src/train.py`; `outs: models/model.pkl` |
| Behavior | Requires `MLFLOW_TRACKING_URI`; splits with `train_test_split(test_size=0.20, random_state=...)`; builds a `RandomForestClassifier` with the configured `n_estimators`/`max_depth`/`random_state`; runs `GridSearchCV` (cv=3) over leaf/split regularization (`min_samples_split`, `min_samples_leaf`); logs params/metrics/artifacts to MLflow and conditionally registers the model; pickles the best estimator. The ML computation (`run_training`) is IO-free and MLflow-free. |

**Artifact ownership:** `train` **owns** `models/model.pkl`. It also owns its
MLflow run (params, metrics, model registry entry).

**Failure conditions (via typed exceptions):**

- `DataError` — dataset missing/empty/invalid, or missing the `target` column.
- `ConfigError` — `MLFLOW_TRACKING_URI` unset/empty; or required `train.*` params
  absent.
- `TrackingError` — MLflow logging fails (URI, credentials, or network).
- `ModelError` — the fitted estimator cannot be pickled.

**Configured parameters govern the run.** `random_state` seeds both the split and
the estimator; `n_estimators`/`max_depth` are set on the estimator (no longer
shadowed by a hardcoded grid). These parameters are therefore live, and training
is deterministic given the same inputs and parameters (resolving the former inert
parameters / non-determinism, C5 / D7-seeding).

### 3.3 `evaluate`

**Purpose:** Measure the trained model against an evaluation dataset and emit
metrics.

| Aspect | As built |
|--------|----------|
| Command | `python src/evaluate.py` |
| Input (data) | `data/processed/data.csv` — read from `params['evaluate']['data']` |
| Input (model) | `models/model.pkl` — the `train` output, read from `params['evaluate']['model']` |
| Input (code) | `src/evaluate.py` |
| Config keys read by code | `evaluate.data`, `evaluate.model`, `evaluate.target`, `evaluate.metrics` (section named `evaluate`, aligned with the stage) |
| Config keys declared in `dvc.yaml` | `evaluate.data`, `evaluate.model`, `evaluate.target`, `evaluate.metrics` |
| Output | `metrics/metrics.json` — a DVC-tracked **metrics artifact** (`cache: false`) |
| DVC declaration | `deps: data/processed/data.csv, models/model.pkl, src/evaluate.py`; `params: evaluate.data, evaluate.model, evaluate.target, evaluate.metrics`; `metrics: metrics/metrics.json (cache: false)` |
| Behavior | Requires `MLFLOW_TRACKING_URI`; loads the model; predicts over the dataset; computes `accuracy_score`; writes the metrics artifact **before** the MLflow boundary; logs the metric to MLflow. The scoring (`compute_metrics`) is IO-free and MLflow-free. |

**Artifact ownership:** `evaluate` **owns** `metrics/metrics.json` and its MLflow
evaluation run.

**Failure conditions (via typed exceptions):**

- `DataError` — dataset missing/empty/invalid, or missing the `target` column.
- `ConfigError` — `MLFLOW_TRACKING_URI` unset/empty; or required `evaluate.*`
  params absent.
- `ModelError` — model file missing/corrupt, or the model fails to predict on the
  provided features (e.g. a feature-schema mismatch).
- `TrackingError` — MLflow logging fails (URI, credentials, or network).

**Evaluation boundary (OPEN limitation).** `evaluate.data` currently points at the
same processed dataset `train` fits on, so the reported `accuracy` is an
**in-sample** figure and must not be presented as a generalization estimate. See
[§8](#8-evaluation-boundary).

---

## 4. Configuration & Parameter Contract

**`params.yaml` is the single authoritative source of pipeline parameters.**
`dvc.yaml` references parameter *keys*; the stage code reads parameter *values*.
All three agree — and the `contract` tests enforce that agreement on every change.

### 4.1 Current `params.yaml` sections

```yaml
preprocess:
  input:  data/raw/data.csv
  output: data/processed/data.csv

train:
  input:  data/processed/data.csv
  output: models/model.pkl
  target: Outcome
  random_state: 42
  n_estimators: 100
  max_depth: 5

evaluate:
  data:   data/processed/data.csv
  model:  models/model.pkl
  target: Outcome
  metrics: metrics/metrics.json
```

### 4.2 Configuration contract — status

The five inconsistencies recorded in the original design contract (C1–C5) are all
resolved. They are retained here as a verification checklist, each now enforced by
a named `contract` test:

| # | Former inconsistency | Status | Enforced by |
|---|----------------------|--------|-------------|
| C1 | `dvc.yaml` `train` declared `train.data`/`train.model`, absent from `params.yaml`. | **Resolved** — `dvc.yaml` references `train.input`/`train.output`, which exist. | `test_every_dvc_param_key_exists_in_params_yaml` |
| C2 | `train.py` read `train.input`/`train.output` while `dvc.yaml` named other keys. | **Resolved** — code, `params.yaml`, and `dvc.yaml` use one naming. | same as C1 + `test_declared_outputs_match_params` |
| C3 | `evaluate.py` read a section named `test`, not `evaluate`. | **Resolved** — the section is `evaluate`, aligned with the stage. | `test_no_orphaned_params` |
| C4 | `dvc.yaml` `evaluate` declared no params. | **Resolved** — `evaluate.data/model/target/metrics` are declared. | `test_no_orphaned_params` |
| C5 | `train.random_state`/`n_estimators`/`max_depth` were loaded but not applied. | **Resolved** — all three govern the split/estimator. | covered by stage unit tests (`tests/unit/test_train.py`) |

### 4.3 Parameter contract rules (enforced)

1. Every parameter a stage reads is defined in `params.yaml`.
2. Every parameter key `dvc.yaml` references exists in `params.yaml`.
3. No parameter is declared for a stage yet referenced by no stage (**no orphaned
   params**).
4. Parameter section names correspond to their stage names.
5. Parameters that affect reproducibility (e.g. seeds) are declared and applied.

---

## 5. Artifact Ownership & Lineage

Each artifact has exactly **one** producing stage ("owner"). Consumers may read it
but never write it. `test_each_artifact_has_exactly_one_producer` enforces the
single-owner rule; `test_processed_data_is_consumed_not_orphaned` enforces that
`preprocess`'s output has a downstream consumer.

| Artifact | Owner (writes) | Consumers (read) | Tracking |
|----------|----------------|------------------|----------|
| `data/raw/data.csv` | External / ingestion (`data/raw/data.csv.dvc`) | `preprocess` only | DVC (`.dvc` pointer) |
| `data/processed/data.csv` | `preprocess` | `train`, `evaluate` | DVC stage output |
| `models/model.pkl` | `train` | `evaluate` | DVC stage output |
| `metrics/metrics.json` | `evaluate` | downstream reporting / CI | DVC metric (`cache: false`) |
| MLflow run (params, metrics, registry) | `train`, `evaluate` | MLflow/DagsHub UI | MLflow (external) |

**Rules:**

- A stage writes only the artifact(s) it owns.
- The raw dataset is owned upstream of the pipeline (DVC-tracked pointer) and is
  not mutated by any stage.
- The raw dataset has exactly one direct consumer (`preprocess`); training and
  evaluation consume the derived processed artifact, not the raw file.

---

## 6. Stage Input/Output Summary

| Stage | Input | Output | Configuration |
|-------|-------|--------|---------------|
| `preprocess` | Raw dataset | Processed dataset | `preprocess.input`, `preprocess.output` |
| `train` | Processed dataset | Model artifact | `train.*` (target + seed + tree hyperparameters) |
| `evaluate` | Model + evaluation dataset | Metrics artifact | `evaluate.*` (data, model, target, metrics) |

The one remaining gap versus the ideal contract is that the `train` "input" and
the `evaluate` "evaluation dataset" are currently the **same** processed file
(in-sample evaluation) — see [§8](#8-evaluation-boundary).

---

## 7. Reproducibility Expectations

Reproducibility is an **engineering requirement**, not a nicety (rationale in
[ADR-006](decisions/ADR-006-pipeline-reproducibility.md)).

**Achieved:**

1. **Declared lineage.** The pipeline is reconstructable from declared inputs,
   parameters, and dependencies. `dvc.yaml` accurately models the real DAG
   `raw → preprocess → processed → train → model → evaluate → metrics`, confirmed
   by `dvc dag` and the `contract` lineage tests.
2. **Deterministic given fixed inputs + params + seed.** The data split and the
   estimator are seeded from `train.random_state`; repeated runs on the same
   inputs and params yield the same model and metrics.
3. **CI enforcement.** Pipeline integrity (valid DAG, consistent params, buildable
   graph, correct lineage) is validated automatically on every push and pull
   request, without production credentials or live external services (see
   [CI/CD](ci-cd.md)).

**Remaining reproducibility limitations (documented, not fixed):**

4. **No committed `dvc.lock`.** The resolved dependency hashes for a run are not
   pinned, so `dvc status` cannot serve as a true "up to date" drift gate and CI
   validates the pipeline **definition**, not a locked **execution**. Committing a
   `dvc.lock` requires a runnable dataset in CI (the raw dataset is remote-only
   today).
5. **No in-CI execution.** CI does not run `dvc repro` — the first stage's data
   dependency lives only on the DagsHub remote. A future execution check would use
   a small committed fixture dataset plus a committed `dvc.lock`.
6. **Dependencies/base image are name-pinned, not digest-pinned** (carried from
   Sprint 3). Byte-for-byte repeatable rebuilds need hash/digest pinning
   ([ADR-005](decisions/ADR-005-containerization-strategy.md)). Until then,
   "reproducible" means *logically* reproducible (same declared inputs, params,
   and seed), not bit-for-bit.

---

## 8. Evaluation Boundary

The evaluation boundary defines **what data proves what claim.**

**As built:**

- `train` performs an internal `train_test_split` (20% test, seeded) purely to
  report an in-training accuracy; that split is not persisted or exported.
- `evaluate` scores the model over the **entire** `data/processed/data.csv` — the
  same dataset the model was trained on.
- Consequently the `accuracy` written to `metrics/metrics.json` and logged to
  MLflow is an **in-sample** metric.

**Honest-evaluation rule (enforced by documentation and code comments):** no
model-performance claim is made unless the methodology supports it. Until a genuine
held-out split exists, any reported accuracy is labeled **in-sample** and must not
be cited as a generalization estimate.

**Path to a held-out split (deviation D5, still open).** Because `evaluate.data` is
already an explicit, configured input — and `evaluate` has no dependency on train's
in-memory split — turning this into a held-out evaluation is a
**configuration/graph change, not a code change**: add a splitting step (or a
second processed artifact) and point `evaluate.data` at the held-out portion. This
is the single most valuable remaining pipeline-correctness improvement and is
tracked for a future sprint.

---

## 9. External Service Boundaries

The pipeline touches two external systems. Both are **boundaries**, isolated from
core ML logic.

| Boundary | Used by | Purpose | Required to run? | Testability |
|----------|---------|---------|------------------|-------------|
| **MLflow on DagsHub** | `train`, `evaluate` | Experiment tracking, metric logging, model registry | **Yes to run a stage end-to-end** — both stages call `require_env("MLFLOW_TRACKING_URI")`. | **Not required for tests.** All MLflow access is funneled through [`src/tracking.py`](../src/tracking.py), imported *lazily* at the boundary. The stages' ML computation (`run_training`, `compute_metrics`) imports no MLflow, and tests replace `tracking` with an in-memory stub. |
| **DVC remote (S3-compatible, DagsHub)** | data/model retrieval | Store/fetch DVC-tracked data & models | Needed to `dvc pull`/`push` artifacts; **not** needed to reason about or validate the graph. | Graph validation in CI (`dvc dag`, local `dvc status`, contract tests) requires no remote credentials. |

**Environment/config boundary:**

- `MLFLOW_TRACKING_URI` (and DagsHub credentials) are provided via `.env`
  (`python-dotenv`); see [`.env.example`](../.env.example). Secrets are never
  committed.
- **Contract rule (enforced):** ordinary **unit tests do not require** live MLflow,
  DagsHub, network access, or production credentials. The `stub_tracking` fixture
  neutralizes the tracking boundary; the `contract` tests parse files only. Tests
  that genuinely exercise external behavior would be **integration** tests, marked
  as such. (Rationale: [ADR-006](decisions/ADR-006-pipeline-reproducibility.md).)

> **Nuance, stated honestly.** The MLflow *coupling* was removed from the ML
> *computation*, which is what makes the stages unit-testable. The stage
> *orchestration* (`train()` / `evaluate()`) still calls `require_env` and will
> refuse to run end-to-end without `MLFLOW_TRACKING_URI` set. That is by design —
> MLflow logging is a preserved capability, not an optional one — and it does not
> weaken testability, because tests exercise the pure functions and the stubbed
> boundary, not the network.

---

## 10. What Constitutes a Valid Pipeline Execution

A pipeline execution is **valid** when all of the following hold.

**Structural validity (enforced by `contract` tests + `dvc dag` in CI):**

1. `dvc.yaml` models the DAG `raw → preprocess → processed → train → model →
   evaluate → metrics`, with `train` depending on the processed data and
   `evaluate` depending on the model and evaluation data.
2. Every `dvc.yaml` parameter key exists in `params.yaml`; no orphaned params.
3. Each stage reads only its declared inputs and writes only the artifact(s) it
   owns.
4. A declared metrics artifact is produced by `evaluate`.

**Execution validity:**

5. Each stage exits non-zero on failure with a typed, actionable error, stopping
   `dvc repro`/CI (via `stage_runner` + typed exceptions).
6. Required configuration is present and validated before compute (via
   `load_params` / `require_env`).
7. Given identical inputs, params, and seed, the run is reproducible (seeded
   split and estimator).

**Evidence validity:**

8. Metrics are emitted as a first-class artifact (`metrics/metrics.json`), not only
   to logs/MLflow.
9. Any performance claim is backed by an evaluation whose methodology supports it.
   **Today the evaluation is in-sample (§8), so the current accuracy supports no
   generalization claim.**
10. *(Not yet achievable)* `dvc status` reports "up to date" for an unchanged,
    committed pipeline — this needs a committed `dvc.lock`, which is absent (§7).

An execution that logs an `accuracy` number but violates the evaluation boundary
(§8) is **not** a valid basis for a generalization claim even though the process
exits 0 and the pipeline is structurally correct.

---

## 11. Deviation Status (Sprint 4)

The eight deviations the design contract enumerated, with their status after the
Sprint 4 implementation PRs:

| ID | Deviation (original) | Status after Sprint 4 |
|----|----------------------|-----------------------|
| D1 | `preprocess` output orphaned; `train` read raw data. | ✅ **Resolved** — `train` consumes `data/processed/data.csv`; raw is `preprocess`-only. |
| D2 | `dvc.yaml` `train` params (`train.data`/`train.model`) mismatched `params.yaml`/code. | ✅ **Resolved** — one authoritative naming (`train.input`/`train.output`). |
| D3 | `evaluate` read a `test` section; no evaluate params in `dvc.yaml`. | ✅ **Resolved** — `evaluate` section; params declared in the graph. |
| D4 | No metrics artifact (MLflow/log only). | ✅ **Resolved** — DVC-tracked `metrics/metrics.json`. |
| D5 | `evaluate` scores on the full training data (in-sample). | ⬜ **Open** — still in-sample; a held-out split is a future config change (§8). |
| D6 | ML logic coupled to MLflow, blocking isolated tests. | ✅ **Resolved** — MLflow isolated behind `tracking.py` (lazy import); ML compute is unit-tested without the service. Stage orchestration still requires the URI to *run* end-to-end (by design; §9). |
| D7 | Non-deterministic training; `random_state` unused; no `dvc.lock`. | 🟡 **Partially resolved** — seeded and deterministic; `random_state` applied. **`dvc.lock` still absent** (§7). |
| D8 | Processed CSV written headerless, incompatible with the `Outcome` column contract. | ✅ **Resolved** — written `header=True`; consumable by `train`/`evaluate`. |

**Summary:** six of eight deviations fully resolved; D7 partially resolved
(determinism yes, `dvc.lock` no); D5 (in-sample evaluation) remains open. D5 and
the `dvc.lock`/execution-check portion of D7 are the pipeline's known limitations
after Sprint 4.

---

## 12. Out of Scope

This contract does **not** cover, and Sprint 4 did not deliver: Kubernetes, Helm,
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
  drift silently. The `contract` test suite makes most such divergences fail CI
  automatically.
- The decision rationale behind this contract lives in
  [ADR-006](decisions/ADR-006-pipeline-reproducibility.md).
