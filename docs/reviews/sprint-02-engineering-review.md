# Sprint-02 Engineering Review — Production-Readiness Assessment

**Project:** MLOps-Project-01 — End-to-End Machine Learning Pipeline
**Sprint:** 02 — Engineering Excellence
**Reviewer role:** Principal Python Engineer (production-readiness review)
**Scope:** Full repository read. Source code, pipeline definition, configuration, dependency management, documentation, and repository hygiene.
**Constraint:** Assessment only. **No source code was modified.** No fixes were implemented.

---

## 1. Executive Summary

The repository is a compact, batch ML pipeline (three stage scripts — `preprocess`, `train`, `evaluate` — orchestrated by DVC, tracked by MLflow on DagsHub). Following the Sprint-01 merge, it now carries **excellent documentation and governance** (ADRs, architecture, roadmap, contributing/security policies, editorconfig, issue/PR templates). The documentation layer is genuinely strong and, to its credit, already discloses several of the code-level defects catalogued below.

The **engineering substance of the code, however, is at prototype maturity.** The three scripts work as a demo but are not production-grade: there are no tests, no logging framework, no exception handling, effectively no type hints, unpinned dependencies, and a set of correctness defects that undermine the project's headline claim of *reproducibility*. Most critically:

- The pipeline **does not train on its own preprocessed output** — `preprocess` is a disconnected dead stage.
- Training is **non-deterministic** despite a `random_state` parameter that is loaded but never used, directly contradicting the project's reproducibility goal.
- `dvc.yaml` references **parameter keys that do not exist** in `params.yaml`, which breaks `dvc repro` parameter tracking.
- The model is evaluated **on its own training data**, making the reported accuracy meaningless as a generalization estimate.
- Models are loaded via **`pickle`** with no integrity control — an arbitrary-code-execution vector.

None of these are large in code volume; the entire `src/` tree is well under 200 lines. The remediation effort is therefore **modest (estimated 5–8 engineering days total)** and aligns exactly with the Sprint-02 scope (logging, exceptions, type hints, testing, tooling). This review prioritizes the work so that correctness and reproducibility are restored first, then observability and testability, then polish.

### Overall Ratings (current state)

| Dimension | Rating (0–10) | Note |
|-----------|---------------|------|
| Documentation & governance | 9 | Excellent; a genuine strength |
| Project structure | 7 | Clean and conventional for its size |
| Code correctness / reproducibility | 3 | Multiple defects undermine the core claim |
| Observability (logging) | 2 | `print()` only |
| Robustness (exception handling) | 1 | None |
| Type safety | 1 | Effectively absent |
| Testability / tests | 1 | No tests, tight IO coupling |
| Dependency reproducibility | 2 | Fully unpinned |
| Security posture | 4 | Good secret hygiene; `pickle` + unpinned deps drag it down |

### Findings by Priority

| Priority | Count | Theme |
|----------|-------|-------|
| **Critical** | 4 | Reproducibility & pipeline correctness defects |
| **High** | 7 | Logging, exceptions, tests, dependency pinning, type hints, packaging, `pickle` |
| **Medium** | 8 | Config duplication, resource leaks, dead code, magic numbers, doc/code drift |
| **Low** | 5 | Naming, typos, tooling, minor polish |

---

## 2. Methodology

Every source file, the DVC pipeline and params, the DVC/S3 remote config, dependency manifest, environment template, and the full Sprint-01 documentation set were read. Findings were cross-checked against the claims made in `docs/architecture.md`, `docs/project-structure.md`, and `SECURITY.md`. Where the documentation already acknowledges a gap, the finding notes it — a documented defect is still a defect for Sprint-02's purposes, since this sprint's mandate is to *fix* engineering quality, not merely describe it.

Each finding is reported with: **Description**, **Why it matters**, **Risk**, **Recommended solution**, **Estimated effort**, and **Engineering impact**.

Effort key: **S** ≈ ≤0.5 day · **M** ≈ 0.5–1.5 days · **L** ≈ 2–4 days.

---

## 3. Critical Findings

These undermine the project's stated goals of **reproducibility** and **correctness**. They should be addressed before any observability or tooling work, because they change program behavior and outputs.

### C-1. `random_state` is loaded but never applied — training is non-deterministic

- **Description:** `params.yaml` defines `train.random_state: 42`, and `train.py` accepts a `random_state` argument (`src/train.py:36`), but it is never passed to `train_test_split` (`src/train.py:58`) or to `RandomForestClassifier` (`src/train.py:18`). The split and the model are seeded from a fresh entropy source on every run.
- **Why it matters:** Reproducibility is the *headline goal* of this project (README §Goals; `docs/architecture.md` §1). Two runs on identical data and params will produce different splits, different best estimators, and different accuracy. The MLflow-logged metrics are therefore not comparable across runs, defeating the purpose of experiment tracking.
- **Risk:** High. Silent, non-obvious. Every experiment comparison and every DVC-cached model is unreproducible; regressions cannot be reliably detected.
- **Recommended solution:** Thread `random_state` into `train_test_split(..., random_state=random_state)` and `RandomForestClassifier(random_state=random_state)`. Consider also fixing `GridSearchCV`/CV shuffling seeds where applicable.
- **Estimated effort:** S
- **Engineering impact:** Restores the project's core reproducibility guarantee for a one-line-per-call change; high value-to-effort ratio.

### C-2. `dvc.yaml` references parameters that do not exist in `params.yaml`

- **Description:** The `train` stage in `dvc.yaml:18-23` declares params `train.data`, `train.model`, `train.max_depth`, `train.n_estimators`, `train.random_state`. But `params.yaml` defines the train keys as `input` and `output` (not `data`/`model`). `train.py:126-127` reads `params['input']`/`params['output']`. So `train.data` and `train.model` are declared dependencies that **do not exist**.
- **Why it matters:** DVC uses declared params to decide when a stage is stale. Referencing non-existent keys causes `dvc repro`/`dvc params` to error or to mis-track changes, breaking the reproducible-pipeline promise. The three-way disagreement (dvc.yaml ↔ params.yaml ↔ train.py) is a latent correctness bug.
- **Risk:** High. Breaks the automated pipeline that is the reason DVC was adopted (`docs/decisions/ADR-003`).
- **Recommended solution:** Choose one canonical key naming (recommend `input`/`output` to match the code, or rename consistently across all three files) and align `dvc.yaml`, `params.yaml`, and the scripts. Add a CI check that runs `dvc status`/`dvc repro --dry` to catch drift.
- **Estimated effort:** S
- **Engineering impact:** Makes the DVC pipeline actually reproducible and CI-verifiable.

### C-3. The `preprocess` stage output is never consumed — a disconnected dead stage

- **Description:** `preprocess` writes `data/processed/data.csv` (`src/preprocess.py:19`), but both `train` and `evaluate` read `data/raw/data.csv` (`params.yaml` `train.input`, `test.data`; confirmed in `dvc.yaml` deps). The processed dataset is produced and versioned but never used. Additionally, `preprocess` writes with `header=None` (`src/preprocess.py:19`), which strips the header row — so if `train.py` *did* consume it, `data.drop(columns=['Outcome'])` (`src/train.py:51`) would raise `KeyError`.
- **Why it matters:** The pipeline graph is a lie: the architecture diagram (`docs/architecture.md` §2) shows `preprocess → train`, but the real data flow bypasses preprocessing. Any preprocessing logic added later would silently have zero effect on the model.
- **Risk:** High. Structurally misleading; guarantees future preprocessing work is wasted until the wiring is fixed. The `header=None` detail is a latent crash.
- **Recommended solution:** Point `train`/`evaluate` at `data/processed/data.csv`, and make `preprocess` write a header (`header=True`) so downstream column-name access works. Update `dvc.yaml` deps so `train` depends on the processed output, making the DAG honest.
- **Estimated effort:** S–M
- **Engineering impact:** Turns three isolated scripts into a genuine dependency-ordered pipeline.

### C-4. Model is evaluated on its own training data

- **Description:** `evaluate.py` loads the full dataset (`test.data: data/raw/data.csv`) and scores the model over all of it (`src/evaluate.py:12-22`). This is the same data the model was trained on (`train.input: data/raw/data.csv`). There is no held-out evaluation set.
- **Why it matters:** The `accuracy` metric logged to MLflow is an in-sample score and materially overstates real-world performance. As a portfolio artifact demonstrating MLOps competence, reporting training accuracy as "model accuracy" is a credibility risk to a knowledgeable reviewer.
- **Risk:** Medium–High. Misleading metrics; encourages overfitting; undermines the experiment-tracking value proposition.
- **Recommended solution:** Persist the train/test split (e.g. write the test split to `data/processed/`, or seed the split deterministically per C-1 and re-derive it) and evaluate strictly on held-out data. Log both train and test accuracy explicitly labelled.
- **Estimated effort:** M
- **Engineering impact:** Produces trustworthy metrics — the entire point of the MLflow integration.

> **Note:** C-2, C-3, and C-4 are candidly pre-documented as "known gaps" in `docs/architecture.md` §3 and `docs/project-structure.md`. That transparency is commendable, but Sprint-02 is the sprint that resolves them.

---

## 4. High-Priority Findings

These are the core Sprint-02 engineering-quality deliverables (logging, exceptions, tests, type hints, tooling) plus dependency and serialization safety.

### H-1. No logging framework — `print()` used throughout

- **Description:** All diagnostic output uses `print()` with hand-written `"INFO:"` prefixes (`src/train.py:26,78,115`; `src/evaluate.py:25`; `src/preprocess.py:21`). There is no central logging configuration, no log levels, no timestamps, no module context.
- **Why it matters:** `print()` cannot be filtered by severity, redirected, formatted structurally, or silenced in tests. It writes to stdout, mixing diagnostics with real program output. The Sprint-02 standard (§7) explicitly requires INFO/WARNING/ERROR/DEBUG levels and structured messages.
- **Risk:** Medium. Blocks observability; every downstream deployment (Docker/K8s in later sprints) expects structured logs.
- **Recommended solution:** Add a `src/logging_config.py` (or a small `utils` package) exposing a configured `logging.Logger` with a consistent formatter; replace all `print()` calls; use levels per the Sprint-02 standard. Enable a lint rule to forbid `print` in `src/`.
- **Estimated effort:** M
- **Engineering impact:** Foundational for every future operational sprint; high portfolio value.

### H-2. No exception handling anywhere

- **Description:** There is not a single `try`/`except` in the codebase. File reads (`pd.read_csv`, `open`), model deserialization (`pickle.load`), MLflow network calls, and environment lookups all fail with raw tracebacks. Environment variables are accessed with `os.environ['MLFLOW_TRACKING_URI']` (`src/train.py:54`, `src/evaluate.py:16`), which raises a bare `KeyError` with no actionable message when unset.
- **Why it matters:** Sprint-02 standards (§7) require: never swallow exceptions, log before raising, user-friendly messages, and a central exception hierarchy. Today a missing `.env`, an unreachable DagsHub endpoint, or a missing data file produces an opaque stack trace with no guidance.
- **Risk:** Medium–High. Poor operability; hard-to-diagnose failures; bad first-run experience for new contributors.
- **Recommended solution:** Introduce a small custom exception hierarchy (e.g. `PipelineError`, `ConfigError`, `DataError`, `TrackingError`) in `src/exceptions.py`; wrap IO/config/network boundaries; validate required env vars up front with a clear message (e.g. "MLFLOW_TRACKING_URI is not set — copy .env.example to .env"). Log before raising.
- **Estimated effort:** M
- **Engineering impact:** Turns cryptic failures into actionable errors; prerequisite for reliable automation.

### H-3. No automated tests

- **Description:** There is no `tests/` directory and no test framework configured. Sprint-02 (Epic 6) requires a testing foundation (smoke tests, critical-component tests, fixtures).
- **Why it matters:** With zero tests, every refactor in this sprint (logging, exceptions, type hints) is unverified. Regression risk is high precisely when the codebase is about to change the most.
- **Risk:** High. No safety net for the refactors this very sprint mandates.
- **Recommended solution:** Add `pytest`; write smoke tests for each stage's pure logic (e.g. preprocess output shape/schema, train produces a fitted estimator on a tiny fixture, evaluate returns a float in [0,1]); provide a small synthetic CSV fixture; mock MLflow so tests are fast, independent, and deterministic (Sprint-02 §7). This finding is coupled to H-6 (testability requires decoupling IO).
- **Estimated effort:** M–L
- **Engineering impact:** Enables safe iteration; single highest-leverage reliability investment.

### H-4. Dependencies are completely unpinned

- **Description:** `requirements.txt` lists six packages (`dvc`, `dagshub`, `scikit-learn`, `mlflow`, `dvc-s3`, `python-dotenv`) with **no version constraints**. `pandas` is used directly (`src/*.py`) but is not declared — it is pulled in only transitively.
- **Why it matters:** Unpinned dependencies directly contradict the reproducibility goal: a fresh `pip install` next month may resolve different major versions of scikit-learn/MLflow, changing model behavior or breaking APIs. A pickled scikit-learn model is version-sensitive — an unpinned sklearn can make an existing `model.pkl` unloadable. The undeclared `pandas` dependency is a latent install failure.
- **Risk:** High. Non-reproducible environments; silent behavioral drift; broken model artifacts.
- **Recommended solution:** Pin with compatible-release specifiers (`scikit-learn>=1.x,<1.y`, etc.), declare `pandas` explicitly, and add a lockfile (`pip-tools`/`uv`/`requirements.lock`). Split dev tooling into `requirements-dev.txt` (pytest, ruff). Record the Python version constraint.
- **Estimated effort:** S–M
- **Engineering impact:** Makes environments reproducible and models loadable across machines/CI.

### H-5. Effectively no type hints

- **Description:** No function in `src/` has type annotations on parameters or return values (`preprocess`, `hyperparameter_tuning`, `train`, `evaluate`). Sprint-02 Epic 5 requires type hints.
- **Why it matters:** Type hints improve IDE support, catch a class of bugs statically (e.g. the `random_state` mis-wiring in C-1 is exactly the kind of thing type-aware review surfaces), and document intent. They are a stated success metric for this sprint.
- **Risk:** Low–Medium. No runtime impact, but weak developer experience and maintainability.
- **Recommended solution:** Annotate all signatures (`def preprocess(input_path: str, output_path: str) -> None:`, etc.); add `mypy` or `pyright` in non-strict mode to CI; type the params dict via a `TypedDict` or a `@dataclass(frozen=True)` config object (see M-1).
- **Estimated effort:** S–M
- **Engineering impact:** Better tooling and safer refactors; directly satisfies a sprint success metric.

### H-6. Tight coupling of pure logic to IO / side effects harms testability

- **Description:** Each function mixes computation with filesystem IO, MLflow network calls, and environment access. `train()` reads a file, sets a global tracking URI, starts an MLflow run, and pickles to disk — all in one function (`src/train.py:34-115`). There are no seams to inject fakes.
- **Why it matters:** This is the structural reason H-3 is hard: you cannot unit-test `train` without a network, a filesystem, and MLflow. Sprint-02 requires fast, independent, deterministic tests (§7).
- **Risk:** Medium. Blocks the testing epic; makes future reuse (e.g. serving) difficult.
- **Recommended solution:** Separate pure logic (data loading, splitting, fitting, scoring) from side effects (MLflow logging, disk IO). Pass the tracking URI and paths as arguments rather than reading globals inside functions. This makes the core logic trivially unit-testable and MLflow mockable.
- **Estimated effort:** M
- **Engineering impact:** Unlocks the testing foundation and prepares the code for a future serving/CLI layer.

### H-7. `pickle` used for model serialization and deserialization

- **Description:** Models are written with `pickle.dump` (`src/train.py:113`) and read with `pickle.load` (`src/evaluate.py:19`). `pickle.load` executes arbitrary code embedded in the payload.
- **Why it matters:** `SECURITY.md` itself instructs contributors to "treat external datasets and artifacts as untrusted input," yet the pipeline deserializes model artifacts pulled from a remote (DagsHub/S3) with `pickle`, which is unsafe against tampered artifacts. This is the classic ML supply-chain risk.
- **Risk:** Medium (High if the DVC remote or MLflow registry is ever shared/public). Arbitrary code execution on model load.
- **Recommended solution:** Prefer a safer serialization path — MLflow's model logging/loading (already partially used, `src/train.py:97`) or `skops` for scikit-learn — and/or verify artifact integrity (checksums) before load. At minimum, document the trust boundary and load only artifacts produced by this pipeline.
- **Estimated effort:** M
- **Engineering impact:** Closes an ACE vector and aligns the code with the repository's own security policy.

---

## 5. Medium-Priority Findings

Maintainability, readability, and configuration hygiene. Individually minor; collectively they are the "technical debt" the sprint targets.

### M-1. Configuration duplication and lack of validation in `params.yaml`

- **Description:** The raw data path `data/raw/data.csv` is repeated across `preprocess.input`, `train.input`, and `test.data`. `test.model` and `train.output` both hardcode `models/model.pkl`. There is no schema/validation; a typo in a path surfaces only as a late runtime error. Section names are inconsistent (`train` vs `test` rather than `train`/`evaluate`).
- **Why it matters:** Duplicated config drifts; a single dataset relocation requires edits in three places. No validation means misconfiguration fails deep in execution.
- **Risk:** Low–Medium. Maintenance friction; late-failing errors.
- **Recommended solution:** Introduce anchors/shared keys or a single `paths:` block; load params into a validated `@dataclass(frozen=True)` or `pydantic` config object at startup with clear error messages; rename `test` → `evaluate` for consistency with the stage name.
- **Estimated effort:** M
- **Engineering impact:** Single source of truth for configuration; earlier, clearer failures.

### M-2. `param_grid` is hardcoded, overriding declared params

- **Description:** `train.py` hardcodes a `param_grid` (`src/train.py:62-67`) that `GridSearchCV` searches. This means `params.yaml`'s `n_estimators`, `max_depth` (and `random_state`, per C-1) are effectively ignored for model selection — the grid overrides them.
- **Why it matters:** The declared parameters give a false impression of controlling training; changing `params.yaml` `n_estimators` has no effect. This is config-that-lies, closely related to C-1/M-1.
- **Risk:** Low–Medium. Misleading configuration surface; DVC param tracking tracks values that don't drive behavior.
- **Recommended solution:** Move the search grid into `params.yaml` (e.g. a `train.grid` block) so the tuning space is declarative and DVC-tracked; or remove the redundant scalar params if the grid is the source of truth.
- **Estimated effort:** S
- **Engineering impact:** Makes the parameter surface honest and DVC-tracked.

### M-3. File handles opened without context managers (resource leaks)

- **Description:** Multiple `open()` calls are never closed: `yaml.safe_load(open("params.yaml"))` (`src/preprocess.py:25`, `src/train.py:123`, `src/evaluate.py:34`), `pickle.dump(best_model, open(file_name,'wb'))` (`src/train.py:113`), `pickle.load(open(model_path,'rb'))` (`src/evaluate.py:19`).
- **Why it matters:** Leaked file descriptors; on Windows the write-then-not-closed pattern can leave a partially flushed pickle. It is also an idiom that a linter/reviewer will flag immediately, weakening the "professionally engineered" claim.
- **Risk:** Low. Real but usually benign in short-lived scripts; poor practice.
- **Recommended solution:** Use `with open(...) as f:` for every file access.
- **Estimated effort:** S
- **Engineering impact:** Correct resource handling; trivially satisfies linters.

### M-4. Dead code and commented-out debug blocks

- **Description:** A large commented-out debug block prints paths and — notably — MLflow credentials (`src/train.py:39-48`). Separator comment banners (`# ----`), a commented `filename = os.path.abspath(...)` (`src/train.py:112`), and the misleading docstring in `preprocess.py` (see M-5) are all dead weight.
- **Why it matters:** Commented code rots and misleads; the commented credential-printing block is a latent secret-leak footgun if ever re-enabled. Sprint-02 Epic 2 explicitly calls for dead-code removal.
- **Risk:** Low (Medium for the credential-print block if uncommented).
- **Recommended solution:** Delete commented-out code; rely on git history. Never log secrets even in debug paths.
- **Estimated effort:** S
- **Engineering impact:** Cleaner, safer, more readable modules.

### M-5. Docstring/comment describes behavior the code does not perform

- **Description:** `preprocess.py`'s docstring says it drops the `'Unnamed: 0'` column (`src/preprocess.py:8-12`), but the implementation only reads and re-writes the CSV — no column is dropped. `docs/architecture.md` §3 already flags this.
- **Why it matters:** Misleading documentation is worse than none; a maintainer will trust the docstring and reason incorrectly about the data.
- **Risk:** Low. Correctness-of-understanding hazard.
- **Recommended solution:** Either implement the drop or correct the docstring to describe actual behavior. Decide the intended preprocessing contract and encode it (coupled to C-3).
- **Estimated effort:** S
- **Engineering impact:** Documentation that matches reality.

### M-6. README drifts from actual artifacts and behavior

- **Description:** `README.md:34` states the model is saved as `models/random_forest.pkl`, but the code writes `models/model.pkl` (`params.yaml`, `src/train.py`). `README.md:28` describes preprocessing as "renaming columns," which the code does not do.
- **Why it matters:** The README is the entry point and portfolio front door; factual drift there is the most visible inconsistency to a reviewer.
- **Risk:** Low. Credibility.
- **Recommended solution:** Reconcile the README with actual filenames and behavior; consider generating the pipeline description from `dvc.yaml` to prevent future drift.
- **Estimated effort:** S
- **Engineering impact:** Accurate first impression.

### M-7. Magic numbers embedded in logic

- **Description:** `test_size=0.20` (`src/train.py:58`), `cv=3` (`src/train.py:23`), and `verbose=2` are hardcoded, not sourced from `params.yaml`.
- **Why it matters:** These are experiment-relevant knobs that should be declarative and DVC-tracked for reproducibility and comparison.
- **Risk:** Low.
- **Recommended solution:** Promote to `params.yaml` (`train.test_size`, `train.cv_folds`) and read them in.
- **Estimated effort:** S
- **Engineering impact:** More of the experiment is reproducible and tracked.

### M-8. Scripts assume the current working directory (fragile relative paths)

- **Description:** Each `__main__` block does `yaml.safe_load(open('params.yaml'))` with a bare relative path (`src/preprocess.py:25`, `src/train.py:123`, `src/evaluate.py:34`). The scripts only work when invoked from the repo root.
- **Why it matters:** Fragile under Docker/K8s/CI (later sprints), or when run from an IDE with a different CWD; fails with an opaque `FileNotFoundError`.
- **Risk:** Low–Medium (rises once containerized).
- **Recommended solution:** Resolve config paths relative to the project root (e.g. via a resolved `Path(__file__)` anchor or an env var), or pass the params path as a CLI argument.
- **Estimated effort:** S
- **Engineering impact:** Portable execution across environments — prepares for Sprint-03 containerization.

---

## 6. Low-Priority Findings

Polish and consistency. Cheap wins that raise the professionalism bar.

### L-1. Inconsistent import style
- **Description:** `preprocess.py` uses `import pandas` (`src/preprocess.py:1`) and calls `pandas.read_csv`, while `train.py`/`evaluate.py` use the conventional `import pandas as pd`.
- **Why it matters / Risk:** Minor readability inconsistency. **Risk:** negligible.
- **Recommended solution:** Standardize on `import pandas as pd`. Enforce via ruff/isort. **Effort:** S. **Impact:** Consistent idiom.

### L-2. Typo in user-facing output
- **Description:** `evaluate.py:25` prints `"INFO: Evaualted Model Accuracy:"` ("Evaualted").
- **Why it matters / Risk:** Cosmetic, but visible in logs. **Risk:** negligible.
- **Recommended solution:** Fix spelling (folds into the logging refactor, H-1). **Effort:** S. **Impact:** Polish.

### L-3. No code-quality automation configured
- **Description:** No `pyproject.toml`, no ruff/formatter/pre-commit config is present (Sprint-02 Epic 7 deliverable). `.editorconfig` exists but no Python toolchain.
- **Why it matters / Risk:** Style drifts without automation; several findings above (M-3, L-1) are exactly what a linter catches. **Risk:** Low.
- **Recommended solution:** Add `pyproject.toml` with ruff + ruff-format config, and a `.pre-commit-config.yaml` running ruff, formatter, and (optionally) mypy. **Effort:** M. **Impact:** Self-enforcing quality; prevents regression of this review's fixes.

### L-4. `src/` is a script collection, not an installable package
- **Description:** `src/__init__.py` is empty and there is no packaging metadata; modules are run as scripts, not imported. There is no shared utilities module for cross-cutting concerns (logging, config, exceptions) that the sprint will introduce.
- **Why it matters / Risk:** As logging/exception/config utilities are added, a flat script layout invites duplication. **Risk:** Low.
- **Recommended solution:** Consider a lightweight package layout (`src/<pkg>/` with `logging_config.py`, `exceptions.py`, `config.py`) and console entry points, or at least a shared `src/utils` module. Keep it proportional to project size. **Effort:** M. **Impact:** A home for shared engineering concerns; easier growth.

### L-5. No structured metrics output from `evaluate` for DVC
- **Description:** The `evaluate` stage declares no `outs`/`metrics` in `dvc.yaml:27-32`; accuracy goes only to MLflow.
- **Why it matters / Risk:** DVC cannot show `dvc metrics diff` across runs; metric history lives solely in a remote service. **Risk:** Low.
- **Recommended solution:** Also write a small `metrics.json` and declare it under DVC `metrics:` so metrics are versioned alongside code. **Effort:** S. **Impact:** Local, versioned metric comparison.

---

## 7. Cross-Cutting Assessment (Review Areas 1–15)

A synthesized view against the fifteen requested areas:

| # | Area | Assessment | Key findings |
|---|------|------------|--------------|
| 1 | Project structure | Good for its scale; clean `src/`-per-stage, strong docs tree | L-4 |
| 2 | Package organization | Weak — scripts, not a package; no home for shared utils | L-4, H-6 |
| 3 | Separation of concerns | Poor — pure logic entangled with IO/network/side effects | H-6, C-4 |
| 4 | Code smells | Several — dead code, magic numbers, bare `open()`, hardcoded grid | M-2, M-3, M-4, M-7 |
| 5 | Technical debt | Moderate and, commendably, partly pre-documented | C-2, C-3, C-4, M-1 |
| 6 | Logging | Absent — `print()` only | H-1 |
| 7 | Exception handling | Absent — zero `try`/`except`; bare env access | H-2 |
| 8 | Configuration management | Duplicated, unvalidated, partly ignored by code | M-1, M-2, M-7, C-2 |
| 9 | Dependency management | Unpinned; undeclared `pandas`; no lockfile/dev split | H-4 |
| 10 | Type hints | Effectively none | H-5 |
| 11 | Testability | Very low — tight IO coupling, no tests | H-3, H-6 |
| 12 | Maintainability | Held up by docs; code has drift and duplication | M-1, M-5, M-6 |
| 13 | Readability | Reasonable; hurt by dead code, typos, inconsistent imports | M-4, L-1, L-2 |
| 14 | Security | Good secret hygiene; `pickle` + unpinned deps are the gaps | H-7, H-4, M-4 |
| 15 | Future scalability | Batch-only by design; CWD-fragility and packaging block containerization | M-8, L-4, C-1 |

**On scalability specifically:** the architecture is explicitly batch/single-machine (documented in `docs/architecture.md` §6, with a cloud roadmap). That is an appropriate scope decision, not a defect. The realistic near-term scalability blockers are *engineering*, not *architecture*: non-deterministic runs (C-1), unpinned/unreproducible environments (H-4), and CWD-dependent execution (M-8) will each bite the moment the pipeline is containerized in Sprint-03. Fixing the Critical/High items is the actual prerequisite for scaling.

---

## 8. Recommended Remediation Sequence

Ordered to restore correctness first, then build the safety net, then refactor under its protection — consistent with the Sprint-02 PR-per-epic strategy.

1. **Correctness & reproducibility (Critical):** C-1, C-2, C-3, C-4. *(~1.5–2 days)* — do these first; they change outputs.
2. **Dependency pinning (H-4):** *(~0.5 day)* — lock the environment before refactoring so behavior is stable.
3. **Testing foundation + decoupling (H-3, H-6):** *(~2–3 days)* — build the safety net *before* the large refactors.
4. **Logging framework (H-1)** and **exception hierarchy (H-2):** *(~1.5–2 days)* — observability and robustness.
5. **Type hints (H-5)** and **`pickle`/serialization safety (H-7):** *(~1–1.5 days)*.
6. **Medium cleanups (M-1…M-8):** *(~1.5 days)* — config consolidation, resource managers, dead-code removal, doc reconciliation.
7. **Tooling & polish (L-1…L-5):** *(~1 day)* — ruff/formatter/pre-commit, packaging, metrics output, typo/import fixes.

**Total estimated effort:** ≈ **8–11 engineering days**, comfortably within a focused Sprint-02.

---

## 9. Closing Assessment

This is a well-documented project sitting on top of code that has not yet received the same engineering rigor as its governance layer — a common and honest state for a portfolio pipeline. The documentation's candor about its own gaps is a strength and makes this sprint's job clear rather than adversarial.

The single most important theme is **reproducibility**, which the project claims as its reason for existing but does not currently deliver: the training seed is ignored, dependencies float, the DVC param graph is broken, and the "preprocessing" stage is bypassed. These are small, well-bounded fixes with outsized value. Address the Critical findings and the dependency pinning first, stand up the test net, then execute the logging/exception/type-hint refactors under its protection. Doing so will move the code's engineering rating into line with its already-strong documentation and legitimately support the Sprint-02 target of a "professionally engineered Python codebase."

*No code was modified in the course of this review.*
