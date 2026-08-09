# ADR-008: Fixture-Based Pipeline Reproducibility (Committed `dvc.lock` + CI `dvc repro`)

- **Status:** Accepted — implemented (proof hardening milestone). Resolves the
  `dvc.lock` / execution-check portion of deviation **D7** tracked in the
  [pipeline contract](../pipeline-contract.md#11-deviation-status-sprint-4).
- **Date:** 2026-08-09
- **Deciders:** Asad Hanif
- **Related:** [ADR-006 (Pipeline Reproducibility)](ADR-006-pipeline-reproducibility.md),
  [ADR-007 (Held-Out Evaluation)](ADR-007-held-out-evaluation.md),
  [ADR-003 (Why DVC)](ADR-003-why-dvc.md),
  [ADR-002 (Why MLflow)](ADR-002-why-mlflow.md),
  [Pipeline Contract §7](../pipeline-contract.md#7-reproducibility-expectations),
  [CI/CD](../ci-cd.md), [architecture.md](../architecture.md)

## Context

After [ADR-006](ADR-006-pipeline-reproducibility.md) (reproducibility as a
requirement) and [ADR-007](ADR-007-held-out-evaluation.md) (held-out evaluation),
the pipeline modeled a **correct, deterministic DAG** — but reproducibility was
still only proven *logically*. Two linked facts, tracked as deviation **D7**,
remained:

- **No committed `dvc.lock`.** The resolved dependency/output hashes for a run
  were never pinned, so `dvc status` could not act as a real "up to date" drift
  gate and the claim "this pipeline reproduces" rested on reading the code, not on
  a checked-in artifact.
- **No in-CI execution.** CI validated the pipeline *definition* (`dvc dag`, local
  `dvc status`, the `contract` tests) but never *ran* it. It could not, honestly:
  the production pipeline's first input, `data/raw/data.csv`, is **remote-only**
  (DagsHub S3, credentialed), and `train`/`evaluate` log to a **networked MLflow**
  server. Running the production pipeline in ordinary CI would require secrets and
  a live external service — exactly what ADR-006 decision (4) forbids for
  automated validation.

So the pipeline could show *what depends on what* and *that the code is seeded*,
but nothing in the repository **executed** the chain
`declared pipeline + declared parameters + declared inputs + code = outputs` and
proved it reproducible. We wanted that evidence without contradicting ADR-006 by
pulling credentials or a network service into CI, and **without replacing the real
dataset** — the production pipeline must keep pointing at the real, remote data.

A key enabling detail already existed in the code: [`src/tracking.py`](../../src/tracking.py)
funnels every MLflow call through one lazily-imported module, and the stages'
ML computation imports no MLflow. The MLflow interaction is a *side effect that
produces no DVC-tracked artifact*.

## Decision

Add a **second, self-contained "fixture" DVC pipeline** whose sole purpose is to
be reproduced in CI, alongside — never replacing — the production pipeline.

1. **A committed fixture dataset, not the real one.** `tests/fixtures/pipeline/`
   holds a small (80-row, ~3 KB), class-balanced CSV that *mimics the schema* of
   the real Pima dataset, generated deterministically by a committed, seeded
   script ([`generate_data.py`](../../tests/fixtures/pipeline/generate_data.py)).
   The production `data/raw/data.csv` is untouched and remains remote-only.

2. **The same code, its own declarations.** The fixture
   [`dvc.yaml`](../../tests/fixtures/pipeline/dvc.yaml) declares the *same four
   stages* and runs the *same* `src/` stage code against the fixture data, with
   its own [`params.yaml`](../../tests/fixtures/pipeline/params.yaml) (seeds
   explicit; a smaller model for CI speed). Both the stage script **and** the
   fixture wrapper are declared `deps`, so code is a tracked input.

3. **A committed `dvc.lock`.** `dvc repro` resolves the fixture pipeline and its
   lock is committed. That lock pins the declared pipeline, the parameter values,
   the input hashes, the code hashes, and the resolved output hashes — a
   checked-in record that the chain resolves.

4. **Offline execution via a tracking stub.** A thin wrapper,
   [`_run_stage.py`](../../tests/fixtures/pipeline/_run_stage.py), runs the real
   stage `main()` but pre-installs a no-op `tracking` module, so `dvc repro` drives
   the whole graph with **no MLflow, no network, and no credentials**. Because
   tracking produces no DVC-tracked artifact, stubbing it cannot change a single
   reproduced byte — this is the same boundary substitution the unit/integration
   tests use (ADR-006 decision 4), applied in a subprocess.

5. **CI runs a real `dvc repro`.** CI reproduces the fixture pipeline end to end,
   asserts the reproduced workspace is coherent with its lock (`dvc status` → "up
   to date"), and asserts determinism by forcing a second run and requiring
   **byte-identical** model and metrics. Two offline `pytest` checks complement
   it, portable to every machine: a structural staleness check that the committed
   lock still matches the fixture `dvc.yaml`, and a determinism check that the
   seeded stage compute reproduces equivalent outputs.

6. **Claim exactly four reproducibility levels, no more.** The contract
   ([§7](../pipeline-contract.md#7-reproducibility-expectations)) now names them
   explicitly: (1) DVC graph correctness, (2) lock-state reproducibility, (3)
   fixture execution — all proven; and (4) production-data reproducibility —
   *documented as out of CI scope* (needs the remote dataset, live MLflow, and
   digest-pinned dependencies).

## Alternatives Considered

1. **Commit a `dvc.lock` for the production pipeline.**
   - *Rejected.* It cannot be generated or verified without the remote dataset and
     MLflow, so it would be an unverifiable, hand-waved artifact — "faking
     reproducibility." A lock nobody can reproduce in CI is worse than none.

2. **Run the production `dvc repro` in CI with `dvc pull` + DagsHub secrets.**
   - *Rejected.* Directly contradicts ADR-006 decision (4): it makes automated
     validation depend on credentials, network, and a live MLflow server, and
     exposes secrets to pull-request CI. Also slow and flaky.

3. **Point the fixture stages at a local MLflow `file:` store instead of a stub.**
   - *Considered, not chosen.* `tracking.py` already supports a `file:` store
     (registration-free), so a real `dvc repro` with `MLFLOW_TRACKING_URI=file:…`
     works — and was verified. But it makes reproduction depend on MLflow being
     installed and writes non-deterministic run directories. The no-op stub is
     lighter, needs nothing beyond scikit-learn, and keeps the fixture's
     reproduced artifacts free of any MLflow influence. The `file:`-store path
     remains available for anyone who wants to exercise real tracking locally.

4. **Only prove reproducibility with a pytest test that runs the stage functions
   (no DVC).**
   - *Rejected as insufficient alone.* That proves the *Python functions* are
     deterministic, but not that **DVC** can reconstruct the pipeline from the
     lock + declared inputs (change detection, stage orchestration, artifact
     hashing). The committed lock + CI `dvc repro` prove the tool-level claim; the
     pytest checks are kept as the portable complement.

5. **Assert the committed lock's output hashes match CI's reproduction
   (`git diff --exit-code dvc.lock`).**
   - *Rejected.* Model pickle bytes are platform-sensitive (a lock authored on
     Windows need not match a Linux CI run), so this would flake without proving
     anything the determinism check doesn't. CI instead asserts *same-environment*
     determinism (force-rerun, compare hashes); the portable pytest test asserts
     *structural* lock freshness. Cross-platform bit-reproducibility is explicitly
     **not** claimed (see level 4).

6. **Make the fixture dataset large / realistic.**
   - *Rejected.* CI cost and repo weight for no proof value. 80 balanced rows are
     enough to exercise the stratified split and the 3-fold `GridSearchCV` while
     keeping a full `dvc repro` to ~a second of compute.

## Consequences

**Positive**

- The repository now contains **executed** evidence that
  `declared pipeline + params + inputs + code = reproducible outputs`, runnable by
  anyone (`dvc repro tests/fixtures/pipeline/dvc.yaml`) with no credentials.
- A committed `dvc.lock` exists and is drift-gated: CI reproduces it every run;
  a `contract` test fails if it goes structurally stale.
- Reproducibility claims are **stratified and honest** — three levels proven, the
  fourth (production-data, bit-for-bit) named as a documented limitation rather
  than implied.
- The MLflow / DagsHub boundary and the real dataset are **untouched**; production
  behavior is unchanged.

**Trade-offs and follow-ups**

- **Two pipelines to keep in step.** The fixture `dvc.yaml`/`params.yaml` mirror
  the production ones; if the production stage contract changes, the fixture must
  change too. The structural `contract` tests and the shared `src/` code keep the
  drift visible, but it is real maintenance surface.
- **The fixture proves the pipeline mechanics, not the model.** It uses synthetic
  data and a smaller model, so its accuracy figure is meaningless as a model claim
  — it validates *reproducible execution*, not performance. The held-out
  methodology itself is owned by [ADR-007](ADR-007-held-out-evaluation.md).
- **Production-data reproducibility (level 4) remains out of CI** by design —
  remote data, live MLflow, and name-pinned (not digest-pinned) dependencies. That
  is the honest boundary; closing it belongs to later CD/roadmap work.
- **Regenerating the lock is a manual step.** After a change to the fixture
  definition or the stage code, run
  `dvc repro tests/fixtures/pipeline/dvc.yaml` and commit the updated lock; the
  `contract` staleness test is the reminder if it is forgotten.
