# Sprint 4 — Final Engineering Validation

- **Date:** 2026-08-06
- **Reviewer:** Engineering validation pass
- **Scope:** Release readiness for `v1.3.0` (Sprint 4 — Pipeline Correctness &
  Reproducibility)
- **Companion:** [Sprint 3 Final Review](sprint-03-final-review.md) (the prior
  release gate, whose "Recommendations for Sprint 4" this sprint addressed),
  [Pipeline Contract](../pipeline-contract.md),
  [ADR-006](../decisions/ADR-006-pipeline-reproducibility.md)

This is the pre-release validation gate for `v1.3.0`. It records the checks run,
their results, an assessment against each Sprint 4 objective, the technical debt
knowingly carried forward, the risks, and the recommended focus for the next
sprint. No functionality was introduced during this validation — it is
verification and documentation only.

---

## 1. Summary

`v1.3.0` turns attention from the infrastructure *around* the ML pipeline to the
pipeline itself. Sprint 4 corrected the DVC dependency graph, made configuration
consistent across `dvc.yaml`/`params.yaml`/code, separated the ML computation
from the MLflow boundary so stage logic is unit-testable, seeded training for
determinism, and enforced the pipeline contract automatically in CI — all without
requiring production credentials or a live external service in CI.

| Capability | Artifact | Status |
|------------|----------|--------|
| Pipeline contract | [pipeline-contract.md](../pipeline-contract.md) + [ADR-006](../decisions/ADR-006-pipeline-reproducibility.md) | ✅ Documented & enforced |
| Corrected DVC graph | [`dvc.yaml`](../../dvc.yaml) | ✅ Implemented & validated (`dvc dag`) |
| Configuration consistency | [`params.yaml`](../../params.yaml) ↔ `dvc.yaml` ↔ `src/` | ✅ Implemented & test-enforced |
| ML-stage refactor (MLflow boundary) | `src/train.py`, `src/evaluate.py`, `src/tracking.py` | ✅ Implemented & unit-tested |
| Pipeline test suite | `tests/{unit,integration,contract}/` | ✅ 84 tests, all green |
| CI pipeline integrity | [`ci.yml`](../../.github/workflows/ci.yml) | ✅ Implemented (offline) |

### Validation results

Every gate below was executed during this pass, on the host, on the release
branch `feature/sprint-04-release`.

| Check | Command | Result |
|-------|---------|--------|
| Lint | `python -m ruff check .` | ✅ All checks passed. |
| Format | `python -m ruff format --check .` | ✅ 19 files already formatted. |
| Types | `python -m mypy` | ✅ Success: no issues in 9 source files. |
| Tests | `python -m pytest` | ✅ 83 passed, 1 skipped (optional `mlflow` absent on host; installed in CI). 84 collected. |
| Contract tests | `python -m pytest -m contract` | ✅ 8 passed — parameter consistency, single-owner artifacts, acyclic graph, declared lineage. |
| DVC graph | `dvc dag` (`DVC_NO_ANALYTICS=true`) | ✅ Exit 0; acyclic; lineage `raw → preprocess → {train, evaluate}`, `train → evaluate`, `evaluate → metrics`. |
| DVC status | `dvc status` (no `--cloud`) | ✅ Exit 0; reports "changed"/"not in cache" because the data is remote-only and no `dvc.lock` is committed — expected (see §7 limitations). |
| DVC/params parse | `yaml.safe_load` on `dvc.yaml`, `params.yaml` | ✅ 3 stages / 3 sections; keys aligned. |
| CI workflow | structural parse of `.github/workflows/ci.yml` | ✅ Valid YAML; `permissions: contents: read`; `docker` needs `quality`; new offline DVC-integrity step present; `DVC_NO_ANALYTICS=true`. |

> **Note on the DVC check.** The host's globally-installed `dvc` is broken by a
> local `pathspec` packaging anomaly (`cannot import name '_DIR_MARK'`), unrelated
> to this repository. The graph was therefore validated in an isolated virtualenv
> with a clean dependency resolve (`dvc 3.59.1`), which is what a fresh CI runner
> also produces. The pre-existing `docker` job already imports `dvc` in the built
> image, so the container path is exercised independently.

---

## 2. Assessment against Sprint 4 objectives

### 2.1 Pipeline correctness

**Verdict: met.** The pipeline is now the single linear chain the architecture
always described. `preprocess` produces the processed dataset with a header row;
`train` consumes that processed dataset (not the raw file); `evaluate` consumes
the model and the processed dataset and writes a metrics artifact. The former
"preprocess output is orphaned" defect (D1) and the headerless-CSV mismatch (D8)
are resolved and guarded by the `contract` tests
(`test_processed_data_is_consumed_not_orphaned`, `test_lineage_matches_contract`).

### 2.2 DVC lineage

**Verdict: met.** `dvc dag` builds an acyclic graph with the correct edges
(`raw.dvc → preprocess`, `preprocess → train`, `preprocess → evaluate`,
`train → evaluate`). Each artifact has exactly one producing stage
(`test_each_artifact_has_exactly_one_producer`), and `evaluate` declares
`metrics/metrics.json` as an uncached DVC metric (D4 resolved).

### 2.3 Configuration consistency

**Verdict: met.** `params.yaml` is the single source of truth; `dvc.yaml` and the
stage code reference the same authoritative names. The `train.data`/`train.model`
mismatch (C1/C2), the misnamed `test` section (C3), the undeclared evaluate params
(C4), and the inert `train.*` hyperparameters (C5) are all resolved. Consistency
is enforced by `test_every_dvc_param_key_exists_in_params_yaml`,
`test_no_orphaned_params`, and `test_declared_outputs_match_params`.

### 2.4 Stage contracts

**Verdict: met.** Each stage documents its inputs, outputs, configuration, and
typed failure conditions in both its docstring and
[pipeline-contract §3](../pipeline-contract.md#3-stage-contracts). Artifact
ownership is explicit and single-owner. Errors are typed (`DataError`,
`ConfigError`, `ModelError`, `TrackingError`) and surface through the uniform
`stage_runner` entry point.

### 2.5 Testability

**Verdict: met.** The ML computation is separated from IO and MLflow:
`run_training` and `compute_metrics` perform no file IO and import no MLflow, so
they are unit-tested directly. The `tracking` boundary is imported lazily and, in
tests, replaced by the `stub_tracking` in-memory recorder — so stage read →
compute → persist paths run end-to-end with no network, no MLflow import, and no
credentials. The suite is four-tier (smoke/unit/integration/contract), 84 tests,
deterministic and offline.

### 2.6 CI validation

**Verdict: met (offline scope).** CI now fails a pull request when the DVC graph
is malformed or cyclic, a parameter is inconsistent, an artifact is mis-owned, or
the declared lineage is broken — via `dvc dag`, local `dvc status`, and the
`contract` tests, with `DVC_NO_ANALYTICS=true` and no remote access. Sprint 3's
`docker` job and least-privilege `contents: read` permissions are preserved. CI
validates the pipeline **definition**, not a full remote **execution** (see §3).

### 2.7 Known limitations

**Verdict: documented, not hidden.** The two open items (in-sample evaluation;
no committed `dvc.lock`) are recorded in the contract, the architecture doc, the
README, and §3 below. No generalization claim is made from the current accuracy.

---

## 3. Remaining technical debt

Carried into `v1.3.0` deliberately, and tracked here rather than hidden.

1. **In-sample evaluation (deviation D5, highest pipeline-correctness priority).**
   `evaluate.data` points at the same processed dataset `train` fits on, so the
   reported accuracy is in-sample and supports no generalization claim. Because
   the evaluation dataset is already an explicit configured input, a held-out
   split is a configuration/graph change, not a code change
   ([contract §8](../pipeline-contract.md#8-evaluation-boundary)).
2. **No committed `dvc.lock` (part of deviation D7).** Training is now
   deterministic (seeded split + estimator), but the resolved pipeline state is
   not pinned, so `dvc status` is not a true "up to date" drift gate and CI cannot
   validate a locked execution. A `dvc.lock` needs a runnable dataset in CI.
3. **No in-CI pipeline execution.** CI validates the definition (`dvc dag` +
   `dvc status` + contract tests) but does not run `dvc repro` — the raw dataset
   is remote-only. A future execution check would use a small committed fixture
   dataset plus a committed `dvc.lock`.
4. **Stage orchestration still requires `MLFLOW_TRACKING_URI` to run end-to-end.**
   By design — MLflow logging is a preserved capability. The *computation* is
   decoupled and testable without it; the *stage* still refuses to run without the
   URI ([contract §9](../pipeline-contract.md#9-external-service-boundaries)).
5. **CI does not run mypy (unchanged from Sprint 3).** `make check` and pre-commit
   run it locally; it is not yet a server-side gate.
6. **CI is not enforced by branch protection (unchanged).** The workflow produces
   the signal; requiring green checks before merge is a repository setting.
7. **Dependencies/base image are name-pinned, not digest-pinned (unchanged).**
   Byte-for-byte reproducible rebuilds need digest/hash pinning
   ([ADR-005](../decisions/ADR-005-containerization-strategy.md)).
8. **No image scanning / publishing, no coverage threshold (unchanged).**
   Deferred by design.

Items 5–8 are carried verbatim from the
[Sprint 3 final review](sprint-03-final-review.md#2-remaining-technical-debt);
Sprint 4 was scoped to pipeline correctness, not CI hardening or supply chain.

---

## 4. Risks

| Risk | Likelihood | Impact | Notes / mitigation |
|------|-----------|--------|--------------------|
| The reported `accuracy` is cited as a generalization estimate. | Medium | Medium | It is in-sample; labeled as such in the README, architecture, and contract. Do not quote it as held-out performance until D5 is closed. |
| A contributor relies on `dvc status` as a drift gate. | Low | Low | No `dvc.lock` is committed; `dvc status` reports "changed" for a remote-only dataset by design. Documented (§3.2). |
| Re-running training locally reproduces a *logically* identical model but not byte-identical artifacts. | Low | Low | Seeds fix the model/metric; name-pinned deps mean the environment is not bit-locked (item 7). |
| A type regression reaches `main` because mypy is not a CI gate. | Medium | Low | mypy runs locally and in pre-commit; add to CI to enforce server-side (item 5). |
| A change breaks the pipeline definition but merges anyway (no branch protection). | Medium | Medium | CI now *detects* graph/contract breakage on every PR; enforce via branch protection to make it binding (item 6). |
| The local `dvc` `pathspec` anomaly is mistaken for a repository defect. | Low | Low | Local packaging issue only; a clean resolve (CI, isolated venv) works. Noted in §1. |
| `docker run` of the pipeline still needs mounted data + MLflow credentials. | Medium | Low | Expected for a batch image; unchanged from Sprint 3. |

---

## 5. Recommendations for Sprint 5

Ordered by leverage:

1. **Close the evaluation boundary (D5).** Add a held-out split (a splitting step
   or a second processed artifact) and point `evaluate.data` at the held-out
   portion. This is the single most valuable remaining pipeline-correctness change
   and turns the reported accuracy into a defensible generalization estimate.
2. **Commit a `dvc.lock` against a fixture dataset and add a `dvc repro`
   execution check to CI (D7).** This converts CI from definition-validation to
   execution-validation and makes `dvc status` a true drift gate — without
   depending on the DagsHub remote.
3. **Harden CI toward the full quality gate.** Add mypy to the workflow (item 5)
   and enable branch protection requiring the `Lint & Test` and `Docker Build &
   Validate` checks (item 6).
4. **Supply-chain hardening.** Digest-pin the base image and dependencies (item 7)
   and add a Trivy scan (item 8) — the prerequisites for publishing the image.

---

## Related Documentation

- [Sprint 3 Final Review](sprint-03-final-review.md)
- [Pipeline Contract](../pipeline-contract.md) · [ADR-006](../decisions/ADR-006-pipeline-reproducibility.md)
- [Sprint 4 Retrospective](../retrospectives/sprint-04-retrospective.md)
- [Sprint 4 Proof Impact](../proof/sprint-04-proof-impact.md)
- [Architecture](../architecture.md) · [CI/CD](../ci-cd.md)
- [Roadmap](../roadmap.md) · [Changelog](../../CHANGELOG.md)
