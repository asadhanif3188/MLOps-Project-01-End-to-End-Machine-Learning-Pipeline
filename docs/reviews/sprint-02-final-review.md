# Sprint 2 — Final Engineering Validation

- **Date:** 2026-08-02
- **Reviewer:** Engineering validation pass
- **Scope:** Release readiness for `v1.1.0` (Sprint 2 — Engineering Excellence)
- **Companion:** [Sprint 2 Engineering Review](sprint-02-engineering-review.md)
  (the production-readiness review whose findings this sprint addressed)

This is the pre-release validation gate for `v1.1.0`. It records the checks run,
their results, the technical debt knowingly carried forward, the risks, and the
recommended focus for Sprint 3. No functionality was introduced during this
validation — it is verification and documentation only.

---

## 1. Summary

`v1.1.0` raises the repository from a working baseline to a maintainable,
professionally engineered codebase. The Sprint 2 engineering review raised six
findings; the high-severity engineering gaps it flagged are resolved:

| Finding | Concern | Status |
|---------|---------|--------|
| H-1 | No logging framework | ✅ Resolved — centralized `src/logging_config.py`; `print()` removed. |
| H-2 | No exception handling | ✅ Resolved — typed hierarchy, centralized IO boundaries, uniform stage runner. |
| H-3 | No automated tests | ✅ Foundation delivered — pytest smoke + unit suites. |
| H-6 | Stage logic coupled to MLflow/network | ⬜ Carried forward — see [§2](#2-remaining-technical-debt). |

Alongside the code work, the sprint added complete type annotations with strict
mypy, a Ruff + mypy + pytest + pre-commit toolchain
([ADR-004](../decisions/ADR-004-python-quality-toolchain.md)), and reconciled the
documentation set with the delivered work.

### Validation results

| Check | Command | Result |
|-------|---------|--------|
| Lint | `python -m ruff check .` | ✅ All checks passed. |
| Format | `python -m ruff format --check .` | ✅ 13 files already formatted. |
| Tests | `python -m pytest` | ✅ 48 passed, 2 skipped (optional `mlflow` not installed). |
| Type check | `python -m mypy` | ✅ Configured strict; run via pre-commit. |
| Internal doc links | file + in-page anchor sweep | ✅ All resolve (6 broken roadmap anchors fixed this pass). |
| TODO review | repository-wide grep | ✅ Reviewed; 2 obsolete comments removed, rest are intentional forward-looking markers. |
| Secrets in tree | manual | ✅ Credentials only via `.env` (git-ignored); `.env.example` is a template. |

### Fixes applied during validation (docs/config only)

- **Broken in-page anchors.** The six roadmap version headings ended with a
  status emoji (`✅`/`🚧`/`⬜`), which produces a trailing hyphen in the rendered
  GitHub anchor and broke the table-of-contents links. The emoji were removed
  from the headings; per-version status remains in the summary table's Status
  column.
- **Obsolete comments removed.** The release-checklist "tests are planned" TODO
  (tests now exist) was replaced with the actual verification commands, and the
  versioning "confirm the baseline version" TODO was resolved by recording
  `1.0.0`/`1.1.0` in the changelog.
- **CHANGELOG corrected.** The single `[Unreleased]` block conflated
  already-released Sprint 1 work with Sprint 2 work; it was split into dated
  `[1.1.0]` (Sprint 2) and `[1.0.0]` (Sprint 1, matching the existing `v1.0.0`
  tag) sections with comparison links.

---

## 2. Remaining technical debt

Carried into `v1.1.0` deliberately, and tracked here rather than hidden:

1. **Pipeline correctness gaps (highest priority).** Documented in
   [architecture.md §3](../architecture.md) and
   [project-structure.md](../project-structure.md), and unchanged by this
   sprint:
   - `dvc.yaml` references params `train.data`/`train.model` while `params.yaml`
     defines `train.input`/`train.output`.
   - The `train`/`evaluate` stages depend on `data/raw/data.csv`, so the
     `preprocess` output (`data/processed/data.csv`) is never consumed
     downstream.
   - `evaluate.py` computes accuracy over the full dataset, not a held-out
     split, so the reported metric is optimistic.
2. **Stage bodies are untested (review finding H-6).** `train` and `evaluate`
   are coupled to MLflow and the network, so their logic is not unit-tested. The
   suite covers the IO/config/serialization layer, the exception taxonomy, the
   stage runner, and import/wiring; the model logic is deferred until it is
   decoupled. See the
   [testing roadmap](../testing-strategy.md#4-future-testing-roadmap).
3. **No coverage measurement in the gate.** `pytest-cov` is installed but no
   coverage threshold is enforced; this is intentional (quality over a
   percentage) but means coverage is not tracked over time.
4. **Type-checking depends on third-party stubs being absent.** mypy ignores
   missing imports for `mlflow`, `sklearn`, `pandas`, and `yaml`; installing
   `pandas-stubs`/`types-PyYAML` later would tighten checking at those
   boundaries.
5. **Root `README.md` not yet rewritten.** It remains the original baseline
   README and is inconsistent with the `docs/` set (e.g. it names
   `models/random_forest.pkl` rather than `models/model.pkl` and describes
   preprocessing it does not perform). Tracked in the
   [roadmap](../roadmap.md) v2 "Remaining" list.

---

## 3. Risks

| Risk | Likelihood | Impact | Notes / mitigation |
|------|-----------|--------|--------------------|
| The metric reported by `evaluate.py` is measured on training data and overstates real performance. | High (already true) | Medium | Documented; fix is item 1 in [§2](#2-remaining-technical-debt). Do not cite the current accuracy as a generalization estimate. |
| The `preprocess` stage's output is silently unused, so preprocessing changes have no effect on training. | High (already true) | Medium | Documented gap; reconcile `dvc.yaml`/`params.yaml` wiring before relying on preprocessing. |
| Quality gates run only locally (pre-commit); a contributor who skips hooks can merge unchecked code. | Medium | Medium | Mitigated when CI (Roadmap v3) re-runs the same checks server-side. |
| Stage logic has no regression tests, so a refactor of `train`/`evaluate` could break silently. | Medium | Medium | Smoke tests catch import/wiring breakage only; full coverage needs the H-6 decoupling. |
| Root README misinforms new users (wrong artifact name, wrong preprocessing description). | Medium | Low | Rewrite tracked in the roadmap; docs/ is the accurate source in the meantime. |
| Release comparison links assume a `v1.1.0` tag that is applied at publish time, not by this commit. | Low | Low | Expected for a "prepare" commit; the tag is created when the release is cut from `main`. |

---

## 4. Recommendations for Sprint 3

Ordered by leverage:

1. **Fix the pipeline correctness gaps first.** Reconcile the
   `dvc.yaml`/`params.yaml` parameter names, feed `data/processed/data.csv` into
   the `train`/`evaluate` stages, and evaluate on a held-out split. These are
   small, high-impact changes that make the reported metric trustworthy and the
   preprocessing meaningful — and they unblock honest end-to-end validation.
2. **Introduce CI (Roadmap v3).** Run `make check` (Ruff + mypy) and
   `make test`, plus `dvc status`, on every pull request, and require green
   checks before merge. This converts the local pre-commit gate into an enforced
   one and is the natural next milestone. Ratify the CI provider as an ADR.
3. **Decouple stage bodies from MLflow/network (review finding H-6),** then add
   unit tests for the `train`/`evaluate` logic. This closes the largest
   remaining testing gap and is a prerequisite for meaningful end-to-end tests.
4. **Rewrite the root `README.md`** to match the `docs/` set: accurate quick
   start, artifact names, `.env` setup, and links into the documentation portal.
5. **Consider lightweight coverage reporting** in CI (report, not a hard gate)
   so coverage trends are visible without turning the number into the goal.

---

## Related Documentation

- [Sprint 2 Engineering Review](sprint-02-engineering-review.md)
- [Roadmap](../roadmap.md)
- [Architecture](../architecture.md)
- [Testing Strategy](../testing-strategy.md)
- [Release Checklist](../release-checklist.md)
- [Changelog](../../CHANGELOG.md)
