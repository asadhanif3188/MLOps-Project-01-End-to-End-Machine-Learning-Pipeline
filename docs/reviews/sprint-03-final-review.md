# Sprint 3 — Final Engineering Validation

- **Date:** 2026-08-04
- **Reviewer:** Engineering validation pass
- **Scope:** Release readiness for `v1.2.0` (Sprint 3 — Containerization & CI)
- **Companion:** [Sprint 2 Final Review](sprint-02-final-review.md) (the prior
  release gate, whose Sprint 3 recommendations this sprint partially addressed)

This is the pre-release validation gate for `v1.2.0`. It records the checks run,
their results, the technical debt knowingly carried forward, the risks, and the
recommended focus for Sprint 4. No functionality was introduced during this
validation — it is verification and documentation only.

---

## 1. Summary

`v1.2.0` packages the pipeline for reproducible execution and automates its
quality gates. Sprint 3 delivered three capabilities, each designed first and
then implemented and validated:

| Capability | Artifact | Status |
|------------|----------|--------|
| Containerization | Multi-stage `Dockerfile` + `.dockerignore` ([ADR-005](../decisions/ADR-005-containerization-strategy.md), [strategy](../containerization.md)) | ✅ Implemented & validated |
| Local dev workflow | `docker-compose.yml` + [docs](../docker-development.md) | ✅ Implemented & validated |
| Continuous integration | GitHub Actions `ci.yml` + [docs](../ci-cd.md) | ✅ Implemented; runs on GitHub |

The container image is **non-root** (UID 10001), built on
`python:3.12-slim-bookworm`, ships only runtime dependencies via a discarded
builder stage, and externalizes all state and secrets. CI validates every push
and pull request and builds the image without publishing it.

### Validation results

Every gate below was executed during this pass.

| Check | Command | Result |
|-------|---------|--------|
| Lint | `python -m ruff check .` | ✅ All checks passed. |
| Format | `python -m ruff format --check .` | ✅ 13 files already formatted. |
| Tests | `python -m pytest` | ✅ 48 passed, 2 skipped (optional `mlflow` absent on host; installed in CI). |
| Docker build (runtime) | `docker build --build-arg VCS_REF=… --build-arg BUILD_VERSION=1.2.0 -t ml-pipeline:1.2.0 .` | ✅ Exit 0. |
| Docker build (development) | `docker build --target development -t ml-pipeline:dev .` | ✅ Exit 0. |
| Image contract | `docker run --rm ml-pipeline:1.2.0 …` | ✅ UID=10001; `import sklearn, mlflow, dvc, pandas` OK; `dvc 3.67.1`. |
| Image metadata | `docker image inspect` | ✅ `User=appuser`, `Cmd=[dvc repro]`, `WorkingDir=/app`; OCI labels incl. `version=1.2.0`, `revision` stamped. |
| Image size | `docker images` | ✅ 1.61 GB runtime / 2.11 GB development (as documented). |
| Compose config | `docker compose config` | ✅ Valid; default `up` → `dev` only; `--profile pipeline` → `dev` + `pipeline`. |
| Compose lifecycle | `up -d dev` → `exec` → `down` | ✅ Toolchain present (Python 3.12.13, Ruff 0.16.1, DVC 3.67.1, pytest 9.1.1); bind mounts live; clean teardown. |
| CI workflow | structural parse of `.github/workflows/ci.yml` | ✅ Triggers push/PR/dispatch; `permissions: contents: read`; `docker` needs `quality`; `push: false`. |
| Internal doc links | file-link sweep across 29 md files | ✅ All resolve (1 broken link fixed this pass — see below). |
| TODO review | repository-wide grep | ✅ None in `src`/`tests`; doc `TODO`s are intentional forward-looking markers (see [§2](#2-remaining-technical-debt)). |
| Secrets in tree | `git ls-files` | ✅ No `.env` tracked; `.env.example` is a template only. |

> **Note on the CI check.** The workflow's *runner orchestration* cannot be
> exercised outside GitHub, but every command it runs was executed locally
> (Ruff, pytest) or as a real container build/validation during this pass, so the
> pipeline's actual work is proven end-to-end.

### Fixes applied during validation (docs only)

- **Broken relative link.** `ADR-005` linked `SECURITY.md` as `../SECURITY.md`;
  from `docs/decisions/` that resolves to a nonexistent `docs/SECURITY.md`.
  Corrected to `../../SECURITY.md` (the file lives at the repository root).

---

## 2. Remaining technical debt

Carried into `v1.2.0` deliberately, and tracked here rather than hidden. Items
1–3 are unchanged from `v1.1.0` — Sprint 3 was scoped to containerization and CI,
not pipeline correctness — and are repeated so the debt stays visible.

1. **Pipeline correctness gaps (highest priority, unchanged).** Documented in
   [architecture.md §3](../architecture.md):
   - `dvc.yaml` references params `train.data`/`train.model` while `params.yaml`
     defines `train.input`/`train.output`.
   - The `train`/`evaluate` stages depend on `data/raw/data.csv`, so the
     `preprocess` output (`data/processed/data.csv`) is never consumed.
   - `evaluate.py` computes accuracy over the full dataset, not a held-out split,
     so the reported metric is optimistic.
2. **Stage bodies are untested (review finding H-6, unchanged).** `train` and
   `evaluate` are coupled to MLflow and the network, so their logic is not
   unit-tested. See the
   [testing roadmap](../testing-strategy.md#4-future-testing-roadmap).
3. **Root `README.md` not yet rewritten.** It still carries baseline content that
   contradicts the `docs/` set — e.g. it names `models/random_forest.pkl` while
   the pipeline produces `models/model.pkl`, and describes preprocessing the
   stages do not perform. Sprint 3 added Docker/CI sections and a CI badge but did
   not do the full rewrite. Tracked in the [roadmap](../roadmap.md) v2 "Remaining".
4. **CI does not run mypy.** The CI job runs Ruff + pytest (the requested stages);
   `make check` additionally runs mypy locally and via pre-commit, but type
   checking is not yet a server-side gate. The `philosophy.md` and
   `github-workflow.md` `TODO`s about server-side enforcement remain accurate.
5. **CI is not yet enforced by branch protection.** The workflow produces the
   signal, but requiring green checks before merge is a GitHub repository setting
   not applied by this commit (see the `github-workflow.md` `TODO`).
6. **No image scanning / publishing.** By design for this release: `push: false`
   and no CVE scan. Publishing to a registry and gating on a Trivy scan are
   deferred (CD, roadmap v3 follow-on).
7. **Dependencies and base image are name-pinned, not digest-pinned.**
   `requirements.txt` pins by name and the base image by codename
   (`python:3.12-slim-bookworm`). Byte-for-byte reproducibility needs digests/
   hashes — the tracked follow-up in [ADR-005](../decisions/ADR-005-containerization-strategy.md).
8. **Image size (~1.6 GB runtime).** Dominated by the scientific/MLOps stack
   (`scipy`, `pyarrow`, `matplotlib`, `mlflow`, `boto3`), not build cruft.
   Reduction paths (mlflow-skinny, distroless, pruning heavy transitives) are
   documented in [containerization.md §15](../containerization.md).
9. **No coverage measurement in the gate (unchanged).** `pytest-cov` is installed
   but no threshold is enforced — intentional (quality over a percentage).

---

## 3. Risks

| Risk | Likelihood | Impact | Notes / mitigation |
|------|-----------|--------|--------------------|
| `evaluate.py` measures accuracy on training data and overstates real performance. | High (already true) | Medium | Documented; item 1 in [§2](#2-remaining-technical-debt). Do not cite current accuracy as a generalization estimate. |
| The `preprocess` output is silently unused, so preprocessing changes have no effect on training. | High (already true) | Medium | Documented; reconcile `dvc.yaml`/`params.yaml` wiring before relying on preprocessing. |
| A contributor who skips pre-commit hooks can merge lint/type/test failures, since branch protection does not yet require CI. | Medium | Medium | CI re-runs Ruff + pytest on every PR; enforce via branch protection (item 5) to fully close. |
| Type regressions reach `main` because mypy is not a CI gate. | Medium | Low | mypy runs locally and in pre-commit; add it to CI (item 4) to enforce server-side. |
| `docker run ml-pipeline:1.2.0` (default `dvc repro`) fails without mounted data and MLflow/DagsHub credentials. | Medium | Low | Expected for a batch image; documented in [containerization.md §15](../containerization.md). Run single stages or the dev image for exploration. |
| A published image could ship a known CVE, since no scan gate exists. | Low (not published) | Medium | Image is not pushed this release; add a Trivy gate before enabling publish (item 6). |
| Rebuilds are not byte-for-byte reproducible (name-pinned deps/base). | Low | Low | Functionally stable; digest-pin per ADR-005 for auditable reproducibility (item 7). |
| Root README misinforms new users (wrong artifact name, wrong preprocessing description). | Medium | Low | `docs/` is the accurate source; rewrite tracked (item 3). |
| Release comparison links assume a `v1.2.0` tag applied at publish time, not by this commit. | Low | Low | Expected for a "prepare" commit; the tag is created when the release is cut from `main`. |

---

## 4. Recommendations for Sprint 4

Ordered by leverage:

1. **Fix the pipeline correctness gaps first (carried since Sprint 2).**
   Reconcile the `dvc.yaml`/`params.yaml` parameter names, feed
   `data/processed/data.csv` into `train`/`evaluate`, and evaluate on a held-out
   split. Small, high-impact changes that make the reported metric trustworthy —
   and, now that the image can run `dvc repro` reproducibly, they unblock honest
   end-to-end validation in a container.
2. **Harden CI toward the full quality gate.** Add mypy to the workflow (item 4)
   and enable branch protection requiring the `Lint & Test` and
   `Docker Build & Validate` checks before merge (item 5). This converts the
   existing signal into an enforced gate.
3. **Add container supply-chain checks.** Introduce a Trivy scan as a CI gate and,
   when ready, digest-pin the base image and dependencies (items 6–7). These are
   the prerequisites for safely publishing the image.
4. **Decouple stage bodies from MLflow/network (review finding H-6),** then add
   unit tests for `train`/`evaluate`. Closes the largest remaining testing gap and
   enables container-based end-to-end tests.
5. **Rewrite the root `README.md`** to match the `docs/` set: correct artifact
   names, accurate preprocessing description, and the Docker/CI workflows as the
   primary quick start.
6. **Trim the image** (mlflow-skinny / distroless / prune transitives, item 8) once
   correctness and CI hardening are done — a size win, not a correctness one, so
   it ranks last.

---

## Related Documentation

- [Sprint 2 Final Review](sprint-02-final-review.md)
- [Containerization Strategy](../containerization.md) · [ADR-005](../decisions/ADR-005-containerization-strategy.md)
- [Docker Development Workflow](../docker-development.md)
- [CI/CD](../ci-cd.md)
- [Roadmap](../roadmap.md)
- [Release Checklist](../release-checklist.md)
- [Changelog](../../CHANGELOG.md)
