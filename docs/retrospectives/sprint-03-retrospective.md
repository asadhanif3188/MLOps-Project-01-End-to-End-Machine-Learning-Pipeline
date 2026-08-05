# Sprint 3 — Retrospective (v1.2.0)

- **Date:** 2026-08-05
- **Release:** `v1.2.0` — Containerization & Continuous Integration
- **Scope:** Make the pipeline portable and self-validating (Docker, Docker
  Compose, GitHub Actions CI) and reconcile the documentation with it.
- **Companion:** [Sprint 2 — Final Validation](../reviews/sprint-02-final-review.md)
  (whose "Sprint 3 recommendations" seeded this sprint),
  [Roadmap v3 (CI/CD)](../roadmap.md#version-3--cicd),
  [Roadmap v4 (Kubernetes)](../roadmap.md#version-4--kubernetes)

This is a look-back on Sprint 3: what was planned, what shipped, the engineering
decisions behind it, and what was deliberately left for later. It records
judgment and rationale — it is not a validation gate and introduced no code.

---

## 1. Planned

The sprint set out to deliver the first half of Roadmap v3 (CI) and the container
prerequisite for Roadmap v4:

- **Containerization** — a production-grade, reproducible image so pipeline
  stages run identically on a laptop, in CI, and (later) on a cluster.
- **Docker Compose** — a one-command local development environment so a new
  contributor needs only Docker.
- **CI** — automated quality gates (lint, format, tests) plus a container build
  on every push and pull request.
- **Documentation** — a containerization strategy, a Compose workflow guide, a
  CI/CD design of record, and an ADR ratifying the container design, with the
  existing docs reconciled to match.

## 2. Delivered

Shipped as **`v1.2.0`** across five feature PRs, each branch → PR → merge to
`main`:

| PR | Branch | Delivered |
|----|--------|-----------|
| #11 | `feature/sprint-03-container-design` | Containerization strategy + ADR-005 (design ratified) |
| #12 | `feature/sprint-03-docker` | Multi-stage `Dockerfile`, `.dockerignore`, `.env.example` |
| #13 | `feature/sprint-03-compose` | `docker-compose.yml` dev workflow + `docs/docker-development.md` |
| #14 | `feature/sprint-03-github-actions` | `.github/workflows/ci.yml` + `docs/ci-cd.md` + CI badge |
| #15 | `feature/sprint-03-documentation` | Sprint 3 documentation refresh |

Concretely, the repository gained:

- **Docker** — a multi-stage [`Dockerfile`](../../Dockerfile) with three named
  targets (`builder` → `development` / `runtime`) on `python:3.12-slim-bookworm`,
  a [`.dockerignore`](../../.dockerignore), and OCI provenance labels.
- **Compose** — a [`docker-compose.yml`](../../docker-compose.yml) with a
  bind-mounted `dev` service and an on-demand `pipeline` profile.
- **GitHub Actions** — a [`ci.yml`](../../.github/workflows/ci.yml) `quality` job
  (Ruff lint + format, pytest) gating a `docker` job that builds the `runtime`
  image and validates it (non-root UID 10001, imports resolve, DVC present).
- **Documentation** — [containerization.md](../containerization.md),
  [docker-development.md](../docker-development.md), [ci-cd.md](../ci-cd.md),
  [ADR-005](../decisions/ADR-005-containerization-strategy.md), plus a refreshed
  architecture / roadmap / project-structure / docs index, and the
  `v1.2.0` [CHANGELOG](../../CHANGELOG.md) entry.

## 3. Engineering decisions

- **Multi-stage Docker build** — a single `Dockerfile` with a `builder` stage
  (compilers, wheel builds) feeding lean downstream stages. Build tooling is
  discarded before runtime, keeping the shipped image small and its attack
  surface minimal, with a cache-friendly layer order (manifest before source).
  Rationale in [ADR-005](../decisions/ADR-005-containerization-strategy.md).
- **Non-root container** — the `runtime` image runs as a dedicated,
  fixed-UID/GID user (`10001`, `nologin` shell). Least privilege by default and a
  prerequisite for the Kubernetes restricted Pod Security Standard (v4). CI
  asserts the UID so the security contract can't silently regress.
- **CI quality gates** — CI is **validation only**: it lints, tests, and
  builds/validates the image, but never deploys, pushes images, or touches
  Kubernetes. `contents: read` is the only permission granted, structurally
  preventing a publish. The image build is gated on the fast lint/test job so
  build minutes are never spent on a change that already fails.
- **Development vs pipeline containers** — one source of truth, two shapes. The
  `development` target (root, full toolchain, bind-mounted source, `docker
  compose up`) serves the inner loop; the `runtime` target (non-root, minimal,
  externalized state, `docker compose --profile pipeline run`) is the artifact CI
  builds and v4 will schedule. Named targets keep dev and prod from drifting the
  way separate Dockerfiles would.

## 4. What went well

- **Clean branch-per-concern flow.** Five focused PRs (design → docker → compose
  → CI → docs) kept each change reviewable and the history legible.
- **Design before build.** Ratifying the container design in ADR-005 *before*
  writing the `Dockerfile` meant implementation followed settled choices rather
  than improvised ones — base image, non-root, externalized state, and cache
  strategy were decided, not discovered.
- **CI encodes the security contract.** The build-validation step (non-root UID,
  core imports, DVC entrypoint) turns "the image is hardened" into an executable
  assertion instead of a claim in a doc.
- **Documentation kept pace with code.** Each PR shipped its own docs, so the
  `docs/` set described the repository as it actually became — no large
  reconciliation debt at the end of the sprint.

## 5. What was difficult

- **Scientific Python in a slim image.** `numpy`/`pandas`/`scikit-learn` drove
  the base-image choice: `slim` (glibc, prebuilt wheels) over the smaller Alpine
  (musl forces slow, brittle source builds). A real size-vs-reliability trade-off,
  resolved in favor of reliability and recorded in ADR-005.
- **Keeping CI honestly "validation only."** It is tempting to let CI push the
  image once it already builds it. Holding the line — no registry, no
  credentials, least-privilege token — required deliberately *not* doing the
  easy next step and documenting why (CD is a separate, ratified decision).
- **Documentation reconciliation surfaced older gaps.** Truthfully describing the
  container flow re-exposed the pre-existing DVC correctness gap (see §6) and the
  stale root README, which had to be handled as documentation truth without
  pulling unplanned engineering work into the sprint.

## 6. What was deliberately deferred

Each of these is a conscious "not this sprint" decision, not an oversight:

- **DVC correctness** — `dvc.yaml`/`params.yaml` parameter-name mismatch
  (`train.data`/`train.model` vs `train.input`/`train.output`) and the
  `preprocess` output not being consumed downstream (train/evaluate read
  `data/raw/data.csv`). Carried from Sprint 2; this is core-pipeline engineering,
  not a release-gate doc fix. **→ Sprint 4.**
- **Digest pinning** — the base image is pinned by codename
  (`python:3.12-slim-bookworm`), not yet by `sha256` digest. Byte-for-byte
  reproducible rebuilds are a follow-up recorded in the `Dockerfile` and ADR-005.
- **Dependency pinning** — `requirements.txt` is not fully hash/version-locked;
  a lockfile (and reproducible resolves) is deferred alongside digest pinning.
- **Continuous delivery** — publishing the image on release, image scanning
  (Trivy), SBOM, and signing are v3's *delivery* half; CI ships the *integration*
  half only. CD needs its own ADR (registry, signing, deploy target).
- **Kubernetes** — orchestrated stage execution, ConfigMaps/Secrets, and resource
  requests/limits are Roadmap v4. The non-root, externalized-state image was
  built to drop into it without rework, but nothing K8s ships this sprint.
- **Observability** — no metrics, tracing, or health/liveness surfaces. The batch
  image intentionally has no `HEALTHCHECK` (run-to-completion, not a service);
  monitoring and alerting belong to Roadmap v6.

## 7. Lessons learned

- **Ratify the design first, then build.** ADR-before-Dockerfile removed
  mid-implementation debate and gave every reviewer a shared reference. Worth
  repeating for the CD and Kubernetes decisions ahead.
- **Make contracts executable.** A security property asserted in CI (non-root
  UID) is worth more than the same property described in prose; extend this to
  future guarantees (e.g. image scan thresholds) as CD lands.
- **Documentation is a correctness surface.** Writing the container docs honestly
  is what re-surfaced the DVC gap and the stale README. Keep docs first-class —
  they catch drift that code review alone misses.
- **Guard scope at the release gate.** Deferrals only stay honest if they are
  written down. Recording DVC correctness, digest/dependency pinning, CD, K8s,
  and observability as explicit "not yet" items is what keeps `v1.2.0` a truthful
  release and Sprint 4 a clear one.

---

## Related documentation

- [Containerization Strategy](../containerization.md)
- [Docker Development Workflow](../docker-development.md)
- [CI/CD](../ci-cd.md)
- [ADR-005 — Containerization Strategy](../decisions/ADR-005-containerization-strategy.md)
- [Roadmap](../roadmap.md)
- [Changelog](../../CHANGELOG.md)
