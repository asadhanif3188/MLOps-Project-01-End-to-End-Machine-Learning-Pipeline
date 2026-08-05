# ADR-005: Containerization Strategy

- **Status:** Accepted
- **Date:** 2026-08-02
- **Deciders:** Asad Hanif
- **Related:** [Containerization Strategy](../containerization.md),
  [Architecture](../architecture.md), [Roadmap](../roadmap.md)
  (v3 CI/CD, v4 Kubernetes),
  [ADR-004 (Python Quality Toolchain)](ADR-004-python-quality-toolchain.md)

> **Numbering note.** This record is **ADR-005**, not ADR-004: ADR-004 is already
> assigned to the Python Quality Toolchain decision. Containerization takes the
> next free number.

> **Scope note.** This ADR ratifies the *design*, and that design was
> **implemented in Sprint 3**: the repository now ships a multi-stage
> `Dockerfile`, a `.dockerignore`, and a `docker-compose.yml` that follow the
> choices recorded here. The record exists so that the implementation followed a
> settled design rather than improvised choices. The full rationale and the
> as-built details live in [containerization.md](../containerization.md).

## Context

The pipeline is a **batch, file-based ML workflow** (`preprocess` → `train` →
`evaluate`) orchestrated by DVC and tracked by MLflow on DagsHub (see
[architecture.md](../architecture.md)). It currently runs directly on a
developer's machine, which makes execution environment-dependent and undermines
the project's central goal of reproducibility.

Two upcoming milestones **require** a container artifact:

- **Roadmap v3 (CI/CD)** — CI should build, test, scan, and reproduce the
  pipeline against a consistent image.
- **Roadmap v4 (Kubernetes)** — the roadmap explicitly calls for containerizing
  the pipeline and running stages as orchestrated workloads.

We therefore need a containerization strategy that delivers environment parity
today and drops into a hardened cluster later, without rework. Requirements:

- reproducible, portable (OCI) execution across laptop, CI, and cluster;
- a clean development/production split from a single source of truth;
- small, secure, non-root images with no baked-in secrets or data; and
- fast, cache-friendly builds for a scientific Python stack
  (`numpy`/`pandas`/`scikit-learn`).

## Decision

Adopt the following strategy (detailed in
[containerization.md](../containerization.md)):

- **Docker/OCI** as the container standard — portable across CI, registries, and
  Kubernetes; the natural on-ramp to Roadmap v4.
- **Multi-stage `Dockerfile`** with a `builder` stage (build toolchain, wheel
  compilation) and a lean `runtime` stage that copies only the installed
  environment and application code. **Development** and **production** are two
  named targets of the *same* file, not separate files.
- **Base image: `python:3.12-slim`** (glibc, prebuilt scientific wheels),
  matching the project's Python 3.12 target. **Alpine is rejected** (musl breaks
  scientific wheels); **distroless is deferred** as a future runtime-stage
  optimization. Base image to be **digest-pinned** at implementation.
- **Non-root production containers** — a dedicated UID/GID, read-only root
  filesystem where feasible, dropped capabilities, `no-new-privileges`.
- **Twelve-factor configuration** — non-secret config via environment variables
  (with safe image defaults, e.g. `LOG_LEVEL`); **secrets injected at runtime**
  (`--env-file`/CI secrets today, ConfigMaps/Secrets under K8s), **never** baked
  into layers. The `.env` pattern stays host-only and `.dockerignore`d.
- **Stateless containers with externalized state** — `data/`, `models/`, and
  `logs/` are mounted volumes (bind → named volume → PVC), never baked in.
- **Build-cache optimization** — dependency manifests copied and installed before
  source; BuildKit cache mounts for pip; a `.dockerignore` to keep the context
  small.
- **Image lifecycle** — build → tag (immutable `sha-*` **and** SemVer, aligned
  with [versioning.md](../versioning.md)) → scan (Trivy) → push (**GHCR**) → run
  (ephemeral job) → retire (retention policy). Published tags are immutable.
- **Forward compatibility** — the image is designed to satisfy the Kubernetes
  restricted Pod Security Standard and to be the artifact CI/CD builds, tests,
  scans, and publishes.

## Alternatives Considered

1. **Virtual machines / Vagrant instead of containers.**
   - *Decision:* rejected — heavyweight boot and footprint for short-lived batch
     jobs, weaker build reproducibility, and not the unit Kubernetes schedules.
2. **`python:3.12-alpine` base (smallest image).**
   - *Decision:* rejected — musl libc forces slow, brittle source builds of
     `numpy`/`scikit-learn`. The size win is not worth the fragility.
3. **Distroless runtime base now.**
   - *Decision:* deferred — strong for minimal surface, but no shell/package
     manager complicates first-cut debugging. Recorded as a future runtime-stage
     optimization once the build stabilizes.
4. **Separate `Dockerfile.dev` and `Dockerfile.prod`.**
   - *Decision:* rejected — two files drift. Multi-stage named targets give the
     same dev/prod split from one source of truth, consistent with
     [ADR-004](ADR-004-python-quality-toolchain.md)'s "one source per concern"
     principle.
5. **Baking data/models into the image.**
   - *Decision:* rejected — large, DVC-versioned, environment-specific artifacts
     belong on mounted volumes; baking them breaks immutability and bloats images.
6. **Running as root for simplicity.**
   - *Decision:* rejected for production — violates least privilege and the
     Kubernetes restricted PSS. Root is a *dev-only* convenience exception.

## Consequences

**Positive**

- Environment parity across laptop, CI, and (later) Kubernetes; "works on my
  machine" is eliminated for pipeline stages.
- Small, non-root, scanned images with no baked-in secrets or data — a
  defensible security posture aligned with [SECURITY.md](../../SECURITY.md).
- The artifact is Kubernetes-ready (v4) and CI/CD-ready (v3) by construction, so
  those sprints inherit a settled design.
- Reproducible builds (pinned base + dependencies) reinforce the project's
  DVC/MLflow reproducibility guarantees.

**Trade-offs and follow-ups**

- **Implemented in Sprint 3.** The `Dockerfile`, `.dockerignore`, and
  `docker-compose.yml` described by this design now exist in the repository;
  Sprint 3 CI also builds and validates the production image. This ADR ratified
  the design that the implementation followed.
- `slim` is larger than Alpine/distroless — an accepted cost for build
  reliability, partly offset by multi-stage builds; distroless revisit is a
  follow-up.
- The **CI provider** and **Kubernetes orchestration/secret handling** are not
  yet chosen; each will get its own ADR (Roadmap v3/v4) that builds on this one.
- Base image should move from a tag pin to a **digest pin** when implemented.
