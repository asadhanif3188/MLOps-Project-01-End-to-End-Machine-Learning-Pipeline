# Containerization Strategy

This document defines **how the pipeline is containerized** and the engineering
reasoning behind each decision.

The strategy was authored first, as a design document, and is now **implemented**:
the repository ships a multi-stage [`Dockerfile`](../Dockerfile) and a
[`.dockerignore`](../.dockerignore) built directly from these decisions. A
`docker-compose.yml` is **not** included yet (deferred, see
[§14](#14-future-cicd-integration) and the roadmap). Concrete build and run
instructions are in [§15](#15-build--run).

The ratified summary lives in
[ADR-005](decisions/ADR-005-containerization-strategy.md); this document carries
the full rationale.

> **Scope note.** Containerization is a prerequisite for both CI/CD
> ([Roadmap v3](roadmap.md#version-3--cicd)) and Kubernetes
> ([Roadmap v4](roadmap.md#version-4--kubernetes)). This strategy is written now,
> during Sprint 3, so that the later implementation sprints inherit a settled
> design instead of ad-hoc choices.

---

## 1. Goals

Containerization must serve the project's core value — **reproducibility** — and
extend it from "reproducible on my machine" to "reproducible anywhere". The
goals, in priority order:

1. **Environment parity.** The same image runs on a developer laptop, in CI, and
   (later) on a cluster. Eliminate "works on my machine" for the DVC pipeline
   stages (`preprocess`, `train`, `evaluate`).
2. **Reproducible execution.** Pin the Python version, OS packages, and
   dependencies so a build is deterministic and re-runnable months later.
3. **Portability.** Produce OCI-compliant images that run unmodified under
   Docker, CI runners, and Kubernetes ([§13](#13-future-kubernetes-compatibility)).
4. **Small, secure images.** Minimize image size and attack surface via
   multi-stage builds and a minimal base ([§6](#6-multi-stage-build-strategy),
   [§7](#7-base-image-selection)).
5. **Fast, cache-friendly builds.** Structure layers so routine code changes do
   not reinstall the dependency tree ([§12](#12-build-cache-optimization)).
6. **Clear dev/prod separation.** Optimize the development image for iteration
   speed and the production image for size and safety
   ([§4](#4-development-vs-production-containers)).

**Non-goals for this iteration.** This strategy does **not** cover model serving
(there is no online inference component today — see
[architecture §1](architecture.md#1-system-overview)), GPU builds, or a full
Kubernetes deployment. Those are deferred to Roadmap v4–v6 and will get their own
ADRs.

---

## 2. Why Docker

Docker (and the broader OCI ecosystem) is the default containerization choice for
this project because:

- **It is the de-facto standard.** Docker images are OCI-compliant and run on
  every major CI system, registry, and orchestrator without modification. This
  directly serves the portability goal.
- **The ecosystem already assumes it.** DVC, MLflow, and the Python data stack
  are routinely packaged and executed in containers; documentation, base images,
  and community patterns are abundant.
- **It composes with the existing tooling.** The pipeline is a set of CLI stages
  (`python src/<stage>.py`) driven by `dvc repro`. That maps cleanly onto a
  container entry point with no application-server machinery required.
- **It is the on-ramp to Kubernetes.** Roadmap v4 targets Kubernetes, which
  schedules OCI images. Adopting Docker now means the artifact we build for local
  use is the same artifact the cluster runs later.
- **BuildKit** provides modern build features we rely on for caching and
  multi-stage builds ([§6](#6-multi-stage-build-strategy),
  [§12](#12-build-cache-optimization)).

We standardize on the **image format and CLI**, not on a specific daemon. The
resulting images are portable to Podman, `buildah`, `nerdctl`, and CI-native
builders; nothing in this strategy is Docker-daemon-specific.

---

## 3. Why Containers Instead of Virtual Machines

Both containers and VMs isolate workloads, but containers fit a **batch,
file-based ML pipeline** far better:

| Dimension | Containers | Virtual machines |
|-----------|-----------|------------------|
| Isolation boundary | Shared host kernel, isolated user space | Full guest OS per instance |
| Start-up time | Milliseconds–seconds | Seconds–minutes |
| Image size | Tens–hundreds of MB | GBs (full OS) |
| Density | Many per host | Few per host |
| Build reproducibility | Declarative `Dockerfile`, layer-cached | Heavier (images/IaC), slower to rebuild |
| Orchestration fit | Native to Kubernetes (Roadmap v4) | Requires a separate virtualization layer |

**Why it matters here.** The pipeline stages are **short-lived, on-demand batch
jobs**. A container that starts in under a second and exits is a natural fit; a
VM's boot overhead and footprint are pure waste for a job that runs and stops.
Containers also give us **immutable, versioned artifacts** that align with the
project's reproducibility ethos, and they are the unit Kubernetes schedules —
making containers the correct rung on the ladder toward v4.

VMs are not rejected as a concept — the cluster nodes that eventually run these
containers may themselves be VMs — but the **pipeline artifact** is a container,
not a VM image.

---

## 4. Development vs Production Containers

The project maintains **two build targets from one `Dockerfile`** (via
multi-stage builds, [§6](#6-multi-stage-build-strategy)) rather than two
unrelated files. This guarantees the production image is a strict subset of what
was validated in development.

| Concern | Development image | Production image |
|---------|-------------------|------------------|
| Dependencies | Runtime **+ dev** (`requirements-dev.txt`: Ruff, mypy, pytest, pre-commit) | Runtime only (`requirements.txt`) |
| Source code | **Bind-mounted** from the host for live edits | **Baked in** at build time (immutable) |
| User | May run as root for convenience | **Non-root** ([§9](#9-non-root-user-rationale)) |
| Entry point | Interactive shell / `make` targets | Pipeline stage command (`dvc repro` or a stage) |
| Filesystem | Writable | Read-only root where feasible ([§8](#8-security-considerations)) |
| Size priority | Low — convenience wins | High — minimal surface and footprint |
| Toolchain | Full quality toolchain ([ADR-004](decisions/ADR-004-python-quality-toolchain.md)) available inside the container | Excluded |

**Development image** optimizes for the inner loop: mount the working tree, run
`make format`/`make lint`/`make test`, and iterate without rebuilding. It mirrors
the [Developer Guide](developer-guide.md) workflow inside a container so
contributors on any OS (notably Windows, where `make` is not native) get an
identical environment.

**Production image** optimizes for a trustworthy, minimal artifact: only runtime
dependencies, code copied in, non-root, and as close to read-only as the stages
allow. This is the image CI builds, scans, and pushes
([§14](#14-future-cicd-integration)), and the one Kubernetes eventually runs
([§13](#13-future-kubernetes-compatibility)).

---

## 5. Image Lifecycle

Images move through a defined lifecycle so that every running container is
traceable to a commit:

```text
 build ──▶ tag ──▶ scan ──▶ push ──▶ run ──▶ retire
   │        │        │        │        │        │
 Dockerfile SemVer  Trivy   registry ephemeral prune old
 + BuildKit + git-sha (CVEs) (GHCR)   job/K8s  tags/images
```

1. **Build.** From the pinned `Dockerfile` using BuildKit, with a build cache
   ([§12](#12-build-cache-optimization)).
2. **Tag.** Every image gets **two** tags:
   - an **immutable** tag — the short git SHA (`:sha-<abcdef1>`) — for exact
     traceability, and
   - a **human** tag — a SemVer release tag (`:1.2.0`) aligned with the project's
     [versioning policy](versioning.md), plus a moving `:latest` for the default
     branch (never used as a deployment reference).
3. **Scan.** A vulnerability scan (e.g. **Trivy**) runs against the built image
   before it is published; the scan is a CI gate in Roadmap v3.
4. **Push.** Publish to a registry. **GitHub Container Registry (GHCR)** is the
   default choice because the repository is on GitHub and CI (Roadmap v3) will
   authenticate to it natively.
5. **Run.** The image runs as an ephemeral job — locally, in CI, or as a
   Kubernetes `Job`/`CronJob` per stage. Containers are cattle: they run to
   completion and are discarded.
6. **Retire.** Old images are pruned on a retention policy (keep releases and the
   last N SHA builds); unreferenced layers are garbage-collected.

**Immutability rule.** A published tag is never rebuilt with different content.
Fixes produce a new tag, never an overwrite — this keeps `sha-*` tags a reliable
audit trail and matches DVC/MLflow's reproducibility guarantees.

---

## 6. Multi-Stage Build Strategy

The `Dockerfile` uses **multi-stage builds** to separate the heavy build-time
environment from the lean runtime image.

```text
┌─ Stage: builder ──────────────┐        ┌─ Stage: runtime ─────────────┐
│ base: python:3.12-slim        │        │ base: python:3.12-slim       │
│ + build toolchain (gcc, etc.) │        │ (no compilers)               │
│ + pip install into a venv /   │  copy  │ COPY --from=builder <venv>   │
│   wheels                      │ ─────▶ │ COPY src/ (prod) or mount    │
│ compiles numpy/scikit-learn   │  only  │ non-root USER                │
│   wheels if needed            │ what   │ ENTRYPOINT: pipeline stage   │
└───────────────────────────────┘ runs   └──────────────────────────────┘
```

**Why multi-stage:**

- **Smaller images.** Compilers, headers, and build caches stay in the discarded
  `builder` stage. Only the installed environment and application code reach the
  final image.
- **Smaller attack surface.** No `gcc`/`build-essential` in the shipped image
  means fewer packages an attacker can leverage
  ([§8](#8-security-considerations)).
- **One source of truth, two targets.** Named stages let a `dev` target extend
  the environment with `requirements-dev.txt` while the default `runtime` target
  stays minimal ([§4](#4-development-vs-production-containers)) — without a second
  file to drift, consistent with the "one source of truth per concern" principle
  from [ADR-004](decisions/ADR-004-python-quality-toolchain.md).

**Relevance to this project.** The scientific stack (`numpy`, `pandas`,
`scikit-learn`) may pull or build native wheels. Isolating that work in a
`builder` stage keeps the runtime image free of compilers while still producing a
correct, fully-installed environment.

---

## 7. Base Image Selection

**Decision: `python:3.12-slim` (Debian-based) as the default base**, for both
stages, with a documented path to distroless for the production runtime later.

Python **3.12** matches the project's declared target (`pyproject.toml`,
`[tool.ruff] target-version` and the mypy config).

| Candidate | Pros | Cons | Verdict |
|-----------|------|------|---------|
| `python:3.12-slim` | Small-ish, glibc, pip-compatible **binary wheels** for numpy/scikit-learn, easy debugging | Larger than distroless/Alpine | **Chosen** — best balance for a scientific Python stack |
| `python:3.12` (full) | Everything included | Large; unnecessary tooling and surface | Rejected — too heavy |
| `python:3.12-alpine` | Very small | **musl libc** breaks/needs source builds for many scientific wheels → slow builds, fragility | Rejected — poor fit for numpy/scikit-learn |
| `gcr.io/distroless/python3` | Minimal surface, no shell/package manager | Harder to debug; no shell; version pinning is coarser | **Deferred** — strong candidate for the *production runtime* stage once the build stabilizes |

**Rationale.** The single biggest base-image risk for an ML project is **Alpine's
musl libc**, which frequently forces slow, brittle source compilation of the
scientific stack. `slim` gives us glibc and prebuilt wheels — fast, reliable
builds — at a modest size cost that multi-stage builds already mitigate.
Distroless is the natural next step for the runtime stage (smallest surface, no
shell) and is recorded as a **future optimization**, not adopted now, so the
first implementation stays debuggable.

**Pinning.** The base image is pinned by **major.minor tag now** and should move
to a **digest pin** (`python:3.12-slim@sha256:…`) when the build is implemented,
so rebuilds are byte-for-byte reproducible.

---

## 8. Security Considerations

Container security is treated as a first-class requirement, consistent with the
repository's existing [Security Policy](../SECURITY.md). The controls:

- **Minimal base + multi-stage** — fewer packages, no compilers in the runtime
  image ([§6](#6-multi-stage-build-strategy), [§7](#7-base-image-selection)).
- **Non-root user** — the container runs as an unprivileged user
  ([§9](#9-non-root-user-rationale)).
- **No secrets in the image.** Credentials (DagsHub/MLflow tokens, S3 keys) are
  **never** baked into layers or `ENV`. They arrive at runtime via environment
  variables or mounted secrets ([§10](#10-environment-variable-strategy)). The
  existing `.env`/`.env.example` pattern stays **host-only** and is excluded via
  `.dockerignore`.
- **`.dockerignore`.** Exclude `.git`, `.env`, `data/`, `models/`, `logs/`,
  caches, and virtualenvs from the build context — smaller context, faster
  builds, and no accidental secret or large-artifact leakage into layers.
- **Pinned dependencies.** Base image digest and Python requirements are pinned
  so a build cannot silently pull a compromised or breaking version.
- **Vulnerability scanning.** Images are scanned (e.g. **Trivy**) in the lifecycle
  ([§5](#5-image-lifecycle)) and gated in CI ([§14](#14-future-cicd-integration)).
- **Read-only root filesystem where feasible.** Runtime containers request a
  read-only root FS, with writes confined to explicit volumes/`tmpfs`
  ([§11](#11-volume-strategy)). This aligns with the Kubernetes restricted
  Pod Security Standard ([§13](#13-future-kubernetes-compatibility)).
- **Drop capabilities / no privilege escalation.** Runtime containers drop all
  Linux capabilities and set `no-new-privileges`; they never run privileged.
- **Supply-chain hygiene (future).** Generate an **SBOM** and image provenance
  during CI builds so published images are auditable.

---

## 9. Non-Root User Rationale

**Production containers run as a dedicated non-root user**, created in the
`Dockerfile` (a fixed UID/GID, e.g. `appuser`), with the application directory
owned by that user.

Why this is non-negotiable for the production image:

- **Least privilege.** A process compromise inside the container is confined to an
  unprivileged user rather than `root`, limiting blast radius.
- **Defense in depth against container escape.** Running as UID 0 inside the
  container maps to elevated risk if an escape or a misconfigured mount exposes
  the host; a non-root UID materially reduces that risk.
- **Kubernetes compatibility.** The **restricted** Pod Security Standard
  effectively requires `runAsNonRoot: true`. Building the image non-root now means
  it drops into a hardened cluster in Roadmap v4 with no rework
  ([§13](#13-future-kubernetes-compatibility)).
- **Read-only-friendly.** A non-root user with writes limited to declared volumes
  pairs naturally with a read-only root filesystem
  ([§8](#8-security-considerations)).

The **development image** may run as root for convenience (bind-mount permission
ergonomics), but this is an explicit, dev-only exception — the artifact that
ships and runs in CI/production is always non-root.

---

## 10. Environment Variable Strategy

Configuration follows **twelve-factor** principles: **config lives in the
environment, secrets are injected at runtime, and neither is baked into the
image.**

- **Config vs secrets, separated.**
  - *Non-secret config* (e.g. `LOG_LEVEL`, already consumed by
    [`src/logging_config.py`](../src/logging_config.py); MLflow tracking URI) may
    have safe defaults set via `ENV` in the `Dockerfile`.
  - *Secrets* (DagsHub token, MLflow credentials, S3 keys) are **never** in the
    image or in `ENV`. They are supplied at run time.
- **Local development.** Keep using the existing `python-dotenv` + `.env` pattern
  (see [`.env.example`](../.env.example)). `.env` is **host-only** and
  `.dockerignore`d; in a container it is passed with `--env-file` or, in Compose,
  an `env_file:` reference — the file itself is never copied into a layer.
- **CI.** Secrets come from the CI provider's secret store and are exposed to the
  build/run as masked environment variables ([§14](#14-future-cicd-integration)).
  Build-time secrets, if ever needed, use BuildKit `--secret` mounts, not build
  args.
- **Kubernetes (future).** Non-secret config comes from **ConfigMaps**; secrets
  from **Secrets** (ideally backed by an external secrets manager). This is
  exactly the externalization Roadmap v4 calls for.
- **Precedence.** Runtime environment overrides image defaults, which override
  in-code defaults — a single, predictable order.

**Rule of thumb:** if a value differs between environments, it is an env var; if
it is sensitive, it is a runtime-injected secret. Nothing sensitive is ever
committed or layered.

---

## 11. Volume Strategy

Containers are **stateless and ephemeral**; all durable state lives **outside** the
image on mounted volumes. This keeps images immutable and lets the DVC-managed
data flow work unchanged.

| Path | Purpose | Baked into image? | Mount type |
|------|---------|-------------------|------------|
| `data/` (raw + processed) | DVC-tracked datasets | **No** | Bind (dev) / volume / PVC (K8s) |
| `models/` | Serialized model artifacts (git-ignored, DVC-tracked) | **No** | Bind / volume / PVC |
| `logs/` | Rotating pipeline logs (`logs/pipeline.log`) | **No** | Volume / `tmpfs` |
| Pip / DVC / MLflow caches | Speed up repeated runs | **No** | Cache volume |
| `src/` (production) | Application code | **Yes** (copied) | — |
| `src/` (development) | Live-edited code | No | Bind mount from host |

**Principles:**

- **Data and models are mounted, never baked.** They are large, versioned by DVC,
  and environment-specific. Baking them would bloat images and break
  immutability. Containers `dvc pull` into the mounted `data/`/`models/` at run
  time, or the volumes are pre-populated.
- **Logs are externalized.** `logs/` is a write target on a volume (or `tmpfs`),
  consistent with the read-only-root goal ([§8](#8-security-considerations)) and
  the [Logging Strategy](logging.md).
- **Dev bind-mounts source; prod bakes it.** This is the practical mechanism
  behind the dev/prod split ([§4](#4-development-vs-production-containers)).
- **Kubernetes mapping.** Bind mounts and named volumes become
  **PersistentVolumeClaims** (for data/models) and `emptyDir`/`tmpfs` (for logs
  and caches) in Roadmap v4 — the same conceptual model, cluster-native
  implementation.

---

## 12. Build Cache Optimization

Build speed matters for the inner loop and for CI cost. The `Dockerfile` is
structured so that **dependency installation is cached independently of source
changes**.

- **Order layers by change frequency.** Copy dependency manifests
  (`requirements.txt`, `requirements-dev.txt`) and install **before** copying
  `src/`. Editing a stage script then invalidates only the cheap final layers,
  not the expensive dependency install.

  ```text
  COPY requirements*.txt .      # changes rarely
  RUN  pip install -r ...       # heavy, cached until requirements change
  COPY src/ ./src/              # changes often — invalidates only from here down
  ```

- **BuildKit cache mounts** for the pip cache
  (`RUN --mount=type=cache,target=/root/.cache/pip …`) so wheels are reused across
  builds without living in an image layer.
- **`.dockerignore`** keeps the build context small (excludes `.git`, `data/`,
  `models/`, `logs/`, caches), which speeds every build and avoids cache-busting
  from irrelevant file churn ([§8](#8-security-considerations)).
- **Deterministic dependencies.** Pinned requirements make cache hits stable and
  builds reproducible ([§7](#7-base-image-selection)).
- **Registry cache in CI.** CI builds use `--cache-from`/`--cache-to` against the
  registry (or the runner's cache) so cold CI builders still benefit from prior
  layers ([§14](#14-future-cicd-integration)).

---

## 13. Future Kubernetes Compatibility

Every decision above is made so the resulting image **drops into Kubernetes
(Roadmap v4) without redesign**:

- **OCI images** run as-is on the cluster ([§2](#2-why-docker)).
- **Non-root + read-only root FS + dropped capabilities** satisfy the
  **restricted Pod Security Standard** out of the box
  ([§8](#8-security-considerations), [§9](#9-non-root-user-rationale)).
- **Externalized config/secrets** map directly onto **ConfigMaps** and
  **Secrets** — no code change from the container's env-var contract
  ([§10](#10-environment-variable-strategy)).
- **Externalized state on volumes** maps onto **PersistentVolumeClaims** for
  `data/`/`models/` and ephemeral volumes for logs/caches
  ([§11](#11-volume-strategy)).
- **Stateless, run-to-completion stages** map onto Kubernetes **`Job`** (one-off
  `dvc repro` or a single stage) and **`CronJob`** (scheduled retraining) — the
  natural primitives for a batch pipeline.
- **Immutable, digest-pinned images** give the cluster reproducible, auditable
  deployments ([§5](#5-image-lifecycle)).
- **Resource limits.** The image imposes no assumptions that block setting CPU/
  memory **requests and limits** per stage — a v4 objective.

The concrete manifests (base image ratification, orchestration approach, secret
handling) will be captured in their **own ADRs** when Roadmap v4 begins, per the
roadmap's existing TODO. This document only guarantees the image is *ready* for
them.

---

## 14. Future CI/CD Integration

Containerization is the artifact CI/CD produces and validates
([Roadmap v3](roadmap.md#version-3--cicd)). The intended integration:

- **Build in CI.** On pull requests, build the image with BuildKit and layer
  caching ([§12](#12-build-cache-optimization)) to validate that the container
  builds cleanly — an extension of the existing quality gates
  ([ADR-004](decisions/ADR-004-python-quality-toolchain.md)).
- **Test in the container.** Run the pytest suite
  ([Testing Strategy](testing-strategy.md)) *inside* the built image so tests
  validate the shipped environment, not just the host.
- **Scan as a gate.** Fail the build on high/critical CVEs via image scanning
  (Trivy) before anything is published ([§5](#5-image-lifecycle)).
- **Publish on merge/release.** On merges to the default branch and on tagged
  releases, push to **GHCR** with the dual tag scheme (`sha-*` + SemVer),
  authenticated via the CI provider's native registry credentials.
- **Provenance & SBOM.** Emit build provenance and an SBOM for published images so
  releases are auditable and supply-chain-verifiable
  ([§8](#8-security-considerations)).
- **Reproduce the pipeline in CI.** Longer term, use the image to run
  `dvc repro`/`dvc status` in CI — the v3 objective of automated pipeline
  validation — on the exact artifact that runs in production.

The CI provider and pipeline design are **not yet selected** (open TODO in
[Roadmap v3](roadmap.md#version-3--cicd)); this section defines the container's
role once they are, and will be ratified alongside the CI ADR.

---

## 15. Build & Run

The repository ships a single multi-stage [`Dockerfile`](../Dockerfile) with
three targets — `builder` (internal), `runtime` (default, production), and
`development` — and a [`.dockerignore`](../.dockerignore) that keeps the build
context small and secret-free. BuildKit is required for the cache mounts and
`# syntax` directive; it is the default in current Docker.

### Build the production image

```bash
# From the repository root. The two build args stamp OCI provenance labels.
docker build \
  --build-arg VCS_REF="$(git rev-parse --short HEAD)" \
  --build-arg BUILD_VERSION="1.2.0" \
  -t ml-pipeline:local .
```

This produces the lean, **non-root** `runtime` image. The build compiles/installs
dependencies in the throwaway `builder` stage and copies only the resulting
virtualenv forward, so no compilers ship in the final image.

### Build the development image

```bash
docker build --target development -t ml-pipeline:dev .
```

Identical runtime environment plus the quality toolchain (Ruff, mypy, pytest,
pre-commit). Intended to be run with the working tree bind-mounted:

```bash
# Live-edit on the host; run the toolchain inside the container.
docker run --rm -it -v "$(pwd)":/app ml-pipeline:dev
# then, inside: make check   /   python -m pytest   /   dvc repro
```

### Run the pipeline

The `runtime` image's default command is `dvc repro`. State (data, models, logs)
is **externalized as volumes** and credentials are **injected at run time** —
nothing sensitive lives in the image ([§10](#10-environment-variable-strategy),
[§11](#11-volume-strategy)).

```bash
docker run --rm \
  --env-file .env \
  -v "$(pwd)/data":/app/data \
  -v "$(pwd)/models":/app/models \
  -v "$(pwd)/logs":/app/logs \
  ml-pipeline:local
```

Run a single stage by overriding the command:

```bash
docker run --rm --env-file .env \
  -v "$(pwd)/data":/app/data -v "$(pwd)/models":/app/models \
  ml-pipeline:local python src/preprocess.py
```

> On Windows PowerShell, replace `$(pwd)` with `${PWD}` and the trailing `\`
> line-continuations with a backtick (`` ` ``), or put the command on one line.

### Run hardened (optional, recommended)

The image is designed to run under a locked-down runtime — useful as a local
preview of the Kubernetes restricted Pod Security Standard
([§13](#13-future-kubernetes-compatibility)):

```bash
docker run --rm \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --tmpfs /tmp \
  --env-file .env \
  -v "$(pwd)/data":/app/data \
  -v "$(pwd)/models":/app/models \
  -v "$(pwd)/logs":/app/logs \
  ml-pipeline:local
```

Writes are confined to the mounted volumes and `/tmp`; the root filesystem stays
read-only.

### Notes & known follow-ups

- **`dvc repro` prerequisites.** The default command operates on the DVC
  pipeline: it needs the DVC-tracked `data/` (mounted, and/or fetched with
  `dvc pull`) and valid MLflow/DagsHub credentials in the environment. Without a
  configured remote and data, run individual stages or use the development image
  for exploration.
- **No `HEALTHCHECK`** by design — this is a run-to-completion batch job, not a
  service ([§8](#8-security-considerations)). A liveness/readiness probe belongs
  to the future serving component (Roadmap v6). A build-time import smoke test in
  the `Dockerfile` validates the environment instead.
- **Dependency pinning.** [`requirements.txt`](../requirements.txt) currently
  pins by name, not version/hash. The base image is pinned by codename
  (`python:3.12-slim-bookworm`). Moving both to digests/hashes for byte-for-byte
  reproducibility is the tracked follow-up from
  [ADR-005](decisions/ADR-005-containerization-strategy.md).
- **Image size.** The validated images measure ~1.6 GB (runtime) and ~2.1 GB
  (development). The footprint is dominated by the scientific/MLOps stack
  (`scipy`, `pyarrow`, `matplotlib`, `mlflow`, `boto3`) rather than avoidable
  build cruft — multi-stage already keeps compilers and caches out of the runtime
  image. Concrete reductions for a later pass: switch tracking to `mlflow-skinny`,
  drop unused heavy transitives, and move the runtime stage to **distroless**
  ([§7](#7-base-image-selection)).

---

## Related Documentation

- [ADR-005 — Containerization Strategy](decisions/ADR-005-containerization-strategy.md)
- [Architecture](architecture.md) — the pipeline this image runs
- [Roadmap](roadmap.md) — v3 (CI/CD) and v4 (Kubernetes) context
- [ADR-004 — Python Quality Toolchain](decisions/ADR-004-python-quality-toolchain.md)
- [Developer Guide](developer-guide.md) — the local workflow the dev image mirrors
- [Testing Strategy](testing-strategy.md) — the suite CI runs inside the image
- [Logging Strategy](logging.md) — why `logs/` is an externalized volume
- [Security Policy](../SECURITY.md)
- [Versioning](versioning.md) — the SemVer scheme reused for image tags
