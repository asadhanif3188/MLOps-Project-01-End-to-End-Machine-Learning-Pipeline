# syntax=docker/dockerfile:1
#
# Production-grade container image for the End-to-End ML Pipeline.
#
# Design of record: docs/containerization.md and
# docs/decisions/ADR-005-containerization-strategy.md.
#
# Multi-stage layout with three named targets from a single source of truth:
#   * builder     — compiles/installs dependencies into an isolated venv.
#   * development — builder + the quality toolchain, for the inner loop.
#   * runtime     — lean, non-root production image (the DEFAULT / last stage).
#
# `runtime` is the LAST stage on purpose: a bare `docker build` (no --target)
# produces the production image. Build the dev image explicitly with
# `--target development`. See docs/containerization.md § "Build & Run".

# Pinned base image. Bookworm is pinned by codename so a moving `latest` cannot
# change the OS underneath a build. Follow-up (ADR-005): pin by digest
# (python:3.12-slim-bookworm@sha256:...) for byte-for-byte reproducible rebuilds.
ARG PYTHON_IMAGE=python:3.12-slim-bookworm


# ---------------------------------------------------------------------------
# Stage 1 — builder: install dependencies into a self-contained virtualenv.
# Compilers and headers live here and are discarded; they never reach runtime.
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS builder

# Deterministic, quiet pip; no interactive prompts; no root warning noise.
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Build toolchain for any dependency that ships only an sdist. Confined to this
# stage, so it adds nothing to the shipped image's size or attack surface.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create the virtualenv the runtime stage will copy verbatim.
RUN python -m venv "$VIRTUAL_ENV"

WORKDIR /app

# Copy ONLY the dependency manifest before the source so this expensive layer is
# cached and reused until requirements.txt itself changes (see ADR-005 § build
# cache). A BuildKit cache mount keeps pip's wheel cache out of the image layer
# while still speeding up repeat builds.
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install -r requirements.txt

# Security remediation for the container image scan (ADR-035). Raise the transitive
# packages the scan flagged as FIXABLE HIGH to their patched versions. TARGETED
# security-floor bumps, not a broad upgrade: the ML stack pinned by requirements.txt
# is untouched, and the runtime import smoke-test in the final stage still gates the
# build, so a bump that broke resolution or imports would fail here. Re-triage on each
# rebuild — drop a floor once the naturally-resolved version already exceeds it.
#   * cryptography >= 50.0.0  — CVE-2026-69247. A REAL installed dependency; this bump
#                               CLEARS the finding (verified: gone from the scan).
#   * msgpack      >= 1.2.1   — GHSA-6v7p-g79w-8964. Patches the venv's REAL msgpack.
#   * setuptools   >= 78.1.1  — CVE-2025-47273. Patches the venv's REAL setuptools.
# NOTE: the scanner ALSO reports msgpack 1.1.2 / setuptools 70.3.0 from `pip`'s own
# VENDORED copies (pip/_vendor/…), which these bumps do NOT touch and which no
# `pip install -U` can fix. pip stays because the mlflow-server image installs on top
# of this one; those two residual, pip-internal findings are handled as documented,
# time-boxed exceptions in .trivyignore.yaml (ADR-035 § Findings/Follow-ups).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade \
        "cryptography>=50.0.0" \
        "msgpack>=1.2.1" \
        "setuptools>=78.1.1"


# ---------------------------------------------------------------------------
# Stage 2 — development: builder + the quality toolchain (Ruff, mypy, pytest,
# pre-commit). Derives from `builder` so requirements.txt (referenced by
# requirements-dev.txt via `-r`) and the venv are already present. Source is
# bind-mounted at run time, so it is not baked in. Built explicitly with
# `--target development`; never the default. Runs as root — a deliberate,
# DEVELOPMENT-ONLY convenience for bind-mount permission ergonomics.
# ---------------------------------------------------------------------------
FROM builder AS development

# git so `dvc repro` and pre-commit work cleanly inside the dev container.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Dev-only tooling. requirements-dev.txt starts with `-r requirements.txt`; those
# runtime deps are already installed in the venv, so this adds just the toolchain.
COPY requirements-dev.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-dev.txt

# Interactive shell by default; developers run `make check` / `dvc repro` / etc.
CMD ["bash"]


# ---------------------------------------------------------------------------
# Stage 3 — runtime: minimal, non-root production image. LAST stage → default.
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

# OCI image metadata for provenance/traceability. Overridable at build time:
#   docker build --build-arg VCS_REF=$(git rev-parse --short HEAD) \
#                --build-arg BUILD_VERSION=1.2.0 ...
ARG VCS_REF=unknown
ARG BUILD_VERSION=0.0.0-dev
LABEL org.opencontainers.image.title="mlops-pipeline" \
      org.opencontainers.image.description="End-to-End ML Pipeline (DVC + MLflow)" \
      org.opencontainers.image.source="https://github.com/asadhanif3188/mlops-platform-on-eks" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${BUILD_VERSION}"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    LOG_LEVEL=INFO

# DVC drives the pipeline (`dvc repro`) and expects an SCM to be present;
# installing git keeps the default command working cleanly. --no-install-recommends
# and cleaning apt lists keep the footprint minimal.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Dedicated unprivileged user (fixed UID/GID). Least privilege by default and a
# prerequisite for the Kubernetes restricted Pod Security Standard (ADR-005 § 9).
RUN groupadd --gid 10001 appgroup \
    && useradd --uid 10001 --gid appgroup --create-home --shell /usr/sbin/nologin appuser

# Bring in the pre-built virtualenv from the builder stage (no compilers here).
COPY --from=builder --chown=appuser:appgroup /opt/venv /opt/venv

WORKDIR /app

# Writable mount points for externalized state (ADR-005 § volume strategy). These
# are populated at runtime via volumes / `dvc pull`, never baked into the image.
# Created up front and owned by the runtime user so a read-only root filesystem
# plus mounted volumes works without permission surprises.
RUN mkdir -p /app/data /app/models /app/logs \
    && chown -R appuser:appgroup /app

# Application code and pipeline definition, copied last (changes most often, so
# it invalidates the fewest cached layers). Data and models are intentionally
# excluded (.dockerignore) and supplied as volumes at run time.
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup dvc.yaml params.yaml ./
COPY --chown=appuser:appgroup .dvc/config ./.dvc/config

# Build-time smoke test: fail the build if the shipped environment can't import
# the core stack. This is the batch-image equivalent of a healthcheck (see below).
RUN python -c "import sklearn, mlflow, dvc, pandas; print('runtime imports OK')"

# Drop to the unprivileged user for everything that follows.
USER appuser

# No HEALTHCHECK by design: this image is a run-to-completion BATCH job, not a
# long-running service, so there is no live endpoint or process to probe — a
# HEALTHCHECK would report UNHEALTHY the moment the pipeline finished.
# Liveness/readiness belong to a serving component (Roadmap v6), which will add
# one then. The build-time import check above validates the environment instead.

# Default command runs the full DVC pipeline. Override to run a single stage,
# e.g. `docker run --rm ml-pipeline:local python src/preprocess.py`. Requires
# mounted data/models and a reachable MLflow tracking server (MLFLOW_TRACKING_URI;
# no credentials — ADR-026) at run time — see docs/containerization.md § "Build & Run".
CMD ["dvc", "repro"]
