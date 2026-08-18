# syntax=docker/dockerfile:1
#
# MLflow Tracking Server image (Sprint 7, PR 6 — in-cluster MLflow platform).
#
# Built FROM the project's own pipeline image (the default CMD of the root
# Dockerfile's `runtime` target). That base already ships, pinned and
# import-checked, the entire stack an MLflow server needs EXCEPT one thing:
#   * mlflow (the tracking server + its gunicorn/flask HTTP stack) — present;
#   * boto3 / botocore / s3fs (the S3 client) — present (via mlflow + dvc-s3);
#   * sqlalchemy — present (via mlflow);
#   * psycopg2   — the ONLY missing piece: the PostgreSQL driver for the
#                  SQLAlchemy backend store. Added here.
#
# Reusing the pipeline base has three benefits over a from-scratch server image:
#   1. Client/server MLflow PARITY — the tracking server runs the EXACT mlflow
#      version the pipeline (client) was built with, eliminating a whole class of
#      client/server API-skew bugs. A separately-pinned server could silently drift.
#   2. It inherits the base's hardening verbatim: the non-root `appuser` (uid
#      10001), the vetted, codename-pinned OS layer, and the same provenance.
#   3. It is minimal and fast to build — a single extra wheel, not the full data
#      stack again.
#
# Why NOT `pip install` at container start: installing in the container command
# would make every server start depend on a reachable package index (a
# development-only hack that fails offline and is non-reproducible). The one added
# dependency is baked and import-checked at BUILD time here, exactly like the base
# image (ADR-005, ADR-026).
#
# Build (local Docker Desktop):
#   docker build -t ml-pipeline:local .                                 # base first
#   docker build -f docker/mlflow/Dockerfile -t mlflow-server:local .   # then this
# On AWS, BASE_IMAGE is the ECR pipeline image; the server image is pushed to ECR
# alongside it. Design of record: docs/decisions/ADR-026-in-cluster-mlflow-platform.md.

# The pipeline image to layer on. Overridable so the AWS build points at the ECR
# pipeline image (e.g. <account>.dkr.ecr.<region>.amazonaws.com/mlops-pipeline:<ver>).
ARG BASE_IMAGE=ml-pipeline:local

FROM ${BASE_IMAGE} AS runtime

# The PostgreSQL driver for the SQLAlchemy backend store. psycopg2-binary ships as
# a self-contained manylinux wheel (its own libpq), so no apt/build tools are
# needed and the image stays lean and non-root. Pinned; bump deliberately.
ARG PSYCOPG2_VERSION=2.9.10

# OCI provenance for the server image, layered over the base's pipeline labels.
ARG VCS_REF=unknown
ARG BUILD_VERSION=0.0.0-dev
LABEL org.opencontainers.image.title="mlflow-server" \
      org.opencontainers.image.description="In-cluster MLflow Tracking Server (PostgreSQL backend + S3 artifacts), layered on the pipeline image for version parity" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${BUILD_VERSION}"

# HOME is a writable path for the numeric runtime user so boto3/mlflow caches never
# try to write under a non-existent home.
ENV HOME=/tmp

# Install the one missing driver into the base's virtualenv (owned by appuser, so
# no root needed) and fail the build if the server stack cannot import.
RUN pip install --no-cache-dir "psycopg2-binary==${PSYCOPG2_VERSION}" \
    && python -c "import mlflow, psycopg2, boto3, sqlalchemy; print('mlflow-server imports OK')"

# The tracking server listens on 5000 (the Kubernetes Service targets this port).
EXPOSE 5000

# Default command. The Kubernetes Deployment OVERRIDES this with the fully
# parameterised `mlflow server` invocation (backend-store-uri, artifacts
# destination, --serve-artifacts) built from ConfigMap/Secret env, so the runtime
# wiring is legible in the manifest. This default keeps the image runnable and
# documents the entrypoint. (Runs as the base image's non-root appuser.)
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
