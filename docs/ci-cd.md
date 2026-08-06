# CI/CD

This document describes the project's **Continuous Integration (CI)** pipeline
and the road to **Continuous Delivery (CD)**. The implemented pipeline lives in
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) and runs on
**GitHub Actions**.

> **Scope today: integration, not delivery.** CI *validates* every change — it
> lints, tests, and builds the container image to prove it assembles and runs. It
> deliberately does **not** deploy, does **not** push images to any registry, and
> does **not** touch Kubernetes. Those are CD concerns, deferred to the roadmap
> below. This keeps the pipeline safe to run on untrusted pull requests: it has
> read-only permissions and no publish credentials.

---

## Triggers

| Event | Why |
|-------|-----|
| `push` to `main` | Guard the integration branch — every landed commit is re-validated. |
| `pull_request` | Validate proposed changes **before** merge; this is what branch protection should require green. |
| `workflow_dispatch` | Manual re-run from the Actions tab (e.g. to retry an infra flake). |

A **concurrency group** keyed on the ref cancels an in-flight run when a newer
commit arrives, so feedback always reflects the latest push and stale runs don't
consume minutes.

---

## Pipeline stages

The workflow has two jobs. `quality` runs first; `docker` runs only if it passes
(`needs: quality`), so image-build minutes are never spent on a change that
already fails lint or tests.

### Job 1 — `quality` (Lint & Test)

| # | Stage | Command | Purpose |
|---|-------|---------|---------|
| 1 | Checkout | `actions/checkout@v4` | Fetch the repository at the triggering commit. |
| 2 | Setup Python | `actions/setup-python@v5` (3.12) | Provision Python 3.12 — matches `pyproject.toml` (`target-version = py312`, mypy `python_version`). Caches pip keyed on the requirements files. |
| 3 | Install dependencies | `pip install -r requirements-dev.txt` | `requirements-dev.txt` begins with `-r requirements.txt`, so one install pulls both the runtime stack and the toolchain (Ruff, pytest). |
| 4 | Ruff | `ruff check --output-format=github .` then `ruff format --check .` | Lint with inline PR annotations, then verify formatting **without** rewriting files (CI reports drift; it never mutates the tree). |
| 5 | Pytest | `pytest` | Runs the suite using the `pyproject.toml` config (`pythonpath=src`, `testpaths=tests`, strict markers) — identical to a local `make test`. Includes the `contract`-marked tests (`tests/contract/`) that statically enforce the [pipeline contract](pipeline-contract.md): `dvc.yaml`↔`params.yaml` parameter consistency, single-owner artifacts, and the declared lineage `raw → preprocess → processed → train → model → evaluate → metrics`. They parse files only — no data, no network, no credentials. |
| 6 | **DVC pipeline integrity** | `dvc dag` then `dvc status` (`DVC_NO_ANALYTICS=true`) | Validates the pipeline **definition** without running it: no stage executes, no dataset is pulled, and the DagsHub remote is never contacted. `dvc dag` proves DVC parses every stage and builds an **acyclic** graph (a malformed `dvc.yaml` or a cycle fails here); local `dvc status` (never `--cloud`) proves the stage/`.dvc` definitions parse coherently. `dvc repro --dry` is intentionally **not** used — it accesses the remote-only raw dataset and would need credentials or fail nondeterministically; the guarantees it would give are enforced offline by the contract tests in Stage 5. |

### Job 2 — `docker` (Docker Build & Validate)

| # | Stage | Purpose |
|---|-------|---------|
| 1 | Checkout | Fresh runner needs its own checkout. |
| — | Set up Buildx | Enables BuildKit (Dockerfile cache mounts) and the GitHub Actions layer cache. |
| 7 | **Docker Build** | Builds the `runtime` (production) target with `docker/build-push-action@v6`, **`push: false`** and **`load: true`** (image loaded locally for validation, never published). Stamps `VCS_REF`/`BUILD_VERSION` build args; caches layers via `type=gha`. |
| 8 | **Build validation** | Runs the freshly built image and asserts its contract: it runs as the **non-root** user (UID `10001`), the core stack (`sklearn`, `mlflow`, `dvc`, `pandas`) imports cleanly, and the `dvc` entrypoint CLI is present. A build that produces an unusable image fails here. |

> Only the **runtime** target is built in CI (it is the shippable artifact and its
> Dockerfile smoke-test already exercises the environment at build time). The
> `development` image is a local convenience and is validated implicitly by the
> `quality` job running the same tools.

---

## Failure strategy

**Fail fast, fail loud, fail safe.**

- **Ordered gates.** `quality` gates `docker`. Cheap, fast checks (lint/test,
  ~a minute) run before the expensive image build, so the common failure is
  reported quickly and no build minutes are wasted on it.
- **Any red step fails the job.** Every step uses its tool's native exit code;
  the validation step runs under `set -euo pipefail` so the first failed
  assertion stops the job immediately with a non-zero status.
- **No mutation on failure (or success).** CI never writes back to the repo. Ruff
  formatting is **checked**, not applied; a drift is surfaced as a failure for the
  author to fix locally with `make format`. This keeps CI a pure, idempotent gate.
- **Fresh, isolated runners.** Each job starts from a clean `ubuntu-latest`
  runner, so a pass depends only on the committed source — not on leftover state.
- **Least privilege.** `permissions: contents: read` and `push: false` mean a
  failing (or malicious) run cannot deploy, publish, or alter anything. Safe to
  run on pull requests from forks.
- **Intended enforcement.** Branch protection on `main` should require the
  `Lint & Test` and `Docker Build & Validate` checks to pass before merge (see
  roadmap v3). CI produces the signal; branch protection makes it binding.
- **Flakes.** Re-run via `workflow_dispatch` or the Actions "Re-run" button;
  concurrency cancellation ensures only the latest attempt matters.

---

## Local validation

CI runs nothing you can't run locally first — reproduce each gate before pushing.

**On the host** (mirrors the `quality` job):

```bash
make install-dev     # one-time: install deps + enable git hooks
make check           # lint + format-check + typecheck + tests (superset of CI)
# or individually:
make lint            # ruff check .
make format-check    # ruff format --check .
make test            # pytest
pytest -m contract   # just the pipeline-definition contract checks (offline)
dvc dag              # the graph DVC builds from dvc.yaml (offline, no remote)
dvc status           # local workspace status (never --cloud)
```

`make check` is a **superset** of the `quality` job — it additionally runs mypy.
Passing it locally guarantees the CI lint/test gate passes.

**The container build** (mirrors the `docker` job):

```bash
docker build --target runtime -t ml-pipeline:ci .
docker run --rm ml-pipeline:ci id -u                                      # -> 10001
docker run --rm ml-pipeline:ci python -c "import sklearn, mlflow, dvc, pandas; print('ok')"
docker run --rm ml-pipeline:ci dvc --version
```

For the full inner-loop workflow, see
[docker-development.md](docker-development.md); pre-commit hooks
(`make install-dev` installs them) catch most lint/format issues before a commit
even reaches CI.

---

## Future CD roadmap

CI stops at "the image builds and runs." **Continuous Delivery** — actually
shipping that image — is intentionally out of scope for now and staged across the
project roadmap ([roadmap.md](roadmap.md)):

1. **Publish the image (v3).** On a tagged release, push the validated image to a
   registry (e.g. GHCR) with immutable, provenance-stamped tags. Requires adding
   `packages: write` permission and registry login — deliberately absent today.
2. **Supply-chain hardening (v3).** Vulnerability scanning (e.g. Trivy/Grype),
   SBOM generation, and image signing (cosign) as release gates.
3. **Automated pipeline validation.** *Partially delivered (Sprint 4).* CI now
   validates the pipeline **definition** offline — `dvc dag` + local `dvc status`
   plus the `contract` tests (see Job 1, stages 5–6). Still future: an
   **execution** check that runs a scoped `dvc repro` against a small committed
   **fixture dataset** (and a committed `dvc.lock`) so `dvc status` becomes a true
   "up to date" drift gate — today the raw dataset is remote-only, so no stage is
   run in CI.
4. **Pin for reproducibility (v3).** Pin runtime dependencies and the base image
   by digest (per [ADR-005](decisions/ADR-005-containerization-strategy.md)) so
   CI builds are byte-for-byte repeatable.
5. **Deploy to Kubernetes (v4–v5).** Roll the published image out to a cluster
   (staging → production) with environment-scoped config and secrets. This is the
   first true **CD** step and is explicitly **not** implemented here.
6. **Progressive delivery & rollback (v5+).** Canary/blue-green strategies,
   automated rollback on failed health checks, and a serving component with real
   liveness/readiness probes.

Each CD stage will be ratified as its own ADR (the roadmap flags the CI provider
choice and the orchestration/secret-handling approach as open decisions).

---

## See also

- [.github/workflows/ci.yml](../.github/workflows/ci.yml) — the pipeline itself
- [containerization.md](containerization.md) — image design and the container's role in CI/CD
- [ADR-005](decisions/ADR-005-containerization-strategy.md) — containerization decision record
- [docker-development.md](docker-development.md) — local Docker Compose workflow
- [roadmap.md](roadmap.md) — v3 (CI/CD) and v4–v6 (deployment) milestones
