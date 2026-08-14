# CI/CD

This document describes the project's **Continuous Integration (CI)** pipeline
and the road to **Continuous Delivery (CD)**. The implemented pipeline lives in
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) and runs on
**GitHub Actions**.

> **Scope today: integration, not delivery.** CI *validates* every change — it
> lints, tests, builds the container image to prove it assembles and runs,
> **statically validates the Kubernetes manifests** (syntax, schema, Kustomize
> rendering, and the workload's security/resource contract), and **statically
> validates the Terraform IaC** (format, provider-only init, `validate`, TFLint,
> and a Trivy misconfiguration scan — with **no AWS access**). It deliberately does
> **not** deploy, does **not** push images to any registry, does **not** run the
> workload on a cluster, and does **not** run `terraform plan`/`apply` or provision
> any cloud resource. The single job that contacts a Kubernetes API server is
> **opt-in** (manual `workflow_dispatch`) and only does a server-side *dry run*.
> Those delivery/provisioning steps are deferred to the roadmap below. This keeps
> the pipeline safe to run on untrusted pull requests: it has read-only
> permissions and **no publish or cloud credentials**.

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

The workflow has five jobs. `quality` runs first; `docker` runs only if it passes
(`needs: quality`), so image-build minutes are never spent on a change that
already fails lint or tests. `k8s-validate` and `terraform-validate` run **in
parallel** (neither needs the Python package or the image), giving fast, static
feedback on the Kubernetes manifests and the Terraform IaC respectively.
`k8s-cluster-dry-run` is **opt-in** — it runs only on a manual `workflow_dispatch`.

### Job 1 — `quality` (Lint & Test)

| # | Stage | Command | Purpose |
|---|-------|---------|---------|
| 1 | Checkout | `actions/checkout@v4` | Fetch the repository at the triggering commit. |
| 2 | Setup Python | `actions/setup-python@v5` (3.12) | Provision Python 3.12 — matches `pyproject.toml` (`target-version = py312`, mypy `python_version`). Caches pip keyed on the requirements files. |
| 3 | Install dependencies | `pip install -r requirements-dev.txt` | `requirements-dev.txt` begins with `-r requirements.txt`, so one install pulls both the runtime stack and the toolchain (Ruff, mypy, pytest) — the type-check step below reuses this install, adding none of its own. |
| 4 | Ruff | `ruff check --output-format=github .` then `ruff format --check .` | Lint with inline PR annotations, then verify formatting **without** rewriting files (CI reports drift; it never mutates the tree). |
| 5 | **mypy** | `python -m mypy` | Static type checking against the single `[tool.mypy]` config in `pyproject.toml` (strict: `disallow_untyped_defs`, `warn_return_any`, `warn_unused_ignores`, …; `files = ["src"]`) — the **same** command `make typecheck` and the pre-commit hook run. Makes the type contract a **binding** server-side gate: a non-zero exit fails the job, so a type regression can no longer reach `main` on green pre-commit alone. Reuses the Stage 3 install (mypy ships in `requirements-dev.txt`), so it adds no dependency install. |
| 6 | Pytest | `pytest` | Runs the suite using the `pyproject.toml` config (`pythonpath=src`, `testpaths=tests`, strict markers) — identical to a local `make test`. Includes the `contract`-marked tests (`tests/contract/`) that statically enforce the [pipeline contract](pipeline-contract.md): `dvc.yaml`↔`params.yaml` parameter consistency, single-owner artifacts, and the declared lineage `raw → preprocess → processed → train → model → evaluate → metrics`. They parse files only — no data, no network, no credentials. |
| 7 | **DVC pipeline integrity** | `dvc dag` then `dvc status` (`DVC_NO_ANALYTICS=true`) | Validates the pipeline **definition** without running it: no stage executes, no dataset is pulled, and the DagsHub remote is never contacted. `dvc dag` proves DVC parses every stage and builds an **acyclic** graph (a malformed `dvc.yaml` or a cycle fails here); local `dvc status` (never `--cloud`) proves the stage/`.dvc` definitions parse coherently. `dvc repro --dry` is intentionally **not** used on the production pipeline — it accesses the remote-only raw dataset and would need credentials or fail nondeterministically; the guarantees it would give are enforced offline by the contract tests in Stage 6 and by the fixture reproduction in Stage 8. |
| 8 | **Fixture pipeline reproduction** | `dvc repro tests/fixtures/pipeline/dvc.yaml` then `dvc status` + forced re-run (`DVC_NO_ANALYTICS=true`) | The one step that actually **runs** the pipeline. The production pipeline cannot run in CI (its raw data is remote-only and its stages log to networked MLflow), so a self-contained **fixture** pipeline reproduces the *same four stages* and the *same `src/` code* against a small committed fixture dataset — with **no remote, no MLflow, no credentials** (the fixture wrapper stubs the tracking boundary). Proves `declared pipeline + params + inputs + code = reproducible execution`: `dvc repro` runs all four stages, `dvc status` must report "up to date" against the committed `dvc.lock`, and a forced re-run must produce **byte-identical** model/metrics (determinism). See [pipeline-contract §7](pipeline-contract.md#7-reproducibility-expectations) and [ADR-008](decisions/ADR-008-fixture-reproducibility.md). |

### Job 2 — `docker` (Docker Build & Validate)

| # | Stage | Purpose |
|---|-------|---------|
| 1 | Checkout | Fresh runner needs its own checkout. |
| — | Set up Buildx | Enables BuildKit (Dockerfile cache mounts) and the GitHub Actions layer cache. |
| 9 | **Docker Build** | Builds the `runtime` (production) target with `docker/build-push-action@v6`, **`push: false`** and **`load: true`** (image loaded locally for validation, never published). Stamps `VCS_REF`/`BUILD_VERSION` build args; caches layers via `type=gha`. |
| 10 | **Build validation** | Runs the freshly built image and asserts its contract: it runs as the **non-root** user (UID `10001`), the core stack (`sklearn`, `mlflow`, `dvc`, `pandas`) imports cleanly, and the `dvc` entrypoint CLI is present. A build that produces an unusable image fails here. |

> Only the **runtime** target is built in CI (it is the shippable artifact and its
> Dockerfile smoke-test already exercises the environment at build time). The
> `development` image is a local convenience and is validated implicitly by the
> `quality` job running the same tools.

### Job 3 — `k8s-validate` (Kubernetes Manifest Validation, static)

Runs on every push/PR, in parallel with `quality`. **Static** validation only — no
cluster is contacted and the workload is never run. Deterministic: tool **and**
schema versions are pinned (job `env`) and the downloaded binaries are
checksum-verified. Design of record:
[ADR-012](decisions/ADR-012-kubernetes-manifest-validation.md).

| # | Stage | Tool (pinned) | Purpose |
|---|-------|---------------|---------|
| 1 | Checkout | `actions/checkout@v4` | Fetch the repo. |
| 2 | Setup Python + PyYAML | `setup-python@v5` (3.12) + `pyyaml==6.0.2` | The project validator's only dependency — no ML stack, so the job stays lean. |
| 3 | Install tools | `kustomize` 5.4.3, `kubeconform` 0.6.7 | Single static binaries, each verified against its release checksums before use. |
| 4 | **Render + schema** | `kustomize build` → `kubeconform -strict` (k8s 1.31.0) | Renders `base/` **and** `overlays/local/` (proves Kustomize builds + YAML parses), then validates every object against the pinned upstream Kubernetes OpenAPI schema and **rejects unknown fields**. |
| 5 | **Security + required fields** | `python k8s/validate.py` | Asserts the PR 1–5 workload contract that a schema check can't express, one PASS/FAIL line per check (below). |

`k8s/validate.py` checks: `runAsNonRoot` + non-root `runAsUser`;
`allowPrivilegeEscalation: false`; `seccompProfile: RuntimeDefault`;
`capabilities: drop [ALL]`; an explicit **non-default** ServiceAccount that exists
in the render; `automountServiceAccountToken: false` (pod + SA); CPU/memory
**requests and limits**; a Job `restartPolicy` of `Never`/`OnFailure`; an
**explicit, pinned** image (no `:latest`); every namespaced object pinned to
`mlops`; and **secret hygiene** (no rendered `Secret`, no inline credential values,
no secret fingerprints anywhere in `k8s/`, template holds only placeholders).

> **Static, not deployment.** This job proves the manifests are well-formed,
> schema-valid, and hardened — **not** that the workload deploys or runs. Its one
> network dependency is `kubeconform` fetching the pinned schema; everything else is
> offline.

### Job 4 — `terraform-validate` (Terraform Validation, static)

Runs on every push/PR, in parallel with `quality` and `k8s-validate`. **Static**
validation of the Terraform IaC under [`terraform/`](../terraform/) — every step
reads only the source and **never contacts AWS**. Tool versions are pinned (job
`env`) so a green run today is a green run tomorrow. Job permissions are
`contents: read`; the job holds **no AWS identity**. Design of record:
[ADR-019](decisions/ADR-019-terraform-ci-validation.md).

| # | Stage | Tool (pinned) | Purpose |
|---|-------|---------------|---------|
| 1 | Checkout | `actions/checkout@v4` | Fetch the repo. |
| 2 | Setup Terraform | `hashicorp/setup-terraform@v3` (TF 1.9.8, `terraform_wrapper: false`) | Pinned CLI; wrapper off so raw exit codes reach the shell. |
| 3 | **fmt** | `terraform fmt -check -recursive` | Canonical formatting is enforced (checked, never rewritten). Drift fails with an actionable message. |
| 4 | **init** | `terraform init -backend=false` | Installs the pinned provider from the committed `.terraform.lock.hcl` — **no backend, no state, no AWS credentials**. Prerequisite for `validate`. |
| 5 | **validate** | `terraform validate` | Syntax, types, references, and provider-schema conformance. **No AWS API calls.** The primary IaC correctness gate. |
| 6 | **TFLint** | `terraform-linters/setup-tflint@v4` (0.54.0) + `tflint` | Language best-practices preset **+ AWS ruleset** (config in [`terraform/.tflint.hcl`](../terraform/.tflint.hcl)). Static lint; contacts no cloud. |
| 7 | **Trivy IaC scan** | `aquasecurity/trivy-action@v0.28.0` (trivy 0.56.1) | `trivy config` misconfiguration scan of `terraform/`. **Fails on CRITICAL/HIGH.** Reads only the source — no AWS access. |

> **Why no `terraform plan`.** A real `plan` reads data sources
> (`aws_caller_identity`, `aws_region`, `aws_availability_zones`), so it needs
> **live AWS credentials**. CI holds none by design, and adding long-lived AWS
> keys to Actions to make `plan` run is the exact security regression this project
> refuses. The `fmt` → `init` → `validate` → lint → scan chain catches formatting,
> syntax, type, reference, and misconfiguration errors **without** any cloud
> access; the un-run part (does this configuration *apply* cleanly against a real
> account) is performed deliberately and out-of-band by an operator against their
> **own** account — see [terraform/README.md § Planning](../terraform/README.md)
> and the boundary below.

> **Trivy suppressions are a triage record, not a mute.** The handful of
> intentional, ADR-ratified exposures for the short-lived validation cluster (the
> open default API CIDR, no KMS envelope encryption) are suppressed **with written
> justification** in [`terraform/.trivyignore`](../terraform/.trivyignore); any
> *new* CRITICAL/HIGH the scanner finds is a real, blocking regression.

### Job 5 — `k8s-cluster-dry-run` (Cluster Admission, opt-in)

The **only** job that talks to a Kubernetes API server, and it is **not** on the
per-PR path — it runs only on a manual `workflow_dispatch` (`if:
github.event_name == 'workflow_dispatch'`), so a real cluster's bootstrap cost and
flake surface never burden ordinary changes. It stands up an **ephemeral kind
cluster** (`helm/kind-action`, k8s v1.31.0) and does a **server-side dry run**:

```bash
kubectl create namespace mlops --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -k k8s/overlays/local --dry-run=server   # admits; persists/ runs nothing
```

Every object passes through the API server's validation, defaulting, and admission
(including Pod Security) — but nothing is persisted and the Job never executes. This
validates **admissibility**, not deployment: a green dry-run does **not** mean the
pipeline completes in-cluster (it still needs an SCM in the image + mounted data —
see [k8s/README.md](../k8s/README.md)). The cluster is torn down automatically.

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
  `Lint & Test`, `Docker Build & Validate`, `K8s Manifest Validation (static)`,
  and `Terraform Validation (static, no AWS)` checks to pass before merge (see
  roadmap v3). CI produces the signal; branch protection makes it binding.
- **Flakes.** Re-run via `workflow_dispatch` or the Actions "Re-run" button;
  concurrency cancellation ensures only the latest attempt matters.

---

## Local validation

CI runs nothing you can't run locally first — reproduce each gate before pushing.

**On the host** (mirrors the `quality` job):

```bash
make install-dev     # one-time: install deps + enable git hooks
make check           # lint + format-check + typecheck + tests (CI's lint/type/test gates)
# or individually:
make lint            # ruff check .
make format-check    # ruff format --check .
make test            # pytest
pytest -m contract   # just the pipeline-definition contract checks (offline)
dvc dag              # the graph DVC builds from dvc.yaml (offline, no remote)
dvc status           # local workspace status (never --cloud)

# Reproduce the self-contained fixture pipeline end to end (offline: no remote,
# no MLflow, no credentials). Proves declared pipeline + params + inputs + code
# = reproducible execution; re-running is a no-op (deterministic).
dvc repro tests/fixtures/pipeline/dvc.yaml
dvc status tests/fixtures/pipeline/dvc.yaml     # -> "up to date"
```

`make check` runs the same lint + format-check + **mypy** + tests the `quality`
job runs (in the same order), so passing it locally guarantees the CI lint/test
gate passes.

**The container build** (mirrors the `docker` job):

```bash
docker build --target runtime -t ml-pipeline:ci .
docker run --rm ml-pipeline:ci id -u                                      # -> 10001
docker run --rm ml-pipeline:ci python -c "import sklearn, mlflow, dvc, pandas; print('ok')"
docker run --rm ml-pipeline:ci dvc --version
```

**Kubernetes manifest validation** (mirrors the `k8s-validate` job):

```bash
# Security + required-field checks (uses your local kustomize/kubectl):
python k8s/validate.py

# Schema validation (install kubeconform once; pin the k8s version to match CI):
kustomize build k8s/base            | kubeconform -strict -summary -kubernetes-version 1.31.0 -schema-location default -
kustomize build k8s/overlays/local  | kubeconform -strict -summary -kubernetes-version 1.31.0 -schema-location default -

# Optional cluster admission dry-run against any local cluster (kind/minikube/Docker Desktop):
kubectl create namespace mlops --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -k k8s/overlays/local --dry-run=server     # admits; runs nothing
```

**Terraform IaC validation** (mirrors the `terraform-validate` job — all offline,
**no AWS credentials**):

```bash
cd terraform
terraform fmt -check -recursive       # canonical formatting (checked, not rewritten)
terraform init -backend=false         # providers only — no backend, no state, no AWS
terraform validate                    # syntax, types, references, provider schema

# Static lint + IaC security scan (install once; pinned to the CI versions):
tflint --init && tflint               # language preset + AWS ruleset (.tflint.hcl)
trivy config .                        # misconfiguration scan; CI fails on CRITICAL/HIGH
```

`terraform plan` is intentionally **not** part of this offline gate — it reads AWS
data sources and needs live credentials. Run it yourself only after
`aws sts get-caller-identity` confirms you are pointed at **your own** account (see
[terraform/README.md § Planning](../terraform/README.md)); CI never runs it.

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
3. **Automated pipeline validation.** *Delivered.* CI validates the pipeline
   **definition** offline — `dvc dag` + local `dvc status` plus the `contract`
   tests (Job 1, stages 6–7) — **and** now runs a real **execution** check: a
   scoped `dvc repro` against a small committed **fixture dataset** with a
   committed `dvc.lock` (Job 1, stage 8), so `dvc status` is a true "up to date"
   drift gate and a forced re-run proves determinism. The **production** raw
   dataset remains remote-only, so the production run itself is still not executed
   in CI — a documented level-4 limitation
   ([pipeline-contract §7](pipeline-contract.md#7-reproducibility-expectations),
   [ADR-008](decisions/ADR-008-fixture-reproducibility.md)).
4. **Pin for reproducibility (v3).** Pin runtime dependencies and the base image
   by digest (per [ADR-005](decisions/ADR-005-containerization-strategy.md)) so
   CI builds are byte-for-byte repeatable.
5. **Deploy to Kubernetes (v4–v5).** Roll the published image out to a cluster
   (staging → production) with environment-scoped config and secrets. This is the
   first true **CD** step and is explicitly **not** implemented here. *Manifest
   **validation** is already in place* (Job 3 `k8s-validate` — static syntax,
   schema, Kustomize, and security/resource checks — plus the opt-in Job 4
   admission dry-run; [ADR-012](decisions/ADR-012-kubernetes-manifest-validation.md)),
   but validating the manifests is not deploying them.
6. **Progressive delivery & rollback (v5+).** Canary/blue-green strategies,
   automated rollback on failed health checks, and a serving component with real
   liveness/readiness probes.

Each CD stage will be ratified as its own ADR (the roadmap flags the CI provider
choice and the orchestration/secret-handling approach as open decisions).

---

## See also

- [.github/workflows/ci.yml](../.github/workflows/ci.yml) — the pipeline itself
- [k8s/README.md](../k8s/README.md) — the manifests and how to validate/run them locally
- [ADR-012](decisions/ADR-012-kubernetes-manifest-validation.md) — Kubernetes manifest validation decision record
- [ADR-019](decisions/ADR-019-terraform-ci-validation.md) — Terraform CI validation decision record
- [terraform/README.md](../terraform/README.md) — the Terraform IaC and how to validate/plan it locally
- [containerization.md](containerization.md) — image design and the container's role in CI/CD
- [ADR-005](decisions/ADR-005-containerization-strategy.md) — containerization decision record
- [docker-development.md](docker-development.md) — local Docker Compose workflow
- [roadmap.md](roadmap.md) — v3 (CI/CD) and v4–v6 (deployment) milestones
