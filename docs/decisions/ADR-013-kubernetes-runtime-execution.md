# ADR-013: Kubernetes Runtime Execution — the DVC/dataset/MLflow runtime contract

- **Status:** Accepted
- **Date:** 2026-08-14
- **Deciders:** Asad Hanif
- **Related:** [`k8s/base/job.yaml`](../../k8s/base/job.yaml),
  [`k8s/base/dvc-config.yaml`](../../k8s/base/dvc-config.yaml),
  [`k8s/overlays/local/job-runtime.yaml`](../../k8s/overlays/local/job-runtime.yaml),
  [ADR-003 (Why DVC)](ADR-003-why-dvc.md),
  [ADR-006 (Pipeline Reproducibility)](ADR-006-pipeline-reproducibility.md),
  [ADR-009 (Kubernetes Workload Model)](ADR-009-kubernetes-workload-model.md),
  [ADR-010 (Security Hardening)](ADR-010-kubernetes-security-hardening.md),
  [ADR-011 (Resource & Lifecycle)](ADR-011-kubernetes-resource-lifecycle.md)

> **Scope.** Sprint 5 PRs 1–7 made the pipeline a *runnable, hardened, validated*
> Kubernetes `Job`, but with one honest gap recorded everywhere: the Job started,
> yet `dvc repro` aborted before any stage ran, so the **pipeline never completed
> in-cluster**. This ADR is the design of record for **closing that gap** — the
> minimal runtime contract that makes the complete pipeline (preprocess → split →
> train → evaluate) execute to `exit 0` inside the secured Job — and for the
> decisions and trade-offs that contract entails.

## Context

Three facts about the runtime image (all deliberate, from earlier ADRs) combine to
stop the pipeline from completing in-cluster:

1. **No SCM in the image.** `.dockerignore` excludes `.git` (ADR-005), so the
   runtime container is not a Git repository. `dvc repro` constructs a DVC `Repo`,
   which by default requires an SCM and aborts immediately:
   `ERROR: /app is not a git repository` — *before* evaluating a single stage. This
   is the first and blocking failure.
2. **No dataset in the image.** `data/` is excluded (`.dockerignore`) and the raw
   dataset is itself DVC-tracked (`data/raw/data.csv.dvc`), so the image ships no
   `data/raw/data.csv`. The preprocess stage's input is absent.
3. **MLflow requires an endpoint (and, for the real one, credentials).** `train`
   and `evaluate` log to MLflow via `require_env("MLFLOW_TRACKING_URI")`; the real
   DagsHub endpoint additionally needs `MLFLOW_TRACKING_USERNAME/PASSWORD`.

Each was diagnosed empirically against the real image, reproduced, and confirmed:

```
docker run --rm --user 10001:10001 ml-pipeline:local dvc repro
# -> ERROR: /app is not a git repository   (exit 255)
```

The constraint from the sprint brief: fix this **without** compromising the
existing container/security design — no `.git` in the production image, no dataset
or credentials committed or baked in, no root, no removed security controls, no
scope creep (no Deployment/Service/Ingress/Helm/Argo).

### Why `dvc repro` needs SCM here, and whether it truly requires `.git`

DVC uses Git for *SCM bookkeeping* — chiefly managing `.gitignore` entries for
tracked outputs and change metadata — not for executing stages. Running the stage
commands, checksumming dependencies/outputs, resolving the DAG, and the local cache
all work with no Git. So the pipeline does **not** require `.git`; DVC merely
*defaults* to requiring an SCM. DVC's first-class, supported escape hatch is no-SCM
mode (`dvc init --no-scm`), which sets `core.no_scm = true` and swaps Git for a
`NoSCM` no-op manager.

Crucially, the pipeline's **reproducibility guarantee does not come from
Git-in-the-container.** It comes from `params.yaml` + pinned dependencies + seeded
RNG + the DVC DAG (ADR-006, ADR-008). Git-based *data/versioning* happens in the
dev/CI environment, which is a real Git repository; the in-cluster Job is an
**ephemeral, run-to-completion executor**, where no-SCM loses nothing that matters
for a single run. (See ADR-003 for DVC's role; this ADR narrows it for runtime.)

## Decision

Provide a **minimal runtime contract** as Kubernetes configuration (no application
code and no production-image changes), split by whether each input is
environment-independent (base) or local-validation-specific (overlay).

### 1. DVC no-SCM — environment-independent, in the base

A base `ConfigMap` (`mlops-pipeline-dvc-config`) carries a `config.local` with:

```ini
[core]
    no_scm = true
```

mounted **read-only** at `/app/.dvc/config.local` (subPath, so it lands beside the
image's `/app/.dvc/config` without shadowing it). `config.local` is DVC's
designated local override layer (git-ignored in dev, highest precedence), so:

- injecting it **only inside the container** turns off SCM for the Job alone; the
  committed `.dvc/config` is untouched, so the **dev/CI Git+DVC workflow keeps
  working** (this is why we do **not** edit the committed config);
- DVC only *reads* it; its writable state (`/app/.dvc/tmp`, `/app/.dvc/cache`,
  `/app/dvc.lock`) lives on the container's writable root filesystem.

This is in the **base** because it is environment-independent: no image, in any
environment, ships `.git`, so every Kubernetes run needs it.

### 2. Runtime dataset — local-validation-specific, in the overlay

The local overlay mounts the dataset read-only at `/app/data/raw` from a
`ConfigMap` (`mlops-pipeline-dataset`) the operator creates **out-of-band** from
the local, git-ignored `data/raw/data.csv` — exactly like the credential Secret:

```bash
kubectl create configmap mlops-pipeline-dataset -n mlops \
  --from-file=data.csv=data/raw/data.csv
```

The volume is `optional: true`, so the Job still applies and starts when the
dataset ConfigMap is absent; the pipeline then **fails fast** at preprocess with a
clear `DataError` (missing `data/raw/data.csv`) rather than the pod being stuck
unschedulable — the intended, graceful missing-input failure mode.

> **This is a LOCAL VALIDATION mechanism and is NOT presented as production
> storage.** A ConfigMap caps at 1 MiB and is the wrong carrier for a real dataset.
> It is chosen here because it is the simplest, most portable, non-root-friendly
> way to hand a small dataset to a *local* Job: it works identically on Docker
> Desktop / kind / minikube, needs no node-path coupling (unlike `hostPath`), and
> needs no network or credentials (unlike an init-container `dvc pull`). Production
> would use a PVC / object store / `dvc pull` from the DVC remote — a separate,
> future decision.

### 3. MLflow — local file store in the overlay; DagsHub preserved in the base

The base keeps `MLFLOW_TRACKING_URI` pointed at the production DagsHub endpoint
(via the existing `ConfigMap`), with credentials supplied by the out-of-band
`Secret`. The local overlay **overrides** it with an in-pod file store so a local
run is fully offline — no external MLflow server and no credentials:

```yaml
env:
  - { name: MLFLOW_TRACKING_URI, value: "file:///app/mlruns" }
  - { name: MLFLOW_ALLOW_FILE_STORE, value: "true" }   # newer MLflow gates the file backend
```

`env` entries take precedence over the base ConfigMap's `envFrom` value for the
same key. The store is written under `/app/mlruns` on the writable root FS.

> DagsHub/MLflow connectivity is therefore **configuration-validated** (the
> Secret + envFrom wiring is correct and unchanged) — **not connectivity-tested**
> — in the local run. To exercise the real DagsHub path locally, create the Secret
> out-of-band and drop the `MLFLOW_*` override.

### Writable paths (security)

The workload writes only under `/app` (owned by uid 10001): `/app/.dvc/tmp`,
`/app/.dvc/cache`, `/app/dvc.lock`, `/app/data/processed`, `/app/models`,
`/app/metrics`, `/app/mlruns`, `/app/logs`. `readOnlyRootFilesystem` stays `false`
(as in ADR-010): those DVC paths are interleaved with the read-only baked-in code
at the `/app` repo root and cannot be carved out with `emptyDir` without shadowing
image files. Relocating DVC's cache/tmp/lock onto declared writable volumes and
then flipping `readOnlyRootFilesystem: true` remains the tracked follow-up (ADR-010
§ "Read-only root filesystem"); it is intentionally **not** bundled into this PR to
keep the change minimal and the green run the single, reviewable outcome.

All other controls are unchanged and re-verified on the live pod: `runAsNonRoot`,
`runAsUser/Group 10001`, seccomp `RuntimeDefault`, `allowPrivilegeEscalation:
false`, `capabilities.drop: [ALL]`, `automountServiceAccountToken: false`. The two
added volumes are read-only ConfigMaps needing no privilege.

## Alternatives Considered

1. **`COPY .git` into the production image / run `git init` at build time.**
   *Rejected* — bloats the image with SCM history (or fabricates an empty repo),
   couples the runtime to Git it does not need, and was explicitly out of bounds.
   No-SCM mode expresses the actual intent (an executor, not a versioning host).
2. **Set `core.no_scm = true` in the committed `.dvc/config`.**
   *Rejected* — it would disable SCM in the dev/CI environment too, breaking
   `dvc add`/`dvc status`/`.gitignore` management there. The mounted `config.local`
   scopes no-SCM to the container only.
3. **Bake the dataset into the image / commit it / `configMapGenerator` from a
   committed file.** *Rejected* — violates "no dataset in the image / in git," and
   a `configMapGenerator` over a *git-ignored* file would break CI's offline
   overlay render. Out-of-band creation (like the Secret) keeps the dataset out of
   git and keeps `kustomize build` working with no dataset present.
4. **`hostPath` for the dataset.** *Rejected for the default path* — couples to the
   node filesystem (on Docker Desktop/kind the "host" is the node container, needing
   `extraMounts`), is not portable across cluster types, and the brief warns against
   dressing it up as production storage. The ConfigMap mount is portable and
   node-agnostic.
5. **Init-container `dvc pull` to fetch the dataset.** *Rejected for local
   validation* — needs the DVC remote credentials and network, adding a flake
   surface and an external dependency unrelated to proving the pipeline runs. It is
   the natural *production* mechanism and is noted as future work.
6. **Keep MLflow pointed at DagsHub for the local run.** *Rejected as the default* —
   makes a green local/CI run depend on external infrastructure and secrets. The
   file-store override keeps local validation hermetic and reproducible (consistent
   with how the CI fixture pipeline already stubs MLflow offline, ADR-008); the
   DagsHub path stays intact in the base for real use.
7. **Enable `readOnlyRootFilesystem: true` in this PR.** *Deferred* — see above and
   ADR-010; it requires relocating DVC's in-tree writable state and is a distinct
   change. Bundling it would enlarge and risk the green-run proof.

## Consequences

**Positive**

- **The complete pipeline runs to completion in-cluster.** Verified end to end on
  Docker Desktop Kubernetes v1.34.3 (2026-08-14): `Job Complete`, pod `Succeeded`,
  **exit code 0**, first attempt (`RESTARTS: 0`), all four stages
  (preprocess 768 rows → split 614/154 → train acc 0.7398 → evaluate acc 0.7078).
  This closes the last Sprint 5 proof gap.
- **Security posture preserved.** The live pod still enforces non-root/10001,
  seccomp RuntimeDefault, no privilege escalation, all capabilities dropped, no API
  token; QoS `Burstable`; the two new mounts are read-only ConfigMaps.
- **Dev/CI unaffected.** No-SCM is scoped to the container via `config.local`; the
  committed `.dvc/config` and the Git+DVC dev workflow are untouched. No application
  code changed; the production image is unchanged.
- **Failure mode is graceful and demonstrated.** With the dataset ConfigMap
  removed, the Job fails fast at preprocess (`DataError`), retries per
  `backoffLimit: 2` as fresh pods, and settles into `Failed: BackoffLimitExceeded`
  — non-zero exit → Job failure → backoff, exactly as designed (ADR-011).
- **The contract is statically checkable.** `k8s/validate.py` gained a
  "Runtime execution contract" section (no-SCM ConfigMap + mount, dataset mount +
  backing volume, MLflow endpoint, `dvc` command) that runs in CI on the rendered
  overlay.

**Trade-offs and follow-ups**

- **The dataset ConfigMap is local-only** (≤1 MiB, hand-created). Production dataset
  provisioning (PVC / object store / init-container `dvc pull`) is future work.
- **Local MLflow is a file store**, so DagsHub connectivity is configuration-
  validated, not connectivity-tested, in the local run.
- **`readOnlyRootFilesystem` is still `false`**, now the *only* remaining item of
  the ADR-010 deferral (the green-run half is done); relocating DVC's writable
  state to enable it is the tracked next step.
- **Image availability on Docker Desktop.** Its Kubernetes runs on containerd whose
  image store is separate from `docker build`; a freshly built image must be
  imported into the node's `k8s.io` namespace or pods run a stale image (documented
  in the operations runbook troubleshooting matrix).

## What This Decision Does *Not* Imply

- It does **not** claim production dataset storage, production MLflow, or a
  production deployment — the run is on a **local** cluster with a local dataset and
  a local MLflow file store.
- It does **not** claim restricted Pod Security Standard *compliance* (no admission
  label/policy engine; read-only root still deferred).
- It does **not** change the pipeline's reproducibility guarantee, which remains
  `params.yaml` + pinned deps + seeded RNG + the DVC DAG (ADR-006), independent of
  whether Git is present in the executor.
