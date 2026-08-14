# Sprint 5 — Retrospective (v1.4.0)

- **Date:** 2026-08-14
- **Release:** `v1.4.0` — Kubernetes Platform Engineering (pending; this sprint's
  release step)
- **Scope:** Express the containerized ML pipeline as a Kubernetes-native **batch
  workload** and make the platform-engineering decisions that entails — workload
  model, identity, configuration/secrets, security hardening, measured resources,
  automated manifest validation, operations/proof documentation — and then **run it
  to completion on a real cluster**.
- **Companion:** [Sprint 4 — Retrospective](sprint-04-retrospective.md),
  [`k8s/README.md`](../../k8s/README.md),
  [Kubernetes Architecture](../kubernetes-architecture.md),
  [Kubernetes Operations](../kubernetes-operations.md),
  [Kubernetes Security](../kubernetes-security.md),
  [Sprint 5 Proof-Impact](../proof/sprint-05-proof-impact.md),
  [ADR-009](../decisions/ADR-009-kubernetes-workload-model.md)–[ADR-013](../decisions/ADR-013-kubernetes-runtime-execution.md)

This is a look-back on Sprint 5: what was planned, what shipped, what changed during
implementation, the problems hit, the engineering decisions behind them, and what
was deliberately left for later. It records judgment and rationale — it is not a
validation gate.

---

## 1. Planned

Sprint 5 set out to take the correct, reproducible, CI-validated container from
Sprint 4 and express it as a **Kubernetes-native workload**, one control at a time,
each introduced against the real image and validated rather than pasted from a
template. The plan was a sequence of focused PRs so every decision stayed
reviewable: workload model → runnable workload → config/identity → security →
resources → CI validation → operations/proof — driving toward a defensible claim of
platform-engineering judgment on a local cluster.

Explicit guardrails: model the workload **honestly** (a `Job`, not a `Deployment`
with an invented service); introduce each security/resource control against the
**real image** with evidence; keep credentials and datasets out of git and out of
the image; and **state the local-vs-production boundary out loud** rather than
imply production capability.

## 2. Delivered

Shipped across eight PRs, each branch → PR → merge, driving to the `v1.4.0` release:

| PR | Delivered |
|----|-----------|
| PR 1 | `mlops` namespace + the pipeline as a `batch/v1` **`Job`**; Kustomize base/overlay structure ([ADR-009](../decisions/ADR-009-kubernetes-workload-model.md)) |
| PR 2 | **Runnable** workload — real `ml-pipeline:local` image, real `dvc repro` command, finite-run lifecycle (`restartPolicy: Never`, `backoffLimit: 2`, `activeDeadlineSeconds`); local runbook; first executed local run |
| PR 3 | Externalized config (`ConfigMap`), an out-of-band `Secret` **template**, and a least-privilege `ServiceAccount` with **token automount off** |
| PR 4 | Hardened pod/container `securityContext` — non-root uid/gid `10001`, seccomp `RuntimeDefault`, `allowPrivilegeEscalation: false`, `capabilities.drop: [ALL]` ([ADR-010](../decisions/ADR-010-kubernetes-security-hardening.md)) |
| PR 5 | **Measured** resource requests/limits (`250m/256Mi`–`1/512Mi`), lifecycle & no-probe decision, failure modes ([ADR-011](../decisions/ADR-011-kubernetes-resource-lifecycle.md)) |
| PR 6 | **Automated static CI validation** — `kustomize` render + `kubeconform -strict` schema + `k8s/validate.py` security/required-field checks; opt-in ephemeral-cluster admission dry-run ([ADR-012](../decisions/ADR-012-kubernetes-manifest-validation.md)) |
| PR 7 | **Operations & proof** documentation — deployment guide, operations runbook + troubleshooting matrix, security document, Sprint 5 Proof-Impact Assessment |
| PR 8 | **Kubernetes Runtime Execution** — the complete pipeline runs to **exit 0** in-cluster via a minimal runtime contract (DVC no-SCM, mounted dataset, in-pod MLflow file store); runtime static-validation section; failure test ([ADR-013](../decisions/ADR-013-kubernetes-runtime-execution.md)) |

Concretely, the repository gained a Kubernetes deployment surface (`k8s/` base +
local overlay), an enforced restricted-style security posture, measured resource
governance, a deterministic CI manifest gate, a full operations/security/proof doc
set — and, in PR 8, a **green end-to-end Job run** (preprocess → split → train →
evaluate, exit 0) on Docker Desktop Kubernetes.

## 3. What changed during implementation

- **The green run became its own PR (PR 8), not a footnote.** PRs 1–7 repeatedly
  and honestly recorded that the Job *started* but `dvc repro` aborted at the SCM
  check. Rather than paper over it, the sprint carried it as a named limitation and
  closed it deliberately once the earlier controls were in place — so the fix could
  be reviewed as the single, focused change it is.
- **The fix was configuration, not code.** The runtime contract (no-SCM, dataset,
  MLflow) was expressed entirely in Kubernetes manifests/overlays — no application
  code and no production-image change — keeping the security/container design of
  the earlier PRs intact.
- **Docker Desktop's image store forced a real diagnosis step.** Its Kubernetes runs
  on a containerd store **separate** from `docker build`; a freshly built image was
  invisible to the kubelet, which silently ran a **stale** cached image. This
  surfaced as a baffling DVC error (a scrambled, old `dvc.yaml`) until an
  in-pod-vs-image `md5sum` comparison pinpointed it. The fix (import into the node's
  `k8s.io` namespace) is now documented in the troubleshooting matrix.

## 4. Problems encountered

- **Structural validity ≠ runtime correctness.** Every static gate was green —
  Kustomize rendered, `kubeconform -strict` passed, `k8s/validate.py` passed, the
  cluster *admitted* the manifests — yet the application would not complete. The
  blocking dependency (`dvc repro` needs an SCM) lived **below** the manifest layer,
  invisible to schema/policy validation. Only executing on a real cluster exposed it.
- **A misleading first cluster failure.** PR 8's first in-cluster run failed with a
  DVC *params* error that did not reproduce under `docker run` — because the kubelet
  was running a stale image (above), not because of the manifests. The lesson:
  verify the pod is actually on the intended image before debugging its behaviour.
- **MLflow's file backend is gated.** Newer MLflow refuses the filesystem tracking
  store unless `MLFLOW_ALLOW_FILE_STORE=true` — discovered empirically and made part
  of the local overlay's offline MLflow contract.

## 5. Engineering decisions

- **DVC no-SCM as the honest runtime expression.** `core.no_scm = true` (DVC's
  supported `dvc init --no-scm` mode) states what is true: the container is an
  ephemeral *executor*, not a versioning host. Reproducibility is carried by
  `params.yaml` + pinned deps + seeded RNG + the DVC DAG (ADR-006/008), not by
  Git-in-the-container — so no-SCM loses nothing that matters for a single run.
- **Scope no-SCM to the container, not the repo.** It is injected via a mounted
  `.dvc/config.local` (DVC's local override layer), leaving the committed
  `.dvc/config` and the dev/CI Git+DVC workflow untouched.
- **Out-of-band dataset, like the Secret.** The dataset is mounted from a ConfigMap
  the operator creates out-of-band from a git-ignored file — nothing committed or
  baked in, CI's overlay render stays offline, and a missing dataset fails fast
  gracefully (`optional: true`). Named explicitly as **local validation, not
  production storage**.
- **Offline local MLflow, DagsHub preserved.** The local overlay overrides MLflow to
  an in-pod file store so a local run needs no external server or credentials; the
  base keeps the DagsHub endpoint + Secret for real use. DagsHub is therefore
  *configuration-validated*, not *connectivity-tested*, locally.
- **Prove the failure path, not just the happy path.** A controlled missing-dataset
  run demonstrated fail-fast → back-off → terminal `Failed (BackoffLimitExceeded)`,
  then the working config was restored — the retry/lifecycle semantics from ADR-011
  shown, not just asserted.
- **Honesty over a clean scorecard, throughout.** Every PR stated what was *not*
  yet true (the non-green run, deferred read-only root, no PSS certification, local
  only), which is what keeps the eventual green-run claim credible.

## 6. What went well

- **One-control-per-PR kept judgment reviewable.** Each control landed against the
  real image with its own evidence and ADR, so no decision was a copy-pasted block.
- **The static gate did its job — and its limits were understood.** CI validation
  caught manifest regressions deterministically; the sprint never mistook it for
  proof that the workload *runs*, which is exactly why PR 8 was scoped.
- **Empirical debugging beat speculation.** The stale-image trap was solved by
  comparing an in-pod `md5sum` to the image's, not by guessing — and the finding
  became durable documentation.
- **The green run required no security regression.** exit 0 was reached with every
  control from PRs 3–5 still enforced (re-verified on the live pod).

## 7. What was deliberately deferred

- **Read-only root filesystem.** Now the **only** remaining item of the ADR-010
  deferral: the green run removed the SCM blocker, but DVC still writes cache/tmp/
  lock in-tree at `/app`. Relocating that state onto writable volumes and flipping
  `readOnlyRootFilesystem: true` is the tracked next step (kept out of PR 8 to keep
  the green-run proof minimal).
- **Production dataset & MLflow.** The local ConfigMap dataset (≤1 MiB) and file
  store are local-validation mechanisms; a PVC/object-store/`dvc pull` dataset and a
  real MLflow connectivity test are future work.
- **Restricted PSS certification, `NetworkPolicy`, RBAC.** Controls are applied and
  cluster-admitted, but no admission label/policy engine ratifies the profile;
  egress policy and (deliberately absent) RBAC remain follow-ups.
- **Production cloud / GitOps / HA / serving / observability.** Roadmap v5–v6,
  explicitly out of scope.
- **Supply-chain hardening.** Digest pinning, signing, scanning, image publishing —
  CD, not this integration sprint.

## 8. Lessons learned

- **A Kubernetes workload can be structurally valid while its application runtime
  contract is still incomplete.** Real cluster execution exposed the DVC SCM
  dependency that static validation (schema, Kustomize, security checks, even
  server-side admission) could not detect — because it lived below the manifest
  layer. The runtime contract was then made **explicit** through Kubernetes
  configuration (no-SCM config, mounted dataset, MLflow backend) and **validated
  through successful end-to-end Job execution**. Static validity proves the manifest
  is well-formed and hardened; only running it proves the *application* runs.
- **Diagnose the image, not just the app.** A "manifest" or "code" bug that won't
  reproduce off-cluster is often a stale-image/artifact-provenance problem. Verify
  the running artifact's identity (an in-pod checksum) before debugging its
  behaviour — provenance is part of the runtime contract.
- **Encode the runtime contract where it can be checked.** The no-SCM config, the
  dataset mount, and the MLflow endpoint are now asserted by `k8s/validate.py`, so
  the contract that made the run green cannot silently regress — the same "make the
  contract executable" lesson from Sprint 4, applied to the *runtime* layer.
- **Carry limitations forward honestly, then close them deliberately.** Naming the
  non-green run across PRs 2–7 turned it into a precise, well-understood backlog item
  that PR 8 could close as a single reviewable change — not a scramble at release.

---

## Related documentation

- [`k8s/README.md`](../../k8s/README.md) — manifests + deployment/runtime records
- [Kubernetes Architecture](../kubernetes-architecture.md) ·
  [Operations](../kubernetes-operations.md) · [Security](../kubernetes-security.md)
- [Sprint 5 — Proof Impact](../proof/sprint-05-proof-impact.md)
- [ADR-009](../decisions/ADR-009-kubernetes-workload-model.md) ·
  [ADR-010](../decisions/ADR-010-kubernetes-security-hardening.md) ·
  [ADR-011](../decisions/ADR-011-kubernetes-resource-lifecycle.md) ·
  [ADR-012](../decisions/ADR-012-kubernetes-manifest-validation.md) ·
  [ADR-013](../decisions/ADR-013-kubernetes-runtime-execution.md)
- [Sprint 4 — Retrospective](sprint-04-retrospective.md)
- [Roadmap](../roadmap.md) · [Changelog](../../CHANGELOG.md)
