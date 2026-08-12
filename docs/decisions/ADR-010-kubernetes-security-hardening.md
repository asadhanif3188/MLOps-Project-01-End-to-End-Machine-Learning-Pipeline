# ADR-010: Kubernetes Workload Security Hardening (Pod/Container `securityContext`)

- **Status:** Accepted
- **Date:** 2026-08-12
- **Deciders:** Asad Hanif
- **Related:** [Kubernetes Architecture](../kubernetes-architecture.md) §6,
  [`k8s/base/job.yaml`](../../k8s/base/job.yaml),
  [ADR-009 (Kubernetes Workload Model)](ADR-009-kubernetes-workload-model.md),
  [ADR-005 (Containerization Strategy)](ADR-005-containerization-strategy.md) §8–9,
  [Roadmap](../roadmap.md) (v4 Kubernetes)

> **Scope note.** [ADR-009](ADR-009-kubernetes-workload-model.md) ratified the
> *workload model* and explicitly **reserved the Kubernetes security baseline for
> this record**. ADR-010 covers exactly that: the pod/container `securityContext`
> applied to the pipeline `Job` in Sprint 5 PR 4 — which controls are enforced,
> **why one control (read-only root filesystem) is deliberately deferred**, and
> what is *not* claimed. Resource requests/limits are a separate concern (PR 5).

## Context

The pipeline runs as a `batch/v1` `Job` ([ADR-009](ADR-009-kubernetes-workload-model.md))
from a non-root image ([ADR-005](ADR-005-containerization-strategy.md) §9: a
dedicated UID/GID `10001`, no baked-in secrets/data). The image being non-root is
necessary but **not sufficient**: nothing at the platform layer *enforces* that
posture, and a container can still request extra Linux capabilities, allow
privilege escalation, or run seccomp-unconfined even from a well-built image. The
[Kubernetes restricted Pod Security Standard](https://kubernetes.io/docs/concepts/security/pod-security-standards/)
codifies the enforced baseline a batch workload of this kind should meet.

The design principle (mirrored throughout this repo) is to introduce each control
against the **real image** and validate it, rather than paste a generic
`securityContext` block. That matters here because the workload's runtime — DVC
driving `dvc repro` — has a specific, load-bearing interaction with one control
(the read-only root filesystem) that a copy-pasted block would get wrong.

Two image-specific facts drive the decisions below:

1. **The image's `USER` is a *name* (`appuser`), not a numeric UID.** The kubelet
   cannot resolve a username to a UID at admission time, so `runAsNonRoot: true`
   *alone* would reject the pod with `CreateContainerConfigError` ("container has
   runAsNonRoot and image has non-numeric user"). An explicit numeric
   `runAsUser`/`runAsGroup` is therefore **required**, not decorative.
2. **`dvc repro` mutates DVC state in-tree at the `/app` repo root.** Empirically,
   under a read-only root filesystem its *first* action fails with
   `[Errno 30] Read-only file system: '/app/.dvc/tmp'`; it further writes
   `/app/.dvc/cache`, updates `/app/dvc.lock`, and needs a writable `/app/.git`
   for the SCM DVC depends on — all at the repo root, interleaved with the
   baked-in, read-only application code and `.dvc/config`.

## Decision

Apply a `securityContext` to the `Job`, split across the two levels Kubernetes
defines, and enable **every restricted-baseline control that is compatible with
the real workload**, deferring only the one that is not.

**Pod-level** (`spec.template.spec.securityContext` — applies to all containers):

| Field | Value | Rationale |
|-------|-------|-----------|
| `runAsNonRoot` | `true` | Refuse to start as uid 0 — platform-enforced, not image-trusted. |
| `runAsUser` | `10001` | Matches the Dockerfile's `useradd --uid 10001`. **Required** because the image's `USER` is a name (see Context #1). |
| `runAsGroup` | `10001` | Matches `groupadd --gid 10001`. |
| `seccompProfile.type` | `RuntimeDefault` | Apply the runtime's default syscall filter instead of running unconfined. Required by restricted PSS. |

**Container-level** (`containers[].securityContext` — these fields *cannot* live at
the pod level):

| Field | Value | Rationale |
|-------|-------|-----------|
| `allowPrivilegeEscalation` | `false` | No setuid/setgid escalation. Verified: `/proc/self/status` reports `NoNewPrivs: 1`. |
| `capabilities.drop` | `[ALL]` | The pipeline runs Python + `dvc` over HTTPS; it needs **no** Linux capabilities, so drop the entire default set (none are added back). |
| `readOnlyRootFilesystem` | `false` | **Deliberate and validated** — see below. |

### Read-only root filesystem — evaluated, and deferred with evidence

`readOnlyRootFilesystem: true` is **not** enabled, and this is a decision, not an
omission. `dvc repro` writes to the `/app` **repo root** it shares with the
baked-in read-only source: `/app/.dvc/tmp` and `/app/.dvc/cache` (DVC state),
`/app/dvc.lock` (rewritten on every repro), and `/app/.git` (the SCM DVC
requires). Those paths cannot be carved out with `emptyDir` mounts without either
**shadowing the image's own files** (`.dvc/config`, the code) or **mounting a
writable volume over the entire code root** — which defeats the control it is
meant to provide. Enabling it now would make the container fail *earlier* than the
existing SCM blocker (at the `/app/.dvc/tmp` write), i.e. **weaken a working
workload to satisfy a checkbox** — explicitly rejected.

The correct home for this control is the *same* future work that makes the
pipeline green in-cluster (ADR-009 §Consequences): relocating DVC's cache/tmp/lock
and the SCM onto **declared writable volumes** outside the read-only code tree,
then flipping `readOnlyRootFilesystem: true` and validating a real run against it.
Until that exists, the honest state is: control evaluated, incompatibility proven,
deferred — recorded here rather than silently skipped.

### Not added (least privilege by omission)

- **No `Role`/`RoleBinding`/`ClusterRole`.** The workload makes no Kubernetes API
  calls; its ServiceAccount token is not even mounted
  (`automountServiceAccountToken: false`, [ADR-009](ADR-009-kubernetes-workload-model.md)
  identity work / PR 3). Granting unused RBAC would violate least privilege.
- **No `NetworkPolicy`.** Egress restriction (allow only DagsHub/MLflow) is a
  meaningful future control but requires a CNI that enforces policy and a separate
  validation; it is out of scope for a `securityContext` PR and noted as a
  follow-up, not claimed.
- **No `fsGroup`.** No shared writable volume needs group-ownership fix-ups today
  (the writable paths are the container's own root FS); revisit when PVCs/`emptyDir`
  volumes are introduced with the read-only-root work.

## Alternatives Considered

1. **Trust the non-root image; add no `securityContext`.**
   - *Rejected* — the image being non-root is unenforced at the platform layer and
     says nothing about capabilities, privilege escalation, or seccomp. Defense in
     depth requires the platform to *enforce*, not assume.
2. **Enable `readOnlyRootFilesystem: true` now with `emptyDir` mounts for the
   writable paths.**
   - *Rejected* — the writable paths are at the `/app` repo root, interleaved with
     read-only baked-in source; covering them means shadowing image files or
     making the code tree writable. It would introduce a *new*, earlier failure
     than the current blocker. Deferred to the read-only-root work above, with
     empirical evidence recorded.
3. **Claim restricted Pod Security Standard compliance.**
   - *Not claimed* — the manifest carries the fields the restricted profile
     requires and was admitted by a live cluster, but no Pod Security **admission**
     controller (a `pod-security.kubernetes.io/enforce: restricted` namespace
     label) or policy engine has *validated* it, and one control (read-only root)
     is not met. We state the concrete controls, not a compliance badge.
4. **Add RBAC and a `NetworkPolicy` in this PR for "completeness".**
   - *Rejected* — scope creep and, for RBAC, an outright least-privilege
     violation (the workload needs no API access). `NetworkPolicy` is a legitimate
     follow-up tracked separately.

## Consequences

**Positive**

- The workload runs under a platform-**enforced** restricted posture: non-root
  (explicit uid/gid), no privilege escalation, zero Linux capabilities, and the
  default seccomp filter — verified enforced by a live cluster (the applied pod's
  `spec.securityContext`/container `securityContext` report exactly these values).
- The hardening is **behaviour-neutral**: with it applied, `dvc repro` reaches the
  *same* pre-existing SCM blocker (`/app is not a git repository`) it hits without
  it — no new failure mode was introduced (proven both in `docker run` and on the
  cluster). The test suite is unaffected (100 passed, 1 pre-existing skip).
- The security fields are at the **correct scope** (pod vs container), asserted by
  a rendered-manifest check, so the manifest is valid and legible.

**Trade-offs and follow-ups**

- **Read-only root filesystem is deferred** (with recorded evidence), tied to the
  future DVC-state-relocation / green-run work — not enabled blindly.
- **Restricted PSS is not formally validated** by an admission controller; adding
  the namespace `enforce: restricted` label (and satisfying read-only root) is the
  path to an actual compliance claim.
- **No egress `NetworkPolicy`** yet; a "DagsHub/MLflow-only" egress policy is a
  sensible next hardening step once a policy-enforcing CNI is assumed.

## What This Decision Does *Not* Imply

- It does **not** claim restricted Pod Security Standard *compliance* — it applies
  the individual controls that are compatible and names the one that is not.
- It does **not** claim a green in-cluster pipeline run; the hardening is validated
  up to the same pre-existing SCM blocker, not beyond it.
- It does **not** add network isolation or RBAC; those are deliberately out of
  scope (RBAC would be an anti-pattern here; `NetworkPolicy` is a tracked
  follow-up).
