# Kubernetes Security

The security posture of the ML pipeline as a Kubernetes **batch `Job`**: the
controls that are enforced, *where* they are enforced (image vs platform), the one
control that is deliberately deferred, the secret-handling model, and — stated
plainly — what is **not** claimed. Every control below is verifiable from the
manifests and was observed enforced on a live local cluster.

> **Framing.** This is a **defense-in-depth, least-privilege** posture for a finite
> batch workload on a **local** cluster. It is not a production security
> certification: **restricted Pod Security Standard (PSS) compliance is not
> claimed**, one control (read-only root filesystem) is deferred with evidence, and
> there is no `NetworkPolicy`, no admission-policy engine, and no managed-cluster
> IAM. Those are future work, not present guarantees. No production Kubernetes
> security expertise is claimed.

Design of record: [ADR-010](decisions/ADR-010-kubernetes-security-hardening.md)
(security hardening) and [ADR-009](decisions/ADR-009-kubernetes-workload-model.md)
(identity). See also [kubernetes-architecture.md §6](kubernetes-architecture.md#6-identity--security-boundary),
[containerization.md](containerization.md) (the image's own hardening), and
[ADR-005](decisions/ADR-005-containerization-strategy.md).

---

## 1. Two layers of enforcement

Security is enforced at **both** the image and the platform layer, so neither is
trusted alone:

| Layer | What it guarantees | Where |
|---|---|---|
| **Image** | A dedicated non-root user (`appuser`, UID/GID `10001`, `nologin` shell); no baked-in data, models, or credentials; build tooling confined to the `builder` stage. | `Dockerfile`, [ADR-005](decisions/ADR-005-containerization-strategy.md) |
| **Platform** | The pod/container `securityContext` *enforces* non-root, drops all capabilities, blocks privilege escalation, applies seccomp, and pins a numeric UID the kubelet can verify. | [`k8s/base/job.yaml`](../k8s/base/job.yaml), [ADR-010](decisions/ADR-010-kubernetes-security-hardening.md) |

The platform layer matters because the image's `USER appuser` is a **name**, which
the kubelet cannot resolve to a UID at admission — so `runAsNonRoot` alone would
reject the pod. The explicit numeric `runAsUser: 10001` is what makes non-root
enforcement actually admit and run.

---

## 2. Workload identity & least privilege

The pipeline runs `dvc repro` and talks only to MLflow/DagsHub over HTTPS — it
**never** calls the Kubernetes API. The identity model reflects exactly that:

| Control | Setting | Rationale |
|---|---|---|
| Dedicated `ServiceAccount` | `mlops-pipeline` (not `default`) | A named identity to scope any future policy to *this* workload, not everything in the namespace. |
| API-token automount | `automountServiceAccountToken: false` on **both** the ServiceAccount and the pod template | The workload needs no API token; not mounting one removes an unused, exfiltratable credential from the pod (defense in depth — asserted at both layers). |
| RBAC | **No `Role`/`RoleBinding`** | Granting permissions the workload never uses would violate least privilege. The absence is deliberate, not an omission. |
| Namespace | `mlops` (dedicated) | Environment/RBAC/quota/Pod-Security boundary scoped to this project. |

**Verified on a live cluster:** the applied pod carried
`serviceAccountName: mlops-pipeline` with **no `kube-api-access-*` projected-token
volume** and no `/var/run/secrets/kubernetes.io/serviceaccount` mount. (Before PR 8
the pod's `spec.volumes` was entirely empty; PR 8 added exactly two **read-only
ConfigMap** volumes — `dvc-runtime-config` and `dataset` — for the runtime contract
([ADR-013](decisions/ADR-013-kubernetes-runtime-execution.md)). The token-absence
guarantee is unchanged: still no projected API-token volume.)

---

## 3. Pod & container security context

Applied to the pipeline `Job` and enforced by the kubelet:

**Pod level** (every container in the pod):

| Field | Value | Purpose |
|---|---|---|
| `runAsNonRoot` | `true` | Refuse to start as UID 0 — enforced, not assumed. |
| `runAsUser` / `runAsGroup` | `10001` | Matches the Dockerfile identity; **required** because the image's `USER` is a name the kubelet can't verify. |
| `seccompProfile.type` | `RuntimeDefault` | Apply the runtime's default syscall filter instead of running unconfined. |

**Container level** (fields that cannot live at pod scope):

| Field | Value | Purpose |
|---|---|---|
| `allowPrivilegeEscalation` | `false` | No setuid/setgid escalation (verified `NoNewPrivs: 1`). |
| `capabilities.drop` | `[ALL]` | The pipeline needs **no** Linux capabilities (pure Python + `dvc` over HTTPS). |
| `readOnlyRootFilesystem` | `false` | **Deliberately deferred** — see [§4](#4-why-read-only-root-filesystem-is-deferred). |

No privilege/escape footguns are present: no `privileged`, `hostNetwork`,
`hostPID`, or `hostIPC`. This is asserted statically (`k8s/validate.py`) and was
observed enforced on the live pod (`kubectl get pod -o jsonpath='{.spec.securityContext}'`
and `{.spec.containers[0].securityContext}` reported exactly the values above).

---

## 4. Why read-only root filesystem is deferred

`readOnlyRootFilesystem: true` is evaluated and intentionally **not** enabled,
because `dvc repro` writes DVC state **in-tree at the `/app` repo root**. Under a
read-only root its first action fails:

```bash
docker run --rm --user 10001:10001 --cap-drop ALL --security-opt no-new-privileges \
  --read-only --tmpfs /tmp ml-pipeline:local dvc repro
# → ERROR: unexpected error - [Errno 30] Read-only file system: '/app/.dvc/tmp'
```

DVC further writes `/app/.dvc/cache`, rewrites `/app/dvc.lock`, and needs a writable
`/app/.git`. These sit at the repo root alongside the read-only baked-in code and
`.dvc/config`, so they cannot be carved out with `emptyDir` mounts without shadowing
image files or making the code tree writable (which defeats the control). Enabling
it now would make the container fail **earlier** than the pre-existing SCM blocker —
weakening a working workload to pass a checkbox.

It is deferred to the **same** work that makes the pipeline green in-cluster
(relocating DVC's cache/tmp/lock onto declared writable volumes), then flipping the
flag to `true`. Until then, the honest state is: *control evaluated, incompatibility
proven, deferred and recorded* — see
[ADR-010 §Read-only root filesystem](decisions/ADR-010-kubernetes-security-hardening.md).

> **PR 8 update.** The *green run* is now achieved
> ([ADR-013](decisions/ADR-013-kubernetes-runtime-execution.md)) — the SCM blocker
> is gone (DVC no-SCM mode, so no writable `/app/.git` is needed) and the dataset is
> mounted. But DVC's cache/tmp/lock still write in-tree at `/app`, so
> `readOnlyRootFilesystem` stays `false` — it is now the **only** remaining item of
> this deferral. Relocating that DVC state to writable volumes and enabling the flag
> is the tracked next step; it was intentionally kept out of the green-run PR to
> keep that change minimal.

---

## 5. Secret & data handling

**No credentials or datasets are committed, rendered, or baked into any image.**

| Concern | Handling |
|---|---|
| MLflow/DagsHub credentials (`MLFLOW_TRACKING_USERNAME`/`_PASSWORD`) | Held in a `Secret` created **out-of-band** from a git-ignored `.env` (`kubectl create secret … --from-env-file=.env`). The repo ships only [`k8s/base/secret.example.yaml`](../k8s/base/secret.example.yaml) with **placeholders**, and it is **excluded from `base/kustomization.yaml`** so no render or apply can ever emit it. |
| Non-secret config (`LOG_LEVEL`, `MLFLOW_TRACKING_URI`) | In the `mlops-pipeline-config` ConfigMap. `MLFLOW_TRACKING_URI` is a public endpoint (the same host already committed as the DVC S3 remote in `.dvc/config`), not a credential. |
| Secret reference in the Job | `secretRef … optional: true` — the pod starts even before the Secret exists; MLflow calls then fail with a clear auth error rather than the pod refusing to schedule. |
| Dataset | Not in the image (`.dockerignore`) and DVC-tracked; mounted at run time. |

**Secret hygiene is enforced in CI.** `k8s/validate.py` asserts that the rendered
workload contains **no `Secret`**, no inline credential values, no secret
fingerprints anywhere in the `k8s/` tree, and that the committed template holds only
placeholders. Manifests are **never** sent to any external scanning service.

> Kubernetes `Secret` `data` is base64, **not** encryption. Committing even the
> example with real values would leak the DagsHub token into git history
> permanently — which is exactly why the real Secret never enters git or a rendered
> manifest.

---

## 6. Controls checklist → evidence

| Control | Status | Evidence |
|---|---|---|
| Non-root enforced (numeric UID) | ✅ | `runAsNonRoot: true` + `runAsUser: 10001`; live pod `spec.securityContext`. |
| No privilege escalation | ✅ | `allowPrivilegeEscalation: false`; `NoNewPrivs: 1` under `docker run`. |
| All Linux capabilities dropped | ✅ | `capabilities.drop: [ALL]`; enforced on the live pod. |
| seccomp default profile | ✅ | `seccompProfile.type: RuntimeDefault`. |
| No API token in pod | ✅ | `automountServiceAccountToken: false` (SA + pod); no `kube-api-access-*` projected-token volume on the live pod (the only volumes are two read-only ConfigMaps, PR 8). |
| No standing RBAC | ✅ | no `Role`/`RoleBinding` in `k8s/`. |
| No baked-in secrets/data | ✅ | out-of-band Secret; `.dockerignore`; `k8s/validate.py` secret-hygiene checks. |
| No host namespaces / privileged | ✅ | no `privileged`/`hostNetwork`/`hostPID`/`hostIPC`; asserted by `k8s/validate.py`. |
| Read-only root filesystem | ⬜ deferred | Incompatibility proven; tied to the green-run/DVC-relocation work ([§4](#4-why-read-only-root-filesystem-is-deferred)). |
| Egress `NetworkPolicy` | ⬜ not present | Needs a policy-enforcing CNI + separate validation; a sensible next hardening step (ADR-010). |
| PSS `restricted` **certification** | ❌ not claimed | Controls applied and cluster-admitted, but no admission label/policy engine ratifies the profile ([§7](#7-what-is-not-claimed)). |

---

## 7. What is *not* claimed

- **Not restricted-PSS-compliant.** The manifest carries the fields the `restricted`
  profile expects and a live cluster admitted it, but **no Pod Security admission
  label or policy engine has validated the profile as a whole**, and read-only root
  is not met. "Hardened toward restricted" — yes; "certified restricted" — no.
- **Not a production security baseline.** No `NetworkPolicy` (egress is unrestricted
  at the cluster layer), no admission-control policy (OPA/Gatekeeper/Kyverno), no
  managed-cluster IAM, no secrets manager (KMS/Vault/CSI driver) — the Secret is a
  plain Kubernetes `Secret` created locally.
- **Not a green-run security proof.** The controls are validated up to the same
  pre-existing SCM blocker the workload hits with or without them; the hardening is
  **behaviour-neutral** (no new failure mode), which is what was proven — not that
  the pipeline completes.
- **No supply-chain provenance guarantee.** The image and its base are name-pinned,
  not digest-pinned or signed; CI tool binaries are version+checksum-pinned, not
  signed. Digest pinning and signing are roadmap items
  ([ADR-005](decisions/ADR-005-containerization-strategy.md),
  [ADR-012](decisions/ADR-012-kubernetes-manifest-validation.md)).

---

## 8. Verification

How to re-check the posture yourself:

```bash
# Static: the security/required-field/secret-hygiene contract (34 checks).
python k8s/validate.py

# Schema: reject unknown/mistyped security fields.
kustomize build k8s/overlays/local | \
  kubeconform -strict -summary -kubernetes-version 1.31.0 -schema-location default -

# Live enforcement (any local cluster): apply, then read the enforced context.
kubectl apply -k k8s/overlays/local
pod=$(kubectl -n mlops get pods -o jsonpath='{.items[0].metadata.name}')
kubectl -n mlops get pod "$pod" -o jsonpath='{.spec.securityContext}{"\n"}{.spec.containers[0].securityContext}{"\n"}{.spec.volumes}{"\n"}'
kubectl delete -k k8s/overlays/local
```

CI runs the first two on every push/PR (the `k8s-validate` job); the live check was
performed on Docker Desktop Kubernetes v1.34.3 (2026-08-12) and reported the exact
values in [§3](#3-pod--container-security-context) and [§2](#2-workload-identity--least-privilege).

---

## Related documentation

- [ADR-010 — Security Hardening](decisions/ADR-010-kubernetes-security-hardening.md)
- [ADR-009 — Workload Model & Identity](decisions/ADR-009-kubernetes-workload-model.md)
- [Kubernetes Architecture](kubernetes-architecture.md) ·
  [Kubernetes Operations](kubernetes-operations.md)
- [Containerization Strategy](containerization.md) ·
  [ADR-005](decisions/ADR-005-containerization-strategy.md)
- [`k8s/README.md`](../k8s/README.md) · [Security Policy](../SECURITY.md)
