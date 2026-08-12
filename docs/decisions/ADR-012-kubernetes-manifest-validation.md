# ADR-012 — Automated Kubernetes Manifest Validation in CI

- **Status:** Accepted
- **Date:** 2026-08-12
- **Supersedes / relates to:** [ADR-009](ADR-009-kubernetes-workload-model.md)
  (workload model), [ADR-010](ADR-010-kubernetes-security-hardening.md) (security
  contract), [ADR-011](ADR-011-kubernetes-resource-lifecycle.md) (resources &
  lifecycle). This ADR makes the guarantees those three describe **enforced by CI**.

## Context

Sprint 5 built the Kubernetes surface incrementally: a `Job` workload model
(PR 1–2), externalized config + a least-privilege identity (PR 3), a hardened
`securityContext` (PR 4), and measured resources + lifecycle (PR 5). Each of
those was validated **once, by hand** (rendered assertions + a live Docker
Desktop run). Nothing stopped a future edit from silently regressing them — a
dropped `runAsNonRoot`, a typo'd field name that Kubernetes would ignore, a
credential pasted into a manifest, an accidentally `:latest` image.

CI already validates the Python package and the container image
([ci.yml](../../.github/workflows/ci.yml), [ADR-004](ADR-004-python-quality-toolchain.md),
[ADR-005](ADR-005-containerization-strategy.md)) but explicitly **did not touch
Kubernetes**. We want the manifests held to the same standard: every push and PR
should re-prove that they are well-formed, schema-valid, render through Kustomize,
and still satisfy the PR 1–5 security/resource contract — **without** deploying,
running the workload, or requiring a cluster on the critical path.

Two constraints shaped the design:

1. **Deterministic and fast.** The gate must not flake and must not add minutes.
   That rules out anything that spins up a cluster on every PR.
2. **Minimal toolchain.** Prefer a couple of small, pinned, single-binary tools
   plus a project-owned script over a large policy framework we would have to
   learn, configure, and keep current.

## Decision

Add Kubernetes validation to CI as **two clearly separated tiers**.

### Tier 1 — Static validation (`k8s-validate` job, every push/PR)

Deterministic, no cluster, workload never runs. Three cooperating checks:

| Concern | Tool | What it proves |
|---|---|---|
| YAML syntax + **Kustomize rendering** | `kustomize build` (pinned v5.4.3) | `base/` and `overlays/local/` render cleanly; the YAML parses. |
| **Kubernetes schema** | `kubeconform` (pinned v0.6.7, `-strict`, schema v1.31.0) | Every field is a real API field of the right type; unknown/misplaced fields are rejected. |
| **Security & required fields** | `k8s/validate.py` (stdlib + PyYAML) | The workload contract a schema check can't express (below). |

`k8s/validate.py` renders `overlays/local` and asserts, with a PASS/FAIL line per
check so a failure names exactly what broke:

- **Security (PR 1–4):** `runAsNonRoot: true` + non-root `runAsUser`; container
  `allowPrivilegeEscalation: false`; pod `seccompProfile: RuntimeDefault`;
  `capabilities: drop [ALL]`; an explicit **non-default** ServiceAccount that
  actually exists in the render; `automountServiceAccountToken: false` on both the
  pod and the ServiceAccount.
- **Required fields (PR 5):** CPU and memory **requests and limits**; a Job
  `restartPolicy` of `Never`/`OnFailure`; an **explicit, pinned** image (no
  floating `:latest`); every namespaced object pinned to `mlops`.
- **Secret hygiene:** no `Secret` object in the rendered workload (credentials are
  out-of-band); no inline credential-keyed values; no high-signal secret
  fingerprints (AWS/GitHub/Slack tokens, private keys) anywhere in `k8s/`; the
  committed Secret **template** carries only placeholders.

Tool **and** schema versions are pinned (env vars in the job) and the downloaded
binaries are checksum-verified, so a green run today is a green run tomorrow.

### Tier 2 — Cluster admission (`k8s-cluster-dry-run` job, opt-in)

`workflow_dispatch` only — **not** on the per-PR path. It stands up an *ephemeral*
kind cluster and does a **server-side dry run**
(`kubectl apply -k … --dry-run=server`): every object passes through a real API
server's validation, defaulting, and admission (including Pod Security) but nothing
is persisted and the Job never runs. This is the one check that needs a cluster, so
it is kept off the critical path (cost + flake surface) and made available on
demand — the sanctioned "implement reliable static validation; document cluster
integration separately" split.

## Alternatives Considered

1. **A full policy engine (OPA/Gatekeeper + Rego, or Conftest).** Most powerful,
   but a whole policy language and ruleset to author and maintain for a handful of
   assertions. Rejected as disproportionate to a single small workload.
2. **`kube-linter` / `kubesec` / `checkov` / `trivy config`.** Recognized static
   security scanners. Each covers *some* of our checks, but none covers all of the
   specific contract items (seccomp `RuntimeDefault`, explicit ServiceAccount name,
   token automount off, namespace pinning, our secret-hygiene rules) out of the
   box, and adopting one still leaves gaps we'd fill with custom config. A ~200-line
   project script does **exactly** our contract with understandable messages and no
   new heavyweight dependency. (`kubeconform` is still adopted — it's the right,
   tiny tool for the one thing a script shouldn't reimplement: OpenAPI schema
   validation.)
3. **`kubeval` for schema.** Deprecated and unmaintained; `kubeconform` is its
   maintained successor.
4. **kind integration on every PR.** Real, but slow and flakier (image pulls,
   cluster bootstrap) and it still would not run the workload to completion (the
   pipeline needs an SCM + mounted data — [ADR-010](ADR-010-kubernetes-security-hardening.md)).
   Static validation gives most of the signal deterministically; the ephemeral
   cluster is retained as an opt-in admission check.
5. **Reuse the PR 5 `assert_pr5.py` scratch script.** It was a throwaway probe.
   `k8s/validate.py` is its productionized, generalized successor, committed and
   runnable locally (`python k8s/validate.py`) and in CI.

## Consequences

- A regression in any PR 1–5 guarantee now **fails CI** rather than shipping
  silently. Verified by temporarily flipping `allowPrivilegeEscalation`/`runAsNonRoot`
  (caught by `k8s/validate.py`) and by giving `activeDeadlineSeconds` a string value
  (caught by `kubeconform -strict`).
- The gate is deterministic and fast: no cluster, workload never executed, tool +
  schema versions pinned. Its only network touch is `kubeconform` fetching the
  pinned upstream schema.
- Contributors get the same checks locally: `python k8s/validate.py` (uses the
  local `kustomize`/`kubectl`), plus the documented `kubeconform` command.
- New maintenance surface: the pinned versions (kustomize, kubeconform, schema) are
  bumped deliberately, and `k8s/validate.py` must evolve with the manifests.

## What This Decision Does **Not** Imply

- **It is not deployment validation.** Tier 1 is *static*; Tier 2 is *admission*
  (dry-run). Neither runs the pipeline in-cluster, so neither proves it completes —
  it still needs an SCM in the image + mounted data
  ([ADR-010](ADR-010-kubernetes-security-hardening.md), k8s/README.md).
- **It does not certify Pod Security Standard compliance.** The static checks
  assert the individual fields the restricted profile expects, but no Pod Security
  admission label or policy engine ratifies the profile as a whole (unchanged from
  [ADR-010](ADR-010-kubernetes-security-hardening.md)).
- **It does not pin supply-chain provenance.** Binaries are pinned by version and
  checksum-verified against their own release, not pinned by digest or signature —
  a future hardening step (roadmap, [ADR-005](ADR-005-containerization-strategy.md)).
