# ADR-036: SBOM generation and immutable image provenance (Sprint 8, PR 9)

- **Status:** Accepted (design; CycloneDX SBOM + git→image provenance assertion added
  to CI, and operator release/verify scripts + optional digest-pinned deploy added. No
  image is pushed and no live EKS deploy happens in this PR — the credentialed push,
  registry-digest capture, and runtime verification are operator steps, run out-of-band.)
- **Date:** 2026-08-20
- **Deciders:** Asad Hanif
- **Related:**
  [ADR-005 (Containerization strategy — the OCI provenance labels this builds on)](ADR-005-containerization-strategy.md),
  [ADR-035 (Container-image scanning — the pinned Trivy this reuses to emit the SBOM)](ADR-035-container-image-scanning.md),
  [ADR-021 (Terraform-managed ECR — IMMUTABLE tags + scan-on-push, what makes tag→digest stable)](ADR-021-terraform-managed-ecr.md),
  [ADR-018 (AWS EKS deployment overlay — the renderer that pins the image)](ADR-018-aws-eks-deployment-overlay.md),
  [ADR-026 (In-cluster MLflow platform — the second image)](ADR-026-in-cluster-mlflow-platform.md),
  [ADR-019 (Terraform CI validation — the credential-free CI invariant)](ADR-019-terraform-ci-validation.md),
  [`docs/supply-chain-provenance.md`](../supply-chain-provenance.md),
  [`scripts/release-image.sh`](../../scripts/release-image.sh),
  [`scripts/verify-deployed-digest.sh`](../../scripts/verify-deployed-digest.sh),
  [`scripts/render-cloud-manifests.sh`](../../scripts/render-cloud-manifests.sh),
  [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)

> **Scope.** This ADR ratifies a **source-to-deployment artifact chain**: a **CycloneDX
> SBOM** for the release image, and a verifiable **git commit → image tag → sha256
> digest → running workload** provenance relationship. It adds an in-CI SBOM +
> provenance-label assertion, an operator `release-image.sh` that captures the immutable
> registry digest and emits the provenance record, a `verify-deployed-digest.sh` that
> checks the live workload, and an **opt-in** digest-pinned deploy in the renderer. It
> **evaluates cosign signing and makes it optional** (see § "Cosign evaluation"). It
> does **not** push an image, deploy to EKS, or introduce a mandatory signing gate.

## Context

Two things were already true before this PR, and one thing was missing.

**Already true:**

1. **The image embeds its own provenance.** The [`Dockerfile`](../../Dockerfile) runtime
   stage sets OCI labels `org.opencontainers.image.revision=${VCS_REF}` and
   `.version=${BUILD_VERSION}`, wired in CI from `github.sha` / the run number and by
   the operator from the git commit / release tag (ADR-005). So *git commit → image* is
   captured in the artifact itself.
2. **The registry makes a tag immutable.** ECR is `image_tag_mutability = IMMUTABLE`
   with `scan_on_push` (ADR-021), and the AWS overlay deploys an explicit version tag,
   never `:latest` (ADR-018). An immutable tag cannot be repointed, so *image tag →
   digest* is a stable, one-to-one binding once pushed.

**Missing:** nothing (a) produced a **Software Bill of Materials** for the image, (b)
**captured the immutable sha256 digest** the tag resolves to, (c) recorded the
end-to-end **git → tag → digest** relationship as evidence, or (d) **verified the
running workload** actually uses that digest. [`docs/ci-cd.md`](../ci-cd.md) § "Future
CD roadmap" earmarks exactly this: "Supply-chain hardening (v3): … SBOM generation, and
image signing (cosign)." ADR-035 delivered the *scanning* slice and explicitly left
SBOM + signing as follow-ups. This PR delivers the **SBOM + provenance-chain** slice.

**The binding constraint** is the same invariant every other PR respects: **ordinary PR
CI holds no AWS credentials and never pushes** (ADR-019). A `docker push`, the registry
digest it returns, and a live-cluster check are therefore inherently **operator-side,
credentialed** activities — the identical split the project already uses for Terraform
(static validation in CI; `apply` in the operator runbook). The design honours that
line rather than smuggling cloud credentials into CI to "complete the chain" there.

## Decision

Deliver the chain in two halves that meet at the digest: **credential-free CI** proves
everything it can without a registry, and a **credentialed operator step** captures the
digest and verifies runtime.

### 1. SBOM — CycloneDX, via the already-pinned Trivy

The SBOM format is **CycloneDX** (JSON), generated with **Trivy** (`trivy image
--format cyclonedx`). Trivy is already installed in the `docker` job as a pinned,
checksum-verified static binary for the ADR-035 scan, so emitting an SBOM adds **no new
tool and no new supply-chain surface** — the same reasoning that chose Trivy for
scanning. The SBOM is generated **from the actual built image** (the image just loaded
into the daemon in CI; the image just pushed, *by digest*, in the release script) — not
from a manifest or a dependency re-resolve — so it inventories the exact OS + Python
packages the artifact ships (requirement 3).

CycloneDX over SPDX: both are supported by Trivy and both are industry standards; the
choice is not load-bearing. CycloneDX is picked for its first-class vulnerability/VEX
alignment with the scanner already in use (a future step can attach the ADR-035
findings to the same BOM), and its compact JSON. SPDX would be an equally defensible
choice and is a one-flag change if a consumer requires it.

### 2. The provenance chain — git commit → image tag → sha256 digest

- **git commit → image (in CI, credential-free, deterministic).** A new `docker` job
  step asserts the built image's `org.opencontainers.image.revision` label equals the
  commit SHA and `.version` equals the expected build version. This is the guard that
  the git→image binding *actually works*: a Dockerfile change that dropped or mis-wired
  the labels fails in CI, before any operator relies on them.
- **image tag → digest (at operator push).** [`release-image.sh`](../../scripts/release-image.sh)
  builds with those labels, pushes to the immutable-tag ECR repo, and captures the
  sha256 **manifest digest** from the pushed image's `RepoDigests` — then **cross-checks
  it against `aws ecr describe-images`**, so the digest is confirmed from two
  independent sources (local daemon + registry API) before it is recorded.
- **The record.** The script emits a small `*.provenance.json` (commit, tag, image tag,
  digest, SBOM filename, signed flag) plus the CycloneDX SBOM into an output directory —
  the operator's evidence for this release.

### 3. Deploy — prefer by digest, never *forced*

The renderer ([`render-cloud-manifests.sh`](../../scripts/render-cloud-manifests.sh))
gains an **opt-in** `IMAGE_DIGEST` / `MLFLOW_IMAGE_DIGEST`: when set, the Kustomize
`images:` transformer renders `newName@sha256:…` (the `digest:` field), binding the
deploy to the exact immutable artifact. It is **opt-in** because the ECR tag is
*already* immutable (ADR-021), so tag-deploy is already reproducible — and the task is
explicit that deployment must not be made **brittle merely to force digest syntax**.
Whichever pinning is used, correctness is established by **verification, not syntax**
(next point).

### 4. Runtime verification — what is ACTUALLY running

[`verify-deployed-digest.sh`](../../scripts/verify-deployed-digest.sh) reads the live
pod's `containerStatuses[].imageID` (the sha256 the kubelet actually pulled) for every
container using the pipeline image — init containers `fetch-dataset` / `wait-for-mlflow`
and the `pipeline` container all share it — and asserts each equals the expected digest,
failing non-zero on any mismatch. This is the "deploy by tag but **verify** the runtime
imageID exactly matches the expected digest" path the task accepts, and it independently
confirms the by-digest path too.

### 5. Artifacts, not committed binaries

CI uploads the SBOMs + provenance record as the **`sbom-and-provenance`** build
artifact (`always()`, 30-day retention), beside the ADR-035 `trivy-image-reports`. The
generated SBOM JSON is **not committed to git**: it is large, regenerated every build,
and changes with every dependency resolution — committing it would be binary-review
noise (the task's "do not commit generated binary junk"). The durable, human-authored
evidence lives in [`docs/proof/sprint-08-sbom-provenance-evidence.md`](../proof/sprint-08-sbom-provenance-evidence.md).

### 6. Secure, credential-free CI integration

The CI additions inspect only the locally-built image and run Trivy (already present).
Job permissions stay `contents: read`; no registry login, no AWS identity, no
`packages:` scope. The vuln-DB / SBOM generation is offline for the SBOM (no DB needed
for the component inventory). The push, ECR digest read, and cluster query live entirely
in the operator scripts, which use the standard AWS credential chain and never read a
secret from the repo.

## Cosign evaluation

**Decision: signing is EVALUATED and made OPTIONAL — an opt-in `--sign` (keyless OIDC)
in `release-image.sh`, off by default, NOT a mandatory gate.**

Reasoning:

- **It cannot live in PR CI.** A signature is attached to a **pushed** image's digest.
  Ordinary CI never pushes and holds no identity (ADR-019), so there is nothing to sign
  there — signing is inherently an operator-release action, exactly like the digest
  capture.
- **What it would add, honestly.** Keyless cosign (Sigstore: an ephemeral Fulcio
  certificate bound to an OIDC identity, logged to the Rekor transparency log) proves
  *who/what built and pushed* the digest. That is real, but it is **additive** on top of
  a chain that is already strong for this project's threat model: the source→artifact
  link is the OCI label (asserted in CI) + the immutable ECR tag + the two-source digest
  capture, and the artifact→runtime link is `verify-deployed-digest.sh`. Signing does
  **not** replace any of those.
- **Cost of making it mandatory.** A hard signing gate pulls in an external transparency
  log + Fulcio OIDC dependency, a key/identity-policy decision (which issuers/identities
  a `cosign verify` should trust), and an admission-time verification story
  (e.g. a policy controller) to *enforce* signatures — none of which a single-operator,
  ephemeral, `provision→prove→destroy` environment (ADR-020) currently has or needs.
  Forcing it now would be ceremony without a verifier.
- **So:** it is wired as a **one-flag opt-in** (`release-image.sh --sign`, guarded on
  `cosign` being present) that signs the pushed image **by digest** and prints the
  matching `cosign verify` command, and it is documented. A future PR can promote it to a
  release gate with an admission-time verifier when the deployment model justifies one
  (roadmap v3). This satisfies "image signing is optional unless it integrates cleanly
  and materially improves proof" and "document the signing decision either way".

## Alternatives considered

- **SPDX instead of CycloneDX.** Equally standard, equally supported by Trivy. *Not
  chosen* only for CycloneDX's tighter fit with the scanner already in the repo; it is a
  one-flag switch if a consumer needs SPDX.
- **Syft/Grype for the SBOM.** Capable, but a second tool alongside the pinned Trivy is
  redundant supply-chain surface — the same "no tool sprawl" call as ADR-035. *Rejected.*
- **BuildKit inline attestations (`docker build --sbom=true --provenance=true`).** Would
  embed SBOM/provenance as attestation manifests in the image. *Rejected:* those
  manifests confused the image scanner (documented in the Sprint 8 PR 8 evidence), are
  not what a deploy pins to, and are harder to consume as a standalone artifact than an
  explicit CycloneDX file. The release build keeps `--provenance=false --sbom=false` and
  emits an explicit SBOM instead.
- **Mandatory cosign signing gate.** *Rejected for now* — see § "Cosign evaluation":
  no push in CI, and no admission-time verifier to enforce it, so it would be ceremony.
- **Always deploy strictly by digest (drop tag support).** *Rejected:* the ECR tag is
  already immutable, so tag-deploy is reproducible; forcing digest-only syntax adds
  brittleness (every render needs a digest lookup) for no additional guarantee once
  `verify-deployed-digest.sh` runs. Digest pinning is offered as the preferred opt-in.
- **Commit the SBOM into git.** *Rejected:* generated, large, changes every build —
  artifact, not source.
- **Capture the digest in CI by pushing to a throwaway registry.** *Rejected:* it would
  put a registry credential (or a GHCR `packages: write` scope) into PR CI, breaking the
  credential-free invariant, to capture a digest that is not the one the operator
  actually deploys. The operator script captures the real one.

## Consequences

**Positive**

- Every release has a **CycloneDX SBOM** of the exact shipped packages and a recorded
  **git → tag → digest** chain, cross-checked against the registry.
- The running workload can be **proven** to match the intended digest at runtime, making
  deploy-by-(immutable-)tag safe and deploy-by-digest verifiable.
- All of it reuses existing tools and patterns (Trivy, the renderer, the operator-runbook
  split); **PR CI stays credential-free**.
- Signing is one opt-in flag away without imposing an unverifiable gate.

**Negative / limitations (stated honestly)**

- **The CI SBOM is of the CI-built image, not the pushed artifact.** Credential-free CI
  never pushes, so the per-PR SBOM + image-ID describe the image CI built (same
  Dockerfile, same commit), while the **release** SBOM + registry digest are produced by
  `release-image.sh` at push time. Until builds are byte-reproducible (below), the two
  are equivalent-by-construction, not bit-identical-by-proof.
- **No reproducible builds yet.** `requirements.txt` pins by name and the base image by
  tag, so the resolved package set can shift build-to-build; digest-pinning the base and
  freezing dependencies (ADR-005) is the prerequisite for a byte-for-byte reproducible
  digest. Noted, not claimed.
- **Runtime verification needs a live cluster.** `verify-deployed-digest.sh` runs
  against a deployed workload; in this ephemeral model that is an operator step, so this
  PR ships the mechanism + local evidence, and the live EKS capture is a documented
  checklist (like the Sprint 8 PR 7 NetworkPolicy runtime evidence).
- **Signing is not enforced.** By decision above; a chain-of-custody *enforcement*
  (admission-time verify) is future work.

**Follow-ups (roadmap, not this PR)**

- Promote cosign to a release gate with an admission-time verifier (e.g. a policy
  controller) once the deployment model warrants enforcement (roadmap v3).
- Digest-pin the base image + freeze dependencies for byte-reproducible builds (ADR-005),
  so the CI SBOM and the release SBOM are provably identical.
- Optionally attach the ADR-035 scan findings to the CycloneDX BOM as VEX, and emit SARIF
  to GitHub code scanning on the publish step (scoped `security-events: write`, still
  AWS-free).
- Execute the live EKS runtime-digest verification on the next enforcing-cluster run and
  record it in the evidence file.
