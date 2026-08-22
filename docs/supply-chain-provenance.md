# SBOM & image provenance

**Sprint 8, PR 9.** Design of record: [ADR-036](decisions/ADR-036-sbom-and-image-provenance.md).

This document is the operator-facing guide to the **source-to-deployment artifact
chain**: the **SBOM** produced for the release image, and the verifiable relationship

```
  git commit (+ tag)  ──▶  image tag (immutable)  ──▶  sha256 digest  ──▶  running workload
```

It is the supply-chain counterpart to the image **vulnerability** scan
([ADR-035](decisions/ADR-035-container-image-scanning.md) ·
[container-image-scanning.md](container-image-scanning.md)) — same tool (Trivy), same
pinned/checksum-verified install, same "prove it, don't claim it" discipline.

> **The credential line.** Everything CI does here is **credential-free** and runs on
> every PR (SBOM + the git→image label assertion). Everything that needs a **registry
> push** or a **live cluster** — capturing the immutable digest, signing, verifying the
> deployed workload — is an **operator step** run from your own AWS account, exactly
> like the Terraform `apply` split ([ADR-019](decisions/ADR-019-terraform-ci-validation.md)).
> CI never holds AWS credentials and never pushes.

---

## 1. What runs where

| Link of the chain | Where | Credential-free? | Mechanism |
|---|---|:---:|---|
| **git commit → image** | CI (`docker` job) + operator build | ✅ | OCI labels `org.opencontainers.image.revision` / `.version` (set by the [`Dockerfile`](../Dockerfile)); CI **asserts** the label equals the commit SHA. |
| **SBOM of the image** | CI (per-PR) + operator (per-release) | ✅ (CI) | `trivy image --format cyclonedx` from the **actual built image**. |
| **image tag → sha256 digest** | operator (`release-image.sh`) | ❌ push | Capture the pushed image's `RepoDigest`, cross-checked against `aws ecr describe-images`. |
| **digest → running workload** | operator (`verify-deployed-digest.sh`) | ❌ cluster | Compare the live pod `imageID` to the expected digest. |
| **signature (optional)** | operator (`release-image.sh --sign`) | ❌ push | Keyless cosign by digest (off by default — see §5). |

---

## 2. SBOM — format, tool, location

- **Format:** **CycloneDX** (JSON). SPDX is a one-flag alternative (`--format spdx-json`)
  if a consumer requires it; see ADR-036 § Decision.
- **Tool:** **Trivy** (`trivy image --format cyclonedx`) — the same pinned,
  checksum-verified binary the vulnerability scan uses, so no new tool is added.
- **Corresponds to the real image:** generated **from the built image** (in CI, the
  image just loaded into the daemon; in the release script, the image just pushed, *by
  digest*) — never from a manifest or a re-resolve.
- **Location:**
  - **CI, per PR:** the **`sbom-and-provenance`** build artifact (both images'
    `*.cdx.json`, their image IDs, and a `provenance.txt`), 30-day retention, uploaded
    with `always()`. It is **not committed to git** — a generated SBOM is large, changes
    every build, and would be review noise (ADR-036 § 5).
  - **Operator, per release:** written by `release-image.sh` into its `--out` directory
    alongside the `*.provenance.json` record.
  - **Durable evidence:** [`docs/proof/sprint-08-sbom-provenance-evidence.md`](proof/sprint-08-sbom-provenance-evidence.md).

---

## 3. Cut a release — capture the git→tag→digest chain

Prerequisites: the operator prerequisites from
[cloud-operations.md § 2](cloud-operations.md#2-prerequisites) (AWS CLI v2, Docker,
Terraform with applied state), plus **Trivy** on PATH, and **cosign** only if you pass
`--sign`. Run from the repo root, against **your own** account.

```bash
# Build → push → capture the immutable digest → emit SBOM + provenance record.
# The ECR repo URL is read from `terraform output` (account id stays out of git).
scripts/release-image.sh --tag 1.7.0 --out ./release-evidence

#   … MLflow server image — layered FROM the pipeline image just released, so pass
#   that image's ECR ref as BASE_IMAGE (it is in the local daemon from the push above;
#   its tag differs from the MLflow --tag, so it must be named explicitly):
BASE_IMAGE="$(terraform -chdir=terraform output -raw ecr_repository_url):1.7.0" \
  scripts/release-image.sh --mlflow --tag 0.1.0 --out ./release-evidence
```

It prints the chain and writes `release-evidence/mlops-pipeline-1.7.0.provenance.json`:

```json
{
  "image": "mlops-pipeline",
  "git": { "commit": "<sha>", "tag": "v1.7.0" },
  "image_tag": "1.7.0",
  "image_digest": "sha256:…",
  "sbom": { "format": "CycloneDX", "tool": "trivy", "file": "mlops-pipeline-1.7.0.cdx.json" }
}
```

The digest is captured from the pushed image **and** cross-checked against
`aws ecr describe-images`, so it is confirmed from two sources before it is recorded.
Because the ECR repository enforces **immutable tags** ([ADR-021](decisions/ADR-021-terraform-managed-ecr.md)),
that `tag → digest` mapping can never change.

> **Dry run (no AWS):** `scripts/release-image.sh --tag 1.7.0 --no-push --repo \
> example.dkr.ecr.us-east-1.amazonaws.com/mlops-pipeline` builds locally and emits the
> SBOM from the local image without pushing — useful to exercise the mechanism.
>
> **Historical note:** Sprint 8 PR 16 captured runtime evidence using the pipeline image tagged `1.6.0`.
> The v1.7.0 release is built from the same final release code (same git commit), but retagged as `1.7.0`
> for deployment. PR16 evidence `mlops-pipeline-1.6.0.provenance.json` remains the authoritative runtime
> validation artifact; the `1.7.0` tag creates a clean release lineage forward from v1.7.0.

---

## 4. Deploy pinned to the digest, then verify at runtime

**Preferred — deploy by digest (immutable by construction):**

```bash
digest=$(jq -r .image_digest release-evidence/mlops-pipeline-1.7.0.provenance.json)
IMAGE_DIGEST="$digest" scripts/render-cloud-manifests.sh --apply
```

The renderer sets the Kustomize `digest:` field, so the workload pulls
`…/mlops-pipeline@sha256:…`. Digest pinning is **opt-in**: with no `IMAGE_DIGEST` the
overlay deploys the **immutable tag** (already reproducible), so this is a hardening
choice, not a requirement — deployment is never made brittle just to force digest
syntax (ADR-036 § 3).

**Verify what is ACTUALLY running** (works for either pinning — the *verification*, not
the syntax, is the guarantee):

```bash
scripts/verify-deployed-digest.sh --record release-evidence/mlops-pipeline-1.7.0.provenance.json
#   or explicitly:  scripts/verify-deployed-digest.sh --expect "$digest"
```

It reads every pipeline-image container's live `imageID` (the init containers
`fetch-dataset` / `wait-for-mlflow` and the `pipeline` container all share the image)
and exits non-zero on any mismatch:

```
  OK    mlops-pipeline-xxxxx/fetch-dataset: sha256:…
  OK    mlops-pipeline-xxxxx/wait-for-mlflow: sha256:…
  OK    mlops-pipeline-xxxxx/pipeline: sha256:…
PASS: all 3 container(s) run the expected digest sha256:…
```

---

## 5. Image signing (cosign) — the decision

**Signing is OPTIONAL and off by default.** `release-image.sh --sign` adds a **keyless**
(Sigstore OIDC → Fulcio cert → Rekor transparency log) signature on the pushed image
**by digest**, and prints the matching `cosign verify` command. It is one flag, guarded
on `cosign` being installed.

It is **not** a mandatory gate because: PR CI never pushes, so there is nothing to sign
there; the chain is already strong for this project's single-operator, ephemeral model
(OCI label asserted in CI + immutable ECR tag + two-source digest capture + runtime
digest verification); and a hard gate would need an admission-time verifier to *enforce*
signatures, which this environment does not run. Full rationale:
[ADR-036 § "Cosign evaluation"](decisions/ADR-036-sbom-and-image-provenance.md#cosign-evaluation).
Promoting signing to an enforced gate is a roadmap v3 follow-up.

---

## 6. What this covers and does not

- **Covers:** a CycloneDX SBOM of the exact shipped packages; a git→tag→digest chain
  captured and cross-checked against the registry; runtime confirmation that the deployed
  workload uses the expected digest; an opt-in signing path.
- **Does not (yet):**
  - **Byte-reproducible builds** — deps pin by name, base by tag (ADR-005), so the CI
    SBOM is *equivalent-by-construction* to the release image, not bit-identical-by-proof.
  - **Enforce signatures at admission** — signing is opt-in, not gated (§5).
  - **Replace ECR scan-on-push** or the ADR-035 image scan — complementary layers.

## See also

- [ADR-036](decisions/ADR-036-sbom-and-image-provenance.md) — the decision record
- [ADR-035](decisions/ADR-035-container-image-scanning.md) · [container-image-scanning.md](container-image-scanning.md) — the sibling vulnerability scan
- [cloud-operations.md](cloud-operations.md) — the full operator runbook (§ 3.7 publish)
- [ADR-021](decisions/ADR-021-terraform-managed-ecr.md) — immutable-tag ECR
- [scripts/release-image.sh](../scripts/release-image.sh) · [scripts/verify-deployed-digest.sh](../scripts/verify-deployed-digest.sh) · [scripts/render-cloud-manifests.sh](../scripts/render-cloud-manifests.sh)
- [Sprint 8 SBOM/provenance evidence](proof/sprint-08-sbom-provenance-evidence.md)
- [SECURITY.md](../SECURITY.md) — the security policy
