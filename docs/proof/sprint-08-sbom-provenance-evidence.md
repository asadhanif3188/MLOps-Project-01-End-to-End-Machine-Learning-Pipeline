# Sprint 8 PR 9 — SBOM & image-provenance evidence (PARTIAL: mechanism EXECUTED)

- **Date:** 2026-08-20
- **Branch / PR:** `feature/sprint-08-sbom-provenance` — *feat(supply-chain): generate
  SBOM and verify immutable image provenance*
- **Design of record:** [ADR-036](../decisions/ADR-036-sbom-and-image-provenance.md) ·
  **Operator guide:** [docs/supply-chain-provenance.md](../supply-chain-provenance.md)
- **Tooling:** Docker 29.3.1 (local Docker Desktop), Trivy `0.74.0` (the repo-pinned
  version, run from `aquasec/trivy:0.74.0`), kustomize `v5.4.3`.

> **What is proven here vs. deferred.** The **credential-free** half of the chain — the
> **git→image** label binding and the **CycloneDX SBOM** of the actual built image — is
> **executed and captured below** against the release image built from this branch. The
> **push → registry digest** and **live-EKS runtime digest** steps require a credentialed
> ECR push and an enforcing cluster, so — exactly like the Sprint 8 PR 7 NetworkPolicy
> runtime evidence — the **mechanism (scripts) is shipped and the digest-render is
> proven**, while the live capture is the operator checklist in § 4. Nothing below is a
> pending checklist item dressed as a result: the numbers are real Trivy/Docker output.

---

## 1. Image under test

Built the release **application** image from this branch's `HEAD` with the release
build-args (the exact command `scripts/release-image.sh` runs):

```bash
docker build --target runtime \
  --build-arg VCS_REF="$(git rev-parse HEAD)" \
  --build-arg BUILD_VERSION="1.6.0" \
  -t ml-pipeline:prov .
```

- **Source commit:** `a6f11d1a3ec26a1417165d964dbf6de5dda703fd` (branch point of
  `feature/sprint-08-sbom-provenance`).
- **Local image ID (config digest):**
  `sha256:a045a3be4b43dea522801b7eee1b163177ab6111a6ee77e1eca65cf535e8d2b3`

## 2. git → image binding (EXECUTED — the CI Stage 15 assertion)

`docker inspect` of the built image's OCI provenance labels:

```
org.opencontainers.image.revision = a6f11d1a3ec26a1417165d964dbf6de5dda703fd
org.opencontainers.image.version  = 1.6.0
org.opencontainers.image.title    = mlops-pipeline
```

`git rev-parse HEAD` = `a6f11d1a3ec26a1417165d964dbf6de5dda703fd`.

**Result:** `revision` **==** the commit SHA, exactly. This is the assertion CI Stage 15
runs on every PR — the first link of the chain (**git commit → image**) is proven from
the artifact's own metadata, not asserted on faith.

## 3. SBOM (EXECUTED — CycloneDX via Trivy)

Generated from the **actual built image** (saved to a tarball, scanned with the pinned
Trivy container — the same cross-platform method the PR 8 scan evidence used):

```bash
docker save ml-pipeline:prov -o img.tar
docker run --rm -v "<host>:/w" aquasec/trivy:0.74.0 \
  image --input /w/img.tar --format cyclonedx --scanners license \
  --skip-version-check --output /w/mlops-pipeline.cdx.json
```

Captured SBOM (`mlops-pipeline.cdx.json`, 501,765 bytes):

| Field | Value |
|---|---|
| `bomFormat` | **CycloneDX** |
| `specVersion` | **1.7** |
| `serialNumber` | `urn:uuid:5ba02bd1-facb-45ba-8dbf-9bcb3f895b8f` |
| Components (total) | **321** |
| — of type `library` | 320 |
| — of type `operating-system` | 1 (Debian 12) |

Sample components: `urllib3`, `requests`, `apt`, `base-files`, `bash`, … — the SBOM
inventories the exact OS + Python packages the image ships (requirement 3). In CI this
runs from the daemon image directly (`trivy image … ml-pipeline:ci`); the release script
runs it against the pushed image **by digest**.

> Note: scanning a `docker save` tarball reports the BOM subject as the input path
> (`/w/img.tar`); when Trivy scans a tagged/digest reference (CI and the release script)
> the subject is the image reference. The component inventory is identical either way.

## 4. tag → digest → running workload (mechanism shipped; live capture deferred)

These require a credentialed ECR push and an enforcing EKS cluster (neither is
provisioned for this PR — no billable environment was stood up). The **mechanism** is
shipped and the **digest-render is proven**; execute the live capture on the next EKS run.

### 4a. Digest-pinned render — PROVEN

`render-cloud-manifests.sh`'s new `IMAGE_DIGEST` path renders the Kustomize `digest:`
field. Verified on a temp copy of the AWS overlay (placeholder account substituted,
`newTag` swapped for a digest):

```
images:
  - name: ml-pipeline
    newName: <account>.dkr.ecr.us-east-1.amazonaws.com/mlops-pipeline
    digest: "sha256:<64-hex>"
# kustomize build → image: <account>.dkr.ecr.us-east-1.amazonaws.com/mlops-pipeline@sha256:<64-hex>
```

The workload is pinned to `…/mlops-pipeline@sha256:…` — deploy **by digest**, cleanly,
through the existing renderer.

### 4b. Operator checklist (run on the next enforcing-cluster session)

> **⏳ Capture this TOGETHER with the Sprint 8 PR 7 NetworkPolicy runtime evidence** —
> [sprint-08-network-policy-runtime-evidence.md](sprint-08-network-policy-runtime-evidence.md),
> also `PENDING` on the next enforcing cluster. Standing up EKS is the billable part
> (`provision → prove → destroy`, ADR-020); both captures need only the same deployed
> workload, so **do both in one session** — deploy once, run the netpol harness *and*
> the digest verification below against the same live cluster, then tear down.

- [ ] `scripts/release-image.sh --tag 1.6.0 --out ./release-evidence` — build+push to
      the Terraform-managed ECR repo; capture the immutable `sha256` digest (cross-checked
      against `aws ecr describe-images`) + emit `*.provenance.json` + the CycloneDX SBOM.
- [ ] Record the `git commit → image tag → sha256 digest` line from the script output.
- [ ] `IMAGE_DIGEST=<digest> scripts/render-cloud-manifests.sh --apply` — deploy pinned
      to the digest.
- [ ] `scripts/verify-deployed-digest.sh --record ./release-evidence/…provenance.json` —
      confirm every pipeline-image container's live `imageID` equals the expected digest
      (expect `PASS: all 3 container(s) …`).
- [ ] (optional) `scripts/release-image.sh --sign …` — keyless cosign signature by
      digest; record the `cosign verify` command.

## 5. Honesty boundary

- The SBOM + image ID above are for the **locally-built** image at commit `a6f11d1`;
  they are equivalent-by-construction to the pushed release image (same Dockerfile, same
  commit) but not **bit-identical-by-proof** until builds are digest-pinned/reproducible
  ([ADR-005](../decisions/ADR-005-containerization-strategy.md)) — stated, not hidden.
- No image was pushed and no EKS cluster was created for this PR, so the **registry
  manifest digest** and the **live runtime-digest PASS** are § 4b operator steps, not
  results claimed here.
- **cosign signing is opt-in and was not executed** (cosign not installed in this
  environment); the decision to keep it optional is recorded in
  [ADR-036 § "Cosign evaluation"](../decisions/ADR-036-sbom-and-image-provenance.md#cosign-evaluation).
