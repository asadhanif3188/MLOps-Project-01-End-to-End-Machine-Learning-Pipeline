#!/usr/bin/env bash
# release-image.sh — build → push → PIN the release application image, and emit the
# source-to-deployment provenance chain:
#
#     git commit (+ tag)  ──▶  image tag (immutable)  ──▶  sha256 registry digest
#
# This is the CREDENTIALED, operator-run counterpart to the credential-free CI
# provenance step (.github/workflows/ci.yml Stage 15-17). CI proves the git→image
# LABEL binding and emits a per-PR SBOM, but it never pushes (no AWS identity), so
# it cannot capture the one thing a deployment actually pins to: the immutable
# registry manifest digest. This script does — against the operator's OWN account,
# using the standard AWS credential chain, exactly like render-cloud-manifests.sh.
#
# Design of record: docs/decisions/ADR-036-sbom-and-image-provenance.md and
# docs/supply-chain-provenance.md. See also docs/cloud-operations.md § 3.7.
#
# What it does, in order:
#   1. Resolve the git provenance (commit SHA + any tag pointing at HEAD) and refuse
#      to release a dirty tree unless --allow-dirty (a released digest must map to a
#      committed source state).
#   2. Read the ECR repository URL from `terraform output` (account id lives in
#      state, never in git — same contract as render-cloud-manifests.sh), unless
#      --repo overrides it.
#   3. `docker build` the image with the OCI provenance labels
#      (org.opencontainers.image.revision=<commit>, .version=<tag>) wired from the
#      Dockerfile, for linux/amd64 to match the EKS node.
#   4. `docker push` to the immutable-tag ECR repository.
#   5. Capture the immutable sha256 registry digest from the pushed image
#      (RepoDigests), and cross-check it against `aws ecr describe-images`.
#   6. Generate a CycloneDX SBOM FROM the pushed image BY DIGEST (Trivy), so the SBOM
#      inventories the exact artifact that will be pulled.
#   7. Write the provenance record (JSON + a human-readable block) into the output
#      directory — NOT into git (it is per-release evidence; generated SBOM/records
#      are artifacts, not tracked source).
#   8. (--sign) Optionally cosign-sign the image BY DIGEST (keyless OIDC). Off by
#      default; see ADR-036 § "Cosign evaluation" for why signing is optional.
#
# Usage:
#   scripts/release-image.sh [options]
#     --tag TAG           image tag / BUILD_VERSION      (default: 1.6.0)
#     --repo URL          ECR repo URL (…/mlops-pipeline); default from terraform
#     --mlflow            release the MLflow server image instead of the pipeline
#                         image (docker/mlflow/Dockerfile, repo mlflow_server_…)
#     --ref GITREF        git ref to stamp as provenance (default: HEAD)
#     --out DIR           write SBOM + provenance record here (default: a temp dir)
#     --sign              cosign-sign the pushed image by digest (keyless OIDC)
#     --allow-dirty       permit a release from an uncommitted working tree
#     --no-push           build + generate locally, skip push/digest capture (dry)
#     -h, --help          show this help
#
# Requirements: bash, git, docker (buildx), aws CLI v2 (unless --repo AND --no-push),
# terraform with applied state (unless --repo), trivy, and — for --sign — cosign.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TF_DIR="${TF_DIR:-${REPO_ROOT}/terraform}"

# --- defaults ------------------------------------------------------------------
TAG="${IMAGE_TAG:-1.6.0}"
REPO=""
GITREF="HEAD"
OUTDIR=""
DO_SIGN=0
ALLOW_DIRTY=0
DO_PUSH=1
VARIANT="pipeline"          # pipeline | mlflow
DOCKERFILE="${REPO_ROOT}/Dockerfile"
BUILD_TARGET="runtime"
TF_REPO_OUTPUT="ecr_repository_url"
IMAGE_TITLE="mlops-pipeline"

log()  { printf '%s\n' "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }
usage() { sed -n '2,52p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# --- parse arguments -----------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --tag)          TAG="${2:?--tag needs a value}"; shift 2 ;;
    --repo)         REPO="${2:?--repo needs a value}"; shift 2 ;;
    --mlflow)       VARIANT="mlflow"; shift ;;
    --ref)          GITREF="${2:?--ref needs a value}"; shift 2 ;;
    --out)          OUTDIR="${2:?--out needs a value}"; shift 2 ;;
    --sign)         DO_SIGN=1; shift ;;
    --allow-dirty)  ALLOW_DIRTY=1; shift ;;
    --no-push)      DO_PUSH=0; shift ;;
    -h|--help)      usage; exit 0 ;;
    *)              die "unknown argument: $1 (try --help)" ;;
  esac
done

if [ "${VARIANT}" = "mlflow" ]; then
  DOCKERFILE="${REPO_ROOT}/docker/mlflow/Dockerfile"
  BUILD_TARGET=""                       # the MLflow Dockerfile is single-stage
  TF_REPO_OUTPUT="mlflow_server_ecr_repository_url"
  IMAGE_TITLE="mlflow-server"
fi

# --- prerequisites -------------------------------------------------------------
command -v git    >/dev/null 2>&1 || die "git not found on PATH"
command -v docker >/dev/null 2>&1 || die "docker not found on PATH"
command -v trivy  >/dev/null 2>&1 || die "trivy not found on PATH (needed for the SBOM)"
[ "${DO_SIGN}" -eq 1 ] && { command -v cosign >/dev/null 2>&1 || die "--sign needs cosign on PATH"; }

# --- resolve git provenance ----------------------------------------------------
COMMIT="$(git -C "${REPO_ROOT}" rev-parse "${GITREF}")" || die "cannot resolve git ref '${GITREF}'"
SHORT="$(git -C "${REPO_ROOT}" rev-parse --short "${GITREF}")"
# The tag(s) pointing exactly at this commit, if any (e.g. v1.6.0). Informational —
# the binding of record is the commit SHA, which is unambiguous.
GITTAG="$(git -C "${REPO_ROOT}" tag --points-at "${COMMIT}" | paste -sd, - || true)"

if [ "${ALLOW_DIRTY}" -eq 0 ] && ! git -C "${REPO_ROOT}" diff --quiet HEAD 2>/dev/null; then
  die "working tree is dirty — a released digest must map to a committed source state. Commit/stash, or pass --allow-dirty for a throwaway build."
fi

# --- resolve the ECR repository URL --------------------------------------------
if [ -z "${REPO}" ]; then
  command -v terraform >/dev/null 2>&1 || die "terraform not found and --repo not given"
  REPO="$(terraform -chdir="${TF_DIR}" output -raw "${TF_REPO_OUTPUT}" 2>/dev/null)" \
    || die "could not read Terraform output '${TF_REPO_OUTPUT}'. Has 'terraform apply' run in ${TF_DIR}? Or pass --repo."
  [ -n "${REPO}" ] || die "Terraform output '${TF_REPO_OUTPUT}' is empty"
fi
REGISTRY="${REPO%%/*}"                  # <account>.dkr.ecr.<region>.amazonaws.com
REPO_NAME="${REPO##*/}"                 # mlops-pipeline | mlflow-server
IMAGE_REF="${REPO}:${TAG}"

# --- output directory ----------------------------------------------------------
if [ -z "${OUTDIR}" ]; then
  OUTDIR="$(mktemp -d "${TMPDIR:-/tmp}/mlops-release.XXXXXX")"
fi
mkdir -p "${OUTDIR}"
SBOM_FILE="${OUTDIR}/${IMAGE_TITLE}-${TAG}.cdx.json"
PROV_JSON="${OUTDIR}/${IMAGE_TITLE}-${TAG}.provenance.json"

log "── Release provenance ──────────────────────────────────────────────"
log "  image      : ${IMAGE_REF}"
log "  git commit : ${COMMIT}${GITTAG:+  (tag: ${GITTAG})}"
log "  dockerfile : ${DOCKERFILE#${REPO_ROOT}/}"
log "  output     : ${OUTDIR}"
log "────────────────────────────────────────────────────────────────────"

# --- build ---------------------------------------------------------------------
log "Building ${IMAGE_REF} (linux/amd64) with provenance labels ..."
build_args=( --platform linux/amd64
             --build-arg "VCS_REF=${COMMIT}"
             --build-arg "BUILD_VERSION=${TAG}"
             -f "${DOCKERFILE}" -t "${IMAGE_REF}" )
[ -n "${BUILD_TARGET}" ] && build_args+=( --target "${BUILD_TARGET}" )
if [ "${VARIANT}" = "mlflow" ]; then
  # The MLflow image layers FROM the pipeline image; pass it through so the base is
  # explicit rather than whatever the Dockerfile's default ARG resolves to.
  build_args+=( --build-arg "BASE_IMAGE=${BASE_IMAGE:-mlops-pipeline:${TAG}}" )
fi
# --provenance/--sbom=false: we generate our OWN CycloneDX SBOM below rather than
# embed BuildKit's attestation manifests, which otherwise confuse image scanners
# (see the Sprint 8 PR 8 scan-evidence note) and are not what we pin to.
docker build --provenance=false --sbom=false "${build_args[@]}" "${REPO_ROOT}"

# --- push + capture the immutable digest ---------------------------------------
DIGEST=""
if [ "${DO_PUSH}" -eq 1 ]; then
  command -v aws >/dev/null 2>&1 || die "aws CLI not found (needed to log in to ECR); use --no-push for a local dry run"
  log "Logging in to ECR registry ${REGISTRY} ..."
  aws ecr get-login-password --region "$(printf '%s' "${REGISTRY}" | cut -d. -f4)" \
    | docker login --username AWS --password-stdin "${REGISTRY}" >/dev/null

  log "Pushing ${IMAGE_REF} ..."
  docker push "${IMAGE_REF}"

  # The immutable manifest digest as the registry recorded it. RepoDigests is
  # populated by the push; it is the sha256 a `pull by digest` resolves to.
  DIGEST="$(docker inspect --format '{{ range .RepoDigests }}{{ println . }}{{ end }}' "${IMAGE_REF}" \
            | grep -m1 "^${REPO}@" | cut -d@ -f2 || true)"
  [ -n "${DIGEST}" ] || die "could not read the pushed image's RepoDigest — push may have failed"

  # Cross-check against ECR's own record, so the digest is confirmed from TWO
  # sources (the local daemon and the registry API), not trusted blindly.
  ECR_DIGEST="$(aws ecr describe-images \
      --repository-name "${REPO_NAME}" \
      --image-ids "imageTag=${TAG}" \
      --region "$(printf '%s' "${REGISTRY}" | cut -d. -f4)" \
      --query 'imageDetails[0].imageDigest' --output text 2>/dev/null || true)"
  if [ -n "${ECR_DIGEST}" ] && [ "${ECR_DIGEST}" != "None" ] && [ "${ECR_DIGEST}" != "${DIGEST}" ]; then
    die "digest mismatch: local RepoDigest ${DIGEST} != ECR ${ECR_DIGEST} — refusing to record an inconsistent chain"
  fi
  log "Immutable digest: ${DIGEST}"
else
  log "--no-push: skipping push + registry digest capture (SBOM/provenance from the local image)."
fi

# --- SBOM (CycloneDX), from the exact artifact ---------------------------------
# By digest when pushed (the artifact that will be pulled); by tag for a local dry
# run. Either way it is the image just built, not a re-resolve.
SBOM_TARGET="${IMAGE_REF}"
[ -n "${DIGEST}" ] && SBOM_TARGET="${REPO}@${DIGEST}"
log "Generating CycloneDX SBOM for ${SBOM_TARGET} ..."
trivy image --format cyclonedx --scanners license --skip-version-check \
  --output "${SBOM_FILE}" "${SBOM_TARGET}"
SBOM_COMPONENTS="$(grep -c '"bom-ref"' "${SBOM_FILE}" || echo 0)"
log "SBOM written: ${SBOM_FILE} (~${SBOM_COMPONENTS} components)"

# --- provenance record ---------------------------------------------------------
# Small, human- and machine-readable. The chain of record is commit → tag → digest.
cat > "${PROV_JSON}" <<EOF
{
  "image": "${IMAGE_TITLE}",
  "repository": "${REPO}",
  "git": { "commit": "${COMMIT}", "tag": "${GITTAG}" },
  "image_tag": "${TAG}",
  "image_digest": "${DIGEST}",
  "sbom": { "format": "CycloneDX", "tool": "trivy", "file": "$(basename "${SBOM_FILE}")" },
  "signed": $( [ "${DO_SIGN}" -eq 1 ] && echo true || echo false )
}
EOF

log ""
log "── Provenance chain ────────────────────────────────────────────────"
log "  git commit : ${COMMIT}"
log "  git tag    : ${GITTAG:-<none pointing at HEAD>}"
log "  image tag  : ${REPO}:${TAG}"
log "  digest     : ${DIGEST:-<not pushed>}"
log "  SBOM       : ${SBOM_FILE}"
log "  record     : ${PROV_JSON}"
log "────────────────────────────────────────────────────────────────────"

# --- optional cosign signing (by digest) ---------------------------------------
if [ "${DO_SIGN}" -eq 1 ]; then
  [ -n "${DIGEST}" ] || die "--sign requires a pushed image (a signature is attached to the registry digest)"
  log "Signing ${REPO}@${DIGEST} with cosign (keyless OIDC) ..."
  # Keyless: identity comes from an OIDC token; the signature + certificate are
  # recorded in the Rekor transparency log. No long-lived key to store or leak.
  COSIGN_EXPERIMENTAL=1 cosign sign --yes "${REPO}@${DIGEST}"
  log "Signed. Verify with:"
  log "  cosign verify ${REPO}@${DIGEST} \\"
  log "    --certificate-identity-regexp '.*' --certificate-oidc-issuer-regexp '.*'"
fi

# --- next steps ----------------------------------------------------------------
log ""
log "Deploy pinned to this exact digest (preferred, immutable):"
log "  IMAGE_DIGEST=${DIGEST:-<digest>} scripts/render-cloud-manifests.sh --apply"
log "Then verify the running workload matches:"
log "  scripts/verify-deployed-digest.sh --expect ${DIGEST:-<digest>}"

# Emit the digest on STDOUT (only) so callers can capture it:
#   digest=$(scripts/release-image.sh --tag 1.6.0 | tail -1)
printf '%s\n' "${DIGEST}"
