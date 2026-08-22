#!/usr/bin/env bash
# render-cloud-manifests.sh — render the account-neutral AWS overlay into a
# concrete, deployable manifest from live Terraform outputs, WITHOUT mutating any
# tracked file.
#
# Flow (see docs/cloud-operations.md, ADR-018):
#
#     terraform outputs
#            │
#            ▼
#     render-cloud-manifests.sh   ──▶  generated AWS overlay (temp copy of k8s/)
#            │
#            ▼
#     kustomize build             ──▶  rendered manifest (STDOUT)
#            │
#            ▼
#     kubectl apply -f -
#
# The committed k8s/overlays/aws stays account-neutral (the `000000000000`
# placeholders and `us-east-1` defaults remain in git). The operator never edits a
# tracked file and never runs `kustomize edit` against the repo, so there is nothing
# to `git checkout --` on teardown. Every account-specific value — the ECR registry
# host, the dataset and MLflow-artifact S3 buckets, and the region — is read straight
# from `terraform output`.
#
# What it substitutes, and the Terraform output each value comes from:
#   * ml-pipeline image        → ecr_repository_url
#   * mlflow-server image      → mlflow_server_ecr_repository_url
#   * DATASET_S3_URI           → dataset_s3_uri
#   * MLFLOW_ARTIFACTS_DESTINATION → s3://<mlflow_artifact_bucket_name>/artifacts
#   * AWS_DEFAULT_REGION       → aws_region
#
# Usage:
#   scripts/render-cloud-manifests.sh [-o OUTFILE] [--apply] [-h]
#
#   (default)      render to STDOUT; all diagnostics go to STDERR, so you can pipe:
#                    scripts/render-cloud-manifests.sh | kubectl apply -f -
#   -o OUTFILE     write the rendered manifest to OUTFILE instead of STDOUT
#   --apply        pipe the rendered manifest straight to `kubectl apply -f -`
#   -h, --help     show this help and exit
#
# Environment overrides (rarely needed; defaults match the committed image tags):
#   IMAGE_TAG          workload image tag         (default: 1.7.0)
#   MLFLOW_IMAGE_TAG   MLflow server image tag    (default: 0.1.0)
#   IMAGE_DIGEST       pin the workload image BY DIGEST (sha256:…) instead of by tag
#   MLFLOW_IMAGE_DIGEST  pin the MLflow image BY DIGEST (sha256:…) instead of by tag
#   TF_DIR             Terraform root directory   (default: <repo>/terraform)
#
# Digest pinning (PREFERRED, opt-in — Sprint 8, PR 9 / ADR-036): set IMAGE_DIGEST to
# the sha256 that scripts/release-image.sh captured at push time and the overlay
# renders `newName@sha256:…` (Kustomize `digest:`), so the deploy is bound to the
# exact immutable artifact rather than a tag. It is OPT-IN so the default flow stays
# simple (the ECR tag is already IMMUTABLE, ADR-021, so tag-deploy is reproducible);
# whether you pin by tag or by digest, scripts/verify-deployed-digest.sh confirms the
# running imageID matches — the verification, not the syntax, is the guarantee.
#
# Requirements: bash, terraform (with applied state), and either `kustomize` or a
# `kubectl` new enough to provide `kubectl kustomize`. No AWS calls are made by this
# script itself — it only reads Terraform state.
set -euo pipefail

# --- locate the repo root (this script lives in <repo>/scripts) ----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TF_DIR="${TF_DIR:-${REPO_ROOT}/terraform}"
OVERLAY_REL="k8s/overlays/aws"

IMAGE_TAG="${IMAGE_TAG:-1.7.0}"
MLFLOW_IMAGE_TAG="${MLFLOW_IMAGE_TAG:-0.1.0}"

OUTFILE=""
DO_APPLY=0

log()  { printf '%s\n' "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

usage() { sed -n '2,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# --- parse arguments -----------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    -o|--out)   OUTFILE="${2:-}"; [ -n "${OUTFILE}" ] || die "-o needs a file path"; shift 2 ;;
    --apply)    DO_APPLY=1; shift ;;
    -h|--help)  usage; exit 0 ;;
    *)          die "unknown argument: $1 (try --help)" ;;
  esac
done

# --- prerequisites -------------------------------------------------------------
command -v terraform >/dev/null 2>&1 || die "terraform not found on PATH"
[ -d "${TF_DIR}" ] || die "Terraform directory not found: ${TF_DIR}"

# Pick a kustomize builder: prefer the standalone binary, else `kubectl kustomize`.
if command -v kustomize >/dev/null 2>&1; then
  kbuild() { kustomize build "$1"; }
elif command -v kubectl >/dev/null 2>&1 && kubectl kustomize --help >/dev/null 2>&1; then
  kbuild() { kubectl kustomize "$1"; }
else
  die "need either 'kustomize' or a 'kubectl' that supports 'kubectl kustomize'"
fi

if [ "${DO_APPLY}" -eq 1 ]; then
  command -v kubectl >/dev/null 2>&1 || die "--apply needs kubectl on PATH"
fi

# --- read the account-specific values from Terraform ---------------------------
# `terraform output -raw` prints the plain value even for outputs marked sensitive.
tf_out() {
  local name="$1" val
  if ! val="$(terraform -chdir="${TF_DIR}" output -raw "${name}" 2>/dev/null)"; then
    die "could not read Terraform output '${name}'. Has 'terraform apply' run in ${TF_DIR}?"
  fi
  [ -n "${val}" ] || die "Terraform output '${name}' is empty"
  printf '%s' "${val}"
}

log "Reading Terraform outputs from ${TF_DIR} ..."
ECR_URL="$(tf_out ecr_repository_url)"
MLFLOW_ECR_URL="$(tf_out mlflow_server_ecr_repository_url)"
DATASET_S3_URI="$(tf_out dataset_s3_uri)"
MLFLOW_ARTIFACT_BUCKET="$(tf_out mlflow_artifact_bucket_name)"
AWS_REGION="$(tf_out aws_region)"

# --- build an isolated, mutable copy of the manifests --------------------------
WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/mlops-aws-render.XXXXXX")"
cleanup() { rm -rf "${WORKDIR}"; }
trap cleanup EXIT

cp -R "${REPO_ROOT}/k8s" "${WORKDIR}/k8s"
OVERLAY="${WORKDIR}/${OVERLAY_REL}"
[ -f "${OVERLAY}/kustomization.yaml" ] || die "AWS overlay not found in copy: ${OVERLAY}"

# Portable in-place edit (works with both GNU and BSD sed): write to a temp file
# then move it back. `|` is used as the sed delimiter because the values contain `/`.
subst() {  # subst FILE SED_EXPR
  local file="$1" expr="$2"
  sed -E "${expr}" "${file}" > "${file}.tmp"
  mv "${file}.tmp" "${file}"
}

# 1. Images — rewrite the placeholder registry host + tag in the overlay's
#    `images:` transformer (matched by the trailing repository name so it is robust
#    to the account/region in the committed placeholder).
subst "${OVERLAY}/kustomization.yaml" "s|newName: .*/mlops-pipeline$|newName: ${ECR_URL}|"
subst "${OVERLAY}/kustomization.yaml" "s|newName: .*/mlflow-server$|newName: ${MLFLOW_ECR_URL}|"

# Pin each image either BY DIGEST (if the matching *_DIGEST env is set — preferred,
# immutable-by-construction) or BY TAG (the default; the committed literals equal
# the script defaults, so retag is a no-op unless IMAGE_TAG/MLFLOW_IMAGE_TAG were
# overridden). Digest pinning replaces the committed `newTag: "<literal>"` line with
# `digest: "sha256:…"`, which Kustomize renders as `newName@sha256:…`.
#   pin_image  COMMITTED_TAG_LITERAL  TAG_OVERRIDE  DIGEST_OVERRIDE
pin_image() {
  local literal="$1" tag="$2" digest="$3"
  if [ -n "${digest}" ]; then
    # STRICT validation: exactly `sha256:` + 64 lowercase hex. A loose check (e.g.
    # `sha256:[0-9a-f]*`, which only constrains the first char) would let a value
    # containing the sed delimiter `|` or `;` break out of the `s|…|…|` below and
    # rewrite ANY line of the rendered kustomization.yaml — including the registry
    # pointer — so the whole 64-char body must be validated before interpolation.
    [[ "${digest}" =~ ^sha256:[0-9a-f]{64}$ ]] \
      || die "invalid digest '${digest}' — expected sha256:<64 lowercase hex> (from scripts/release-image.sh)"
    # `newTag: "<literal>"` (with its surrounding indentation) becomes `digest: "…"`.
    subst "${OVERLAY}/kustomization.yaml" "s|newTag: \"${literal}\"|digest: \"${digest}\"|"
  else
    subst "${OVERLAY}/kustomization.yaml" "s|newTag: \"${literal}\"|newTag: \"${tag}\"|"
  fi
}
pin_image "1.7.0" "${IMAGE_TAG}"        "${IMAGE_DIGEST:-}"
pin_image "0.1.0" "${MLFLOW_IMAGE_TAG}" "${MLFLOW_IMAGE_DIGEST:-}"

# 2. Runtime dataset source (job-cloud.yaml) — the whole DATASET_S3_URI value.
subst "${OVERLAY}/job-cloud.yaml" "s|value: \"s3://[^\"]*datasets[^\"]*\"|value: \"${DATASET_S3_URI}\"|"

# 3. MLflow artifact store (mlflow-cloud.yaml).
subst "${OVERLAY}/mlflow-cloud.yaml" "s|value: \"s3://[^\"]*mlflow-artifacts[^\"]*\"|value: \"s3://${MLFLOW_ARTIFACT_BUCKET}/artifacts\"|"

# 4. Region (both patch files) — replace the committed us-east-1 default.
subst "${OVERLAY}/job-cloud.yaml"    "s|value: \"us-east-1\"|value: \"${AWS_REGION}\"|"
subst "${OVERLAY}/mlflow-cloud.yaml" "s|value: \"us-east-1\"|value: \"${AWS_REGION}\"|"

# --- render --------------------------------------------------------------------
log "Rendering ${OVERLAY_REL} with account-specific values ..."
RENDERED="${WORKDIR}/rendered.yaml"
kbuild "${OVERLAY}" > "${RENDERED}"

# Safety net: the placeholder account id must not survive into a deployable manifest.
if grep -q '000000000000' "${RENDERED}"; then
  die "placeholder account id '000000000000' still present after render — substitution missed something; refusing to emit"
fi

log "Rendered OK (registry ${ECR_URL%%/*}, region ${AWS_REGION})."

# --- deliver -------------------------------------------------------------------
if [ "${DO_APPLY}" -eq 1 ]; then
  log "Applying with kubectl ..."
  kubectl apply -f "${RENDERED}"
elif [ -n "${OUTFILE}" ]; then
  cp "${RENDERED}" "${OUTFILE}"
  log "Wrote rendered manifest to ${OUTFILE}"
  log "Next: kubectl apply -f ${OUTFILE}"
else
  cat "${RENDERED}"
fi
