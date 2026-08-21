#!/usr/bin/env bash
# verify-deployed-digest.sh — prove the workload RUNNING on the cluster is the exact
# image the release recorded, by comparing the live container `imageID` (the sha256
# the kubelet actually pulled and ran) against the EXPECTED immutable digest.
#
# This closes the last link of the supply-chain contract at RUNTIME:
#
#     git commit ─▶ image tag ─▶ sha256 digest ─▶ [what is ACTUALLY running]
#
# It makes deploy-by-tag safe: even though the AWS overlay pins an IMMUTABLE ECR tag
# (so a tag can never be repointed), this check confirms — from the cluster's own
# reported state, not from the manifest we hoped it applied — that the pulled digest
# equals the one release-image.sh captured. If you deploy BY DIGEST (the preferred
# path, IMAGE_DIGEST=… scripts/render-cloud-manifests.sh), this is the independent
# confirmation that it took effect. Either way the digest syntax is never FORCED —
# the verification is what provides the guarantee.
#
# Design of record: docs/decisions/ADR-036-sbom-and-image-provenance.md and
# docs/supply-chain-provenance.md.
#
# Usage:
#   scripts/verify-deployed-digest.sh --expect sha256:<hex>
#   scripts/verify-deployed-digest.sh --record <provenance.json>
#     --expect DIGEST   expected digest (sha256:<hex>, or bare <hex>)
#     --record FILE     read the expected digest from a release-image.sh record
#                       (its "image_digest" field) instead of --expect
#     --namespace NS    Kubernetes namespace         (default: mlops)
#     --selector SEL    pod label selector           (default: app.kubernetes.io/name=mlops-pipeline)
#     --repo SUBSTR     only check containers whose image repo contains SUBSTR
#                       (default: mlops-pipeline) — skips unrelated sidecars
#     -h, --help        show this help
#
# Exit status: 0 iff EVERY matching container (init + main) reports the expected
# digest; non-zero on any mismatch, or if no matching container is found.
#
# Requirements: bash, kubectl (context pointed at the target cluster), and either
# `jq` or a shell (the parsing is pure bash). No AWS calls.
set -euo pipefail

EXPECT=""
RECORD=""
NAMESPACE="mlops"
SELECTOR="app.kubernetes.io/name=mlops-pipeline"
REPO_SUBSTR="mlops-pipeline"

die()  { printf 'error: %s\n' "$*" >&2; exit 1; }
usage() { sed -n '2,37p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --expect)     EXPECT="${2:?--expect needs a value}"; shift 2 ;;
    --record)     RECORD="${2:?--record needs a value}"; shift 2 ;;
    --namespace)  NAMESPACE="${2:?--namespace needs a value}"; shift 2 ;;
    --selector)   SELECTOR="${2:?--selector needs a value}"; shift 2 ;;
    --repo)       REPO_SUBSTR="${2:?--repo needs a value}"; shift 2 ;;
    -h|--help)    usage; exit 0 ;;
    *)            die "unknown argument: $1 (try --help)" ;;
  esac
done

command -v kubectl >/dev/null 2>&1 || die "kubectl not found on PATH"

# --- resolve the expected digest -----------------------------------------------
if [ -z "${EXPECT}" ] && [ -n "${RECORD}" ]; then
  [ -f "${RECORD}" ] || die "record file not found: ${RECORD}"
  if command -v jq >/dev/null 2>&1; then
    EXPECT="$(jq -r '.image_digest // empty' "${RECORD}")"
  else
    EXPECT="$(grep -o '"image_digest"[[:space:]]*:[[:space:]]*"[^"]*"' "${RECORD}" \
              | sed 's/.*"\(sha256:[^"]*\)".*/\1/')"
  fi
  [ -n "${EXPECT}" ] || die "no image_digest found in ${RECORD}"
fi
[ -n "${EXPECT}" ] || die "need --expect <digest> or --record <file>"
# Normalise: accept a bare hex digest too.
case "${EXPECT}" in sha256:*) ;; *) EXPECT="sha256:${EXPECT}" ;; esac

echo "Expecting digest: ${EXPECT}"
echo "Namespace/selector: ${NAMESPACE} / ${SELECTOR}"

# --- gather the live container image IDs ---------------------------------------
pods="$(kubectl -n "${NAMESPACE}" get pods -l "${SELECTOR}" \
          -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null || true)"
[ -n "${pods}" ] || die "no pods found in ${NAMESPACE} matching ${SELECTOR} (is the workload deployed?)"

checked=0
mismatch=0
while IFS= read -r pod; do
  [ -n "${pod}" ] || continue
  # Both init and regular containers — the pipeline image is used by all three
  # (fetch-dataset, wait-for-mlflow, pipeline), so all must match.
  lines="$(kubectl -n "${NAMESPACE}" get pod "${pod}" -o jsonpath='
    {range .status.initContainerStatuses[*]}{.name}{"|"}{.imageID}{"\n"}{end}
    {range .status.containerStatuses[*]}{.name}{"|"}{.imageID}{"\n"}{end}' 2>/dev/null || true)"
  while IFS='|' read -r cname imageid; do
    cname="$(printf '%s' "${cname}" | tr -d '[:space:]')"
    [ -n "${cname}" ] || continue
    # Only containers pulled from the repo we are attesting.
    case "${imageid}" in *"${REPO_SUBSTR}"*) ;; *) continue ;; esac
    got="sha256:${imageid##*@sha256:}"
    checked=$((checked + 1))
    if [ "${got}" = "${EXPECT}" ]; then
      echo "  OK    ${pod}/${cname}: ${got}"
    else
      echo "  FAIL  ${pod}/${cname}: ${got} (expected ${EXPECT})"
      mismatch=$((mismatch + 1))
    fi
  done <<< "${lines}"
done <<< "${pods}"

echo "----------------------------------------------------------------"
[ "${checked}" -gt 0 ] || die "no container matched repo substring '${REPO_SUBSTR}' — nothing verified"
if [ "${mismatch}" -eq 0 ]; then
  echo "PASS: all ${checked} container(s) run the expected digest ${EXPECT}"
  exit 0
fi
echo "FAIL: ${mismatch}/${checked} container(s) run an UNEXPECTED digest"
exit 1
