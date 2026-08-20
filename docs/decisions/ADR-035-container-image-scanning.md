# ADR-035: Container-image vulnerability scanning as a CI gate (Sprint 8, PR 8)

- **Status:** Accepted (design; Trivy image scan + exception mechanism + docs added to CI, no image published / no registry push in this PR)
- **Date:** 2026-08-20
- **Deciders:** Asad Hanif
- **Related:**
  [ADR-005 (Containerization strategy — the multi-stage image being scanned)](ADR-005-containerization-strategy.md),
  [ADR-019 (Terraform CI validation — the Trivy IaC scan this reuses the pattern of)](ADR-019-terraform-ci-validation.md),
  [ADR-021 (Terraform-managed ECR — registry-side scan-on-push, the complementary layer)](ADR-021-terraform-managed-ecr.md),
  [ADR-026 (In-cluster MLflow platform — the second image scanned)](ADR-026-in-cluster-mlflow-platform.md),
  [ADR-012 (Kubernetes manifest validation — sibling static CI gate)](ADR-012-kubernetes-manifest-validation.md),
  [`docs/container-image-scanning.md`](../container-image-scanning.md),
  [`docs/ci-cd.md`](../ci-cd.md),
  [`.trivyignore.yaml`](../../.trivyignore.yaml),
  [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)

> **Scope.** This ADR ratifies **container-image vulnerability scanning** as a CI
> gate: the two images the project ships — the `mlops-pipeline` runtime image
> ([`Dockerfile`](../../Dockerfile)) and the `mlflow-server` image layered on it
> ([`docker/mlflow/Dockerfile`](../../docker/mlflow/Dockerfile)) — are scanned with
> Trivy on every push/PR, with a severity + fixability policy, a time-boxed exception
> mechanism, and a published report artifact. It **adds a scan + exception file +
> docs**; it does **not** publish an image, push to a registry, sign images, or emit
> an SBOM (those remain roadmap v3 supply-chain items). It reuses — does not
> re-decide — the pinned, checksum-verified Trivy install pattern from
> [ADR-019](ADR-019-terraform-ci-validation.md).

## Context

CI already builds the production image and proves it assembles and runs (the `docker`
job: non-root UID, core imports, DVC entrypoint). What it did **not** do is ask
whether that image ships **known-vulnerable** OS or Python packages. The base is
`python:3.12-slim-bookworm` (Debian 12) plus a pip-installed scientific stack
(scikit-learn, mlflow, dvc, boto3, …); the `mlflow-server` image adds one wheel
(`psycopg2-binary`). Both are assembled from upstream packages that accumulate CVEs
over time. Shipping them unscanned means a HIGH/CRITICAL vulnerability in, say, a
transitive library or an OS package could reach a cluster unnoticed.

Three facts shape the decision:

1. **Trivy is already the project's scanner.** The `terraform-validate` job runs
   `trivy config` (pinned `0.74.0`, installed as a checksum-verified static binary)
   for IaC misconfiguration, and `terraform/.trivyignore` is an established,
   ADR-referenced triage record. Adding **image** scanning with the same tool and the
   same install/exception discipline is the aligned choice — no new tool, no new
   pattern to learn or maintain.
2. **The roadmap already earmarks this.** [`docs/ci-cd.md`](../ci-cd.md) § "Future CD
   roadmap" lists "Supply-chain hardening (v3): Vulnerability scanning (e.g.
   Trivy/Grype), SBOM generation, and image signing (cosign) as release gates." This
   PR delivers the **scanning** slice of that item; SBOM and signing stay on the
   roadmap.
3. **The registry already scans on push — but that is a *different*, later layer.**
   ECR is configured with `scan_on_push` ([ADR-021](ADR-021-terraform-managed-ecr.md)).
   That scan happens **after** an operator publishes an image to AWS, is
   AWS-account-coupled, and is not part of the per-PR gate. It is a valuable *second*
   line of defence at the registry, not a substitute for catching a vulnerable image
   **before** it is ever built into a release — and it cannot run in ordinary,
   credential-free PR CI. The two are complementary.

The risk to avoid is a **meaningless checkbox gate**: a scan that either fails on
everything (so teams reflexively `|| true` it or blanket-ignore a whole severity) or
passes on everything (so it proves nothing). The policy below is designed so a red
gate is always **actionable** and a green gate is **meaningful**.

## Decision

**Scan both shipped images with Trivy in the `docker` job, and fail CI on *fixable*
HIGH/CRITICAL vulnerabilities**, with non-fixable findings reported (not gated) and a
specific, time-boxed exception mechanism for the rare justified case.

### 1. What is scanned — the actual built images, locally

The scan runs in the existing `docker` job, immediately after the image is built and
loaded into the local daemon. It scans `ml-pipeline:ci` (the real production image
this PR's CI just built) and `mlflow-server:ci` (built here FROM that image). The
images are **never pulled from a registry**, so the job stays credential-free and
**AWS-independent** — satisfying the "normal PR CI holds no cloud credentials"
invariant this project enforces everywhere.

**Both images are scanned** because the `mlflow-server` image adds a package
(`psycopg2-binary`) the pipeline image does not carry; scanning it proves that one
extra layer. It is a cheap, single-wheel build on the same base, so the added scope
is minimal — the "keep scope manageable" bar the task set.

### 2. Severity policy

**HIGH and CRITICAL are the gating severities.** MEDIUM/LOW are not gated (they would
drown the signal), consistent with the IaC scan, which also gates on `CRITICAL,HIGH`.

### 3. Fixable vs non-fixable treatment — the core of the policy

- **Fixable HIGH/CRITICAL** (a patched package version exists) → **fail the build**
  (`--ignore-unfixed --exit-code 1`). These are genuinely actionable: rebuild on a
  patched base image, or bump the offending package. A red gate here always has a
  concrete fix.
- **Non-fixable HIGH/CRITICAL** (no upstream fix yet) → **reported, not gated**. There
  is *no action* that resolves them, so gating would force either a permanent red
  build or a blanket mute — the checkbox anti-pattern. They are **surfaced in every
  run** (the report pass below) and **auto-promote to the gate the moment a fix ships
  upstream**, because `--ignore-unfixed` stops ignoring them as soon as Trivy's DB
  records a fixed version. This is *treatment*, not suppression.

This is explicitly **not** a global ignore of HIGH/CRITICAL (the task's requirement 7):
fixable ones hard-fail; non-fixable ones remain visible and tracked.

### 4. Two passes per image — gate + report

- **Report (non-gating):** every HIGH/CRITICAL, fixable or not, printed to the log and
  written as both a human table and machine-readable JSON.
- **Gate (blocking):** `--ignore-unfixed`, non-zero exit on any remaining
  (fixable) HIGH/CRITICAL.

### 5. Exception mechanism — [`​.trivyignore.yaml`](../../.trivyignore.yaml)

For the rare case where a **fixable** HIGH/CRITICAL cannot be remediated immediately
(the fix needs a major upgrade under regression test, or the CVE is a confirmed
false-positive / unreachable in our usage), a **specific** exception is recorded in
`.trivyignore.yaml`. Trivy's structured YAML ignore format requires, per entry:

- **`id`** — the exact vulnerability identifier (CVE-…/GHSA-…),
- **`statement`** — the rationale,
- **`expired_at`** — the review/expiry date; Trivy **automatically** stops honouring
  the entry after it, so a stale exception re-breaks CI rather than hiding risk
  forever.

There is **no blanket severity mute and no un-scoped ignore** — the same discipline
`terraform/.trivyignore` follows. The file ships with **zero active exceptions**.

### 6. Artifact

The report tables + JSON are uploaded as the `trivy-image-reports` build artifact with
`if: always()`, so a *failing* run — exactly when triage is needed — has the full
findings (including the non-gating, non-fixable ones) one click away.

### 7. Secure, credential-free integration

- Trivy is installed as a **pinned (`TRIVY_VERSION`), checksum-verified** static binary
  — the same supply-chain pattern as kustomize/kubeconform/promtool/the IaC Trivy.
  `TRIVY_VERSION` is pinned **once** at the workflow level and shared by the IaC and
  image scans so they cannot drift.
- Job permissions stay `contents: read`; **no** registry login, **no** AWS identity,
  **no** `packages:`/`security-events:` write scope. The image lives only in the local
  daemon.
- The vuln-DB fetch is the **only** network dependency — the same class as the IaC
  scan's DB and kubeconform's schema fetch.

## Findings from the initial scan (2026-08-20)

The scan was **run** on both images (local Docker daemon, the CI method); full record in
[`docs/proof/sprint-08-image-scan-evidence.md`](../proof/sprint-08-image-scan-evidence.md).
Baseline: **3 fixable HIGH** (identical on both images — the `psycopg2-binary` layer
adds none) plus **70 non-fixable** HIGH/CRITICAL (18 CRITICAL + 52 HIGH), **all Debian
base-image OS packages** with no published fix (so `apt-get upgrade` would not clear
them — they are correctly reported, not gated).

Of the 3 fixable HIGH:

- **`cryptography` 49.0.0 → 50.0.0 (CVE-2026-69247)** — a real installed dependency;
  **remediated** by a targeted security-floor bump in the [`Dockerfile`](../../Dockerfile).
  The rebuilt image's import smoke-test and `dvc --version` pass, so the bump is safe;
  the finding is gone.
- **`msgpack` (GHSA-6v7p-g79w-8964)** and **`setuptools` (CVE-2025-47273)** — the flagged
  copies are **pip's *vendored*** packages (`pip/_vendor/…`), not independently
  upgradable, not on the runtime execution path (`dvc repro` never invokes pip), and not
  removable by dropping pip (the `mlflow-server` image `pip install`s on top of this
  base). Recorded as **two time-boxed exceptions** in
  [`.trivyignore.yaml`](../../.trivyignore.yaml) (`expired_at: 2026-11-18`).

**After the fix + exceptions, both images pass the gate (exit 0)** while all 70
non-fixable findings remain reported — the intended, meaningful-not-checkbox outcome.

## Alternatives considered

- **Grype (instead of Trivy).** Comparable capability, but Trivy is already vetted,
  pinned, and documented in this repo for IaC. A second scanner is redundant surface.
  *Rejected* for tool sprawl.
- **Rely on ECR `scan_on_push` only.** That is a real layer (ADR-021) but fires
  **after** publish, is AWS-coupled, and is absent from per-PR CI. It cannot catch a
  vulnerable image before build/merge and cannot run credential-free. *Kept as a
  complementary second layer, not a replacement.*
- **Gate on ALL severities (incl. MEDIUM/LOW).** Would bury the signal and train
  reviewers to ignore the gate. *Rejected*; HIGH/CRITICAL matches the IaC scan.
- **Gate on non-fixable findings too.** Produces an un-actionable red build that forces
  blanket muting — the exact checkbox this design avoids. *Rejected* in favour of
  report-but-don't-gate + auto-promotion when a fix appears.
- **`aquasecurity/trivy-action` (marketplace action).** Convenient, but the repo
  deliberately installs security tools as pinned, checksum-verified binaries after a
  past experience of transitive setup-action pins breaking (ADR-019). *Rejected* for
  consistency and supply-chain control.
- **A separate `image-scan` job.** Cleaner separated status, but each job gets a fresh
  runner, so the built image would have to be exported/imported between jobs (extra
  minutes + an artifact round-trip). *Rejected*; scanning in the job that owns the
  image is simpler and the image is already loaded there. (The job keeps its existing
  name so branch-protection required-checks are unaffected.)

## Consequences

**Positive**

- A vulnerable image is caught **before** it can be published or deployed, on every PR,
  with a fix that is always actionable when the gate is red.
- One scanner, one pinned version, one exception discipline across IaC **and** images.
- The published JSON artifact makes findings auditable and feeds future triage/SBOM
  work.
- Ordinary PR CI remains fully **credential-free and AWS-independent**.

**Negative / limitations (stated honestly)**

- **The gate is time-varying by design.** Trivy pulls a fresh vuln DB each run, so a
  commit that is green today can go red on a later re-run when a **new** fixable
  CVE is published for an already-shipped package. That is correct security behaviour
  (the world learned something new), but it means a red `docker` job is not always
  caused by the diff under review — the report artifact names exactly which package
  and CVE, and the fix is a base/dependency bump.
- **Scanned versions reflect the build, and runtime deps are not yet pinned.**
  `requirements.txt` pins by name, so pip resolves the latest compatible versions at
  build time; the scan reflects whatever was resolved. Digest/version pinning for
  byte-reproducible builds is the separate ADR-005 roadmap item — until then the exact
  scanned set can shift build-to-build.
- **This scans images, not the running cluster or the registry.** It complements, and
  does not replace, ECR scan-on-push (registry layer) and does not add runtime
  admission scanning.
- **No SBOM and no image signing yet** — deliberately out of scope; roadmap v3.
- **A DB-fetch flake fails the job.** Same failure mode as the other network-dependent
  gates; re-run to recover.

**Follow-ups (roadmap, not this PR)**

- **Clear the two pip-vendored exceptions** (`msgpack` / `setuptools` in `pip/_vendor`)
  at their `2026-11-18` review: pick up a base image / pip release whose vendored deps
  are patched, or restructure the `mlflow-server` image so the base no longer needs pip
  (e.g. install `psycopg2-binary` in a builder and copy the wheel), after which pip can
  be dropped from the runtime image entirely — removing both findings at the source.
- SBOM generation (`trivy image --format cyclonedx/spdx`) and image signing (cosign) as
  release gates (roadmap v3).
- Pin the base image and runtime dependencies by digest (ADR-005) so the scanned set is
  reproducible.
- On the publish step (roadmap v3), optionally emit SARIF to GitHub code scanning
  (needs `security-events: write` — a scoped, still-AWS-free permission).
