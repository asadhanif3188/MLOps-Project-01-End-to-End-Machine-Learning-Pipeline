# Container-image vulnerability scanning

**Sprint 8, PR 8.** Design of record: [ADR-035](decisions/ADR-035-container-image-scanning.md).

This document is the operator-facing guide to the container-image vulnerability
scan: **what** is scanned, the **severity + fixability policy**, how **exceptions**
work, and how to **run it locally**. It is the image-layer counterpart to the
Terraform IaC scan (`trivy config`, [ADR-019](decisions/ADR-019-terraform-ci-validation.md))
— the same tool, the same pinned/checksum-verified install, the same "justified
suppressions only" discipline.

> **Where it runs.** In the `docker` job of [`.github/workflows/ci.yml`](../.github/workflows/ci.yml),
> on every push to `main` and every pull request, right after the image is built and
> loaded into the local daemon. The images are **never pulled from a registry**, so
> the scan is **credential-free and AWS-independent** — normal PR CI carries no cloud
> identity.

---

## 1. Images scanned

| Image | Built from | Why it is scanned |
|---|---|---|
| `mlops-pipeline` (`ml-pipeline:ci`) | [`Dockerfile`](../Dockerfile) `runtime` target — `python:3.12-slim-bookworm` + the pip stack | The shippable production image (the pipeline `Job`). |
| `mlflow-server` (`mlflow-server:ci`) | [`docker/mlflow/Dockerfile`](../docker/mlflow/Dockerfile) — the pipeline image **+ `psycopg2-binary`** | The in-cluster MLflow platform image ([ADR-026](decisions/ADR-026-in-cluster-mlflow-platform.md)). Scanning it proves the **one** package the pipeline image does not carry; everything else is inherited from the shared base, so it is a cheap single-layer build. |

Both are the **actual built images**, scanned for **OS packages and language
(Python) libraries** (`--pkg-types os,library`).

---

## 2. Severity + fixability policy

**HIGH and CRITICAL** are the gating severities (MEDIUM/LOW are not gated — the same
threshold as the IaC scan). Within those, findings are treated by **fixability**:

| Finding | Fix available upstream? | CI behaviour |
|---|---|---|
| **Fixable** HIGH/CRITICAL | yes (a patched version exists) | **FAILS the build** — actionable: rebuild on a patched base, or bump the package. |
| **Non-fixable** HIGH/CRITICAL | no fix yet | **Reported, not gated** — surfaced in every run's report + artifact; **auto-promotes to the gate** the moment a fix ships upstream (`--ignore-unfixed` stops ignoring it). |
| MEDIUM / LOW | — | Not scanned into the gate (noise control). |

This is deliberately **not** a global ignore of HIGH/CRITICAL. Fixable ones hard-fail;
non-fixable ones stay **visible and tracked** rather than muted — there is simply no
action that resolves them, so blocking on them would only force a blanket suppression
(the "meaningless checkbox" this gate exists to avoid).

Each image is scanned **twice**:

1. **Report** (non-gating) — every HIGH/CRITICAL, fixable or not, as a table + JSON.
2. **Gate** (`--ignore-unfixed --exit-code 1`) — fails only on **fixable** HIGH/CRITICAL.

The reports are uploaded as the **`trivy-image-reports`** artifact with `always()`, so
a failing run has the full findings one click away.

---

## 3. Exceptions — [`.trivyignore.yaml`](../.trivyignore.yaml)

For the rare case where a **fixable** HIGH/CRITICAL cannot be remediated immediately
(the fix needs a major upgrade under regression test, or the CVE is a confirmed
false-positive / not reachable in how we use the package), record a **specific,
time-boxed** exception. Trivy's structured YAML format requires, per entry:

```yaml
vulnerabilities:
  - id: CVE-0000-00000        # the exact CVE / GHSA id — never a whole severity
    statement: >-
      Why this fixable finding is accepted right now (the concrete reason the fix
      cannot land immediately, or why it is not exploitable in our usage).
    expired_at: 2026-11-01     # review/expiry date — Trivy AUTO-EXPIRES the entry
```

Rules:

- **No blanket severity mute, no un-scoped ignore.** Every entry is one CVE id with a
  rationale and an expiry — the same discipline `terraform/.trivyignore` follows.
- **`expired_at` is enforced by Trivy:** after that date the entry is ignored and the
  finding re-breaks CI, so a stale exception cannot hide a risk forever.
- **Prefer remediation over exception.** Rebuild on a patched base or bump the package
  first; add an entry only when you genuinely cannot. (Two whole suppression sets in
  the IaC `.trivyignore` were later *removed* once the config was actually fixed — the
  same expectation applies here.)

The file ships with **zero active exceptions** — the images pass the fixable gate on
their own.

---

## 4. Run it locally

You need Docker and Trivy. The scan is the same two passes CI runs.

```bash
# 1. Build both images (mirrors the docker job).
docker build --target runtime -t ml-pipeline:ci .
docker build -f docker/mlflow/Dockerfile --build-arg BASE_IMAGE=ml-pipeline:ci \
  -t mlflow-server:ci .

# 2. Report pass — every HIGH/CRITICAL (fixable or not), for awareness.
trivy image --scanners vuln --pkg-types os,library \
  --severity HIGH,CRITICAL --ignorefile .trivyignore.yaml ml-pipeline:ci

# 3. Gate pass — what CI blocks on: FIXABLE HIGH/CRITICAL only.
trivy image --scanners vuln --pkg-types os,library \
  --ignore-unfixed --severity HIGH,CRITICAL --exit-code 1 \
  --ignorefile .trivyignore.yaml ml-pipeline:ci      # exit 1 == a real regression
```

No Trivy binary on your host? Run it from its container against a saved image (works
cross-platform, no daemon socket mount needed):

```bash
docker save ml-pipeline:ci -o img.tar
docker run --rm -v "$PWD:/w" aquasec/trivy:0.74.0 \
  image --input /w/img.tar --ignore-unfixed --severity HIGH,CRITICAL --exit-code 1
```

> The **vuln DB download** is the only network dependency (as with the IaC scan). The
> gate is **time-varying by design**: a new CVE published for an already-shipped
> package can turn a previously-green image red on a later run — that is correct, and
> the report names the exact package + CVE to bump.

---

## 5. What this does and does not cover

- **Covers:** known-CVE OS + Python packages in the two shipped images, before publish,
  on every PR, credential-free.
- **Does not replace ECR `scan_on_push`** ([ADR-021](decisions/ADR-021-terraform-managed-ecr.md)):
  that is a **second**, registry-side layer that fires *after* an operator publishes to
  AWS — complementary, not a substitute, and not runnable in credential-free PR CI.
- **Not yet:** SBOM generation or image signing (cosign) — roadmap v3 supply-chain
  items ([ci-cd.md § Future CD roadmap](ci-cd.md)); and digest-pinned, byte-reproducible
  builds ([ADR-005](decisions/ADR-005-containerization-strategy.md)).

## See also

- [ADR-035](decisions/ADR-035-container-image-scanning.md) — the decision record
- [ci-cd.md](ci-cd.md) — the CI pipeline (Job 2 `docker`)
- [containerization.md](containerization.md) — image design
- [terraform/.trivyignore](../terraform/.trivyignore) — the sibling IaC suppression record
- [SECURITY.md](../SECURITY.md) — the security policy
