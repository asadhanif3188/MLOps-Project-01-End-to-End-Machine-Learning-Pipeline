# Sprint 8 PR 8 — Container-image scan evidence (EXECUTED)

- **Date:** 2026-08-20
- **Branch / PR:** `feature/sprint-08-image-scanning` — *feat(security): enforce
  container vulnerability scanning in CI*
- **Design of record:** [ADR-035](../decisions/ADR-035-container-image-scanning.md) ·
  **Policy + local run:** [docs/container-image-scanning.md](../container-image-scanning.md)
- **Scanner:** Trivy `0.74.0` (the repo-pinned version), vuln DB as of 2026-08-20.
- **Environment:** local Docker Desktop; images built from this branch's `HEAD`
  Dockerfiles and scanned via the Docker daemon — the **same method** the CI `docker`
  job uses. This is **real captured evidence**, not a pending checklist.

> **What this proves.** The scan policy (severity + fixability gate), the applied fix,
> and the two justified exceptions produce a **green gate on both shipped images**,
> while every non-fixable HIGH/CRITICAL remains **reported**. Numbers below are the
> actual Trivy output; the JSON reports are the CI `trivy-image-reports` artifact.

---

## 1. Images scanned

| Image | Build | Base |
|---|---|---|
| `mlops-pipeline` | [`Dockerfile`](../../Dockerfile) `runtime` target | `python:3.12-slim-bookworm` (Debian 12.15) + pip stack |
| `mlflow-server` | [`docker/mlflow/Dockerfile`](../../docker/mlflow/Dockerfile) `FROM` the pipeline image | pipeline image + `psycopg2-binary` |

Scanned for **OS packages + Python libraries** (`--scanners vuln --pkg-types os,library`),
severity **HIGH,CRITICAL**.

## 2. Severity policy (recap)

Gate on **HIGH + CRITICAL**; within those, **fail on *fixable* only** (`--ignore-unfixed`).
Non-fixable HIGH/CRITICAL are **reported, not gated** (no upstream fix to apply; they
auto-promote to the gate when one ships). Full rationale: ADR-035 § Decision.

## 3. Findings — baseline (before fix)

Both images, identical result (the mlflow image inherits the base; its `psycopg2-binary`
layer added **no** fixable HIGH/CRITICAL):

| Class | CRITICAL | HIGH | Gates? |
|---|---:|---:|---|
| **Fixable** (HIGH/CRITICAL) | 0 | **3** | **YES → gate FAILED (exit 1)** |
| **Non-fixable** (HIGH/CRITICAL) | 18 | 52 | No — reported only |

The **3 fixable HIGH**:

| Package | Installed → Fixed | ID | Source |
|---|---|---|---|
| `cryptography` | 49.0.0 → 50.0.0 | CVE-2026-69247 | real venv dependency |
| `msgpack` | 1.1.2 → 1.2.1 | GHSA-6v7p-g79w-8964 | **pip-vendored** (`pip/_vendor/msgpack`) |
| `setuptools` | 70.3.0 → 78.1.1 | CVE-2025-47273 | **pip-vendored** (`pip/_vendor/pkg_resources`) |

The **70 non-fixable** HIGH/CRITICAL are **all Debian base-image OS packages** (top
offenders: `perl`/`perl-base`/`libperl5.36`/`perl-modules-5.36`, `libcurl3-gnutls`,
`libexpat1`, `zlib1g`, `libsqlite3-0`) — no fixed Debian package is published, so
`apt-get upgrade` would not clear them either. They are correctly reported, not gated.

## 4. Fixes applied

- **`cryptography` 49.0.0 → 50.0.0** — a real installed transitive dependency; raised
  via a **targeted security-floor bump** in the [`Dockerfile`](../../Dockerfile) builder
  stage (not a broad upgrade). Rebuilt image → **finding cleared**. The in-Dockerfile
  import smoke-test (`import sklearn, mlflow, dvc, pandas`) and `dvc --version` both
  **passed** on the rebuilt image, so the bump broke neither resolution nor imports.
- `msgpack` and `setuptools` are **also** raised in the venv (their real copies →
  1.2.1 / ≥78.1.1) as hygiene, but that does **not** clear their *reported* findings —
  see next section.

## 5. Exceptions (justified, time-boxed)

The residual `msgpack` and `setuptools` findings come from **pip's vendored copies**
(`pip/_vendor/msgpack`, `pip/_vendor/pkg_resources`), confirmed by locating every copy
in the image. These:

- **cannot be upgraded independently** — pip pins its own vendored deps; no
  `pip install -U` touches them;
- are **not on the runtime execution path** — the container runs `dvc repro`, never
  pip; pip's vendored msgpack is used only by pip's install-time HTTP cache;
- **cannot be removed by dropping pip** — the `mlflow-server` image is built `FROM`
  this image and runs `pip install psycopg2-binary`, so pip must stay.

They are recorded as **two specific, time-boxed exceptions** in
[`.trivyignore.yaml`](../../.trivyignore.yaml) (id + rationale + `expired_at: 2026-11-18`,
auto-expired by Trivy) — **not** a blanket severity mute. Remediation (base-image/pip
bump, or restructuring the mlflow image so pip is unneeded) is tracked in ADR-035
§ Follow-ups.

## 6. Final scan result (after fix + exceptions)

Re-scanned via the Docker daemon with `--ignore-unfixed --severity HIGH,CRITICAL
--ignorefile .trivyignore.yaml` — the exact CI gate:

| Image | Gating (fixable HIGH/CRITICAL, minus exceptions) | Gate exit | Non-fixable still reported |
|---|---:|:---:|---:|
| `mlops-pipeline` (`ml-pipeline:fixed`) | **0** | **0 (PASS)** | 70 |
| `mlflow-server` (`mlflow-server:fixed`) | **0** | **0 (PASS)** | 70 |

```
GATE_EXIT[ml-pipeline:fixed]=0
GATE_EXIT[mlflow-server:fixed]=0
```

**Result: green gate on both images** — one real fix applied and verified, two
pip-vendored findings excepted with expiry, and all 70 non-fixable base-image CVEs
transparently reported.

## 7. Honesty boundary

- The gate is **time-varying by design**: a new CVE for an already-shipped package can
  turn a green image red on a later run (correct security behaviour; the report names
  the package to bump). ADR-035 § Consequences.
- Runtime deps are **not yet digest/version-pinned** ([ADR-005](../decisions/ADR-005-containerization-strategy.md)
  roadmap), so the exact resolved set can shift build-to-build; these numbers are the
  2026-08-20 build.
- This scans **images**, complementing (not replacing) ECR `scan_on_push`
  ([ADR-021](../decisions/ADR-021-terraform-managed-ecr.md)); **SBOM** and **signing**
  remain roadmap v3.
