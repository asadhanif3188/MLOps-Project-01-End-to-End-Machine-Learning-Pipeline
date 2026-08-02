# Versioning

This project follows [Semantic Versioning 2.0.0](https://semver.org/).

Version numbers take the form:

```text
MAJOR.MINOR.PATCH
```

Given a version `X.Y.Z`, each component is incremented as follows:

| Component | Increment when… |
|-----------|-----------------|
| **MAJOR** (`X`) | You make incompatible or breaking changes. |
| **MINOR** (`Y`) | You add functionality in a backward-compatible manner. |
| **PATCH** (`Z`) | You make backward-compatible bug fixes. |

When a higher-order component is incremented, lower-order components reset to `0`
(e.g. `1.4.2` → `2.0.0`).

---

## What Counts as a Change Here

Because this is an ML pipeline rather than a public API, "compatibility" is
interpreted in terms of the pipeline's interfaces and reproducibility.

### MAJOR — breaking changes

Examples relevant to this repository:

- Changing the pipeline's expected inputs or outputs in an incompatible way
  (e.g. the dataset schema or the target column `Outcome`).
- Restructuring `params.yaml` or `dvc.yaml` such that existing configurations no
  longer work.
- Replacing the model type or serialization format in a way that invalidates
  previously produced `models/model.pkl` artifacts.

*Example:* `1.3.0` → `2.0.0` when the preprocessing output format changes and
downstream stages must be updated.

### MINOR — new, backward-compatible functionality

Examples:

- Adding a new pipeline stage (e.g. a dedicated feature-engineering step).
- Adding new tracked metrics or artifacts to MLflow without changing existing
  ones.
- Adding a new optional parameter to `params.yaml` with a sensible default.

*Example:* `1.2.0` → `1.3.0` when a held-out evaluation split is added without
changing existing interfaces.

### PATCH — backward-compatible fixes

Examples:

- Fixing the `dvc.yaml` / `params.yaml` parameter-name mismatch (see
  [ADR-003](decisions/ADR-003-why-dvc.md)).
- Correcting the `preprocess.py` behavior to match its documented intent.
- Documentation-only corrections that don't change behavior.

*Example:* `1.2.0` → `1.2.1` for a bug fix that produces the intended result
without changing the pipeline's interface.

---

## Pre-1.0.0 Development

While the project is at `0.y.z`, anything may change at any time; the public
interface should not be considered stable. The first stable, tagged release is
`1.0.0`.

The project has passed that point: `1.0.0` was tagged for the Sprint 1
professional-repository baseline (2026-08-01) and `1.1.0` for the Sprint 2
engineering-excellence work. Both are recorded in
[`CHANGELOG.md`](../CHANGELOG.md).

---

## Tagging & Changelog

- Each release is tagged in Git using the `vMAJOR.MINOR.PATCH` format
  (e.g. `v1.0.0`).
- `CHANGELOG.md` follows [Keep a Changelog](https://keepachangelog.com/) and is
  updated with every release.
- See the [release checklist](release-checklist.md) for the full process.

---

## Related Documentation

- [Release Checklist](release-checklist.md)
- [GitHub Workflow](github-workflow.md)
- [Roadmap](roadmap.md)
