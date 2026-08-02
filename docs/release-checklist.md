# Release Checklist

Use this checklist when preparing a release. It ensures each release is
consistent, documented, and reproducible. Releases follow
[Semantic Versioning](versioning.md) and are cut from `main`.

---

## Pre-Release

- [ ] Confirm the target version number per [Semantic Versioning](versioning.md).
- [ ] Confirm the associated [roadmap](roadmap.md) milestone scope is complete.
- [ ] Ensure `main` is up to date and all intended PRs are merged.

## Documentation

- [ ] **Update `CHANGELOG.md`** — move items from **Unreleased** into a new
      dated version section.
- [ ] **Review documentation** — verify `docs/` is accurate and consistent with
      the code (architecture, project structure, workflow, versioning).
- [ ] Resolve or re-scope any `TODO` markers that this release addresses.

## Verification

- [ ] **Verify tests pass** — run `make test` (or `python -m pytest`); the suite
      must be green (skips for optional, uninstalled runtime deps are expected).
- [ ] **Verify lint, format, and types** — `make check` (Ruff lint +
      format-check + mypy) reports no findings.
- [ ] Confirm the pipeline reproduces cleanly (`dvc repro` / `dvc status`).
- [ ] Confirm no secrets or credentials are present in the tree.

## Roadmap & Decisions

- [ ] **Update the roadmap** — mark completed milestones and adjust upcoming
      versions as needed.
- [ ] **Review ADRs** — confirm existing
      [decisions](decisions/) still hold; add new ADRs for any decisions made
      during this cycle and update statuses/dates.

## Tag & Publish

- [ ] **Tag the release** in Git using `vMAJOR.MINOR.PATCH` (e.g. `v1.0.0`).
- [ ] Push the tag and create a GitHub release with notes derived from
      `CHANGELOG.md`.

## Post-Release

- [ ] Open a new **Unreleased** section in `CHANGELOG.md`.
- [ ] Communicate the release (if applicable).

---

## Related Documentation

- [Versioning](versioning.md)
- [GitHub Workflow](github-workflow.md)
- [Roadmap](roadmap.md)
- [Architecture Decision Records](decisions/)
