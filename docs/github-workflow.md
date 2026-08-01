# GitHub Workflow

Work in this repository is organized and delivered through a consistent process:
the branching model, commit conventions, pull request flow, labels, milestones,
and release strategy described below.

The project follows **[GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow)**
— a lightweight, branch-based workflow suited to continuous, incremental
delivery.

---

## Branch Strategy

- **`main` is always deployable.** It reflects the latest reviewed, working state
  of the project.
- **Work happens on short-lived feature branches** created from `main`.
- **Branches are merged back into `main` via pull request** after review.
- **Delete branches after merge** to keep the repository tidy.

### Branch naming

Use a `type/short-description` convention, matching the commit types below:

| Prefix | Purpose | Example |
|--------|---------|---------|
| `feat/` | New feature | `feat/model-serving-endpoint` |
| `fix/` | Bug fix | `fix/param-name-mismatch` |
| `docs/` | Documentation | `docs/architecture-diagrams` |
| `refactor/` | Refactoring | `refactor/train-module` |
| `test/` | Tests | `test/preprocess-coverage` |
| `chore/` | Tooling/maintenance | `chore/update-dependencies` |
| `ci/` | CI/CD changes | `ci/add-lint-workflow` |

---

## Commit Conventions

Commits follow the [Conventional Commits](https://www.conventionalcommits.org/)
format:

```text
<type>: <short description>

<optional body explaining what and why>
```

**Types:** `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`, `ci`.

**Guidelines:**

- Use the imperative mood ("add", not "added" or "adds").
- Keep the subject line concise (~72 characters).
- Explain *why* in the body when the change isn't self-evident.
- One logical change per commit where practical.

**Examples:**

```text
docs: expand architecture document with data flow
fix: align dvc.yaml params with params.yaml
feat: add held-out evaluation split
```

---

## Pull Request Workflow

1. **Create a branch** from `main` using the naming convention above.
2. **Make focused commits** following the commit conventions.
3. **Open a pull request** into `main` using the
   [PR template](../.github/pull_request_template.md).
4. **Link related issues** (e.g. `Closes #123`).
5. **Complete the checklist**, including documentation and `CHANGELOG.md`
   updates where relevant.
6. **Request review.** Address feedback with additional commits.
7. **Merge** once approved and all checks pass, then **delete the branch**.

> <!-- TODO: once CI is added (see docs/roadmap.md, v3), require status checks to
> pass before merge and document any branch protection rules here. -->

---

## Labels

Labels categorize issues and pull requests. The recommended baseline set:

| Label | Description |
|-------|-------------|
| `bug` | Something isn't working |
| `enhancement` | New feature or request |
| `documentation` | Documentation-related work |
| `good first issue` | Suitable for newcomers |
| `help wanted` | Extra attention is welcome |
| `question` | Further information is requested |
| `wontfix` | Will not be worked on |
| `duplicate` | Already reported elsewhere |
| `dependencies` | Dependency updates |
| `ci` | Continuous integration/tooling |

> <!-- TODO: finalize the label set in GitHub settings; this table documents the
> recommended baseline only. -->

---

## Milestones

Milestones group issues and pull requests toward a specific objective, typically
aligned with the versioned entries in the [roadmap](roadmap.md).

- Create one milestone per roadmap version (e.g. "v2 — Engineering Improvements").
- Assign issues and PRs to the milestone they contribute to.
- Use milestone progress to track readiness for a release.

---

## Release Strategy

- Releases follow [Semantic Versioning](versioning.md).
- Releases are cut from `main` once a milestone's scope is complete and the
  [release checklist](release-checklist.md) is satisfied.
- Each release is tagged (e.g. `v1.0.0`) and accompanied by an updated
  `CHANGELOG.md` and GitHub release notes.

> <!-- TODO: decide whether releases are automated via CI or cut manually
> (see docs/roadmap.md, v3). -->

---

## Related Documentation

- [Versioning](versioning.md)
- [Release Checklist](release-checklist.md)
- [Contributing](../CONTRIBUTING.md)
- [Roadmap](roadmap.md)
