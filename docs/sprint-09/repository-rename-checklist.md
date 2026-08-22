# Repository Rename Checklist (Sprint 9, PR 1)

This is the **manual** operator checklist for renaming the GitHub repository from
its original slug to the final identity chosen in
[Repository Naming Evaluation](repository-naming-evaluation.md).

- **From:** `asadhanif3188/MLOps-Project-01-End-to-End-Machine-Learning-Pipeline`
- **To:** `asadhanif3188/mlops-platform-on-eks`

> **Claude Code does not perform the GitHub rename.** Renaming the remote
> repository is a GitHub Settings operation and must be done by the repository
> owner. The steps below are the runbook for that operation and the follow-up
> verification.

> **Status (2026-08-22, Sprint 9 PR 12):** The rename is **applied** — the git
> remote resolves to `asadhanif3188/mlops-platform-on-eks` and prior PRs merged
> against it (GitHub also keeps the old-slug redirect). The unchecked items below
> that remain relevant are the GitHub *presentation* settings (About, topics,
> social-preview image, profile pin) and the external link updates, which stay
> owner-driven.

## Sequencing note

This PR updates repo-name-dependent URLs (CI badge, OCI image source label,
Terraform repository variable, alert `runbook_url`s, discussion/support links) to
the **new** slug. Those URLs resolve correctly **only after** the GitHub rename
below is executed. GitHub keeps a permanent redirect from the **old** slug to the
new one, so any residual old-slug links (e.g. historical CHANGELOG release links)
continue to work. To avoid a badge-broken window, perform the rename close to
merging this PR.

## Rename steps

- [ ] **Rename the repository in GitHub Settings.** Settings → General →
      Repository name → set to `mlops-platform-on-eks` → Rename.
- [ ] **Confirm the redirect from the old URL.** Visit the old URL
      (`https://github.com/asadhanif3188/MLOps-Project-01-End-to-End-Machine-Learning-Pipeline`)
      and confirm it redirects to the new one.
- [ ] **Update the local git remote.**
      ```bash
      git remote set-url origin https://github.com/asadhanif3188/mlops-platform-on-eks.git
      git remote -v   # verify
      ```
      Advise any other clones/collaborators to do the same (old remote still works
      via redirect, but should be updated).

## Presence / links to update

- [ ] **Update the GitHub profile pin** — re-pin the repo if the pin title/URL is
      cached; set the About description and topics per
      [Repository Metadata Recommendations](../repository-metadata.md).
- [ ] **Update the website / portfolio link** pointing at the repo.
- [ ] **Update LinkedIn / resume links** referencing the old slug or old title.

## Post-rename verification

- [ ] **Verify Actions & badges.** Confirm the README CI badge renders (points at
      `.../mlops-platform-on-eks/actions/workflows/ci.yml/badge.svg`) and that
      Actions still run on push/PR.
- [ ] **Verify release URLs.** Spot-check a release/tag URL and a CHANGELOG
      compare link — both the new-slug and old-slug (redirected) forms should
      resolve.
- [ ] **Verify clone instructions.** `git clone https://github.com/asadhanif3188/mlops-platform-on-eks.git`
      succeeds.
- [ ] **Verify the OCI image source label** on the published image matches the new
      URL (`org.opencontainers.image.source`), so provenance links stay valid.
- [ ] **Verify alert runbook links.** An alert's `runbook_url` opens the correct
      `docs/alerting.md#…` anchor on the renamed repo.

## Out of scope for the GitHub rename

These identities are **separate services** and are **not** changed by renaming
the GitHub repository. Leave them unless independently migrated:

- **DagsHub DVC remote** — `.dvc/config` points at
  `dagshub.com/asadhanif3188/MLOps-Project-01-End-to-End-Machine-Learning-Pipeline.s3`.
  This is the functional data remote; changing it without renaming the DagsHub
  repo would break `dvc pull`.
- **ECR image repositories** — `mlops-pipeline` / `mlflow-server` are named
  independently of the GitHub slug (see `scripts/release-image.sh`).
