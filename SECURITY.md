# Security Policy

## Supported Versions

This project is under active development. Security fixes are applied to the
latest released version and the `main` branch.

| Version | Supported |
|---------|-----------|
| Latest release | ✅ |
| `main` (development) | ✅ |
| Older releases | ❌ |

> <!-- TODO: refine this table once tagged releases exist (see docs/versioning.md). -->

## Reporting a Vulnerability

Please **do not report security vulnerabilities through public GitHub issues.**

Instead, report them privately using one of the following:

- GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
  ("Report a vulnerability" under the repository's **Security** tab), or
- <!-- TODO: add a dedicated security contact email if one is available. -->

When reporting, please include:

- A description of the vulnerability and its potential impact.
- Steps to reproduce, or a proof of concept.
- Any relevant environment details (OS, Python version, dependency versions).

## Responsible Disclosure

We ask that you:

- Give us a reasonable opportunity to investigate and address the issue before
  any public disclosure.
- Avoid accessing, modifying, or deleting data that does not belong to you.
- Act in good faith and avoid privacy violations, service disruption, or data
  destruction.

We will:

- Acknowledge your report as soon as reasonably possible.
- Keep you informed of progress toward a fix.
- Credit reporters who wish to be acknowledged, once the issue is resolved.

> <!-- TODO: define concrete response/acknowledgement time targets when the
> project's maintenance cadence is established. -->

## Security Best Practices

Contributors and users should follow these practices:

- **Never commit secrets.** Credentials (e.g. `MLFLOW_TRACKING_URI`,
  `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`) belong in a local
  `.env` file, which is git-ignored. Only `.env.example` (a template) is
  committed.
- **Rotate credentials** if they are ever exposed, and remove them from history.
- **Review dependencies.** Keep `requirements.txt` up to date and monitor for
  known vulnerabilities.
- **Least privilege.** Use scoped tokens for the DagsHub/MLflow and DVC remotes
  rather than broad credentials.
- **Validate data sources.** Treat external datasets and artifacts as untrusted
  input.
- **Infrastructure credentials & IAM.** The Terraform ([`terraform/`](terraform/))
  AWS platform never stores static AWS credentials — no access keys or secret
  keys are committed; Terraform authenticates via the standard AWS credential
  chain. AWS IAM roles are least-privilege and dedicated to purpose (see
  [ADR-016](docs/decisions/ADR-016-aws-iam-foundation.md)): permissions come from
  the AWS-managed policies EKS requires, with no `AdministratorAccess` and no
  project-authored wildcard. State files and kubeconfigs are git-ignored and
  never committed. **CI holds no AWS credentials or cloud identity**: the
  `terraform-validate` job validates the IaC *statically* (`fmt`/`init
  -backend=false`/`validate`, TFLint, Trivy) and never runs `terraform
  plan`/`apply` — real provisioning is a deliberate, operator-driven step against
  one's own account (see [ADR-019](docs/decisions/ADR-019-terraform-ci-validation.md)).

---

## Related Documentation

- [Support](SUPPORT.md)
- [Contributing](CONTRIBUTING.md)
- [Engineering Philosophy](docs/philosophy.md)
