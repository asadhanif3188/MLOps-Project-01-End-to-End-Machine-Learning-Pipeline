# Contributing

Thank you for your interest in contributing to this project. This guide explains
how to propose changes and what to expect. It complements the more detailed
[GitHub Workflow](docs/github-workflow.md) documentation.

## Code of Conduct

This project and everyone participating in it is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to
uphold it.

## How to Contribute

The project follows [GitHub Flow](docs/github-workflow.md#branch-strategy):

1. Fork the repository and create a branch from `main` using the
   [branch naming convention](docs/github-workflow.md#branch-strategy).
2. Make your changes with clear, focused commits.
3. Open a pull request describing the change and its motivation.

## Development Setup

<!-- TODO: Document detailed local setup once the README Quick Start is written.
     In brief: create a Python 3.12 environment, `pip install -r requirements.txt`,
     copy `.env.example` to `.env`, and configure DVC/MLflow credentials. -->

For an overview of the repository layout, see
[Project Structure](docs/project-structure.md).

## Commit Messages

Follow the [Conventional Commits](docs/github-workflow.md#commit-conventions)
format documented in the GitHub Workflow guide (`type: description`, imperative
mood).

## Reporting Issues

Use the provided issue templates:

- [Bug report](.github/ISSUE_TEMPLATE/bug_report.md)
- [Feature request](.github/ISSUE_TEMPLATE/feature_request.md)
- [Documentation](.github/ISSUE_TEMPLATE/documentation.md)

For questions and support, see [SUPPORT.md](SUPPORT.md). For security issues, see
the [Security Policy](SECURITY.md).

## Pull Request Guidelines

- Open PRs against `main` using the
  [pull request template](.github/pull_request_template.md).
- Complete the checklist, including documentation and `CHANGELOG.md` updates
  where relevant.
- Link the related issue (e.g. `Closes #123`).
- See the full [pull request workflow](docs/github-workflow.md#pull-request-workflow)
  for details.
