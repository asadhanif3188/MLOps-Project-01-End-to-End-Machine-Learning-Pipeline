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

Create a Python 3.12 environment, install the development dependencies, and
enable the git hooks:

```bash
conda create -p myenv python=3.12 -y && conda activate myenv
make install-dev            # installs dev deps + registers pre-commit hooks
cp .env.example .env        # then set the MLflow / DagsHub values (see README)
```

If you do not have `make`, the equivalent is `pip install -r requirements-dev.txt`
followed by `pre-commit install` and `pre-commit install --hook-type pre-push`.

The [Developer Guide](docs/developer-guide.md) documents the full local workflow.
For an overview of the repository layout, see
[Project Structure](docs/project-structure.md).

## Code Quality

The project uses [Ruff](https://docs.astral.sh/ruff/) (linter and formatter),
[mypy](https://mypy.readthedocs.io/) (static typing), and
[pre-commit](https://pre-commit.com/) to keep the codebase consistent. All are
configured in `pyproject.toml` and wired into the pre-commit hooks, so the same
checks apply in every environment (and will run in CI once it is added).

Run the full gate before opening a pull request:

```bash
make check      # lint + format-check + typecheck + test
```

Individual steps are available as `make format`, `make lint`, `make typecheck`,
and `make test` (run `make help` for the full list). Once `pre-commit` is
installed, formatting and static checks run automatically on every commit and the
test suite on every push. See the
[Developer Guide](docs/developer-guide.md) for details on each tool and the
pre-commit workflow.

## Running Tests

Install the development dependencies and run the suite from the repository root:

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Select a slice with markers — `python -m pytest -m smoke` or `-m unit`. See the
[Testing Strategy](docs/testing-strategy.md) for the suite layout and the
conventions for adding tests. Please accompany behavioural changes with tests.

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
