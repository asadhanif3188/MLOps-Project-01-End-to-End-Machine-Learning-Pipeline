# ADR-004: Python Quality Toolchain (Ruff, mypy, pytest, pre-commit)

- **Status:** Accepted
- **Date:** 2026-08-02
- **Deciders:** Asad Hanif
- **Related:** [Developer Guide](../developer-guide.md),
  [Type Safety](../type-safety.md), [Testing Strategy](../testing-strategy.md),
  [Sprint 2 Engineering Review](../reviews/sprint-02-engineering-review.md)

## Context

The Sprint 2 engineering review found the baseline codebase had no enforced
style, no linting, no type checking, no tests, and no automated quality gates.
Sprint 2's engineering-excellence work (logging, exception handling, type
annotations, a test suite) needed a toolchain to keep those standards enforced
rather than aspirational.

Requirements:

- consistent formatting and linting with minimal contributor friction,
- static type checking to back the new annotations,
- a test runner for the new suite,
- enforcement at commit time, not only by reviewer vigilance, and
- **one source of truth per concern** — each rule configured in exactly one
  file, with every entry point (CLI, Makefile, hooks, editor) deferring to it.

## Decision

Adopt a four-tool chain, configured centrally:

- **Ruff** as both linter and formatter (`[tool.ruff]` in
  [`pyproject.toml`](../../pyproject.toml)): 88-character lines, Python 3.12
  target, and a deliberate rule selection (pycodestyle, pyflakes, isort,
  pyupgrade, bugbear, comprehensions, simplify, blind-except, Ruff-specific).
- **mypy** in a strict configuration (`[tool.mypy]`): complete annotations
  required in `src/`, no suppressions, third-party stubs scoped via overrides.
- **pytest** as the test runner (`[tool.pytest.ini_options]`): `tests/` tree,
  `smoke`/`unit` markers, strict marker enforcement.
- **pre-commit** to wire it together
  ([`.pre-commit-config.yaml`](../../.pre-commit-config.yaml)): Ruff and
  file-hygiene checks plus mypy at commit time, the test suite at push time.

A [`Makefile`](../../Makefile) provides memorable entry points (`make format`,
`make lint`, `make typecheck`, `make test`, `make check`), each a thin wrapper
over `python -m <tool>`. VS Code workspace settings (`.vscode/`) align the
editor with the same configuration.

## Alternatives Considered

1. **black + isort + flake8 (traditional stack).**
   - *Pros:* widely known, battle-tested.
   - *Cons:* three tools, three configs, plugin management for flake8, and
     slower runs. Ruff implements the same rule families in one fast tool, and
     its formatter is black-compatible.
2. **pylint (instead of or in addition to a linter).**
   - *Decision:* rejected — heavier, slower, and noisier than the curated Ruff
     rule set; marginal benefit for a codebase of this size.
3. **No git hooks (CI-only enforcement).**
   - *Decision:* rejected — CI does not exist yet (Roadmap v3), and even with
     CI, hooks catch issues before they reach a PR. Pre-commit also gives
     contributors a single install step (`make install-dev`).
4. **unittest (stdlib) instead of pytest.**
   - *Decision:* rejected — pytest's fixtures, markers, and terse assertions
     fit the contract-focused suite better; it is the de-facto community
     standard.

## Consequences

**Positive**

- Style, imports, types, and tests are enforced mechanically; review can focus
  on substance.
- One config file per concern (`pyproject.toml`, `.pre-commit-config.yaml`,
  `.editorconfig`) with no duplicate settings to drift.
- The chain runs fast enough to sit in commit hooks without friction.

**Trade-offs and follow-ups**

- Contributors need the dev environment installed (`requirements-dev.txt`) for
  hooks to run; the [Developer Guide](../developer-guide.md) covers setup.
- `make` is not native on Windows; every target documents its underlying
  `python -m …` command as a fallback.
- Enforcement is local until CI (Roadmap v3) re-runs the same checks
  server-side.
