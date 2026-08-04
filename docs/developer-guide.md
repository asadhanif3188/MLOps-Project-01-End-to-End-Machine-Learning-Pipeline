# Developer Guide

This guide is the practical companion to the [Contributing](../CONTRIBUTING.md)
page: it explains how to set up a local environment and run the day-to-day
tooling — formatting, linting, type checking, tests, and the pre-commit hooks —
that keep the codebase consistent. It is the reference for the *how*; the
[Testing Strategy](testing-strategy.md) and [Type Safety](type-safety.md)
documents cover the *why* behind two of those tools.

The guiding principle is **one source of truth per concern**. Line length,
lint rules, type strictness, and test configuration each live in exactly one
file, and every entry point — the command line, the [`Makefile`](../Makefile),
the [pre-commit hooks](../.pre-commit-config.yaml), and the editor — defers to
it. There is no second place to keep in sync.

---

## 1. The toolchain at a glance

| Concern | Tool | Configured in | Run it with |
|---------|------|---------------|-------------|
| Formatting | Ruff formatter | `[tool.ruff]` in `pyproject.toml` | `make format` |
| Linting | Ruff linter | `[tool.ruff.lint]` in `pyproject.toml` | `make lint` |
| Type checking | mypy | `[tool.mypy]` in `pyproject.toml` | `make typecheck` |
| Tests | pytest | `[tool.pytest.ini_options]` in `pyproject.toml` | `make test` |
| Git hooks | pre-commit | `.pre-commit-config.yaml` | `make pre-commit` |
| Editor | EditorConfig + VS Code | `.editorconfig`, `.vscode/` | automatic |

[Ruff](https://docs.astral.sh/ruff/) provides both the linter and the formatter,
replacing the traditional flake8 + isort + black stack with a single, fast tool.
The formatter is black-compatible, so the 88-character line length in
`.editorconfig`, the Ruff config, and the formatter all agree on one number.

> **Note on `make` (Windows).** `make` is not installed by default on Windows.
> Every target below is a thin wrapper over a `python -m …` command, so if you
> do not have `make`, run the underlying command shown in each section directly.
> Install `make` via [Chocolatey](https://chocolatey.org/) (`choco install make`)
> or use WSL if you prefer the shortcuts. A few targets (`help`, `clean`) also
> use POSIX utilities (`grep`, `awk`, `find`, `rm`), so on Windows run them from
> Git Bash or WSL rather than `cmd`/PowerShell.

---

## 2. Local development

### Prerequisites

- **Python 3.12** — the version the pipeline is written and type-checked against.
- **Git**, and optionally **DVC** credentials if you intend to pull data or
  reproduce the pipeline (see the [architecture](architecture.md) overview).

### First-time setup

```bash
# 1. Create and activate an isolated environment (conda shown; venv works too).
conda create -p myenv python=3.12 -y
conda activate myenv

# 2. Install the development dependencies (this also pulls in the runtime deps)
#    and enable the git hooks in one step.
make install-dev

# 3. Provide configuration. Copy the template and fill in the values
#    (MLflow tracking URI and DagsHub credentials — see the README).
cp .env.example .env
```

Without `make`, step 2 is:

```bash
pip install -r requirements-dev.txt
pre-commit install
pre-commit install --hook-type pre-push
```

`make install-dev` installs the tooling **and** registers the pre-commit hooks
for this clone, so quality checks start running automatically from your next
commit. See [§6](#6-pre-commit-workflow) for what those hooks do.

### The everyday loop

While working, the fastest feedback comes from your editor (formatting and lint
fixes apply on save — see [§7](#7-editor-configuration)). Before opening a pull
request, run the full quality gate — the same checks the pre-commit hooks apply,
and a superset of what [continuous integration](ci-cd.md) runs on every pull
request:

```bash
make check      # lint + format-check + typecheck + test
```

`make help` lists every available target with a one-line description.

### Developing in a container (alternative)

If you would rather not install Python and the toolchain on your host, the
project ships a Docker Compose development environment that provides all of it:

```bash
cp .env.example .env          # credentials (optional for lint/type/test)
docker compose up -d          # build + start the dev container
docker compose exec dev bash  # shell in; the same `make` targets work here
```

Your working tree is bind-mounted, so edits on the host are live in the
container — no rebuild for a code change. The full lifecycle (startup, logs,
rebuild, troubleshooting) is documented in the
[Docker Development Workflow](docker-development.md); the image itself is
described in the [Containerization Strategy](containerization.md).

---

## 3. Formatting

Formatting is **not a matter of taste** here — it is applied mechanically by the
Ruff formatter so that diffs stay small and reviews focus on behaviour, not
whitespace.

```bash
make format          # rewrite files in place  (python -m ruff format .)
make format-check    # report only, change nothing  (python -m ruff format --check .)
```

Use `format` locally; `format-check` is the non-mutating variant that `make
check` and CI use to fail a build when something is unformatted. The formatter
governs the pipeline's `.py` sources only — Markdown is excluded so the
illustrative code snippets in `docs/` are never rewritten.

---

## 4. Linting

The linter catches likely bugs and enforces import order and modern syntax. The
enabled rule families (pyflakes, pycodestyle, isort, pyupgrade, bugbear,
comprehensions, simplify, blind-except, and Ruff's own rules) are documented
inline in `[tool.ruff.lint]` — each is turned on deliberately rather than by
enabling everything and suppressing the fallout.

```bash
make lint       # report issues            (python -m ruff check .)
make lint-fix   # apply safe autofixes     (python -m ruff check --fix .)
```

Most findings are auto-fixable. When a lint rule must genuinely be broken, add a
scoped, *explained* `# noqa: <CODE>` on the line — never a bare `# noqa`, which
silences everything. Ruff's `RUF100` rule flags any `# noqa` that is no longer
needed and removes it on `--fix`, so suppressions cannot silently rot: a
suppression only survives while the rule it names actually fires on that line.

---

## 5. Running tests

Tests use pytest and live under [`tests/`](../tests/). They are fast, isolated,
and deterministic by design — the full suite runs in well under a second and
needs no network, MLflow, or real data. The [Testing Strategy](testing-strategy.md)
explains the layout and conventions in depth.

```bash
make test         # the whole suite            (python -m pytest)
make test-smoke   # fast import/wiring checks   (python -m pytest -m smoke)
make test-unit    # isolated component tests    (python -m pytest -m unit)
make coverage     # suite + coverage report     (python -m pytest --cov=src --cov-report=term-missing)
```

The `smoke` and `unit` markers are declared (and enforced with
`--strict-markers`) in `pyproject.toml`; select any slice with `pytest -m
<marker>`. Coverage is available on demand via `make coverage`, but note the
project **does not chase a coverage percentage** — tests earn their place by
guarding a contract worth protecting, not by moving a number.

---

## 6. Pre-commit workflow

[pre-commit](https://pre-commit.com/) runs the project's quality gates locally,
so problems are caught before they are ever pushed — the same checks
[continuous integration](ci-cd.md) re-runs on every pull request. The hooks
are defined in [`.pre-commit-config.yaml`](../.pre-commit-config.yaml).

### How it is wired

The hooks run in two stages, chosen to keep the inner loop fast:

| Stage | Hooks | When |
|-------|-------|------|
| **commit** | Ruff lint (+fix), Ruff format, file-hygiene checks, mypy | every `git commit` |
| **push** | the full pytest suite | every `git push` |

Formatting and fast static checks run on every commit; the test suite is gated
at push time so individual commits stay quick. This split is enforced by
`default_stages: [pre-commit]` in the hook config, with the pytest hook opting
into `pre-push` explicitly. The tool versions pinned in the hook config match
`requirements-dev.txt`, so every environment applies identical rules.

### Using it

```bash
make install-dev          # one-time: installs hooks (commit + pre-push)
```

After that the hooks fire automatically. If a formatting or autofix hook changes
a file, the commit is aborted so you can review the change — simply `git add` the
modifications and commit again. To run every hook against the whole tree on
demand (useful after changing the config or before a big PR):

```bash
make pre-commit           # python -m pre_commit run --all-files
```

### Bypassing (rarely)

`git commit --no-verify` skips the hooks. Reserve it for genuine emergencies —
CI runs the same checks, so anything skipped locally will surface there anyway.

---

## 7. Editor configuration

Two layers configure the editor so that what you see locally matches what the
command line and hooks enforce:

- **[`.editorconfig`](../.editorconfig)** — editor-agnostic baseline: UTF-8, LF
  line endings, a final newline, trailing-whitespace trimming, indentation, and
  the 88-character Python line length. Most editors honour it out of the box;
  VS Code needs the EditorConfig extension (recommended below).
- **[`.vscode/`](../.vscode/)** — for VS Code users, `settings.json` wires Ruff
  in as the formatter with format-and-fix-on-save, and `extensions.json`
  recommends the Ruff, Python, mypy, and EditorConfig extensions. Opening the
  repository prompts you to install them.

These settings are conveniences: they apply the *same* tools configured in
`pyproject.toml`, never a competing set of rules. Contributors using other
editors get the identical result from `.editorconfig` plus the `make` targets.

---

## 8. Troubleshooting

- **`make: command not found` (Windows).** `make` is not installed by default.
  Run the underlying `python -m …` command shown in each section, or install
  `make` (see [§1](#1-the-toolchain-at-a-glance)).
- **A hook reformats files and the commit fails.** This is expected — the hook
  fixed something. Review the change, `git add` it, and commit again.
- **`pre-commit` hook install fails to execute on Windows.** Some corporate
  security policies (Controlled Folder Access / antivirus) block execution from
  the pre-commit cache. Run the checks directly via the `make` targets in the
  meantime; the checks themselves are unaffected.
- **Import errors when running tests or mypy.** Both add `src/` to the path
  (`pythonpath`/`mypy_path` in `pyproject.toml`) so stages import their siblings
  by bare module name. Run tools from the repository root so those settings apply.
