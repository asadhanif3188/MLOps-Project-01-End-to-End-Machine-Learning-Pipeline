# Developer command shortcuts. Run `make help` (or just `make`) for the list.
#
# Every target is a thin wrapper over the underlying tool, invoked as
# `python -m <tool>` so it always resolves to the current environment's version
# (the ones pinned in requirements-dev.txt) rather than whatever is on PATH.
# The tools themselves are configured in pyproject.toml and
# .pre-commit-config.yaml — this file only provides memorable entry points.

# Override on the command line if needed, e.g. `make test PYTHON=python3.12`.
PYTHON ?= python

.DEFAULT_GOAL := help

.PHONY: help install install-dev format format-check lint lint-fix typecheck \
        test test-smoke test-unit coverage check pre-commit repro clean

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime dependencies only
	$(PYTHON) -m pip install -r requirements.txt

install-dev: ## Install dev dependencies and enable git hooks
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pre_commit install
	$(PYTHON) -m pre_commit install --hook-type pre-push

format: ## Auto-format the code with the Ruff formatter
	$(PYTHON) -m ruff format .

format-check: ## Check formatting without modifying files (CI-safe)
	$(PYTHON) -m ruff format --check .

lint: ## Lint the code with Ruff
	$(PYTHON) -m ruff check .

lint-fix: ## Lint and apply safe autofixes
	$(PYTHON) -m ruff check --fix .

typecheck: ## Run static type checking with mypy
	$(PYTHON) -m mypy

test: ## Run the full test suite
	$(PYTHON) -m pytest

test-smoke: ## Run only the fast smoke tests
	$(PYTHON) -m pytest -m smoke

test-unit: ## Run only the unit tests
	$(PYTHON) -m pytest -m unit

coverage: ## Run tests with a terminal coverage report
	$(PYTHON) -m pytest --cov=src --cov-report=term-missing

check: lint format-check typecheck test ## Run all quality gates (what CI runs)

pre-commit: ## Run every pre-commit hook against all files
	$(PYTHON) -m pre_commit run --all-files

repro: ## Reproduce the DVC pipeline
	dvc repro

clean: ## Remove caches and Python bytecode
	rm -rf .ruff_cache .mypy_cache .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
