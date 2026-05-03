# FastWoe Development Makefile

.PHONY: help install test test-all lint format typecheck clean check-all

help:  ## Show this help message
	@echo "FastWoe Development Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install development dependencies
	uv sync --dev

test:  ## Run tests (current Python)
	uv run pytest

test-all:  ## Run tests across all supported Python versions (3.9 → 3.14)
	@for ver in 3.9 3.10 3.11 3.12 3.13 3.14; do \
		echo "\n▶ Python $$ver"; \
		uv run --python $$ver pytest tests/ -q -m "not compatibility" || exit 1; \
	done
	@echo "\n[SUCCESS] All versions passed"

lint:  ## Run linting
	uv run ruff check fastwoe

format:  ## Format code
	uv run ruff format fastwoe

format-check:  ## Check code formatting
	uv run ruff format --check fastwoe

typecheck:  ## Run mypy type checking
	@echo "Running mypy type checking..."
	@uv run mypy fastwoe/ --check-untyped-defs
	@echo "[SUCCESS] Mypy type checking passed"

clean:  ## Clean build artifacts and virtual environments
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .venv/
	rm -rf .venv.*/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

check-all: format-check lint typecheck  ## Run all checks (format, lint, typecheck)

# CI-friendly target
ci-check: format-check lint  ## Run CI checks (format, lint, typecheck)
	@echo "🔍 Running mypy type checking..."
	@uv run mypy fastwoe/ --check-untyped-defs
	@echo "[SUCCESS] Mypy type checking passed"
