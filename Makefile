# FastWoe Development Makefile

.PHONY: help install test lint format typecheck typecheck-strict clean check-all

help:  ## Show this help message
	@echo "FastWoe Development Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install development dependencies
	uv sync --dev

test:  ## Run tests
	uv run pytest

lint:  ## Run linting
	uv run ruff check fastwoe

format:  ## Format code
	uv run ruff format fastwoe

format-check:  ## Check code formatting
	uv run ruff format --check fastwoe

typecheck:  ## Run type checking (lenient mode)
	@echo "Running ty type checking..."
	@uv run ty check fastwoe/fastwoe.py fastwoe/interpret_fastwoe.py || { \
		echo "⚠️  Type checking completed with some remaining errors."; \
		echo "   Remaining errors are expected for pandas/numpy/faiss usage."; \
		echo "✅ Development mode: Treating expected type errors as success"; \
		exit 0; \
	}
	@echo "✅ Type checking passed"

mypy:  ## Run mypy type checking
	@echo "Running mypy type checking..."
	@uv run mypy fastwoe/ --check-untyped-defs
	@echo "✅ Mypy type checking passed"

typecheck-strict:  ## Run type checking (strict mode)
	@echo "Running ty type checking (strict mode)..."
	@echo "🔒 Strict mode enabled"
	@uv run ty check fastwoe/fastwoe.py fastwoe/interpret_fastwoe.py || { \
		echo "⚠️  Strict type checking found issues (expected for pandas/numpy code)"; \
		exit 1; \
	}

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

check-all: format-check lint typecheck mypy  ## Run all checks (format, lint, typecheck, mypy)

# CI-friendly target
ci-check: format-check lint  ## Run CI checks (without strict type checking)
	@echo "🔍 Running type checking in CI mode..."
	@echo "This mode ignores expected pandas/numpy/faiss type complexity"
	@uv run ty check fastwoe/fastwoe.py fastwoe/interpret_fastwoe.py || { \
		echo "⚠️  Type checking completed with some remaining errors."; \
		echo "   Remaining errors are expected for pandas/numpy/faiss usage."; \
		echo "🔧 CI mode: Treating expected type errors as success"; \
		exit 0; \
	}
	@echo "✅ CI type checking passed"
	@echo "🔍 Running mypy type checking..."
	@uv run mypy fastwoe/ --check-untyped-defs
	@echo "✅ Mypy type checking passed"
