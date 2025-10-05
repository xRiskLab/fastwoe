# Type Checking with ty

This document explains the type checking setup for FastWoe using ty.

## Overview

FastWoe uses [ty](https://github.com/astral-sh/ty) for static type analysis. Due to the heavy use of pandas, numpy, and FAISS, many type errors are expected and appropriately ignored.

## Configuration

Type checking is configured through:

1. **pyproject.toml**: Basic ty configuration
2. **Makefile**: Convenient commands for development
3. **GitHub Actions**: CI/CD integration

## Usage

### Local Development

```bash
# Lenient type checking (recommended for development)
make typecheck

# Strict type checking (will show all errors)
make typecheck-strict
```

### CI/CD

The type checker automatically detects CI environments and uses lenient mode:

```bash
# In CI, this will pass even with expected pandas/numpy type issues
make ci-check
```

## Ignored Error Types

The following error types are ignored because they're inherent to pandas/numpy/faiss usage:

- **unresolved-reference**: Dynamic pandas/numpy attributes (`.empty`, `.idxmax`, `.isna`, etc.)
- **possibly-unbound-attribute**: Complex pandas/numpy operations
- **unresolved-import**: Optional dependencies like FAISS

## Expected Behavior

- **Development**: Type checking may show 1-5 remaining errors (expected)
- **CI (lenient mode)**: Passes with expected pandas/numpy type issues
- **CI (strict mode)**: May fail but continues build process

## Troubleshooting

### "ty not found"
```bash
uv add --dev ty
```

### Running type checking
```bash
make typecheck  # Uses configuration from pyproject.toml
```

### CI failures
The CI workflow automatically uses lenient mode for type checking.

## Adding New Code

When adding new code:

1. Add proper type annotations where possible
2. Use `# type: ignore[error-code]` for unavoidable pandas/numpy issues
3. Test with `make typecheck` locally
4. Ensure CI passes with lenient mode

## Philosophy

FastWoe prioritizes:
1. **Functionality over perfect typing**: Pandas/numpy code is complex
2. **Practical type safety**: Catch real bugs, ignore library limitations
3. **CI stability**: Don't break builds on expected type issues
4. **Developer experience**: Easy local development with helpful feedback
