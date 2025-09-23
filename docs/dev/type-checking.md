# Type Checking with Pyrefly

This document explains the type checking setup for FastWoe using pyrefly.

## Overview

FastWoe uses [pyrefly](https://pyrefly.org/) for static type analysis. Due to the heavy use of pandas, numpy, and FAISS, many type errors are expected and appropriately ignored.

## Configuration

Type checking is configured through:

1. **pyproject.toml**: Basic pyrefly configuration
2. **scripts/typecheck.py**: Standalone script with ignore flags
3. **Makefile**: Convenient commands for development
4. **GitHub Actions**: CI/CD integration

## Usage

### Local Development

```bash
# Lenient type checking (recommended for development)
make typecheck
# or
python scripts/typecheck.py

# Strict type checking (will show all errors)
make typecheck-strict
# or
PYREFLY_STRICT=true python scripts/typecheck.py
```

### CI/CD

The type checker automatically detects CI environments and uses lenient mode:

```bash
# In CI, this will pass even with expected pandas/numpy type issues
CI=true python scripts/typecheck.py
```

To enable strict mode in CI, set `PYREFLY_STRICT=true`.

## Ignored Error Types

The following error types are ignored because they're inherent to pandas/numpy/faiss usage:

- **missing-attribute**: Dynamic pandas/numpy attributes (`.empty`, `.idxmax`, `.isna`, etc.)
- **bad-argument-type**: pandas DataFrame constructor complexity
- **unsupported-operation**: Complex pandas/numpy operations
- **not-iterable**: Type checker can't infer pandas iterability
- **no-matching-overload**: `zip()` and `str.join()` with dynamic types
- **missing-argument**: FAISS API type inference issues
- **bad-return**: Complex pandas return types
- **bad-assignment**: pandas Series.mean() return type inference
- **missing-module-attribute**: scipy.stats import edge cases

## Expected Behavior

- **Development**: Type checking may show 1-5 remaining errors (expected)
- **CI (lenient mode)**: Passes with expected pandas/numpy type issues
- **CI (strict mode)**: May fail but continues build process

## Troubleshooting

### "pyrefly not found"
```bash
uv add --dev pyrefly
```

### Too many type errors
The ignore flags should handle most pandas/numpy issues. If you see many errors, ensure you're using the custom script:
```bash
python scripts/typecheck.py  # Uses ignore flags
# NOT: pyrefly check fastwoe/  # Raw pyrefly without ignores
```

### CI failures
Set lenient mode in your workflow:
```yaml
env:
  CI: "true"
  # PYREFLY_STRICT: "true"  # Uncomment for strict mode
```

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
