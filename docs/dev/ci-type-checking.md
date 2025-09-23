# CI Type Checking Behavior

This document explains exactly what happens when the type checking workflow runs in CI.

## Workflow Overview

The `.github/workflows/type-check.yml` workflow:

1. **Runs on**: Push to `main`/`develop`, PRs to `main`/`develop`
2. **Python version**: 3.11 (single version for speed)
3. **Two steps**: CI mode (required) + Strict mode (informational)

## Expected Behavior

### ✅ CI Mode (Required Step)
```bash
CI=true uv run type-check
```

**Expected outcome**: ✅ **PASS**
- Ignores 65+ pandas/numpy/faiss type errors
- Shows ~1 remaining error (expected)
- Returns exit code 0 (success)
- Build continues

**What it does**:
- Runs pyrefly with comprehensive ignore flags
- Detects CI environment automatically
- Treats expected data science library type issues as success
- Only fails on genuine type errors in your code

### ℹ️ Strict Mode (Informational Step)
```bash
PYREFLY_STRICT=true uv run type-check
```

**Expected outcome**: ⚠️ **SHOWS ISSUES** (but doesn't fail build)
- Shows all type errors including pandas/numpy issues
- Returns exit code 1 (but `continue-on-error: true`)
- Provides full type checking information
- Build continues regardless

## Failure Scenarios

### ❌ When CI Mode Will Fail

The CI mode will only fail if:

1. **Script execution fails**: pyrefly not installed, import errors, etc.
2. **New genuine type errors**: Errors in your code logic (not pandas/numpy)
3. **Missing dependencies**: Required packages not available

### ✅ When CI Mode Will Pass

CI mode will pass even with:
- pandas DataFrame constructor type issues
- numpy array method inference problems
- FAISS API type complexity
- scipy.stats import edge cases
- Complex generic type inference

## Debugging CI Failures

If the workflow fails:

1. **Check the logs** for the specific error
2. **Test locally**:
   ```bash
   CI=true uv run type-check  # Should pass
   ```
3. **Check for new code issues**:
   ```bash
   PYREFLY_STRICT=true uv run type-check  # Shows all errors
   ```

## Configuration

### Making CI Stricter
To fail on any type errors, uncomment in the workflow:
```yaml
env:
  CI: "true"
  PYREFLY_STRICT: "true"  # Uncomment this line
```

### Making CI More Lenient
The current configuration is already lenient. If you need even more leniency, you can:
1. Add more ignore flags in `fastwoe/scripts/type_check.py`
2. Exclude more files in the pyrefly configuration

## Summary

**The workflow is designed to be CI-friendly**:
- ✅ Won't break your builds on expected pandas/numpy type complexity
- ✅ Will catch genuine type errors in your application logic
- ✅ Provides both lenient (CI) and strict (informational) feedback
- ✅ Easy to debug and configure based on your needs
