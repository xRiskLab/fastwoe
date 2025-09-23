# Local GitHub Actions Testing with Act

This guide explains how to use [act](https://github.com/nektos/act) to run your GitHub Actions workflows locally, providing fast feedback and debugging capabilities for your CI/CD pipelines.

## What is Act?

Act is a tool that allows you to run GitHub Actions locally using Docker. It reads your `.github/workflows/` files and executes them in a local environment that mimics GitHub's runner environment.

## Benefits

- **Fast Feedback**: Test workflow changes without pushing to GitHub
- **Debugging**: Identify issues before they reach CI
- **Resource Efficiency**: No GitHub Actions minutes consumed
- **Local Development**: Validate syntax and logic locally
- **Matrix Testing**: Test all Python version combinations

## Installation

### macOS (using Homebrew)
```bash
brew install act
```

### Other Platforms
See the [act installation guide](https://nektosact.com/installation/) for your platform.

## Prerequisites

- **Docker**: Act requires Docker to be running
- **Docker Desktop**: Recommended for macOS/Windows users

### Starting Docker
```bash
# macOS
open -a Docker

# Linux
sudo systemctl start docker
```

## Configuration

### Apple Silicon (M1/M2/M3) Setup

If you're on Apple Silicon, you'll need to specify the container architecture:

```bash
# Create act configuration
mkdir -p "/Users/$USER/Library/Application Support/act"
echo "-P ubuntu-latest=catthehacker/ubuntu:act-latest" > "/Users/$USER/Library/Application Support/act/actrc"
```

### Environment Variables

Act reads from your `.env` file. Ensure environment variable names use underscores (not hyphens):

```bash
# ✅ Correct format
fastwoe_token=pypi-...
xrisklab_token=pypi-...

# ❌ Incorrect format (act doesn't like hyphens)
fastwoe-token=pypi-...
xrisklab-token=pypi-...
```

## Basic Usage

### List Available Workflows
```bash
act --container-architecture linux/amd64 --list
```

### Run All Jobs in a Workflow
```bash
# Run the CI workflow
act --container-architecture linux/amd64 -W .github/workflows/ci.yml

# Run the type checking workflow
act --container-architecture linux/amd64 -W .github/workflows/typecheck.yml

# Run compatibility tests
act --container-architecture linux/amd64 -W .github/workflows/compatibility.yml
```

### Run Specific Jobs
```bash
# Run only the test job
act --container-architecture linux/amd64 -j test -W .github/workflows/ci.yml

# Run only the lint job
act --container-architecture linux/amd64 -j lint -W .github/workflows/ci.yml

# Run only the type-check job
act --container-architecture linux/amd64 -j type-check -W .github/workflows/ci.yml
```

### Dry Run (Recommended for Testing)
```bash
# Validate workflow without executing
act --container-architecture linux/amd64 -W .github/workflows/ci.yml --dryrun

# Test specific job
act --container-architecture linux/amd64 -j lint -W .github/workflows/ci.yml --dryrun
```

## FastWoe-Specific Examples

### Testing New Features

When testing Kiro's new IV standard errors functionality:

```bash
# Test the enhanced type checking workflow
act --container-architecture linux/amd64 -W .github/workflows/typecheck.yml --dryrun

# Test CI type checking
act --container-architecture linux/amd64 -j type-check -W .github/workflows/ci.yml --dryrun
```

### Matrix Testing

Test specific Python versions:

```bash
# Test Python 3.11 specifically
act --container-architecture linux/amd64 -j test -W .github/workflows/ci.yml --matrix python-version:3.11 --dryrun

# Test Python 3.12 specifically
act --container-architecture linux/amd64 -j test -W .github/workflows/ci.yml --matrix python-version:3.12 --dryrun
```

### Event Simulation

Test different trigger events:

```bash
# Test pull request event
act --container-architecture linux/amd64 -W .github/workflows/ci.yml --event pull_request --dryrun

# Test push event (default)
act --container-architecture linux/amd64 -W .github/workflows/ci.yml --event push --dryrun
```

## Workflow-Specific Commands

### CI Workflow (`ci.yml`)
```bash
# Test all CI jobs
act --container-architecture linux/amd64 -W .github/workflows/ci.yml --dryrun

# Test only tests (Python 3.9-3.12 matrix)
act --container-architecture linux/amd64 -j test -W .github/workflows/ci.yml --dryrun

# Test only linting
act --container-architecture linux/amd64 -j lint -W .github/workflows/ci.yml --dryrun

# Test only type checking
act --container-architecture linux/amd64 -j type-check -W .github/workflows/ci.yml --dryrun
```

### Type Checking Workflow (`typecheck.yml`)
```bash
# Test the dedicated type checking workflow
act --container-architecture linux/amd64 -W .github/workflows/typecheck.yml --dryrun

# This workflow includes:
# - Python 3.11 setup
# - uv installation
# - Dependencies installation
# - FastWoe & pyrefly verification
# - CI checks (format, lint, typecheck)
# - Strict type checking
```

### Compatibility Workflow (`compatibility.yml`)
```bash
# Test Python version compatibility
act --container-architecture linux/amd64 -W .github/workflows/compatibility.yml --dryrun
```

### Release Workflow (`release.yml`)
```bash
# Test release process
act --container-architecture linux/amd64 -W .github/workflows/release.yml --dryrun
```

## Advanced Usage

### Verbose Output
```bash
act --container-architecture linux/amd64 -W .github/workflows/ci.yml -v --dryrun
```

### Custom Environment Variables
```bash
act --container-architecture linux/amd64 -W .github/workflows/ci.yml --env CUSTOM_VAR=value --dryrun
```

### Skip Jobs
```bash
# Skip specific jobs
act --container-architecture linux/amd64 -W .github/workflows/ci.yml --skip-job test --dryrun
```

### Use Different Docker Image
```bash
# Use a different base image
act --container-architecture linux/amd64 -P ubuntu-latest=ubuntu:22.04 -W .github/workflows/ci.yml --dryrun
```

## Troubleshooting

### Common Issues

#### Docker Not Running
```bash
# Error: Cannot connect to the Docker daemon
# Solution: Start Docker Desktop
open -a Docker  # macOS
```

#### Apple Silicon Architecture Issues
```bash
# Error: Container architecture mismatch
# Solution: Always use --container-architecture linux/amd64
act --container-architecture linux/amd64 [other-options]
```

#### Environment Variable Issues
```bash
# Error: unexpected character "-" in variable name
# Solution: Use underscores instead of hyphens in .env file
fastwoe_token=...  # ✅ Correct
fastwoe-token=...   # ❌ Incorrect
```

#### Image Pull Issues
```bash
# Error: Failed to pull image
# Solution: Check Docker is running and has internet access
docker pull catthehacker/ubuntu:act-latest
```

### Debugging Tips

1. **Start with dry runs**: Always use `--dryrun` first to validate syntax
2. **Test individual jobs**: Use `-j job-name` to isolate issues
3. **Check Docker**: Ensure Docker is running and accessible
4. **Verify environment**: Check your `.env` file format
5. **Use verbose mode**: Add `-v` for detailed output

## Integration with Development Workflow

### Pre-commit Testing
```bash
# Test workflows before committing
act --container-architecture linux/amd64 -W .github/workflows/ci.yml --dryrun

# If dry run passes, commit your changes
git add .
git commit -m "Add new feature"
git push
```

### Feature Development
```bash
# Test new workflow changes
act --container-architecture linux/amd64 -W .github/workflows/typecheck.yml --dryrun

# Test specific functionality
act --container-architecture linux/amd64 -j type-check -W .github/workflows/ci.yml --dryrun
```

### Release Preparation
```bash
# Test release workflow
act --container-architecture linux/amd64 -W .github/workflows/release.yml --dryrun

# Test compatibility across Python versions
act --container-architecture linux/amd64 -W .github/workflows/compatibility.yml --dryrun
```

## Best Practices

1. **Always use dry runs first**: Validate syntax before execution
2. **Test individual jobs**: Isolate issues by testing specific jobs
3. **Use consistent architecture flags**: Always use `--container-architecture linux/amd64` on Apple Silicon
4. **Keep environment clean**: Ensure `.env` file uses proper naming conventions
5. **Test matrix combinations**: Verify all Python version combinations work
6. **Validate new workflows**: Test new workflow files before pushing

## Resources

- [Act Documentation](https://nektosact.com/)
- [Act GitHub Repository](https://github.com/nektos/act)
- [Docker Documentation](https://docs.docker.com/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

## FastWoe Workflow Overview

Your project includes these workflows:

- **`ci.yml`**: Main CI pipeline with tests, linting, and type checking
- **`typecheck.yml`**: Dedicated type checking with pyrefly integration
- **`compatibility.yml`**: Python version compatibility testing
- **`release.yml`**: Release automation and publishing

Each workflow can be tested locally using the commands above, providing fast feedback for your development process.
