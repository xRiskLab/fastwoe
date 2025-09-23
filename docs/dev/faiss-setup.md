# FAISS Setup Guide

FAISS (Facebook AI Similarity Search) is an optional dependency for FastWoe that enables efficient KMeans clustering-based binning for numerical features.

## Installation Options

### CPU Version (Recommended)
```bash
# Option 1: Install with FastWoe
pip install fastwoe[faiss]

# Option 2: Install directly
pip install faiss-cpu>=1.12.0
```

### GPU Version (CUDA 12)
```bash
# Option 1: Install with FastWoe
pip install fastwoe[faiss-gpu]

# Option 2: Install directly
pip install faiss-gpu-cu12>=1.12.0
```

## Usage

### Basic Usage (CPU)
```python
from fastwoe import FastWoe

# Use FAISS KMeans clustering for binning
woe = FastWoe(
    binning_method="faiss_kmeans",
    faiss_kwargs={
        "k": 5,              # Number of clusters
        "niter": 20,         # Number of iterations
        "verbose": False,    # Show progress
        "gpu": False         # Use CPU
    }
)
```

### GPU Usage
```python
from fastwoe import FastWoe

# Use GPU acceleration (requires faiss-gpu-cu12)
woe = FastWoe(
    binning_method="faiss_kmeans",
    faiss_kwargs={
        "k": 10,
        "niter": 50,
        "verbose": True,
        "gpu": True          # Use GPU if available
    }
)
```

## Compatibility

- **Python**: 3.7-3.12
- **NumPy**: Both 1.x and 2.x
- **CUDA**: Version 12 for GPU support

## Without FAISS

If FAISS is not installed, you can still use FastWoe with other binning methods:

```python
from fastwoe import FastWoe

# Use KBinsDiscretizer (default)
woe = FastWoe(binning_method="kbins")

# Use decision tree-based binning
woe = FastWoe(binning_method="tree")
```

## Troubleshooting

### Import Errors
If you get `ModuleNotFoundError: No module named 'faiss'`:
```bash
pip install faiss-cpu>=1.12.0
# or
pip install faiss-gpu-cu12>=1.12.0
```

### NumPy Compatibility Issues
If you get NumPy-related errors:
```bash
# Upgrade to compatible versions
pip install --upgrade faiss-cpu>=1.12.0 numpy>=1.21.0
```

### GPU Issues
If GPU version doesn't work:
1. Check CUDA version: `nvidia-smi`
2. Ensure CUDA 12 compatibility
3. Fallback to CPU version: `pip install faiss-cpu>=1.12.0`

## Performance Comparison

| Method | Speed | Memory | Use Case |
|--------|-------|--------|----------|
| `kbins` | Fast | Low | General purpose |
| `tree` | Medium | Medium | Feature-aware binning |
| `faiss_kmeans` (CPU) | Medium | Medium | Large datasets |
| `faiss_kmeans` (GPU) | Very Fast | High | Very large datasets |

## When to Use FAISS

- **Large datasets** (>10k samples)
- **High-dimensional features**
- **Need for clustering-based bins**
- **GPU acceleration available**

For most use cases, the default `kbins` method is sufficient and doesn't require additional dependencies.
