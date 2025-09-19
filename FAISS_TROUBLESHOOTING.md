# FAISS Troubleshooting Guide for FastWoe

## The Problem

You're getting this error:
```
ImportError: FAISS is required for faiss_kmeans binning method. Install it with: pip install faiss-cpu
```

This happens when you try to use `binning_method='faiss_kmeans'` but FAISS is not properly installed or accessible.

## Quick Solutions

### Option 1: Install FastWoe with FAISS support (Recommended)
```bash
# Using pip
pip install 'fastwoe[faiss]'

# Using uv
uv add 'fastwoe[faiss]'
```

### Option 2: Install FAISS separately
```bash
# CPU version (recommended for most users)
pip install faiss-cpu

# GPU version (if you have CUDA)
pip install faiss-gpu
```

### Option 3: Use a different binning method
If you don't need FAISS KMeans, you can use other binning methods:
```python
# Use default KBinsDiscretizer
woe = FastWoe()  # binning_method='kbins' by default

# Use tree-based binning
woe = FastWoe(binning_method='tree')
```

## Verification

After installing, verify it works:
```python
import faiss
print("FAISS version:", faiss.__version__)

from fastwoe import FastWoe
import pandas as pd
import numpy as np

# Test data
X = pd.DataFrame({'feature': np.random.randn(100)})
y = np.random.binomial(1, 0.3, 100)

# Test FAISS KMeans
woe = FastWoe(binning_method='faiss_kmeans', faiss_kwargs={'k': 3})
woe.fit(X, y)
print("✅ FAISS KMeans works!")
```

## Common Issues

### 1. Wrong Environment
Make sure you're installing in the correct virtual environment:
```bash
# Check current environment
which python
pip list | grep fastwoe
pip list | grep faiss
```

### 2. Multiple Python Installations
If you have multiple Python versions, make sure you're using the right one:
```bash
# Use specific Python version
python3.11 -m pip install 'fastwoe[faiss]'
```

### 3. Permission Issues
If you get permission errors, use `--user` flag:
```bash
pip install --user 'fastwoe[faiss]'
```

### 4. Version Conflicts
If you have version conflicts, try upgrading:
```bash
pip install --upgrade 'fastwoe[faiss]'
```

## Troubleshooting Script

Run this script to diagnose issues:
```python
# Save as faiss_troubleshooting.py and run
python faiss_troubleshooting.py
```

## Correct Usage

Once FAISS is installed, use it like this:
```python
from fastwoe import FastWoe
import pandas as pd
import numpy as np

# Create your data
X = pd.DataFrame({'feature': np.random.randn(1000)})
y = np.random.binomial(1, 0.3, 1000)

# Use FAISS KMeans binning
woe = FastWoe(
    binning_method='faiss_kmeans',
    faiss_kwargs={
        'k': 5,           # Number of clusters
        'niter': 20,      # Number of iterations
        'verbose': False, # Verbose output
        'gpu': False      # Use GPU (requires faiss-gpu)
    }
)

# Fit and transform
woe.fit(X, y)
X_transformed = woe.transform(X)
```

## Still Having Issues?

1. **Check your Python version**: FAISS 1.12.0 supports Python 3.7-3.12
2. **Check your NumPy version**: FAISS works with both NumPy 1.x and 2.x
3. **Try a fresh environment**: Create a new virtual environment and install from scratch
4. **Check the logs**: Look for more detailed error messages in the full traceback

## Need Help?

If you're still having issues, please share:
1. Your Python version: `python --version`
2. Your pip/uv version: `pip --version` or `uv --version`
3. The full error traceback
4. What you tried to install: `pip list | grep -E "(fastwoe|faiss)"`
