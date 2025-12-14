# Marginal Somers' D (MSD) for Feature Selection

**Denis Burakov** | **December 2025** | **xRiskLab**

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-FastWoe-black?logo=github)](https://github.com/xRiskLab/fastwoe)
[![PyPI](https://img.shields.io/badge/PyPI-fastwoe-blue?logo=pypi)](https://pypi.org/project/fastwoe/)

**Fast and efficient Python implementation of WOE encoding and MSD feature selection**

</div>

---

## 1. Introduction

**Marginal Somers' D (MSD)** is a feature selection method that uses rank correlation (Somers' D) instead of traditional Information Value (IV). It implements greedy forward selection that:

1. Transforms features using WOE encoding
2. Selects features based on their Somers' D with the target
3. Filters out features highly correlated with already-selected features
4. Works with both binary and continuous targets

**Key advantage:** Unlike IV-based methods limited to binary classification, MSD handles continuous targets through rank correlation.

---

## 2. Mathematical Foundation

### Somers' D Definition

Somers' D measures monotonic association between two variables:

```math
D_{Y|X} = \frac{\text{Concordant} - \text{Discordant}}{\text{Total pairs (excluding ties in Y)}}
```

Where:
- **Concordant**: $(x_i > x_j \text{ and } y_i > y_j)$ or $(x_i < x_j \text{ and } y_i < y_j)$
- **Discordant**: $(x_i > x_j \text{ and } y_i < y_j)$ or $(x_i < x_j \text{ and } y_i > y_j)$

For binary classification:

```math
\text{Gini} = 2 \times \text{AUC} - 1 = D_{Y|X}
```

### WOE Transformation

All features are transformed using Weight of Evidence (WOE) before computing Somers' D. This:
- Handles categorical variables
- Creates monotonic transformations
- Works with both binary and continuous targets

---

## 3. The MSD Algorithm

### Step-by-Step Process

1. **Pre-processing**
   - Transform all features using WOE encoding
   - Compute pairwise Somers' D correlation matrix

2. **Initialization**
   - Calculate univariate Somers' D for each feature
   - Select feature with highest univariate Somers' D

3. **Iterative Selection**
   - Fit model with currently selected features
   - For each remaining feature:
     - Compute univariate Somers' D between feature WOE and target
     - Check correlation with already-selected features (using pairwise feature correlation)
   - Add feature with highest Somers' D if correlation < threshold
   - Repeat until stopping criteria met

4. **Stopping Criteria**
   - Marginal Somers' D < `min_msd`, OR
   - Maximum features reached, OR
   - All remaining features too correlated with selected features

### What Makes It "Marginal"?

The "marginal" aspect comes from:
- **Iterative evaluation**: Features are evaluated at each step after some features are already selected
- **Correlation filtering**: Features with Somers' D correlation > threshold with already-selected features are skipped
- **Greedy selection**: The selection order implicitly accounts for redundancy through correlation filtering

> [!NOTE]
> The term "marginal" here refers to the iterative, step-wise evaluation process. At each step, features are evaluated using their univariate Somers' D with the target, but redundant features are filtered out based on their correlation with already-selected features.

Feature correlation is computed as:

```math
\text{correlation}(f_i, f_j) = \frac{|D_{f_i|f_j}| + |D_{f_j|f_i}|}{2}
```

---

## 4. Basic Usage

### Binary Classification

```python
import numpy as np
import pandas as pd
from fastwoe.modeling import marginal_somersd_selection

# Prepare data
X = pd.DataFrame({
    'feature1': np.random.choice(['A', 'B', 'C'], 1000),
    'feature2': np.random.choice(['X', 'Y', 'Z'], 1000),
    'feature3': np.random.choice(['P', 'Q', 'R'], 1000),
})
y = np.random.binomial(1, 0.3, 1000)

# Run selection
result = marginal_somersd_selection(
    X, y,
    min_msd=0.01,              # Minimum marginal Somers' D
    max_features=5,            # Maximum features
    correlation_threshold=0.5  # Correlation threshold
)

print(result['selected_features'])
print(result['msd_history'])
```

### With Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

result = marginal_somersd_selection(
    X_train, y_train,
    X_test=X_test,
    y_test=y_test,
    min_msd=0.01
)

# Monitor performance at each step
# Note: test_performance has length len(selected_features) - 1
# (computed at start of each iteration after first feature)
for i, (feat, msd) in enumerate(zip(
    result['selected_features'],
    result['msd_history']
)):
    if i > 0:  # test_performance starts from step 2
        test_perf = result['test_performance'][i - 1]
        print(f"{feat}: Train MSD={msd:.4f}, Test D={test_perf:.4f}")
    else:
        print(f"{feat}: Train MSD={msd:.4f}")
```

### Continuous Target

```python
# Works with continuous targets
y_continuous = np.random.normal(0, 1, 1000)

result = marginal_somersd_selection(
    X, y_continuous,
    min_msd=0.01
)
```

---

## 5. Output Structure

The function returns a dictionary with:

| Key | Type | Description |
|-----|------|-------------|
| `selected_features` | `list[str]` | Feature names in selection order |
| `msd_history` | `list[float]` | Marginal Somers' D at each step (same length as selected_features) |
| `univariate_somersd` | `dict[str, float]` | Univariate Somers' D for all features |
| `model` | `FastWoe` | Trained WOE model with selected features |
| `test_performance` | `list[float]` | Test Somers' D at each step (length = len(selected_features) - 1, if test set provided) |
| `correlation_matrix` | `pd.DataFrame` | Pairwise correlations of selected features |

---

## 6. When to Use MSD

**Use MSD when:**
- You have categorical or mixed-type features
- You need rank correlation-based selection (robust to outliers)
- You want to handle both binary and continuous targets
- You want automatic redundancy filtering
- You're building credit scoring or risk models

**Consider alternatives when:**
- You have extremely high-dimensional data (thousands of features)
- You need very fast selection with minimal computation
- Your features are already numeric and well-scaled

---

## 7. Complete Example

```python
import numpy as np
import pandas as pd
from fastwoe.modeling import marginal_somersd_selection
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Generate data
np.random.seed(42)
n = 2000
X = pd.DataFrame({
    'age_group': np.random.choice(['18-25', '26-35', '36-45', '46+'], n),
    'income': np.random.choice(['Low', 'Medium', 'High'], n),
    'employment': np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], n),
    'education': np.random.choice(['HS', 'Bachelor', 'Master', 'PhD'], n),
})

# Create target
y = (
    (X['income'] == 'High').astype(int) * 0.3 +
    (X['education'].isin(['Master', 'PhD'])).astype(int) * 0.2 +
    np.random.normal(0, 0.1, n)
)
y = (y > 0.3).astype(int)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Select features
result = marginal_somersd_selection(
    X_train, y_train,
    X_test=X_test,
    y_test=y_test,
    min_msd=0.01,
    max_features=5
)

# Results
print("Selected features:", result["selected_features"])
print("\nUnivariate Somers' D:")
for feat, val in sorted(
    result["univariate_somersd"].items(), key=lambda x: x[1], reverse=True
):
    print(f"{feat}: {val:.4f}")

# Evaluate
model = result["model"]
y_pred = model.predict_proba(X_test[result["selected_features"]])[:, 1]
print(f"\nTest AUC: {roc_auc_score(y_test, y_pred):.4f}")
```

---

## References

1. Somers, R.H. (1962). A new asymmetric measure of association for ordinal variables. *American Sociological Review*, 27(6), 799-811.

2. Spinella, F., & Krisciunas, T. (2025). Enhancing Credit Risk Models at Revolut by Combining Deep Feature Synthesis and Marginal Information Value. *Credit Research Centre, University of Edinburgh Business School*. Available at: https://www.crc.business-school.ed.ac.uk/sites/crc/files/2025-11/Enhancing-Credit-Risk-Models-at-Revolut-by-combining-Deep-Feature-Synthesis-and-Marginal-Information-Value-paper.pdf
