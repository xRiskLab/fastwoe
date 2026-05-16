# Piecewise WOE Usage Guide

## Overview

FastWoe supports piecewise Weight of Evidence encoding through the `PiecewiseWoeMixin`. Instead of collapsing all bins of a feature into a single WOE-transformed variable, bins are grouped into *pieces* -- each piece becomes its own column so that downstream logistic regression can learn a separate coefficient per piece.

This implements the approach described by Raymond Anderson (Standard Bank of South Africa) in *Piecewise Logistic Regression: an Application in Credit Scoring*, presented at Credit Scoring and Control Conference XIV, Edinburgh, 2015.

## Background

In standard WOE-based scorecard development, each characteristic (feature) is represented by a single variable carrying its WOE value. Logistic regression then learns one coefficient per characteristic. This assumes a linear relationship between the WOE and the log-odds -- which may not hold when a characteristic has complex non-linear patterns.

Anderson (2015) describes three approaches:

| Approach | Variables per characteristic | Description |
|---|---|---|
| **Base Case** | 1 | Single WOE value per characteristic (standard) |
| **Piecewise** | 2-3 typically | Bins grouped into pieces, each with its own coefficient |
| **Dummy** | = number of bins | One variable per bin |

The piecewise approach is a "middle way" that often outperforms both the base case and the dummy approach in terms of Gini coefficient, while keeping the model interpretable.

## Basic Usage

### 1. Fit and Assign Pieces

```python
import pandas as pd
import numpy as np
from fastwoe import FastWoe

# Prepare data
X = pd.DataFrame({
    'income': np.random.choice(['low', 'mid', 'high', 'very_high'], 1000),
    'age': np.random.choice(['young', 'middle', 'senior'], 1000),
})
y = pd.Series(np.random.binomial(1, 0.3, 1000))

# Fit the model (standard WOE computation)
woe = FastWoe()
woe.fit(X, y)

# Assign pieces using the sign strategy
woe.assign_pieces(strategy="sign")
```

### 2. Inspect Piece Assignments

```python
# View the mapping with piece labels
for feature, mapping in woe.mappings_.items():
    print(f"\n{feature}:")
    print(mapping[["woe", "piece"]])
```

Output:

```
income:
                woe  piece
category
high       0.2597      1
low       -0.2912      0
mid       -0.1336      0
very_high  0.0892      1

age:
               woe  piece
category
middle    0.0562      1
senior   -0.1948      0
young     0.0950      1
```

### 3. Transform with Piecewise Output

```python
# Standard output: 1 column per feature
X_woe = woe.transform(X, output="woe")
print(X_woe.shape)  # (1000, 2)

# Piecewise output: 1 column per (feature, piece) pair
X_pw = woe.transform(X, output="piecewise")
print(X_pw.shape)  # (1000, 4)
print(X_pw.columns.tolist())
# ['income__piece_0', 'income__piece_1', 'age__piece_0', 'age__piece_1']
```

Each piecewise column contains the WOE value when the observation's bin belongs to that piece, and 0 otherwise.

## Piece Assignment Strategies

### Sign Strategy (Default)

Splits bins into two pieces based on the sign of the WOE value:

- **Piece 0**: bins with WOE < 0 (higher risk than average)
- **Piece 1**: bins with WOE >= 0 (lower risk than average)

```python
woe.assign_pieces(strategy="sign")
```

This is the simplest heuristic and aligns with Anderson's recommendation of starting with a negative/positive split.

### Custom Piece Map

For full control, supply a dictionary mapping each category to a piece index:

```python
woe.assign_pieces(piece_map={
    "income": {"low": 0, "mid": 1, "high": 1, "very_high": 2},
    "age": {"young": 0, "middle": 0, "senior": 1},
})

X_pw = woe.transform(X, output="piecewise")
print(X_pw.columns.tolist())
# ['income__piece_0', 'income__piece_1', 'income__piece_2',
#  'age__piece_0', 'age__piece_1']
```

You can mix strategies: features in `piece_map` use the custom assignment, while features not in `piece_map` fall back to the `strategy` parameter:

```python
# Custom pieces for income, sign strategy for everything else
woe.assign_pieces(
    strategy="sign",
    piece_map={"income": {"low": 0, "mid": 0, "high": 1, "very_high": 1}},
)
```

## Using Piecewise Output with Logistic Regression

The main motivation is to feed the piecewise columns into logistic regression, giving each piece its own coefficient:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit WOE and assign pieces
woe = FastWoe()
woe.fit(X_train, y_train)
woe.assign_pieces(strategy="sign")

# Transform
X_train_pw = woe.transform(X_train, output="piecewise")
X_test_pw = woe.transform(X_test, output="piecewise")

# Fit logistic regression on piecewise features
lr = LogisticRegression(penalty=None)
lr.fit(X_train_pw, y_train)

# Each piece gets its own coefficient
for name, coef in zip(X_train_pw.columns, lr.coef_[0]):
    print(f"{name}: {coef:.4f}")
```

Compare with the standard approach:

```python
X_train_woe = woe.transform(X_train, output="woe")
X_test_woe = woe.transform(X_test, output="woe")

lr_base = LogisticRegression(penalty=None)
lr_base.fit(X_train_woe, y_train)

from sklearn.metrics import roc_auc_score

auc_base = roc_auc_score(y_test, lr_base.predict_proba(X_test_woe)[:, 1])
auc_pw = roc_auc_score(y_test, lr.predict_proba(X_test_pw)[:, 1])

print(f"Base Case AUC: {auc_base:.4f}")
print(f"Piecewise AUC: {auc_pw:.4f}")
```

## How It Works

Given a feature with bins A, B, C, D and WOE values:

| Bin | WOE   | Piece (sign strategy) |
|-----|-------|-----------------------|
| A   | -0.30 | 0                     |
| B   | -0.10 | 0                     |
| C   |  0.15 | 1                     |
| D   |  0.40 | 1                     |

Standard WOE produces one column:

| Row | feature |
|-----|---------|
| 1   | -0.30   |
| 2   |  0.15   |
| 3   |  0.40   |

Piecewise produces two columns:

| Row | feature__piece_0 | feature__piece_1 |
|-----|------------------|------------------|
| 1   | -0.30            | 0.00             |
| 2   |  0.00            | 0.15             |
| 3   |  0.00            | 0.40             |

Logistic regression learns separate coefficients for piece_0 and piece_1, allowing different slopes for the "risky" and "safe" regions of the characteristic.

## Key Differences from Standard WOE

1. **Multiple columns per feature**: Each piece is a separate regression input
2. **Separate coefficients**: The model can weight different risk regions independently
3. **Same WOE values**: The underlying WOE computation is identical -- only the output structure changes
4. **Post-fit step**: `assign_pieces()` is called after `fit()`, so you can experiment with different piece assignments without refitting

## Best Practices

1. **Start with the sign strategy**: A negative/positive split is the simplest and most interpretable starting point
2. **Review piece assignments**: Check `mappings_[feature][["woe", "piece"]]` to ensure the groupings make domain sense
3. **Watch for small pieces**: Pieces with very few observations may lead to unstable coefficients -- consider merging them
4. **Enforce positive coefficients**: Anderson recommends ensuring all beta coefficients are positive (use `LogisticRegression` with bounds or manual review)
5. **Compare Gini across approaches**: Test base case, piecewise, and dummy approaches on out-of-time samples
6. **Monitor VIF**: Piecewise variables from the same characteristic are correlated by construction -- check variance inflation factors

## Limitations

- Not supported for multiclass targets (raises `ValueError`)
- The `"sign"` strategy always produces exactly 2 pieces per feature; use `piece_map` for finer control
- `assign_pieces()` must be called before `transform(output="piecewise")`

## Reference

Anderson, R. (2015). *Piecewise Logistic Regression: an Application in Credit Scoring*. Presented at Credit Scoring and Control Conference XIV, Edinburgh, 26-28 August 2015.
