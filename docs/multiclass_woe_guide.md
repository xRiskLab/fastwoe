# Multiclass WOE Usage Guide

## Overview

FastWoe now supports multiclass targets through the `MulticlassWoeMixin`. This guide shows you how to use multiclass functionality and provides utility methods for analysis.

## Basic Usage

### 1. Creating Multiclass Targets

```python
from fastwoe import FastWoe
from examples.multiclass_woe_utils import MulticlassWoeUtils

# Method 1: From continuous target using thresholds
y_continuous = np.random.normal(0, 1, 1000)
y_multiclass = MulticlassWoeUtils.create_multiclass_target(
    y_continuous, method="threshold", n_bins=3
)

# Method 2: From continuous target using quantiles
y_multiclass = MulticlassWoeUtils.create_multiclass_target(
    y_continuous, method="quantile", n_bins=4
)

# Method 3: Custom thresholds
y_multiclass = MulticlassWoeUtils.create_multiclass_target(
    y_continuous, method="threshold", thresholds=[-0.5, 0.5]
)
```

### 2. Fitting and Predicting

```python
# Fit the model
woe_model = FastWoe()
woe_model.fit(X_train, y_multiclass)

# Check if it's multiclass
print(f"Is multiclass: {woe_model.is_multiclass_target}")
print(f"Classes: {woe_model.classes_}")

# Make predictions
y_pred_proba = woe_model.predict_proba(X_test)  # Shape: (n_samples, n_classes)
y_pred = woe_model.predict(X_test)  # Shape: (n_samples,)
```

### 3. Getting Class-Specific Predictions

```python
# Get probabilities for a specific class
class_0_proba = woe_model.predict_proba_class(X_test, class_label=0)

# Get confidence intervals for a specific class
class_0_ci = woe_model.predict_ci_class(X_test, class_label=0, alpha=0.05)
```

## Analysis Methods

### 1. WOE Mappings

```python
# Get WOE mapping for a specific class
mapping_class_0 = woe_model.get_mapping("feature_name", class_label=0)

# Compare WOE mappings across all classes
woe_comparison = MulticlassWoeUtils.compare_class_woe_mappings(
    woe_model, "feature_name"
)
print(woe_comparison)
```

### 2. Feature Statistics

```python
# Get feature stats for a specific class
stats_class_0 = woe_model.get_feature_stats(class_label=0)

# Get IV analysis for a specific class
iv_analysis_class_0 = woe_model.get_iv_analysis(class_label=0)

# Get top features by class
top_features = MulticlassWoeUtils.get_top_features_by_class(woe_model, top_n=5)
```

### 3. Performance Analysis

```python
# Analyze multiclass performance
performance = MulticlassWoeUtils.analyze_multiclass_performance(
    y_test, y_pred, class_names=woe_model.classes_
)

print(f"Overall Accuracy: {performance['overall_accuracy']:.3f}")
print(f"Class Distribution: {performance['class_distribution']}")
```

## Static Methods for Multiclass Analysis

The `MulticlassWoeUtils` class provides several static methods:

### 1. `create_multiclass_target()`
- **Purpose**: Convert continuous/binary targets to multiclass
- **Methods**: "threshold", "quantile", "kmeans"
- **Use case**: When you need to create multiclass targets from continuous variables

### 2. `analyze_multiclass_performance()`
- **Purpose**: Comprehensive performance analysis
- **Returns**: Classification report, confusion matrix, class distribution
- **Use case**: Model evaluation and comparison

### 3. `compare_class_woe_mappings()`
- **Purpose**: Compare WOE values across classes for a feature
- **Returns**: Pivot table with WOE values by class
- **Use case**: Understanding how features behave differently across classes

### 4. `get_top_features_by_class()`
- **Purpose**: Get top N features by IV for each class
- **Returns**: Dictionary with top features per class
- **Use case**: Feature selection and understanding class-specific importance

### 5. `calculate_class_separation_score()`
- **Purpose**: Measure how well a feature separates classes
- **Returns**: Separation score (higher = better separation)
- **Use case**: Feature ranking and selection

## Complete Workflow Example

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from fastwoe import FastWoe
from examples.multiclass_woe_utils import MulticlassWoeUtils

# 1. Prepare data
X = pd.DataFrame({
    'feature_1': np.random.normal(0, 1, 1000),
    'feature_2': np.random.normal(0, 1, 1000),
    'feature_3': np.random.choice(['A', 'B', 'C'], 1000),
})

# Create multiclass target
y_continuous = X['feature_1'] * 0.5 + X['feature_2'] * 0.3 + np.random.normal(0, 0.2, 1000)
y_multiclass = MulticlassWoeUtils.create_multiclass_target(y_continuous, method="quantile", n_bins=3)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_multiclass, test_size=0.3, random_state=42)

# 3. Fit model
woe_model = FastWoe()
woe_model.fit(X_train, y_train)

# 4. Make predictions
y_pred_proba = woe_model.predict_proba(X_test)
y_pred = woe_model.predict(X_test)

# 5. Analyze results
performance = MulticlassWoeUtils.analyze_multiclass_performance(y_test, y_pred)
print(f"Accuracy: {performance['overall_accuracy']:.3f}")

# 6. Feature analysis
top_features = MulticlassWoeUtils.get_top_features_by_class(woe_model, top_n=3)
for class_label, features in top_features.items():
    print(f"\nTop features for {class_label}:")
    print(features[['feature', 'iv', 'iv_significance']])
```

## Key Differences from Binary WOE

1. **Multiple Encoders**: One encoder per class (one-vs-rest approach)
2. **Multiple Mappings**: Separate WOE mappings for each class
3. **Class-Specific Methods**: Methods like `predict_proba_class()` and `predict_ci_class()`
4. **Probability Output**: `predict_proba()` returns shape `(n_samples, n_classes)`
5. **Class Labels**: Access via `woe_model.classes_` attribute

## Best Practices

1. **Target Creation**: Use quantile-based binning for balanced classes
2. **Feature Selection**: Use `get_top_features_by_class()` to understand class-specific importance
3. **Model Evaluation**: Always use `analyze_multiclass_performance()` for comprehensive evaluation
4. **WOE Analysis**: Compare mappings across classes using `compare_class_woe_mappings()`
5. **Class Imbalance**: Consider stratification in train/test splits

## Troubleshooting

- **"Model must be fitted on multiclass target"**: Ensure your target has more than 2 unique values
- **Class not found errors**: Check available classes with `woe_model.classes_`
- **Performance issues**: Use `get_top_features_by_class()` to focus on most important features
