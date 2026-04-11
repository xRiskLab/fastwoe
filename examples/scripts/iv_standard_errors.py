#!/usr/bin/env python3
"""
Demonstration of IV Standard Errors in FastWoe

This example shows how to use the new IV standard error functionality
to assess the statistical significance of Information Value calculations.
"""

import numpy as np
import pandas as pd

from fastwoe import FastWoe

# Set random seed for reproducibility
np.random.seed(42)
n_samples = 2000

print("FastWoe IV Standard Errors Demo")
print("=" * 50)

# Create sample dataset with features of varying predictive power
df = pd.DataFrame(
    {
        "strong_predictor": np.random.choice(
            ["Low", "Medium", "High"], n_samples, p=[0.4, 0.3, 0.3]
        ),
        "weak_predictor": np.random.choice(["A", "B"], n_samples, p=[0.55, 0.45]),
        "noise_feature": np.random.choice(["X", "Y", "Z", "W"], n_samples),
        "numerical_feature": np.random.normal(0, 1, n_samples),
    }
)

# Create target with correlation to strong_predictor
y = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])

# Add strong correlation for strong_predictor
strong_mask = df["strong_predictor"] == "High"
y[strong_mask] = np.random.choice([0, 1], strong_mask.sum(), p=[0.2, 0.8])

# Add weak correlation for weak_predictor
weak_mask = df["weak_predictor"] == "B"
y[weak_mask] = np.random.choice([0, 1], weak_mask.sum(), p=[0.55, 0.45])

y = pd.Series(y)

print(f"Dataset: {n_samples} samples, {len(df.columns)} features")
print(f"Target distribution: {y.value_counts().to_dict()}")

# Fit FastWoe
print("\nFitting FastWoe...")
woe = FastWoe()
woe.fit(df, y)

# Get IV analysis with confidence intervals
print("\n1. IV Analysis with Confidence Intervals:")
iv_analysis = woe.get_iv_analysis()
print(
    iv_analysis[
        ["feature", "iv", "iv_se", "iv_ci_lower", "iv_ci_upper", "iv_significance"]
    ].to_string(index=False)
)

# Get detailed analysis for the strongest predictor
print("\n2. Detailed Analysis for Strongest Predictor:")
strongest_feature = iv_analysis.loc[iv_analysis["iv"].idxmax(), "feature"]
single_analysis = woe.get_iv_analysis(strongest_feature)

print(f"\nFeature: {strongest_feature}")
print(f"IV: {single_analysis['iv'].iloc[0]:.4f}")
print(f"Standard Error: {single_analysis['iv_se'].iloc[0]:.4f}")
print(
    f"95% Confidence Interval: [{single_analysis['iv_ci_lower'].iloc[0]:.4f}, {single_analysis['iv_ci_upper'].iloc[0]:.4f}]"
)
print(f"Statistical Significance: {single_analysis['iv_significance'].iloc[0]}")

# Show mapping table for the strongest predictor
print(f"\n3. WOE Mapping for {strongest_feature}:")
mapping = woe.get_mapping(strongest_feature)
print(
    mapping[["count", "event_rate", "woe", "woe_se", "woe_ci_lower", "woe_ci_upper"]]
    .round(4)
    .to_string()
)

# Compare regular feature stats with IV analysis
print("\n4. Enhanced Feature Statistics:")
feature_stats = woe.get_feature_stats()
enhanced_cols = [
    "feature",
    "iv",
    "iv_se",
    "iv_ci_lower",
    "iv_ci_upper",
    "gini",
    "n_categories",
]
print(feature_stats[enhanced_cols].round(4).to_string(index=False))

# Interpretation guide
print("\n5. IV Interpretation Guide:")
print("IV Range     | Predictive Power | Statistical Assessment")
print("-------------|------------------|----------------------")
for _, row in iv_analysis.iterrows():
    iv_val = row["iv"]
    feature = row["feature"]
    significance = row["iv_significance"]

    if iv_val < 0.02:
        power = "Not useful"
    elif iv_val < 0.1:
        power = "Weak"
    elif iv_val < 0.3:
        power = "Medium"
    elif iv_val < 0.5:
        power = "Strong"
    else:
        power = "Very Strong"

    print(f"{iv_val:8.4f} | {power:15} | {significance} ({feature})")
