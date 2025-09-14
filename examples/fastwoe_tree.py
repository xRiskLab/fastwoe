#!/usr/bin/env python3
"""
FastWoe Tree Binning Example

This example demonstrates the new decision tree-based binning functionality
in FastWoe, comparing it with traditional binning methods.
"""

import numpy as np
import pandas as pd

from fastwoe import FastWoe


def _decorated_header(arg0, arg1):
    """Decorate a header with a line of equals signs."""
    print("\n" + "=" * arg0)
    print(arg1)
    print("=" * arg0)


def create_sample_data(n_samples=1000, random_state=42):
    """Create sample data with non-linear relationships."""
    np.random.seed(random_state)

    # Create numerical feature with non-linear relationship to target
    X_num = np.random.normal(0, 1.5, n_samples)

    # Create target with non-linear relationship
    # High probability for very low values (< -1.5) and high values (> 1.0)
    y = ((X_num < -1.5) | (X_num > 1.0)).astype(int)

    # Add some noise
    y = y ^ (np.random.random(n_samples) < 0.15)

    # Create categorical feature
    X_cat = np.random.choice(["A", "B", "C", "D"], n_samples, p=[0.4, 0.3, 0.2, 0.1])

    return pd.DataFrame(
        {"numerical_feature": X_num, "categorical_feature": X_cat, "target": y}
    )


def compare_binning_methods(data):  # sourcery skip: extract-duplicate-method
    """Compare different binning methods."""
    _decorated_header(60, "FastWoe Tree Binning Example")

    print(f"Dataset: {len(data)} samples")
    print(f"Target distribution: {data['target'].value_counts().to_dict()}")
    print(
        f"Numerical feature range: [{data['numerical_feature'].min():.2f}, {data['numerical_feature'].max():.2f}]"
    )

    # Method 1: Traditional KBinsDiscretizer (quantile)
    _decorated_header(40, "Method 1: Traditional Binning (Quantile)")

    fw_traditional = FastWoe(
        binning_method="kbins",
        binner_kwargs={"n_bins": 5, "strategy": "quantile", "encode": "ordinal"},
        numerical_threshold=10,
        warn_on_numerical=False,
    )

    fw_traditional.fit(
        data[["numerical_feature", "categorical_feature"]], data["target"]
    )

    # Get binning summary
    summary_traditional = fw_traditional.get_binning_summary()
    print("Binning Summary:")
    print(summary_traditional)

    # Get split values
    splits_traditional = fw_traditional.get_split_value_histogram("numerical_feature")
    print(f"\nSplit values: {splits_traditional}")

    # Method 2: Decision Tree Binning (default)
    print("\n" + "=" * 40)
    print("Method 2: Decision Tree Binning (Default)")
    print("=" * 40)

    fw_tree = FastWoe(
        binning_method="tree",
        tree_kwargs={
            "max_depth": 3,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "random_state": 42,
        },
        numerical_threshold=10,
        warn_on_numerical=False,
    )

    fw_tree.fit(data[["numerical_feature", "categorical_feature"]], data["target"])

    # Get binning summary
    summary_tree = fw_tree.get_binning_summary()
    print("Binning Summary:")
    print(summary_tree)

    # Get split values
    splits_tree = fw_tree.get_split_value_histogram("numerical_feature")
    print(f"\nSplit values: {splits_tree}")

    return fw_traditional, fw_tree


def analyze_woe_mappings(fw_traditional, fw_tree):
    """Analyze and compare WOE mappings."""
    _decorated_header(60, "WOE Mappings Comparison")
    # Get mappings for numerical feature
    mapping_traditional = fw_traditional.get_mapping("numerical_feature")
    mapping_tree = fw_tree.get_mapping("numerical_feature")

    print("\nTraditional Binning WOE Mapping:")
    print(
        mapping_traditional[["category", "count", "event_rate", "woe", "woe_se"]].round(
            4
        )
    )

    print("\nTree Binning WOE Mapping:")
    print(mapping_tree[["category", "count", "event_rate", "woe", "woe_se"]].round(4))

    _decorated_header(40, "Feature Statistics Comparison")
    stats_traditional = fw_traditional.get_feature_stats("numerical_feature")
    stats_tree = fw_tree.get_feature_stats("numerical_feature")

    comparison = pd.DataFrame(
        {
            "Method": ["KBins", "Tree"],
            "Gini": [
                stats_traditional["gini"].iloc[0],
                stats_tree["gini"].iloc[0],
            ],
            "IV": [
                stats_traditional["iv"].iloc[0],
                stats_tree["iv"].iloc[0],
            ],
            "N_Categories": [
                stats_traditional["n_categories"].iloc[0],
                stats_tree["n_categories"].iloc[0],
            ],
        }
    )

    print(comparison.round(4))


def demonstrate_predictions(data, fw_traditional, fw_tree):
    """Demonstrate prediction capabilities."""
    _decorated_header(60, "Prediction Capabilities")

    # Get predictions
    X = data[["numerical_feature", "categorical_feature"]]

    pred_traditional = fw_traditional.predict_proba(X)[:, 1]
    pred_tree = fw_tree.predict_proba(X)[:, 1]

    print(f"Traditional binning - Mean prediction: {pred_traditional.mean():.4f}")
    print(f"Tree binning - Mean prediction: {pred_tree.mean():.4f}")
    print(f"Actual target mean: {data['target'].mean():.4f}")

    # Get confidence intervals
    ci_traditional = fw_traditional.predict_ci(X)
    ci_tree = fw_tree.predict_ci(X)

    print("\nConfidence Intervals (first 5 samples):")
    print("Traditional:", ci_traditional[:5].round(4))
    print("Tree:", ci_tree[:5].round(4))


def main():
    """Main example function."""
    # Create sample data
    data = create_sample_data()

    # Compare binning methods
    fw_traditional, fw_tree = compare_binning_methods(data)

    # Analyze WOE mappings
    analyze_woe_mappings(fw_traditional, fw_tree)

    # Demonstrate predictions
    demonstrate_predictions(data, fw_traditional, fw_tree)

    _decorated_header(60, "Key Takeaways")
    print("1. Tree-based binning creates bins optimized for the target variable")
    print("2. Number of bins is determined by tree structure, not fixed parameters")
    print("3. Tree binning can capture non-linear relationships better")
    print("4. All methods maintain the same FastWoe API for easy comparison")
    print("\n✅ Example completed successfully!")


if __name__ == "__main__":
    main()
