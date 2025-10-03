#!/usr/bin/env python3
"""
Multiclass WOE Example

This example demonstrates how to use FastWoe for multiclass classification problems
using one-vs-rest Weight of Evidence encoding.

The key insight is that for multiclass targets, we create separate WOE encodings
for each class against all others, resulting in multiple columns per feature.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from fastwoe import FastWoe


def create_multiclass_data(n_samples=1000, random_state=42):
    """Create a realistic multiclass dataset with different patterns per class."""
    np.random.seed(random_state)

    # Create job categories with different class distributions
    jobs = []
    ages = []
    incomes = []
    y = []

    for i in range(n_samples):
        if i < n_samples // 3:  # Class 0: High-income professionals
            if np.random.random() < 0.6:
                jobs.append("doctor")
            elif np.random.random() < 0.8:
                jobs.append("engineer")
            else:
                jobs.append("lawyer")
            ages.append(np.random.choice(["30-50", "50+"], p=[0.7, 0.3]))
            incomes.append(np.random.normal(120000, 20000))
            y.append(0)

        elif i < 2 * n_samples // 3:  # Class 1: Mid-income workers
            if np.random.random() < 0.5:
                jobs.append("teacher")
            elif np.random.random() < 0.8:
                jobs.append("engineer")
            else:
                jobs.append("nurse")
            ages.append(np.random.choice(["<30", "30-50"], p=[0.4, 0.6]))
            incomes.append(np.random.normal(60000, 15000))
            y.append(1)

        else:  # Class 2: Lower-income workers
            if np.random.random() < 0.7:
                jobs.append("artist")
            elif np.random.random() < 0.9:
                jobs.append("teacher")
            else:
                jobs.append("retail")
            ages.append(np.random.choice(["<30", "30-50", "50+"], p=[0.5, 0.3, 0.2]))
            incomes.append(np.random.normal(35000, 10000))
            y.append(2)

    X = pd.DataFrame(
        {
            "job": jobs,
            "age_group": ages,
            "income": incomes,
        }
    )
    y = pd.Series(y)

    return X, y


def main():  # sourcery skip: extract-duplicate-method
    print("=== Multiclass WOE Example ===\n")

    # Create dataset
    print("1. Creating multiclass dataset...")
    X, y = create_multiclass_data(n_samples=1000)

    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    print(f"Features: {list(X.columns)}")

    # Show class patterns
    print("\nClass patterns:")
    for class_label in sorted(y.unique()):
        mask = y == class_label
        print(f"Class {class_label}:")
        print(f"Job distribution: {X[mask]['job'].value_counts().to_dict()}")
        print(f"Age distribution: {X[mask]['age_group'].value_counts().to_dict()}")
        print(f"Avg income: ${X[mask]['income'].mean():.0f}")

    # Split data
    print("\n2. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Apply WOE encoding
    print("\n3. Applying multiclass WOE encoding...")
    woe_encoder = FastWoe()
    woe_encoder.fit(X_train, y_train)

    print(f"Target type: {woe_encoder.encoder_kwargs['target_type']}")
    print(f"Classes: {woe_encoder.classes_}")
    print(f"Class priors: {woe_encoder.y_prior_}")

    # Transform data
    X_train_woe = woe_encoder.transform(X_train)
    X_test_woe = woe_encoder.transform(X_test)

    print(f"Original features: {X_train.shape[1]}")
    print(f"WOE features: {X_train_woe.shape[1]}")
    print(f"WOE columns: {list(X_train_woe.columns)}")

    # Show WOE value ranges
    print("\nWOE value ranges:")
    for col in X_train_woe.columns:
        print(f"{col}: [{X_train_woe[col].min():.3f}, {X_train_woe[col].max():.3f}]")

    # Train a classifier
    print("\n4. Training Random Forest classifier...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_woe, y_train)

    # Make predictions
    y_pred = rf.predict(X_test_woe)

    print("\n5. Model performance:")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=[f"Class {i}" for i in range(3)]
        )
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Show feature importance
    print("\n6. Feature importance (top 10):")
    feature_importance = pd.DataFrame(
        {"feature": X_train_woe.columns, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)

    print(feature_importance.head(10).to_string(index=False))

    # Show WOE mappings for most important feature
    print(
        f"\n7. WOE mapping for most important feature: {feature_importance.iloc[0]['feature']}"
    )
    most_important_feature = feature_importance.iloc[0]["feature"]

    # Extract original feature name and class
    if "_class_" in most_important_feature:
        orig_feature, class_label = most_important_feature.split("_class_")
        class_label = int(class_label)
        mapping = woe_encoder.mappings_[orig_feature][class_label]
        print(f"\n   WOE mapping for {orig_feature} (Class {class_label} vs rest):")
        print(mapping[["count", "event_rate", "woe"]].to_string())

    # Demonstrate class-specific predictions
    print("\n8. Class-specific predictions:")
    print("\nMethod 1: Extract from full results")
    all_probs = woe_encoder.predict_proba(X_test)
    all_ci = woe_encoder.predict_ci(X_test)

    class_0_probs = all_probs[:, 0]
    class_0_ci = all_ci[:, [0, 1]]
    print(f"Class 0 probabilities (first 5): {class_0_probs[:5]}")
    print(f"Class 0 CI (first 5):\n{class_0_ci[:5]}")

    print("\nMethod 2: Use class-specific methods")
    class_0_probs_method = woe_encoder.predict_proba_class(X_test, class_label=0)
    class_0_ci_method = woe_encoder.predict_ci_class(X_test, class_label=0)
    print(f"Class 0 probabilities (first 5): {class_0_probs_method[:5]}")
    print(f"Class 0 CI (first 5):\n{class_0_ci_method[:5]}")

    print("\n9. Practical usage examples:")
    # High-risk detection
    high_risk_mask = woe_encoder.predict_proba_class(X_test, class_label=0) > 0.5
    print(f"Samples with high probability of being Class 0: {high_risk_mask.sum()}")

    # High-confidence predictions for Class 2
    class_2_ci = woe_encoder.predict_ci_class(X_test, class_label=2)
    high_confidence_mask = class_2_ci[:, 0] > 0.3  # Lower bound > 0.3
    print(
        f"Samples with high confidence of being Class 2: {high_confidence_mask.sum()}"
    )

    # Uncertain predictions (wide CI)
    ci_widths = class_0_ci_method[:, 1] - class_0_ci_method[:, 0]
    uncertain_mask = ci_widths > 0.1
    print(f"Samples with uncertain predictions (wide CI): {uncertain_mask.sum()}")

    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
