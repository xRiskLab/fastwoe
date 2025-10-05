#!/usr/bin/env python3
"""
FastWoe Monotonic Constraints Comprehensive Example

This example demonstrates monotonic constraints across all binning methods:
- Tree method: Native scikit-learn monotonic constraints
- KBins method: Isotonic regression post-processing
- FAISS KMeans method: Isotonic regression post-processing

Shows how different binning methods handle the same monotonic constraints
and compares their effectiveness for credit scoring scenarios.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fastwoe import FastWoe


def create_credit_scoring_data(n_samples=2000, random_state=42):
    """Create synthetic credit scoring data with monotonic relationships."""
    np.random.seed(random_state)

    # Generate base features
    income = np.random.lognormal(
        mean=10, sigma=0.5, size=n_samples
    )  # Income in thousands
    age = np.random.normal(35, 12, n_samples)
    age = np.clip(age, 18, 80)  # Reasonable age range
    credit_score = np.random.normal(650, 100, n_samples)
    credit_score = np.clip(credit_score, 300, 850)

    # Create monotonic relationships
    # Income: higher income -> lower risk (decreasing)
    income_risk = 1 / (1 + np.exp((income - np.median(income)) / 20))

    # Age: higher age -> higher risk (increasing)
    age_risk = 1 / (1 + np.exp(-(age - 35) / 8))

    # Credit score: higher score -> lower risk (decreasing)
    credit_risk = 1 / (1 + np.exp((credit_score - 650) / 50))

    # Combine risks with some noise
    combined_risk = (income_risk + age_risk + credit_risk) / 3
    noise = np.random.normal(0, 0.1, n_samples)
    final_risk = np.clip(combined_risk + noise, 0, 1)

    # Convert to binary target
    y = (final_risk > 0.5).astype(int)

    # Create DataFrame
    X = pd.DataFrame({"income": income, "age": age, "credit_score": credit_score})

    return X, y


def print_data_summary(X, y):
    """Print data summary statistics."""
    print("=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)

    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution: {y.mean():.3f} positive rate")
    print(f"Features: {list(X.columns)}")

    print("\nFeature Statistics:")
    print("-" * 50)
    for col in X.columns:
        print(
            f"{col:15} | Range: {X[col].min():8.1f} - {X[col].max():8.1f} | Mean: {X[col].mean():8.1f}"
        )

    # Check correlations with target
    print("\nCorrelations with Target:")
    print("-" * 30)
    for col in X.columns:
        corr = np.corrcoef(X[col], y)[0, 1]
        print(f"{col:15} | Correlation: {corr:6.3f}")


def compare_binning_methods(X, y):
    """Compare monotonic constraints across different binning methods."""

    print("\n" + "=" * 80)
    print("COMPREHENSIVE MONOTONIC CONSTRAINTS COMPARISON")
    print("=" * 80)

    # Define monotonic constraints
    monotonic_cst = {
        "income": -1,  # Decreasing: higher income -> lower risk
        "age": 1,  # Increasing: higher age -> higher risk
        "credit_score": -1,  # Decreasing: higher score -> lower risk
    }

    methods = {
        "Tree (Native)": {
            "binning_method": "tree",
            "tree_kwargs": {"max_depth": 3, "random_state": 42},
        },
        "KBins (Isotonic)": {
            "binning_method": "kbins",
            "binner_kwargs": {"n_bins": 5, "strategy": "quantile"},
        },
        "FAISS KMeans (Isotonic)": {
            "binning_method": "faiss_kmeans",
            "faiss_kwargs": {"k": 5, "niter": 20},
        },
    }

    results = {}

    for method_name, method_config in methods.items():
        print(f"\n{method_name.upper()}")
        print("-" * len(method_name))

        try:
            woe = FastWoe(
                monotonic_cst=monotonic_cst, numerical_threshold=10, **method_config
            )

            woe.fit(X, y)

            # Store results
            results[method_name] = {
                "woe": woe,
                "auc": None,
                "constraints_applied": True,
            }

            # Calculate AUC
            X_woe = woe.transform(X)
            woe_score = X_woe.sum(axis=1)
            auc = roc_auc_score(y, woe_score)
            results[method_name]["auc"] = auc

            print("✅ Successfully applied monotonic constraints")
            print(f"📊 AUC Score: {auc:.4f}")

            # Show constraint application
            summary = woe.get_binning_summary()
            print("📋 Constraints applied:")
            for _, row in summary.iterrows():
                constraint_map = {-1: "Decreasing", 1: "Increasing", 0: "None"}
                print(
                    f"   {row['feature']}: {constraint_map[row['monotonic_constraint']]}"
                )

        except ImportError as e:
            if "faiss" not in str(e).lower():
                raise
            print(f"⚠️  FAISS not available, skipping {method_name}")
            results[method_name] = {
                "woe": None,
                "auc": None,
                "constraints_applied": False,
                "error": "FAISS not available",
            }
        except (ValueError, RuntimeError, AttributeError) as e:
            print(f"❌ Error with {method_name}: {e}")
            results[method_name] = {
                "woe": None,
                "auc": None,
                "constraints_applied": False,
                "error": str(e),
            }

    return results


def compare_kbins_strategies(X, y):
    """Compare different KBins strategies with monotonic constraints."""

    print("\n" + "=" * 80)
    print("KBINS STRATEGIES COMPARISON")
    print("=" * 80)

    monotonic_cst = {
        "income": -1,  # Decreasing: higher income -> lower risk
        "age": 1,  # Increasing: higher age -> higher risk
        "credit_score": -1,  # Decreasing: higher score -> lower risk
    }

    strategies = {
        "Uniform": {"strategy": "uniform"},
        "Quantile": {"strategy": "quantile"},
        "KMeans": {"strategy": "kmeans"},
    }

    results = {}

    for strategy_name, strategy_config in strategies.items():
        print(f"\n{strategy_name.upper()} Strategy:")
        print("-" * len(strategy_name))

        try:
            woe = FastWoe(
                binning_method="kbins",
                monotonic_cst=monotonic_cst,
                numerical_threshold=10,
                binner_kwargs={"n_bins": 5, **strategy_config},
            )

            woe.fit(X, y)

            # Calculate AUC
            X_woe = woe.transform(X)
            woe_score = X_woe.sum(axis=1)
            auc = roc_auc_score(y, woe_score)

            results[strategy_name] = {
                "woe": woe,
                "auc": auc,
                "strategy": strategy_config["strategy"],
            }

            print("✅ Successfully applied monotonic constraints")
            print(f"📊 AUC Score: {auc:.4f}")

            # Show constraint application
            summary = woe.get_binning_summary()
            print("📋 Constraints applied:")
            for _, row in summary.iterrows():
                constraint_map = {-1: "Decreasing", 1: "Increasing", 0: "None"}
                print(
                    f"   {row['feature']}: {constraint_map[row['monotonic_constraint']]}"
                )

        except (ValueError, RuntimeError, AttributeError) as e:
            print(f"❌ Error with {strategy_name}: {e}")
            results[strategy_name] = {
                "woe": None,
                "auc": None,
                "strategy": strategy_config["strategy"],
                "error": str(e),
            }

    return results


def analyze_monotonic_patterns(results, X, y):
    """Analyze monotonic patterns across different methods."""

    print(f"\n{'=' * 80}")
    print("MONOTONIC PATTERN ANALYSIS")
    print(f"{'=' * 80}")

    features = ["income", "age", "credit_score"]
    expected_directions = [-1, 1, -1]  # Decreasing, Increasing, Decreasing

    for feature, expected_direction in zip(features, expected_directions):
        print(f"\n{feature.upper()} Analysis:")
        print("-" * 30)

        direction_name = "Decreasing" if expected_direction == -1 else "Increasing"
        print(f"Expected: {direction_name}")

        for method_name, result in results.items():
            if result["woe"] is None:
                print(f"  {method_name}: Skipped")
                continue

            woe = result["woe"]
            if feature not in woe.mappings_:
                print(f"  {method_name}: Feature not found")
                continue

            mapping = woe.get_mapping(feature)
            woe_values = mapping["woe"].values

            # Extract bin centers
            bin_centers = []
            for _, row in mapping.iterrows():
                bin_str = row["category"]
                if "(" in bin_str and "," in bin_str:
                    try:
                        start = float(bin_str.split("(")[1].split(",")[0])
                        end = float(bin_str.split(",")[1].split("]")[0])
                        center = (start + end) / 2
                        bin_centers.append(center)
                    except (ValueError, IndexError):
                        bin_centers.append(len(bin_centers))
                else:
                    bin_centers.append(len(bin_centers))

            # Check monotonicity
            if len(bin_centers) >= 2:
                sorted_indices = np.argsort(bin_centers)
                sorted_woe = woe_values[sorted_indices]

                is_monotonic = True
                for i in range(1, len(sorted_woe)):
                    if expected_direction == -1:  # Decreasing
                        if sorted_woe[i] > sorted_woe[i - 1] + 1e-10:
                            is_monotonic = False
                            break
                    elif sorted_woe[i] < sorted_woe[i - 1] - 1e-10:
                        is_monotonic = False
                        break

                status = "✅ Monotonic" if is_monotonic else "❌ Non-monotonic"
                print(f"  {method_name}: {status}")

                # Show WOE values in a readable format
                print(f"WOE values: {[f'{v:.3f}' for v in sorted_woe]}")
            else:
                print(f"{method_name}: Insufficient bins")


def print_performance_summary(results):
    """Print performance summary across methods."""

    print(f"\n{'=' * 80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'=' * 80}")

    print(f"{'Method':<30} {'AUC Score':<12} {'Status':<15}")
    print("-" * 60)

    for method_name, result in results.items():
        if result["woe"] is None:
            status = "Skipped"
            auc_str = "N/A"
        elif result["constraints_applied"]:
            status = "✅ Success"
            auc_str = f"{result['auc']:.4f}"
        else:
            status = "❌ Failed"
            auc_str = "N/A"

        print(f"{method_name:<30} {auc_str:<12} {status:<15}")

    if valid_results := {k: v for k, v in results.items() if v["auc"] is not None}:
        best_method = max(valid_results.keys(), key=lambda k: valid_results[k]["auc"])
        best_auc = valid_results[best_method]["auc"]
        print(f"\n🏆 Best performing method: {best_method} (AUC: {best_auc:.4f})")


def print_kbins_strategies_summary(kbins_results):
    """Print KBins strategies performance summary."""

    print(f"\n{'=' * 80}")
    print("KBINS STRATEGIES PERFORMANCE SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Strategy':<15} {'AUC Score':<12} {'Status':<15}")
    print("-" * 45)

    for strategy_name, result in kbins_results.items():
        if result["woe"] is None:
            status = "Failed"
            auc_str = "N/A"
        else:
            status = "Success"
            auc_str = f"{result['auc']:.4f}"
        print(f"{strategy_name:<15} {auc_str:<12} {status:<15}")

    if valid_kbins := {k: v for k, v in kbins_results.items() if v["auc"] is not None}:
        best_kbins = max(valid_kbins.keys(), key=lambda k: valid_kbins[k]["auc"])
        best_kbins_auc = valid_kbins[best_kbins]["auc"]
        print(f"\n🏆 Best KBins strategy: {best_kbins} (AUC: {best_kbins_auc:.4f})")


def print_detailed_woe_analysis(results, X, y):
    """Print detailed WOE analysis for each method."""

    print(f"\n{'=' * 80}")
    print("DETAILED WOE ANALYSIS")
    print(f"{'=' * 80}")

    features = ["income", "age", "credit_score"]
    expected_directions = [-1, 1, -1]

    for method_name, result in results.items():
        if result["woe"] is None:
            continue

        print(f"\n{method_name.upper()}")
        print("-" * len(method_name))

        woe = result["woe"]

        for feature, _ in zip(features, expected_directions):
            if feature not in woe.mappings_:
                continue

            mapping = woe.get_mapping(feature)

            print(f"\n{feature.upper()}:")
            print("Bin Range" + " " * 20 + "WOE Value" + " " * 8 + "Event Rate")
            print("-" * 50)

            # Extract bin centers and WOE values
            bin_centers = []
            woe_values = []
            event_rates = []

            for _, row in mapping.iterrows():
                bin_str = row["category"]
                if "(" in bin_str and "," in bin_str:
                    try:
                        start = float(bin_str.split("(")[1].split(",")[0])
                        end = float(bin_str.split(",")[1].split("]")[0])
                        center = (start + end) / 2
                        bin_centers.append(center)
                        woe_values.append(row["woe"])
                        event_rates.append(row["event_rate"])
                    except (ValueError, IndexError):
                        continue

            if len(bin_centers) >= 2:
                sorted_indices = np.argsort(bin_centers)
                sorted_centers = np.array(bin_centers)[sorted_indices]
                sorted_woe = np.array(woe_values)[sorted_indices]
                sorted_rates = np.array(event_rates)[sorted_indices]

                for center, woe_val, rate in zip(
                    sorted_centers, sorted_woe, sorted_rates
                ):
                    print(
                        f"{center:8.1f}"
                        + " " * 20
                        + f"{woe_val:8.3f}"
                        + " " * 8
                        + f"{rate:.3f}"
                    )


def main():
    """Main example function."""
    print("FastWoe Comprehensive Monotonic Constraints Example")
    print("=" * 60)

    # Create synthetic credit scoring data
    print("\nCreating synthetic credit scoring data...")
    X, y = create_credit_scoring_data(n_samples=2000, random_state=42)

    # Print data summary
    print_data_summary(X, y)

    # Compare binning methods
    results = compare_binning_methods(X, y)

    # Compare KBins strategies
    kbins_results = compare_kbins_strategies(X, y)

    # Analyze monotonic patterns
    analyze_monotonic_patterns(results, X, y)

    # Performance summary
    print_performance_summary(results)

    # KBins strategies summary
    print_kbins_strategies_summary(kbins_results)

    # Detailed WOE analysis
    print_detailed_woe_analysis(results, X, y)


if __name__ == "__main__":
    main()
