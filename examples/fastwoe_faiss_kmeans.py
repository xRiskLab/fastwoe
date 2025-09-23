#!/usr/bin/env python3
"""
Example demonstrating FAISS KMeans binning in FastWoe.

This example shows how to use the new FAISS KMeans binning method
and compares it with the traditional KBinsDiscretizer approach.

Requirements:
    pip install fastwoe[faiss]      # CPU version
    # or
    pip install fastwoe[faiss-gpu]  # GPU version
"""

import importlib.util
import warnings

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def _decorated_header(arg0, arg1):
    """Decorate a header with a line of equals signs."""
    print("\n" + "=" * arg0)
    print(arg1)
    print("=" * arg0)


try:
    from fastwoe import FastWoe
except ImportError:
    print("Error: fastwoe not installed. Please install it first.")
    exit(1)

FAISS_AVAILABLE = importlib.util.find_spec("faiss") is not None

if not FAISS_AVAILABLE:
    print("Error: FAISS not available. Please install with:")
    print("  CPU version: pip install faiss-cpu")
    print("  GPU version: pip install faiss-gpu-cu12")
    exit(1)


def create_sample_data(n_samples=1000, n_features=5, random_state=42):
    """Create sample data for demonstration."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=3,
        n_redundant=1,
        n_clusters_per_class=1,
        random_state=random_state,
    )

    # Convert to DataFrame
    feature_names = [f"feature_{i:02d}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)

    return X_df, pd.Series(y)


def demonstrate_faiss_kmeans():
    """Demonstrate FAISS KMeans binning."""
    _decorated_header(50, "FAISS KMeans Binning Demonstration")

    # Create sample data
    X, y = create_sample_data(n_samples=2000, n_features=4)
    print(f"Data shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    # Method 1: FAISS KMeans binning
    _decorated_header(60, "Method 1: FAISS KMeans Binning")

    woe_faiss = FastWoe(
        binning_method="faiss_kmeans",
        faiss_kwargs={
            "k": 5,  # Number of clusters
            "niter": 20,  # Number of iterations
            "verbose": True,  # Show progress
            "gpu": False,  # Use CPU (set to True if you have GPU)
        },
        numerical_threshold=10,
        warn_on_numerical=False,
    )

    # Fit and transform
    print("Fitting FAISS KMeans model...")
    woe_faiss.fit(X_train, y_train)

    print("Transforming data...")
    X_train_faiss = woe_faiss.transform(X_train)
    X_test_faiss = woe_faiss.transform(X_test)

    # Get binning summary
    print("\nBinning Summary:")
    binning_summary = woe_faiss.get_binning_summary()
    print(binning_summary)

    # Get feature statistics
    print("\nFeature Statistics:")
    feature_stats = woe_faiss.get_feature_stats()
    print(feature_stats[["feature", "gini", "iv", "n_categories"]].round(4))

    # Show mapping for one feature
    print(f"\nWOE Mapping for {feature_stats.iloc[0]['feature']}:")
    mapping = woe_faiss.get_mapping(feature_stats.iloc[0]["feature"])
    print(mapping[["category", "count", "event_rate", "woe"]].round(4))

    return woe_faiss, X_train_faiss, X_test_faiss


def compare_with_kbins():
    """Compare FAISS KMeans with KBinsDiscretizer."""
    _decorated_header(60, "Method 2: KBinsDiscretizer (for comparison)")

    # Create sample data
    X, y = create_sample_data(n_samples=2000, n_features=4)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    woe_kbins = FastWoe(
        binning_method="kbins",
        binner_kwargs={
            "n_bins": 5,
            "strategy": "quantile",
            "encode": "ordinal",
        },
        numerical_threshold=10,
        warn_on_numerical=False,
    )

    # Fit and transform
    print("Fitting KBinsDiscretizer model...")
    woe_kbins.fit(X_train, y_train)

    print("Transforming data...")
    X_train_kbins = woe_kbins.transform(X_train)
    X_test_kbins = woe_kbins.transform(X_test)

    # Get feature statistics
    print("\nFeature Statistics:")
    feature_stats = woe_kbins.get_feature_stats()
    print(feature_stats[["feature", "gini", "iv", "n_categories"]].round(4))

    return woe_kbins, X_train_kbins, X_test_kbins


def compare_performance():
    """Compare performance between methods."""
    _decorated_header(60, "Performance Comparison")

    import time

    # Create sample data
    X, y = create_sample_data(n_samples=5000, n_features=6)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    methods = {
        "FAISS KMeans": FastWoe(
            binning_method="faiss_kmeans",
            faiss_kwargs={"k": 5, "niter": 20, "verbose": False, "gpu": False},
            numerical_threshold=10,
            warn_on_numerical=False,
        ),
        "KBinsDiscretizer": FastWoe(
            binning_method="kbins",
            binner_kwargs={"n_bins": 5, "strategy": "quantile", "encode": "ordinal"},
            numerical_threshold=10,
            warn_on_numerical=False,
        ),
        "Decision Tree": FastWoe(
            binning_method="tree",
            tree_kwargs={
                "max_depth": 3,
                "min_samples_split": 20,
                "min_samples_leaf": 10,
            },
            numerical_threshold=10,
            warn_on_numerical=False,
        ),
    }

    results = {}

    for method_name, woe in methods.items():
        print(f"\nTesting {method_name}...")

        # Measure fit time
        start_time = time.time()
        woe.fit(X_train, y_train)
        fit_time = time.time() - start_time

        # Measure transform time
        start_time = time.time()
        X_transformed = woe.transform(X_test)
        transform_time = time.time() - start_time

        # Get quality metrics
        feature_stats = woe.get_feature_stats()
        avg_gini = feature_stats["gini"].mean()
        avg_iv = feature_stats["iv"].mean()

        results[method_name] = {
            "fit_time": fit_time,
            "transform_time": transform_time,
            "total_time": fit_time + transform_time,
            "avg_gini": avg_gini,
            "avg_iv": avg_iv,
        }

        print(f"  Fit time: {fit_time:.4f}s")
        print(f"  Transform time: {transform_time:.4f}s")
        print(f"  Average Gini: {avg_gini:.4f}")
        print(f"  Average IV: {avg_iv:.4f}")

    # Print comparison table
    _decorated_header(80, "COMPARISON SUMMARY")
    print(
        f"{'Method':<20} {'Fit (s)':<10} {'Transform (s)':<15} {'Total (s)':<10} {'Gini':<8} {'IV':<8}"
    )
    print("-" * 80)

    for method_name, metrics in results.items():
        print(
            f"{method_name:<20} {metrics['fit_time']:<10.4f} {metrics['transform_time']:<15.4f} "
            f"{metrics['total_time']:<10.4f} {metrics['avg_gini']:<8.4f} {metrics['avg_iv']:<8.4f}"
        )


def demonstrate_advanced_features():
    """Demonstrate advanced FAISS KMeans features."""
    _decorated_header(60, "Advanced FAISS KMeans Features")

    # Create sample data
    X, y = create_sample_data(n_samples=1000, n_features=5)

    # Test different k values
    print("\nTesting different k values:")
    for k in [3, 5, 7, 10]:
        woe = FastWoe(
            binning_method="faiss_kmeans",
            faiss_kwargs={"k": k, "niter": 10, "verbose": False, "gpu": False},
            numerical_threshold=10,
            warn_on_numerical=False,
        )
        woe.fit(X, y)

        feature_stats = woe.get_feature_stats()
        avg_gini = feature_stats["gini"].mean()
        binning_summary = woe.get_binning_summary()
        total_bins = binning_summary["n_bins"].sum()

        print(f"  k={k}: Gini={avg_gini:.4f}, Total bins={total_bins}")

    # Test with different strategies
    print("\nTesting different binning strategies:")
    strategies = ["quantile", "uniform", "kmeans"]

    for strategy in strategies:
        woe = FastWoe(
            binning_method="kbins",
            binner_kwargs={"n_bins": 5, "strategy": strategy, "encode": "ordinal"},
            numerical_threshold=10,
            warn_on_numerical=False,
        )
        woe.fit(X, y)

        feature_stats = woe.get_feature_stats()
        avg_gini = feature_stats["gini"].mean()

        print(f"  {strategy}: Gini={avg_gini:.4f}")


def main():
    """Main demonstration function."""
    _decorated_header(50, "FastWoe FAISS KMeans Binning Example")

    if not FAISS_AVAILABLE:
        print("FAISS is not available. Please install it with:")
        print("  CPU version: pip install faiss-cpu")
        print("  GPU version: pip install faiss-gpu-cu12")
        return

    # Demonstrate FAISS KMeans
    woe_faiss, X_train_faiss, X_test_faiss = demonstrate_faiss_kmeans()

    # Compare with KBins
    woe_kbins, X_train_kbins, X_test_kbins = compare_with_kbins()

    # Compare performance
    compare_performance()

    # Demonstrate advanced features
    demonstrate_advanced_features()

    _decorated_header(60, "Example completed!")
    print("\nKey takeaways:")
    print("1. FAISS KMeans provides efficient clustering-based binning")
    print("2. It can be faster than traditional methods for large datasets")
    print("3. The quality depends on the choice of k (number of clusters)")
    print("4. It works well with both binary and continuous targets")
    print("5. GPU acceleration is available for even better performance")


if __name__ == "__main__":
    main()
