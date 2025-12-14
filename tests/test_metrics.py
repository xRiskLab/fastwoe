"""test_metrics.py."""

import time
import warnings
from typing import Any

import numpy as np
import pytest
from scipy import stats

from fastwoe.fast_somersd import somersd_pairwise, somersd_xy, somersd_yx
from fastwoe.logging_config import logger, setup_logger

# Configure logger with RichHandler for better formatting
setup_logger(level="INFO")


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(42)


def _log_case(name, y, x):
    """Log our D_Y|X and D_X|Y next to SciPy's for a given case."""
    ryx = somersd_yx(y, x)
    rxy = somersd_xy(y, x)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        syx = stats.somersd(y, x)
        sxy = stats.somersd(x, y)

    logger.info(f"=== {name} ===")
    logger.info(f"Our D_Y|X: {ryx.statistic}")
    logger.info(f"SciPy D_Y|X: {syx.statistic}")
    logger.info(f"Our D_X|Y: {rxy.statistic}")
    logger.info(f"SciPy D_X|Y: {sxy.statistic}")

    return ryx, rxy, syx, sxy


def test_metric_comparisons():
    """Compare Somers' D calculations against SciPy with logging."""
    cases = [
        ("Perfect positive", np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5])),
        ("Perfect negative", np.array([1, 2, 3, 4, 5]), np.array([5, 4, 3, 2, 1])),
        ("Random normal (n=100)", np.random.randn(100), np.random.randn(100)),
        (
            "Random integers w/ ties",
            np.random.randint(0, 10, 100),
            np.random.randint(0, 10, 100),
        ),
        ("Ties in Y", np.array([1, 1, 2, 2, 3, 3]), np.array([1, 2, 3, 4, 5, 6])),
        ("Ties in X", np.array([1, 2, 3, 4, 5, 6]), np.array([1, 1, 2, 2, 3, 3])),
    ]

    logger.info("Starting Somers' D comparison tests...")

    # sourcery skip: no-loop-in-tests
    for name, y, x in cases:
        ryx, rxy, syx, sxy = _log_case(name, y, x)

        # Actually test something instead of just printing
        assert abs(ryx.statistic - syx.statistic) < 1e-10, f"D_Y|X mismatch for {name}"
        assert abs(rxy.statistic - sxy.statistic) < 1e-10, f"D_X|Y mismatch for {name}"

    logger.info("All comparisons passed!")


@pytest.mark.slow
def test_performance_vs_scipy():
    """Compare runtime performance to SciPy with logging."""
    n = 100
    y = np.random.randn(n)
    x = np.random.randn(n)

    logger.info(f"Performance test with n={n} samples")

    # Our implementation timing
    t0 = time.time()
    r = somersd_yx(y, x)
    our_time = time.time() - t0

    # SciPy timing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t1 = time.time()
        s = stats.somersd(y, x)
        scipy_time = time.time() - t1

    speedup = scipy_time / our_time if our_time > 0 else float("inf")

    logger.info(f"FastWoe time: {our_time:.4f}s")
    logger.info(f"SciPy time: {scipy_time:.4f}s")
    logger.info(f"Speedup: {speedup:.2f}x")
    logger.info(f"FastWoe D_Y|X: {r.statistic:.6f}")
    logger.info(f"SciPy D_Y|X: {s.statistic:.6f}")
    logger.info(f"FastWoe D_X|Y: {r.statistic:.6f}")
    logger.info(f"SciPy D_X|Y: {s.statistic:.6f}")

    # Actually verify correctness
    assert abs(r.statistic - s.statistic) < 1e-10


def test_weighted_somersd():
    """Test weighted Somers' D against sklearn's weighted AUC."""
    from sklearn.metrics import roc_auc_score

    logger.info("Testing weighted Somers' D implementation...")

    # Generate test data
    np.random.seed(42)
    n = 1000
    y_true = np.random.binomial(1, 0.3, n)
    y_pred = y_true + np.random.normal(0, 0.3, n)
    weights = np.random.uniform(0.5, 2.0, n)

    # Test 1: Unweighted case
    logger.info("=== Test 1: Unweighted ===")
    somers_unweighted = somersd_yx(y_true, y_pred).statistic
    auc_unweighted = roc_auc_score(y_true, y_pred)
    gini_unweighted = 2 * auc_unweighted - 1

    logger.info(f"Somers' D (unweighted):     {somers_unweighted:.8f}")
    logger.info(f"Gini from AUC (unweighted): {gini_unweighted:.8f}")

    assert np.isclose(somers_unweighted, gini_unweighted, atol=1e-10), (
        "Unweighted Somers' D should match Gini"
    )

    # Test 2: Weighted case
    logger.info("=== Test 2: Weighted ===")
    somers_weighted = somersd_yx(y_true, y_pred, weights).statistic
    auc_weighted = roc_auc_score(y_true, y_pred, sample_weight=weights)
    gini_weighted = 2 * auc_weighted - 1

    logger.info(f"Somers' D (weighted):       {somers_weighted:.8f}")
    logger.info(f"Gini from AUC (weighted):   {gini_weighted:.8f}")

    assert np.isclose(somers_weighted, gini_weighted, atol=1e-10), (
        "Weighted Somers' D should match weighted Gini"
    )

    # Test 3: Verify weights have effect
    logger.info("=== Test 3: Weights Effect ===")
    logger.info(f"Unweighted Gini: {gini_unweighted:.6f}")
    logger.info(f"Weighted Gini:   {gini_weighted:.6f}")
    logger.info(f"Difference:      {abs(gini_weighted - gini_unweighted):.6f}")

    assert not np.isclose(gini_weighted, gini_unweighted, atol=1e-3), (
        "Weights should change the result"
    )

    logger.info("✓ All weighted tests passed!")


def test_somersd_pairwise():
    """Test pairwise Somers' D computation (works for both continuous and binary)."""
    from sklearn.metrics import roc_auc_score

    logger.info("Testing somersd_pairwise function...")

    # Test 1: Perfect separation
    logger.info("=== Test 1: Perfect separation ===")
    pos_scores = np.array([0.8, 0.9, 0.7])
    neg_scores = np.array([0.3, 0.4, 0.2])
    gini = somersd_pairwise(pos_scores, neg_scores, ties="y")
    logger.info(f"Perfect separation Somers' D: {gini}")
    assert gini == 1.0, "Perfect separation should give Somers' D = 1.0"

    # Test 2: No separation (all ties)
    logger.info("=== Test 2: No separation ===")
    pos_scores = np.array([0.5, 0.5, 0.5])
    neg_scores = np.array([0.5, 0.5, 0.5])
    gini = somersd_pairwise(pos_scores, neg_scores, ties="y")
    logger.info(f"No separation Somers' D: {gini}")
    assert gini == 0.0, "All ties should give Somers' D = 0.0"

    # Test 3: Random data - compare with AUC-based Gini
    logger.info("=== Test 3: Random data vs AUC ===")
    np.random.seed(42)
    n_pos, n_neg = 50, 50
    pos_scores = np.random.uniform(0.5, 1.0, n_pos)
    neg_scores = np.random.uniform(0.0, 0.5, n_neg)

    # Create full arrays for AUC calculation
    y_true = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    y_pred = np.concatenate([pos_scores, neg_scores])

    somersd_pairwise_val = somersd_pairwise(pos_scores, neg_scores, ties="y")
    auc = roc_auc_score(y_true, y_pred)
    gini_from_auc = 2 * auc - 1

    logger.info(f"Pairwise Somers' D: {somersd_pairwise_val:.8f}")
    logger.info(f"Gini from AUC: {gini_from_auc:.8f}")

    assert np.isclose(somersd_pairwise_val, gini_from_auc, atol=1e-10), (
        "Pairwise Somers' D should match Gini from AUC"
    )

    # Test 4: Empty arrays
    logger.info("=== Test 4: Empty arrays ===")
    assert somersd_pairwise(np.array([]), np.array([1, 2]), ties="y") is None
    assert somersd_pairwise(np.array([1, 2]), np.array([]), ties="y") is None
    assert somersd_pairwise(np.array([]), np.array([]), ties="y") is None

    # Test 5: NaN handling
    logger.info("=== Test 5: NaN handling ===")
    pos_scores = np.array([0.8, np.nan, 0.9, 0.7])
    neg_scores = np.array([0.3, 0.4, np.nan, 0.2])
    gini = somersd_pairwise(pos_scores, neg_scores, ties="y")
    # Should work without NaNs
    assert gini is not None
    assert 0 <= gini <= 1

    logger.info("✓ All somersd_pairwise tests passed!")


def test_somersd_pairwise_ties_parameter():
    """Test somersd_pairwise with different ties parameter values."""
    logger.info("Testing somersd_pairwise with ties parameter...")

    # Test 1: Default ties="y"
    logger.info("=== Test 1: Default ties='y' ===")
    pos_scores = np.array([0.8, 0.9, 0.7])
    neg_scores = np.array([0.3, 0.4, 0.2])
    result_y = somersd_pairwise(pos_scores, neg_scores)
    result_y_explicit = somersd_pairwise(pos_scores, neg_scores, ties="y")
    assert result_y == result_y_explicit, "Default should be ties='y'"

    # Test 2: ties="x"
    logger.info("=== Test 2: ties='x' ===")
    result_x = somersd_pairwise(pos_scores, neg_scores, ties="x")
    logger.info(f"ties='y': {result_y:.8f}")
    logger.info(f"ties='x': {result_x:.8f}")

    # Test 3: Invalid ties parameter
    logger.info("=== Test 3: Invalid ties parameter ===")
    with pytest.raises(ValueError, match="ties must be 'x' or 'y'"):
        somersd_pairwise(pos_scores, neg_scores, ties="invalid")

    # Test 4: Compare with scipy for both ties options
    logger.info("=== Test 4: Compare with scipy for both ties options ===")
    np.random.seed(42)
    pos_scores = np.random.uniform(0.5, 1.0, 20)
    neg_scores = np.random.uniform(0.0, 0.5, 20)

    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_pred = np.concatenate([pos_scores, neg_scores])

    result_y = somersd_pairwise(pos_scores, neg_scores, ties="y")
    result_x = somersd_pairwise(pos_scores, neg_scores, ties="x")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scipy_yx = stats.somersd(y_true, y_pred).statistic
        scipy_xy = stats.somersd(y_pred, y_true).statistic

    logger.info(f"somersd_pairwise(ties='y'): {result_y:.8f}, scipy: {scipy_yx:.8f}")
    logger.info(f"somersd_pairwise(ties='x'): {result_x:.8f}, scipy: {scipy_xy:.8f}")

    assert np.isclose(result_y, scipy_yx, atol=1e-10)
    assert np.isclose(result_x, scipy_xy, atol=1e-10)

    logger.info("✓ All ties parameter tests passed!")


def test_somersd_clustered_matrix():
    """Test clustered Somers' D matrix computation."""
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from fastwoe.metrics import somersd_clustered_matrix

    logger.info("Testing somersd_clustered_matrix function...")

    # Create test data with clusters
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "score": np.random.uniform(0, 1, n),
            "label": np.random.binomial(1, 0.3, n),
            "cluster": np.random.choice(["C1", "C2", "C3"], n),
        }
    )

    # Compute clustered Somers' D matrix
    somersd_matrix, global_somersd = somersd_clustered_matrix(df, "score", "label", "cluster")

    logger.info("=== Clustered Somers' D Matrix ===")
    logger.info(f"\n{somersd_matrix}")

    # Test 1: Global Somers' D should match AUC * 2 - 1 (Gini)
    logger.info("=== Test 1: Global Somers' D == AUC * 2 - 1 ===")
    auc = roc_auc_score(df["label"], df["score"])
    gini_from_auc = 2 * auc - 1

    logger.info(f"Global Somers' D (from matrix): {global_somersd:.8f}")
    logger.info(f"Gini from AUC:                  {gini_from_auc:.8f}")

    assert np.isclose(global_somersd, gini_from_auc, atol=1e-10), (
        "Global Somers' D should match Gini from AUC"
    )

    # Test 2: Diagonal elements are intra-cluster Somers' D
    logger.info("=== Test 2: Diagonal elements ===")
    clusters = sorted(df["cluster"].unique())
    for cluster in clusters:
        cluster_df = df[df["cluster"] == cluster]
        if len(cluster_df) > 0:
            cluster_pos = cluster_df[cluster_df["label"] == 1]["score"].values
            cluster_neg = cluster_df[cluster_df["label"] == 0]["score"].values
            if len(cluster_pos) > 0 and len(cluster_neg) > 0:
                intra_cluster_somersd = somersd_pairwise(cluster_pos, cluster_neg, ties="y")
                diagonal_value = somersd_matrix.loc[cluster, cluster]

                logger.info(
                    f"Cluster {cluster}: diagonal={diagonal_value:.6f}, "
                    f"computed={intra_cluster_somersd:.6f}"
                )

                if intra_cluster_somersd is not None:
                    assert np.isclose(diagonal_value, intra_cluster_somersd, atol=1e-10), (
                        f"Diagonal element for {cluster} should match intra-cluster Somers' D"
                    )

    # Test 3: Matrix shape
    logger.info("=== Test 3: Matrix shape ===")
    assert somersd_matrix.shape[0] == len(clusters)
    assert somersd_matrix.shape[1] == len(clusters)
    assert list[Any](somersd_matrix.index) == clusters
    assert list[Any](somersd_matrix.columns) == clusters

    logger.info("✓ All somersd_clustered_matrix tests passed!")


def test_somersd_clustered_matrix_requires_binary():
    """Test that somersd_clustered_matrix raises error for non-binary labels."""
    import pandas as pd

    from fastwoe.metrics import somersd_clustered_matrix

    logger.info("Testing somersd_clustered_matrix error handling for non-binary labels...")

    # Create test data with continuous target
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "score": np.random.uniform(0, 1, n),
            "target": np.random.uniform(0, 100, n),  # Continuous target
            "cluster": np.random.choice(["C1", "C2", "C3"], n),
        }
    )

    # Should raise ValueError for non-binary labels
    try:
        somersd_clustered_matrix(df, "score", "target", "cluster")
        raise AssertionError("Should have raised ValueError for non-binary labels")
    except ValueError as e:
        assert "requires binary labels" in str(e).lower()
        logger.info(f"✓ Correctly raised ValueError: {e}")

    logger.info("✓ Non-binary label error handling test passed!")


def test_somersd_pairwise_binary_vs_auc():
    """Test somersd_pairwise against roc_auc * 2 - 1 for binary cases in individual segments."""
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    logger.info("Testing somersd_pairwise vs roc_auc * 2 - 1 for binary segments...")

    # Test 1: Individual segments with different characteristics
    logger.info("=== Test 1: Individual segments ===")
    np.random.seed(42)

    test_cases = [
        ("Perfect separation", np.array([0.8, 0.9, 0.95]), np.array([0.2, 0.1, 0.05])),
        ("No separation", np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])),
        (
            "Random small",
            np.random.uniform(0.6, 1.0, 20),
            np.random.uniform(0.0, 0.4, 20),
        ),
        (
            "Random medium",
            np.random.uniform(0.5, 1.0, 100),
            np.random.uniform(0.0, 0.5, 100),
        ),
        (
            "Random large",
            np.random.uniform(0.4, 1.0, 500),
            np.random.uniform(0.0, 0.6, 500),
        ),
        (
            "Overlapping",
            np.random.uniform(0.3, 0.7, 50),
            np.random.uniform(0.3, 0.7, 50),
        ),
    ]

    for name, pos_scores, neg_scores in test_cases:
        # Compute using somersd_pairwise
        somersd_result = somersd_pairwise(pos_scores, neg_scores, ties="y")

        # Compute using AUC
        y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        y_pred = np.concatenate([pos_scores, neg_scores])
        auc = roc_auc_score(y_true, y_pred)
        gini_from_auc = 2 * auc - 1

        logger.info(f"{name}:")
        logger.info(f"somersd_pairwise: {somersd_result:.8f}")
        logger.info(f"roc_auc * 2 - 1: {gini_from_auc:.8f}")
        logger.info(f"Difference: {abs(somersd_result - gini_from_auc):.2e}")

        assert np.isclose(somersd_result, gini_from_auc, atol=1e-10), (
            f"{name}: somersd_pairwise should match roc_auc * 2 - 1"
        )

    # Test 2: Clustered segments (like in somersd_clustered_matrix)
    logger.info("=== Test 2: Clustered segments ===")
    n = 200
    df = pd.DataFrame(
        {
            "score": np.random.uniform(0, 1, n),
            "label": np.random.binomial(1, 0.3, n),
            "cluster": np.random.choice(["C1", "C2", "C3"], n),
        }
    )

    clusters = sorted(df["cluster"].unique())
    for cluster in clusters:
        cluster_df = df[df["cluster"] == cluster]
        cluster_pos = cluster_df[cluster_df["label"] == 1]["score"].values
        cluster_neg = cluster_df[cluster_df["label"] == 0]["score"].values

        if len(cluster_pos) > 0 and len(cluster_neg) > 0:
            # Compute using somersd_pairwise
            somersd_result = somersd_pairwise(cluster_pos, cluster_neg, ties="y")

            # Compute using AUC
            y_true = np.concatenate([np.ones(len(cluster_pos)), np.zeros(len(cluster_neg))])
            y_pred = np.concatenate([cluster_pos, cluster_neg])
            auc = roc_auc_score(y_true, y_pred)
            gini_from_auc = 2 * auc - 1

            logger.info(f"Cluster {cluster}:")
            logger.info(f"  somersd_pairwise: {somersd_result:.8f}")
            logger.info(f"  roc_auc * 2 - 1:  {gini_from_auc:.8f}")

            assert np.isclose(somersd_result, gini_from_auc, atol=1e-10), (
                f"Cluster {cluster}: somersd_pairwise should match roc_auc * 2 - 1"
            )

    logger.info("✓ All binary vs AUC tests passed!")


def test_somersd_pairwise_vs_scipy():
    """Test somersd_pairwise against scipy.somersd for smaller datasets."""
    logger.info("Testing somersd_pairwise vs scipy.somersd for smaller datasets...")

    # Test 1: Small datasets with different characteristics
    logger.info("=== Test 1: Small datasets ===")
    np.random.seed(42)

    test_cases = [
        ("Perfect positive", np.array([0.8, 0.9, 0.95]), np.array([0.2, 0.1, 0.05])),
        ("Perfect negative", np.array([0.2, 0.1, 0.05]), np.array([0.8, 0.9, 0.95])),
        ("No separation", np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])),
        (
            "Small random",
            np.random.uniform(0.6, 1.0, 10),
            np.random.uniform(0.0, 0.4, 10),
        ),
        (
            "Medium random",
            np.random.uniform(0.5, 1.0, 50),
            np.random.uniform(0.0, 0.5, 50),
        ),
        ("With ties", np.array([0.5, 0.5, 0.7, 0.8]), np.array([0.3, 0.3, 0.4, 0.4])),
    ]

    for name, pos_scores, neg_scores in test_cases:
        # Compute using somersd_pairwise with ties="y"
        somersd_yx_result = somersd_pairwise(pos_scores, neg_scores, ties="y")

        # Compute using somersd_pairwise with ties="x"
        somersd_xy_result = somersd_pairwise(pos_scores, neg_scores, ties="x")

        # Compute using scipy
        y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        y_pred = np.concatenate([pos_scores, neg_scores])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scipy_yx = stats.somersd(y_true, y_pred).statistic
            scipy_xy = stats.somersd(y_pred, y_true).statistic

        logger.info(f"{name}:")
        logger.info(f"  somersd_pairwise(ties='y'): {somersd_yx_result:.8f}")
        logger.info(f"scipy.somersd(y, x): {scipy_yx:.8f}")
        if not (np.isnan(scipy_yx) and somersd_yx_result == 0.0):
            logger.info(f"Difference (yx): {abs(somersd_yx_result - scipy_yx):.2e}")
        else:
            logger.info("Difference (yx): Both are 0.0 or NaN (no separation)")

        if somersd_xy_result is not None:
            logger.info(f"somersd_pairwise(ties='x'): {somersd_xy_result:.8f}")
        else:
            logger.info("somersd_pairwise(ties='x'): None")
        logger.info(f"scipy.somersd(x, y): {scipy_xy:.8f}")
        if somersd_xy_result is not None and not (np.isnan(scipy_xy) and somersd_xy_result == 0.0):
            logger.info(f"Difference (xy): {abs(somersd_xy_result - scipy_xy):.2e}")
        else:
            logger.info("Difference (xy): Both are 0.0, None, or NaN (no separation)")

        # Handle NaN cases: when scipy returns NaN (no separation), our function returns 0.0 or None
        if np.isnan(scipy_yx):
            assert (
                somersd_yx_result == 0.0
                or (somersd_yx_result is not None and np.isnan(somersd_yx_result))
                or somersd_yx_result is None
            ), f"{name}: When scipy returns NaN, somersd_pairwise should return 0.0, None, or NaN"
        else:
            assert somersd_yx_result is not None, (
                f"{name}: somersd_pairwise(ties='y') should not return None"
            )
            assert np.isclose(somersd_yx_result, scipy_yx, atol=1e-10), (
                f"{name}: somersd_pairwise(ties='y') should match scipy.somersd(y, x)"
            )

        if np.isnan(scipy_xy):
            assert (
                somersd_xy_result == 0.0
                or (somersd_xy_result is not None and np.isnan(somersd_xy_result))
                or somersd_xy_result is None
            ), f"{name}: When scipy returns NaN, somersd_pairwise should return 0.0, None, or NaN"
        else:
            assert somersd_xy_result is not None, (
                f"{name}: somersd_pairwise(ties='x') should not return None"
            )
            assert np.isclose(somersd_xy_result, scipy_xy, atol=1e-10), (
                f"{name}: somersd_pairwise(ties='x') should match scipy.somersd(x, y)"
            )

    # Test 2: Multiple random datasets of varying sizes
    logger.info("=== Test 2: Varying dataset sizes ===")
    sizes = [5, 10, 20, 50, 100]

    for size in sizes:
        np.random.seed(42 + size)  # Different seed for each size
        pos_scores = np.random.uniform(0.4, 1.0, size)
        neg_scores = np.random.uniform(0.0, 0.6, size)

        # Compute using somersd_pairwise
        somersd_yx_result = somersd_pairwise(pos_scores, neg_scores, ties="y")
        somersd_xy_result = somersd_pairwise(pos_scores, neg_scores, ties="x")

        # Compute using scipy
        y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        y_pred = np.concatenate([pos_scores, neg_scores])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scipy_yx = stats.somersd(y_true, y_pred).statistic
            scipy_xy = stats.somersd(y_pred, y_true).statistic

        logger.info(f"Size {size}:")
        logger.info(f"somersd_pairwise(ties='y'): {somersd_yx_result:.8f}, scipy: {scipy_yx:.8f}")
        logger.info(f"somersd_pairwise(ties='x'): {somersd_xy_result:.8f}, scipy: {scipy_xy:.8f}")

        # Handle NaN cases
        if np.isnan(scipy_yx):
            assert (
                somersd_yx_result == 0.0
                or (somersd_yx_result is not None and np.isnan(somersd_yx_result))
                or somersd_yx_result is None
            ), (
                f"Size {size}: When scipy returns NaN, somersd_pairwise should return 0.0, None, or NaN"
            )
        else:
            assert somersd_yx_result is not None, (
                f"Size {size}: somersd_pairwise(ties='y') should not return None"
            )
            assert np.isclose(somersd_yx_result, scipy_yx, atol=1e-10), (
                f"Size {size}: somersd_pairwise(ties='y') should match scipy"
            )

        if np.isnan(scipy_xy):
            assert (
                somersd_xy_result == 0.0
                or (somersd_xy_result is not None and np.isnan(somersd_xy_result))
                or somersd_xy_result is None
            ), (
                f"Size {size}: When scipy returns NaN, somersd_pairwise should return 0.0, None, or NaN"
            )
        else:
            assert somersd_xy_result is not None, (
                f"Size {size}: somersd_pairwise(ties='x') should not return None"
            )
            assert np.isclose(somersd_xy_result, scipy_xy, atol=1e-10), (
                f"Size {size}: somersd_pairwise(ties='x') should match scipy"
            )

    logger.info("✓ All scipy comparison tests passed!")


if __name__ == "__main__":
    # Allow running this file directly for development
    print("🧪 Running FastSomersD tests...")
    pytest.main([__file__, "-v"])
