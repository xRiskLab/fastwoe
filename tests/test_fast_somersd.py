"""test_fast_somersd.py."""

import time
import warnings

import numpy as np
import pytest
from loguru import logger
from rich.logging import RichHandler
from scipy import stats

from fastwoe.fast_somersd import somersd_xy, somersd_yx

# Configure logger with RichHandler for better formatting
logger.remove()  # Remove default handler
logger.add(
    RichHandler(markup=True, rich_tracebacks=True),
    format="{message}",
    level="INFO",
)


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


if __name__ == "__main__":
    # Allow running this file directly for development
    print("🧪 Running FastSomersD tests...")
    pytest.main([__file__, "-v"])
