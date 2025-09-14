"""test_fast_somersd.py."""

import logging
import time
import warnings

import numpy as np
import pytest
from scipy import stats

from fastwoe.fast_somersd import somersd_xy, somersd_yx

logger = logging.getLogger(__name__)


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

    logger.info("=== %s ===", name)
    logger.info("Our D_Y|X: %s", ryx.statistic)
    logger.info("SciPy D_Y|X: %s", syx.statistic)
    logger.info("Our D_X|Y: %s", rxy.statistic)
    logger.info("SciPy D_X|Y: %s", sxy.statistic)

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

    logger.info("Performance test with n=%s samples", n)

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

    logger.info("FastWoe time: %.4fs", our_time)
    logger.info("SciPy time: %.4fs", scipy_time)
    logger.info("Speedup: %.2fx", speedup)
    logger.info("FastWoe D_Y|X: %.6f", r.statistic)
    logger.info("SciPy D_Y|X: %.6f", s.statistic)
    logger.info("FastWoe D_X|Y: %.6f", r.statistic)
    logger.info("SciPy D_X|Y: %.6f", s.statistic)

    # Actually verify correctness
    assert abs(r.statistic - s.statistic) < 1e-10


if __name__ == "__main__":
    # Allow running this file directly for development
    print("🧪 Running FastSomersD tests...")
    pytest.main([__file__, "-v"])
