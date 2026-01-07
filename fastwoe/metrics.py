"""
Model performance metrics.

Implements Somers' D, Gini coefficient, and clustered Gini analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

# Try to import numba, with fallback for environments where it's not available or has issues
try:
    from numba import njit

    _HAS_NUMBA = True
except (ImportError, OSError, MemoryError) as e:
    # Fallback: use a no-op decorator when numba is not available
    # This allows the code to run, but without JIT compilation (slower)
    import warnings

    warnings.warn(
        f"Numba not available or failed to import ({type(e).__name__}: {e}). "
        "Performance will be degraded. If this is unexpected, check numba/llvmlite installation.",
        UserWarning,
        stacklevel=2,
    )
    _HAS_NUMBA = False

    def njit(func: Callable) -> Callable:
        """No-op decorator when numba is not available."""
        return func


@dataclass(frozen=True)
class SomersDResult:
    """Container for Somers' D computation results."""

    statistic: float
    concordant_pairs: int
    discordant_pairs: int
    ties: int
    total_pairs: int
    denominator: int

    def __repr__(self):
        return (
            f"SomersDResult(statistic={self.statistic:.6f}, "
            f"concordant_pairs={self.concordant_pairs}, "
            f"discordant_pairs={self.discordant_pairs}, "
            f"ties={self.ties}, "
            f"total_pairs={self.total_pairs}, "
            f"denominator={self.denominator})"
        )


@njit
def _fenwick_update(bit: np.ndarray, i: int, delta: int) -> None:
    n = bit.size
    while i <= n:
        bit[i - 1] += delta
        i += i & -i


@njit
def _fenwick_query(bit: np.ndarray, i: int) -> int:
    s = 0
    while i > 0:
        s += int(bit[i - 1])
        i -= i & -i
    return s


@njit
def _somers_yx_weighted(
    y: np.ndarray, x: np.ndarray, weights: np.ndarray
) -> tuple[float, float, float, float, float, float]:
    """Compute weighted Somers' D_{Y|X}."""
    n = y.size
    if n < 2:
        return np.nan, 0.0, 0.0, 0.0, 0.0, 0.0

    # Calculate weighted concordant and discordant pairs
    concordant = 0.0
    discordant = 0.0
    ties_y = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            w_ij = weights[i] * weights[j]

            if y[i] == y[j]:  # Tied in Y
                ties_y += w_ij

            elif (y[i] < y[j] and x[i] < x[j]) or (y[i] > y[j] and x[i] > x[j]):
                concordant += w_ij
            elif (y[i] < y[j] and x[i] > x[j]) or (y[i] > y[j] and x[i] < x[j]):
                discordant += w_ij
    total_pairs = concordant + discordant + ties_y
    denom = concordant + discordant  # Exclude ties in Y from denominator

    stat = (concordant - discordant) / denom if denom > 0 else np.nan
    return stat, concordant, discordant, ties_y, total_pairs, denom


@njit
def _somers_yx_core(y: np.ndarray, x: np.ndarray) -> tuple[float, int, int, int, int, int]:
    """Compute Somers' D_{Y|X} (ties computed in Y)."""
    n = y.size
    if n < 2:
        return np.nan, 0, 0, 0, 0, 0

    # Compute ties in Y
    y_sorted = np.sort(y)
    Ty = 0
    run = 1
    for i in range(1, n):
        if y_sorted[i] == y_sorted[i - 1]:
            run += 1
        else:
            Ty += run * (run - 1) // 2
            run = 1
    Ty += run * (run - 1) // 2

    uniq = [y_sorted[0]]
    for i in range(1, n):  # sourcery skip: for-append-to-extend
        if y_sorted[i] != y_sorted[i - 1]:
            uniq.append(y_sorted[i])
    m = len(uniq)

    # ranks array
    y_rank = np.empty(n, dtype=np.int64)
    for i in range(n):
        lo, hi = 0, m
        v = y[i]
        while lo < hi:
            mid = (lo + hi) // 2
            if uniq[mid] < v:
                lo = mid + 1
            else:
                hi = mid
        y_rank[i] = lo + 1

    idx = np.argsort(x, kind="mergesort")
    bit = np.zeros(m, dtype=np.int64)
    processed = 0
    S = 0

    t = 0
    while t < n:
        t2 = t + 1
        xv = x[idx[t]]
        while t2 < n and x[idx[t2]] == xv:
            t2 += 1

        for k in range(t, t2):
            r = y_rank[idx[k]]
            less = _fenwick_query(bit, r - 1)
            greater = processed - _fenwick_query(bit, r)
            S += less - greater

        for k in range(t, t2):
            r = y_rank[idx[k]]
            _fenwick_update(bit, r, 1)
            processed += 1

        t = t2

    P = n * (n - 1) // 2
    denom = P - Ty
    stat = S / denom if denom > 0 else np.nan

    # Fix the concordant/discordant calculation
    concordant = (S + denom) // 2  # C - D = S, C + D = denom, so C = (S + denom) / 2
    discordant = denom - concordant  # D = denom - C

    return stat, concordant, discordant, Ty, P, denom


@njit
def _somers_xy_core(y: np.ndarray, x: np.ndarray) -> tuple[float, int, int, int, int, int]:
    """Compute Somers' D_{X|Y} (ties computed in X)."""
    n = y.size
    if n < 2:
        return np.nan, 0, 0, 0, 0, 0

    # Compute ties in X
    x_sorted = np.sort(x)
    Tx = 0
    run = 1
    for i in range(1, n):
        if x_sorted[i] == x_sorted[i - 1]:
            run += 1
        else:
            Tx += run * (run - 1) // 2
            run = 1
    Tx += run * (run - 1) // 2

    # Get unique Y values for ranking
    y_sorted = np.sort(y)
    uniq = [y_sorted[0]]
    for i in range(1, n):  # sourcery skip: for-append-to-extend
        if y_sorted[i] != y_sorted[i - 1]:
            uniq.append(y_sorted[i])
    m = len(uniq)

    # Rank Y values
    y_rank = np.empty(n, dtype=np.int64)
    for i in range(n):
        lo, hi = 0, m
        v = y[i]
        while lo < hi:
            mid = (lo + hi) // 2
            if uniq[mid] < v:
                lo = mid + 1
            else:
                hi = mid
        y_rank[i] = lo + 1

    # Sort by X values and process with Fenwick tree
    idx = np.argsort(x, kind="mergesort")
    bit = np.zeros(m, dtype=np.int64)
    processed = 0
    S = 0

    t = 0
    while t < n:
        t2 = t + 1
        xv = x[idx[t]]
        while t2 < n and x[idx[t2]] == xv:
            t2 += 1

        for k in range(t, t2):
            r = y_rank[idx[k]]
            less = _fenwick_query(bit, r - 1)
            greater = processed - _fenwick_query(bit, r)
            S += less - greater

        for k in range(t, t2):
            r = y_rank[idx[k]]
            _fenwick_update(bit, r, 1)
            processed += 1

        t = t2

    P = n * (n - 1) // 2
    denom = P - Tx  # Denominator excludes ties in X
    stat = S / denom if denom > 0 else np.nan

    # Fix the concordant/discordant calculation
    concordant = (S + denom) // 2  # C - D = S, C + D = denom, so C = (S + denom) / 2
    discordant = denom - concordant  # D = denom - C

    return stat, concordant, discordant, Tx, P, denom


def somersd_yx(
    y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray | None = None
) -> SomersDResult:
    """Compute Somers' D_{Y|X} (ties in Y excluded from denominator).

    Args:
        y_true: True binary labels
        y_pred: Predicted scores
        weights: Optional sample weights. If provided, uses weighted AUC calculation.

    Returns:
        SomersDResult with statistic, concordant, discordant, ties, total_pairs, denominator

    Note:
        When weights are provided, falls back to sklearn's weighted AUC calculation
        since weighted concordance requires different algorithm.
    """
    y = np.asarray(y_true, dtype=np.float64)
    x = np.asarray(y_pred, dtype=np.float64)
    mask = ~(np.isnan(y) | np.isnan(x))
    y = y[mask]
    x = x[mask]

    if weights is not None:
        # Weighted case: use weighted concordance calculation
        weights = np.asarray(weights, dtype=np.float64)[mask]
        stat, S, D, Ty, P, denom = _somers_yx_weighted(y, x, weights)
        return SomersDResult(stat, S, D, Ty, P, denom)

    # Unweighted case: use fast Numba implementation
    stat, S, D, Ty, P, denom = _somers_yx_core(y, x)
    return SomersDResult(stat, S, D, Ty, P, denom)


def somersd_xy(y_true: np.ndarray, y_pred: np.ndarray) -> SomersDResult:
    """Compute Somers' D_{X|Y} (ties in X excluded from denominator)."""
    y = np.asarray(y_true, dtype=np.float64)
    x = np.asarray(y_pred, dtype=np.float64)
    mask = ~(np.isnan(y) | np.isnan(x))
    y = y[mask]
    x = x[mask]
    stat, S, D, Tx, P, denom = _somers_xy_core(y, x)
    return SomersDResult(stat, S, D, Tx, P, denom)


def somersd_pairwise(
    pos_scores: np.ndarray, neg_scores: np.ndarray, ties: str = "y"
) -> Optional[float]:
    """Compute pairwise Somers' D between positive and negative scores.

    This function computes Somers' D by comparing all positive scores
    against all negative scores. It's used for clustered Gini analysis where
    you want to measure separation between different groups.

    The computation leverages the fast Somers' D implementation for optimal
    performance, which uses efficient Numba-accelerated algorithms.

    Args:
        pos_scores: Array of scores for positive class (label=1)
        neg_scores: Array of scores for negative class (label=0)
        ties: How to handle ties. "y" (default) computes D_Y|X (ties in Y excluded),
              "x" computes D_X|Y (ties in X excluded).

    Returns:
        Somers' D statistic (net concordant pairs / total pairs), or None if
        either array is empty.

    Note:
        Somers' D is computed by combining the scores into a single array with
        binary labels (1 for positive, 0 for negative). This leverages the
        efficient O(n log n) algorithm instead of O(n_pos * n_neg).

        For binary classification, Somers' D equals the Gini coefficient
        (2 * AUC - 1).

    Examples:
        >>> pos = np.array([0.8, 0.9, 0.7])
        >>> neg = np.array([0.3, 0.4, 0.2])
        >>> somersd_pairwise(pos, neg)
        1.0  # Perfect separation
        >>> somersd_pairwise(pos, neg, ties="x")
        1.0  # Same result for perfect separation
    """
    if ties not in ("x", "y"):
        raise ValueError(f"ties must be 'x' or 'y', got {ties}")

    pos_scores = np.asarray(pos_scores, dtype=np.float64)
    neg_scores = np.asarray(neg_scores, dtype=np.float64)

    # Remove NaN values
    pos_mask = ~np.isnan(pos_scores)
    neg_mask = ~np.isnan(neg_scores)
    pos_scores = pos_scores[pos_mask]
    neg_scores = neg_scores[neg_mask]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return None

    # Combine scores and create binary labels
    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate(
        [
            np.ones(len(pos_scores), dtype=np.float64),
            np.zeros(len(neg_scores), dtype=np.float64),
        ]
    )

    # Use fast Somers' D implementation (O(n log n) instead of O(n_pos * n_neg))
    if ties == "y":
        result = somersd_yx(all_labels, all_scores)
    else:  # ties == "x"
        result = somersd_xy(all_labels, all_scores)

    statistic = result.statistic

    return None if np.isnan(statistic) else float(statistic)


def somersd_clustered_matrix(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,  # Must be binary (0/1)
    cluster_col: str,
    ties: str = "y",
) -> tuple[pd.DataFrame, Optional[float]]:
    """Compute intra/inter-cluster Somers' D matrix for binary classification.

    Based on: Sudjianto and Liu (2025), https://doi.org/10.48550/arXiv.2508.07495.

    Each element (i, j) represents Somers' D between:
    - Positive scores from cluster i
    - Negative scores from cluster j

    Diagonal elements (i, i) measure intra-cluster separation.
    Off-diagonal elements (i, j) measure inter-cluster separation.

    Args:
        df: DataFrame containing scores, binary labels, and cluster assignments
        score_col: Column name for model scores
        label_col: Column name for binary labels (must be 0/1)
        cluster_col: Column name for cluster assignments
        ties: "y" (default) for D_Y|X, "x" for D_X|Y

    Returns:
        (somersd_matrix, global_somersd) where:
        - somersd_matrix: DataFrame with clusters as index/columns
        - global_somersd: Overall Somers' D across all data

    Raises:
        ValueError: If label_col is not binary (0/1)

    Examples:
        >>> df = pd.DataFrame({
        ...     'score': [0.8, 0.9, 0.3, 0.4, 0.7, 0.6],
        ...     'label': [1, 1, 0, 0, 1, 0],
        ...     'cluster': ['C1', 'C1', 'C1', 'C2', 'C2', 'C2']
        ... })
        >>> matrix, global_somersd = somersd_clustered_matrix(
        ...     df, 'score', 'label', 'cluster'
        ... )
        >>> print(matrix)
        >>> print(f"Global Somers' D: {global_somersd}")
    """
    if ties not in ("x", "y"):
        raise ValueError(f"ties must be 'x' or 'y', got {ties}")

    # Detect if target is binary
    unique_labels = df[label_col].dropna().unique()
    is_binary = set[Any](unique_labels).issubset({0, 1}) and len(unique_labels) <= 2

    if not is_binary:
        raise ValueError(
            f"somersd_clustered_matrix requires binary labels (0/1). "
            f"Got labels: {sorted(unique_labels)}"
        )

    clusters = sorted(df[cluster_col].unique())
    somersd_matrix = pd.DataFrame(index=clusters, columns=clusters, dtype=float)

    high_mask = df[label_col] == 1
    low_mask = df[label_col] == 0

    # Compute intra/inter-cluster Somers' D
    for ci in clusters:
        for cj in clusters:
            high_scores = df[(df[cluster_col] == ci) & high_mask][score_col].values
            low_scores = df[(df[cluster_col] == cj) & low_mask][score_col].values
            somersd_matrix.loc[ci, cj] = somersd_pairwise(high_scores, low_scores, ties=ties)

    # Compute global Somers' D
    global_high_scores = df[high_mask][score_col].values
    global_low_scores = df[low_mask][score_col].values
    global_somersd = somersd_pairwise(global_high_scores, global_low_scores, ties=ties)

    return somersd_matrix, global_somersd
