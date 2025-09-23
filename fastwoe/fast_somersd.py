"""fast_somersd.py."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


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
        s += bit[i - 1]
        i -= i & -i
    return s


@njit
def _somers_yx_core(
    y: np.ndarray, x: np.ndarray
) -> tuple[float, int, int, int, int, int]:
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
def _somers_xy_core(
    y: np.ndarray, x: np.ndarray
) -> tuple[float, int, int, int, int, int]:
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


def somersd_yx(y_true: np.ndarray, y_pred: np.ndarray) -> SomersDResult:
    """Compute Somers' D_{Y|X} (ties in Y excluded from denominator)."""
    y = np.asarray(y_true, dtype=np.float64)
    x = np.asarray(y_pred, dtype=np.float64)
    mask = ~(np.isnan(y) | np.isnan(x))
    y = y[mask]
    x = x[mask]
    stat, S, D, Ty, P, denom = _somers_yx_core(y, x)  # type: ignore[misc]
    return SomersDResult(stat, S, D, Ty, P, denom)


def somersd_xy(y_true: np.ndarray, y_pred: np.ndarray) -> SomersDResult:
    """Compute Somers' D_{X|Y} (ties in X excluded from denominator)."""
    y = np.asarray(y_true, dtype=np.float64)
    x = np.asarray(y_pred, dtype=np.float64)
    mask = ~(np.isnan(y) | np.isnan(x))
    y = y[mask]
    x = x[mask]
    stat, S, D, Tx, P, denom = _somers_xy_core(y, x)  # type: ignore[misc]
    return SomersDResult(stat, S, D, Tx, P, denom)
