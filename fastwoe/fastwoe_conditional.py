"""Conditional Weight of Evidence via Good's chain rule.

A standard scorecard measures every feature against the whole book and adds
the weights. That is exact only when the features are independent. When they
are not, the shared signal is counted once per feature.

Good's chain rule gives the exact decomposition instead::

    W(H : E1 E2) = W(H : E1) + W(H : E2 | E1)

Each weight after the first is measured inside the population the earlier
features already selected, so the weights still add and no independence is
assumed. The price is sparsity: conditioning cells multiply, and a cell with
too few observations in either class gives an unstable or infinite weight.
Cells below ``min_cell_count`` therefore fall back to the marginal weight,
and every fallback is recorded rather than hidden.

The decomposition is order-dependent. ``W(E2 | E1) != W(E1 | E2)``, though the
total is the same either way: order changes how credit is attributed across
features, not the score.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd

__all__ = ["ConditionalWoeMixin"]


class ConditionalWoeMixin:
    """Conditional WOE along an explicit conditioning order.

    Mixed into :class:`FastWoe`, which supplies the binning, the prior and
    :meth:`_calculate_woe_se`.
    """

    # populated by fit_conditional()
    conditional_order_: list[str]
    conditional_weights_: dict[tuple[str, tuple], pd.DataFrame]
    conditional_fallbacks_: list[dict[str, Any]]
    min_cell_count_: int

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _binned_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted binners, so conditioning happens on bins."""
        out = X.copy()
        for col in X.columns:
            if col in self.binners_:
                out[col] = self._apply_binning_to_column(out, col)
        return out

    @staticmethod
    def _counts(mask: np.ndarray, y: np.ndarray) -> tuple[int, int]:
        """(bad, good) counts under a boolean mask."""
        sel = y[mask]
        return int((sel == 1).sum()), int((sel == 0).sum())

    def _weight(
        self,
        bad_sel: int,
        good_sel: int,
        bad_tot: int,
        good_tot: int,
    ) -> tuple[float, float]:
        """WOE and its standard error for one cell, or (nan, nan) if degenerate."""
        if min(bad_sel, good_sel, bad_tot, good_tot) == 0:
            return float("nan"), float("nan")
        lr = (bad_sel / bad_tot) / (good_sel / good_tot)
        return float(np.log(lr)), float(self._calculate_woe_se(good_sel, bad_sel))

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit_conditional(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        order: list[str],
        min_cell_count: int = 30,
    ):
        """Learn conditional weights along ``order``.

        Parameters
        ----------
        X, y
            Training data. ``fit`` must have been called first, so the
            binners and the prior are available.
        order
            Feature names in conditioning order. Step ``i`` is measured
            inside the cell picked out by steps ``1..i-1``. The first
            feature's weight is the ordinary marginal WOE.
        min_cell_count
            Minimum observations of each class in a conditioning cell. Below
            this the marginal weight is used instead and the fallback is
            recorded in ``conditional_fallbacks_``.
        """
        if not getattr(self, "is_fitted_", False):
            raise ValueError("call fit() before fit_conditional()")
        missing = [c for c in order if c not in X.columns]
        if missing:
            raise ValueError(f"features not in X: {missing}")
        if len(set(order)) != len(order):
            raise ValueError(f"order must not repeat a feature: {order}")

        Xb = self._binned_frame(X[order])
        yv = np.asarray(y)

        self.conditional_order_ = list(order)
        self.conditional_weights_ = {}
        self.conditional_fallbacks_ = []
        self.min_cell_count_ = int(min_cell_count)

        # paths: the distinct value combinations of the features already used
        paths: list[tuple] = [()]
        for depth, col in enumerate(order):
            prev = order[:depth]
            for path in paths:
                mask = np.ones(len(Xb), dtype=bool)
                for c, v in zip(prev, path):
                    mask &= (Xb[c] == v).to_numpy()
                bad_tot, good_tot = self._counts(mask, yv)
                rows = []
                for value in sorted(Xb.loc[mask, col].unique(), key=str):
                    cell = mask & (Xb[col] == value).to_numpy()
                    bad_sel, good_sel = self._counts(cell, yv)
                    thin = min(bad_tot, good_tot) < self.min_cell_count_
                    woe, se = self._weight(bad_sel, good_sel, bad_tot, good_tot)
                    fallback = thin or np.isnan(woe)
                    if fallback:
                        woe, se = self._marginal_weight(col, value)
                        self.conditional_fallbacks_.append(
                            {
                                "feature": col,
                                "given": dict(zip(prev, path)),
                                "category": value,
                                "bad": bad_sel,
                                "good": good_sel,
                                "bad_in_cell": bad_tot,
                                "good_in_cell": good_tot,
                                "reason": "thin cell" if thin else "empty class",
                            }
                        )
                    rows.append(
                        {
                            "category": value,
                            "bad_count": bad_sel,
                            "good_count": good_sel,
                            "bad_in_cell": bad_tot,
                            "good_in_cell": good_tot,
                            "woe": woe,
                            "woe_se": se,
                            "fallback": fallback,
                        }
                    )
                if rows:
                    self.conditional_weights_[(col, path)] = pd.DataFrame(rows).set_index(
                        "category"
                    )
            # extend the paths by this feature's observed values
            if depth + 1 < len(order):
                paths = [
                    (*p, v)
                    for p in paths
                    for v in self.conditional_weights_.get(
                        (col, p), pd.DataFrame()
                    ).index
                ]

        if self.conditional_fallbacks_:
            warnings.warn(
                f"{len(self.conditional_fallbacks_)} conditioning cells had fewer than "
                f"{self.min_cell_count_} of a class and fell back to the marginal "
                f"weight. See conditional_fallbacks_; consider a shorter order, "
                f"coarser bins, or a lower min_cell_count.",
                UserWarning,
                stacklevel=2,
            )
        return self

    def _marginal_weight(self, col: str, value: Any) -> tuple[float, float]:
        """The ordinary scorecard weight, used where a conditional cell is too thin."""
        mapping = self.mappings_.get(col)
        if mapping is None or value not in mapping.index:
            return 0.0, float("nan")
        return float(mapping.loc[value, "woe"]), float(mapping.loc[value, "woe_se"])

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------

    def transform_conditional(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-feature conditional weights, one column per feature in order.

        The columns sum to the total weight of evidence for each row, so the
        decomposition is exact by construction rather than approximated.
        """
        if not hasattr(self, "conditional_weights_"):
            raise ValueError("call fit_conditional() before transform_conditional()")
        order = self.conditional_order_
        Xb = self._binned_frame(X[order])
        out = pd.DataFrame(index=X.index, columns=order, dtype=float)
        for pos, row in enumerate(Xb.itertuples(index=False)):
            values = dict(zip(order, row))
            for depth, col in enumerate(order):
                path = tuple(values[c] for c in order[:depth])
                table = self.conditional_weights_.get((col, path))
                value = values[col]
                if table is None or value not in table.index:
                    woe, _ = self._marginal_weight(col, value)
                else:
                    woe = float(table.loc[value, "woe"])
                out.iloc[pos, depth] = woe
        return out

    def predict_conditional_log_odds(self, X: pd.DataFrame) -> np.ndarray:
        """Prior log-odds plus the conditional weights: the scorecard score."""
        prior = float(np.log(self.odds_prior_))
        return prior + self.transform_conditional(X).sum(axis=1).to_numpy()

    # ------------------------------------------------------------------
    # inspection
    # ------------------------------------------------------------------

    def conditional_summary(self, feature: Optional[str] = None) -> pd.DataFrame:
        """All learned conditional weights as one tidy frame."""
        if not hasattr(self, "conditional_weights_"):
            raise ValueError("call fit_conditional() first")
        frames = []
        for (col, path), table in self.conditional_weights_.items():
            if feature is not None and col != feature:
                continue
            given = ", ".join(
                f"{c}={v}" for c, v in zip(self.conditional_order_, path)
            )
            frame = table.reset_index()
            frame.insert(0, "given", given or "-")
            frame.insert(0, "feature", col)
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def check_chain_rule(self, X: pd.DataFrame, y: pd.Series, tol: float = 1e-9) -> dict:
        """Verify the weights add to the joint weight of evidence.

        For every path with no fallback, the conditional weights along it must
        equal the log likelihood ratio of the whole combination. A failure here
        means the decomposition is not the thing it claims to be.
        """
        order = self.conditional_order_
        Xb = self._binned_frame(X[order])
        yv = np.asarray(y)
        bad_tot, good_tot = self._counts(np.ones(len(Xb), dtype=bool), yv)
        checked = failed = skipped = 0
        worst = 0.0
        for combo, group in Xb.groupby(order, observed=True):
            combo = combo if isinstance(combo, tuple) else (combo,)
            mask = np.zeros(len(Xb), dtype=bool)
            mask[Xb.index.get_indexer(group.index)] = True
            bad_sel, good_sel = self._counts(mask, yv)
            if min(bad_sel, good_sel) == 0:
                skipped += 1
                continue
            joint = np.log((bad_sel / bad_tot) / (good_sel / good_tot))
            total, used_fallback = 0.0, False
            for depth, col in enumerate(order):
                table = self.conditional_weights_.get((col, combo[:depth]))
                if table is None or combo[depth] not in table.index:
                    used_fallback = True
                    break
                row = table.loc[combo[depth]]
                used_fallback |= bool(row["fallback"])
                total += float(row["woe"])
            if used_fallback:
                skipped += 1
                continue
            checked += 1
            worst = max(worst, abs(joint - total))
            failed += abs(joint - total) > tol
        return {
            "paths_checked": checked,
            "paths_skipped": skipped,
            "failures": failed,
            "max_abs_error": worst,
            # nothing checked is not evidence of exactness
            "exact": failed == 0 and checked > 0,
        }
