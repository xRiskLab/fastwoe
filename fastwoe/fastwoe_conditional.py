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
A cell falls back to the marginal weight when it, or the group it is
conditioned within, has fewer than ``conditional_min_count`` of either class,
and every fallback is recorded rather than hidden.

The decomposition is order-dependent. ``W(E2 | E1) != W(E1 | E2)``, though the
total is the same either way: order changes how credit is attributed across
features, not the score. With two features that holds even with fallbacks (the
second step's cell is the same leaf in either order); with three or more,
fallbacks happen in different cells for different orders and can change it.

Enabled with ``FastWoe(conditional=True)``; ``fit``, ``transform``,
``predict_proba``, ``predict_ci`` and ``get_mapping`` then use the conditional
weights.
"""

from __future__ import annotations

import warnings
from collections import Counter
from typing import Any, Optional, cast

import numpy as np
import pandas as pd

__all__ = ["ConditionalWoeMixin"]


class ConditionalWoeMixin:
    """Conditional WOE along a conditioning order.

    Mixed into :class:`FastWoe`, which supplies the binning, the prior and
    :meth:`_calculate_woe_se`, and calls into this mixin when
    ``conditional=True``.
    """

    # set in FastWoe
    conditional: bool
    conditional_order: Optional[list[str]]
    conditional_min_count: int
    mappings_: dict[str, pd.DataFrame]
    binners_: dict[str, Any]
    is_binary_target: Optional[bool]
    odds_prior_: Optional[float]
    unseen_counts_: dict[str, dict[Any, int]]
    feature_stats_: dict[str, dict[str, Any]]
    _apply_binning_to_column: Any
    _calculate_woe_se: Any
    _calculate_iv: Any
    _calculate_iv_standard_error: Any
    _calculate_iv_confidence_interval: Any
    _handle_unseen: Any
    _ordered_mapping: Any

    # populated by fit() when conditional=True
    conditional_order_: list[str]
    conditional_weights_: dict[tuple[str, tuple], pd.DataFrame]
    conditional_fallbacks_: list[dict[str, Any]]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _binned_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted binners, so conditioning happens on bins.

        Missing values are normalised to ``np.nan`` so they compare and hash
        consistently when used as conditioning keys.
        """
        out = cast(pd.DataFrame, X.copy())
        for col in X.columns:
            if col in self.binners_:
                out[col] = self._apply_binning_to_column(out, col)
            elif out[col].dtype == object:
                out[col] = out[col].where(out[col].notna(), np.nan)
        return out

    @staticmethod
    def _is(column: pd.Series, value: Any) -> np.ndarray:
        """Rows of ``column`` equal to ``value``, where NaN matches NaN."""
        if pd.isna(value):
            return np.asarray(column.isna(), dtype=bool)
        return np.asarray(column == value, dtype=bool)

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

    def _marginal_row(self, col: str, value: Any) -> Optional[pd.Series]:
        """The marginal mapping row for ``value``, matching None and NaN alike."""
        mapping = self.mappings_.get(col)
        if mapping is None:
            return None
        if pd.isna(value):
            hits = np.asarray(mapping.index.isna())
            return mapping[hits].iloc[0] if hits.any() else None
        return mapping.loc[value] if value in mapping.index else None

    def _marginal_weight(self, col: str, value: Any) -> tuple[float, float]:
        """The ordinary scorecard weight, used where a conditional cell is too thin."""
        row = self._marginal_row(col, value)
        if row is None:
            return 0.0, float("nan")
        return float(row["woe"]), float(row["woe_se"])

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def _fit_conditional(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Learn conditional weights along the conditioning order.

        Called from ``fit()`` after the marginal fit, which supplies the
        binners, the prior and the marginal fallback weights. Step ``i`` is
        measured inside the cell picked out by steps ``1..i-1``; the first
        feature's weight is the ordinary marginal WOE.
        """
        if not self.is_binary_target:
            raise ValueError("conditional=True is only supported for binary (0/1) targets")
        order = list(X.columns) if self.conditional_order is None else list(self.conditional_order)
        if len(set(order)) != len(order):
            raise ValueError(f"conditional_order must not repeat a feature: {order}")
        if set(order) != set(X.columns):
            raise ValueError(
                f"conditional_order must list every column of X exactly once; "
                f"missing {sorted(set(X.columns) - set(order))}, "
                f"unknown {sorted(set(order) - set(X.columns))}"
            )

        Xb = self._binned_frame(X[order])
        yv = np.asarray(y)
        min_count = int(self.conditional_min_count)

        self.conditional_order_ = order
        self.conditional_weights_ = {}
        self.conditional_fallbacks_ = []

        # paths: the distinct value combinations of the features already used
        paths: list[tuple] = [()]
        for depth, col in enumerate(order):
            prev = order[:depth]
            for path in paths:
                mask = np.ones(len(Xb), dtype=bool)
                for c, v in zip(prev, path):
                    mask &= self._is(Xb[c], v)
                bad_tot, good_tot = self._counts(mask, yv)
                rows = []
                for value in sorted(Xb.loc[mask, col].unique(), key=str):
                    cell = mask & self._is(Xb[col], value)
                    bad_sel, good_sel = self._counts(cell, yv)
                    # both the group conditioned within and the cell itself need
                    # enough of each class for the weight to be worth using
                    thin_parent = min(bad_tot, good_tot) < min_count
                    thin_cell = min(bad_sel, good_sel) < min_count
                    thin = thin_parent or thin_cell
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
                                "reason": (
                                    "empty class"
                                    if min(bad_sel, good_sel) == 0
                                    else "thin parent"
                                    if thin_parent
                                    else "thin cell"
                                ),
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
                    for v in self.conditional_weights_.get((col, p), pd.DataFrame()).index
                ]

        self._conditional_iv(yv)

        if self.conditional_fallbacks_:
            warnings.warn(
                f"{len(self.conditional_fallbacks_)} conditioning cells had fewer than "
                f"{min_count} of a class and fell back to the marginal weight. See "
                f"conditional_fallbacks_; consider fewer features, coarser bins, or a "
                f"lower conditional_min_count.",
                UserWarning,
                stacklevel=3,
            )

    def _conditional_iv(self, y: np.ndarray) -> None:
        """Add each feature's IV given the features before it to ``feature_stats_``.

        ``IV(E2 | E1) = sum over cells of (p_bad - p_good) * W(E2 | E1)``, with
        ``p_bad`` and ``p_good`` each cell's share of all bads and goods. Like
        the weights, these add up to the joint IV of the features where no cell
        fell back; the order changes the split, not the total.

        Each cell is laid out like a marginal bin, so the IV, its delta-method
        standard error and its confidence interval come from the same helpers
        as the marginal IV.
        """
        total_bad, total_good = int((y == 1).sum()), int((y == 0).sum())
        for depth, col in enumerate(self.conditional_order_):
            cells = [
                table for (feature, _), table in self.conditional_weights_.items() if feature == col
            ]
            frame = pd.concat(cells, ignore_index=True)
            count = frame["bad_count"] + frame["good_count"]
            bins = pd.DataFrame(
                {
                    "count": count,
                    "event_rate": frame["bad_count"] / count,
                    "woe": frame["woe"],
                    "woe_se": frame["woe_se"].fillna(0.0),
                }
            )
            iv = float(self._calculate_iv(bins, total_good, total_bad))
            iv_se = float(self._calculate_iv_standard_error(bins, total_good, total_bad))
            lower, upper = self._calculate_iv_confidence_interval(iv, iv_se)
            self.feature_stats_[col].update(
                {
                    "iv_conditional": iv,
                    "iv_conditional_se": iv_se,
                    "iv_conditional_ci_lower": lower,
                    "iv_conditional_ci_upper": upper,
                    "conditioned_on": ", ".join(self.conditional_order_[:depth]) or "-",
                }
            )

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------

    def _conditional_path_weights(
        self, combo: tuple
    ) -> tuple[list[float], list[float], list[bool], list[bool]]:
        """Weights, SEs, fallback flags and unseen flags along one row's path."""
        order = self.conditional_order_
        woes, ses, fallbacks, unseen = [], [], [], []
        for depth, col in enumerate(order):
            value = combo[depth]
            table = self.conditional_weights_.get((col, combo[:depth]))
            if table is not None and value in table.index:
                row = table.loc[value]
                woe, se, fell_back = float(row["woe"]), float(row["woe_se"]), bool(row["fallback"])
            else:
                woe, se = self._marginal_weight(col, value)
                fell_back = True
            woes.append(woe)
            ses.append(se)
            fallbacks.append(fell_back)
            unseen.append(self._marginal_row(col, value) is None)
        return woes, ses, fallbacks, unseen

    def _transform_conditional(
        self, X: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Per-row conditional weights, their SEs and fallback flags.

        Each is a frame in X's column order. The weight columns sum to the
        total weight of evidence for each row. Weights are computed once per
        distinct combination of (binned) values and broadcast back to rows.
        """
        if not hasattr(self, "conditional_weights_"):
            raise ValueError("Model must be fitted before transforming data")
        order = self.conditional_order_
        missing = [c for c in order if c not in X.columns]
        if missing:
            raise ValueError(f"features seen during fit are missing from X: {missing}")
        Xb = self._binned_frame(X[order])

        groups = Xb.groupby(order, dropna=False, sort=False, observed=True)
        codes = groups.ngroup().to_numpy()
        sizes = np.bincount(codes)
        firsts = groups.head(1)

        n_groups = len(firsts)
        woe = np.zeros((n_groups, len(order)))
        se = np.zeros((n_groups, len(order)))
        fell_back = np.zeros((n_groups, len(order)), dtype=bool)
        unseen: dict[str, Counter] = {col: Counter() for col in order}
        for g, combo in enumerate(firsts.itertuples(index=False, name=None)):
            w, s, f, u = self._conditional_path_weights(tuple(combo))
            woe[g], se[g], fell_back[g] = w, s, f
            for col, value, is_unseen in zip(order, combo, u):
                if is_unseen:
                    unseen[col][value] += int(sizes[g])

        # unseen categories follow the same policy as the marginal transform
        self.unseen_counts_ = {col: dict(c) for col, c in unseen.items() if c}
        for col, counts in unseen.items():
            if counts:
                self._handle_unseen(col, counts, len(X))

        def frame(values: np.ndarray) -> pd.DataFrame:
            out = pd.DataFrame(values[codes], index=X.index, columns=order)
            return out[[c for c in X.columns if c in order]]

        return frame(woe), frame(se), frame(fell_back)

    def _conditional_output(self, X: pd.DataFrame, output: str) -> pd.DataFrame:
        """``transform(X, output=...)`` when ``conditional=True``."""
        woe, se, _ = self._transform_conditional(X)
        if output == "woe":
            return woe
        if output == "woe_norm":
            return (woe / se.replace(0, np.nan)).fillna(0)
        z = 1.959963984540054  # 95%, as in the marginal mappings
        if output == "woe_upper_ci":
            return woe + z * se
        if output == "woe_lower_ci":
            return woe - z * se
        raise ValueError(
            f"output='{output}' is not supported with conditional=True; "
            f"use 'woe', 'woe_norm', 'woe_upper_ci' or 'woe_lower_ci'"
        )

    def _conditional_score_se(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Total WOE per row and its standard error.

        Along a path with no fallback the weights telescope to the WOE of the
        full cell, so the total's SE is the SE of the last weight: the SE of
        that cell's counts. Once a step falls back, every deeper cell is a
        subset and falls back too; those marginal weights add their variances.
        """
        woe, se, fell_back = self._transform_conditional(X)
        order = self.conditional_order_
        woe_v = woe[order].to_numpy()
        se_v = np.nan_to_num(se[order].to_numpy())
        fb = fell_back[order].to_numpy()

        n_exact = np.where(fb.any(axis=1), fb.argmax(axis=1), len(order))
        rows = np.arange(len(woe_v))
        chain_se = np.where(n_exact > 0, se_v[rows, np.maximum(n_exact - 1, 0)], 0.0)
        fallback_var = (np.where(fb, se_v, 0.0) ** 2).sum(axis=1)
        return woe_v.sum(axis=1), np.sqrt(chain_se**2 + fallback_var)

    # ------------------------------------------------------------------
    # inspection
    # ------------------------------------------------------------------

    @staticmethod
    def _in_order(table: pd.DataFrame, category_order: list) -> pd.DataFrame:
        """Rows of ``table`` in ``category_order`` (bin order for binned features)."""
        rank = {c: i for i, c in enumerate(category_order)}
        return table.iloc[
            sorted(range(len(table)), key=lambda i: rank.get(table.index[i], len(rank)))
        ]

    def _category_order(self, feature: str) -> list:
        """A feature's categories as get_mapping() orders them for a marginal model."""
        return list(self._ordered_mapping(feature, self.mappings_[feature].copy()).index)

    def _conditional_mapping(self, feature: str, category_order: list) -> pd.DataFrame:
        """All conditional weights for one feature, one row per (given, category)."""
        frames = []
        for (col, path), table in self.conditional_weights_.items():
            if col != feature:
                continue
            given = ", ".join(f"{c}={v}" for c, v in zip(self.conditional_order_, path))
            frame = self._in_order(table, category_order).reset_index()
            frame.insert(0, "given", given or "-")
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return cast(pd.DataFrame, pd.concat(frames, ignore_index=True))

    def export_text(
        self, max_depth: Optional[int] = None, decimals: int = 3, bar_width: int = 30
    ) -> str:
        """The conditional weights as a text tree, like ``sklearn.tree.export_text``.

        Each level splits on the next feature in the conditioning order. A node
        shows its conditional weight ``W`` with its 95% interval (1.96 x SE),
        its size and its event rate (share of rows with target = 1). The
        intervals are also drawn as bars on one shared scale with a zero line,
        so weights can be compared by eye. Cells that used the marginal weight
        are drawn hollow and marked ``[fallback]``. A leaf's log-odds are the
        prior plus the weights along its path.

        Parameters
        ----------
        max_depth : int, optional
            Show only the first ``max_depth`` levels; deeper branches are
            summarised in one line.
        decimals : int, default=3
            Decimal places for weights and intervals.
        bar_width : int, default=30
            Width of the interval bars in characters; 0 hides them.

        Returns:
        -------
        str
        """
        if not getattr(self, "conditional", False):
            raise ValueError("export_text() requires FastWoe(conditional=True)")
        if not hasattr(self, "conditional_weights_"):
            raise ValueError("FastWoe must be fitted before export_text()")
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be at least 1")

        order = self.conditional_order_
        depth_limit = len(order) if max_depth is None else min(max_depth, len(order))
        orders = {col: self._category_order(col) for col in order}
        z = 1.959963984540054

        def label(value: Any) -> str:
            return "NaN" if pd.isna(value) else str(value)

        # collect rows first, lay them out once widths are known
        rows: list[dict[str, Any]] = []
        root = self.conditional_weights_[(order[0], ())]
        n_root = int(root["bad_count"].sum() + root["good_count"].sum())
        rows.append({"node": "root", "n": n_root, "rate": float(root["bad_count"].sum()) / n_root})

        def walk(depth: int, path: tuple, prefix: str) -> None:
            table = self._in_order(
                self.conditional_weights_[(order[depth], path)], orders[order[depth]]
            )
            for i, (value, row) in enumerate(table.iterrows()):
                last = i == len(table) - 1
                cell_n = int(row["bad_count"] + row["good_count"])
                woe, se = float(row["woe"]), float(row["woe_se"])
                rows.append(
                    {
                        "node": f"{prefix}{'└── ' if last else '├── '}"
                        f"{order[depth]} = {label(value)}",
                        "woe": woe,
                        "lo": woe - z * se,
                        "hi": woe + z * se,
                        "n": cell_n,
                        "rate": float(row["bad_count"]) / cell_n,
                        "fallback": bool(row["fallback"]),
                    }
                )
                child_prefix = prefix + ("    " if last else "│   ")
                child = (*path, value)
                if depth + 1 < depth_limit:
                    walk(depth + 1, child, child_prefix)
                elif depth + 1 < len(order):
                    leaves = sum(
                        len(t)
                        for (col, p), t in self.conditional_weights_.items()
                        if col == order[-1] and p[: depth + 1] == child
                    )
                    rows.append(
                        {
                            "node": f"{child_prefix}└── ... {len(order) - depth - 1} "
                            f"more level(s), {leaves} leaves"
                        }
                    )

        walk(0, (), "")

        target = getattr(self, "target_name_", "target")
        prior = float(np.log(cast(float, self.odds_prior_)))
        header = [
            f"Conditional WOE tree  ·  target: {target}  ·  "
            f"event rate = share of rows with {target} = 1",
            f"Conditioning order: {' → '.join(order)}  ·  prior log-odds {prior:+.{decimals}f}",
            "",
        ]

        weighted = [r for r in rows if "woe" in r]
        fmt = f"{{:+.{decimals}f}}"
        node_w = max(len(r["node"]) for r in rows)
        w_w = max(len("W"), *(len(fmt.format(r["woe"])) for r in weighted))
        ci = {id(r): f"[{fmt.format(r['lo'])}, {fmt.format(r['hi'])}]" for r in weighted}
        ci_w = max(len("95% interval"), *(len(c) for c in ci.values()))
        n_w = max(len("n"), *(len(f"{r['n']:,}") for r in rows if "n" in r))
        rate_w = len("event rate")

        lo_axis = min(0.0, *(r["lo"] for r in weighted))
        hi_axis = max(0.0, *(r["hi"] for r in weighted))
        span = hi_axis - lo_axis or 1.0

        def pos(v: float) -> int:
            return int(round((v - lo_axis) / span * (bar_width - 1)))

        def bar(r: dict) -> str:
            cells = [" "] * bar_width
            cells[pos(0.0)] = "┊"
            for k in range(pos(r["lo"]), pos(r["hi"]) + 1):
                cells[k] = "─"
            if r["lo"] <= 0.0 <= r["hi"]:  # the interval covers zero
                cells[pos(0.0)] = "┼"
            cells[pos(r["woe"])] = "○" if r["fallback"] else "●"
            return "".join(cells)

        def axis() -> str:
            cells = [" "] * bar_width
            tick = "{:+.2f}"
            marks = [(0, tick.format(lo_axis)), (pos(0.0), "0")]
            right = tick.format(hi_axis)
            marks.append((bar_width - len(right), right))
            for start, text in marks:
                if all(c == " " for c in cells[max(0, start - 1) : start + len(text) + 1]):
                    cells[start : start + len(text)] = list(text)
            return "".join(cells)

        show_bars = bar_width > 0
        head = (
            f"{'':<{node_w}}  {'W':>{w_w}}  {'95% interval':<{ci_w}}  "
            f"{'n':>{n_w}}  {'event rate':>{rate_w}}"
        )
        lines = header + [head + (f"  {axis()}" if show_bars else "")]
        for r in rows:
            if "n" not in r:  # truncation note
                lines.append(r["node"])
                continue
            if "woe" not in r:  # root
                woe_s, ci_s, bar_s = "", "", ""
            else:
                woe_s, ci_s = fmt.format(r["woe"]), ci[id(r)]
                bar_s = bar(r) if show_bars else ""
            line = (
                f"{r['node']:<{node_w}}  {woe_s:>{w_w}}  {ci_s:<{ci_w}}  "
                f"{r['n']:>{n_w},}  {r['rate']:>{rate_w}.1%}"
            )
            if show_bars and bar_s:
                line += f"  {bar_s}"
            if r.get("fallback"):
                line += "  [fallback]"
            lines.append(line.rstrip())
        return "\n".join(lines)

    def _check_chain_rule(self, X: pd.DataFrame, y: pd.Series, tol: float = 1e-9) -> dict:
        """Verify the weights add to the joint weight of evidence.

        For every path with no fallback, the conditional weights along it must
        equal the log likelihood ratio of the whole combination.
        """
        order = self.conditional_order_
        Xb = self._binned_frame(X[order])
        yv = np.asarray(y)
        bad_tot, good_tot = self._counts(np.ones(len(Xb), dtype=bool), yv)
        checked = failed = skipped = 0
        worst = 0.0
        for key, group in Xb.groupby(order, observed=True, dropna=False):
            combo: tuple[Any, ...] = key if isinstance(key, tuple) else (key,)
            mask = np.zeros(len(Xb), dtype=bool)
            mask[Xb.index.get_indexer(group.index)] = True
            bad_sel, good_sel = self._counts(mask, yv)
            if min(bad_sel, good_sel) == 0:
                skipped += 1
                continue
            joint = np.log((bad_sel / bad_tot) / (good_sel / good_tot))
            woes, _, fallbacks, _ = self._conditional_path_weights(combo)
            if any(fallbacks):
                skipped += 1
                continue
            checked += 1
            error = abs(joint - sum(woes))
            worst = max(worst, error)
            failed += error > tol
        return {
            "paths_checked": checked,
            "paths_skipped": skipped,
            "failures": failed,
            "max_abs_error": worst,
            # nothing checked is not evidence of exactness
            "exact": failed == 0 and checked > 0,
        }
