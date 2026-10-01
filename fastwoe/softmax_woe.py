"""Softmax WOE: conditional weight of evidence from a hierarchical softmax.

SoftmaxWoe is a generative classifier: per class, an autoregressive chain of
penalized multinomial logistic regressions, fitted by maximum likelihood of the
features given the class; its per-feature weights are Good's conditional
weights of evidence under that model and sum exactly to the posterior log-odds.

Conditional WOE (``FastWoe(conditional=True)``) measures each feature's weight
inside the cell picked out by the features before it, by counting. With more
than a few features the cells run thin and fall back to marginal weights.

Softmax WOE replaces the counts with a small model per feature. Within each
class, the bin of feature i is predicted from the bins of the earlier features
by a multinomial logistic regression (the "node"); the first node is the bin
shares. Per class this is the hierarchical softmax of Goodfellow et al.
(2016, sec. 12.4.3.2), with applicant profiles in place of words. The weight at
node i is

    W(H : bin_i | bin_<i) = log P(bin_i | bin_<i, H) - log P(bin_i | bin_<i, not-H)

Each node is a proper distribution over its bins, so in each class the product
along a path is a proper distribution over all profiles, and the weights add
up exactly to the log likelihood ratio of the profile:

    log-odds = prior + sum_i W_i

The node penalty ``C`` moves along a ladder of assumptions: as C goes to 0 the
nodes shrink to their bin shares and the result is marginal WOE (naive Bayes);
as C grows the nodes fit main effects of the earlier bins freely. Unlike the
counting chain, the score depends on the order of the features, because each
order defines a different smoothed distribution.
"""

from __future__ import annotations

import warnings
from collections import Counter
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .fastwoe import FastWoe

__all__ = ["SoftmaxWoe"]

MISSING = "Missing"
_FLOOR = 1e-6  # probability of a bin a class never showed at a node, before renormalizing


class SoftmaxWoe:
    """Conditional WOE by hierarchical softmax (binary targets, categorical features).

    SoftmaxWoe is a generative classifier: per class, an autoregressive chain of
    penalized multinomial logistic regressions, fitted by maximum likelihood of the
    features given the class; its per-feature weights are Good's conditional
    weights of evidence under that model and sum exactly to the posterior log-odds.

    Parameters
    ----------
    order : list of str, optional
        Order in which features are conditioned on. Defaults to the column order
        of X at fit. Unlike counted conditional WOE, the order changes the score
        as well as how evidence is attributed across features.
    C : float, default=1.0
        Inverse L2 penalty of every node model (smaller means more shrinkage
        toward marginal WOE), applied to one coefficient row per bin, so nodes
        with two bins are shrunk like the rest. Choose it by cross-validation.
    root_pseudo_count : float, default=0.5
        Pseudo-count added to each bin when shares are counted: at the first
        node, and at any node where a class shows a single bin.
    max_iter : int, default=5000
        Iteration limit of each node's logistic regression.
    unseen : {"warn", "prior", "raise"}, default="warn"
        What transform() does with a category absent at fit: give that
        feature's weight 0 and warn, do so silently, or raise.
    binning_kwargs : dict, optional
        Keyword arguments for the FastWoe that bins numerical features, e.g.
        ``{"binning_method": "kbins", "numerical_threshold": 10}`` or
        ``{"monotonic_cst": {"income": -1}}``. Defaults to FastWoe's own
        defaults (decision-tree bins for numeric columns with at least 20
        distinct values; numeric columns with fewer are treated as categories).
        A monotonic constraint shapes the bins, so it holds for the marginal
        weights and for the first feature in ``order``; the node models are not
        constrained, so a later feature's conditional weights need not be
        monotone.

    Attributes:
    ----------
    binner_ : FastWoe
        The fitted FastWoe that supplies the bins. Its ``get_binning_summary()``
        shows them, and its ``transform()`` gives the marginal WOE for comparison.
    levels_ : dict
        Bins of each feature, in the order the chain uses them.

    Notes:
    -----
    Numerical features are binned exactly as FastWoe bins them; every other
    feature is treated as categorical. Missing values form their own level,
    "Missing".
    """

    def __init__(
        self,
        order: Optional[list[str]] = None,
        C: float = 1.0,
        root_pseudo_count: float = 0.5,
        max_iter: int = 5000,
        unseen: str = "warn",
        binning_kwargs: Optional[dict[str, Any]] = None,
    ):
        """Configure the model; parameters are described in the class docstring."""
        if C <= 0:
            raise ValueError("C must be positive")
        if root_pseudo_count < 0:
            raise ValueError("root_pseudo_count must be non-negative")
        if unseen not in ("warn", "prior", "raise"):
            raise ValueError(f"unseen must be 'warn', 'prior' or 'raise', got {unseen!r}")
        self.order = None if order is None else list(order)
        self.C = C
        self.root_pseudo_count = root_pseudo_count
        self.max_iter = max_iter
        self.unseen = unseen
        self.binning_kwargs = dict(binning_kwargs or {})
        if conditional := [k for k in self.binning_kwargs if k.startswith("conditional")]:
            raise ValueError(f"binning_kwargs configures binning only; remove {conditional}")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_frame(X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """X as a DataFrame with string column names."""
        frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        frame.columns = [str(c) for c in frame.columns]
        return frame

    def _frame(self, X: Union[pd.DataFrame, np.ndarray], features: list[str]) -> pd.DataFrame:
        """The given features of X, binned by binner_, as objects with missing values filled."""
        binned = self.binner_.transform_bins(self._as_frame(X)[features])
        filled: pd.DataFrame = binned.astype(object).where(binned.notna(), MISSING)
        return filled

    @staticmethod
    def _sort_key(value: Any) -> tuple[int, Any]:
        """Numbers in numeric order, then strings."""
        return (1, value) if isinstance(value, str) else (0, value)

    def _codes(self, column: pd.Series, feature: str) -> np.ndarray:
        """Position of each value in levels_[feature]; -1 for a level not seen at fit."""
        lookup = self._index[feature]
        return np.array([lookup.get(v, -1) for v in column], dtype=int)

    def _onehot(self, codes: dict[str, np.ndarray], features: list[str], n: int) -> np.ndarray:
        """Every bin of every earlier feature, no reference level; unseen bins are all zero."""
        blocks = []
        for f in features:
            block = np.zeros((n, len(self.levels_[f])))
            known = codes[f] >= 0
            block[np.flatnonzero(known), codes[f][known]] = 1.0
            blocks.append(block)
        return np.hstack(blocks) if blocks else np.zeros((n, 0))

    def _shares(self, codes: np.ndarray, n_levels: int) -> np.ndarray:
        """Return the share of each level, with pseudo-count smoothing."""
        counts = np.bincount(codes, minlength=n_levels).astype(float)
        counts += self.root_pseudo_count
        shares: np.ndarray = counts / counts.sum()
        return shares

    def _node_probs(self, i: int, cls: int, codes: dict[str, np.ndarray], n: int) -> np.ndarray:
        """P(bin of feature i | earlier bins, class): rows x bins, each row summing to 1."""
        feature = self.order_[i]
        node = self.nodes_[(feature, cls)]
        if isinstance(node, np.ndarray):
            return np.tile(node, (n, 1))
        probs = np.full((n, len(self.levels_[feature])), _FLOOR)
        probs[:, node.classes_] = node.predict_proba(self._onehot(codes, self.order_[:i], n))
        normalized: np.ndarray = probs / probs.sum(axis=1, keepdims=True)
        return normalized

    def _prepare(
        self, X: Union[pd.DataFrame, np.ndarray], features: list[str]
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """Check the model is fitted and X has the features; return X and its bin codes."""
        if not hasattr(self, "nodes_"):
            raise ValueError("SoftmaxWoe must be fitted first")
        if missing := [f for f in features if f not in self._as_frame(X).columns]:
            raise ValueError(f"features seen during fit are missing from X: {missing}")
        frame = self._frame(X, features)
        return frame, {f: self._codes(frame[f], f) for f in features}

    # ------------------------------------------------------------------
    # fit / transform
    # ------------------------------------------------------------------

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> SoftmaxWoe:
        """Fit one node model per feature and class.

        Returns self, as the scikit-learn estimator API expects.
        """
        raw = self._as_frame(X)
        target = np.asarray(y)
        if not set(np.unique(target)) <= {0, 1} or len(np.unique(target)) != 2:
            raise ValueError("SoftmaxWoe needs a binary (0/1) target with both classes present")
        order = list(raw.columns) if self.order is None else [str(c) for c in self.order]
        if len(set(order)) != len(order) or set(order) != set(raw.columns):
            raise ValueError(
                "order must list every column of X exactly once; got "
                f"{order} for columns {list(raw.columns)}"
            )

        self.order_ = order
        self.binner_ = FastWoe(**self.binning_kwargs).fit(raw[order], target)
        binned = self.binner_.transform_bins(raw[order])
        frame = self._frame(raw, order)
        self.levels_ = {}
        for f in order:
            seen = set(frame[f].unique())
            if isinstance(binned[f].dtype, pd.CategoricalDtype):
                self.levels_[f] = [b for b in binned[f].cat.categories if b in seen]
            else:
                self.levels_[f] = sorted(seen, key=self._sort_key)
        self._index = {
            f: {v: k for k, v in enumerate(levels)} for f, levels in self.levels_.items()
        }
        n_bad, n_good = int(target.sum()), int(len(target) - target.sum())
        self.prior_log_odds_ = float(np.log(n_bad / n_good))

        codes = {f: self._codes(frame[f], f) for f in order}
        self.nodes_: dict[tuple[str, int], Any] = {}
        for i, f in enumerate(order):
            n_levels = len(self.levels_[f])
            for cls in (1, 0):
                rows = target == cls
                node_codes = codes[f][rows]
                if i == 0 or len(np.unique(node_codes)) < 2:
                    self.nodes_[(f, cls)] = self._shares(node_codes, n_levels)
                    continue
                inputs = self._onehot(
                    {g: codes[g][rows] for g in order[:i]}, order[:i], int(rows.sum())
                )
                # With two bins scikit-learn fits one logit, bin 1 against bin 0,
                # instead of a coefficient row per bin. Under the same C that is
                # penalized twice as hard as the symmetric softmax; 2C makes the
                # two fits identical, so every node is shrunk alike.
                two_bins = len(np.unique(node_codes)) == 2
                node_C = 2 * self.C if two_bins else self.C
                self.nodes_[(f, cls)] = LogisticRegression(C=node_C, max_iter=self.max_iter).fit(
                    inputs, node_codes
                )
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Per-feature weights, one column per feature in X's column order.

        prior_log_odds_ plus the row sum is the log-odds of the positive class.
        """
        frame, codes = self._prepare(X, self.order_)
        n = len(frame)
        rows = np.arange(n)
        weights = np.zeros((n, len(self.order_)))
        self.unseen_counts_: dict[str, dict[Any, int]] = {}
        for i, f in enumerate(self.order_):
            known = codes[f] >= 0
            if not known.all():
                self.unseen_counts_[f] = dict(Counter(frame[f][~known]))
            bins = np.where(known, codes[f], 0)
            p_bad = self._node_probs(i, 1, codes, n)[rows, bins]
            p_good = self._node_probs(i, 0, codes, n)[rows, bins]
            weights[:, i] = np.where(known, np.log(p_bad) - np.log(p_good), 0.0)
        self._handle_unseen(n)
        out = pd.DataFrame(weights, index=frame.index, columns=self.order_)
        ordered: pd.DataFrame = out[[c for c in self._as_frame(X).columns if c in self.order_]]
        return ordered

    def _handle_unseen(self, n_rows: int) -> None:
        """Handle unseen categories according to the unseen policy."""
        if not self.unseen_counts_ or self.unseen == "prior":
            return
        detail = "; ".join(
            f"'{f}': " + ", ".join(f"{v!r} ({k})" for v, k in counts.items())
            for f, counts in self.unseen_counts_.items()
        )
        msg = (
            f"categories not seen during fit, given weight 0 (the prior) in {n_rows} rows: "
            f"{detail}. Set unseen='prior' to silence this, or unseen='raise' to fail."
        )
        if self.unseen == "raise":
            raise ValueError(msg)
        warnings.warn(msg, UserWarning, stacklevel=3)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Probabilities [P(0), P(1)] from the prior plus the summed weights."""
        log_odds = self.prior_log_odds_ + self.transform(X).to_numpy().sum(axis=1)
        p = 1.0 / (1.0 + np.exp(-log_odds))
        return np.column_stack([1.0 - p, p])

    def node_proba(self, X: Union[pd.DataFrame, np.ndarray], feature: str) -> pd.DataFrame:
        """Probability of each row's bin of ``feature`` at its node, in each class.

        Columns ``p_event`` = P(bin | earlier bins, y=1) and ``p_nonevent`` =
        P(bin | earlier bins, y=0); the log of their ratio is the feature's
        weight in transform(). X needs ``feature`` and every feature before it
        in ``order_``. A bin not seen at fit gives NaN.
        """
        if not hasattr(self, "nodes_"):
            raise ValueError("SoftmaxWoe must be fitted first")
        feature = feature
        if feature not in self.order_:
            raise ValueError(f"unknown feature {feature!r}; expected one of {self.order_}")
        i = self.order_.index(feature)
        frame, codes = self._prepare(X, self.order_[: i + 1])
        n = len(frame)
        known = codes[feature] >= 0
        bins = np.where(known, codes[feature], 0)
        rows = np.arange(n)
        out = {
            name: np.where(known, self._node_probs(i, cls, codes, n)[rows, bins], np.nan)
            for name, cls in (("p_event", 1), ("p_nonevent", 0))
        }
        probs: pd.DataFrame = pd.DataFrame(out, index=frame.index)
        return probs
