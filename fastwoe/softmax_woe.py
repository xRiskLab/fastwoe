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

import copy
import warnings
from collections import Counter
from typing import Any, Optional, Protocol, Union, runtime_checkable

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.linear_model import LogisticRegression

from .fastwoe import FastWoe

__all__ = ["Binner", "SoftmaxWoe"]

MISSING = "Missing"
_MANY_LEVELS = 50  # more bins than this after binning suggests an unbinned column
_FLOOR = 1e-6  # probability of a bin a class never showed at a node, before renormalizing


@runtime_checkable
class Binner(Protocol):
    """What SoftmaxWoe needs from a binner: fit on X and y, then label every value with a bin.

    ``transform`` returns a DataFrame or array with X's columns and a bin label
    in every cell (strings, numbers, intervals or categoricals). Any object with
    these two methods qualifies; it need not inherit from this class.
    """

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Any:
        """Learn the bins from X and the binary target y; the return value is ignored."""

    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        """Return the bin label of every value of X, with X's columns."""


class SoftmaxWoe(ClassifierMixin, TransformerMixin, BaseEstimator):
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
    binner : Binner, optional
        Turns X into bins. Any object with ``fit(X, y)`` and ``transform(X)``
        (the ``Binner`` protocol),
        where ``transform`` returns a DataFrame or array with X's columns and a
        bin label in every cell: strings, numbers, intervals or categoricals.
        Ordered categoricals keep their order; other labels are sorted (numbers
        and intervals numerically). Missing values become the level "Missing",
        and a column the binner leaves unchanged is treated as categories. The
        binner is cloned before fitting, and ``fit`` may return None. It always
        receives every column seen at fit; where ``node_proba`` is given only
        some, the others are passed as NaN. Defaults to a FastWoe configured by
        ``binning_kwargs``.
    binning_kwargs : dict, optional
        Keyword arguments for the default FastWoe binner, e.g.
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
    binner_ : object
        The fitted binner. With the default it is a FastWoe: its
        ``get_binning_summary()`` shows the bins, and its ``transform()`` gives
        the marginal WOE for comparison.
    levels_ : dict
        Bins of each feature, in the order the chain uses them.
    class_counts_ : dict
        Training rows per class, {1: events, 0: non-events}.
    classes_ : ndarray
        The class labels, [0, 1].

    Notes on scikit-learn: SoftmaxWoe is a scikit-learn estimator, so it works
    with ``clone``, ``GridSearchCV`` (for example over ``C``), ``cross_val_score``
    and ``Pipeline``. Parameters are checked when ``fit`` is called.

    Notes:
    -----
    By default numerical features are binned exactly as FastWoe bins them; every
    other feature is treated as categorical. Missing values form their own level,
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
        binner: Optional[Binner] = None,
    ):
        """Store the parameters as given; they are checked in fit (scikit-learn convention)."""
        self.order = order
        self.C = C
        self.root_pseudo_count = root_pseudo_count
        self.max_iter = max_iter
        self.unseen = unseen
        self.binning_kwargs = binning_kwargs
        self.binner = binner

    def _check_params(self) -> None:
        """Validate the constructor parameters."""
        if self.C <= 0:
            raise ValueError("C must be positive")
        if self.root_pseudo_count < 0:
            raise ValueError("root_pseudo_count must be non-negative")
        if self.unseen not in ("warn", "prior", "raise"):
            raise ValueError(f"unseen must be 'warn', 'prior' or 'raise', got {self.unseen!r}")
        kwargs = self.binning_kwargs or {}
        if conditional := [k for k in kwargs if k.startswith("conditional")]:
            raise ValueError(f"binning_kwargs configures binning only; remove {conditional}")
        if self.binner is not None:
            if kwargs:
                raise ValueError("pass either binner or binning_kwargs, not both")
            if not isinstance(self.binner, Binner):
                raise TypeError("binner must have fit(X, y) and transform(X) methods")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_frame(X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """X as a DataFrame with string column names."""
        frame: pd.DataFrame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        frame.columns = [str(c) for c in frame.columns]
        return frame

    def _fit_binner(self, raw: pd.DataFrame, target: np.ndarray) -> None:
        """Fit the default FastWoe, or a clone of the given binner."""
        if self.binner is None:
            self.binner_ = FastWoe(**(self.binning_kwargs or {})).fit(raw, target)
            return
        try:
            binner = clone(self.binner)
        except TypeError:  # not a scikit-learn estimator
            binner = copy.deepcopy(self.binner)
        binner.fit(raw, target)
        self.binner_ = binner

    def _bins(self, raw: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        """Bin labels of the given features of raw, checked against the binner contract."""
        if self.binner is None:
            bins: pd.DataFrame = self.binner_.transform_bins(raw[features])
            return bins
        full = raw.copy()
        for f in self.order_:  # a custom binner always sees every fitted column
            if f not in full.columns:
                full[f] = np.nan
        full = full[self.order_]
        out = self.binner_.transform(full)
        if isinstance(out, pd.DataFrame):
            out = out.copy()
            out.columns = [str(c) for c in out.columns]
            if missing := [f for f in self.order_ if f not in out.columns]:
                raise ValueError(f"binner.transform() dropped columns {missing}")
            if len(out) != len(full):
                raise ValueError("binner.transform() must return one row per row of X")
            out.index = full.index
        else:
            values = np.asarray(out)
            if values.shape != full.shape:
                raise ValueError(
                    f"binner.transform() returned shape {values.shape}; expected {full.shape}"
                )
            out = pd.DataFrame(values, index=full.index, columns=full.columns)
        selected: pd.DataFrame = out[features]
        return selected

    def _frame(self, X: Union[pd.DataFrame, np.ndarray], features: list[str]) -> pd.DataFrame:
        """The given features of X, binned by binner_, as objects with missing values filled."""
        binned = self._bins(self._as_frame(X), features)
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
        self._check_params()
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
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = raw.shape[1]
        self._fit_binner(raw[order], target)
        binned = self._bins(raw[order], order)
        frame = binned.astype(object).where(binned.notna(), MISSING)
        self.levels_ = {}
        for f in order:
            seen = set(frame[f].unique())
            if isinstance(binned[f].dtype, pd.CategoricalDtype):
                ordered = [b for b in binned[f].cat.categories if b in seen]
                self.levels_[f] = ordered + sorted(seen - set(ordered), key=self._sort_key)
            else:
                self.levels_[f] = sorted(seen, key=self._sort_key)
            if len(self.levels_[f]) > _MANY_LEVELS:
                warnings.warn(
                    f"'{f}' has {len(self.levels_[f])} distinct values after binning and is "
                    "treated as that many categories; was a continuous column left unbinned?",
                    UserWarning,
                    stacklevel=2,
                )
        self._index = {
            f: {v: k for k, v in enumerate(levels)} for f, levels in self.levels_.items()
        }
        n_bad, n_good = int(target.sum()), int(len(target) - target.sum())
        self.prior_log_odds_ = float(np.log(n_bad / n_good))
        self.class_counts_ = {1: n_bad, 0: n_good}
        codes = {f: self._codes(frame[f], f) for f in order}
        # kept to compute standard errors on demand; covariances are cached on first use
        self._fit_codes, self._fit_target = codes, target
        self._node_C: dict[tuple[str, int], float] = {}
        self._node_cov: dict[tuple[str, int], np.ndarray] = {}
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
                node = LogisticRegression(C=node_C, max_iter=self.max_iter).fit(inputs, node_codes)
                self.nodes_[(f, cls)] = node
                self._node_C[(f, cls)] = node_C
        return self

    # ------------------------------------------------------------------
    # standard errors
    # ------------------------------------------------------------------

    def _distinct(
        self, columns: dict[str, np.ndarray], features: list[str]
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Distinct rows of the given bin codes, with each row's index into them and their counts.

        Codes (-1 for unseen) are packed into one integer per row so a 1-D unique
        does the work; if that integer could overflow, rows are compared directly.
        """
        key = np.column_stack([columns[f] + 1 for f in features])
        sizes = [len(self.levels_[f]) + 1 for f in features]
        if float(np.prod(sizes, dtype=float)) < 2**62:
            packed = np.zeros(len(key), dtype=np.int64)
            for j, size in enumerate(sizes):
                packed = packed * size + key[:, j]
            _, first, back, counts = np.unique(
                packed, return_index=True, return_inverse=True, return_counts=True
            )
            rows = key[first]
        else:
            rows, back, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        distinct = {f: rows[:, j] - 1 for j, f in enumerate(features)}
        return distinct, back.ravel(), counts

    @staticmethod
    def _with_intercept(inputs: np.ndarray) -> np.ndarray:
        return np.hstack([inputs, np.ones((len(inputs), 1))])

    @staticmethod
    def _coef_rows(node: LogisticRegression) -> np.ndarray:
        """Coefficients and intercept of each softmax row: rows x (inputs + 1)."""
        rows: np.ndarray = np.hstack([node.coef_, node.intercept_[:, None]])
        return rows

    def _coef_covariance(
        self, node: LogisticRegression, inputs: np.ndarray, counts: np.ndarray, node_C: float
    ) -> np.ndarray:
        """Covariance of a node's coefficients: the inverse penalized Hessian.

        The Hessian is that of the node's objective in units of log-likelihood,
        with the L2 penalty 1/C on every coefficient and none on the intercept.
        The softmax is over-parametrized (a constant added to every row changes
        nothing), so the pseudo-inverse is used; the gradients it is applied to
        are orthogonal to those directions.

        The inputs are one-hot bins, so rows repeat: the Hessian is a sum over
        rows, computed once per distinct row (``inputs``) weighted by how many
        training rows share it (``counts``). The
        multinomial Hessian sum_i (diag(p_i) - p_i p_i') kron a_i a_i' is built
        as one matrix product of the stacked p kron a, plus the diagonal blocks.
        """
        A = self._with_intercept(inputs)
        d = A.shape[1]
        rows = self._coef_rows(node)
        penalty = np.r_[np.full(d - 1, 1.0 / node_C), 0.0]
        if len(rows) == 1:  # binary node: one logit
            q = 1.0 / (1.0 + np.exp(-A @ rows[0]))
            H = (A * (counts * q * (1 - q))[:, None]).T @ A + np.diag(penalty)
        else:
            Z = A @ rows.T
            P = np.exp(Z - Z.max(axis=1, keepdims=True))
            P /= P.sum(axis=1, keepdims=True)
            K = len(rows)
            stacked = (P[:, :, None] * A[:, None, :]).reshape(len(A), K * d)
            stacked *= np.sqrt(counts)[:, None]
            H = -(stacked.T @ stacked)  # - sum_i p_i p_i' kron a_i a_i'
            for k in range(K):  # + sum_i diag(p_i) kron a_i a_i'
                block = slice(k * d, (k + 1) * d)
                H[block, block] += (A.T * (counts * P[:, k])) @ A
            H += np.diag(np.tile(penalty, K))
        cov: np.ndarray = np.linalg.pinv(H)
        return cov

    def _covariance(self, i: int, cls: int) -> np.ndarray:
        """A fitted node's coefficient covariance, computed on first use and cached."""
        key = (self.order_[i], cls)
        if key not in self._node_cov:
            rows = self._fit_target == cls
            earlier = self.order_[:i]
            distinct, _, counts = self._distinct(
                {g: self._fit_codes[g][rows] for g in earlier}, earlier
            )
            inputs = self._onehot(distinct, earlier, len(counts))
            self._node_cov[key] = self._coef_covariance(
                self.nodes_[key], inputs, counts, self._node_C[key]
            )
        return self._node_cov[key]

    def _log_prob_variance(
        self, i: int, cls: int, codes: dict[str, np.ndarray], n: int
    ) -> np.ndarray:
        """Delta-method variance of log P(row's bin of feature i | earlier bins, class).

        Counted shares use the multinomial variance (1 - p) / (n p); fitted
        nodes use the gradient of log p against the coefficient covariance.
        A bin unseen at fit gets 0 (its weight is fixed at 0).
        """
        feature = self.order_[i]
        node = self.nodes_[(feature, cls)]
        n_class = self.class_counts_[cls]
        if isinstance(node, np.ndarray):
            k = codes[feature]
            known = k >= 0
            p = node[np.where(known, k, 0)]
            return np.where(known, (1 - p) / (n_class * p), 0.0)
        # rows with the same earlier bins and own bin share a variance: compute
        # it once per distinct combination and map it back
        earlier = self.order_[:i]
        dcodes, back, _ = self._distinct(codes, [*earlier, feature])
        dk = dcodes[feature]
        u = len(dk)
        x = self._with_intercept(self._onehot(dcodes, earlier, u))
        rows = self._coef_rows(node)
        cov = self._covariance(i, cls)
        # position of each bin among the classes the node was fitted on
        position = {c: j for j, c in enumerate(node.classes_)}
        pos = np.array([position.get(c, -1) for c in dk])
        if len(rows) == 1:
            q = 1.0 / (1.0 + np.exp(-x @ rows[0]))
            grad = np.where(pos == 1, 1 - q, -q)[:, None] * x
        else:
            Z = x @ rows.T
            P = np.exp(Z - Z.max(axis=1, keepdims=True))
            P /= P.sum(axis=1, keepdims=True)
            E = -P
            hit = pos >= 0
            E[np.flatnonzero(hit), pos[hit]] += 1.0
            grad = (E[:, :, None] * x[:, None, :]).reshape(u, -1)
        var = ((grad @ cov) * grad).sum(axis=1)
        # a bin this class never showed here sits at the probability floor
        floor_var = (1 - _FLOOR) / (n_class * _FLOOR)
        per_distinct = np.where(dk < 0, 0.0, np.where(pos < 0, floor_var, var))
        out: np.ndarray = per_distinct[back]
        return out

    def _weight_se(self, codes: dict[str, np.ndarray], n: int) -> np.ndarray:
        """Standard error of each feature's weight: rows x features in order_."""
        var = [
            self._log_prob_variance(i, 1, codes, n) + self._log_prob_variance(i, 0, codes, n)
            for i in range(len(self.order_))
        ]
        se: np.ndarray = np.sqrt(np.column_stack(var))
        return se

    def transform(self, X: Union[pd.DataFrame, np.ndarray], output: str = "woe") -> pd.DataFrame:
        """Per-feature weights, or their standard errors, one column per feature of X.

        Parameters
        ----------
        X : DataFrame or ndarray
            Features seen during fit.
        output : {"woe", "se"}, default="woe"
            "woe": the conditional weights; prior_log_odds_ plus the row sum is
            the log-odds of the positive class. "se": the delta-method standard
            error of each weight, from the event and non-event nodes, which are
            fitted on separate rows.
        """
        if output not in ("woe", "se"):
            raise ValueError(f"output must be 'woe' or 'se', got {output!r}")
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
        if output == "se":
            weights = self._weight_se(codes, n)
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

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Class labels: 1 where P(y=1) >= 0.5, else 0."""
        labels: np.ndarray = (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
        return labels

    def predict_ci(self, X: Union[pd.DataFrame, np.ndarray], alpha: float = 0.05) -> np.ndarray:
        """Confidence interval for P(y=1): columns [lower, upper].

        Within each class the log-likelihood is a sum of node terms with
        separate coefficients, so the node estimates are asymptotically
        independent: the variance of the log-odds is the prior's (1/n1 + 1/n0)
        plus the sum of the weight variances. The interval on the log-odds,
        score +/- z * SE, is mapped through the sigmoid.

        The weights are penalized, so they are biased toward marginal WOE; the
        interval covers sampling noise, not that bias. With a strong penalty
        (small C) read it as an interval under the model's assumptions.
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        frame, codes = self._prepare(X, self.order_)
        n = len(frame)
        score = self.prior_log_odds_ + self.transform(X).to_numpy().sum(axis=1)
        prior_var = 1 / self.class_counts_[1] + 1 / self.class_counts_[0]
        se = np.sqrt(prior_var + (self._weight_se(codes, n) ** 2).sum(axis=1))
        z = norm.ppf(1 - alpha / 2)
        bounds: np.ndarray = 1.0 / (
            1.0 + np.exp(-np.column_stack([score - z * se, score + z * se]))
        )
        return bounds

    def node_proba(self, X: Union[pd.DataFrame, np.ndarray], feature: str) -> pd.DataFrame:
        """Probability of each row's bin of ``feature`` at its node, in each class.

        Columns ``p_event`` = P(bin | earlier bins, y=1) and ``p_nonevent`` =
        P(bin | earlier bins, y=0); the log of their ratio is the feature's
        weight in transform(). X needs ``feature`` and every feature before it
        in ``order_``. A bin not seen at fit gives NaN.
        """
        if not hasattr(self, "nodes_"):
            raise ValueError("SoftmaxWoe must be fitted first")
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
