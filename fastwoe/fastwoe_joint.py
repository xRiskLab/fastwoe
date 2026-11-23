"""Joint WOE: a scorecard whose WOE tables are fitted together.

Marginal WOE (``FastWoe``) measures each feature's table on its own, so
evidence that two correlated features share is counted in both. JointWoe keeps
the same form, one weight per bin of every feature,

    log-odds = intercept + sum_j w_j[bin_j],

but fits all tables at once: each table is the feature's weight of evidence
given the other features' tables, and shared evidence is counted once. The
model is logistic regression on the one-hot bins (every bin, no reference
level) with an L2 penalty on the weights and none on the intercept. At the
penalized optimum every table sums to zero over its bins.

It is the discriminative counterpart of SoftmaxWoe, which models how each
class's features look; with many correlated features, SoftmaxWoe's chain
accumulates estimation error, while JointWoe fits one regression.

How evidence is split between strongly correlated features is set by the
penalty alone: the score is well determined, but the weights of two
near-duplicate features are not separately identified by the data. Read the
conditional information value of such features with that in mind.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from scipy.special import expit
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.linear_model import LogisticRegression

from ._binned import MISSING, Binner, _BinnedMixin

__all__ = ["JointWoe"]

_CHUNK = 4096  # rows per block when computing score variances


class JointWoe(_BinnedMixin, ClassifierMixin, TransformerMixin, BaseEstimator):
    """WOE tables fitted jointly: L2 logistic regression on the one-hot bins.

    Parameters
    ----------
    C : float, default=1.0
        Inverse L2 penalty on the bin weights, as in scikit-learn's
        LogisticRegression: the objective is C times the log loss (summed over
        rows) plus half the sum of squared weights. Smaller C shrinks thin bins
        toward 0. Choose it by cross-validation.
    max_iter : int, default=1000
        Iteration limit of the solver (lbfgs).
    tol : float, default=1e-6
        Solver tolerance. Tighter than scikit-learn's default, which can leave
        the weights of correlated features visibly unconverged.
    unseen : {"warn", "prior", "raise"}, default="warn"
        What transform() does with a bin absent at fit: give that feature's
        weight 0 and warn, do so silently, or raise.
    binner : Binner, optional
        Turns X into bins: any object with ``fit(X, y)`` and ``transform(X)``
        (see ``SoftmaxWoe``). Defaults to a FastWoe configured by ``binning_kwargs``.
    binning_kwargs : dict, optional
        Keyword arguments for the default FastWoe binner, e.g.
        ``{"binning_method": "tree", "tree_kwargs": {"max_leaf_nodes": 5}}``.

    Attributes:
    ----------
    binner_ : object
        The fitted binner (a FastWoe by default).
    levels_ : dict
        Bins of each feature, in table order; missing values are the bin "Missing".
    tables_ : dict of pandas.Series
        Each feature's weights, indexed by bin.
    intercept_ : float
        Log-odds when every feature takes weight 0.
    classes_ : ndarray
        [0, 1].
    feature_names_in_ : ndarray
        The features seen at fit, in X's column order.

    Notes:
    -----
    The target may be 0/1 or a proportion in [0, 1] (a soft target, such as a
    model's probabilities to be distilled into a scorecard); ``sample_weight``
    weights rows. Standard errors come from the inverse of the penalized
    Hessian, so they include the covariance between features' weights; the
    weights are penalized, hence biased toward 0, and the errors cover sampling
    noise, not that bias.
    """

    _cov: Optional[np.ndarray]

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-6,
        unseen: str = "warn",
        binning_kwargs: Optional[dict[str, Any]] = None,
        binner: Optional[Binner] = None,
    ):
        """Store the parameters as given; they are checked in fit (scikit-learn convention)."""
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.unseen = unseen
        self.binning_kwargs = binning_kwargs
        self.binner = binner

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _design(self, codes: dict[str, np.ndarray], n: int) -> sparse.csr_matrix:
        """Sparse one-hot of every bin of every feature, in table order; unseen bins are all zero."""
        cols, rows = [], []
        for f in self._binned_features:
            known = codes[f] >= 0
            rows.append(np.flatnonzero(known))
            cols.append(codes[f][known] + self._offsets[f])
        r, c = np.concatenate(rows), np.concatenate(cols)
        return sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, self._n_bins))

    def _prepare(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """Check the model is fitted and X has the features; return the binned X and its codes."""
        if not hasattr(self, "tables_"):
            raise ValueError("JointWoe must be fitted first")
        features = self._binned_features
        if missing := [f for f in features if f not in self._as_frame(X).columns]:
            raise ValueError(f"features seen during fit are missing from X: {missing}")
        frame = self._frame(X, features)
        codes = {f: self._codes(frame[f], f) for f in features}
        self.unseen_counts_ = {
            f: dict(Counter(frame[f][codes[f] < 0])) for f in features if (codes[f] < 0).any()
        }
        self._handle_unseen(len(frame))
        return frame, codes

    def _weight_se(self) -> np.ndarray:
        """Standard error of every bin weight as reported, centered within its table.

        Shifting a whole table by c and the intercept by -c changes no prediction,
        only the penalty, so the raw covariance carries the uncertainty of that
        arbitrary level (about sqrt(C / K) for a table of K bins) in every weight.
        The reported tables sum to zero, so each weight's variance is taken from
        the covariance projected onto centered tables, P S P with P = I - 11'/K.
        """
        cov = self._covariance()
        out = np.empty(self._n_bins)
        for f in self._binned_features:
            start, k = self._offsets[f], len(self.levels_[f])
            block = cov[start : start + k, start : start + k]
            centered = block - block.mean(axis=0) - block.mean(axis=1)[:, None] + block.mean()
            out[start : start + k] = np.sqrt(np.clip(np.diag(centered), 0, None))
        return out

    def _covariance(self) -> np.ndarray:
        """Covariance of [bin weights..., intercept]: the inverse penalized Hessian, computed once."""
        if self._cov is None:
            A = sparse.hstack([self._fit_design, np.ones((self._fit_design.shape[0], 1))]).tocsr()
            p = expit(A @ np.r_[self._coef, self.intercept_])
            H = (A.T @ A.multiply((self._fit_weight * p * (1 - p))[:, None])).toarray()
            H += np.diag(np.r_[np.full(self._n_bins, 1.0 / self.C), 0.0])
            self._cov = np.linalg.pinv(H)
        cov: np.ndarray = self._cov
        return cov

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sample_weight: Optional[np.ndarray] = None,
    ) -> JointWoe:
        """Bin X, then fit every feature's table at once.

        y holds 0/1 labels or proportions in [0, 1]. Returns self.
        """
        if not np.isfinite(self.C) or self.C <= 0:
            raise ValueError("C must be positive and finite")
        self._check_binning_params()
        raw = self._as_frame(X)
        target = np.asarray(y, dtype=float).ravel()
        if len(target) != len(raw):
            raise ValueError(f"X has {len(raw)} rows but y has {len(target)}")
        if np.isnan(target).any() or (target < 0).any() or (target > 1).any():
            raise ValueError("y must hold 0/1 labels or proportions in [0, 1]")
        weight = (
            np.ones(len(raw)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        )
        if len(weight) != len(raw) or (weight < 0).any():
            raise ValueError("sample_weight must be non-negative, one value per row")
        events = float((weight * target).sum())
        if not 0 < events < weight.sum():
            raise ValueError("JointWoe needs both events and non-events in the target")

        features = list(raw.columns)
        self._binned_features = features
        self.feature_names_in_ = np.array(features, dtype=object)
        self.n_features_in_ = len(features)
        self.classes_ = np.array([0, 1])
        self._fit_binner(raw, target)
        binned = self._bins(raw, features)
        frame = binned.astype(object).where(binned.notna(), MISSING)
        self._set_levels(binned, frame, features)
        sizes = [len(self.levels_[f]) for f in features]
        starts = np.r_[0, np.cumsum(sizes)]
        self._offsets = {f: int(s) for f, s in zip(features, starts)}
        self._n_bins = int(starts[-1])
        codes = {f: self._codes(frame[f], f) for f in features}
        design = self._design(codes, len(raw))

        lr = LogisticRegression(C=self.C, max_iter=self.max_iter, tol=self.tol)
        if np.isin(target, (0.0, 1.0)).all():
            lr.fit(design, target.astype(int), sample_weight=weight)
        else:  # soft target: every row once as an event (weight t) and once as a non-event (1 - t)
            n = len(target)
            lr.fit(
                sparse.vstack([design, design]),
                np.r_[np.ones(n), np.zeros(n)],
                sample_weight=np.r_[weight * target, weight * (1 - target)],
            )
        self._coef = lr.coef_.ravel()
        self.intercept_ = float(lr.intercept_[0])
        self.n_iter_ = int(lr.n_iter_[0])
        self.tables_ = {
            f: pd.Series(
                self._coef[self._offsets[f] : self._offsets[f] + k], index=self.levels_[f], name=f
            )
            for f, k in zip(features, sizes)
        }
        # kept for standard errors and the mapping table, which are computed on demand
        self._fit_design, self._fit_weight, self._fit_target = design, weight, target
        self._cov = None
        return self

    # ------------------------------------------------------------------
    # scores
    # ------------------------------------------------------------------

    def transform(self, X: Union[pd.DataFrame, np.ndarray], output: str = "woe") -> pd.DataFrame:
        """Each row's weight per feature ("woe"), or its standard error ("se").

        intercept_ plus the row sum of the weights is the log-odds.
        """
        if output not in ("woe", "se"):
            raise ValueError(f"output must be 'woe' or 'se', got {output!r}")
        frame, codes = self._prepare(X)
        if output == "se":
            se = self._weight_se()
        values = np.zeros(frame.shape)
        for j, f in enumerate(self._binned_features):
            known = codes[f] >= 0
            source = se if output == "se" else self._coef
            values[known, j] = source[codes[f][known] + self._offsets[f]]
        out: pd.DataFrame = pd.DataFrame(values, index=frame.index, columns=self._binned_features)
        return out

    def decision_function(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Log-odds of the event: the intercept plus the summed weights."""
        scores: np.ndarray = self.intercept_ + self.transform(X).to_numpy().sum(axis=1)
        return scores

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """[P(y=0), P(y=1)] for every row."""
        p = expit(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """1 where P(y=1) >= 0.5, else 0."""
        labels: np.ndarray = (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
        return labels

    def predict_ci(self, X: Union[pd.DataFrame, np.ndarray], alpha: float = 0.05) -> np.ndarray:
        """Confidence interval for P(y=1): columns [lower, upper].

        The variance of a row's log-odds is a' S a, where a marks the row's bins
        and the intercept and S is the coefficient covariance, so the covariance
        between features' weights is included. The interval on the log-odds is
        mapped through the sigmoid. It covers sampling noise, not the penalty's
        shrinkage.
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        frame, codes = self._prepare(X)
        n = len(frame)
        A = sparse.hstack([self._design(codes, n), np.ones((n, 1))]).tocsr()
        S = self._covariance()
        var = np.concatenate(
            [
                np.asarray((A[i : i + _CHUNK] @ S) * A[i : i + _CHUNK].toarray()).sum(axis=1)
                for i in range(0, n, _CHUNK)
            ]
        )
        score = A @ np.r_[self._coef, self.intercept_]
        half = norm.ppf(1 - alpha / 2) * np.sqrt(np.clip(var, 0, None))
        bounds: np.ndarray = expit(np.column_stack([score - half, score + half]))
        return bounds

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------

    def iv(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> pd.Series:
        """Conditional information value of each feature against 0/1 labels.

        The mean weight over events minus the mean over non-events. With
        marginal WOE this is the usual information value; with the joint
        tables it is each feature's share of the evidence given the others.
        """
        labels = np.asarray(y).ravel()
        if not np.isin(labels, (0, 1)).all() or len(np.unique(labels)) < 2:
            raise ValueError("iv needs 0/1 labels with both classes present")
        weights = self.transform(X).to_numpy()
        values = weights[labels == 1].mean(axis=0) - weights[labels == 0].mean(axis=0)
        iv: pd.Series = pd.Series(values, index=self._binned_features, name="iv")
        return iv

    def get_mapping(self, feature: str) -> pd.DataFrame:
        """One feature's table: bin, training count and event rate, marginal WOE, joint weight and its SE.

        ``marginal_woe`` is the bin's WOE on its own, log of (events share /
        non-events share), from the (weighted, possibly soft) training target;
        ``woe`` is the jointly fitted weight, given the other features.
        """
        if not hasattr(self, "tables_"):
            raise ValueError("JointWoe must be fitted first")
        if feature not in self.tables_:
            raise ValueError(f"unknown feature {feature!r}")
        start, k = self._offsets[feature], len(self.levels_[feature])
        block = self._fit_design[:, start : start + k]
        w, t = self._fit_weight, self._fit_target
        count = np.asarray(block.T @ w).ravel()
        events = np.asarray(block.T @ (w * t)).ravel()
        nonevents = count - events
        with np.errstate(divide="ignore", invalid="ignore"):
            marginal = np.log((events / events.sum()) / (nonevents / nonevents.sum()))
            rate = events / count
        se = self._weight_se()[start : start + k]
        table: pd.DataFrame = pd.DataFrame(
            {
                "category": self.levels_[feature],
                "count": count,
                "event_rate": rate,
                "marginal_woe": marginal,
                "woe": self.tables_[feature].to_numpy(),
                "woe_se": se,
            }
        )
        return table
