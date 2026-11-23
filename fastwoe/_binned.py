"""Binning shared by the estimators that fit models on bins (SoftmaxWoe, JointWoe).

The ``Binner`` protocol, and a mixin that fits a binner (FastWoe by default),
turns X into bin labels, keeps each feature's levels in a fixed order and maps
labels to integer codes, with -1 for a level not seen at fit.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any, Protocol, Union, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.base import clone

from .fastwoe import FastWoe

MISSING = "Missing"
_MANY_LEVELS = 100  # more bins than this after binning suggests an unbinned column


@runtime_checkable
class Binner(Protocol):
    """What a bin-based estimator needs from a binner: fit on X and y, then label every value with a bin.

    ``transform`` returns a DataFrame or array with X's columns and a bin label
    in every cell (strings, numbers, intervals or categoricals). Any object with
    these two methods qualifies; it need not inherit from this class.
    """

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Any:
        """Learn the bins from X and the target y; the return value is ignored."""

    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        """Return the bin label of every value of X, with X's columns."""


class _BinnedMixin:
    """Binner fitting, bin labels and integer codes.

    The estimator sets ``binner``, ``binning_kwargs`` and ``unseen`` as parameters, and
    ``_binned_features`` (the fitted features in the estimator's order) during fit.
    """

    binner: Any
    binning_kwargs: Any
    unseen: str
    _binned_features: list[str]

    def _check_binning_params(self) -> None:
        """Validate the binning and unseen parameters."""
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
        fitted = self._binned_features
        full = raw.copy()
        for f in fitted:  # a custom binner always sees every fitted column
            if f not in full.columns:
                full[f] = np.nan
        full = full[fitted]
        out = self.binner_.transform(full)
        if isinstance(out, pd.DataFrame):
            out = out.copy()
            out.columns = [str(c) for c in out.columns]
            if missing := [f for f in fitted if f not in out.columns]:
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

    def _set_levels(self, binned: pd.DataFrame, frame: pd.DataFrame, features: list[str]) -> None:
        """levels_ (each feature's bins: ordered categoricals in their order, others sorted) and the code lookup."""
        self.levels_: dict[str, list[Any]] = {}
        for f in features:
            seen = set(frame[f].unique())
            if isinstance(binned[f].dtype, pd.CategoricalDtype):
                ordered = [b for b in binned[f].cat.categories if b in seen]
                self.levels_[f] = ordered + sorted(seen - set(ordered), key=self._sort_key)
            else:
                self.levels_[f] = sorted(seen, key=self._sort_key)
            if len(self.levels_[f]) > _MANY_LEVELS:
                warnings.warn(
                    f"'{f}' has {len(self.levels_[f])} distinct values after binning and is "
                    "treated as that many categories; was a continuous column left unbinned? "
                    "For a high-cardinality category, pool rare values first, e.g. "
                    f"make_pipeline(WoePreprocessor(min_count=30), {type(self).__name__}()).",
                    UserWarning,
                    stacklevel=3,
                )
        self._index = {
            f: {v: k for k, v in enumerate(levels)} for f, levels in self.levels_.items()
        }

    def _codes(self, column: pd.Series, feature: str) -> np.ndarray:
        """Position of each value in levels_[feature]; -1 for a level not seen at fit."""
        lookup = self._index[feature]
        return np.array([lookup.get(v, -1) for v in column], dtype=int)

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

    unseen_counts_: dict[str, dict[Any, int]]
