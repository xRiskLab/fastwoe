"""Piecewise Weight of Evidence encoding functionality.

Implements the "Piecewise Logistic Regression" approach described by
Raymond Anderson (Standard Bank of South Africa, 2015).  Instead of
collapsing all bins of a feature into a single WOE-transformed variable,
bins are grouped into user-defined *pieces*.  Each piece becomes its own
column so that downstream logistic regression can learn a separate
coefficient per piece, capturing non-linear relationships within a
characteristic.

Reference
---------
Anderson, R. (2015). *Piecewise Logistic Regression: an Application in
Credit Scoring*.  Presented at Credit Scoring and Control Conference XIV,
Edinburgh, 26-28 August 2015.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd


class PiecewiseWoeMixin:
    """Mixin class providing piecewise WOE functionality for FastWoe.

    After ``fit()``, call :meth:`assign_pieces` to label each bin with a
    piece index.  Then ``transform(output="piecewise")`` fans out each
    feature into multiple columns (one per piece).
    """

    # ------------------------------------------------------------------
    # Type stubs for attributes set in FastWoe
    # ------------------------------------------------------------------
    mappings_: dict[str, pd.DataFrame]
    is_fitted_: bool
    binners_: dict[str, Any]
    _apply_binning_to_column: Any
    _ensure_dataframe: Any
    is_multiclass_target: Optional[bool]

    # ------------------------------------------------------------------
    # Piece assignment
    # ------------------------------------------------------------------

    def assign_pieces(
        self,
        strategy: str = "sign",
        piece_map: Optional[dict[str, dict[str, int]]] = None,
    ) -> "PiecewiseWoeMixin":
        """Assign bins to pieces for every fitted feature.

        Parameters
        ----------
        strategy : str, default="sign"
            Built-in strategy for automatic piece assignment:

            - ``"sign"`` -- two pieces per feature: bins with WOE < 0
              (piece 0) and bins with WOE >= 0 (piece 1).  This is the
              simplest heuristic suggested by Anderson (2015).

        piece_map : dict[str, dict[str, int]], optional
            Manual override.  Outer key = feature name, inner key = bin
            category (must match the ``mappings_`` index), value = piece
            index (int >= 0).  Features present in *piece_map* ignore
            *strategy*; features absent from *piece_map* fall back to
            *strategy*.

        Returns:
        -------
        self
            Returns self for chaining.

        Raises:
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before assigning pieces")
        if self.is_multiclass_target:
            raise ValueError("Piecewise output is not supported for multiclass targets")

        if piece_map is None:
            piece_map = {}

        for feature, _mapping in self.mappings_.items():
            if feature in piece_map:
                self._assign_pieces_from_map(feature, piece_map[feature])
            else:
                self._assign_pieces_auto(feature, strategy)

        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assign_pieces_auto(self, feature: str, strategy: str) -> None:
        """Assign pieces to a single feature using a built-in strategy."""
        mapping = self.mappings_[feature]

        if strategy == "sign":
            # Piece 0 = negative WOE, piece 1 = non-negative WOE
            mapping["piece"] = np.where(mapping["woe"] < 0, 0, 1).astype(int)
        else:
            raise ValueError(f"Unknown piece strategy '{strategy}'. Available: 'sign'")

    def _assign_pieces_from_map(
        self,
        feature: str,
        cat_to_piece: dict,  # noqa: ANN001
    ) -> None:
        """Assign pieces to a single feature from a user-supplied mapping.

        Keys in *cat_to_piece* may be either the internal ``mappings_``
        index values (bin labels such as ``'(-∞, 707.5]'``) **or**
        positional integers (0, 1, 2, ...) matching the row order of
        ``get_mapping(feature)``.  When integer keys are detected and
        they don't already match the internal index, they are translated
        to the corresponding bin labels automatically.
        """
        mapping = self.mappings_[feature]

        # Detect whether keys are positional integers that need translating
        provided_keys = set(cat_to_piece.keys())
        internal_keys = set(mapping.index)

        if not provided_keys.issubset(internal_keys):
            # Try interpreting keys as positional indices
            try:
                idx_list = mapping.index.tolist()
                translated = {idx_list[int(k)]: v for k, v in cat_to_piece.items()}
                cat_to_piece = translated
                provided_keys = set(cat_to_piece.keys())
            except (IndexError, ValueError, TypeError):
                pass  # Fall through to the missing-check below

        pieces = pd.Series(cat_to_piece, name="piece")

        missing = internal_keys - provided_keys
        if missing:
            raise ValueError(f"piece_map for '{feature}' is missing categories: {sorted(missing)}")

        mapping["piece"] = pieces.reindex(mapping.index).astype(int)

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def _transform_piecewise(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with one column per (feature, piece) pair.

        For each feature, bins belonging to piece *k* carry their WOE
        value in column ``{feature}__piece_{k}``; all other rows get 0.

        Parameters
        ----------
        X : pd.DataFrame
            Input data (already converted to DataFrame by the caller).

        Returns:
        -------
        pd.DataFrame
        """
        # Apply binning to numerical features
        X_processed = X.copy()
        for col in X.columns:
            if col in self.binners_:
                X_processed[col] = self._apply_binning_to_column(X_processed, col)

        result_columns: dict[str, np.ndarray] = {}

        for col in X_processed.columns:
            mapping = self.mappings_[col]

            if "piece" not in mapping.columns:
                raise ValueError(
                    f"No piece assignment for feature '{col}'. "
                    "Call assign_pieces() before transform(output='piecewise')."
                )

            pieces = sorted(mapping["piece"].unique())
            woe_lookup = mapping["woe"].to_dict()
            piece_lookup = mapping["piece"].to_dict()

            col_values = X_processed[col].values

            for piece_id in pieces:
                col_name = f"{col}__piece_{piece_id}"
                out: np.ndarray = np.zeros(len(col_values), dtype=float)

                for cat, p in piece_lookup.items():
                    if p == piece_id:
                        mask = col_values == cat
                        out[mask] = woe_lookup[cat]

                result_columns[col_name] = out

        return pd.DataFrame(result_columns, index=X.index)
