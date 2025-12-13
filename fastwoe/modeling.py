"""
Feature selection using Marginal Information Value (MIV) with rank correlation.

This module implements MIV-based feature selection using Somers' D (rank correlation)
instead of traditional WOE-based Information Value. It leverages fastwoe's predict_proba
to get model scores and uses the fast Somers' D implementation for efficient computation.

Based on: Revolut MIV methodology (see docs/revolut_miv.md)
"""

from __future__ import annotations

import contextlib
import itertools
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from .fastwoe import FastWoe
from .metrics import gini_pairwise, somersd_pairwise


def _create_woe_model(template: Optional[FastWoe] = None) -> FastWoe:
    """
    Create a new FastWoe instance, optionally using a template for configuration.

    Parameters
    ----------
    template : FastWoe, optional
        Template FastWoe instance to copy configuration from. If None, creates
        a new instance with default parameters.

    Returns:
    -------
    FastWoe
        New FastWoe instance with configuration from template (if provided).
    """
    if template is None:
        return FastWoe()

    # Create new instance with same configuration as template
    return FastWoe(
        encoder_kwargs=template.encoder_kwargs.copy() if template.encoder_kwargs else None,
        random_state=template.random_state,
        binner_kwargs=getattr(template, "binner_kwargs", None),
        warn_on_numerical=getattr(template, "warn_on_numerical", False),
        numerical_threshold=getattr(template, "numerical_threshold", 20),
        binning_method=getattr(template, "binning_method", "tree"),
        tree_estimator=getattr(template, "tree_estimator", None),
        tree_kwargs=getattr(template, "tree_kwargs", None),
        faiss_kwargs=getattr(template, "faiss_kwargs", None),
        monotonic_cst=getattr(template, "monotonic_cst", None),
    )


def marginal_somersd_selection(
    X: pd.DataFrame,
    y: Union[np.ndarray, pd.Series],
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[Union[np.ndarray, pd.Series]] = None,
    min_miv: float = 0.02,
    max_features: Optional[int] = None,
    correlation_threshold: float = 0.5,
    ties: str = "y",
    random_state: Optional[int] = None,
    woe_model: Optional[FastWoe] = None,
) -> dict:
    """
    Feature selection using Marginal Information Value (MIV) with Somers' D.

    This implements a greedy forward-selection algorithm similar to MIV, but uses
    rank correlation (Somers' D) instead of WOE-based Information Value. The method:

    1. Starts with the feature with highest univariate Somers' D
    2. At each step, fits a FastWoe model with current features
    3. Gets model scores using predict_proba
    4. For each candidate feature, calculates marginal Somers' D (residual correlation)
    5. Adds the feature with highest marginal Somers' D
    6. Continues until MIV falls below threshold or max_features reached

    Parameters
    ----------
    X : pd.DataFrame
        Training features (categorical or mixed types)
    y : np.ndarray or pd.Series
        Binary target variable (0/1)
    X_test : pd.DataFrame, optional
        Test features for performance monitoring. If None, uses X.
    y_test : np.ndarray or pd.Series, optional
        Test target for performance monitoring. If None, uses y.
    min_miv : float, default=0.02
        Minimum marginal Somers' D threshold to continue selection
    max_features : int, optional
        Maximum number of features to select. If None, continues until min_miv.
    correlation_threshold : float, default=0.5
        Maximum pairwise correlation allowed between selected features
    ties : str, default="y"
        How to handle ties in Somers' D calculation:
        - "y": D_Y|X (default, measures how well X predicts Y)
        - "x": D_X|Y (measures how well Y predicts X)
    random_state : int, optional
        Random seed for reproducibility
    woe_model : FastWoe, optional
        Pre-configured FastWoe instance to use as a template. If provided, new instances
        will be created with the same configuration at each iteration. If None, creates
        FastWoe() with default parameters.

    Returns:
    -------
    dict
        Dictionary containing:
        - 'selected_features': List of selected feature names in order
        - 'miv_history': List of marginal Somers' D values for each step
        - 'univariate_somersd': Dict mapping feature names to univariate Somers' D
        - 'model': Trained FastWoe model with selected features
        - 'test_performance': Dict with test set Somers' D at each step
        - 'correlation_matrix': Pairwise correlations of selected features

    Examples:
    --------
    >>> from fastwoe.modeling import marginal_somersd_selection
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Prepare data
    >>> X = pd.DataFrame({
    ...     'feature1': ['A', 'B', 'A', 'B'] * 100,
    ...     'feature2': ['X', 'Y', 'Y', 'X'] * 100,
    ... })
    >>> y = np.random.binomial(1, 0.3, 400)
    >>>
    >>> # Run feature selection
    >>> result = marginal_somersd_selection(
    ...     X, y,
    ...     min_miv=0.01,
    ...     max_features=5
    ... )
    >>>
    >>> print(f"Selected features: {result['selected_features']}")
    >>> print(f"MIV values: {result['miv_history']}")
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Use test set if provided, otherwise use training set
    X_eval = X_test if X_test is not None else X
    y_eval = y_test if y_test is not None else y

    # Convert y to numpy array
    y = np.asarray(y)
    y_eval = np.asarray(y_eval)

    # Pre-compute WOE values for all features (Step 1: Build the matrix on WOE-transformed features)
    candidate_features = list[Any](X.columns)
    woe_values_dict: dict[str, np.ndarray] = {}
    woe_encoders_dict: dict[str, FastWoe] = {}

    print("Pre-computing WOE values for all features...")
    for feature in candidate_features:
        woe_encoder = _create_woe_model(woe_model)
        try:
            woe_encoder.fit(X[[feature]], y)
            X_woe = woe_encoder.transform(X[[feature]])
            woe_values_dict[feature] = X_woe.iloc[:, 0].values
            woe_encoders_dict[feature] = woe_encoder
        except Exception:
            # If fitting fails, store zeros as placeholder
            woe_values_dict[feature] = np.zeros(len(X))
            woe_encoders_dict[feature] = woe_encoder

    # Build pairwise correlation matrix between features using Gini pairwise (for correlation checking)
    print("Building pairwise feature correlation matrix using Gini pairwise...")
    feature_correlation_matrix: dict[tuple[str, str], float] = {}
    for i, f_i in enumerate[Any](candidate_features):
        for f_j in candidate_features[i + 1 :]:
            try:
                # Use Gini pairwise: split one feature by median and compute Gini
                woe_i = woe_values_dict[f_i]
                woe_j = woe_values_dict[f_j]

                # Create binary split for feature j using median
                median_j = np.median(woe_j)
                binary_j = (woe_j > median_j).astype(int)

                # Compute Gini pairwise: use feature i WOE values as scores,
                # split by feature j binary values
                pos_scores = woe_i[binary_j == 1]
                neg_scores = woe_i[binary_j == 0]

                if len(pos_scores) > 0 and len(neg_scores) > 0:
                    gini_ij = gini_pairwise(pos_scores, neg_scores)
                    if gini_ij is not None and not np.isnan(gini_ij):
                        feature_correlation_matrix[(f_i, f_j)] = abs(gini_ij)
                    else:
                        feature_correlation_matrix[(f_i, f_j)] = 0.0
                else:
                    feature_correlation_matrix[(f_i, f_j)] = 0.0

                # Also compute in reverse direction and take average for symmetry
                median_i = np.median(woe_i)
                binary_i = (woe_i > median_i).astype(int)
                pos_scores_rev = woe_j[binary_i == 1]
                neg_scores_rev = woe_j[binary_i == 0]

                if len(pos_scores_rev) > 0 and len(neg_scores_rev) > 0:
                    gini_ji = gini_pairwise(pos_scores_rev, neg_scores_rev)
                    if gini_ji is not None and not np.isnan(gini_ji):
                        # Average the two directions for symmetric correlation
                        avg_gini = (
                            feature_correlation_matrix.get((f_i, f_j), 0.0) + abs(gini_ji)
                        ) / 2.0
                        feature_correlation_matrix[(f_i, f_j)] = avg_gini
                        feature_correlation_matrix[(f_j, f_i)] = avg_gini
                    else:
                        feature_correlation_matrix[(f_j, f_i)] = feature_correlation_matrix.get(
                            (f_i, f_j), 0.0
                        )
                else:
                    feature_correlation_matrix[(f_j, f_i)] = feature_correlation_matrix.get(
                        (f_i, f_j), 0.0
                    )
            except Exception:
                feature_correlation_matrix[(f_i, f_j)] = 0.0
                feature_correlation_matrix[(f_j, f_i)] = 0.0

    # Calculate univariate Somers' D for all features using pre-computed WOE values
    print("Calculating univariate Somers' D for all features...")
    univariate_somersd = {}
    for feature in candidate_features:
        woe_values = woe_values_dict[feature]
        pos_scores = woe_values[y == 1]
        neg_scores = woe_values[y == 0]

        if len(pos_scores) > 0 and len(neg_scores) > 0:
            somersd = somersd_pairwise(pos_scores, neg_scores, ties=ties)
            univariate_somersd[feature] = somersd if somersd is not None else 0.0
        else:
            univariate_somersd[feature] = 0.0

    # Sort features by univariate Somers' D
    sorted_features = sorted(univariate_somersd.items(), key=lambda x: x[1], reverse=True)

    # Start with feature with highest univariate Somers' D
    selected_features = [sorted_features[0][0]]
    miv_history = [sorted_features[0][1]]
    remaining_features = [f for f, _ in sorted_features[1:]]

    # Track test performance
    test_performance = []

    print(f"\nStep 1: Selected '{selected_features[0]}' (Somers' D: {miv_history[0]:.4f})")

    # Iterative selection
    step = 2
    while remaining_features:
        if max_features is not None and len(selected_features) >= max_features:
            print(f"\nReached max_features limit ({max_features})")
            break

        # Fit model with current features
        current_woe_model = _create_woe_model(woe_model)
        try:
            current_woe_model.fit(X[selected_features], y)
        except Exception as e:
            print(f"Error fitting model: {e}")
            break

        # Get model scores (probabilities) on training set
        train_probs = current_woe_model.predict_proba(X[selected_features])[:, 1]

        # Calculate test performance
        try:
            test_probs = current_woe_model.predict_proba(X_eval[selected_features])[:, 1]
            test_pos = test_probs[y_eval == 1]
            test_neg = test_probs[y_eval == 0]
            if len(test_pos) > 0 and len(test_neg) > 0:
                test_somersd = somersd_pairwise(test_pos, test_neg, ties=ties)
                test_performance.append(test_somersd if test_somersd is not None else 0.0)
            else:
                test_performance.append(0.0)
        except Exception:
            test_performance.append(0.0)

        # Calculate marginal Somers' D for each remaining feature
        marginal_somersd = {}

        for feature in remaining_features:
            try:
                # Use pre-computed WOE values for candidate feature
                feature_woe = woe_values_dict[feature]

                # Calculate residual correlation: Somers' D between feature and target,
                # controlling for current model scores
                # We use the feature WOE values directly and measure correlation with target
                # The "marginal" aspect comes from using rank correlation which naturally
                # accounts for what's already captured
                pos_scores = feature_woe[y == 1]
                neg_scores = feature_woe[y == 0]

                if len(pos_scores) > 0 and len(neg_scores) > 0:
                    # Calculate conformal residual metric: residual correlation between
                    # feature and target after controlling for current model scores
                    # This is done by computing residuals from regressing feature WOE on model scores,
                    # then computing Somers' D between residuals and target

                    # Compute residuals: feature_woe - predicted_feature_woe_from_model
                    # Use simple linear regression to get predicted values
                    try:
                        # Fit linear regression: feature_woe ~ train_probs
                        # Using least squares: beta = (X'X)^(-1)X'y
                        X_reg = np.column_stack([np.ones(len(train_probs)), train_probs])
                        beta = np.linalg.lstsq(X_reg, feature_woe, rcond=None)[0]
                        predicted_feature_woe = X_reg @ beta
                        residuals = feature_woe - predicted_feature_woe
                    except Exception:
                        # If regression fails, use original feature WOE values
                        residuals = feature_woe

                    # Compute conformal residual metric: Somers' D between residuals and target
                    residual_pos = residuals[y == 1]
                    residual_neg = residuals[y == 0]

                    if len(residual_pos) > 0 and len(residual_neg) > 0:
                        conformal_residual = somersd_pairwise(residual_pos, residual_neg, ties=ties)
                        if conformal_residual is not None:
                            marginal_somersd[feature] = max(0.0, abs(conformal_residual))
                        else:
                            marginal_somersd[feature] = 0.0
                    else:
                        marginal_somersd[feature] = 0.0
                else:
                    marginal_somersd[feature] = 0.0
            except Exception:
                marginal_somersd[feature] = 0.0

        if not marginal_somersd:
            break

        # Find feature with highest marginal Somers' D
        best_feature = max(marginal_somersd.items(), key=lambda x: x[1])
        best_miv = best_feature[1]

        # Check if MIV is above threshold
        if best_miv < min_miv:
            print(f"\nMIV ({best_miv:.4f}) below threshold ({min_miv})")
            break

        # Check correlation with already selected features using pre-computed matrix
        if selected_features:
            max_corr = 0.0
            for sel_feat in selected_features:
                corr = feature_correlation_matrix.get((best_feature[0], sel_feat), 0.0)
                max_corr = max(max_corr, corr)

            if max_corr > correlation_threshold:
                print(
                    f"\nFeature '{best_feature[0]}' has high correlation "
                    f"({max_corr:.3f}) with selected features, skipping..."
                )
                remaining_features.remove(best_feature[0])
                continue
        # Add best feature
        selected_features.append(best_feature[0])
        miv_history.append(best_miv)
        remaining_features.remove(best_feature[0])

        print(f"Step {step}: Selected '{best_feature[0]}' (Marginal Somers' D: {best_miv:.4f})")
        step += 1

    # Build final model
    final_model = _create_woe_model(woe_model)
    final_model.fit(X[selected_features], y)

    # Calculate correlation matrix of selected features
    correlation_matrix = None
    if len(selected_features) > 1:
        with contextlib.suppress(Exception):
            woe_features = final_model.transform(X[selected_features])
            correlation_matrix = woe_features.corr()
    return {
        "selected_features": selected_features,
        "miv_history": miv_history,
        "univariate_somersd": univariate_somersd,
        "model": final_model,
        "test_performance": test_performance,
        "correlation_matrix": correlation_matrix,
    }


def gini_shapley(
    score_dict: dict[str, np.ndarray],
    y: Union[np.ndarray, pd.Series],
    availability_mask: Optional[dict[str, np.ndarray]] = None,
    base_score_name: Optional[str] = None,
    ties: str = "y",
) -> pd.DataFrame:
    """
    Exact Shapley-value attribution of Gini under a fixed aggregation rule.

    This function computes Shapley values for each score source by enumerating
    all subsets (2^n). It attributes the total Gini of the *combined score*
    fairly across individual score sources.

    When base_score_name is provided:
    - Population is fixed to where base score is available
    - Base score is always included in the averaged score
    - Shapley values are computed only for "extras" (scores other than base)
    - Returns base-only Gini separately and incremental effects for extras

    - This IS a fair attribution of Gini under:
        * a fixed population (base-available if base_score_name provided, else intersection of all)
        * a fixed score aggregation rule (simple averaging)
        * a fixed availability regime

    The Shapley values:
        - are order-invariant
        - sum to the total combined Gini (or incremental effects sum to final - base)
        - account for all interactions

    Parameters
    ----------
    score_dict : dict[str, np.ndarray]
        Mapping of score names to score arrays.
    y : np.ndarray or pd.Series
        Binary target (0/1).
    availability_mask : dict[str, np.ndarray], optional
        Availability masks per score. If base_score_name is provided, population
        is fixed to base-available samples. Otherwise, uses intersection of all masks.
    base_score_name : str, optional
        Name of the base score. If provided:
        - Population is fixed to where base score is available
        - Base score is always included in averaged scores
        - Shapley values computed only for extras
        - Returns base-only Gini and incremental effects
    ties : str, default="y"
        Tie handling for Somers' D.

    Returns:
    -------
    pd.DataFrame
        If base_score_name is provided:
        - component: Score name
        - effect_on_gini: Base Gini for base, Shapley value for extras
        - role: "base_score_only" or "increment_over_base"
        - base_only_gini: Gini of base score alone
        - final_system_gini: Gini of all scores combined
        - sum_incremental_effects: Sum of Shapley values for extras
        - base_plus_incrementals: base_only_gini + sum_incremental_effects
        - n_samples_used: Number of samples in fixed population
        - n_pos, n_neg: Positive/negative counts
        Otherwise:
        - score_name: Score name
        - shapley_value: Shapley value
        - total_gini: Total Gini of all scores
        - n_scores: Number of scores
        - n_subsets: Number of subsets evaluated
    """
    y = np.asarray(y, dtype=float)
    n_samples = len(y)

    # Normalize inputs
    scores = {k: np.asarray(v, dtype=float) for k, v in score_dict.items()}
    score_names = list(scores.keys())

    if availability_mask is None:
        availability_mask = {k: np.ones(n_samples, dtype=bool) for k in score_names}
    else:
        availability_mask = {k: np.asarray(m, dtype=bool) for k, m in availability_mask.items()}
        for k in score_names:
            availability_mask.setdefault(k, np.ones(n_samples, dtype=bool))

    # Handle base score case
    if base_score_name is not None:
        if base_score_name not in scores:
            raise ValueError(f"base_score_name='{base_score_name}' not found in score_dict")

        # Fix population to where base score + y exist
        base_score = scores[base_score_name]
        base_valid = availability_mask[base_score_name] & ~np.isnan(base_score) & ~np.isnan(y)

        if base_valid.sum() == 0:
            raise ValueError("No valid samples after applying base availability and NaN filtering")

        y_valid = y[base_valid]
        if len(np.unique(y_valid)) < 2:
            raise ValueError("Target has no class variation on the base-valid population")

        extras = [k for k in score_names if k != base_score_name]

        # For extras, availability also requires score not-NaN
        extra_avail = {k: (availability_mask[k] & ~np.isnan(scores[k])) for k in extras}

        def gini_of(score_1d: np.ndarray) -> float:
            s = score_1d[base_valid]
            pos = s[y_valid == 1]
            neg = s[y_valid == 0]
            if len(pos) == 0 or len(neg) == 0:
                return 0.0
            g = somersd_pairwise(pos, neg, ties=ties)
            return float(g) if g is not None else 0.0

        cache: dict[tuple[str, ...], float] = {}

        def v(subset: tuple[str, ...]) -> float:
            """Gini of averaged score: base + subset of extras."""
            subset = tuple[str, ...](sorted(subset))
            if subset in cache:
                return cache[subset]

            numer = np.zeros(n_samples, dtype=float)
            denom = np.zeros(n_samples, dtype=float)

            # Base is always included for base_valid samples
            numer[base_valid] = base_score[base_valid]
            denom[base_valid] = 1.0

            # Add subset components only where they are available
            for k in subset:
                m = base_valid & extra_avail[k]
                numer[m] += scores[k][m]
                denom[m] += 1.0

            combined = np.zeros(n_samples, dtype=float)
            combined[base_valid] = numer[base_valid] / denom[base_valid]

            val = gini_of(combined)
            cache[subset] = val
            return val

        base_only_gini = v(())
        final_system_gini = v(tuple[str, ...](extras))

        # Exact Shapley for each extra component
        shapley = dict.fromkeys(extras, 0.0)
        M = len(extras)

        for i in extras:
            others = [k for k in extras if k != i]
            for r in range(M + 1):  # r = |S|, from 0 to M
                for S in itertools.combinations(others, r):
                    # Shapley weight: |S|! (M-|S|-1)! / M!
                    weight = (
                        np.math.factorial(r) * np.math.factorial(M - r - 1) / np.math.factorial(M)
                    )
                    shapley[i] += weight * (v(S + (i,)) - v(S))

        # Build output DataFrame
        rows = []
        rows.append(
            {
                "component": base_score_name,
                "effect_on_gini": base_only_gini,
                "role": "base_score_only",
            }
        )

        for k in sorted(extras, key=lambda z: shapley[z], reverse=True):
            rows.append(
                {
                    "component": k,
                    "effect_on_gini": shapley[k],
                    "role": "increment_over_base",
                }
            )

        out = pd.DataFrame(rows)
        out["base_only_gini"] = base_only_gini
        out["final_system_gini"] = final_system_gini

        return out

    # Original behavior: no base score, compute Shapley for all scores
    # Fix population to intersection of all availability masks
    global_mask = np.ones(n_samples, dtype=bool)
    for name in score_names:
        global_mask &= availability_mask[name]

    if global_mask.sum() == 0:
        raise ValueError("No samples available after intersecting availability masks")

    y_valid = y[global_mask]
    if len(np.unique(y_valid)) < 2:
        raise ValueError("Target has no class variation in the valid population")

    # Cache value function v(S)
    value_cache: dict[tuple[str, ...], float] = {}

    def v(subset: tuple[str, ...]) -> float:
        """Gini of the averaged score for a subset (cached)."""
        if subset in value_cache:
            return value_cache[subset]

        if not subset:
            value_cache[subset] = 0.0
            return 0.0

        cumulative = np.zeros(n_samples)
        for s in subset:
            cumulative[global_mask] += scores[s][global_mask]

        averaged_score = cumulative[global_mask] / len(subset)

        pos = averaged_score[y_valid == 1]
        neg = averaged_score[y_valid == 0]

        if len(pos) == 0 or len(neg) == 0:
            g = 0.0
        else:
            g = somersd_pairwise(pos, neg, ties=ties) or 0.0

        value_cache[subset] = g
        return g

    # Exact Shapley computation
    shapley = dict.fromkeys(score_names, 0.0)
    n_scores = len(score_names)

    for k in range(n_scores + 1):
        for subset in itertools.combinations(score_names, k):
            subset = tuple[str, ...](sorted(subset))
            v_subset = v(subset)

            for s in score_names:
                if s in subset:
                    continue

                extended = tuple[str, ...](sorted(subset + (s,)))
                v_extended = v(extended)

                # Shapley weight: |S|! (n-|S|-1)! / n!
                weight = (
                    np.math.factorial(len(subset))
                    * np.math.factorial(n_scores - len(subset) - 1)
                    / np.math.factorial(n_scores)
                )

                shapley[s] += weight * (v_extended - v_subset)

    total_gini = v(tuple[str, ...](sorted(score_names)))

    # Output
    result = pd.DataFrame(
        {
            "score_name": score_names,
            "shapley_value": [shapley[s] for s in score_names],
        }
    )

    result["total_gini"] = total_gini
    result["n_scores"] = n_scores
    result["n_subsets"] = 2**n_scores

    result = result.sort_values("shapley_value", ascending=False).reset_index(drop=True)

    return result
