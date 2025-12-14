"""
Feature selection using Marginal Somers' D (MSD) with rank correlation.

This module implements MSD-based feature selection using Somers' D (rank correlation)
instead of traditional WOE-based Information Value. It leverages fastwoe's predict_proba
to get model scores and uses the fast Somers' D implementation for efficient computation.
"""

from __future__ import annotations

import contextlib
import itertools
import math
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from .fastwoe import FastWoe
from .logging_config import _HAS_LOGGING, logger
from .metrics import somersd_pairwise, somersd_xy, somersd_yx


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
    min_msd: float = 0.02,
    max_features: Optional[int] = None,
    correlation_threshold: float = 0.5,
    ties: str = "y",
    random_state: Optional[int] = None,
    woe_model: Optional[FastWoe] = None,
    verbose: bool = False,
) -> dict:
    """
    Feature selection using Marginal Somers' D (MSD).

    Greedy forward selection that adds features based on their marginal rank correlation
    with the target, controlling for already-selected features. Uses WOE-transformed
    features and filters out highly correlated candidates.

    Parameters
    ----------
    X : pd.DataFrame
        Training features (categorical or mixed types)
    y : np.ndarray or pd.Series
        Target variable (binary 0/1 or continuous)
    X_test : pd.DataFrame, optional
        Test set for performance monitoring
    y_test : np.ndarray or pd.Series, optional
        Test target for performance monitoring
    min_msd : float, default=0.02
        Minimum marginal Somers' D to continue selection
    max_features : int, optional
        Maximum features to select
    correlation_threshold : float, default=0.5
        Skip features with Somers' D correlation above this threshold
    ties : str, default="y"
        Tie handling: "y" for D_Y|X, "x" for D_X|Y
    random_state : int, optional
        Random seed for reproducibility
    woe_model : FastWoe, optional
        Template FastWoe instance for configuration
    verbose : bool, default=False
        Enable detailed logging (requires loguru and rich)

    Returns:
    -------
    dict
        - selected_features: List of feature names in selection order
        - msd_history: Marginal Somers' D at each step
        - univariate_somersd: Dict of univariate Somers' D values
        - model: Trained FastWoe with selected features
        - test_performance: Test Somers' D at each step
        - correlation_matrix: Pairwise correlations of selected features

    Examples:
    --------
    >>> result = marginal_somersd_selection(X, y, min_msd=0.01, max_features=5)
    >>> print(result['selected_features'])
    >>> print(result['msd_history'])
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Use test set if provided, otherwise use training set
    X_eval = X_test if X_test is not None else X
    y_eval = y_test if y_test is not None else y

    # Convert y to numpy array
    y = np.asarray(y)
    y_eval = np.asarray(y_eval)

    # Detect if target is binary (only 0 and 1) or continuous
    unique_labels = np.unique(y[~np.isnan(y)])
    is_binary = set[Any](unique_labels).issubset({0, 1}) and len(unique_labels) <= 2

    # Helper function for output
    # Only outputs when verbose=True, otherwise silent
    def log_or_print(message: str, use_logger: bool = False) -> None:
        if not verbose:
            return  # Silent when verbose=False
        if use_logger and _HAS_LOGGING:
            logger.info(message)
        else:
            print(message)

    # Pre-compute WOE values for all features (Step 1: Build the matrix on WOE-transformed features)
    candidate_features = list[Any](X.columns)
    woe_values_dict: dict[str, np.ndarray] = {}
    woe_encoders_dict: dict[str, FastWoe] = {}

    log_or_print("Pre-computing WOE values for all features...", use_logger=verbose)
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

    # Build pairwise correlation matrix between features using Somers' D (for correlation checking)
    log_or_print(
        "Building pairwise feature correlation matrix using Somers' D...", use_logger=verbose
    )
    feature_correlation_matrix: dict[tuple[str, str], float] = {}
    for i, f_i in enumerate[Any](candidate_features):
        for f_j in candidate_features[i + 1 :]:
            try:
                # Compute Somers' D directly between two continuous WOE value arrays
                # This gives us rank correlation between the features
                woe_i = woe_values_dict[f_i]
                woe_j = woe_values_dict[f_j]

                # Compute Somers' D in both directions and take average for symmetry
                # D_Y|X: how well woe_j predicts woe_i
                result_ij = somersd_yx(woe_i, woe_j)
                somersd_ij = 0.0 if np.isnan(result_ij.statistic) else result_ij.statistic

                # D_Y|X: how well woe_i predicts woe_j
                result_ji = somersd_yx(woe_j, woe_i)
                somersd_ji = 0.0 if np.isnan(result_ji.statistic) else result_ji.statistic

                # Average absolute values for symmetric correlation measure
                avg_corr = (abs(somersd_ij) + abs(somersd_ji)) / 2.0
                feature_correlation_matrix[(f_i, f_j)] = avg_corr
                feature_correlation_matrix[(f_j, f_i)] = avg_corr

                if verbose and _HAS_LOGGING:
                    logger.debug(
                        f"  Pairwise correlation [{f_i} ↔ {f_j}]: "
                        f"Somers' D = {avg_corr:.4f} (ij={somersd_ij:.4f}, ji={somersd_ji:.4f})"
                    )
            except Exception:
                feature_correlation_matrix[(f_i, f_j)] = 0.0
                feature_correlation_matrix[(f_j, f_i)] = 0.0

    # Calculate univariate Somers' D for all features using pre-computed WOE values
    log_or_print("Calculating univariate Somers' D for all features...", use_logger=verbose)
    univariate_somersd = {}
    for feature in candidate_features:
        woe_values = woe_values_dict[feature]

        if is_binary:
            # Binary target: use pairwise comparison
            pos_scores = woe_values[y == 1]
            neg_scores = woe_values[y == 0]
            if len(pos_scores) > 0 and len(neg_scores) > 0:
                somersd = somersd_pairwise(pos_scores, neg_scores, ties=ties)
                univariate_somersd[feature] = abs(somersd) if somersd is not None else 0.0
            else:
                univariate_somersd[feature] = 0.0
        else:
            # Continuous target: use somersd_yx directly
            if ties == "y":
                result = somersd_yx(y, woe_values)
            else:
                result = somersd_xy(y, woe_values)
            univariate_somersd[feature] = (
                0.0 if np.isnan(result.statistic) else abs(result.statistic)
            )

        if verbose and _HAS_LOGGING:
            logger.debug(f"  Univariate Somers' D [{feature}]: {univariate_somersd[feature]:.4f}")

    # Sort features by univariate Somers' D
    sorted_features = sorted(univariate_somersd.items(), key=lambda x: x[1], reverse=True)

    # Start with feature with highest univariate Somers' D
    selected_features = [sorted_features[0][0]]
    msd_history = [sorted_features[0][1]]
    remaining_features = [f for f, _ in sorted_features[1:]]

    # Track test performance
    test_performance = []

    log_or_print(
        f"\nStep 1: Selected '{selected_features[0]}' (Somers' D: {msd_history[0]:.4f})",
        use_logger=verbose,
    )

    # Iterative selection
    step = 2
    while remaining_features:
        if max_features is not None and len(selected_features) >= max_features:
            log_or_print(f"\nReached max_features limit ({max_features})", use_logger=verbose)
            break

        # Fit model with current features
        current_woe_model = _create_woe_model(woe_model)
        try:
            current_woe_model.fit(X[selected_features], y)
        except Exception as e:
            log_or_print(f"Error fitting model: {e}", use_logger=verbose)
            break

        # Get model scores on training set
        # For binary targets, use predict_proba; for continuous, use predict
        if is_binary:
            train_scores = current_woe_model.predict_proba(X[selected_features])[:, 1]
        else:
            # For continuous targets, FastWoe might not support predict directly
            # Use predict_proba and take the mean or use a different approach
            # For now, we'll use predict_proba[:, 1] as a score
            try:
                train_scores = current_woe_model.predict_proba(X[selected_features])[:, 1]
            except Exception:
                # Fallback: use transformed WOE values as scores
                woe_transformed = current_woe_model.transform(X[selected_features])
                train_scores = woe_transformed.sum(axis=1).values

        # Calculate test performance
        try:
            if is_binary:
                test_scores = current_woe_model.predict_proba(X_eval[selected_features])[:, 1]
                test_pos = test_scores[y_eval == 1]
                test_neg = test_scores[y_eval == 0]
                if len(test_pos) > 0 and len(test_neg) > 0:
                    test_somersd = somersd_pairwise(test_pos, test_neg, ties=ties)
                    test_performance.append(test_somersd if test_somersd is not None else 0.0)
                else:
                    test_performance.append(0.0)
            else:
                # Continuous target: compute Somers' D directly
                test_scores = current_woe_model.predict_proba(X_eval[selected_features])[:, 1]
                if ties == "y":
                    result = somersd_yx(y_eval, test_scores)
                else:
                    result = somersd_xy(y_eval, test_scores)
                test_performance.append(
                    0.0 if np.isnan(result.statistic) else abs(result.statistic)
                )
        except Exception:
            test_performance.append(0.0)

        if verbose and _HAS_LOGGING:
            logger.debug(
                f"\n[bold cyan]Step {step}:[/bold cyan] Evaluating {len(remaining_features)} "
                f"candidate features (current model Somers' D: {test_performance[-1]:.4f})"
            )

        # Calculate marginal Somers' D for each remaining feature
        marginal_somersd = {}

        for feature in remaining_features:
            feature_woe = woe_values_dict[feature]

            if is_binary:
                # Binary target: use pairwise comparison
                pos_scores = feature_woe[y == 1]
                neg_scores = feature_woe[y == 0]
                if len(pos_scores) > 0 and len(neg_scores) > 0:
                    somersd = somersd_pairwise(pos_scores, neg_scores, ties=ties)
                    marginal_somersd[feature] = abs(somersd) if somersd is not None else 0.0
                else:
                    marginal_somersd[feature] = 0.0
            else:
                # Continuous target: use somersd_yx directly
                if ties == "y":
                    result = somersd_yx(y, feature_woe)
                else:
                    result = somersd_xy(y, feature_woe)
                marginal_somersd[feature] = (
                    0.0 if np.isnan(result.statistic) else abs(result.statistic)
                )

        if not marginal_somersd:
            break

        # Show pairwise marginal Somers' D values in verbose mode
        if verbose and _HAS_LOGGING:
            sorted_marginal = sorted(marginal_somersd.items(), key=lambda x: x[1], reverse=True)
            logger.debug("  [bold]Marginal Somers' D values:[/bold]")
            for feat, msd_val in sorted_marginal[:10]:  # Show top 10
                logger.debug(f"{feat:30s}: {msd_val:.4f}")

        # Find feature with highest marginal Somers' D
        best_feature = max(marginal_somersd.items(), key=lambda x: x[1])
        best_msd = best_feature[1]

        # Check if MSD is above threshold
        if best_msd < min_msd:
            log_or_print(f"\nMSD ({best_msd:.4f}) below threshold ({min_msd})", use_logger=verbose)
            break

        # Check correlation with already selected features using pre-computed matrix
        if selected_features:
            max_corr = 0.0
            max_corr_feature = None
            for sel_feat in selected_features:
                corr = feature_correlation_matrix.get((best_feature[0], sel_feat), 0.0)
                if corr > max_corr:
                    max_corr = corr
                    max_corr_feature = sel_feat

            if verbose and _HAS_LOGGING:
                logger.debug(
                    f"  Checking correlations with selected features "
                    f"(max: {max_corr:.4f} with '{max_corr_feature}')"
                )

            if max_corr > correlation_threshold:
                log_or_print(
                    f"\nFeature '{best_feature[0]}' has high correlation "
                    f"({max_corr:.3f}) with selected features, skipping...",
                    use_logger=verbose,
                )
                remaining_features.remove(best_feature[0])
                continue
        # Add best feature
        selected_features.append(best_feature[0])
        msd_history.append(best_msd)
        remaining_features.remove(best_feature[0])

        log_or_print(
            f"Step {step}: Selected '{best_feature[0]}' (Marginal Somers' D: {best_msd:.4f})",
            use_logger=verbose,
        )
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
        "msd_history": msd_history,
        "univariate_somersd": univariate_somersd,
        "model": final_model,
        "test_performance": test_performance,
        "correlation_matrix": correlation_matrix,
    }


def somersd_shapley(
    score_dict: dict[str, np.ndarray],
    y: Union[np.ndarray, pd.Series],
    availability_mask: Optional[dict[str, np.ndarray]] = None,
    base_score_name: Optional[str] = None,
    ties: str = "y",
) -> pd.DataFrame:
    """
    Exact Shapley value attribution of Somers' D under score averaging.

    Computes fair attribution of combined score performance across individual score
    sources by enumerating all 2^n subsets. Handles variable score availability and
    optional base score conditioning.

    Parameters
    ----------
    score_dict : dict[str, np.ndarray]
        Mapping of score names to score arrays
    y : np.ndarray or pd.Series
        Target variable (binary 0/1 or continuous)
    availability_mask : dict[str, np.ndarray], optional
        Per-score availability masks. Population fixed to intersection of all masks,
        or to base availability if base_score_name provided
    base_score_name : str, optional
        Base score name. If provided, returns incremental Shapley values for extras
        conditional on always including the base score
    ties : str, default="y"
        Tie handling: "y" for D_Y|X, "x" for D_X|Y

    Returns:
    -------
    pd.DataFrame
        Without base_score_name:
            - score_name, shapley_value, total_somersd, n_scores, n_subsets
        With base_score_name:
            - component, effect_on_somersd, role, base_only_somersd, final_system_somersd

    Notes:
    -----
    Shapley values are order-invariant, sum to total performance, and account for
    all interactions. Computational complexity is O(2^n) - impractical beyond ~15 scores.
    """
    y = np.asarray(y, dtype=float)
    n_samples = len(y)

    # Normalize inputs
    scores = {k: np.asarray(v, dtype=float) for k, v in score_dict.items()}
    score_names = list[str](scores.keys())

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
            raise ValueError("Target has no variation on the base-valid population")

        # Detect if binary or continuous target
        unique_labels = np.unique(y_valid)
        is_binary = set(unique_labels).issubset({0, 1}) and len(unique_labels) <= 2

        extras = [k for k in score_names if k != base_score_name]

        # For extras, availability also requires score not-NaN
        extra_avail = {k: (availability_mask[k] & ~np.isnan(scores[k])) for k in extras}

        def somersd_of(score_1d: np.ndarray) -> float:
            """Compute Somers' D for a score array, handling both binary and continuous targets."""
            s = score_1d[base_valid]
            if is_binary:
                pos = s[y_valid == 1]
                neg = s[y_valid == 0]
                if len(pos) == 0 or len(neg) == 0:
                    return 0.0
                result = somersd_pairwise(pos, neg, ties=ties)
                return float(result) if result is not None else 0.0
            else:
                # Continuous target: use direct Somers' D
                result = somersd_yx(y_valid, s) if ties == "y" else somersd_xy(y_valid, s)
                return 0.0 if np.isnan(result.statistic) else float(result.statistic)

        cache: dict[tuple[str, ...], float] = {}

        def v(subset: tuple[str, ...]) -> float:
            """Somers' D of averaged score: base + subset of extras."""
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

            val = somersd_of(combined)
            cache[subset] = val
            return val

        base_only_somersd = v(())
        final_system_somersd = v(tuple[str, ...](extras))

        # Exact Shapley for each extra component
        shapley = dict.fromkeys(extras, 0.0)
        M = len(extras)

        for i in extras:
            others = [k for k in extras if k != i]
            for r in range(M + 1):  # r = |S|, from 0 to M
                for S in itertools.combinations(others, r):
                    # Shapley weight: |S|! (M-|S|-1)! / M!
                    weight = math.factorial(r) * math.factorial(M - r - 1) / math.factorial(M)
                    shapley[i] += weight * (v(S + (i,)) - v(S))

        # Build output DataFrame
        rows = []
        rows.append(
            {
                "component": base_score_name,
                "effect_on_somersd": base_only_somersd,
                "role": "base_score_only",
            }
        )

        for k in sorted(extras, key=lambda z: shapley[z], reverse=True):
            rows.append(
                {
                    "component": k,
                    "effect_on_somersd": shapley[k],
                    "role": "increment_over_base",
                }
            )

        out = pd.DataFrame(rows)
        out["base_only_somersd"] = base_only_somersd
        out["final_system_somersd"] = final_system_somersd

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
        raise ValueError("Target has no variation in the valid population")

    # Detect if binary or continuous target
    unique_labels = np.unique(y_valid)
    is_binary = set(unique_labels).issubset({0, 1}) and len(unique_labels) <= 2

    # Cache value function v(S)
    value_cache: dict[tuple[str, ...], float] = {}

    def v(subset: tuple[str, ...]) -> float:
        """Somers' D of the averaged score for a subset (cached)."""
        if subset in value_cache:
            return value_cache[subset]

        if not subset:
            value_cache[subset] = 0.0
            return 0.0

        cumulative = np.zeros(n_samples)
        for s in subset:
            cumulative[global_mask] += scores[s][global_mask]

        averaged_score = cumulative[global_mask] / len(subset)

        if is_binary:
            pos = averaged_score[y_valid == 1]
            neg = averaged_score[y_valid == 0]
            if len(pos) == 0 or len(neg) == 0:
                d = 0.0
            else:
                result = somersd_pairwise(pos, neg, ties=ties)
                d = float(result) if result is not None else 0.0
        else:
            # Continuous target: use direct Somers' D
            result = (
                somersd_yx(y_valid, averaged_score)
                if ties == "y"
                else somersd_xy(y_valid, averaged_score)
            )
            d = 0.0 if np.isnan(result.statistic) else float(result.statistic)

        value_cache[subset] = d
        return d

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
                    math.factorial(len(subset))
                    * math.factorial(n_scores - len(subset) - 1)
                    / math.factorial(n_scores)
                )

                shapley[s] += weight * (v_extended - v_subset)

    total_somersd = v(tuple[str, ...](sorted(score_names)))

    # Output
    result = pd.DataFrame(
        {
            "score_name": score_names,
            "shapley_value": [shapley[s] for s in score_names],
        }
    )

    result["total_somersd"] = total_somersd
    result["n_scores"] = n_scores
    result["n_subsets"] = 2**n_scores

    result = result.sort_values("shapley_value", ascending=False).reset_index(drop=True)

    return result
