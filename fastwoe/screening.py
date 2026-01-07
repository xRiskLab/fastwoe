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
from .metrics import SomersDResult, somersd_pairwise, somersd_xy, somersd_yx


def _compute_somersd(
    y: np.ndarray,
    scores: np.ndarray,
    is_binary: bool,
    ties: str = "y",
) -> float:
    """Compute Somers' D for scores against target, handling binary and continuous targets."""
    if is_binary:
        pos_scores = scores[y == 1]
        neg_scores = scores[y == 0]
        if len(pos_scores) > 0 and len(neg_scores) > 0:
            pairwise_result = somersd_pairwise(pos_scores, neg_scores, ties=ties)
            return abs(pairwise_result) if pairwise_result is not None else 0.0
        return 0.0
    else:
        somers_result: SomersDResult = (
            somersd_yx(y, scores) if ties == "y" else somersd_xy(y, scores)
        )
        return 0.0 if np.isnan(somers_result.statistic) else abs(somers_result.statistic)


def _build_feature_correlation_matrix(
    candidate_features: list[str],
    woe_values_dict: dict[str, np.ndarray],
    verbose: bool = False,
) -> dict[tuple[str, str], float]:
    """Build pairwise correlation matrix between features using Somers' D.

    Computes symmetric correlation by averaging Somers' D in both directions
    for each feature pair.
    """
    feature_correlation_matrix: dict[tuple[str, str], float] = {}
    for i, f_i in enumerate[str](candidate_features):
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
                        f"Pairwise correlation [{f_i} ↔ {f_j}]: "
                        f"Somers' D = {avg_corr:.4f} (ij={somersd_ij:.4f}, ji={somersd_ji:.4f})"
                    )
            except Exception:
                feature_correlation_matrix[(f_i, f_j)] = 0.0
                feature_correlation_matrix[(f_j, f_i)] = 0.0
    return feature_correlation_matrix


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

    Selects features based on their Somers' D correlation with model residuals,
    measuring true incremental contribution beyond already-selected features.

    Algorithm
    ---------
    1. Step 1: Select feature with highest univariate Somers' D with target y
    2. Step 2+: For each subsequent step:
    a. Fit model with currently selected features
    b. Compute residuals: ε = y - model.predict_proba(X[selected])
    c. For each remaining feature, compute |Somers' D(ε, feature_WOE)|
    d. Select feature with highest |D(ε, feature)|
    e. Skip if correlation with selected features > correlation_threshold

    This is "truly marginal" - each feature's contribution is measured relative
    to what prior features fail to explain (residual variance).

    Parameters
    ----------
    ...
    correlation_threshold : float, default=0.5
        Skip features with pairwise Somers' D correlation above this threshold
        with already-selected features. Acts as a safety net against multicollinearity,
        though residual-based selection already penalizes redundancy naturally.
    ...

    Notes:
    -----
    - Marginal Somers' D values decrease at each step (measuring against shrinking
    residual variance), unlike univariate methods that measure against original y
    - First feature is selected using univariate Somers' D (no residuals yet)
    - From Step 2 onwards, selection is based on residual correlation
    - The method relies on rankable residual variance - monotonic patterns in what
    the current model doesn't capture

    Examples:
    --------
    >>> # Select features using marginal (residual-based) approach
    >>> result = marginal_somersd_selection(
    ...     X_train, y_train,
    ...     X_test=X_test, y_test=y_test,
    ...     min_msd=0.01,
    ...     max_features=10,
    ...     correlation_threshold=0.6
    ... )
    >>> print(result['selected_features'])
    >>> print(result['msd_history'])  # Note: Values decrease with each step
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
    is_binary = set(unique_labels).issubset({0, 1}) and len(unique_labels) <= 2

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
            woe_values_dict[feature] = np.asarray(X_woe.iloc[:, 0].values, dtype=float)
            woe_encoders_dict[feature] = woe_encoder
        except Exception:
            # If fitting fails, store zeros as placeholder
            woe_values_dict[feature] = np.zeros(len(X))
            woe_encoders_dict[feature] = woe_encoder

    # Build pairwise correlation matrix between features using Somers' D (for correlation checking)
    log_or_print(
        "Building pairwise feature correlation matrix using Somers' D...", use_logger=verbose
    )
    feature_correlation_matrix = _build_feature_correlation_matrix(
        candidate_features, woe_values_dict, verbose
    )

    # Calculate univariate Somers' D for all features using pre-computed WOE values
    log_or_print("Calculating univariate Somers' D for all features...", use_logger=verbose)
    y_arr = np.asarray(y, dtype=float)
    univariate_somersd = {
        feature: _compute_somersd(
            y_arr, np.asarray(woe_values_dict[feature], dtype=float), is_binary, ties
        )
        for feature in candidate_features
    }

    if verbose and _HAS_LOGGING:
        for feature in candidate_features:
            logger.debug(f"Univariate Somers' D [{feature}]: {univariate_somersd[feature]:.4f}")

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

        # Calculate test performance
        try:
            test_scores = current_woe_model.predict_proba(X_eval[selected_features])[:, 1]
            y_eval_arr = np.asarray(y_eval, dtype=float)
            test_performance.append(
                _compute_somersd(y_eval_arr, np.asarray(test_scores, dtype=float), is_binary, ties)
            )
        except Exception:
            test_performance.append(0.0)

        # Compute residuals for truly marginal selection (Step 2 onwards)
        # At this point, selected_features is non-empty (contains at least first feature)
        # We measure against residuals to capture what the current model doesn't explain
        try:
            current_predictions = current_woe_model.predict_proba(X[selected_features])[:, 1]
            residuals = y - current_predictions
            target_for_msd = residuals
            is_binary_for_msd = False  # Residuals are always continuous
        except Exception as e:
            if verbose and _HAS_LOGGING:
                logger.warning(f"Failed to compute residuals: {e}. Using original target.")
            # Fallback to original target if residual computation fails
            target_for_msd = y
            is_binary_for_msd = is_binary

        if verbose and _HAS_LOGGING:
            target_desc = "residuals (y - ŷ)"
            logger.debug(
                f"\n[bold cyan]Step {step}:[/bold cyan] Evaluating {len(remaining_features)} "
                f"candidate features (current model Somers' D: {test_performance[-1]:.4f}, "
                f"measuring against {target_desc})"
            )

        # Calculate marginal Somers' D for each remaining feature
        marginal_somersd = {
            feature: _compute_somersd(
                target_for_msd, woe_values_dict[feature], is_binary_for_msd, ties
            )
            for feature in remaining_features
        }

        if not marginal_somersd:
            break

        # Show pairwise marginal Somers' D values in verbose mode
        if verbose and _HAS_LOGGING:
            sorted_marginal = sorted(marginal_somersd.items(), key=lambda x: x[1], reverse=True)
            logger.debug("[bold]Marginal Somers' D values:[/bold]")
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
                    f"Checking correlations with selected features: "
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
    score_names = list(scores.keys())

    if availability_mask is None:
        availability_mask = {key: np.ones(n_samples, dtype=bool) for key in score_names}
    else:
        availability_mask = {key: np.asarray(m, dtype=bool) for key, m in availability_mask.items()}
        for key in score_names:
            availability_mask.setdefault(key, np.ones(n_samples, dtype=bool))

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
            return _compute_somersd(y_valid, s, is_binary, ties)

        cache: dict[tuple[str, ...], float] = {}

        def v_base(subset: tuple[str, ...]) -> float:
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

        base_only_somersd = v_base(())
        final_system_somersd = v_base(tuple[str, ...](extras))

        # Exact Shapley for each extra component
        shapley = dict.fromkeys(extras, 0.0)
        M = len(extras)

        for i in extras:
            others = [k for k in extras if k != i]
            for r in range(M + 1):  # r = |S|, from 0 to M
                for S in itertools.combinations(others, r):
                    # Shapley weight: |S|! (M-|S|-1)! / M!
                    weight = math.factorial(r) * math.factorial(M - r - 1) / math.factorial(M)
                    shapley[i] += weight * (v_base(S + (i,)) - v_base(S))

        # Build output DataFrame
        rows = []
        rows.append(
            {
                "component": base_score_name,
                "effect_on_somersd": base_only_somersd,
                "role": "base_score_only",
            }
        )

        for component_name in sorted(extras, key=lambda z: shapley[z], reverse=True):
            rows.append(
                {
                    "component": component_name,
                    "effect_on_somersd": shapley[component_name],
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

        d = _compute_somersd(y_valid, averaged_score, is_binary, ties)

        value_cache[subset] = d
        return d

    # Exact Shapley computation
    shapley = dict.fromkeys(score_names, 0.0)
    n_scores = len(score_names)

    for k_int in range(n_scores + 1):
        k: int = k_int
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
