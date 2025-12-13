"""
Feature selection using Marginal Information Value (MIV) with rank correlation.

This module implements MIV-based feature selection using Somers' D (rank correlation)
instead of traditional WOE-based Information Value. It leverages fastwoe's predict_proba
to get model scores and uses the fast Somers' D implementation for efficient computation.

Based on: Revolut MIV methodology (see docs/revolut_miv.md)
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from .fastwoe import FastWoe
from .metrics import somersd_pairwise


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

    # Calculate univariate Somers' D for all features
    univariate_somersd = {}
    candidate_features = list(X.columns)

    print("Calculating univariate Somers' D for all features...")
    for feature in candidate_features:
        # Get WOE values for this feature
        woe_encoder = FastWoe()
        try:
            woe_encoder.fit(X[[feature]], y)
            X_woe = woe_encoder.transform(X[[feature]])
            woe_values = X_woe.iloc[:, 0].values

            # Calculate Somers' D between WOE values and target
            pos_scores = woe_values[y == 1]
            neg_scores = woe_values[y == 0]

            if len(pos_scores) > 0 and len(neg_scores) > 0:
                somersd = somersd_pairwise(pos_scores, neg_scores, ties=ties)
                univariate_somersd[feature] = somersd if somersd is not None else 0.0
            else:
                univariate_somersd[feature] = 0.0
        except Exception:
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
        woe_model = FastWoe()
        try:
            woe_model.fit(X[selected_features], y)
        except Exception as e:
            print(f"Error fitting model: {e}")
            break

        # Get model scores (probabilities) on training set
        train_probs = woe_model.predict_proba(X[selected_features])[:, 1]

        # Calculate test performance
        try:
            test_probs = woe_model.predict_proba(X_eval[selected_features])[:, 1]
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
                # Get WOE values for candidate feature
                woe_encoder = FastWoe()
                woe_encoder.fit(X[[feature]], y)
                X_woe = woe_encoder.transform(X[[feature]])
                feature_woe = X_woe.iloc[:, 0].values

                # Calculate residual correlation: Somers' D between feature and target,
                # controlling for current model scores
                # We use the feature WOE values directly and measure correlation with target
                # The "marginal" aspect comes from using rank correlation which naturally
                # accounts for what's already captured
                pos_scores = feature_woe[y == 1]
                neg_scores = feature_woe[y == 0]

                if len(pos_scores) > 0 and len(neg_scores) > 0:
                    # Calculate correlation between feature and model residuals
                    # Residual = target - predicted probability
                    residuals = y.astype(float) - train_probs

                    # Calculate Somers' D between feature WOE and residuals
                    # This measures how much additional information the feature provides
                    pos_residuals = residuals[y == 1]
                    neg_residuals = residuals[y == 0]

                    # Alternative: Use partial correlation approach
                    # Calculate Somers' D of feature WOE with target, but weight by
                    # how much the feature adds beyond current model
                    feature_somersd = somersd_pairwise(pos_scores, neg_scores, ties=ties)

                    if feature_somersd is not None:
                        # Marginal contribution: feature's correlation minus
                        # correlation with current model scores
                        # This is a simplified approach - full MIV would use WOE differences
                        model_feature_corr = np.corrcoef(feature_woe, train_probs)[0, 1]
                        if np.isnan(model_feature_corr):
                            model_feature_corr = 0.0

                        # Marginal Somers' D: reduce by correlation with current model
                        marginal = feature_somersd * (1 - abs(model_feature_corr))
                        marginal_somersd[feature] = max(0.0, marginal)
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

        # Check correlation with already selected features
        if len(selected_features) > 0:
            try:
                # Calculate correlation between candidate and selected features
                candidate_woe = FastWoe()
                candidate_woe.fit(X[[best_feature[0]]], y)
                candidate_woe_vals = candidate_woe.transform(X[[best_feature[0]]]).iloc[:, 0]

                max_corr = 0.0
                for sel_feat in selected_features:
                    sel_woe = FastWoe()
                    sel_woe.fit(X[[sel_feat]], y)
                    sel_woe_vals = sel_woe.transform(X[[sel_feat]]).iloc[:, 0]
                    corr = abs(np.corrcoef(candidate_woe_vals, sel_woe_vals)[0, 1])
                    if not np.isnan(corr):
                        max_corr = max(max_corr, corr)

                if max_corr > correlation_threshold:
                    print(
                        f"\nFeature '{best_feature[0]}' has high correlation "
                        f"({max_corr:.3f}) with selected features, skipping..."
                    )
                    remaining_features.remove(best_feature[0])
                    continue
            except Exception:
                pass

        # Add best feature
        selected_features.append(best_feature[0])
        miv_history.append(best_miv)
        remaining_features.remove(best_feature[0])

        print(f"Step {step}: Selected '{best_feature[0]}' (Marginal Somers' D: {best_miv:.4f})")
        step += 1

    # Build final model
    final_model = FastWoe()
    final_model.fit(X[selected_features], y)

    # Calculate correlation matrix of selected features
    correlation_matrix = None
    if len(selected_features) > 1:
        try:
            woe_features = final_model.transform(X[selected_features])
            correlation_matrix = woe_features.corr()
        except Exception:
            pass

    return {
        "selected_features": selected_features,
        "miv_history": miv_history,
        "univariate_somersd": univariate_somersd,
        "model": final_model,
        "test_performance": test_performance,
        "correlation_matrix": correlation_matrix,
    }


def cumulative_gini_analysis(
    score_dict: dict[str, np.ndarray],
    y: Union[np.ndarray, pd.Series],
    availability_mask: Optional[dict[str, np.ndarray]] = None,
    ties: str = "y",
) -> pd.DataFrame:
    """
    Analyze cumulative Gini from multiple score vectors (e.g., different models/credit checks).

    This function calculates how Gini improves as you add each score/model sequentially.
    It's useful for understanding the incremental value of different data sources or models
    in a credit decisioning process.

    Parameters
    ----------
    score_dict : dict[str, np.ndarray]
        Dictionary mapping score names to score arrays. Each array should have the same length.
        Example: {'model1': scores1, 'bureau_score': scores2, 'internal_score': scores3}
    y : np.ndarray or pd.Series
        Binary target variable (0/1)
    availability_mask : dict[str, np.ndarray], optional
        Dictionary mapping score names to boolean arrays indicating availability.
        If None, assumes all scores are available for all samples.
        Example: {'bureau_score': has_bureau_data, 'model1': np.ones(n, dtype=bool)}
    ties : str, default="y"
        How to handle ties in Somers' D calculation:
        - "y": D_Y|X (default, measures how well X predicts Y)
        - "x": D_X|Y (measures how well Y predicts X)

    Returns:
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'step': Step number (0 = baseline, 1 = first score, etc.)
        - 'score_name': Name of score added at this step
        - 'gini': Cumulative Gini after adding this score
        - 'marginal_gini': Marginal Gini contribution of this score
        - 'n_samples': Number of samples with this score available
        - 'n_pos': Number of positive samples with this score
        - 'n_neg': Number of negative samples with this score

    Examples:
    --------
    >>> import numpy as np
    >>> from fastwoe.modeling import cumulative_gini_analysis
    >>>
    >>> n = 1000
    >>> y = np.random.binomial(1, 0.3, n)
    >>> score_dict = {
    ...     'model1': np.random.uniform(0, 1, n),
    ...     'bureau_score': np.random.uniform(0, 1, n),
    ...     'internal_score': np.random.uniform(0, 1, n),
    ... }
    >>> # Only 80% have bureau data
    >>> availability_mask = {
    ...     'model1': np.ones(n, dtype=bool),
    ...     'bureau_score': np.random.binomial(1, 0.8, n).astype(bool),
    ...     'internal_score': np.ones(n, dtype=bool),
    ... }
    >>>
    >>> results = cumulative_gini_analysis(score_dict, y, availability_mask)
    >>> print(results)
    """
    y = np.asarray(y)
    n_samples = len(y)

    # Initialize availability mask if not provided
    if availability_mask is None:
        availability_mask = {name: np.ones(n_samples, dtype=bool) for name in score_dict.keys()}

    # Validate inputs
    for name, scores in score_dict.items():
        if len(scores) != n_samples:
            raise ValueError(f"Score '{name}' has length {len(scores)}, expected {n_samples}")
        if name not in availability_mask:
            availability_mask[name] = np.ones(n_samples, dtype=bool)
        if len(availability_mask[name]) != n_samples:
            raise ValueError(
                f"Availability mask for '{name}' has length {len(availability_mask[name])}, "
                f"expected {n_samples}"
            )

    results = []

    # Baseline: no scores
    baseline_gini = 0.0
    results.append(
        {
            "step": 0,
            "score_name": "baseline",
            "gini": baseline_gini,
            "marginal_gini": 0.0,
            "n_samples": n_samples,
            "n_pos": np.sum(y == 1),
            "n_neg": np.sum(y == 0),
        }
    )

    # Track cumulative score (sum of all scores added so far)
    cumulative_score = np.zeros(n_samples)
    used_scores = []

    step = 1
    remaining_scores = list(score_dict.keys())

    while remaining_scores:
        best_score_name = None
        best_marginal_gini = -np.inf
        best_gini = -np.inf

        # Try each remaining score
        for score_name in remaining_scores:
            scores = score_dict[score_name]
            mask = availability_mask[score_name]

            # Only consider samples where this score is available
            # and where we have both positive and negative cases
            valid_mask = mask & ~np.isnan(scores)
            valid_y = y[valid_mask]
            valid_scores = scores[valid_mask]

            if len(valid_y) == 0 or len(np.unique(valid_y)) < 2:
                continue

            # Calculate Gini for this score alone
            pos_scores = valid_scores[valid_y == 1]
            neg_scores = valid_scores[valid_y == 0]

            if len(pos_scores) == 0 or len(neg_scores) == 0:
                continue

            score_gini = somersd_pairwise(pos_scores, neg_scores, ties=ties)
            if score_gini is None:
                continue

            # Calculate marginal contribution: Gini of combined score vs cumulative
            # Create combined score (simple average for now, could be weighted)
            combined_score = cumulative_score.copy()
            combined_score[valid_mask] += valid_scores

            # Normalize combined score for samples with this score
            n_used = len(used_scores) + 1
            combined_score[valid_mask] = combined_score[valid_mask] / n_used

            # Calculate Gini of combined score
            combined_pos = combined_score[valid_mask & (y == 1)]
            combined_neg = combined_score[valid_mask & (y == 0)]

            if len(combined_pos) > 0 and len(combined_neg) > 0:
                combined_gini = somersd_pairwise(combined_pos, combined_neg, ties=ties)
                if combined_gini is not None:
                    marginal = combined_gini - results[-1]["gini"]
                    if combined_gini > best_gini:
                        best_score_name = score_name
                        best_marginal_gini = marginal
                        best_gini = combined_gini

        if best_score_name is None:
            break

        # Add best score
        scores = score_dict[best_score_name]
        mask = availability_mask[best_score_name]
        valid_mask = mask & ~np.isnan(scores)

        # Update cumulative score
        cumulative_score[valid_mask] += scores[valid_mask]
        used_scores.append(best_score_name)
        remaining_scores.remove(best_score_name)

        # Calculate final stats
        valid_y = y[valid_mask]
        valid_scores = cumulative_score[valid_mask] / len(used_scores)

        pos_scores = valid_scores[valid_y == 1]
        neg_scores = valid_scores[valid_y == 0]

        results.append(
            {
                "step": step,
                "score_name": best_score_name,
                "gini": best_gini,
                "marginal_gini": best_marginal_gini,
                "n_samples": np.sum(valid_mask),
                "n_pos": np.sum(valid_y == 1),
                "n_neg": np.sum(valid_y == 0),
            }
        )

        step += 1

    return pd.DataFrame(results)
