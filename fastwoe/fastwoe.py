"""fastwoe.py."""

from typing import Optional, Union, List

import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder


class WoePreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocess high-cardinality categorical features for stable WOE encoding.
    Controls cardinality by keeping top categories and grouping rare ones.
    """

    def __init__(
        self, max_categories=None, top_p=0.95, min_count=10, other_token="__other__"
    ):
        """
        Parameters:
        -----------
        max_categories : int, optional
            Maximum number of categories to keep per feature
        top_p : float, default=0.95
            Keep categories that cover top_p% of cumulative frequency
        min_count : int, default=10
            Minimum count required for a category to be kept
        other_token : str, default="__other__"
            Token for grouping rare categories
        """
        assert max_categories or top_p, "Set either `max_categories` or `top_p`"
        self.max_categories = max_categories
        self.top_p = top_p
        self.min_count = min_count
        self.other_token = other_token
        self.category_maps = {}
        self.cat_features_ = None

    def fit(self, X: pd.DataFrame, y=None, cat_features: Union[List[str], None] = None):  # pylint: disable=invalid-name
        """Fit the preprocessor to identify top categories."""
        self.cat_features_ = (
            cat_features
            or X.select_dtypes(include=["object", "category"]).columns.tolist()
        )

        for col in self.cat_features_:
            vc = (
                X[col]
                .astype(str)
                .value_counts(dropna=False)
                .sort_values(ascending=False)
            )
            original_cats = len(vc)

            # Skip filtering if the number of categories is ≤ 2
            if original_cats <= 2:
                self.category_maps[col] = set(vc.index.tolist())
                continue

            # Apply min_count filter
            vc_filtered = vc[vc >= self.min_count]

            # Fallback: if ALL categories are below min_count, keep the most frequent
            if vc_filtered.empty:
                top_cats = [vc.idxmax()]
            elif self.top_p is not None:
                # Calculate cumulative as percentage of ORIGINAL total
                cumulative = vc_filtered.cumsum() / vc.sum()
                top_cats = cumulative[cumulative <= self.top_p].index.tolist() or [
                    vc_filtered.idxmax()
                ]
            else:
                top_cats = vc_filtered.nlargest(self.max_categories).index.tolist()

            self.category_maps[col] = set(top_cats)

        return self

    # pylint: disable=invalid-name
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by replacing rare categories with other_token."""
        X_ = X.copy()
        for col in self.cat_features_:
            if col in X_.columns:
                allowed = self.category_maps[col]
                X_[col] = (
                    X_[col]
                    .astype(str)
                    .apply(
                        lambda x, allowed=allowed: x
                        if x in allowed
                        else self.other_token
                    )
                )
        return X_

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        """Fit and transform in one step."""
        cat_features = fit_params.get("cat_features", None)
        return self.fit(X, y, cat_features).transform(X)

    def get_category_mapping(self) -> dict:
        """Get mapping of kept categories per feature."""
        return self.category_maps

    def get_reduction_summary(self, X: pd.DataFrame) -> pd.DataFrame:  # pylint: disable=invalid-name
        """Get summary of cardinality reduction per feature."""
        summary = []
        for col in self.cat_features_:
            if col in X.columns:
                original_cats = X[col].nunique()
                kept_cats = len(self.category_maps[col])
                summary.append(
                    {
                        "feature": col,
                        "original_categories": original_cats,
                        "kept_categories": kept_cats,
                        "reduction_pct": (1 - kept_cats / original_cats) * 100,
                    }
                )
        return pd.DataFrame(summary)


class FastWoe:  # pylint: disable=invalid-name
    """
    Fast Weight of Evidence (WOE) Encoder using scikit-learn's TargetEncoder.
    Stores mapping tables for each categorical feature, including:
    - Category value
    - Number of observations (count)
    - Average target (event rate)
    - WOE value
    - Feature-level statistics (Gini, IV, etc.)

    Parameters
    ----------
    encoder_kwargs : dict, optional
        Additional keyword arguments for scikit-learn TargetEncoder.
    random_state : int, optional
        Random state for reproducibility.

    Attributes
    ----------
    mappings_ : dict
        Per-feature mapping DataFrames with WOE info.
    encoders_ : dict
        Fitted TargetEncoder per feature.
    feature_stats_ : dict
        Per-feature statistics (Gini, IV, etc.).
    y_prior_ : float
        Mean target in fit data.
    """

    def __init__(self, encoder_kwargs=None, random_state=42):
        self.encoder_kwargs = encoder_kwargs or {"smooth": 1e-5}
        self.random_state = random_state
        self.encoders_ = {}
        self.mappings_ = {}
        self.feature_stats_ = {}
        self.y_prior_ = None
        self.is_fitted_ = False

    def _calculate_gini(self, y_true, y_pred):
        """Calculate Gini coefficient from AUC."""
        try:
            auc = roc_auc_score(y_true, y_pred)
            return 2 * auc - 1
        except ValueError:
            return np.nan  # Handle cases with single class

    def _calculate_woe_se(self, good_count, bad_count):
        """
        Calculate standard error of WOE using actual counts.

        Formula: SE(WOE) = sqrt(1/n_good + 1/n_bad)

        Parameters:
        -----------
        good_count : int
            Number of non-events (y=0) in the category
        bad_count : int
            Number of events (y=1) in the category

        Returns:
        --------
        float
            Standard error of the WOE value
        """
        # Avoid division by zero
        if good_count <= 0 or bad_count <= 0:
            return np.inf

        return np.sqrt(1.0 / good_count + 1.0 / bad_count)

    def _calculate_woe_ci(self, woe_value, se_value, alpha=0.05):
        """
        Calculate confidence interval for WOE value.

        Parameters:
        -----------
        woe_value : float
            The WOE value
        se_value : float
            Standard error of the WOE value
        alpha : float, default=0.05
            Significance level (0.05 for 95% CI)

        Returns:
        --------
        tuple
            (lower_bound, upper_bound) of the confidence interval
        """
        if np.isinf(se_value) or np.isnan(se_value):
            return (np.nan, np.nan)

        z_crit = norm.ppf(1 - alpha / 2)
        margin = z_crit * se_value

        return (woe_value - margin, woe_value + margin)

    def _calculate_iv(self, mapping_df, total_good, total_bad):
        """Calculate Information Value (IV) for a feature."""
        iv = 0
        for _, row in mapping_df.iterrows():
            bad_rate = (
                (row["count"] * row["event_rate"]) / total_bad if total_bad > 0 else 0
            )
            good_rate = (
                (row["count"] * (1 - row["event_rate"])) / total_good
                if total_good > 0
                else 0
            )

            if good_rate > 0 and bad_rate > 0:
                iv += (bad_rate - good_rate) * row["woe"]
        return iv

    def _calculate_feature_stats(self, col, X, y, mapping_df):
        """Calculate comprehensive statistics for a feature."""

        # Basic counts
        total_obs = len(y)
        total_bad = y.sum()
        total_good = total_obs - total_bad

        # Get WOE transformed values for this feature
        woe_values = self.transform(X[[col]])[col]  # Use original column name

        return {
            "feature": col,
            "n_categories": len(mapping_df),
            "total_observations": total_obs,
            "missing_count": X[col].isna().sum(),
            "missing_rate": X[col].isna().mean(),
            "gini": self._calculate_gini(y, woe_values),
            "iv": self._calculate_iv(mapping_df, total_good, total_bad),
            "min_woe": mapping_df["woe"].min(),
            "max_woe": mapping_df["woe"].max(),
        }

    # pylint: disable=invalid-name, too-many-locals
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the FastWoe encoder to categorical features.
        
        This method:
        1. Fits a TargetEncoder for each categorical feature
        2. Calculates WOE values and standard errors for each category
        3. Computes confidence intervals using normal approximation
        4. Generates comprehensive feature statistics (Gini, IV, etc.)
        
        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame with categorical features to encode
        y : pd.Series
            Binary target variable (0/1)
            
        Returns
        -------
        self : FastWoe
            The fitted encoder instance
        """
        self.y_prior_ = y.mean()
        odds_prior = self.y_prior_ / (1 - self.y_prior_)
        self.encoders_ = {}
        self.mappings_ = {}
        self.feature_stats_ = {}

        for col in X.columns:
            enc = TargetEncoder(**self.encoder_kwargs, random_state=self.random_state)
            enc.fit(X[[col]], y)
            self.encoders_[col] = enc

            # Get unique categories as seen by the encoder
            categories = enc.categories_[0]
            event_rates = enc.encodings_[0]

            # Defensive clipping to avoid log(0)
            event_rates = np.clip(event_rates, 1e-15, 1 - 1e-15)
            odds_cat = event_rates / (1 - event_rates)
            woe = np.log(odds_cat / odds_prior)

            # Count for each category in training data
            value_counts = X[col].value_counts(dropna=False)
            # Map in same order as categories_
            count = (
                pd.Series(value_counts).reindex(categories).fillna(0).astype(int).values
            )

            # Enhanced mapping with more details
            good_counts = np.round(count * (1 - event_rates)).astype(int)
            bad_counts = np.round(count * event_rates).astype(int)

            # Calculate WOE standard errors for each category
            woe_se = np.array(
                [
                    self._calculate_woe_se(good_count, bad_count)
                    for good_count, bad_count in zip(good_counts, bad_counts)
                ]
            )

            # Calculate 95% confidence intervals for WOE values
            woe_ci_lower = []
            woe_ci_upper = []
            for woe_val, se_val in zip(woe, woe_se):
                ci_lower, ci_upper = self._calculate_woe_ci(woe_val, se_val)
                woe_ci_lower.append(ci_lower)
                woe_ci_upper.append(ci_upper)

            mapping_df = pd.DataFrame(
                {
                    "category": categories,
                    "count": count,
                    "count_pct": count / len(X) * 100,
                    "event_rate": event_rates,
                    "woe": woe,
                    "woe_se": woe_se,
                    "woe_ci_lower": woe_ci_lower,
                    "woe_ci_upper": woe_ci_upper,
                    "good_count": good_counts,
                    "bad_count": bad_counts,
                }
            ).set_index("category")

            self.mappings_[col] = mapping_df

            # Calculate feature-level statistics
            self.feature_stats_[col] = self._calculate_feature_stats(
                col, X, y, mapping_df
            )

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical features to Weight of Evidence (WOE) values.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame with categorical features to transform
            
        Returns
        -------
        pd.DataFrame
            DataFrame with same shape as input, but with WOE values
            instead of categorical values. Column names are preserved.
        """
        odds_prior = self.y_prior_ / (1 - self.y_prior_)
        woe_df = pd.DataFrame(index=X.index)

        for col in X.columns:
            enc = self.encoders_[col]
            event_rate = enc.transform(X[[col]])
            # scikit-learn returns np.ndarray
            if isinstance(event_rate, pd.DataFrame):
                event_rate = event_rate.values.flatten()
            event_rate = np.clip(event_rate, 1e-15, 1 - 1e-15)
            odds_cat = event_rate / (1 - event_rate)
            woe = np.log(odds_cat / odds_prior)
            woe_df[col] = woe  # Keep original column name

        return woe_df

    def fit_transform(self, X: pd.DataFrame, y=None, **_fit_params) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_mapping(self, col: str) -> pd.DataFrame:
        """Return the mapping table for a feature (category, count, event_rate, woe)."""
        return self.mappings_[col].reset_index()

    def get_all_mappings(self) -> dict:
        """Get all mappings (useful for serialization, audit, or compact storage)."""
        return {col: mapping.reset_index() for col, mapping in self.mappings_.items()}

    def get_feature_stats(self, col: Optional[str] = None) -> pd.DataFrame:
        """Get feature statistics. If col is None, return stats for all features."""
        if col is not None:
            return pd.DataFrame([self.feature_stats_[col]])
        else:
            return pd.DataFrame(list(self.feature_stats_.values()))

    def get_feature_summary(self) -> pd.DataFrame:
        """Get a summary table of all features ranked by predictive power."""
        stats_df = self.get_feature_stats()
        return stats_df.sort_values("gini", ascending=False)[
            ["feature", "gini", "iv", "n_categories"]
        ].round(4)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities using WOE-transformed features.
        This is a simple linear combination - for real scoring you'd use logistic regression.
        """
        X_woe = self.transform(X)
        odds_prior = self.y_prior_ / (1 - self.y_prior_)
        woe_score = X_woe.sum(axis=1) + np.log(odds_prior)

        # Convert to probability (simple sigmoid transformation)
        prob = sigmoid(woe_score)
        return np.column_stack([1 - prob, prob])

    def predict_ci(self, X: pd.DataFrame, alpha=0.05) -> pd.DataFrame:
        """
        Predict confidence intervals for WOE values and probabilities.

        Simple approach: For each category, add/subtract WOE standard error
        to get confidence bounds, then transform to probability scale.

        Parameters:
        -----------
        X : pd.DataFrame
            Input features to predict confidence intervals for
        alpha : float, default=0.05
            Significance level (0.05 for 95% CI)

        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: prediction, ci_lower, ci_upper
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before calling predict_ci")

        # Get WOE-transformed features
        X_woe = self.transform(X)

        # Calculate WOE confidence intervals by looking up standard errors
        z_crit = norm.ppf(1 - alpha / 2)
        woe_ci_lower = np.zeros_like(X_woe.values)
        woe_ci_upper = np.zeros_like(X_woe.values)

        for i, col in enumerate(X.columns):
            if col in self.encoders_:
                mapping = self.mappings_[col]

                # For each row, find the WOE standard error
                for j in range(len(X)):
                    cat_value = X.iloc[j, i]

                    # Look up in mapping (handle unseen categories)
                    if cat_value in mapping.index:
                        woe_se = mapping.loc[cat_value, "woe_se"]
                    else:
                        # For unseen categories, use the average SE
                        woe_se = mapping["woe_se"].mean()

                    woe_val = X_woe.iloc[j, i]
                    margin = z_crit * woe_se

                    woe_ci_lower[j, i] = woe_val - margin
                    woe_ci_upper[j, i] = woe_val + margin

        # Sum WOE values across features for final score
        woe_score_lower = woe_ci_lower.sum(axis=1)
        woe_score_upper = woe_ci_upper.sum(axis=1)

        # Convert to probability scale
        odds_prior = self.y_prior_ / (1 - self.y_prior_)
        logit_lower = woe_score_lower + np.log(odds_prior)
        logit_upper = woe_score_upper + np.log(odds_prior)

        prob_lower = sigmoid(logit_lower)
        prob_upper = sigmoid(logit_upper)

        return pd.DataFrame(
            {
                "ci_lower": prob_lower,
                "ci_upper": prob_upper,
            },
            index=X.index,
        )

    def transform_standardized(
        self, X: pd.DataFrame, output="woe", col_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Transform features to standardized WOE scores or Wald statistics.

        Parameters:
        -----------
        X : pd.DataFrame
            Input features to transform
        output : str, default="woe"
            Type of output to return:
            - "woe": Return standardized WOE scores (WOE / SE) for each feature
            - "wald": Return overall Wald statistic (sum of squared z-scores)
        col_name : str, optional
            If specified, only return results for this column

        Returns:
        --------
        pd.DataFrame
            Standardized scores based on output parameter
        """
        if not self.is_fitted_:
            raise ValueError(
                "Model must be fitted before calling transform_standardized"
            )

        # Get WOE-transformed features
        X_woe = self.transform(X)

        # Filter to specific column if requested
        cols_to_process = (
            [col_name] if col_name and col_name in X.columns else X.columns
        )
        cols_to_process = [col for col in cols_to_process if col in self.encoders_]

        # Calculate standardized WOE scores (z-scores)
        z_scores = pd.DataFrame(index=X.index)

        for col in cols_to_process:
            mapping = self.mappings_[col]
            z_col = np.zeros(len(X))

            # For each row, calculate z-score
            for i in range(len(X)):
                cat_value = X.iloc[i][col]

                # Look up WOE and SE in mapping
                if cat_value in mapping.index:
                    woe_se = mapping.loc[cat_value, "woe_se"]
                else:
                    # For unseen categories, use average SE
                    woe_se = mapping["woe_se"].mean()

                woe_val = X_woe.iloc[i][col]
                if output == "woe":
                    z_col[i] = woe_val / woe_se if woe_se > 0 else 0
                elif output == "wald":
                    prior_log_odds = np.log(self.y_prior_ / (1 - self.y_prior_))
                    final_log_odds = woe_val + prior_log_odds
                    z_col[i] = final_log_odds / woe_se if woe_se > 0 else 0
                else:
                    raise ValueError("output must be 'woe' or 'wald'")

            z_scores[col] = z_col

            return z_scores
