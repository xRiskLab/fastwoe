"""fastwoe.py."""

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid  # pylint: disable=no-name-in-module
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder
from sklearn.tree import DecisionTreeClassifier

from .fast_somersd import somersd_yx


class WoePreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocess high-cardinality categorical features for stable WOE encoding.
    Controls cardinality by keeping top categories and grouping rare ones.
    """

    def __init__(
        self, max_categories=None, top_p=0.95, min_count=10, other_token="__other__"
    ):
        """
        Parameters
        ----------
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

    # pylint: disable=invalid-name, unused-argument
    def fit(self, X: pd.DataFrame, y=None, cat_features: Union[list[str], None] = None):
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
    binner_kwargs : dict, optional
        Additional keyword arguments for KBinsDiscretizer (when binning_method="kbins").
    warn_on_numerical : bool, default=True
        Whether to warn when numerical features are automatically binned.
    numerical_threshold : int, default=20
        Minimum number of unique values to trigger binning for numerical features.
    binning_method : str, default="kbins"
        Method for binning numerical features. Options:
        - "kbins": Use KBinsDiscretizer (default)
        - "tree": Use decision tree-based binning
    tree_estimator : estimator object, optional
        Custom tree estimator for binning (when binning_method="tree").
        Must implement fit() and have a tree_ attribute. Default: DecisionTreeClassifier.
    tree_kwargs : dict, optional
        Additional keyword arguments for the tree estimator.

    Attributes:
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

    def __init__(
        self,
        encoder_kwargs=None,
        random_state=42,
        binner_kwargs=None,
        warn_on_numerical=True,
        numerical_threshold=20,
        binning_method="kbins",
        tree_estimator=None,
        tree_kwargs=None,
    ):
        # Enforce binary target type for TargetEncoder
        default_kwargs = {"smooth": 1e-5, "target_type": "binary"}
        if encoder_kwargs is None:
            self.encoder_kwargs = default_kwargs
        else:
            # Merge user kwargs with defaults, enforcing target_type
            self.encoder_kwargs = {
                **default_kwargs,
                **encoder_kwargs,
                "target_type": "binary",
            }

        # Check if user tried to set multiclass
        if encoder_kwargs and encoder_kwargs.get("target_type") == "multiclass":
            raise NotImplementedError(
                "FastWoe currently only supports binary classification. "
                "Multiclass WOE support will be available in future versions."
            )

        # Numerical binning configuration
        default_binner_kwargs = {
            "n_bins": 5,
            "strategy": "quantile",
            "encode": "ordinal",
        }
        if binner_kwargs is None:
            self.binner_kwargs = default_binner_kwargs
        else:
            self.binner_kwargs = {**default_binner_kwargs, **binner_kwargs}

        # Binning method configuration
        if binning_method not in ["kbins", "tree"]:
            raise ValueError("binning_method must be 'kbins' or 'tree'")
        self.binning_method = binning_method

        # Tree estimator configuration
        if tree_estimator is None:
            self.tree_estimator = DecisionTreeClassifier
        else:
            self.tree_estimator = tree_estimator

        # Tree parameters
        default_tree_kwargs = {
            "max_depth": 3,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "random_state": random_state,
        }
        if tree_kwargs is None:
            self.tree_kwargs = default_tree_kwargs
        else:
            self.tree_kwargs = {**default_tree_kwargs, **tree_kwargs}

        self.warn_on_numerical = warn_on_numerical
        self.numerical_threshold = (
            numerical_threshold  # Apply binning if unique values >= threshold
        )

        self.random_state = random_state
        self.encoders_ = {}
        self.mappings_ = {}
        self.feature_stats_ = {}
        self.binners_ = {}  # Store fitted binners for numerical features
        self.binning_info_ = {}  # Store binning summary info
        self.y_prior_ = None
        self.is_fitted_ = False

    def _calculate_gini(self, y_true, y_pred):
        """Calculate Gini coefficient from AUC."""
        try:
            auc = roc_auc_score(y_true, y_pred)
            return 2 * auc - 1
        except ValueError:
            return somersd_yx(y_true, y_pred).statistic

    def _calculate_woe_se(self, good_count: int, bad_count: int) -> float:
        """
        Calculate standard error of WOE using actual counts.

        If either count is zero, estimate using a conservative rule-of-thumb
        for binomial log-odds variance.

        Parameters
        ----------
        good_count : int
            Number of non-events (y=0) in the category.
        bad_count : int
            Number of events (y=1) in the category.

        Returns:
        -------
        float
            Standard error of the WOE value.
        """
        n = good_count + bad_count

        if good_count <= 0 or bad_count <= 0:
            p = 3.0 / n if bad_count <= 0 else 1.0 - (3.0 / n)
            variance = 1.0 if p <= 0 or p >= 1 or n <= 0 else 1.0 / (p * (1.0 - p) * n)
        else:
            variance = 1.0 / good_count + 1.0 / bad_count

        return np.sqrt(variance)

    def _calculate_woe_ci(self, woe_value, se_value, alpha=0.05):
        """
        Calculate confidence interval for WOE value.

        Parameters
        ----------
        woe_value : float
            The WOE value
        se_value : float
            Standard error of the WOE value
        alpha : float, default=0.05
            Significance level (0.05 for 95% CI)

        Returns:
        -------
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

        # Calculate WOE values directly to avoid circular dependency during fit
        odds_prior = self.y_prior_ / (1 - self.y_prior_)
        enc = self.encoders_[col]
        event_rate = enc.transform(X[[col]])
        if isinstance(event_rate, pd.DataFrame):
            event_rate = event_rate.values.flatten()
        event_rate = np.clip(event_rate, 1e-15, 1 - 1e-15)
        odds_cat = event_rate / (1 - event_rate)
        woe_values = np.log(odds_cat / odds_prior)

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

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """
        Fit the FastWoe encoder to features (both categorical and numerical).

        This method:
        1. Detects numerical features and applies automatic binning if needed
        2. Fits a TargetEncoder for each feature (categorical or binned numerical)
        3. Calculates WOE values and standard errors for each category
        4. Computes confidence intervals using normal approximation
        5. Generates comprehensive feature statistics (Gini, IV, etc.)

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Input features to encode. If numpy array, will be converted to DataFrame with generic column names.
        y : Union[pd.Series, np.ndarray]
            Binary target variable (0/1). If numpy array, will be converted to Series.

        Returns:
        -------
        self : FastWoe
            The fitted encoder instance
        """
        # Convert numpy arrays to pandas if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
            warnings.warn(
                "Input X is a numpy array. Converting to pandas DataFrame with generic column names. "
                "For better control, convert to DataFrame with meaningful column names before passing to fit().",
                stacklevel=2,
            )

        if isinstance(y, np.ndarray):
            y = pd.Series(y)
            warnings.warn(
                "Input y is a numpy array. Converting to pandas Series. "
                "For better control, convert to Series before passing to fit().",
                stacklevel=2,
            )

        # Validate target is binary
        unique_targets = pd.Series(y).nunique()
        unique_values = sorted(pd.Series(y).unique())

        if unique_targets > 2:
            raise ValueError(
                "FastWoe only supports binary targets. "
                f"Found {unique_targets} unique values: {unique_values}. "
                "Multiclass support will be available in a future release."
            )
        elif unique_targets == 1:
            raise ValueError(
                f"Target variable must have exactly 2 classes. "
                f"Found only 1 unique value: {unique_values}"
            )

        # Ensure target values are 0 and 1
        if not set(unique_values).issubset({0, 1}):
            raise ValueError(
                f"Target variable must contain only 0 and 1. Found values: {unique_values}"
            )

        # Detect numerical features that need binning
        numerical_features = self._detect_numerical_features(X)

        # Warn user about automatic binning if enabled
        if numerical_features and self.warn_on_numerical:
            if self.binning_method == "kbins":
                warnings.warn(
                    f"Detected numerical features: {numerical_features}. "
                    f"Applying automatic binning with {self.binner_kwargs['n_bins']} bins using "
                    f"'{self.binner_kwargs['strategy']}' strategy. Use get_binning_summary() to view details.",
                    UserWarning,
                    stacklevel=2,
                )
            else:  # tree method
                warnings.warn(
                    f"Detected numerical features: {numerical_features}. "
                    f"Applying decision tree-based binning with max_depth={self.tree_kwargs['max_depth']}. "
                    f"Use get_binning_summary() to view details.",
                    UserWarning,
                    stacklevel=2,
                )

        # Apply binning to numerical features
        X_processed = X.copy()
        for col in numerical_features:
            X_processed[[col]] = self._bin_numerical_feature(X_processed, col, y)

        self.y_prior_ = y.mean()
        odds_prior = self.y_prior_ / (1 - self.y_prior_)
        self.encoders_ = {}
        self.mappings_ = {}
        self.feature_stats_ = {}

        for col in X_processed.columns:
            enc = TargetEncoder(**self.encoder_kwargs, random_state=self.random_state)
            enc.fit(X_processed[[col]], y)
            self.encoders_[col] = enc

            # Get unique categories as seen by the encoder
            categories = enc.categories_[0]
            event_rates = enc.encodings_[0]

            # Defensive clipping to avoid log(0)
            event_rates = np.clip(event_rates, 1e-15, 1 - 1e-15)
            odds_cat = event_rates / (1 - event_rates)
            woe = np.log(odds_cat / odds_prior)

            # Count for each category in training data
            value_counts = X_processed[col].value_counts(dropna=False)
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
                    "count_pct": count / len(X_processed) * 100,
                    "good_count": good_counts,
                    "bad_count": bad_counts,
                    "event_rate": event_rates,
                    "woe": woe,
                    "woe_se": woe_se,
                    "woe_ci_lower": woe_ci_lower,
                    "woe_ci_upper": woe_ci_upper,
                }
            ).set_index("category")

            self.mappings_[col] = mapping_df

            # Calculate feature-level statistics
            self.feature_stats_[col] = self._calculate_feature_stats(
                col, X_processed, y, mapping_df
            )

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features to Weight of Evidence (WOE) values.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame with features to transform

        Returns:
        -------
        pd.DataFrame
            DataFrame with same shape as input, but with WOE values
            instead of categorical values. Column names are preserved.

        """
        odds_prior = self.y_prior_ / (1 - self.y_prior_)
        woe_df = pd.DataFrame(index=X.index)

        # Apply binning to numerical features if they were binned during fit
        X_processed = X.copy()
        for col in X.columns:
            if col in self.binners_:
                # Apply the same binning as during fit
                binner = self.binners_[col]
                binning_info = self.binning_info_[col]
                X_col = X_processed[[col]].copy()
                mask_missing = X_col[col].isna()

                if not mask_missing.all():  # If there are any non-missing values
                    # Ensure column is object type to accept strings
                    X_processed[col] = X_processed[col].astype("object")

                    if binning_info.get("method") == "kbins":
                        # KBinsDiscretizer method
                        X_processed.loc[~mask_missing, col] = binner.transform(
                            X_col[~mask_missing]
                        ).ravel()
                    else:  # tree method
                        # Tree method - use bin edges for binning
                        bin_edges = np.array(binning_info["bin_edges"])
                        binned_values = np.digitize(
                            X_col[~mask_missing][col].values,
                            bin_edges[1:-1],
                            right=False,
                        )
                        binned_values = np.clip(
                            binned_values - 1, 0, len(bin_edges) - 2
                        )
                        X_processed.loc[~mask_missing, col] = binned_values

                # Convert to string categories with meaningful labels (same as fit)
                if binning_info.get("method") == "kbins" and hasattr(
                    binner, "bin_edges_"
                ):
                    edges = binner.bin_edges_[0]
                elif (
                    binning_info.get("method") == "tree" and "bin_edges" in binning_info
                ):
                    edges = np.array(binning_info["bin_edges"])
                else:
                    edges = None

                if edges is not None:
                    bin_labels = []
                    for i in range(len(edges) - 1):
                        if i == 0:
                            label = f"(-∞, {edges[i + 1]:.1f}]"
                        elif i == len(edges) - 2:
                            label = f"({edges[i]:.1f}, ∞)"
                        else:
                            label = f"({edges[i]:.1f}, {edges[i + 1]:.1f}]"
                        bin_labels.append(label)

                    # Map ordinal values to labels
                    non_missing_values = X_processed.loc[~mask_missing, col]
                    if len(non_missing_values) > 0:
                        X_processed.loc[~mask_missing, col] = (
                            non_missing_values.astype(int)
                            .map(dict(enumerate(bin_labels)))
                            .astype(str)
                        )

                # Handle missing values
                if mask_missing.any():
                    X_processed.loc[mask_missing, col] = "Missing"

        for col in X_processed.columns:
            enc = self.encoders_[col]
            event_rate = enc.transform(X_processed[[col]])
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

    def get_mapping(self, feature: str) -> pd.DataFrame:
        """
        Get the WOE mapping DataFrame for a specific feature, preserving correct bin order for binned features.
        """
        if not self.is_fitted_:
            raise ValueError("FastWoe must be fitted before getting mappings")
        if feature not in self.mappings_:
            raise ValueError(f"Feature '{feature}' not found in fitted features")
        mapping = self.mappings_[feature].copy()
        # For binned numerical features, reindex by bin_labels
        if feature in getattr(self, "binners_", {}):
            binning_info = self.binning_info_[feature]

            if binning_info.get("method") == "kbins" and hasattr(
                self.binners_[feature], "bin_edges_"
            ):
                binner = self.binners_[feature]
                edges = binner.bin_edges_[0]
                bin_labels = []
                for i in range(len(edges) - 1):
                    if i == 0:
                        label = f"(-∞, {edges[i + 1]:.1f}]"
                    elif i == len(edges) - 2:
                        label = f"({edges[i]:.1f}, ∞)"
                    else:
                        label = f"({edges[i]:.1f}, {edges[i + 1]:.1f}]"
                    bin_labels.append(label)
                mapping = mapping.reindex(bin_labels)
            elif binning_info.get("method") == "tree" and "bin_edges" in binning_info:
                edges = np.array(binning_info["bin_edges"])
                bin_labels = []
                for i in range(len(edges) - 1):
                    if i == 0:
                        label = f"(-∞, {edges[i + 1]:.1f}]"
                    elif i == len(edges) - 2:
                        label = f"({edges[i]:.1f}, ∞)"
                    else:
                        label = f"({edges[i]:.1f}, {edges[i + 1]:.1f}]"
                    bin_labels.append(label)
                mapping = mapping.reindex(bin_labels)
        return mapping.reset_index()

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

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict probabilities using WOE-transformed features.
        """
        # Handle numpy array input
        if isinstance(X, np.ndarray):
            warnings.warn(
                "Input X is a numpy array. Converting to pandas DataFrame with generic column names. "
                "For better control, convert to DataFrame with meaningful column names before passing to predict_proba().",
                UserWarning,
                stacklevel=2,
            )
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        X_woe = self.transform(X)
        odds_prior = self.y_prior_ / (1 - self.y_prior_)
        woe_score = X_woe.sum(axis=1) + np.log(odds_prior)

        # Convert to probability (simple sigmoid transformation)
        prob = sigmoid(woe_score)
        return np.column_stack([1 - prob, prob])

    def predict_ci(self, X: Union[pd.DataFrame, np.ndarray], alpha=0.05) -> np.ndarray:
        """
        Predict confidence intervals for WOE values and probabilities.

        Simple approach: For each category, add/subtract WOE standard error
        to get confidence bounds, then transform to probability scale.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input features to predict confidence intervals for
        alpha : float, default=0.05
            Significance level (0.05 for 95% CI)

        Returns:
        -------
        np.ndarray
            Array with shape (n_samples, 2) containing:
            - Column 0: Lower confidence bound
            - Column 1: Upper confidence bound

        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before calling predict_ci")

        # Handle numpy array input
        if isinstance(X, np.ndarray):
            warnings.warn(
                "Input X is a numpy array. Converting to pandas DataFrame with generic column names. "
                "For better control, convert to DataFrame with meaningful column names before passing to predict_ci().",
                UserWarning,
                stacklevel=2,
            )
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

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

        # Return numpy array with shape (n_samples, 2)
        return np.column_stack([prob_lower, prob_upper])

    def transform_standardized(
        self, X: pd.DataFrame, output="woe", col_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Transform features to standardized WOE scores or Wald statistics.

        Parameters
        ----------
        X : pd.DataFrame
            Input features to transform
        output : str, default="woe"
            Type of output to return:
            - "woe": Return standardized WOE scores (WOE / SE) for each feature
            - "wald": Return overall Wald statistic (sum of squared z-scores)
        col_name : str, optional
            If specified, only return results for this column

        Returns:
        -------
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

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict binary outcomes using WOE-transformed features.

        Predicts class 1 if WOE score > 0, class 0 otherwise.
        This is based on the WOE interpretation where 0 represents
        the center (average log odds).
        """
        # Handle numpy array input
        if isinstance(X, np.ndarray):
            warnings.warn(
                "Input X is a numpy array. Converting to pandas DataFrame with generic column names. "
                "For better control, convert to DataFrame with meaningful column names before passing to predict().",
                UserWarning,
                stacklevel=2,
            )
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        woe_score = self.transform(X).sum(axis=1)
        return (woe_score > 0).astype(int).values

    def _detect_numerical_features(self, X: pd.DataFrame) -> list[str]:
        """
        Detect numerical features that should be binned.

        Returns list of column names that are:
        1. Numerical (using numbers.Number)
        2. Have >= numerical_threshold unique values
        """
        numerical_features = []

        numerical_features.extend(
            col
            for col in X.columns
            if pd.api.types.is_numeric_dtype(X[col])
            and X[col].nunique() >= self.numerical_threshold
        )
        return numerical_features

    def _bin_numerical_feature(
        self, X: pd.DataFrame, col: str, y: pd.Series
    ) -> pd.DataFrame:
        """
        Apply binning to a numerical feature and return the binned data.
        Supports both KBinsDiscretizer and decision tree-based binning.
        """
        # Handle missing values
        X_col = X[[col]].copy()
        mask_missing = X_col[col].isna()

        if mask_missing.any():
            # Check if we have enough non-missing values
            X_fit = X_col[~mask_missing]
            if len(X_fit) == 0:
                raise ValueError(
                    f"Column '{col}' has no non-missing values for binning"
                )
        else:
            X_fit = X_col

        if self.binning_method == "kbins":
            return self._bin_with_kbins(X_col, col, mask_missing, X_fit)
        else:  # tree method
            return self._bin_with_tree(X_col, col, y, mask_missing, X_fit)

    def _bin_with_kbins(
        self,
        X_col: pd.DataFrame,
        col: str,
        mask_missing: pd.Series,
        X_fit: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply KBinsDiscretizer binning to a numerical feature."""
        # Create and fit the binner
        binner_kwargs = self.binner_kwargs.copy()

        # Set quantile_method to avoid sklearn future warning
        if "quantile_method" not in binner_kwargs:
            binner_kwargs["quantile_method"] = "averaged_inverted_cdf"

        binner = KBinsDiscretizer(random_state=self.random_state, **binner_kwargs)
        binner.fit(X_fit)

        # Transform all values
        X_binned = X_col.copy()
        if not mask_missing.all():  # If there are any non-missing values
            # Ensure column is object type to accept strings
            X_binned[col] = X_binned[col].astype("object")
            X_binned.loc[~mask_missing, col] = binner.transform(X_fit).ravel()

        # Convert to string categories with meaningful labels
        if hasattr(binner, "bin_edges_"):
            edges = binner.bin_edges_[0]
            bin_labels = []
            for i in range(len(edges) - 1):
                if i == 0:
                    label = f"(-∞, {edges[i + 1]:.1f}]"
                elif i == len(edges) - 2:
                    label = f"({edges[i]:.1f}, ∞)"
                else:
                    label = f"({edges[i]:.1f}, {edges[i + 1]:.1f}]"
                bin_labels.append(label)

            # Map ordinal values to labels for non-missing values only
            non_missing_values = X_binned.loc[~mask_missing, col]
            if len(non_missing_values) > 0:
                X_binned.loc[~mask_missing, col] = (
                    non_missing_values.astype(int)
                    .map(dict(enumerate(bin_labels)))
                    .astype(str)
                )

        # Handle missing values
        if mask_missing.any():
            X_binned.loc[mask_missing, col] = "Missing"

        # Store binner and info
        self.binners_[col] = binner
        self.binning_info_[col] = {
            "values": X_col[col].nunique(),
            "n_bins": len(bin_labels)
            if hasattr(binner, "bin_edges_")
            else binner.n_bins,
            "missing": mask_missing.sum(),
            "bin_edges": edges.tolist() if hasattr(binner, "bin_edges_") else None,
            "method": "kbins",
        }

        return X_binned

    def _bin_with_tree(
        self,
        X_col: pd.DataFrame,
        col: str,
        y: pd.Series,
        mask_missing: pd.Series,
        X_fit: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply decision tree-based binning to a numerical feature."""
        # Create and fit the tree
        tree_kwargs = self.tree_kwargs.copy()
        tree = self.tree_estimator(**tree_kwargs)

        # Fit tree on non-missing data
        y_fit = y[~mask_missing] if mask_missing.any() else y
        tree.fit(X_fit, y_fit)

        # Extract split points from the tree
        split_points = self._extract_tree_splits(tree, col)

        # Create bin edges from split points
        bin_edges = self._create_bin_edges_from_splits(split_points, X_fit[col].values)

        # Create bin labels
        bin_labels = []
        for i in range(len(bin_edges) - 1):
            if i == 0:
                label = f"(-∞, {bin_edges[i + 1]:.1f}]"
            elif i == len(bin_edges) - 2:
                label = f"({bin_edges[i]:.1f}, ∞)"
            else:
                label = f"({bin_edges[i]:.1f}, {bin_edges[i + 1]:.1f}]"
            bin_labels.append(label)

        # Apply binning
        X_binned = X_col.copy()
        if not mask_missing.all():  # If there are any non-missing values
            # Ensure column is object type to accept strings
            X_binned[col] = X_binned[col].astype("object")

            # Bin the non-missing values
            binned_values = np.digitize(X_fit[col].values, bin_edges[1:-1], right=False)
            # digitize returns 1-based indices, but we want 0-based
            binned_values = np.clip(binned_values - 1, 0, len(bin_labels) - 1)

            X_binned.loc[~mask_missing, col] = binned_values

            # Map ordinal values to labels
            non_missing_values = X_binned.loc[~mask_missing, col]
            if len(non_missing_values) > 0:
                X_binned.loc[~mask_missing, col] = (
                    non_missing_values.astype(int)
                    .map(dict(enumerate(bin_labels)))
                    .astype(str)
                )

        # Handle missing values
        if mask_missing.any():
            X_binned.loc[mask_missing, col] = "Missing"

        # Store tree and info
        self.binners_[col] = tree
        self.binning_info_[col] = {
            "values": X_col[col].nunique(),
            "n_bins": len(bin_labels),
            "missing": mask_missing.sum(),
            "bin_edges": bin_edges.tolist(),
            "method": "tree",
            "split_points": split_points.tolist() if len(split_points) > 0 else [],
        }

        return X_binned

    def _extract_tree_splits(self, tree, feature_name: str) -> np.ndarray:
        """Extract split points from a fitted decision tree."""
        if not hasattr(tree, "tree_"):
            raise ValueError("Tree estimator must have a tree_ attribute")

        tree_obj = tree.tree_
        split_points = []

        def extract_splits_recursive(node_id, depth=0):
            if tree_obj.children_left[node_id] != tree_obj.children_right[node_id]:
                # This is a split node
                feature_idx = tree_obj.feature[node_id]
                threshold = tree_obj.threshold[node_id]

                # Only consider splits for our feature (should be 0 for single feature)
                if feature_idx == 0:  # Assuming single feature
                    split_points.append(threshold)

                # Recursively process children
                extract_splits_recursive(tree_obj.children_left[node_id], depth + 1)
                extract_splits_recursive(tree_obj.children_right[node_id], depth + 1)

        extract_splits_recursive(0)
        return np.array(sorted(split_points))

    def _create_bin_edges_from_splits(
        self, split_points: np.ndarray, data: np.ndarray
    ) -> np.ndarray:
        """Create bin edges from split points, including min and max boundaries."""
        if len(split_points) == 0:
            # No splits found, create single bin
            return np.array([-np.inf, np.inf])

        min_val = np.min(data)
        max_val = np.max(data)

        return np.concatenate([[-np.inf], split_points, [np.inf]])

    def get_binning_summary(self) -> pd.DataFrame:
        """
        Get summary of binning applied to numerical features.

        Returns:
        -------
        pd.DataFrame
            Summary with columns: feature, values, n_bins, missing, method

        """
        if not self.binning_info_:
            return pd.DataFrame()

        return pd.DataFrame(
            [
                {
                    "feature": col,
                    "values": info["values"],
                    "n_bins": info["n_bins"],
                    "missing": info["missing"],
                    "method": info.get("method", "unknown"),
                }
                for col, info in self.binning_info_.items()
            ]
        )

    def get_split_value_histogram(
        self, feature: str, as_array: bool = True
    ) -> Union[np.ndarray, list]:
        """
        Get the actual split values (bin edges) for a numerical binned feature.

        This function returns the exact numerical thresholds used to create
        the bins for a numerical feature, rather than string representations.

        Parameters
        ----------
        feature : str
            Name of the numerical feature to get split values for
        as_array : bool, default=True
            If True, return as numpy array. If False, return as list.

        Returns:
        -------
        np.ndarray or list
            Array/list of split values (bin edges) used for binning.
            For n bins, returns n+1 edges (including min and max boundaries).
        """
        if not self.is_fitted_:
            raise ValueError(
                "FastWoe must be fitted before calling get_split_value_histogram"
            )

        if feature not in self.mappings_:
            raise ValueError(f"Feature '{feature}' not found in fitted features")

        if feature not in self.binners_:
            raise ValueError(
                f"Feature '{feature}' is not a binned numerical feature. "
                f"This function only works with numerical features that were automatically binned."
            )

        # Get the binning info for this feature
        binning_info = self.binning_info_[feature]

        if binning_info.get("method") == "kbins":
            # For KBinsDiscretizer, get edges from the binner
            binner = self.binners_[feature]
            if not hasattr(binner, "bin_edges_"):
                raise ValueError(
                    f"Unable to extract bin edges for feature '{feature}'. "
                    f"The binner may not support edge extraction."
                )
            edges = binner.bin_edges_[0].copy()
        elif binning_info.get("method") == "tree" and "bin_edges" in binning_info:
            # For tree method, get edges from binning_info
            edges = np.array(binning_info["bin_edges"])
        else:
            raise ValueError(
                f"Unable to extract bin edges for feature '{feature}'. "
                f"Binning info may be incomplete or method not supported."
            )

        # Replace first and last edges with -inf and +inf
        edges[0] = -np.inf
        edges[-1] = np.inf

        return edges if as_array else edges.tolist()
