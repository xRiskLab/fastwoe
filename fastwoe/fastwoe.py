"""fastwoe.py."""

import contextlib
import math
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid  # pylint: disable=no-name-in-module

if TYPE_CHECKING:

    @runtime_checkable
    class NormProtocol(Protocol):
        """Protocol for a normal distribution with a ppf method."""

        def ppf(self, q: float) -> float: ...

    # type checkers will think norm always matches this interface
    norm: NormProtocol
else:
    try:
        from scipy.stats import norm  # type: ignore
    except ImportError:

        class MockNorm:
            """Fallback normal distribution with rough ppf approximation."""

            def ppf(self, q: float) -> float:
                if q <= 0:
                    return float("-inf")
                if q >= 1:
                    return float("inf")
                if q == 0.5:
                    return 0.0
                # crude Box-Muller style approximation
                return (
                    math.sqrt(-2 * math.log(q))
                    if q < 0.5
                    else -math.sqrt(-2 * math.log(1 - q))
                )

        norm = MockNorm()

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

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
        self.category_maps: dict[str, set] = {}
        self.cat_features_: Optional[list[str]] = None

    # pylint: disable=invalid-name, unused-argument
    def fit(self, X: pd.DataFrame, y=None, cat_features: Union[list[str], None] = None):
        """Fit the preprocessor to identify top categories."""
        self.cat_features_ = (
            cat_features
            if cat_features is not None
            else X.select_dtypes(include=["object", "category"]).columns.tolist()
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
            # pyrefly: ignore  # missing-attribute
            if vc_filtered.empty:
                top_cats = [vc.idxmax()]
            elif self.top_p is not None:
                # Calculate cumulative as percentage of ORIGINAL total
                cumulative = vc_filtered.cumsum() / vc.sum()
                top_cats = cumulative[cumulative <= self.top_p].index.tolist() or [
                    # pyrefly: ignore  # missing-attribute
                    vc_filtered.idxmax()
                ]
            else:
                # pyrefly: ignore  # missing-attribute
                top_cats = vc_filtered.nlargest(self.max_categories).index.tolist()

            self.category_maps[col] = set(top_cats)

        return self

    # pylint: disable=invalid-name
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by replacing rare categories with other_token."""
        X_ = X.copy()
        # pyrefly: ignore  # not-iterable
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
        # pyrefly: ignore  # not-iterable
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
    warn_on_numerical : bool, default=False
        Whether to warn when numerical features are automatically binned.
    numerical_threshold : int, default=20
        Minimum number of unique values to trigger binning for numerical features.
    binning_method : str, default="tree"
        Method for binning numerical features. Options:
        - "tree": Use decision tree-based binning (default, optimized for credit scoring)
        - "kbins": Use KBinsDiscretizer
        - "faiss_kmeans": Use FAISS KMeans clustering for binning
    tree_estimator : estimator object, optional
        Custom tree estimator for binning (when binning_method="tree").
        Must implement fit() and have a tree_ attribute. Default: DecisionTreeClassifier.
    tree_kwargs : dict, optional
        Additional keyword arguments for the tree estimator.
    faiss_kwargs : dict, optional
        Additional keyword arguments for FAISS KMeans (when binning_method="faiss_kmeans").

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
        warn_on_numerical=False,
        numerical_threshold=20,
        binning_method="tree",  # Changed default to 'tree' for version 0.1.4
        tree_estimator=None,
        tree_kwargs=None,
        faiss_kwargs=None,
    ):
        # Set up encoder kwargs - will be updated in fit() based on target type
        default_kwargs: dict[str, Any] = {"smooth": 1e-5}
        if encoder_kwargs is None:
            self.encoder_kwargs: dict[str, Any] = default_kwargs
        else:
            # Merge user kwargs with defaults
            self.encoder_kwargs = {**default_kwargs, **encoder_kwargs}

        # Multiclass support: one-vs-rest WOE encoding for multiclass targets

        # Binning method configuration
        if binning_method not in ["kbins", "tree", "faiss_kmeans"]:
            raise ValueError(
                "binning_method must be 'kbins', 'tree', or 'faiss_kmeans'"
            )
        self.binning_method = binning_method

        # Tree estimator configuration
        # Will be set during fit() based on target type
        self.tree_estimator = tree_estimator

        # Unified binner kwargs - interpret based on binning_method
        self._setup_binner_kwargs(
            binning_method, binner_kwargs, tree_kwargs, faiss_kwargs, random_state
        )

        self.warn_on_numerical = warn_on_numerical
        self.numerical_threshold = (
            numerical_threshold  # Apply binning if unique values >= threshold
        )

        self.random_state = random_state
        self.encoders_: dict[str, Any] = {}
        self.mappings_: dict[str, pd.DataFrame] = {}
        self.feature_stats_: dict[str, dict[str, Any]] = {}
        self.binners_: dict[
            str, Any
        ] = {}  # Store fitted binners for numerical features
        self.binning_info_: dict[str, dict[str, Any]] = {}  # Store binning summary info
        self.y_prior_: Optional[Union[float, Dict[Union[int, str], float]]] = None
        self.is_fitted_: bool = False
        self.is_continuous_target: Optional[bool] = None
        self.is_binary_target: Optional[bool] = None

    def _setup_binner_kwargs(
        self, binning_method, binner_kwargs, tree_kwargs, faiss_kwargs, random_state
    ):
        """Setup binner kwargs based on the binning method, with backward compatibility."""
        if binning_method == "faiss_kmeans":
            # FAISS KMeans parameters
            default_kwargs = {
                "k": 5,
                "niter": 20,
                "verbose": False,
                "gpu": False,
            }
            # Use faiss_kwargs if provided, otherwise use binner_kwargs
            if faiss_kwargs is not None:
                self.faiss_kwargs = {**default_kwargs, **faiss_kwargs}
            elif binner_kwargs is not None:
                self.faiss_kwargs = {**default_kwargs, **binner_kwargs}
            else:
                self.faiss_kwargs = default_kwargs
            # For backward compatibility, also set binner_kwargs and tree_kwargs
            self.binner_kwargs = {}
            self.tree_kwargs = {}
        elif binning_method == "kbins":
            # KBinsDiscretizer parameters
            default_kwargs = {
                "n_bins": 5,
                "strategy": "quantile",
                "encode": "ordinal",
            }
            self.binner_kwargs = (
                default_kwargs
                if binner_kwargs is None
                else {**default_kwargs, **binner_kwargs}
            )
            # For backward compatibility, also set tree_kwargs and faiss_kwargs to empty
            self.tree_kwargs = {}
            self.faiss_kwargs = {}

        elif binning_method == "tree":
            # Tree parameters optimized for credit scoring
            default_kwargs = {
                "max_depth": 3,  # Optimal for credit scoring: 4-8 bins per feature
                "random_state": random_state,
            }
            # Use tree_kwargs if provided, otherwise use binner_kwargs
            if tree_kwargs is not None:
                self.tree_kwargs = {**default_kwargs, **tree_kwargs}
            elif binner_kwargs is not None:
                self.tree_kwargs = {**default_kwargs, **binner_kwargs}
            else:
                self.tree_kwargs = default_kwargs
            # For backward compatibility, also set binner_kwargs and faiss_kwargs
            self.binner_kwargs = {}
            self.faiss_kwargs = {}

    def _calculate_gini(self, y_true, y_pred):
        """Calculate Gini coefficient from AUC."""
        # Ensure inputs are 1D numpy arrays
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        # Check if arrays have the same length
        if len(y_true) != len(y_pred):
            # If lengths don't match, this might be per-category stats
            # Return NaN for now as this case needs different handling
            return np.nan

        # Remove any NaN values
        # pyrefly: ignore  # unsupported-operation
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if not mask.any():
            return np.nan

        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) < 2:
            return np.nan

        try:
            auc = roc_auc_score(y_true, y_pred)
            return 2 * auc - 1
        except ValueError:
            return somersd_yx(y_true, y_pred).statistic

    def _calculate_woe_se(self, good_count: int, bad_count: int) -> float:
        """
        Calculate standard error of WOE using actual counts.

        If either count is zero, estimate using a conservative rule-of-three
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

    def _calculate_iv_standard_error(self, mapping_df, total_good, total_bad):
        """
        Calculate standard error of Information Value using delta method.

        Mathematical Framework:
        ----------------------
        IV = Σ_j (bad_rate_j - good_rate_j) * WOE_j

        Using delta method:
        Var(IV) ≈ Σ_j (bad_rate_j - good_rate_j)² * Var(WOE_j)
                + Σ_j WOE_j² * Var(bad_rate_j - good_rate_j)

        Parameters
        ----------
        mapping_df : DataFrame
            Mapping table with WOE statistics
        total_good : int
            Total number of good observations
        total_bad : int
            Total number of bad observations

        Returns:
        -------
        float
            Standard error of IV
        """
        if total_good <= 0 or total_bad <= 0:
            return np.nan

        iv_variance = 0.0

        for _, row in mapping_df.iterrows():
            # Calculate bad and good rates for this bin
            bin_bad = row["count"] * row["event_rate"]
            bin_good = row["count"] * (1 - row["event_rate"])

            bad_rate = bin_bad / total_bad
            good_rate = bin_good / total_good

            # Weight in IV formula: (bad_rate - good_rate)
            iv_weight = bad_rate - good_rate

            # WOE standard error from mapping
            woe_se = row.get("woe_se", 0)
            # Delta method: Var(IV) ≈ Σ weight² * Var(WOE)
            iv_variance += (iv_weight**2) * (woe_se**2)

            # Add sampling variance for the rates
            if bin_bad > 0 and bin_good > 0:
                # Sampling variance of bad_rate - good_rate
                bad_rate_var = bad_rate * (1 - bad_rate) / total_bad
                good_rate_var = good_rate * (1 - good_rate) / total_good
                rate_diff_var = bad_rate_var + good_rate_var

                woe_value = row["woe"]

                # Add contribution: WOE² * Var(rate_diff)
                iv_variance += (woe_value**2) * rate_diff_var

        return np.sqrt(iv_variance)

    def _calculate_iv_confidence_interval(self, iv_value, iv_se, alpha=0.05):
        """
        Calculate confidence interval for IV.

        Parameters
        ----------
        iv_value : float
            Information Value
        iv_se : float
            Standard error of IV
        alpha : float, default=0.05
            Significance level (0.05 for 95% CI)

        Returns:
        -------
        tuple
            (lower_bound, upper_bound) of the confidence interval
        """
        if np.isnan(iv_se) or np.isinf(iv_se):
            return (np.nan, np.nan)

        z_crit = norm.ppf(1 - alpha / 2)
        margin = z_crit * iv_se

        # IV is always non-negative, so lower bound should be at least 0
        lower_bound = max(0, iv_value - margin)
        upper_bound = iv_value + margin

        return (lower_bound, upper_bound)

    def _calculate_feature_stats(self, col, X, y, mapping_df):
        # sourcery skip: extract-method, inline-immediately-returned-variable
        """Calculate comprehensive statistics for a feature."""
        # Basic counts
        total_obs = len(y)

        # Handle different y types (list, Series, array)
        if hasattr(y, "sum") and not isinstance(
            y.iloc[0] if hasattr(y, "iloc") else y[0], str
        ):
            total_bad = y.sum()
        else:
            # For multiclass or string labels, use the mapping to get total bad count
            total_bad = mapping_df["bad_count"].sum()

        total_good = total_obs - total_bad

        # Calculate WOE values directly to avoid circular dependency during fit
        if self.is_multiclass_target:
            # For multiclass, we need to pass the class-specific prior
            # This method is called with the original y, so we need to extract the class
            # from the mapping_df or use a different approach
            # For now, let's use the mapping_df directly
            woe_values = mapping_df["woe"].values
        else:
            if self.y_prior_ is None:
                raise ValueError("Model must be fitted before calculating WOE values")
            odds_prior = self.y_prior_ / (1 - self.y_prior_)
            enc = self.encoders_[col]
            event_rate = enc.transform(X[[col]])
            if isinstance(event_rate, pd.DataFrame):
                event_rate = event_rate.values.flatten()
            event_rate = np.clip(event_rate, 1e-15, 1 - 1e-15)
            odds_cat = event_rate / (1 - event_rate)  # pyrefly: ignore
            woe_values = np.log(odds_cat / odds_prior)

        # Calculate IV and its standard error
        iv_value = self._calculate_iv(mapping_df, total_good, total_bad)
        iv_se = self._calculate_iv_standard_error(mapping_df, total_good, total_bad)
        iv_ci_lower, iv_ci_upper = self._calculate_iv_confidence_interval(
            iv_value, iv_se
        )

        return {
            "feature": col,
            "n_categories": len(mapping_df),
            "total_observations": total_obs,
            "missing_count": X[col].isna().sum(),
            "missing_rate": X[col].isna().mean(),
            "gini": self._calculate_gini(y, woe_values),
            "iv": iv_value,
            "iv_se": iv_se,
            "iv_ci_lower": iv_ci_lower,
            "iv_ci_upper": iv_ci_upper,
            "min_woe": mapping_df["woe"].min(),
            "max_woe": mapping_df["woe"].max(),
        }

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray, pd.Series],
        y: Union[pd.Series, np.ndarray],
    ):
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
        X : Union[pd.DataFrame, np.ndarray, pd.Series]
            Input features to encode. If numpy array, will be converted to DataFrame with generic column names.
            If Series, will be converted to single-column DataFrame.
        y : Union[pd.Series, np.ndarray]
            Target variable. Supports:
            - Binary targets (0/1) for classification
            - Continuous proportions (0-1) for regression/aggregated data
            If numpy array, will be converted to Series.

        Returns:
        -------
        self : FastWoe
            The fitted encoder instance
        """
        # Convert numpy arrays and Series to pandas DataFrame if needed
        if isinstance(X, np.ndarray):
            column_names = [f"feature_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=column_names)  # type: ignore[arg-type]
            warnings.warn(
                "Input X is a numpy array. Converting to pandas DataFrame with generic column names. "
                "For better control, convert to DataFrame with meaningful column names before passing to fit().",
                stacklevel=2,
            )
        elif isinstance(X, pd.Series):
            # Convert Series to DataFrame with the Series name as column name
            column_name = X.name if X.name is not None else "feature_0"
            X = pd.DataFrame({column_name: X})
            warnings.warn(
                f"Input X is a pandas Series. Converting to DataFrame with column name '{column_name}'. "
                "For better control, convert to DataFrame before passing to fit().",
                stacklevel=2,
            )

        if isinstance(y, np.ndarray):
            y = pd.Series(y)
            warnings.warn(
                "Input y is a numpy array. Converting to pandas Series. "
                "For better control, convert to Series before passing to fit().",
                stacklevel=2,
            )

        # Validate target type
        unique_targets = pd.Series(y).nunique()
        unique_values = sorted(pd.Series(y).unique())
        y_series = pd.Series(y)

        # Check if target is continuous (proportions), binary, or multiclass
        is_continuous = (
            y_series.dtype in ["float32", "float64"]
            and (y_series >= 0).all()
            and (y_series <= 1).all()
            and unique_targets > 2
        )

        is_binary = unique_targets == 2 and set(unique_values).issubset({0, 1})

        # Multiclass: more than 2 unique values and not continuous proportions
        is_multiclass = unique_targets > 2 and not is_continuous

        if not (is_continuous or is_binary or is_multiclass):
            if unique_targets == 1:
                raise ValueError(
                    f"Target variable must have at least 2 unique values. "
                    f"Found only 1 unique value: {unique_values}"
                )
            elif unique_targets == 2 and not set(unique_values).issubset({0, 1}):
                raise ValueError(
                    f"Target variable must be binary (0/1), multiclass, or continuous proportions (0-1). "
                    f"Found binary values: {unique_values}. Expected 0/1 for binary or 0-1 for proportions."
                )

        # Store target type for later use in WOE calculation
        self.is_continuous_target = None
        self.is_binary_target = None
        self.is_multiclass_target = None

        self.is_continuous_target = is_continuous
        self.is_binary_target = is_binary
        self.is_multiclass_target = is_multiclass

        # Update encoder_kwargs based on target type
        if is_continuous:
            self.encoder_kwargs["target_type"] = "continuous"
        elif is_multiclass:
            self.encoder_kwargs["target_type"] = "multiclass"
        else:
            self.encoder_kwargs["target_type"] = "binary"

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
            elif self.binning_method == "tree":
                warnings.warn(
                    f"Detected numerical features: {numerical_features}. "
                    f"Applying decision tree-based binning with max_depth={self.tree_kwargs['max_depth']}. "
                    f"Use get_binning_summary() to view details.",
                    UserWarning,
                    stacklevel=2,
                )
            else:  # faiss_kmeans method
                warnings.warn(
                    f"Detected numerical features: {numerical_features}. "
                    f"Applying FAISS KMeans clustering with k={self.faiss_kwargs['k']} clusters. "
                    f"Use get_binning_summary() to view details.",
                    UserWarning,
                    stacklevel=2,
                )

        # Apply binning to numerical features
        X_processed = X.copy()
        for col in numerical_features:
            X_processed[col] = self._bin_numerical_feature(X_processed, col, y)[col]

        # Handle multiclass targets with one-vs-rest approach
        if self.is_multiclass_target:
            self.classes_ = sorted(y_series.unique())
            self.n_classes_ = len(self.classes_)
            self.y_prior_ = {}  # Dictionary of priors for each class
            self.odds_prior_per_class_ = {}

            # Calculate priors for each class
            for class_label in self.classes_:
                class_prior = float((y_series == class_label).mean())
                self.y_prior_[class_label] = class_prior
                self.odds_prior_per_class_[class_label] = class_prior / (
                    1 - class_prior
                )
        else:
            self.y_prior_ = float(y.mean())
            odds_prior = self.y_prior_ / (1 - self.y_prior_)

        self.encoders_ = {}
        self.mappings_ = {}
        self.feature_stats_ = {}

        for col in X_processed.columns:
            if self.is_multiclass_target:
                # For multiclass, create one encoder per class (one-vs-rest)
                self.encoders_[col] = {}
                self.mappings_[col] = {}
                self.feature_stats_[col] = {}

                for class_label in self.classes_:
                    # Create binary target: 1 for current class, 0 for all others
                    y_binary = (y_series == class_label).astype(int)

                    # Use binary target type for one-vs-rest
                    encoder_kwargs = self.encoder_kwargs.copy()
                    encoder_kwargs["target_type"] = "binary"

                    enc = TargetEncoder(
                        **encoder_kwargs, random_state=self.random_state
                    )
                    enc.fit(X_processed[[col]], y_binary)
                    self.encoders_[col][class_label] = enc

                    # Get unique categories as seen by the encoder
                    categories = enc.categories_[0]
                    event_rates = enc.encodings_[0]

                    # Defensive clipping to avoid log(0)
                    event_rates = np.clip(event_rates, 1e-15, 1 - 1e-15)
                    odds_cat = event_rates / (1 - event_rates)
                    woe = np.log(odds_cat / self.odds_prior_per_class_[class_label])

                    # Count for each category in training data
                    value_counts = X_processed[col].value_counts(dropna=False)
                    count = (
                        pd.Series(value_counts)
                        .reindex(categories)
                        .fillna(0)
                        .astype(int)
                        .values
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

                    # Calculate confidence intervals for WOE values
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
                            "count_pct": (
                                count.astype(float) / len(X_processed) * 100
                            ).tolist(),
                            "good_count": good_counts,
                            "bad_count": bad_counts,
                            "event_rate": np.round(event_rates, 6),
                            "woe": woe,
                            "woe_se": woe_se,
                            "woe_ci_lower": woe_ci_lower,
                            "woe_ci_upper": woe_ci_upper,
                        }
                    ).set_index("category")

                    self.mappings_[col][class_label] = mapping_df

                    # Calculate feature-level statistics for this class
                    self.feature_stats_[col][class_label] = (
                        self._calculate_feature_stats(col, X_processed, y, mapping_df)
                    )
            else:
                # Binary or continuous case
                enc = TargetEncoder(
                    **self.encoder_kwargs, random_state=self.random_state
                )
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
                    pd.Series(value_counts)
                    .reindex(categories)
                    .fillna(0)
                    .astype(int)
                    .values
                )

                # Enhanced mapping with more details
                # For both continuous and binary targets, we derive good/bad counts from probabilities
                # This works because: good_count = total_count * (1 - probability)
                # and bad_count = total_count * probability
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
                        "count_pct": (
                            count.astype(float) / len(X_processed) * 100
                        ).tolist(),
                        "good_count": good_counts,
                        "bad_count": bad_counts,
                        "event_rate": np.round(event_rates, 6),
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

    def transform(self, X: Union[pd.DataFrame, np.ndarray, pd.Series]) -> pd.DataFrame:
        """
        Transform features to Weight of Evidence (WOE) values.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray, pd.Series]
            Input DataFrame with features to transform. If numpy array, will be converted to DataFrame with generic column names.
            If Series, will be converted to single-column DataFrame.

        Returns:
        -------
        pd.DataFrame
            DataFrame with same shape as input, but with WOE values
            instead of categorical values. Column names are preserved.

        """
        # Convert numpy arrays and Series to pandas DataFrame if needed
        if isinstance(X, np.ndarray):
            column_names = [f"feature_{i}" for i in range(X.shape[1])]
            # pyrefly: ignore  # bad-argument-type
            X = pd.DataFrame(X, columns=column_names)
            warnings.warn(
                "Input X is a numpy array. Converting to pandas DataFrame with generic column names. "
                "For better control, convert to DataFrame with meaningful column names before passing to transform().",
                stacklevel=2,
            )
        elif isinstance(X, pd.Series):
            # Convert Series to DataFrame with the Series name as column name
            column_name = X.name if X.name is not None else "feature_0"
            X = pd.DataFrame({column_name: X})
            warnings.warn(
                f"Input X is a pandas Series. Converting to DataFrame with column name '{column_name}'. "
                "For better control, convert to DataFrame before passing to transform().",
                stacklevel=2,
            )

        if not self.is_fitted_:
            raise ValueError("Model must be fitted before transforming data")

        if not self.is_multiclass_target:
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
                if hasattr(X_col[col], "isna"):
                    # pyrefly: ignore  # missing-attribute
                    mask_missing = X_col[col].isna()
                else:
                    # For numpy arrays, use pd.isna
                    mask_missing = pd.isna(X_col[col])

                # pyrefly: ignore  # missing-attribute
                if not mask_missing.all():  # If there are any non-missing values
                    # Ensure column is object type to accept strings
                    X_processed[col] = X_processed[col].astype("object")

                    if binning_info.get("method") == "kbins":
                        # KBinsDiscretizer method
                        X_processed.loc[~mask_missing, col] = binner.transform(
                            X_col[~mask_missing]
                        ).ravel()
                    elif binning_info.get("method") == "tree":
                        # Tree method - use bin edges for binning
                        bin_edges = np.array(binning_info["bin_edges"])
                        col_data = X_col[~mask_missing][col]
                        if hasattr(col_data, "values"):
                            col_values = col_data.values
                        else:
                            col_values = np.array(col_data)
                        binned_values = np.digitize(
                            col_values,
                            bin_edges[1:-1],
                            right=False,
                        )
                        binned_values = np.clip(
                            binned_values - 1, 0, len(bin_edges) - 2
                        )
                        X_processed.loc[~mask_missing, col] = binned_values
                    elif binning_info.get("method") == "faiss_kmeans":
                        # FAISS KMeans method - use FAISS model for prediction
                        faiss_model = binner
                        col_data = X_col[~mask_missing][col]
                        if hasattr(col_data, "values"):
                            col_values = col_data.values
                        else:
                            col_values = np.array(col_data)
                        data = col_values.astype(np.float32).reshape(-1, 1)
                        _, labels = faiss_model.index.search(data, 1)
                        cluster_labels = (
                            labels.flatten() + 1
                        )  # Convert to 1-based indexing

                        # Map cluster labels to bin labels
                        bin_edges = np.array(binning_info["bin_edges"])
                        bin_labels = []
                        for i in range(len(bin_edges) - 1):
                            if i == 0:
                                label = f"(-∞, {bin_edges[i + 1]:.1f}]"
                            elif i == len(bin_edges) - 2:
                                label = f"({bin_edges[i]:.1f}, ∞)"
                            else:
                                label = f"({bin_edges[i]:.1f}, {bin_edges[i + 1]:.1f}]"
                            bin_labels.append(label)

                        cluster_to_bin = dict(
                            zip(range(1, len(bin_labels) + 1), bin_labels)
                        )
                        binned_labels = [
                            cluster_to_bin[label] for label in cluster_labels
                        ]
                        X_processed.loc[~mask_missing, col] = binned_labels

                # Convert to string categories with meaningful labels (same as fit)
                if binning_info.get("method") == "kbins" and hasattr(
                    binner, "bin_edges_"
                ):
                    edges = binner.bin_edges_[0]
                elif (
                    binning_info.get("method") == "tree" and "bin_edges" in binning_info
                ):
                    edges = np.array(binning_info["bin_edges"])
                elif (
                    binning_info.get("method") == "faiss_kmeans"
                    and "bin_edges" in binning_info
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
                    if (
                        len(non_missing_values) > 0
                        and binning_info.get("method") != "faiss_kmeans"
                    ):
                        X_processed.loc[~mask_missing, col] = (
                            non_missing_values.astype(int)
                            .map(dict(enumerate(bin_labels)))
                            .astype(str)
                        )

                # Handle missing values
                # pyrefly: ignore  # missing-attribute
                if mask_missing.any():
                    X_processed.loc[mask_missing, col] = "Missing"

        for col in X_processed.columns:
            if self.is_multiclass_target:
                # For multiclass, create one column per class
                for class_label in self.classes_:
                    enc = self.encoders_[col][class_label]
                    event_rate = enc.transform(X_processed[[col]])

                    # Binary case for this class
                    if isinstance(event_rate, pd.DataFrame):
                        event_rate = event_rate.values.flatten()
                    event_rate = np.clip(event_rate, 1e-15, 1 - 1e-15)
                    odds_cat = event_rate / (1 - event_rate)
                    woe = np.log(odds_cat / self.odds_prior_per_class_[class_label])

                    # Create column name with class suffix
                    col_name = f"{col}_class_{class_label}"
                    woe_df[col_name] = woe
            else:
                # Binary or continuous case
                enc = self.encoders_[col]
                event_rate = enc.transform(X_processed[[col]])

                if isinstance(event_rate, pd.DataFrame):
                    event_rate = event_rate.values.flatten()
                event_rate = np.clip(event_rate, 1e-15, 1 - 1e-15)
                odds_cat = event_rate / (1 - event_rate)  # pyrefly: ignore
                woe = np.log(odds_cat / odds_prior)
                woe_df[col] = woe  # Keep original column name

        return woe_df

    def fit_transform(self, X: pd.DataFrame, y=None, **_fit_params) -> pd.DataFrame:
        """Fit and transform in one step."""
        # pyrefly: ignore  # bad-argument-type
        return self.fit(X, y).transform(X)

    def get_mapping(
        self, feature: str, class_label: Optional[Union[int, str]] = None
    ) -> pd.DataFrame:
        """
        Get the WOE mapping DataFrame for a specific feature, preserving correct bin order for binned features.

        For multiclass targets, specify class_label to get mapping for a specific class.
        If class_label is None and target is multiclass, returns mapping for first class.
        """
        if not self.is_fitted_:
            raise ValueError("FastWoe must be fitted before getting mappings")
        if feature not in self.mappings_:
            raise ValueError(f"Feature '{feature}' not found in fitted features")

        if self.is_multiclass_target:
            if class_label is None:
                class_label = self.classes_[0]
            if class_label not in self.mappings_[feature]:
                raise ValueError(
                    f"Class '{class_label}' not found for feature '{feature}'. Available classes: {self.classes_}"
                )
            mapping = self.mappings_[feature][class_label].copy()
        else:
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
            elif (
                binning_info.get("method") == "faiss_kmeans"
                and "bin_edges" in binning_info
            ):
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

    def get_probability_mapping(self, feature: str) -> pd.DataFrame:
        """
        Get the probability mapping DataFrame for a specific feature.
        This is particularly useful for continuous targets where you want the raw probabilities
        instead of WOE values.

        Parameters
        ----------
        feature : str
            Name of the feature to get probability mapping for

        Returns:
        -------
        pd.DataFrame
            DataFrame with columns: category, count, count_pct, probability, probability_se
        """
        if not self.is_fitted_:
            raise ValueError(
                "FastWoe must be fitted before getting probability mappings"
            )

        if feature not in self.mappings_:
            raise ValueError(f"Feature '{feature}' not found in fitted mappings")

        mapping = self.mappings_[feature].copy()

        # Extract probability information
        prob_mapping = pd.DataFrame(
            {
                "category": mapping.index,
                "count": mapping["count"],
                "count_pct": mapping["count_pct"],
                "probability": mapping["event_rate"],  # This is the actual probability
            }
        )

        # Add probability standard error if available
        if "woe_se" in mapping.columns:
            # Convert WOE SE to probability SE using delta method
            # For probability p, if WOE = log(p/(1-p)), then dWOE/dp = 1/(p*(1-p))
            # So SE(p) = SE(WOE) * p * (1-p)
            probabilities = mapping["event_rate"].values
            woe_se = mapping["woe_se"].values
            # pyrefly: ignore  # unsupported-operation
            prob_se = woe_se * probabilities * (1 - probabilities)
            prob_mapping["probability_se"] = prob_se

        return prob_mapping

    def get_feature_stats(
        self, col: Optional[str] = None, class_label: Optional[Union[int, str]] = None
    ) -> pd.DataFrame:
        """
        Get feature statistics. If col is None, return stats for all features.

        For multiclass targets, specify class_label to get stats for a specific class.
        If class_label is None and target is multiclass, returns stats for first class.
        """
        if self.is_multiclass_target:
            if col is not None:
                if class_label is None:
                    class_label = self.classes_[0]
                if class_label not in self.feature_stats_[col]:
                    raise ValueError(
                        f"Class '{class_label}' not found for feature '{col}'. Available classes: {self.classes_}"
                    )
                return pd.DataFrame([self.feature_stats_[col][class_label]])
            else:
                # Return stats for all features and all classes
                all_stats = []
                for feature_name, feature_stats in self.feature_stats_.items():
                    for class_label, stats in feature_stats.items():
                        stats_copy = stats.copy()
                        stats_copy["feature"] = f"{feature_name}_class_{class_label}"
                        stats_copy["class"] = class_label
                        all_stats.append(stats_copy)
                return pd.DataFrame(all_stats)
        else:
            # Binary case (original implementation)
            if col is not None:
                return pd.DataFrame([self.feature_stats_[col]])
            else:
                return pd.DataFrame(list(self.feature_stats_.values()))

    def get_feature_summary(self) -> pd.DataFrame:
        """Get a summary table of all features ranked by predictive power."""
        stats_df = self.get_feature_stats()
        # pyrefly: ignore  # bad-return
        return stats_df.sort_values("gini", ascending=False)[
            ["feature", "gini", "iv", "n_categories"]
        ].round(4)

    def get_iv_analysis(
        self,
        col: Optional[str] = None,
        class_label: Optional[Union[int, str]] = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Get detailed Information Value analysis with confidence intervals.

        Parameters
        ----------
        col : str, optional
            Feature name. If None, returns analysis for all features.
        class_label : int or str, optional
            For multiclass targets, specify class to analyze. If None, uses first class.
        alpha : float, default=0.05
            Significance level for confidence intervals (0.05 for 95% CI)

        Returns:
        -------
        pd.DataFrame
            DataFrame with IV statistics including standard errors and confidence intervals
        """
        if not self.is_fitted_:
            raise ValueError("FastWoe must be fitted before getting IV analysis")

        if self.is_multiclass_target:
            if col is not None:
                if class_label is None:
                    class_label = self.classes_[0]
                if col not in self.feature_stats_:
                    raise ValueError(f"Feature '{col}' not found in fitted features")
                if class_label not in self.feature_stats_[col]:
                    raise ValueError(
                        f"Class '{class_label}' not found for feature '{col}'. Available classes: {self.classes_}"
                    )
                stats = self.feature_stats_[col][class_label]
                return pd.DataFrame(
                    [
                        {
                            "feature": f"{stats['feature']}_class_{class_label}",
                            "class": class_label,
                            "iv": stats["iv"],
                            "iv_se": stats["iv_se"],
                            "iv_ci_lower": stats["iv_ci_lower"],
                            "iv_ci_upper": stats["iv_ci_upper"],
                            "iv_significance": "Significant"
                            if stats["iv_ci_lower"] > 0
                            else "Not Significant",
                            "n_categories": stats["n_categories"],
                            "gini": stats["gini"],
                        }
                    ]
                )
            else:
                # Return analysis for all features and all classes
                analysis_data = []
                for _, feature_stats in self.feature_stats_.items():
                    for class_label, stats in feature_stats.items():
                        analysis_data.append(
                            {
                                "feature": f"{stats['feature']}_class_{class_label}",
                                "class": class_label,
                                "iv": stats["iv"],
                                "iv_se": stats["iv_se"],
                                "iv_ci_lower": stats["iv_ci_lower"],
                                "iv_ci_upper": stats["iv_ci_upper"],
                                "iv_significance": (
                                    "Significant"
                                    if stats["iv_ci_lower"] > 0
                                    else "Not Significant"
                                ),
                                "n_categories": stats["n_categories"],
                                "gini": stats["gini"],
                            }
                        )
                df = pd.DataFrame(analysis_data)
                return df.sort_values("iv", ascending=False).round(4)
        else:
            # Binary case (original implementation)
            if col is not None:
                if col not in self.feature_stats_:
                    raise ValueError(f"Feature '{col}' not found in fitted features")
                stats = self.feature_stats_[col]
                return pd.DataFrame(
                    [
                        {
                            "feature": stats["feature"],
                            "iv": stats["iv"],
                            "iv_se": stats["iv_se"],
                            "iv_ci_lower": stats["iv_ci_lower"],
                            "iv_ci_upper": stats["iv_ci_upper"],
                            "iv_significance": "Significant"
                            if stats["iv_ci_lower"] > 0
                            else "Not Significant",
                            "n_categories": stats["n_categories"],
                            "gini": stats["gini"],
                        }
                    ]
                )
            else:
                analysis_data = [
                    {
                        "feature": stats["feature"],
                        "iv": stats["iv"],
                        "iv_se": stats["iv_se"],
                        "iv_ci_lower": stats["iv_ci_lower"],
                        "iv_ci_upper": stats["iv_ci_upper"],
                        "iv_significance": (
                            "Significant"
                            if stats["iv_ci_lower"] > 0
                            else "Not Significant"
                        ),
                        "n_categories": stats["n_categories"],
                        "gini": stats["gini"],
                    }
                    for _feature_name, stats in self.feature_stats_.items()
                ]
                df = pd.DataFrame(analysis_data)
                return df.sort_values("iv", ascending=False).round(4)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict probabilities using WOE-transformed features.

        For binary targets: returns 2 probabilities [P(0), P(1)]
        For multiclass targets: returns probabilities for each class [P(0), P(1), P(2), ...]
        """
        # Handle numpy array input
        if isinstance(X, np.ndarray):
            warnings.warn(
                "Input X is a numpy array. Converting to pandas DataFrame with generic column names. "
                "For better control, convert to DataFrame with meaningful column names before passing to predict_proba().",
                UserWarning,
                stacklevel=2,
            )
            column_names = [f"feature_{i}" for i in range(X.shape[1])]
            # pyrefly: ignore  # bad-argument-type
            X = pd.DataFrame(X, columns=column_names)

        X_woe = self.transform(X)
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before predicting probabilities")

        if self.is_multiclass_target:
            # For multiclass: calculate WOE score for each class
            n_samples = len(X_woe)
            n_classes = len(self.classes_)
            woe_scores = np.zeros((n_samples, n_classes))

            for i, class_label in enumerate(self.classes_):
                # Get WOE columns for this class
                class_cols = [
                    col
                    for col in X_woe.columns
                    if col.endswith(f"_class_{class_label}")
                ]
                if class_cols:
                    class_woe = X_woe[class_cols].sum(axis=1)
                    # Add log prior for this class
                    log_prior = np.log(
                        self.y_prior_[class_label] / (1 - self.y_prior_[class_label])
                    )
                    woe_scores[:, i] = class_woe + log_prior

            # Convert to probabilities using softmax
            exp_scores = np.exp(woe_scores - np.max(woe_scores, axis=1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            return probs
        else:
            # Binary case
            if self.y_prior_ is None:
                raise ValueError("Model must be fitted before predicting probabilities")
            odds_prior = self.y_prior_ / (1 - self.y_prior_)
            woe_score = X_woe.sum(axis=1) + np.log(odds_prior)

            # Convert to probability (simple sigmoid transformation)
            prob = sigmoid(woe_score)
            return np.column_stack([1 - prob, prob])

    def predict_ci(self, X: Union[pd.DataFrame, np.ndarray], alpha=0.05) -> np.ndarray:
        """
        Predict confidence intervals for WOE values and probabilities.

        For binary targets: returns 2 columns [lower, upper] for class 1 probability
        For multiclass targets: returns 2 columns per class [lower_0, upper_0, lower_1, upper_1, ...]

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input features to predict confidence intervals for
        alpha : float, default=0.05
            Significance level (0.05 for 95% CI)

        Returns:
        -------
        np.ndarray
            For binary: (n_samples, 2) containing [lower, upper] bounds
            For multiclass: (n_samples, 2*n_classes) containing [lower_0, upper_0, lower_1, upper_1, ...]

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
            column_names = [f"feature_{i}" for i in range(X.shape[1])]
            # pyrefly: ignore  # bad-argument-type
            X = pd.DataFrame(X, columns=column_names)

        # Get WOE-transformed features
        X_woe = self.transform(X)

        if self.is_multiclass_target:
            # For multiclass: calculate CI for each class
            n_samples = len(X_woe)
            n_classes = len(self.classes_)
            z_crit = norm.ppf(1 - alpha / 2)

            # Initialize arrays for each class
            all_ci_lower = []
            all_ci_upper = []

            for _, class_label in enumerate(self.classes_):
                # Get WOE columns for this class
                class_cols = [
                    col
                    for col in X_woe.columns
                    if col.endswith(f"_class_{class_label}")
                ]
                if not class_cols:
                    continue

                # Calculate WOE confidence intervals for this class
                woe_ci_lower = np.zeros((n_samples, len(class_cols)))
                woe_ci_upper = np.zeros((n_samples, len(class_cols)))

                # Map class columns back to original features
                original_features = []
                for col in class_cols:
                    # Extract original feature name (remove _class_X suffix)
                    orig_feature = col.rsplit("_class_", 1)[0]
                    original_features.append(orig_feature)

                for i, orig_feature in enumerate(original_features):
                    if orig_feature in self.mappings_:
                        mapping = self.mappings_[orig_feature][class_label]

                        # For each row, find the WOE standard error
                        for j in range(n_samples):
                            cat_value = X.iloc[j, X.columns.get_loc(orig_feature)]

                            # Look up in mapping (handle unseen categories)
                            if cat_value in mapping.index:
                                woe_se = mapping.loc[cat_value, "woe_se"]
                            else:
                                # For unseen categories, use the average SE
                                woe_se = mapping["woe_se"].mean()

                            woe_val = X_woe.iloc[
                                j, X_woe.columns.get_loc(class_cols[i])
                            ]
                            margin = z_crit * woe_se

                            woe_ci_lower[j, i] = woe_val - margin
                            woe_ci_upper[j, i] = woe_val + margin

                # Sum WOE values across features for this class
                woe_score_lower = woe_ci_lower.sum(axis=1)
                woe_score_upper = woe_ci_upper.sum(axis=1)

                # Convert to probability scale for this class
                log_prior = np.log(
                    self.y_prior_[class_label] / (1 - self.y_prior_[class_label])
                )
                logit_lower = woe_score_lower + log_prior
                logit_upper = woe_score_upper + log_prior

                prob_lower = sigmoid(logit_lower)
                prob_upper = sigmoid(logit_upper)

                all_ci_lower.append(prob_lower)
                all_ci_upper.append(prob_upper)

            # Combine all classes: [lower_0, upper_0, lower_1, upper_1, ...]
            result = []
            for lower, upper in zip(all_ci_lower, all_ci_upper):
                result.extend([lower, upper])

            return np.column_stack(result)
        else:
            # Binary case (original implementation)
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
            if self.y_prior_ is None:
                raise ValueError(
                    "Model must be fitted before predicting confidence intervals"
                )
            odds_prior = self.y_prior_ / (1 - self.y_prior_)
            logit_lower = woe_score_lower + np.log(odds_prior)
            logit_upper = woe_score_upper + np.log(odds_prior)

            prob_lower = sigmoid(logit_lower)
            prob_upper = sigmoid(logit_upper)

        # Return numpy array with shape (n_samples, 2)
        return np.column_stack([prob_lower, prob_upper])

    def predict_proba_class(
        self, X: Union[pd.DataFrame, np.ndarray], class_label: Union[int, str]
    ) -> np.ndarray:
        """
        Predict probabilities for a specific class in multiclass scenarios.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input features to predict probabilities for
        class_label : int or str
            Class label to predict probabilities for

        Returns:
        -------
        np.ndarray
            Array with shape (n_samples,) containing probabilities for the specified class
        """
        if not self.is_multiclass_target:
            raise ValueError(
                "predict_proba_class() is only available for multiclass targets"
            )

        if class_label not in self.classes_:
            raise ValueError(
                f"Class '{class_label}' not found. Available classes: {self.classes_}"
            )

        # Get all probabilities
        all_probs = self.predict_proba(X)

        # Find the index of the requested class
        class_idx = self.classes_.index(class_label)

        # Return probabilities for the specified class
        return all_probs[:, class_idx]

    def predict_ci_class(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        class_label: Union[int, str],
        alpha: float = 0.05,
    ) -> np.ndarray:
        """
        Predict confidence intervals for a specific class in multiclass scenarios.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input features to predict confidence intervals for
        class_label : int or str
            Class label to predict confidence intervals for
        alpha : float, default=0.05
            Significance level (0.05 for 95% CI)

        Returns:
        -------
        np.ndarray
            Array with shape (n_samples, 2) containing [lower, upper] bounds for the specified class
        """
        if not self.is_multiclass_target:
            raise ValueError(
                "predict_ci_class() is only available for multiclass targets"
            )

        if class_label not in self.classes_:
            raise ValueError(
                f"Class '{class_label}' not found. Available classes: {self.classes_}"
            )

        # Get all confidence intervals
        all_ci = self.predict_ci(X, alpha)

        # Find the index of the requested class
        class_idx = self.classes_.index(class_label)

        # Extract CI for the specified class (2 columns per class)
        lower_col = class_idx * 2
        upper_col = class_idx * 2 + 1

        return all_ci[:, [lower_col, upper_col]]

    def transform_standardized(
        self,
        X: pd.DataFrame,
        output="woe",
        col_name: Optional[str] = None,
        # pyrefly: ignore  # bad-return
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
                    if self.y_prior_ is None:
                        raise ValueError(
                            "Model must be fitted before calculating Wald statistics"
                        )
                    # Handle both binary (float) and multiclass (dict) priors
                    if isinstance(self.y_prior_, dict):
                        # For multiclass, use the first class prior (main class)
                        first_class = list(self.y_prior_.keys())[0]
                        prior = self.y_prior_[first_class]
                    else:
                        prior = self.y_prior_
                    prior_log_odds = np.log(prior / (1 - prior))
                    final_log_odds = woe_val + prior_log_odds
                    z_col[i] = final_log_odds / woe_se if woe_se > 0 else 0
                else:
                    raise ValueError("output must be 'woe' or 'wald'")

            z_scores[col] = z_col

        return z_scores

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict outcomes using WOE-transformed features.

        For binary targets: predicts class 1 if WOE score > 0, class 0 otherwise.
        For multiclass targets: predicts the class with the highest WOE score + prior.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before transforming data")

        # Handle numpy array input
        if isinstance(X, np.ndarray):
            warnings.warn(
                "Input X is a numpy array. Converting to pandas DataFrame with generic column names. "
                "For better control, convert to DataFrame with meaningful column names before passing to predict().",
                UserWarning,
                stacklevel=2,
            )
            column_names = [f"feature_{i}" for i in range(X.shape[1])]
            # pyrefly: ignore  # bad-argument-type
            X = pd.DataFrame(X, columns=column_names)

        if self.is_multiclass_target:
            # For multiclass: return class with highest probability
            probs = self.predict_proba(X)
            class_indices = np.argmax(probs, axis=1)
            return np.array([self.classes_[i] for i in class_indices])
        else:
            # Binary case
            woe_score = self.transform(X).sum(axis=1)
            return (woe_score > 0).astype(int).values

    def _detect_numerical_features(self, X: pd.DataFrame) -> list[str]:
        """
        Detect numerical features that should be binned.

        Returns list of column names that are:
        1. Numerical (using numbers.Number)
        2. Have >= numerical_threshold unique values
        """
        numerical_features: list[str] = []

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
        if hasattr(X_col[col], "isna"):
            # pyrefly: ignore  # missing-attribute
            mask_missing = X_col[col].isna()
        else:
            # For numpy arrays, use pd.isna
            mask_missing = pd.isna(X_col[col])

        # pyrefly: ignore  # missing-attribute
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
            # pyrefly: ignore  # bad-argument-type
            return self._bin_with_kbins(X_col, col, mask_missing, X_fit)
        elif self.binning_method == "faiss_kmeans":
            # pyrefly: ignore  # bad-argument-type
            return self._bin_with_faiss_kmeans(X_col, col, mask_missing, X_fit)
        else:  # tree method
            # Determine if target is continuous (proportions) or binary
            unique_targets = y.nunique()
            unique_values = sorted(y.unique())
            is_continuous = (
                y.dtype in ["float32", "float64"]
                and (y >= 0).all()
                and (y <= 1).all()
                and unique_targets > 2
            )
            return self._bin_with_tree(
                # pyrefly: ignore  # bad-argument-type
                X_col,
                col,
                y,
                # pyrefly: ignore  # bad-argument-type
                mask_missing,
                # pyrefly: ignore  # bad-argument-type
                X_fit,
                is_continuous,
            )

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

        # Set quantile_method to avoid sklearn future warning (only available in sklearn >= 1.7)
        if "quantile_method" not in binner_kwargs:
            with contextlib.suppress(ImportError, AttributeError):
                import sklearn
                from packaging import version

                if version.parse(sklearn.__version__) >= version.parse("1.7.0"):
                    binner_kwargs["quantile_method"] = "averaged_inverted_cdf"
        # Try to create the binner, removing quantile_method if it fails
        try:
            binner = KBinsDiscretizer(random_state=self.random_state, **binner_kwargs)
        except TypeError as e:
            if "quantile_method" not in str(e):
                raise
            # Remove quantile_method and try again
            binner_kwargs.pop("quantile_method", None)
            binner = KBinsDiscretizer(random_state=self.random_state, **binner_kwargs)
        binner.fit(X_fit)

        # Transform all values
        X_binned = X_col.copy()
        if not mask_missing.all():  # If there are any non-missing values
            # Ensure column is object type to accept strings
            X_binned[col] = X_binned[col].astype("object")
            # pyrefly: ignore  # missing-attribute
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
        is_continuous: bool = False,
    ) -> pd.DataFrame:
        """Apply decision tree-based binning to a numerical feature."""
        # Select appropriate tree estimator based on target type
        # pylint: disable=redefined-outer-name
        if self.tree_estimator is None:
            if is_continuous:
                tree_estimator = DecisionTreeRegressor
            else:
                tree_estimator = DecisionTreeClassifier
        else:
            tree_estimator = self.tree_estimator

        # Create and fit the tree
        tree_kwargs = self.tree_kwargs.copy()
        tree = tree_estimator(**tree_kwargs)

        # Fit tree on non-missing data
        y_fit = y[~mask_missing] if mask_missing.any() else y
        tree.fit(X_fit, y_fit)

        # Extract split points from the tree
        split_points = self._extract_tree_splits(tree, col)

        # Create bin edges from split points
        col_data = X_fit[col]
        if hasattr(col_data, "values"):
            col_values = col_data.values
        else:
            col_values = np.array(col_data)
        # pyrefly: ignore  # bad-argument-type
        bin_edges = self._create_bin_edges_from_splits(split_points, col_values)

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
            col_data = X_fit[col]
            if hasattr(col_data, "values"):
                col_values = col_data.values
            else:
                col_values = np.array(col_data)
            # Custom binning logic to handle right-inclusive intervals correctly
            binned_values = np.zeros(len(col_values), dtype=int)
            for i, val in enumerate(col_values):
                for bin_idx in range(len(bin_edges) - 1):
                    if bin_edges[bin_idx] < val <= bin_edges[bin_idx + 1]:
                        binned_values[i] = bin_idx
                        break
                else:
                    # Handle edge case where value is exactly at -inf or +inf
                    if val <= bin_edges[0]:
                        binned_values[i] = 0
                    elif val > bin_edges[-1]:
                        binned_values[i] = len(bin_edges) - 2

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

    def _bin_with_faiss_kmeans(
        self,
        X_col: pd.DataFrame,
        col: str,
        mask_missing: pd.Series,
        X_fit: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply FAISS KMeans clustering to a numerical feature."""
        try:
            import faiss  # noqa: F401

            print("FAISS is available:", faiss)
        except ImportError as e:
            raise ImportError(
                "FAISS is required for faiss_kmeans binning method. "
                "Install CPU version: pip install faiss-cpu "
                "or GPU version: pip install faiss-gpu-cu12"
            ) from e

        # Prepare data for FAISS
        col_data = X_fit[col]
        if hasattr(col_data, "values"):
            col_values = col_data.values
        else:
            col_values = np.array(col_data)
        data = col_values.astype(np.float32).reshape(-1, 1)
        d = data.shape[1]  # dimension
        k = self.faiss_kwargs["k"]
        niter = self.faiss_kwargs["niter"]
        verbose = self.faiss_kwargs["verbose"]
        gpu = self.faiss_kwargs["gpu"]

        # Create FAISS KMeans
        from faiss.extra_wrappers import Kmeans

        faiss_kmeans = Kmeans(d=d, k=k, niter=niter, verbose=verbose, gpu=gpu)
        faiss_kmeans.train(data)

        # Assign cluster labels
        n_samples = data.shape[0]
        distances = np.zeros((n_samples, 1), dtype=np.float32)
        labels = np.zeros((n_samples, 1), dtype=np.int64)
        faiss_kmeans.index.search(n_samples, data, 1, distances, labels)
        cluster_labels = labels.flatten() + 1  # Convert to 1-based indexing

        # Create bin edges from cluster centroids
        centroids_raw = faiss_kmeans.centroids
        if centroids_raw is None:
            raise ValueError(
                "FAISS KMeans centroids are None - clustering may have failed"
            )
        centroids = centroids_raw.flatten()
        sorted_centroids = np.sort(centroids)

        # Create bin edges: midpoints between consecutive centroids
        bin_edges = np.zeros(len(sorted_centroids) + 1)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        for i in range(len(sorted_centroids) - 1):
            bin_edges[i + 1] = (sorted_centroids[i] + sorted_centroids[i + 1]) / 2

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

            # Map cluster labels to bin labels
            cluster_to_bin = dict(zip(range(1, k + 1), bin_labels))
            # Convert to pandas Series to use .map() method
            cluster_series = pd.Series(cluster_labels)
            X_binned.loc[~mask_missing, col] = cluster_series.map(cluster_to_bin)

        # Handle missing values
        if mask_missing.any():
            X_binned.loc[mask_missing, col] = "Missing"

        # Store FAISS model and info
        self.binners_[col] = faiss_kmeans
        self.binning_info_[col] = {
            "values": X_col[col].nunique(),
            "n_bins": k,
            "missing": mask_missing.sum(),
            "bin_edges": bin_edges.tolist(),
            "method": "faiss_kmeans",
            "centroids": sorted_centroids.tolist(),
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

        # Remove duplicate split points and sort
        unique_splits = np.unique(split_points)

        # Ensure we have valid splits that actually divide the data
        min_val = np.min(data)
        max_val = np.max(data)

        # Filter out splits that are at the boundaries (they don't create meaningful bins)
        valid_splits = unique_splits[
            (unique_splits > min_val) & (unique_splits < max_val)
        ]

        if len(valid_splits) == 0:
            # No valid splits found, create single bin
            return np.array([-np.inf, np.inf])

        return np.concatenate([[-np.inf], valid_splits, [np.inf]])

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

    def get_tree_estimator(self, feature: str):
        """
        Get the fitted tree estimator for a numerical feature that was binned using tree method.

        Parameters
        ----------
        feature : str
            Name of the numerical feature to get the tree estimator for

        Returns:
        -------
        sklearn.tree.DecisionTreeClassifier or sklearn.tree.DecisionTreeRegressor
            The fitted tree estimator used for binning this feature

        Raises:
        ------
        ValueError
            If the feature was not binned using tree method or not found
        """
        if not self.is_fitted_:
            raise ValueError("FastWoe must be fitted before calling get_tree_estimator")

        if feature not in self.mappings_:
            raise ValueError(f"Feature '{feature}' not found in fitted features")

        if feature not in self.binners_:
            raise ValueError(
                f"Feature '{feature}' is not a binned numerical feature. "
                f"This function only works with numerical features that were automatically binned."
            )

        # Check if this feature was binned using tree method
        binning_info = self.binning_info_[feature]
        if binning_info.get("method") != "tree":
            raise ValueError(
                f"Feature '{feature}' was not binned using tree method. "
                f"Found method: {binning_info.get('method', 'unknown')}. "
                f"Use get_binning_summary() to see which features used tree binning."
            )

        return self.binners_[feature]

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
        elif (
            binning_info.get("method") == "faiss_kmeans" and "bin_edges" in binning_info
        ):
            # For FAISS KMeans method, get edges from binning_info
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
