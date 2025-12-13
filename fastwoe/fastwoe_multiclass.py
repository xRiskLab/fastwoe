"""Multiclass Weight of Evidence encoding functionality.

This module contains the multiclass-specific implementation for FastWoe,
providing one-vs-rest WOE encoding for multiclass targets.
"""

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from scipy.stats import norm
from sklearn.preprocessing import TargetEncoder


class MulticlassWoeMixin:
    """Mixin class providing multiclass functionality for FastWoe.

    This mixin adds multiclass support to the base FastWoe class,
    implementing one-vs-rest WOE encoding for multiclass targets.
    """

    def _detect_multiclass_target(self, y: Union[pd.Series, np.ndarray]) -> bool:
        """Detect if target is multiclass.

        Parameters
        ----------
        y : pd.Series or np.ndarray
            Target variable

        Returns:
        -------
        bool
            True if target is multiclass, False otherwise
        """
        y_series = y if isinstance(y, pd.Series) else pd.Series(y)
        unique_targets = len(y_series.unique())

        # Check if target is continuous (proportions), binary, or multiclass
        is_continuous = (
            y_series.dtype in ["float64", "float32"]
            and y_series.min() >= 0
            and y_series.max() <= 1
            and unique_targets > 2
        )

        # Multiclass: more than 2 unique values and not continuous proportions
        is_multiclass = unique_targets > 2 and not is_continuous

        return is_multiclass

    def _setup_multiclass_target(self, y: Union[pd.Series, np.ndarray]) -> None:
        """Setup multiclass target attributes and priors.

        Parameters
        ----------
        y : pd.Series or np.ndarray
            Target variable
        """
        y_series = y if isinstance(y, pd.Series) else pd.Series(y)

        self.classes_ = sorted(y_series.unique())
        self.n_classes_ = len(self.classes_)
        self.y_prior_ = {}  # Dictionary of priors for each class
        self.odds_prior_per_class_ = {}

        # Calculate priors for each class
        for class_label in self.classes_:
            class_prior = float((y_series == class_label).mean())
            self.y_prior_[class_label] = class_prior
            self.odds_prior_per_class_[class_label] = class_prior / (1 - class_prior)

    def _create_multiclass_encoders(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> None:
        """Create one-vs-rest encoders for multiclass targets.

        Parameters
        ----------
        X : pd.DataFrame
            Input features
        y : pd.Series or np.ndarray
            Target variable
        """
        y_series = y if isinstance(y, pd.Series) else pd.Series(y)

        for col in X.columns:
            if self.is_multiclass_target:
                # For multiclass, create one encoder per class (one-vs-rest)
                self.encoders_[col] = {}
                self.mappings_[col] = {}
                self.feature_stats_[col] = {}

                for class_label in self.classes_:
                    # Create binary target: 1 for current class, 0 for all others
                    y_binary = (y_series == class_label).astype(int)

                    # Create encoder for this class
                    encoder_kwargs = self.encoder_kwargs.copy()
                    encoder_kwargs["target_type"] = "binary"

                    encoder = TargetEncoder(**encoder_kwargs, random_state=self.random_state)
                    encoder.fit(X[[col]], y_binary)
                    self.encoders_[col][class_label] = encoder

                    # Create mapping for this class
                    mapping_df = self._create_mapping_df(encoder, col, X, y_binary, class_label)
                    self.mappings_[col][class_label] = mapping_df

                    # Calculate feature-level statistics for this class
                    self.feature_stats_[col][class_label] = self._calculate_feature_stats(
                        col, X, y, mapping_df
                    )

    def _create_mapping_df(
        self,
        encoder,
        col: str,
        X: pd.DataFrame,
        y_binary: pd.Series,
        class_label: Union[int, str],
    ) -> pd.DataFrame:
        """Create mapping DataFrame for a specific class."""
        # Get unique categories as seen by the encoder
        categories = encoder.categories_[0]
        event_rates = encoder.encodings_[0]

        # Defensive clipping to avoid log(0)
        event_rates = np.clip(event_rates, 1e-15, 1 - 1e-15)
        odds_cat = event_rates / (1 - event_rates)
        woe = np.log(odds_cat / self.odds_prior_per_class_[class_label])

        # Count for each category in training data
        value_counts = X[col].value_counts(dropna=False)
        count = pd.Series(value_counts).reindex(categories).fillna(0).astype(int).values

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
                "count_pct": (count.astype(float) / len(y_binary) * 100).tolist(),
                "good_count": good_counts,
                "bad_count": bad_counts,
                "event_rate": np.round(event_rates, 6),
                "woe": woe,
                "woe_se": woe_se,
                "woe_ci_lower": woe_ci_lower,
                "woe_ci_upper": woe_ci_upper,
            }
        ).set_index("category")

        # Warn if monotonic constraints specified for non-tree methods
        # Note: In multiclass scenarios, monotonic constraints are only supported
        # for tree-based binning methods in the main FastWoe class
        if col in self.monotonic_cst and self.monotonic_cst[col] != 0:
            warnings.warn(
                f"Monotonic constraints for feature '{col}' are ignored in multiclass WOE encoding. "
                f"Monotonic constraints only work with binning_method='tree' in the main FastWoe class. "
                f"Consider using binary classification or binning_method='tree' for monotonic constraints.",
                UserWarning,
                stacklevel=3,
            )

        return mapping_df

    def _transform_multiclass(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Transform features for multiclass targets.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input features

        Returns:
        -------
        pd.DataFrame
            Transformed features with one column per class per feature
        """
        # Convert numpy arrays and Series to pandas DataFrame if needed
        if isinstance(X, np.ndarray):
            column_names = [f"feature_{i}" for i in range(X.shape[1])]
            X_processed = pd.DataFrame(X, columns=column_names)
        elif isinstance(X, pd.Series):
            column_name = X.name if X.name is not None else "feature_0"
            X_processed = pd.DataFrame({column_name: X})
        else:
            X_processed = X.copy()

        woe_df = pd.DataFrame(index=X_processed.index)

        # Apply binning to numerical features if they were binned during fit
        for col in X_processed.columns:
            if col in self.binners_:
                binner = self.binners_[col]
                if hasattr(binner, "transform"):
                    # For KBinsDiscretizer and other sklearn transformers
                    X_processed[col] = binner.transform(X_processed[[col]]).flatten()
                else:
                    # For DecisionTreeClassifier, check if data is already binned
                    col_data = X_processed[col]
                    if col_data.dtype != "object" and not isinstance(col_data.iloc[0], str):
                        # Data needs binning, use predict
                        X_processed[col] = binner.predict(X_processed[[col]])

        for col in X_processed.columns:
            if self.is_multiclass_target:
                # For multiclass, create one column per class
                for class_label in self.classes_:
                    enc = self.encoders_[col][class_label]
                    event_rate = enc.transform(X_processed[[col]])

                    # Ensure event_rate is 1D
                    if isinstance(event_rate, pd.DataFrame):
                        event_rate = event_rate.iloc[:, 0].values
                    elif isinstance(event_rate, np.ndarray) and event_rate.ndim == 2:
                        event_rate = event_rate.flatten()

                    # Calculate WOE for this class
                    event_rate_clipped = np.clip(event_rate, 1e-15, 1 - 1e-15)
                    odds_cat = event_rate_clipped / (1 - event_rate_clipped)
                    woe_values = np.log(odds_cat) - np.log(
                        self.y_prior_[class_label] / (1 - self.y_prior_[class_label])
                    )

                    woe_df[f"{col}_class_{class_label}"] = woe_values

        return woe_df

    def _get_multiclass_mapping(
        self, feature: str, class_label: Optional[Union[int, str]] = None
    ) -> pd.DataFrame:
        """Get WOE mapping for a specific class in multiclass scenarios."""
        if not self.is_multiclass_target:
            return self.mappings_[feature].copy()

        if class_label is None:
            class_label = self.classes_[0]
        if class_label not in self.mappings_[feature]:
            raise ValueError(
                f"Class '{class_label}' not found for feature '{feature}'. Available classes: {self.classes_}"
            )
        return self.mappings_[feature][class_label].copy()

    def _get_multiclass_feature_stats(
        self, col: Optional[str] = None, class_label: Optional[Union[int, str]] = None
    ) -> pd.DataFrame:
        """Get feature statistics for multiclass scenarios."""
        if self.is_multiclass_target:
            if col is not None:
                if class_label is None:
                    class_label = self.classes_[0]
                if class_label not in self.feature_stats_[col]:
                    raise ValueError(
                        f"Class '{class_label}' not found for feature '{col}'. "
                        f"Available classes: {self.classes_}"
                    )
                return pd.DataFrame([self.feature_stats_[col][class_label]])
            else:
                # Return stats for all features, first class only
                all_stats = []
                for feature_name in self.feature_stats_:
                    first_class = self.classes_[0]
                    if first_class in self.feature_stats_[feature_name]:
                        stats = self.feature_stats_[feature_name][first_class].copy()
                        stats["feature"] = feature_name
                        all_stats.append(stats)
                return pd.DataFrame(all_stats)
        else:
            if col is not None:
                return pd.DataFrame([self.feature_stats_[col]])
            all_stats = []
            for feature_name, stats in self.feature_stats_.items():
                stats_copy = stats.copy()
                stats_copy["feature"] = feature_name
                all_stats.append(stats_copy)
            return pd.DataFrame(all_stats)

    def _get_multiclass_iv_analysis(
        self,
        col: Optional[str] = None,
        class_label: Optional[Union[int, str]] = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Get IV analysis for multiclass scenarios."""
        if self.is_multiclass_target:
            if col is not None:
                if class_label is None:
                    class_label = self.classes_[0]
                if col not in self.feature_stats_:
                    raise ValueError(f"Feature '{col}' not found in fitted features")
                if class_label not in self.feature_stats_[col]:
                    raise ValueError(
                        f"Class '{class_label}' not found for feature '{col}'. "
                        f"Available classes: {self.classes_}"
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
                # Return IV analysis for all features, first class only
                all_iv = []
                for feature_name in self.feature_stats_:
                    first_class = self.classes_[0]
                    if first_class in self.feature_stats_[feature_name]:
                        stats = self.feature_stats_[feature_name][first_class]
                        all_iv.append(
                            {
                                "feature": f"{feature_name}_class_{first_class}",
                                "class": first_class,
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
                        )
                return pd.DataFrame(all_iv)
        elif col is None:
            all_iv = []
            all_iv.extend(
                {
                    "feature": feature_name,
                    "iv": stats["iv"],
                    "iv_se": stats["iv_se"],
                    "iv_ci_lower": stats["iv_ci_lower"],
                    "iv_ci_upper": stats["iv_ci_upper"],
                    "iv_significance": (
                        "Significant" if stats["iv_ci_lower"] > 0 else "Not Significant"
                    ),
                    "n_categories": stats["n_categories"],
                    "gini": stats["gini"],
                }
                for feature_name, stats in self.feature_stats_.items()
            )
            return pd.DataFrame(all_iv)

        else:
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

    def _predict_multiclass_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict probabilities for multiclass targets."""
        X_woe = self.transform(X)

        if self.is_multiclass_target:
            # For multiclass: calculate WOE score for each class
            n_samples = len(X_woe)
            n_classes = len(self.classes_)
            woe_scores = np.zeros((n_samples, n_classes))

            for i, class_label in enumerate(self.classes_):
                if class_cols := [
                    col for col in X_woe.columns if col.endswith(f"_class_{class_label}")
                ]:
                    # For simple-vs-composite hypothesis (Good, 1950)
                    # Sum of WOEs gives us the log likelihood ratio for this class
                    class_woe = X_woe[class_cols].sum(axis=1)
                    # Add log prior to get log posterior odds
                    log_prior = np.log(
                        self.y_prior_[class_label] / (1 - self.y_prior_[class_label])
                    )
                    log_posterior_odds = class_woe + log_prior
                    woe_scores[:, i] = log_posterior_odds

            # Convert to probabilities and return
            return sigmoid(woe_scores)
        else:
            # Binary case
            if self.y_prior_ is None:
                raise ValueError("Model must be fitted before predicting probabilities")
            odds_prior = self.y_prior_ / (1 - self.y_prior_)
            woe_score = X_woe.sum(axis=1) + np.log(odds_prior)

            # Convert to probability (simple sigmoid transformation)
            prob = sigmoid(woe_score)
            return np.column_stack([1 - prob, prob])

    def _predict_multiclass_ci(
        self, X: Union[pd.DataFrame, np.ndarray], alpha: float = 0.05
    ) -> np.ndarray:
        """Predict confidence intervals for multiclass targets."""
        X_woe = self.transform(X)

        if self.is_multiclass_target:
            # For multiclass: calculate CI for each class
            n_samples = len(X_woe)
            n_classes = len(self.classes_)
            z_crit = norm.ppf(1 - alpha / 2)

            # Initialize arrays for each class
            ci_lower = np.zeros((n_samples, n_classes))
            ci_upper = np.zeros((n_samples, n_classes))

            for _, class_label in enumerate(self.classes_):
                if class_cols := [
                    col for col in X_woe.columns if col.endswith(f"_class_{class_label}")
                ]:
                    class_woe = X_woe[class_cols].sum(axis=1)

                    # Calculate WOE standard error using proper delta method
                    # For multiclass, we need to calculate SE for each sample individually
                    woe_se_array = np.zeros(n_samples)

                    for i in range(n_samples):
                        sample_se_squared = 0.0

                        for col in class_cols:
                            # Extract original feature name (remove _class_X suffix)
                            orig_feature = col.rsplit("_class_", 1)[0]

                            # Get the mapping for this feature and class
                            if (
                                orig_feature in self.mappings_
                                and class_label in self.mappings_[orig_feature]
                            ):
                                mapping = self.mappings_[orig_feature][class_label]

                                # Get the category value for this sample
                                cat_value = X.iloc[i, X.columns.get_loc(orig_feature)]

                                # Look up WOE standard error in mapping
                                if cat_value in mapping.index:
                                    woe_se_feature = mapping.loc[cat_value, "woe_se"]
                                else:
                                    # For unseen categories, use the average SE
                                    woe_se_feature = mapping["woe_se"].mean()

                                sample_se_squared += woe_se_feature**2

                        woe_se_array[i] = np.sqrt(sample_se_squared)

                    # Calculate confidence intervals using per-sample standard errors
                    woe_score_lower = class_woe - z_crit * woe_se_array
                    woe_score_upper = class_woe + z_crit * woe_se_array

                    log_prior = np.log(
                        self.y_prior_[class_label] / (1 - self.y_prior_[class_label])
                    )
                    logit_lower = woe_score_lower + log_prior
                    logit_upper = woe_score_upper + log_prior

                    # Convert to probabilities
                    prob_lower = sigmoid(logit_lower)
                    prob_upper = sigmoid(logit_upper)

                    ci_lower[:, _] = prob_lower
                    ci_upper[:, _] = prob_upper

            # Interleave lower and upper bounds
            ci_result = np.zeros((n_samples, 2 * n_classes))
            for i in range(n_classes):
                ci_result[:, 2 * i] = ci_lower[:, i]
                ci_result[:, 2 * i + 1] = ci_upper[:, i]

            return ci_result
        else:
            # Binary case
            if self.y_prior_ is None:
                raise ValueError("Model must be fitted before predicting confidence intervals")
            odds_prior = self.y_prior_ / (1 - self.y_prior_)
            logit_lower = woe_score_lower + np.log(odds_prior)
            logit_upper = woe_score_upper + np.log(odds_prior)

            # Convert to probabilities
            prob_lower = sigmoid(logit_lower)
            prob_upper = sigmoid(logit_upper)

            return np.column_stack([prob_lower, prob_upper])

    def _predict_multiclass(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class labels for multiclass targets."""
        if self.is_multiclass_target:
            # For multiclass: return class with highest probability
            probs = self.predict_proba(X)
            class_indices = np.argmax(probs, axis=1)
            return np.array([self.classes_[i] for i in class_indices])
        else:
            # Binary case
            woe_score = self.transform(X).sum(axis=1)
            if self.y_prior_ is None:
                raise ValueError("Model must be fitted before predicting")

            # Handle both binary (float) and multiclass (dict) priors
            if isinstance(self.y_prior_, dict):
                # For multiclass, use the first class prior (main class)
                first_class = list(self.y_prior_.keys())[0]
                prior = self.y_prior_[first_class]
            else:
                prior = self.y_prior_

            odds_prior = prior / (1 - prior)
            logit_score = woe_score + np.log(odds_prior)
            prob = sigmoid(logit_score)
            return (prob > 0.5).astype(int)

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
            raise ValueError("predict_proba_class() is only available for multiclass targets")

        if class_label not in self.classes_:
            raise ValueError(f"Class '{class_label}' not found. Available classes: {self.classes_}")

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
            raise ValueError("predict_ci_class() is only available for multiclass targets")

        if class_label not in self.classes_:
            raise ValueError(f"Class '{class_label}' not found. Available classes: {self.classes_}")

        # Get all confidence intervals
        all_ci = self.predict_ci(X, alpha)

        # Find the index of the requested class
        class_idx = self.classes_.index(class_label)

        # Extract CI for the specified class (2 columns per class)
        lower_col = class_idx * 2
        upper_col = class_idx * 2 + 1

        return all_ci[:, [lower_col, upper_col]]
