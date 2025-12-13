"""
interpret_fastwoe.py.

FastWoe interpretability module for explaining predictions.

The explanations help users understand why a particular prediction was made and
how much each feature contributed to the final decision.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.base import BaseEstimator

from .fastwoe import FastWoe

# Initialize Rich console for beautiful output
console = Console()


class WeightOfEvidence(BaseEstimator):
    """
    Weight of Evidence explainer for FastWoe classifiers.

    Provides interpretability for FastWoe predictions by computing
    Weight of Evidence scores that measure how much evidence features provide
    for one hypothesis over another.

    Parameters
    ----------
    classifier : FastWoe, optional
        A trained FastWoe classifier with predict and predict_proba methods.
        If None, automatically creates and fits a FastWoe classifier
    X_train : array-like of shape (n_samples, n_features), optional
        Training data (original categorical data) used to fit the FastWoe classifier.
        Required when classifier is None
    y_train : array-like of shape (n_samples,), optional
        Training labels. Required when classifier is None
    feature_names : list of str, optional
        Names of features for interpretation. If None, auto-inferred
    class_names : list of str, optional
        Names of classes for interpretation. If None, auto-inferred
    auto_infer : bool, default=True
        Whether to automatically infer missing parameters

    Attributes:
    ----------
    is_fitted_ : bool
        Whether the explainer has been fitted
    n_samples_ : int
        Number of training samples
    n_features_ : int
        Number of features
    n_classes_ : int
        Number of classes
    classes_ : ndarray
        Unique class labels

    """

    @staticmethod
    def _is_fastwoe_classifier(classifier) -> bool:
        """Detect if the classifier is a FastWoe instance."""
        return (
            hasattr(classifier, "transform")
            and hasattr(classifier, "y_prior_")
            and hasattr(classifier, "mappings_")
        )

    @staticmethod
    def _extract_training_data_from_fastwoe(classifier):
        """Extract training data from a fitted FastWoe classifier."""
        if hasattr(classifier, "_X_train_original") and hasattr(classifier, "_y_train"):
            return classifier._X_train_original, classifier._y_train  # pylint: disable=protected-access
        return None, None

    @staticmethod
    def _infer_feature_names(X_train, classifier=None):  # pylint: disable=invalid-name
        """Automatically infer feature names."""
        if isinstance(X_train, pd.DataFrame):
            return list(X_train.columns)
        elif hasattr(classifier, "feature_names_"):
            return list(classifier.feature_names_)
        elif hasattr(classifier, "mappings_"):
            # FastWoe case
            return list(classifier.mappings_.keys())
        else:
            # Fallback to generic names
            n_features = X_train.shape[1] if hasattr(X_train, "shape") else len(X_train[0])
            return [f"feature_{i}" for i in range(n_features)]

    @staticmethod
    def _infer_class_names(y_train, n_classes=None):
        """Generate sensible default class names."""
        if n_classes is None:
            n_classes = len(np.unique(y_train))

        if n_classes == 2:
            return ["Negative", "Positive"]
        else:
            return [f"Class_{i}" for i in range(n_classes)]

    # pylint: disable=invalid-name
    def __init__(
        self,
        classifier=None,
        X_train: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_train: Optional[Union[np.ndarray, pd.Series]] = None,
        feature_names: Optional[list[str]] = None,
        class_names: Optional[list[str]] = None,
        auto_infer: bool = True,
    ):
        """
        Initialize WeightOfEvidence explainer with automatic parameter inference.
        """
        self.auto_infer = auto_infer

        # Handle case where no classifier is provided - create FastWoe automatically
        if classifier is None:
            if X_train is None or y_train is None:
                raise ValueError(
                    "When classifier is None, X_train and y_train must be provided "
                    "to automatically create a FastWoe classifier."
                )
            # Import and create FastWoe classifier
            try:
                classifier = FastWoe()
                classifier.fit(X_train, y_train)
                console.print("✅ Automatically created and fitted FastWoe classifier")
            except ImportError as e:
                raise ImportError(
                    "FastWoe not available. Please provide a classifier explicitly."
                ) from e

        self.classifier = classifier

        # Validate that this is a FastWoe classifier
        if not self._is_fastwoe_classifier(classifier):
            raise ValueError(
                "Only FastWoe classifiers are supported. "
                "Please provide a FastWoe classifier or use auto-creation."
            )

        # Auto-infer training data if not provided
        if auto_infer and X_train is None and y_train is None:
            X_train, y_train = self._extract_training_data_from_fastwoe(classifier)
            if X_train is None:
                raise ValueError(
                    "Cannot auto-infer training data from FastWoe classifier. "
                    "Please provide X_train and y_train explicitly, or use "
                    "FastWoe.store_training_data=True during fitting."
                )

        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train must be provided if not auto-inferable")

        # Auto-infer feature names for FastWoe first
        if auto_infer and feature_names is None:
            feature_names = self._infer_feature_names(X_train, classifier)

        if feature_names and not isinstance(X_train, pd.DataFrame):
            # type: ignore[bad-argument-type]
            self._original_X_train = pd.DataFrame(X_train, columns=list(feature_names))
        elif isinstance(X_train, pd.DataFrame):
            self._original_X_train = X_train
        else:
            self._original_X_train = pd.DataFrame(X_train)
        self.X_train_ = np.asarray(classifier.transform(self._original_X_train))
        self.y_train_ = np.asarray(y_train)

        # Update feature names to match WOE-transformed features if not set
        if feature_names is None:
            self.feature_names: Optional[list[str]] = list(self._original_X_train.columns)
        else:
            self.feature_names = feature_names

        # Auto-infer class names
        if auto_infer and class_names is None:
            class_names = self._infer_class_names(y_train)
        self.class_names: Optional[list[str]] = class_names

        # Initialize fitted attributes
        self.is_fitted_ = False
        self._fit()

    def _prepare_input_for_prediction(
        self, x: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Prepare input for FastWoe prediction (needs original categorical data as DataFrame)."""
        if isinstance(x, np.ndarray):
            if hasattr(self, "_original_X_train"):
                # Use original feature names
                feature_names = list(self._original_X_train.columns)
                if len(x.shape) == 1:
                    # type: ignore[bad-argument-type]
                    return pd.DataFrame([x], columns=feature_names)
                else:
                    # type: ignore[bad-argument-type]
                    return pd.DataFrame(x, columns=feature_names)
            else:
                # Fallback to generic names
                if len(x.shape) == 1:
                    return pd.DataFrame([x])
                else:
                    return pd.DataFrame(x)
        elif isinstance(x, pd.Series):
            return pd.DataFrame([x])
        else:
            return x  # Already DataFrame

    def _resolve_class_identifier(self, class_id: Union[int, str]) -> int:
        """Convert class identifier to class index."""
        if isinstance(class_id, (int, np.integer)):
            class_id = int(class_id)  # Convert numpy integers to Python int
            if class_id < 0 or class_id >= self.n_classes_:
                raise ValueError(f"Class index {class_id} out of range")
            return class_id
        elif isinstance(class_id, str):
            # type: ignore[unsupported-operation]
            if class_id not in self.class_names:
                raise ValueError(f"Class name '{class_id}' not found in {self.class_names}")
            # type: ignore[missing-attribute]
            return self.class_names.index(class_id)
        else:
            # Raise error if class identifier is not int or str
            raise ValueError(f"Class identifier must be int or str, got {type(class_id)}")

    def _fit(self):
        """Fit the Weight of Evidence explainer."""
        # Set basic properties
        self.n_samples_, self.n_features_ = self.X_train_.shape
        self.classes_ = np.unique(self.y_train_)
        self.n_classes_ = len(self.classes_)

        # Set default feature and class names if still None
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(self.n_features_)]
        if self.class_names is None:
            self.class_names = self._infer_class_names(self.y_train_, self.n_classes_)

        # Validate inputs
        self._validate_inputs()

        self.is_fitted_ = True
        return self

    def _validate_inputs(self):
        """Validate input parameters and data consistency."""
        # Check shapes
        if self.X_train_.shape[0] != self.y_train_.shape[0]:
            raise ValueError(
                f"X_train and y_train must have same number of samples. "
                f"Got {self.X_train_.shape[0]} and {self.y_train_.shape[0]}"
            )

        # Check feature names
        # type: ignore[bad-argument-type]
        if len(self.feature_names) != self.n_features_:
            raise ValueError(
                # type: ignore[bad-argument-type]
                f"feature_names length ({len(self.feature_names)}) must match "
                f"n_features ({self.n_features_})"
            )

        # Check class names
        # type: ignore[bad-argument-type]
        if len(self.class_names) != self.n_classes_:
            raise ValueError(
                # type: ignore[bad-argument-type]
                f"class_names length ({len(self.class_names)}) must match "
                f"n_classes ({self.n_classes_})"
            )

        # Check for minimum samples per class
        for cls in self.classes_:
            n_samples_cls = np.sum(self.y_train_ == cls)
            if n_samples_cls < 2:
                raise ValueError(
                    f"Class {cls} has only {n_samples_cls} samples. "
                    "Need at least 2 samples per class for covariance estimation."
                )

    def explain(
        self,
        x: Union[np.ndarray, pd.Series, pd.DataFrame],
        sample_idx: Optional[int] = None,
        class_to_explain: Optional[Union[int, str]] = None,
        true_labels: Optional[Union[np.ndarray, pd.Series]] = None,
        return_dict: bool = True,
    ) -> Optional[dict]:
        """
        Explain a prediction using Weight of Evidence.

        This method handles two usage patterns:
        1. explain(sample) - explain a single sample
        2. explain(dataset, index) - explain a sample from a dataset

        Parameters
        ----------
        x : array-like of shape (n_features,) or (n_samples, n_features)
            Input sample to explain, or dataset containing samples
        sample_idx : int, optional
            Index of sample to explain (when x is a dataset)
        class_to_explain : int or str, optional
            Class to explain. If None, uses predicted class
        true_labels : array-like of shape (n_samples,), optional
            True labels array (when using dataset + sample_idx)
        return_dict : bool, default=True
            Whether to return explanation as a dictionary. If False, prints nice
            formatted output and returns None

        Returns:
        -------
        dict or None
            If return_dict=True: Explanation dictionary with keys:
            - 'predicted_label': predicted label name
            - 'predicted_proba': prediction probabilities
            - 'explained_label': label being explained
            - 'total_woe': total Weight of Evidence
            - 'interpretation': human-readable interpretation
            - 'true_label': true label (if provided)
            - 'feature_contributions': individual feature contributions

            If return_dict=False: None (prints formatted explanation instead)
        """
        if not self.is_fitted_:
            raise ValueError("WeightOfEvidence must be fitted before explaining")

        if sample_idx is not None:
            # Dataset + index pattern: extract sample and explain
            if isinstance(x, pd.DataFrame):
                sample = x.iloc[sample_idx]
                sample_dict = {
                    k: v.item() if hasattr(v, "item") else v for k, v in sample.to_dict().items()
                }
            else:
                sample = x[sample_idx]
                sample_dict = {
                    k: v.item() if hasattr(v, "item") else v
                    # type: ignore[no-matching-overload]
                    for k, v in zip(self.feature_names, sample)
                }

            # Get true label from true_labels if provided
            if true_labels is not None and hasattr(true_labels, "iloc"):
                # true_labels is a pandas Series
                true_label = true_labels.iloc[sample_idx]  # type: ignore
            elif true_labels is not None:
                # true_labels is a numpy array or list
                true_label = true_labels[sample_idx]
            else:
                true_label = None

            # Get explanation using the core method
            explanation = self._explain_single_sample(
                # type: ignore[bad-argument-type]
                sample,
                class_to_explain,
                # type: ignore[bad-argument-type]
                true_label,
            )

            if not return_dict:
                sample_info = f"Sample Index: {sample_idx}\nOriginal Features: {sample_dict}"
                if true_label is not None:
                    sample_info += f"\nTrue Label: {self._format_class_name(true_label)}"
                sample_info += (
                    f"Predicted Label: {explanation['predicted_label']}\n"
                    f"Predicted Probabilities: {explanation['predicted_proba']}\n"
                    f"WOE Evidence: {explanation['total_woe']:.4f}\n"
                    f"Interpretation: {explanation['interpretation']}"
                )
                console.print(Panel(sample_info, title="Sample Explanation"))
                if "feature_contributions" in explanation:
                    self._render_centered_bars(explanation["feature_contributions"])
                return None
        else:
            # Single sample pattern: explain(sample)
            if isinstance(x, pd.DataFrame) and len(x) > 1:
                raise ValueError(
                    f"explain() received DataFrame with {len(x)} rows but no sample_idx. "
                    f"Use explain(dataset, sample_idx=i) or extract single sample: dataset.iloc[i]"
                )
            elif isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[0] > 1:
                raise ValueError(
                    f"explain() received array with shape {x.shape} but no sample_idx. "
                    f"Use explain(dataset, sample_idx=i) or extract single sample: dataset[i]"
                )
            # Extract true_label from true_labels for single sample
            true_label = None
            if true_labels is not None:
                if isinstance(x, pd.Series) and isinstance(true_labels, pd.Series):
                    if hasattr(x, "name") and x.name is not None and x.name in true_labels.index:
                        true_label = true_labels.loc[x.name]
                    else:
                        raise ValueError(
                            f"Cannot automatically extract true_label from true_labels. "
                            f"Sample index {getattr(x, 'name', 'unknown')} not found in true_labels index. "
                            f"Please use true_labels.iloc[position] or true_labels.loc[index] explicitly."
                        )
                else:
                    true_label = (
                        true_labels[-1] if hasattr(true_labels, "__getitem__") else true_labels
                    )
            # type: ignore[bad-argument-type]
            explanation = self._explain_single_sample(x, class_to_explain, true_label)
            if not return_dict:
                if isinstance(x, pd.Series):
                    sample_dict = {
                        k: v.item() if hasattr(v, "item") else v for k, v in x.to_dict().items()
                    }
                elif isinstance(x, np.ndarray):
                    sample_dict = {
                        k: v.item() if hasattr(v, "item") else v
                        # type: ignore[no-matching-overload]
                        for k, v in zip(self.feature_names, x)
                    }
                else:
                    raw_dict = x.to_dict() if hasattr(x, "to_dict") else dict(x)
                    sample_dict = {
                        k: v.item() if hasattr(v, "item") else v for k, v in raw_dict.items()
                    }
                sample_info = f"**Original Features**: {sample_dict}"
                if true_label is not None:
                    sample_info += f"\n**True Label**: {self._format_class_name(true_label)}"
                sample_info += f"""
                    Predicted Label: {explanation["predicted_label"]}
                    Predicted Probabilities: {explanation["predicted_proba"]}
                    WOE Evidence: {explanation["total_woe"]:.4f}
                    Interpretation: {explanation["interpretation"]}"""
                console.print(Panel(sample_info, title="Sample Explanation"))
                if "feature_contributions" in explanation:
                    self._render_centered_bars(explanation["feature_contributions"])
                return None
        return explanation

    def _explain_single_sample(
        self,
        x: Union[np.ndarray, pd.Series],
        class_to_explain: Optional[Union[int, str]] = None,
        true_label: Optional[Union[int, str]] = None,
    ) -> dict:
        explanation = self._explain_fastwoe(x, class_to_explain)
        if true_label is not None:
            formatted_true_label = self._format_class_name(true_label)
            explanation = {"true_label": formatted_true_label, **explanation}
        return explanation

    def _explain_fastwoe(
        self,
        x: Union[np.ndarray, pd.Series],
        class_to_explain: Optional[Union[int, str]] = None,
    ) -> dict:
        """Explain predictions for FastWoe classifier."""
        # Prepare input for FastWoe (needs DataFrame with categorical data)
        x_df = self._prepare_input_for_prediction(x)

        # Get predictions using FastWoe's native methods
        prediction = self.classifier.predict(x_df)[0]
        prediction_proba = self.classifier.predict_proba(x_df)[0]

        # Get WOE-transformed features for explanation
        x_woe = self.classifier.transform(x_df)
        woe_values = x_woe.iloc[0].to_dict()

        # Calculate total WOE (sum of individual WOE values)
        total_woe = sum(woe_values.values())

        # Determine class to explain
        if class_to_explain is None:
            # type: ignore[bad-assignment]
            class_to_explain = prediction

        # Convert to index if needed
        # type: ignore[bad-argument-type]
        class_idx = self._resolve_class_identifier(class_to_explain)

        return {
            "predicted_label": self._format_class_name(prediction),
            "predicted_proba": self._format_probabilities(prediction_proba),
            "explained_label": self._format_class_name(class_idx),
            "total_woe": float(total_woe),
            "interpretation": self._interpret_woe(total_woe),
            "feature_contributions": {k: float(v) for k, v in woe_values.items()},
        }

    def _format_class_name(self, class_value) -> str:
        """Format class name for clean display."""
        # Handle string class names that might already be formatted
        if isinstance(class_value, str):
            # If it's already one of our custom class names, return as-is
            if self.class_names and class_value in self.class_names:
                return class_value
            # If it's a numeric string, convert to int and process
            try:
                class_value = int(class_value)
            except (ValueError, TypeError):
                # If can't convert, return as-is
                return str(class_value)

        # Handle numpy integers
        if hasattr(class_value, "item") and callable(class_value.item):
            class_value = class_value.item()  # type: ignore

        # Now handle as integer index
        if isinstance(class_value, (int, float)):
            class_value = int(class_value)
            # Always use class names for consistency (both predicted and true labels)
            return (
                self.class_names[class_value]
                if self.class_names and class_value < len(self.class_names)
                else str(class_value)
            )

        # Fallback for any other type
        return str(class_value)

    def _format_probabilities(self, proba_array) -> dict:
        """Format probability array for clean display."""
        # Convert numpy floats to regular floats with 5 decimal places
        # Always use class names for consistency
        if self.class_names:
            return {
                name: round(float(prob), 5) for name, prob in zip(self.class_names, proba_array)
            }
        else:
            # Fallback to numeric labels if no class names
            return {str(i): round(float(prob), 5) for i, prob in enumerate(proba_array)}

    def _interpret_woe(self, woe: float) -> str:
        """Interpret WOE value in human-readable terms."""
        if woe > 2:
            return "Very strong evidence FOR the hypothesis"
        elif woe > 1:
            return "Strong evidence FOR the hypothesis"
        elif woe > 0.5:
            return "Moderate evidence FOR the hypothesis"
        elif woe > 0:
            return "Weak evidence FOR the hypothesis"
        elif woe > -0.5:
            return "Weak evidence AGAINST the hypothesis"
        elif woe > -1:
            return "Moderate evidence AGAINST the hypothesis"
        elif woe > -2:
            return "Strong evidence AGAINST the hypothesis"
        else:
            return "Very strong evidence AGAINST the hypothesis"

    def _render_centered_bars(self, contributions: dict, width: int = 20, min_bar: int = 1) -> None:
        """
        Render feature contributions as centered horizontal bars.
        """
        if not contributions:
            return

        table = Table(title="Feature Contributions", show_lines=True)
        table.add_column("Feature")
        table.add_column("WOE")
        table.add_column("Contribution", min_width=width)

        max_abs = max(abs(v) for v in contributions.values())
        half_width = width // 2  # half for each side

        # Sort features by value (largest to smallest)
        sorted_contributions = sorted(contributions.items(), key=lambda x: x[1], reverse=True)

        for feat, val in sorted_contributions:
            if max_abs == 0 or val >= 0 and val <= 0:
                left = " " * half_width
                right = " " * half_width
            elif val < 0:
                bar_len = max(int((abs(val) / max_abs) * half_width), min_bar)
                left = "█" * bar_len
                left = left.rjust(half_width)
                right = " " * half_width
            else:
                bar_len = max(int((val / max_abs) * half_width), min_bar)
                left = " " * half_width
                right = "█" * bar_len
                right = right.ljust(half_width)
            bar_ = left + right
            table.add_row(feat, f"{val:7.4f}", bar_)

        console.print(table)

    def summary(self) -> str:
        """Return a summary of the explainer."""
        if not self.is_fitted_:
            return "WeightOfEvidence (not fitted)"

        # Get class distribution from training data
        _, counts = np.unique(self.y_train_, return_counts=True)
        class_dist = counts / len(self.y_train_)

        # type: ignore[no-matching-overload]
        features_str = ", ".join(self.feature_names)
        # type: ignore[no-matching-overload]
        classes_str = ", ".join(self.class_names)

        return f"""
            Weight of Evidence Explainer (FastWoe)
            ==========================================
            Features: {self.n_features_} ({features_str})
            Classes: {self.n_classes_} ({classes_str})
            Training samples: {self.n_samples_}
            Class distribution: {[round(float(v), 3) for v in class_dist]}"""

    def explain_ci(
        self,
        x: Union[np.ndarray, pd.Series, pd.DataFrame],
        sample_idx: Optional[int] = None,
        class_to_explain: Optional[Union[int, str]] = None,
        true_labels: Optional[Union[np.ndarray, pd.Series]] = None,
        alpha: float = 0.05,
        return_dict: bool = True,
    ) -> Optional[dict]:
        """
        Explain predictions with confidence intervals using FastWoe's predict_ci.

        Parameters
        ----------
        x : array-like of shape (n_features,) or (n_samples, n_features)
            Input sample to explain, or dataset containing samples
        sample_idx : int, optional
            Index of sample to explain (when x is a dataset)
        class_to_explain : int or str, optional
            Class to explain. If None, uses predicted class
        true_labels : array-like of shape (n_samples,), optional
            True labels array (when using dataset + sample_idx)
        alpha : float, default=0.05
            Significance level (0.05 for 95% CI)
        return_dict : bool, default=True
            Whether to return explanation as a dictionary

        Returns:
        -------
        dict or None
            If return_dict=True: Explanation dictionary with CI information
            If return_dict=False: None (prints formatted explanation instead)
        """
        if not self.is_fitted_:
            raise ValueError("WeightOfEvidence must be fitted before explaining")

        if sample_idx is not None:
            if isinstance(x, pd.DataFrame):
                sample = x.iloc[sample_idx]
                sample_dict = {
                    col: sample[col].item() if hasattr(sample[col], "item") else sample[col]
                    for col in x.columns
                }
            else:
                sample = x[sample_idx]
                sample_dict = {
                    k: v.item() if hasattr(v, "item") else v
                    # type: ignore[no-matching-overload]
                    for k, v in zip(self.feature_names, sample)
                }
            if true_labels is not None and hasattr(true_labels, "iloc"):
                # true_labels is a pandas Series
                true_label = true_labels.iloc[sample_idx]  # type: ignore
            elif true_labels is not None:
                # true_labels is a numpy array or list
                true_label = true_labels[sample_idx]
            else:
                true_label = None
            explanation = self._explain_single_sample_ci(
                # type: ignore[bad-argument-type]
                sample,
                class_to_explain,
                # type: ignore[bad-argument-type]
                true_label,
                alpha,
            )
            if not return_dict:
                sample_info = f"Sample Index: {sample_idx}\nOriginal Features: {sample_dict}"
                if true_label is not None:
                    sample_info += f"\nTrue Label: {self._format_class_name(true_label)}"
                console.print(Panel(sample_info, title="Sample with Confidence Intervals"))
                self._print_ci_explanation(explanation)
                return None
        else:
            if isinstance(x, pd.DataFrame) and len(x) > 1:
                raise ValueError(
                    f"explain_ci() received DataFrame with {len(x)} rows but no sample_idx. "
                    f"Use explain_ci(dataset, sample_idx=i) or extract single sample: dataset.iloc[i]"
                )
            elif isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[0] > 1:
                raise ValueError(
                    f"explain_ci() received array with shape {x.shape} but no sample_idx. "
                    f"Use explain_ci(dataset, sample_idx=i) or extract single sample: dataset[i]"
                )
            true_label = None
            if true_labels is not None:
                if hasattr(true_labels, "__len__") and len(true_labels) > 1:
                    raise ValueError("true_labels has multiple values but no sample_idx specified")
                true_label = true_labels[0] if hasattr(true_labels, "__getitem__") else true_labels
            explanation = self._explain_single_sample_ci(
                # type: ignore[bad-argument-type]
                x,
                class_to_explain,
                # type: ignore[bad-argument-type]
                true_label,
                alpha,
            )
            if not return_dict:
                if isinstance(x, pd.Series):
                    sample_dict = {
                        col: x[col].item() if hasattr(x[col], "item") else x[col] for col in x.index
                    }
                elif isinstance(x, np.ndarray):
                    sample_dict = {
                        k: v.item() if hasattr(v, "item") else v
                        # type: ignore[no-matching-overload]
                        for k, v in zip(self.feature_names, x)
                    }
                else:
                    raw_dict = x.to_dict() if hasattr(x, "to_dict") else dict(x)
                    sample_dict = {
                        k: v.item() if hasattr(v, "item") else v for k, v in raw_dict.items()
                    }
                sample_info = f"**Original Features**: {sample_dict}"
                if true_label is not None:
                    sample_info += f"\n**True Label**: {self._format_class_name(true_label)}"
                console.print(Panel(sample_info, title="🎯 Sample with Confidence Intervals"))
                self._print_ci_explanation(explanation)
                return None
        return explanation

    def _explain_single_sample_ci(
        self,
        x: Union[np.ndarray, pd.Series],
        class_to_explain: Optional[Union[int, str]] = None,
        true_label: Optional[Union[int, str]] = None,
        alpha: float = 0.05,
    ) -> dict:
        base_explanation = self._explain_fastwoe(x, class_to_explain)
        x_df = self._prepare_input_for_prediction(x)
        ci_results = self.classifier.predict_ci(x_df, alpha=alpha)
        # ci_results is now a numpy array with shape (n_samples, 2): [ci_lower, ci_upper]
        ci_lower_probs = ci_results[:, 0]  # Lower confidence bounds
        ci_upper_probs = ci_results[:, 1]  # Upper confidence bounds
        if self.classifier.y_prior_ is None:
            raise ValueError("Classifier must be fitted before explaining confidence intervals")
        odds_prior = self.classifier.y_prior_ / (1 - self.classifier.y_prior_)
        eps = 1e-15
        ci_lower_safe = np.clip(ci_lower_probs, eps, 1 - eps)
        ci_upper_safe = np.clip(ci_upper_probs, eps, 1 - eps)
        logit_lower = np.log(ci_lower_safe / (1 - ci_lower_safe))  # type: ignore
        logit_upper = np.log(ci_upper_safe / (1 - ci_upper_safe))  # type: ignore
        woe_lower = logit_lower - np.log(odds_prior)
        woe_upper = logit_upper - np.log(odds_prior)
        ci_conservative = {
            "predicted_label": "Positive" if woe_lower > 0 else "Negative",
            "predicted_proba_ci": float(ci_lower_probs[0].item()),
            "total_woe": float(woe_lower.item()),
            "interpretation": self._interpret_woe(woe_lower),
        }
        ci_optimistic = {
            "predicted_label": "Positive" if woe_upper > 0 else "Negative",
            "predicted_proba_ci": float(ci_upper_probs[0].item()),
            "total_woe": float(woe_upper.item()),
            "interpretation": self._interpret_woe(woe_upper),
            "scenario": "Optimistic (Upper CI)",
        }
        explanation = {
            **base_explanation,
            "confidence_level": f"{(1 - alpha) * 100:.0f}%",
            "ci_conservative": ci_conservative,
            "ci_optimistic": ci_optimistic,
            "uncertainty_range": {
                "woe_range": float((woe_upper - woe_lower).item()),
                "prob_range": float((ci_upper_probs[0] - ci_lower_probs[0]).item()),
            },
        }
        if true_label is not None:
            formatted_true_label = self._format_class_name(true_label)
            explanation = {"true_label": formatted_true_label, **explanation}
        return explanation

    def _print_ci_explanation(self, explanation: dict) -> None:
        # sourcery skip: extract-duplicate-method
        """Print a formatted confidence interval explanation."""
        # Header information
        header_info = (
            f"Predicted Label: {explanation['predicted_label']}\n"
            f"Predicted Probabilities: {explanation['predicted_proba']}\n"
            f"Confidence Level: {explanation['confidence_level']}"
        )

        console.print(Panel(header_info, title="Inference Summary"))

        # Create scenarios comparison table
        scenarios_table = Table(
            title="Confidence Intervals",
            show_header=True,
        )
        scenarios_table.add_column("Scenario")
        scenarios_table.add_column("Predicted Label")
        scenarios_table.add_column("WOE Evidence")
        scenarios_table.add_column("Probability")
        scenarios_table.add_column("Interpretation", max_width=70)

        # Conservative scenario row (first)
        ci_cons = explanation["ci_conservative"]
        cons_predicted_label = "Positive" if ci_cons["total_woe"] > 0 else "Negative"
        scenarios_table.add_row(
            "Lower Bound",
            cons_predicted_label,
            f"{ci_cons['total_woe']:.4f}",
            f"{ci_cons['predicted_proba_ci']:.4f}",
            ci_cons["interpretation"],
        )

        # Point estimate row (middle)
        base_prob = list(explanation["predicted_proba"].values())[1]  # Get positive class prob
        point_predicted_label = "Positive" if explanation["total_woe"] > 0 else "Negative"
        scenarios_table.add_row(
            "Point Estimate",
            point_predicted_label,
            f"{explanation['total_woe']:.4f}",
            f"{base_prob:.4f}",
            explanation["interpretation"],
        )

        # Optimistic scenario row (last)
        ci_opt = explanation["ci_optimistic"]
        opt_predicted_label = "Positive" if ci_opt["total_woe"] > 0 else "Negative"
        scenarios_table.add_row(
            "Upper Bound",
            opt_predicted_label,
            f"{ci_opt['total_woe']:.4f}",
            f"{ci_opt['predicted_proba_ci']:.4f}",
            ci_opt["interpretation"],
        )

        console.print(scenarios_table)

        # Uncertainty summary table
        unc = explanation["uncertainty_range"]
        uncertainty_table = Table(title="Uncertainty Summary", show_header=True)
        uncertainty_table.add_column("Metric")
        uncertainty_table.add_column("Range (±)")

        uncertainty_table.add_row("WOE Range", f"{unc['woe_range'] / 2:.4f}")
        uncertainty_table.add_row("Probability Range", f"{unc['prob_range'] / 2:.4f}")

        console.print(uncertainty_table)

        # Feature contributions if available
        if "feature_contributions" in explanation:
            self._render_centered_bars(explanation["feature_contributions"])

    def predict_ci(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        alpha: float = 0.05,
        return_probabilities: bool = False,
    ) -> dict:
        """
        Make predictions using confidence interval bounds for decision thresholds.

        This method provides three prediction scenarios:
        - Base Estimate: Standard WOE > 0 prediction
        - Conservative (Lower Bound): Uses lower CI bound > 0 for prediction
        - Optimistic (Upper Bound): Uses upper CI bound > 0 for prediction

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples to predict
        alpha : float, default=0.05
            Significance level (0.05 for 95% CI)
        return_probabilities : bool, default=False
            Whether to include probability estimates in the output

        Returns:
        -------
        dict
            Dictionary containing prediction scenarios with rich output:
            - 'base_estimate': Point estimate predictions
            - 'lower_bound': Conservative predictions (lower CI) with full details
            - 'upper_bound': Optimistic predictions (upper CI) with full details
            - 'uncertainty_summary': Uncertainty metrics and confidence levels

            Each scenario contains:
            - 'predictions': Binary predictions (0/1)
            - 'predicted_labels': Formatted class names
            - 'predicted_proba': Probability dictionaries
            - 'woe_scores': WOE evidence values
            - 'interpretation': Human-readable WOE interpretations
            - 'scenario': Description of the prediction scenario
        """
        if not self.is_fitted_:
            raise ValueError("WeightOfEvidence must be fitted before predicting")

        # Prepare input for FastWoe
        if isinstance(X, np.ndarray):
            # Convert to DataFrame with proper feature names
            if hasattr(self, "_original_X_train"):
                feature_names = list(self._original_X_train.columns)
            else:
                feature_names = self.feature_names
            # type: ignore[bad-argument-type], no-matching-overload
            X_df = pd.DataFrame(X, columns=list(feature_names))
        else:
            X_df = X.copy()

        # Get base predictions and probabilities
        base_predictions = self.classifier.predict(X_df)
        base_probabilities = self.classifier.predict_proba(X_df)

        # Get confidence intervals from FastWoe
        ci_results = self.classifier.predict_ci(X_df, alpha=alpha)
        # ci_results is now a numpy array with shape (n_samples, 2): [ci_lower, ci_upper]
        ci_lower_probs = ci_results[:, 0]  # Lower confidence bounds
        ci_upper_probs = ci_results[:, 1]  # Upper confidence bounds

        # Get WOE scores
        X_woe = self.classifier.transform(X_df)
        base_woe_scores = X_woe.sum(axis=1).values

        # Calculate WOE confidence intervals
        if self.classifier.y_prior_ is None:
            raise ValueError("Classifier must be fitted before explaining confidence intervals")
        odds_prior = self.classifier.y_prior_ / (1 - self.classifier.y_prior_)

        # Convert probability bounds back to WOE bounds
        eps = 1e-15
        ci_lower_safe = np.clip(ci_lower_probs, eps, 1 - eps)
        ci_upper_safe = np.clip(ci_upper_probs, eps, 1 - eps)

        logit_lower = np.log(ci_lower_safe / (1 - ci_lower_safe))  # type: ignore
        logit_upper = np.log(ci_upper_safe / (1 - ci_upper_safe))  # type: ignore

        # Remove prior to get WOE bounds
        woe_lower = logit_lower - np.log(odds_prior)
        woe_upper = logit_upper - np.log(odds_prior)

        # Make predictions using CI bounds for thresholds
        conservative_predictions = (woe_lower > 0).astype(int)  # Lower bound > 0
        optimistic_predictions = (woe_upper > 0).astype(int)  # Upper bound > 0

        # Calculate uncertainty metrics
        prediction_agreement = np.mean(conservative_predictions == optimistic_predictions)
        woe_uncertainty = woe_upper - woe_lower
        prob_uncertainty = ci_upper_probs - ci_lower_probs

        # Format predicted labels for each scenario
        base_labels = [self._format_class_name(pred) for pred in base_predictions]
        conservative_labels = [self._format_class_name(pred) for pred in conservative_predictions]
        optimistic_labels = [self._format_class_name(pred) for pred in optimistic_predictions]

        # Format probabilities for each scenario
        base_probs = [self._format_probabilities(prob) for prob in base_probabilities]
        conservative_probs = [
            self._format_probabilities(np.array([1 - p, p])) for p in ci_lower_probs
        ]
        optimistic_probs = [
            self._format_probabilities(np.array([1 - p, p])) for p in ci_upper_probs
        ]

        # Generate interpretations for each scenario
        base_interpretations = [self._interpret_woe(woe) for woe in base_woe_scores]
        conservative_interpretations = [self._interpret_woe(woe) for woe in woe_lower]
        optimistic_interpretations = [self._interpret_woe(woe) for woe in woe_upper]

        # Build result dictionary
        result = {
            "base_estimate": {
                "predictions": base_predictions,
                "predicted_labels": base_labels,
                "predicted_proba": base_probs,
                "woe_scores": base_woe_scores,
                "interpretation": base_interpretations,
                "scenario": "Standard WOE > 0 threshold",
            },
            "lower_bound": {
                "predictions": conservative_predictions,
                "predicted_labels": conservative_labels,
                "predicted_proba": conservative_probs,
                "woe_scores": woe_lower,
                "interpretation": conservative_interpretations,
                "scenario": "Conservative (Lower CI > 0 threshold)",
            },
            "upper_bound": {
                "predictions": optimistic_predictions,
                "predicted_labels": optimistic_labels,
                "predicted_proba": optimistic_probs,
                "woe_scores": woe_upper,
                "interpretation": optimistic_interpretations,
                "scenario": "Optimistic (Upper CI > 0 threshold)",
            },
            "uncertainty_summary": {
                "confidence_level": f"{(1 - alpha) * 100:.0f}%",
                "prediction_agreement": float(prediction_agreement),
                "mean_woe_uncertainty": float(np.mean(woe_uncertainty)),
                "mean_prob_uncertainty": float(np.mean(prob_uncertainty)),
                "max_woe_uncertainty": float(np.max(woe_uncertainty)),
                "max_prob_uncertainty": float(np.max(prob_uncertainty)),
            },
        }

        # Add probabilities if requested
        if return_probabilities:
            # type: ignore[unsupported-operation]
            result["base_estimate"]["probabilities"] = base_probabilities
            # type: ignore[unsupported-operation]
            result["lower_bound"]["probabilities"] = np.column_stack(
                [1 - ci_lower_probs, ci_lower_probs]
            )
            # type: ignore[unsupported-operation]
            result["upper_bound"]["probabilities"] = np.column_stack(
                [1 - ci_upper_probs, ci_upper_probs]
            )

        return result
