"""Tests for interpret_fastwoe module."""

import warnings

import numpy as np
import pandas as pd
import pytest

from fastwoe import FastWoe, WeightOfEvidence


class TestWeightOfEvidence:
    """Test cases for WeightOfEvidence class with FastWoe."""

    @pytest.fixture
    def sample_data(self):
        """Create sample categorical data for FastWoe testing."""
        np.random.seed(42)

        # Create categorical data suitable for FastWoe
        n_samples = 200
        data = {
            "cat_feature_1": np.random.choice(["A", "B", "C"], n_samples),
            "cat_feature_2": np.random.choice(["X", "Y"], n_samples),
            "num_feature_1": np.random.normal(0, 1, n_samples),
            "num_feature_2": np.random.normal(0, 1, n_samples),
        }

        X = pd.DataFrame(data)
        # Create target with some correlation to features
        y = ((X["num_feature_1"] > 0) & (X["cat_feature_1"] == "A")).astype(int)

        # Train FastWoe classifier - suppress expected warning about numerical features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            clf = FastWoe()
            clf.fit(X, y)

        return X, y, clf

    def test_init_basic(self, sample_data):
        """Test basic initialization with FastWoe."""
        X, y, clf = sample_data

        woe = WeightOfEvidence(clf, X, y)

        assert woe.is_fitted_
        assert woe.classifier == clf
        assert woe.n_samples_ == len(X)
        assert woe.n_features_ == X.shape[1]
        assert woe.n_classes_ == 2
        assert woe.feature_names is not None
        assert woe.class_names is not None
        assert len(woe.feature_names) == X.shape[1]
        assert len(woe.class_names) == 2

    def test_init_with_custom_params(self, sample_data):
        """Test initialization with custom parameters."""
        X, y, clf = sample_data

        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        class_names = ["negative", "positive"]

        woe = WeightOfEvidence(
            clf,
            X,
            y,
            feature_names=feature_names,
            class_names=class_names,
        )

        assert woe.feature_names == feature_names
        assert woe.class_names == class_names

    def test_init_with_pandas(self, sample_data):
        """Test initialization with pandas DataFrames."""
        X, y, clf = sample_data

        y_series = pd.Series(y)

        woe = WeightOfEvidence(clf, X, y_series)

        assert woe.is_fitted_
        assert woe.n_samples_ == len(X)
        assert woe.n_features_ == X.shape[1]

    def test_init_invalid_shapes(self, sample_data):
        """Test initialization with mismatched shapes."""
        X, y, clf = sample_data

        with pytest.raises(ValueError, match="must have same number of samples"):
            WeightOfEvidence(clf, X, y[:-10])

    def test_init_invalid_feature_names(self, sample_data):
        """Test initialization with wrong number of feature names."""
        X, y, clf = sample_data

        with pytest.raises(ValueError, match="feature_names length"):
            WeightOfEvidence(clf, X, y, feature_names=["feat1", "feat2"])

    def test_init_invalid_class_names(self, sample_data):
        """Test initialization with wrong number of class names."""
        X, y, clf = sample_data

        with pytest.raises(ValueError, match="class_names length"):
            WeightOfEvidence(clf, X, y, class_names=["class1"])

    def test_init_insufficient_samples(self, sample_data):
        """Test initialization with insufficient samples per class."""
        X, y, clf = sample_data

        # Create data with only one sample for one class
        X_small = X[:2]
        y_small = np.array([0, 1])

        with pytest.raises(ValueError, match="Need at least 2 samples per class"):
            WeightOfEvidence(clf, X_small, y_small)

    def test_multiclass_support(self):
        """Test that multiclass classification is now supported."""
        np.random.seed(42)

        n_samples = 100
        data = {
            "cat_feature_1": np.random.choice(["A", "B", "C"], n_samples),
            "num_feature_1": np.random.normal(0, 1, n_samples),
        }

        X = pd.DataFrame(data)
        # Create 3-class target (now supported)
        y = np.random.choice([0, 1, 2], n_samples)

        clf = FastWoe()
        # Multiclass is now supported
        clf.fit(X, y)
        assert clf.is_multiclass_target

    def test_resolve_class_identifier(self, sample_data):
        """Test class identifier resolution."""
        X, y, clf = sample_data

        woe = WeightOfEvidence(clf, X, y, class_names=["neg", "pos"])

        # Test with integer
        assert woe._resolve_class_identifier(0) == 0
        assert woe._resolve_class_identifier(1) == 1

        # Test with string
        assert woe._resolve_class_identifier("neg") == 0
        assert woe._resolve_class_identifier("pos") == 1

        # Test with invalid string
        with pytest.raises(ValueError, match="Class name 'invalid' not found"):
            woe._resolve_class_identifier("invalid")

    def test_explain_basic(self, sample_data):
        """Test basic explanation functionality."""
        X, y, clf = sample_data

        woe = WeightOfEvidence(clf, X, y)
        sample = X.iloc[0]

        explanation = woe.explain(sample)

        assert isinstance(explanation, dict)
        assert "predicted_label" in explanation
        assert "predicted_proba" in explanation
        assert "total_woe" in explanation
        assert "interpretation" in explanation
        assert "feature_contributions" in explanation
        assert isinstance(explanation["total_woe"], float)
        assert not np.isnan(explanation["total_woe"])

    def test_explain_with_specific_class(self, sample_data):
        """Test explanation with specific class to explain."""
        X, y, clf = sample_data

        woe = WeightOfEvidence(clf, X, y)
        sample = X.iloc[0]

        explanation = woe.explain(sample, class_to_explain=1)
        assert explanation is not None

        assert explanation["explained_label"] in ["1", "Positive"]

    def test_explain_not_fitted(self, sample_data):
        """Test that explain raises error when not fitted."""
        X, y, clf = sample_data

        woe = WeightOfEvidence.__new__(WeightOfEvidence)
        woe.is_fitted_ = False

        with pytest.raises(ValueError, match="must be fitted before explaining"):
            woe.explain(X.iloc[0])

    def test_summary(self, sample_data):
        """Test summary method."""
        X, y, clf = sample_data

        woe = WeightOfEvidence(clf, X, y)
        summary = woe.summary()

        assert isinstance(summary, str)
        assert "FastWoe" in summary
        assert str(woe.n_features_) in summary
        assert str(woe.n_samples_) in summary

    def test_summary_not_fitted(self, sample_data):
        """Test summary when not fitted."""
        woe = WeightOfEvidence.__new__(WeightOfEvidence)
        woe.is_fitted_ = False

        summary = woe.summary()
        assert "not fitted" in summary

    # pylint: disable=protected-access
    def test_interpret_woe_thresholds(self, sample_data):
        """Test WoE interpretation thresholds."""
        X, y, clf = sample_data

        woe = WeightOfEvidence(clf, X, y)

        # Test different WoE values
        assert "Very strong evidence FOR" in woe._interpret_woe(3.0)
        assert "Strong evidence FOR" in woe._interpret_woe(1.5)
        assert "Moderate evidence FOR" in woe._interpret_woe(0.7)
        assert "Weak evidence FOR" in woe._interpret_woe(0.2)
        assert "Weak evidence AGAINST" in woe._interpret_woe(-0.2)
        assert "Moderate evidence AGAINST" in woe._interpret_woe(-0.7)
        assert "Strong evidence AGAINST" in woe._interpret_woe(-1.5)
        assert "Very strong evidence AGAINST" in woe._interpret_woe(-3.0)

    def test_pandas_series_input(self, sample_data):
        """Test explanation with pandas Series input."""
        X, y, clf = sample_data

        woe = WeightOfEvidence(clf, X, y)
        sample_series = X.iloc[0]

        explanation = woe.explain(sample_series)

        assert isinstance(explanation, dict)
        assert "total_woe" in explanation

    def test_explain_input_validation(self, sample_data):
        """Test input validation for explain method."""
        X, y, clf = sample_data

        woe = WeightOfEvidence(clf, X, y)

        # Test error when passing full dataset without sample_idx
        with pytest.raises(ValueError, match="no sample_idx"):
            woe.explain(X)

        # Test error when passing array without sample_idx
        X_array = X.values
        with pytest.raises(ValueError, match="no sample_idx"):
            woe.explain(X_array)

    def test_class_name_formatting_consistency(self, sample_data):
        """Test consistent class name formatting."""
        X, y, clf = sample_data

        woe = WeightOfEvidence(clf, X, y, class_names=["negative", "positive"])

        # Test with different input types
        assert woe._format_class_name(0) == "negative"
        assert woe._format_class_name(1) == "positive"
        assert woe._format_class_name("negative") == "negative"
        assert woe._format_class_name("positive") == "positive"

    def test_non_fastwoe_classifier_error(self, sample_data):
        """Test that non-FastWoe classifiers are rejected."""
        X, y, _ = sample_data

        # Mock a non-FastWoe classifier
        class MockClassifier:
            """Mock classifier for testing."""

            def predict(self, X):
                """Mock predict method."""
                return np.zeros(len(X))

            def predict_proba(self, X):
                """Mock predict_proba method."""
                return np.ones((len(X), 2)) * 0.5

        mock_clf = MockClassifier()

        with pytest.raises(ValueError, match="Only FastWoe classifiers are supported"):
            WeightOfEvidence(mock_clf, X, y)

    def test_true_labels_consistency(self, sample_data):
        """Test that true_labels parameter works consistently across methods."""
        X, y, clf = sample_data

        woe = WeightOfEvidence(clf, X, y, class_names=["Negative", "Positive"])
        sample = X.iloc[0]
        true_labels = pd.Series([1, 0, 1, 0, 1])  # Some test labels

        # Test explain with true_labels
        explanation = woe.explain(X.iloc[:5], sample_idx=0, true_labels=true_labels)
        assert explanation is not None
        assert explanation["true_label"] == "Positive"  # Should use class name

        # Test explain_ci with true_labels
        explanation_ci = woe.explain_ci(X.iloc[:5], sample_idx=1, true_labels=true_labels)
        assert explanation_ci is not None
        assert explanation_ci["true_label"] == "Negative"  # Should use class name

        # Test with numpy array
        true_labels_np = np.array([1, 0, 1, 0, 1])
        explanation_np = woe.explain(X.iloc[:5], sample_idx=2, true_labels=true_labels_np)
        assert explanation_np is not None
        assert explanation_np["true_label"] == "Positive"

        # Test with list
        true_labels_list = np.array([1, 0, 1, 0, 1])
        explanation_list = woe.explain(X.iloc[:5], sample_idx=3, true_labels=true_labels_list)
        assert explanation_list is not None
        assert explanation_list["true_label"] == "Negative"


class TestIntegration:
    """Integration tests for WeightOfEvidence with FastWoe."""

    def test_with_custom_names(self):
        """Test with custom feature and class names."""
        np.random.seed(42)

        data = {
            "income": np.random.choice(["low", "medium", "high"], 100),
            "age_group": np.random.choice(["young", "old"], 100),
            "score": np.random.normal(0, 1, 100),
        }

        X = pd.DataFrame(data)
        y = (X["score"] > 0).astype(int)

        clf = FastWoe()
        clf.fit(X, y)

        woe = WeightOfEvidence(
            clf,
            X,
            y,
            feature_names=["Income", "Age_Group", "Credit_Score"],
            class_names=["Reject", "Accept"],
        )

        explanation = woe.explain(X.iloc[0])
        assert explanation is not None

        assert explanation["predicted_label"] in ["Reject", "Accept"]
        # Note: FastWoe preserves original column names in feature_contributions
        assert "income" in explanation["feature_contributions"]
        assert "age_group" in explanation["feature_contributions"]
        assert "score" in explanation["feature_contributions"]

    def test_explain_unified_api(self):
        """Test unified explain API with different input methods."""
        np.random.seed(42)

        data = {
            "feature1": np.random.choice(["A", "B"], 50),
            "feature2": np.random.normal(0, 1, 50),
        }

        X = pd.DataFrame(data)
        y = np.random.choice([0, 1], 50)

        clf = FastWoe()
        clf.fit(X, y)
        woe = WeightOfEvidence(clf, X, y)

        # Test single sample explanation
        explanation1 = woe.explain(X.iloc[0])
        assert isinstance(explanation1, dict)

        # Test dataset + index explanation
        explanation2 = woe.explain(X, sample_idx=0)
        assert isinstance(explanation2, dict)

        # Both should give same result for same sample
        assert explanation1["total_woe"] == explanation2["total_woe"]

    def test_explain_return_dict_false(self):
        """Test explain with return_dict=False (print mode)."""
        np.random.seed(42)

        data = {
            "feature1": np.random.choice(["A", "B"], 30),
            "feature2": np.random.normal(0, 1, 30),
        }

        X = pd.DataFrame(data)
        y = np.random.choice([0, 1], 30)

        clf = FastWoe()
        clf.fit(X, y)
        woe = WeightOfEvidence(clf, X, y)

        # Should return None and print output
        result = woe.explain(X.iloc[0], return_dict=False)
        assert result is None

        # Test with dataset + index
        result = woe.explain(X, sample_idx=0, return_dict=False)
        assert result is None

    def test_automatic_fastwoe_creation(self):
        """Test automatic FastWoe classifier creation."""
        np.random.seed(42)

        data = {
            "cat_feature": np.random.choice(["A", "B", "C"], 100),
            "num_feature": np.random.normal(0, 1, 100),
        }

        X = pd.DataFrame(data)
        y = np.random.choice([0, 1], 100)

        # Should automatically create FastWoe
        woe = WeightOfEvidence(None, X, y)

        assert woe.is_fitted_
        assert hasattr(woe.classifier, "mappings_")  # FastWoe attribute

    def test_automatic_fastwoe_missing_data(self):
        """Test error when trying auto-creation without data."""
        with pytest.raises(ValueError, match="X_train and y_train must be provided"):
            WeightOfEvidence(None, None, None)

    def test_feature_contributions_structure(self):
        """Test that feature contributions are properly structured."""
        np.random.seed(42)

        data = {
            "feature_a": np.random.choice(["X", "Y"], 50),
            "feature_b": np.random.normal(0, 1, 50),
            "feature_c": np.random.choice(["P", "Q", "R"], 50),
        }

        X = pd.DataFrame(data)
        y = np.random.choice([0, 1], 50)

        clf = FastWoe()
        clf.fit(X, y)
        woe = WeightOfEvidence(clf, X, y)

        explanation = woe.explain(X.iloc[0])
        assert explanation is not None

        contributions = explanation["feature_contributions"]

        # Should have contribution for each feature
        assert len(contributions) == len(X.columns)

        # sourcery skip: no-loop-in-tests
        for contribution in contributions.values():
            assert isinstance(contribution, (int, float))
            assert not np.isnan(contribution)

        # Total WoE should equal sum of contributions
        total_contribution = sum(contributions.values())
        assert abs(explanation["total_woe"] - total_contribution) < 1e-10


class TestCoverageImprovements:
    """Tests to improve coverage for edge cases and error handling."""

    @pytest.fixture
    def mock_fastwoe_without_training_data(self):
        """Create a mock FastWoe that doesn't store training data."""

        class MockFastWoe:
            def __init__(self):
                # Mock FastWoe attributes but without training data
                self.mappings_ = {"feature1": None, "feature2": None}
                self.y_prior_ = 0.3  # Required for _is_fastwoe_classifier
                self.is_fitted_ = True
                # Deliberately missing _X_train_original and _y_train

            def predict(self, X):
                return np.zeros(len(X))

            def predict_proba(self, X):
                return np.column_stack([np.ones(len(X)) * 0.6, np.ones(len(X)) * 0.4])

            def transform(self, X):
                # Return mock WOE scores
                return pd.DataFrame(
                    {
                        "feature1": np.ones(len(X)) * 0.5,
                        "feature2": np.ones(len(X)) * -0.3,
                    }
                )

        return MockFastWoe()

    def test_extract_training_data_failure(self, mock_fastwoe_without_training_data):
        """Test when FastWoe doesn't store training data."""
        # Add the _is_fastwoe_classifier method to make it pass FastWoe validation
        mock_fastwoe_without_training_data.__class__.__name__ = "FastWoe"
        mock_fastwoe_without_training_data.__module__ = "fastwoe.fastwoe"

        # Should fail when trying to auto-infer training data
        with pytest.raises(ValueError, match="Cannot auto-infer training data"):
            WeightOfEvidence(mock_fastwoe_without_training_data, auto_infer=True)

    def test_infer_feature_names_fallbacks(self):
        """Test feature name inference fallbacks."""

        # Test with classifier that has feature_names_
        class MockClassifierWithFeatureNames:
            """Mock classifier with feature names."""

            feature_names_ = ["feat1", "feat2"]

        result = WeightOfEvidence._infer_feature_names(None, MockClassifierWithFeatureNames())
        assert result == ["feat1", "feat2"]

        # Test with classifier that has mappings_ (FastWoe case)
        class MockClassifierWithMappings:
            """Mock classifier with mappings."""

            mappings_ = {"map1": None, "map2": None}

        result = WeightOfEvidence._infer_feature_names(None, MockClassifierWithMappings())
        assert result == ["map1", "map2"]

        # Test fallback to generic names
        class MockClassifierMinimal:
            """Mock classifier with minimal attributes."""

            pass

        # pylint: disable=protected-access
        class MockXTrain:
            """Mock X_train with shape."""

            shape = (100, 3)

        result = WeightOfEvidence._infer_feature_names(MockXTrain(), MockClassifierMinimal())
        assert result == ["feature_0", "feature_1", "feature_2"]

        # Test with list-like X_train
        X_train_list = [[1, 2, 3], [4, 5, 6]]
        result = WeightOfEvidence._infer_feature_names(X_train_list, MockClassifierMinimal())
        assert result == ["feature_0", "feature_1", "feature_2"]

    def test_prepare_input_fallback_without_original_train(self):
        """Test input preparation when no original training data is available."""
        np.random.seed(42)

        data = pd.DataFrame(
            {
                "feature1": np.random.choice(["A", "B"], 50),
                "feature2": np.random.choice(["X", "Y"], 50),
            }
        )
        y = np.random.choice([0, 1], 50)

        clf = FastWoe()
        clf.fit(data, y)

        # Create explainer and remove _original_X_train to test fallback
        woe = WeightOfEvidence(clf, data, y)
        delattr(woe, "_original_X_train")

        # Test with numpy array (should fallback to generic DataFrame)
        x_array = np.array(["A", "X"])
        result = woe._prepare_input_for_prediction(x_array)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.shape[1] == 2

        # Test with 2D numpy array
        x_array_2d = np.array([["A", "X"], ["B", "Y"]])
        result = woe._prepare_input_for_prediction(x_array_2d)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)

    def test_class_identifier_resolution_errors(self):
        """Test error cases in class identifier resolution."""
        np.random.seed(42)

        data = pd.DataFrame(
            {
                "feature1": np.random.choice(["A", "B"], 30),
                "feature2": np.random.choice(["X", "Y"], 30),
            }
        )
        y = np.random.choice([0, 1], 30)

        clf = FastWoe()
        clf.fit(data, y)
        woe = WeightOfEvidence(clf, data, y, class_names=["Negative", "Positive"])

        # Test out of range class index
        with pytest.raises(ValueError, match="Class index 5 out of range"):
            woe._resolve_class_identifier(5)

        with pytest.raises(ValueError, match="Class index -1 out of range"):
            woe._resolve_class_identifier(-1)

        # Test invalid class name
        with pytest.raises(ValueError, match="Class name 'InvalidClass' not found"):
            woe._resolve_class_identifier("InvalidClass")

        # Test invalid type
        with pytest.raises(ValueError, match="Class identifier must be int or str"):
            woe._resolve_class_identifier([1, 2])  # type: ignore

    def test_validation_insufficient_samples_per_class(self):
        """Test validation error when insufficient samples per class."""
        # Create data with only 1 sample for class 1
        data = pd.DataFrame({"feature1": ["A", "B", "A"], "feature2": ["X", "Y", "X"]})
        y = pd.Series([0, 0, 1])  # Only 1 sample for class 1

        clf = FastWoe()
        clf.fit(data, y)

        with pytest.raises(ValueError, match="Class 1 has only 1 samples"):
            WeightOfEvidence(clf, data, y)

    def test_explain_multi_sample_input_errors(self):
        """Test errors when passing multiple samples without sample_idx."""
        np.random.seed(42)

        data = pd.DataFrame(
            {
                "feature1": np.random.choice(["A", "B"], 50),
                "feature2": np.random.choice(["X", "Y"], 50),
            }
        )
        y = np.random.choice([0, 1], 50)

        clf = FastWoe()
        clf.fit(data, y)
        woe = WeightOfEvidence(clf, data, y)

        # Test DataFrame with multiple rows, no sample_idx
        with pytest.raises(ValueError, match="received DataFrame with 50 rows but no sample_idx"):
            woe.explain(data)

        # Test 2D numpy array with multiple rows, no sample_idx
        data_array = data.values
        with pytest.raises(
            ValueError, match="received array with shape \\(50, 2\\) but no sample_idx"
        ):
            woe.explain(data_array)

    def test_true_labels_extraction_edge_cases(self):
        """Test edge cases in true labels extraction."""
        np.random.seed(42)

        data = pd.DataFrame(
            {
                "feature1": np.random.choice(["A", "B"], 30),
                "feature2": np.random.choice(["X", "Y"], 30),
            }
        )
        y = np.random.choice([0, 1], 30)

        clf = FastWoe()
        clf.fit(data, y)
        woe = WeightOfEvidence(clf, data, y)

        # Create a Series with a specific name/index
        sample = data.iloc[5].copy()
        sample.name = "sample_5"

        # Create true_labels Series with different index
        true_labels = pd.Series([1, 0, 1], index=["sample_1", "sample_2", "sample_3"])

        # Should raise error when index doesn't match
        with pytest.raises(ValueError, match="Cannot automatically extract true_label"):
            woe.explain(sample, true_labels=true_labels)

        # Test with non-pandas true_labels (should use fallback)
        true_labels_list = np.array([0, 1, 0])
        explanation = woe.explain(sample, true_labels=true_labels_list)
        assert explanation is not None
        assert explanation["true_label"] == "Negative"  # Should use last element (0)

    def test_format_class_name_edge_cases(self):
        """Test edge cases in class name formatting."""
        np.random.seed(42)

        data = pd.DataFrame(
            {
                "feature1": np.random.choice(["A", "B"], 30),
                "feature2": np.random.choice(["X", "Y"], 30),
            }
        )
        y = np.random.choice([0, 1], 30)

        clf = FastWoe()
        clf.fit(data, y)
        woe = WeightOfEvidence(clf, data, y, class_names=["Negative", "Positive"])

        # Test string that can't be converted to int
        result = woe._format_class_name("invalid_string")
        assert result == "invalid_string"

        # Test numpy integer
        numpy_int = np.int64(1)
        result = woe._format_class_name(numpy_int)
        assert result == "Positive"

        # Test float input
        result = woe._format_class_name(1.0)
        assert result == "Positive"

        # Test class value beyond available class names
        woe_no_names = WeightOfEvidence(clf, data, y, class_names=None)
        result = woe_no_names._format_class_name(5)
        assert result == "5"

    def test_format_probabilities_without_class_names(self):
        """Test probability formatting when no class names are available."""
        np.random.seed(42)

        data = pd.DataFrame(
            {
                "feature1": np.random.choice(["A", "B"], 30),
                "feature2": np.random.choice(["X", "Y"], 30),
            }
        )
        y = np.random.choice([0, 1], 30)

        clf = FastWoe()
        clf.fit(data, y)

        # Create explainer and manually set class_names to None to test fallback
        woe = WeightOfEvidence(clf, data, y)
        woe.class_names = None

        proba_array = np.array([0.3, 0.7])
        result = woe._format_probabilities(proba_array)

        assert result == {"0": 0.3, "1": 0.7}

    def test_summary_not_fitted(self):
        """Test summary method when explainer is not fitted."""
        # Create a mock explainer that's not fitted
        woe = WeightOfEvidence.__new__(WeightOfEvidence)
        woe.is_fitted_ = False

        result = woe.summary()
        assert result == "WeightOfEvidence (not fitted)"

    def test_automatic_fastwoe_creation_with_more_data(self):
        """Test automatic FastWoe creation with sufficient data."""
        # Create sufficient data to pass validation (at least 2 samples per class)
        data = {"cat1": ["A", "B", "A", "B"], "cat2": ["X", "Y", "X", "Y"]}
        X = pd.DataFrame(data)
        y = pd.Series([0, 1, 0, 1])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            woe = WeightOfEvidence(None, X, y)

        assert woe.is_fitted_
        assert woe.classifier is not None
        assert woe._is_fastwoe_classifier(woe.classifier)


class TestConfidenceIntervals:
    """Test cases for confidence interval functionality."""

    @pytest.fixture
    def sample_data_ci(self):
        """Create sample data for CI testing."""
        np.random.seed(42)

        # Create categorical data suitable for FastWoe
        n_samples = 200
        data = {
            "cat_feature_1": np.random.choice(["A", "B", "C"], n_samples),
            "cat_feature_2": np.random.choice(["X", "Y"], n_samples),
            "num_feature_1": np.random.normal(0, 1, n_samples),
            "num_feature_2": np.random.normal(0, 1, n_samples),
        }

        X = pd.DataFrame(data)
        # Create target with some correlation to features
        y = ((X["num_feature_1"] > 0) & (X["cat_feature_1"] == "A")).astype(int)

        # Train FastWoe classifier - suppress expected warning about numerical features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            clf = FastWoe()
            clf.fit(X, y)

        return X, y, clf

    def test_explain_ci_basic(self, sample_data_ci):
        """Test basic confidence interval explanation functionality."""
        X, y, clf = sample_data_ci

        woe = WeightOfEvidence(clf, X, y)
        sample = X.iloc[0]

        explanation = woe.explain_ci(sample)

        # Check basic structure
        assert isinstance(explanation, dict)
        assert "confidence_level" in explanation
        assert "ci_conservative" in explanation
        assert "ci_optimistic" in explanation
        assert "uncertainty_range" in explanation

        # Check confidence level
        assert explanation["confidence_level"] == "95%"

        # Check conservative scenario
        ci_cons = explanation["ci_conservative"]
        assert "total_woe" in ci_cons
        assert "predicted_proba_ci" in ci_cons
        assert "interpretation" in ci_cons

        # Check optimistic scenario
        ci_opt = explanation["ci_optimistic"]
        assert "total_woe" in ci_opt
        assert "predicted_proba_ci" in ci_opt
        assert "interpretation" in ci_opt

    def test_explain_ci_with_true_label(self, sample_data_ci):
        """Test CI explanation with true label."""
        X, y, clf = sample_data_ci

        woe = WeightOfEvidence(clf, X, y)
        sample = X.iloc[0]
        true_labels = pd.Series([y[0]])  # Convert to Series

        explanation = woe.explain_ci(sample, true_labels=true_labels)
        assert isinstance(explanation, dict)
        assert "true_label" in explanation
        assert explanation["true_label"] in ["Positive", "Negative"]

    def test_explain_ci_custom_alpha(self, sample_data_ci):
        """Test confidence interval with custom alpha level."""
        X, y, clf = sample_data_ci

        woe = WeightOfEvidence(clf, X, y)
        sample = X.iloc[0]

        # Test 90% CI (alpha=0.1)
        explanation_90 = woe.explain_ci(sample, alpha=0.1)
        assert explanation_90 is not None
        assert explanation_90["confidence_level"] == "90%"

        # Test 99% CI (alpha=0.01)
        explanation_99 = woe.explain_ci(sample, alpha=0.01)
        assert explanation_99 is not None
        assert explanation_99["confidence_level"] == "99%"

        # 99% CI should be wider than or equal to 90% CI (account for numerical precision)
        range_90 = explanation_90["uncertainty_range"]["woe_range"]
        range_99 = explanation_99["uncertainty_range"]["woe_range"]
        assert range_99 >= range_90

    def test_explain_ci_with_dataset_index(self, sample_data_ci):
        """Test CI explanation with dataset and sample index."""
        X, y, clf = sample_data_ci

        woe = WeightOfEvidence(clf, X, y)

        explanation = woe.explain_ci(X, sample_idx=0)

        assert isinstance(explanation, dict)
        assert "ci_conservative" in explanation
        assert "ci_optimistic" in explanation

    def test_explain_ci_print_format(self, sample_data_ci, capsys):
        """Test CI explanation print format."""
        X, y, clf = sample_data_ci

        woe = WeightOfEvidence(clf, X, y)
        sample = X.iloc[0]

        # Test with return_dict=False
        result = woe.explain_ci(sample, return_dict=False)
        assert result is None

        # Check printed output
        captured = capsys.readouterr()
        assert "Point Estimate" in captured.out
        assert "Lower Bound" in captured.out
        assert "Upper Bound" in captured.out
        assert "Uncertainty Summary" in captured.out

    def test_explain_ci_not_fitted(self, sample_data_ci):
        """Test that explain_ci raises error when not fitted."""
        X, y, clf = sample_data_ci

        woe = WeightOfEvidence.__new__(WeightOfEvidence)
        woe.is_fitted_ = False

        with pytest.raises(ValueError, match="must be fitted before explaining"):
            woe.explain_ci(X.iloc[0])

    def test_explain_ci_input_validation(self, sample_data_ci):
        """Test input validation for explain_ci method."""
        X, y, clf = sample_data_ci

        woe = WeightOfEvidence(clf, X, y)

        # Test error when passing full dataset without sample_idx
        with pytest.raises(ValueError, match="no sample_idx"):
            woe.explain_ci(X)

        # Test error when passing array without sample_idx
        X_array = X.values
        with pytest.raises(ValueError, match="no sample_idx"):
            woe.explain_ci(X_array)

    def test_explain_ci_uncertainty_properties(self, sample_data_ci):
        """Test mathematical properties of uncertainty estimates."""
        X, y, clf = sample_data_ci

        woe = WeightOfEvidence(clf, X, y)
        sample = X.iloc[0]

        explanation = woe.explain_ci(sample)

        # Check that conservative and optimistic bounds make sense
        assert explanation is not None
        ci_cons = explanation["ci_conservative"]
        ci_opt = explanation["ci_optimistic"]
        base = explanation

        # Conservative should be more negative (or less positive) for WOE
        assert ci_cons["total_woe"] <= base["total_woe"]
        assert base["total_woe"] <= ci_opt["total_woe"]

        # Probability bounds should be ordered
        assert ci_cons["predicted_proba_ci"] <= ci_opt["predicted_proba_ci"]

        # Uncertainty range should be positive
        unc = explanation["uncertainty_range"]
        assert unc["woe_range"] >= 0
        assert unc["prob_range"] >= 0

    def test_predict_ci_basic(self, sample_data_ci):
        """Test basic predict_ci functionality."""
        X, y, clf = sample_data_ci

        woe = WeightOfEvidence(clf, X, y)

        # Test with a small subset
        X_test = X.iloc[:5]
        results = woe.predict_ci(X_test)

        # Check structure
        assert isinstance(results, dict)
        assert "base_estimate" in results
        assert "lower_bound" in results
        assert "upper_bound" in results
        assert "uncertainty_summary" in results

        # Check base estimate
        base = results["base_estimate"]
        assert "predictions" in base
        assert "woe_scores" in base
        assert "scenario" in base
        assert len(base["predictions"]) == 5

        # Check lower bound (conservative)
        lower = results["lower_bound"]
        assert "predictions" in lower
        assert "woe_scores" in lower
        assert "scenario" in lower
        assert len(lower["predictions"]) == 5

        # Check upper bound (optimistic)
        upper = results["upper_bound"]
        assert "predictions" in upper
        assert "woe_scores" in upper
        assert "scenario" in upper
        assert len(upper["predictions"]) == 5

        # Check uncertainty summary
        unc = results["uncertainty_summary"]
        assert "confidence_level" in unc
        assert "prediction_agreement" in unc
        assert "mean_woe_uncertainty" in unc
        assert "mean_prob_uncertainty" in unc
        assert unc["confidence_level"] == "95%"
        assert 0 <= unc["prediction_agreement"] <= 1

    def test_predict_ci_with_probabilities(self, sample_data_ci):
        """Test predict_ci with probability output."""
        X, y, clf = sample_data_ci

        woe = WeightOfEvidence(clf, X, y)

        X_test = X.iloc[:3]
        results = woe.predict_ci(X_test, return_probabilities=True)

        # Check that probabilities are included
        assert "probabilities" in results["base_estimate"]
        assert "probabilities" in results["lower_bound"]
        assert "probabilities" in results["upper_bound"]

        # Check probability shapes
        base_probs = results["base_estimate"]["probabilities"]
        lower_probs = results["lower_bound"]["probabilities"]
        upper_probs = results["upper_bound"]["probabilities"]

        assert base_probs.shape == (3, 2)
        assert lower_probs.shape == (3, 2)
        assert upper_probs.shape == (3, 2)

        # Check probability bounds ordering
        # sourcery skip: no-loop-in-tests
        for i in range(3):
            # Lower bound probabilities should be <= upper bound probabilities
            assert lower_probs[i, 1] <= upper_probs[i, 1]  # Positive class prob

    def test_predict_ci_prediction_logic(self, sample_data_ci):
        """Test the core prediction logic with CI bounds."""
        X, y, clf = sample_data_ci

        woe = WeightOfEvidence(clf, X, y)

        X_test = X.iloc[:10]
        results = woe.predict_ci(X_test)

        base_preds = results["base_estimate"]["predictions"]
        conservative_preds = results["lower_bound"]["predictions"]
        optimistic_preds = results["upper_bound"]["predictions"]

        # Conservative should be less aggressive (fewer positive predictions)
        assert np.sum(conservative_preds) <= np.sum(base_preds)

        # Optimistic should be more aggressive (more positive predictions)
        assert np.sum(base_preds) <= np.sum(optimistic_preds)

        # Conservative should be less aggressive than optimistic
        assert np.sum(conservative_preds) <= np.sum(optimistic_preds)

    def test_predict_ci_uncertainty_metrics(self, sample_data_ci):
        """Test uncertainty metrics calculation."""
        X, y, clf = sample_data_ci

        woe = WeightOfEvidence(clf, X, y)

        X_test = X.iloc[:5]
        results = woe.predict_ci(X_test)

        unc = results["uncertainty_summary"]

        # Check that uncertainty metrics are non-negative
        assert unc["mean_woe_uncertainty"] >= 0
        assert unc["mean_prob_uncertainty"] >= 0
        assert unc["max_woe_uncertainty"] >= unc["mean_woe_uncertainty"]
        assert unc["max_prob_uncertainty"] >= unc["mean_prob_uncertainty"]

        # Prediction agreement should be between 0 and 1
        assert 0 <= unc["prediction_agreement"] <= 1

    def test_predict_ci_different_alpha(self, sample_data_ci):
        """Test predict_ci with different alpha levels."""
        X, y, clf = sample_data_ci

        woe = WeightOfEvidence(clf, X, y)

        X_test = X.iloc[:3]

        # Test different confidence levels
        results_90 = woe.predict_ci(X_test, alpha=0.1)
        results_99 = woe.predict_ci(X_test, alpha=0.01)

        assert results_90["uncertainty_summary"]["confidence_level"] == "90%"
        assert results_99["uncertainty_summary"]["confidence_level"] == "99%"

    def test_predict_ci_not_fitted(self, sample_data_ci):
        """Test that predict_ci raises error when not fitted."""
        X, y, clf = sample_data_ci

        woe = WeightOfEvidence.__new__(WeightOfEvidence)
        woe.is_fitted_ = False

        with pytest.raises(ValueError, match="must be fitted before predicting"):
            woe.predict_ci(X.iloc[:1])

    def test_predict_ci_numpy_input(self, sample_data_ci):
        """Test predict_ci with numpy array input."""
        X, y, clf = sample_data_ci

        woe = WeightOfEvidence(clf, X, y)

        # Convert to numpy array
        X_array = X.iloc[:3].values
        results = woe.predict_ci(X_array)

        # Should work the same as DataFrame input
        assert isinstance(results, dict)
        assert len(results["base_estimate"]["predictions"]) == 3

    def test_predict_ci_rich_output(self, sample_data_ci):
        """Test that predict_ci includes rich formatted output."""
        X, y, clf = sample_data_ci

        woe = WeightOfEvidence(clf, X, y)

        X_test = X.iloc[:3]
        results = woe.predict_ci(X_test)

        # Check that all scenarios have rich output
        for scenario_key in ["base_estimate", "lower_bound", "upper_bound"]:
            scenario = results[scenario_key]

            # Check basic arrays
            assert "predictions" in scenario
            assert "woe_scores" in scenario
            assert "scenario" in scenario

            # Check new rich output
            assert "predicted_labels" in scenario
            assert "predicted_proba" in scenario
            assert "interpretation" in scenario

            # Check lengths match
            n_samples = len(scenario["predictions"])
            assert len(scenario["predicted_labels"]) == n_samples
            assert len(scenario["predicted_proba"]) == n_samples
            assert len(scenario["interpretation"]) == n_samples

            # Check types
            assert isinstance(scenario["predicted_labels"][0], str)
            assert isinstance(scenario["predicted_proba"][0], dict)
            assert isinstance(scenario["interpretation"][0], str)

            # Check predicted labels are valid class names
            assert woe.class_names is not None
            for label in scenario["predicted_labels"]:
                assert label in woe.class_names
