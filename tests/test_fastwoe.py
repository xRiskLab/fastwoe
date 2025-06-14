"""Tests for FastWoe library."""

import warnings

import numpy as np
import pandas as pd
import pytest

from fastwoe import FastWoe, WoePreprocessor


class TestWoePreprocessor:
    """Test cases for WoePreprocessor class."""

    def test_init(self):
        """Test initialization of WoePreprocessor."""
        # Test default initialization
        preprocessor = WoePreprocessor()
        assert preprocessor.max_categories is None
        assert preprocessor.top_p == 0.95
        assert preprocessor.min_count == 10
        assert preprocessor.other_token == "__other__"

        # Test custom initialization
        preprocessor = WoePreprocessor(
            max_categories=5, top_p=0.8, min_count=5, other_token="OTHER"
        )
        assert preprocessor.max_categories == 5
        assert preprocessor.top_p == 0.8
        assert preprocessor.min_count == 5
        assert preprocessor.other_token == "OTHER"

    def test_init_invalid_params(self):
        """Test initialization with invalid parameters."""
        with pytest.raises(AssertionError):
            WoePreprocessor(max_categories=None, top_p=None)

    def test_fit_basic(self):
        """Test basic fitting functionality."""
        df = pd.DataFrame(
            {
                "cat1": ["A"] * 50 + ["B"] * 30 + ["C"] * 5,
                "cat2": ["X"] * 40 + ["Y"] * 30 + ["Z"] * 15,
                "num": range(85),
            }
        )

        preprocessor = WoePreprocessor(min_count=10)
        preprocessor.fit(df)

        assert "cat1" in preprocessor.category_maps
        assert "cat2" in preprocessor.category_maps
        assert len(preprocessor.cat_features_) == 2

    def test_fit_with_cat_features(self):
        """Test fitting with specified categorical features."""
        df = pd.DataFrame(
            {
                "cat1": ["A"] * 50 + ["B"] * 30,
                "cat2": ["X"] * 40 + ["Y"] * 40,
                "num": range(80),
            }
        )

        preprocessor = WoePreprocessor()
        preprocessor.fit(df, cat_features=["cat1"])

        assert "cat1" in preprocessor.category_maps
        assert "cat2" not in preprocessor.category_maps
        assert len(preprocessor.cat_features_) == 1

    def test_transform(self):
        """Test transformation functionality."""
        df = pd.DataFrame(
            {
                "cat1": ["A"] * 50 + ["B"] * 30 + ["C"] * 5,
            }
        )

        preprocessor = WoePreprocessor(min_count=10)
        preprocessor.fit(df)
        df_transformed = preprocessor.transform(df)

        # Check that rare category 'C' was replaced
        assert "__other__" in df_transformed["cat1"].values
        assert "C" not in df_transformed["cat1"].values

    def test_fit_transform(self):
        """Test fit_transform method."""
        df = pd.DataFrame(
            {
                "cat1": ["A"] * 50 + ["B"] * 30 + ["C"] * 5,
            }
        )

        preprocessor = WoePreprocessor(min_count=10)
        df_transformed = preprocessor.fit_transform(df)

        assert "__other__" in df_transformed["cat1"].values
        assert "C" not in df_transformed["cat1"].values

    def test_binary_categories(self):
        """Test handling of binary categories."""
        df = pd.DataFrame(
            {
                "binary_cat": ["Yes", "No"] * 25,
            }
        )

        preprocessor = WoePreprocessor(min_count=10)
        preprocessor.fit(df)

        # Binary categories should be kept as-is
        assert len(preprocessor.category_maps["binary_cat"]) == 2
        assert "Yes" in preprocessor.category_maps["binary_cat"]
        assert "No" in preprocessor.category_maps["binary_cat"]

    def test_get_reduction_summary(self):
        """Test cardinality reduction summary."""
        df = pd.DataFrame(
            {
                "cat1": ["A"] * 50 + ["B"] * 30 + ["C"] * 5 + ["D"] * 3,
            }
        )

        preprocessor = WoePreprocessor(min_count=10)
        preprocessor.fit(df)
        summary = preprocessor.get_reduction_summary(df)

        assert isinstance(summary, pd.DataFrame)
        assert "feature" in summary.columns
        assert "original_categories" in summary.columns
        assert "kept_categories" in summary.columns
        assert "reduction_pct" in summary.columns


class TestFastWoe:
    """Test cases for FastWoe class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "cat1": ["A"] * 50 + ["B"] * 30 + ["C"] * 20,
                "cat2": ["X"] * 40 + ["Y"] * 35 + ["Z"] * 25,
                "target": np.random.binomial(1, 0.3, 100),
            }
        )
        # Create some correlation with target
        data.loc[data["cat1"] == "A", "target"] = np.random.binomial(1, 0.5, 50)
        data.loc[data["cat1"] == "B", "target"] = np.random.binomial(1, 0.2, 30)
        data.loc[data["cat1"] == "C", "target"] = np.random.binomial(1, 0.1, 20)

        return data

    def test_init(self):
        """Test initialization of FastWoe."""
        woe = FastWoe()
        assert woe.random_state == 42
        assert woe.encoder_kwargs == {"smooth": 1e-5, "target_type": "binary"}
        assert not woe.is_fitted_

        # Test custom initialization
        woe = FastWoe(encoder_kwargs={"smooth": 1e-3}, random_state=123)
        assert woe.random_state == 123
        assert woe.encoder_kwargs == {"smooth": 1e-3, "target_type": "binary"}

    def test_init_multiclass_error(self):
        """Test that multiclass target_type raises NotImplementedError."""
        with pytest.raises(
            NotImplementedError,
            match="FastWoe currently only supports binary classification",
        ):
            FastWoe(encoder_kwargs={"target_type": "multiclass"})

    def test_fit(self, sample_data):
        """Test fitting functionality."""
        X = sample_data[["cat1", "cat2"]]
        y = sample_data["target"]

        woe = FastWoe()
        woe.fit(X, y)

        assert woe.is_fitted_
        assert "cat1" in woe.mappings_
        assert "cat2" in woe.mappings_
        assert woe.y_prior_ == y.mean()

    def test_transform(self, sample_data):
        """Test transformation functionality."""
        X = sample_data[["cat1", "cat2"]]
        y = sample_data["target"]

        woe = FastWoe()
        woe.fit(X, y)
        X_transformed = woe.transform(X)  # pylint: disable=invalid-name

        assert X_transformed.shape == X.shape
        assert isinstance(X_transformed, pd.DataFrame)
        # Check that values are WOE scores (floats)
        assert X_transformed.dtypes["cat1"] == "float64"
        assert X_transformed.dtypes["cat2"] == "float64"

    def test_transform_not_fitted(self, sample_data):
        """Test transform raises error when not fitted."""
        X = sample_data[["cat1", "cat2"]]
        woe = FastWoe()

        with pytest.raises(TypeError):  # Will get TypeError because y_prior_ is None
            woe.transform(X)

    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        X = sample_data[["cat1", "cat2"]]
        y = sample_data["target"]

        woe = FastWoe()
        X_transformed = woe.fit_transform(X, y)  # pylint: disable=invalid-name

        assert X_transformed.shape == X.shape
        assert isinstance(X_transformed, pd.DataFrame)
        assert woe.is_fitted_

    def test_get_mapping(self, sample_data):
        """Test getting mapping for specific column."""
        X = sample_data[["cat1", "cat2"]]
        y = sample_data["target"]

        woe = FastWoe()
        woe.fit(X, y)
        mapping = woe.get_mapping("cat1")

        assert isinstance(mapping, pd.DataFrame)
        assert "category" in mapping.columns
        assert "count" in mapping.columns
        assert "event_rate" in mapping.columns
        assert "woe" in mapping.columns

    def test_get_all_mappings(self, sample_data):
        """Test getting all mappings."""
        X = sample_data[["cat1", "cat2"]]
        y = sample_data["target"]

        woe = FastWoe()
        woe.fit(X, y)
        all_mappings = woe.get_all_mappings()

        assert isinstance(all_mappings, dict)
        assert "cat1" in all_mappings
        assert "cat2" in all_mappings

    def test_get_feature_stats(self, sample_data):
        """Test getting feature statistics."""
        X = sample_data[["cat1", "cat2"]]
        y = sample_data["target"]

        woe = FastWoe()
        woe.fit(X, y)

        # Test single feature stats
        stats_cat1 = woe.get_feature_stats("cat1")
        assert isinstance(stats_cat1, pd.DataFrame)
        assert "feature" in stats_cat1.columns

        # Test all features stats
        all_stats = woe.get_feature_stats()
        assert len(all_stats) == 2  # Two features

    def test_predict_proba(self, sample_data):
        """Test predict_proba method."""
        X = sample_data[["cat1", "cat2"]]
        y = sample_data["target"]

        woe = FastWoe()
        woe.fit(X, y)
        probas = woe.predict_proba(X)

        assert isinstance(probas, np.ndarray)
        assert probas.shape == (len(X), 2)  # Returns [neg_prob, pos_prob]
        assert np.all((probas >= 0) & (probas <= 1))
        assert np.allclose(probas.sum(axis=1), 1.0)  # Probabilities sum to 1

    def test_predict_ci(self, sample_data):
        """Test confidence interval prediction."""
        X = sample_data[["cat1", "cat2"]]
        y = sample_data["target"]

        woe = FastWoe()
        woe.fit(X, y)
        ci_df = woe.predict_ci(X)

        assert isinstance(ci_df, pd.DataFrame)
        assert "ci_lower" in ci_df.columns
        assert "ci_upper" in ci_df.columns
        assert len(ci_df) == len(X)
        assert np.all((ci_df["ci_lower"] >= 0) & (ci_df["ci_lower"] <= 1))
        assert np.all((ci_df["ci_upper"] >= 0) & (ci_df["ci_upper"] <= 1))

    def test_calculate_woe_se(self):
        """Test WOE standard error calculation."""
        woe = FastWoe()

        # Test normal case
        se = woe._calculate_woe_se(good_count=100, bad_count=50)
        expected_se = np.sqrt(1 / 100 + 1 / 50)
        assert np.isclose(se, expected_se)

        # Test symmetry
        se_reverse = woe._calculate_woe_se(good_count=50, bad_count=100)
        assert np.isclose(se, se_reverse)  # SE should be same when swapping counts

        # Test larger counts give smaller SE
        se_large = woe._calculate_woe_se(good_count=1000, bad_count=500)
        assert se_large < se  # SE should decrease with sqrt of sample size

    def test_calculate_woe_ci(self):
        """Test WOE confidence interval calculation."""
        woe = FastWoe()

        woe_value = 0.5
        se_value = 0.1
        lower, upper = woe._calculate_woe_ci(woe_value, se_value, alpha=0.05)

        assert lower < woe_value < upper
        assert isinstance(lower, float)
        assert isinstance(upper, float)

    def test_transform_standardized(self, sample_data):
        """Test standardized transformation."""
        X = sample_data[["cat1", "cat2"]]
        y = sample_data["target"]

        woe = FastWoe()
        woe.fit(X, y)

        # Test WOE output - returns original column names
        woe_result = woe.transform_standardized(X, output="woe")
        assert "cat1" in woe_result.columns or "cat2" in woe_result.columns

        # Test wald output
        wald_result = woe.transform_standardized(X, output="wald")
        assert "cat1" in wald_result.columns or "cat2" in wald_result.columns

    def test_numerical_binning_basic(self):
        """Test basic numerical binning functionality."""
        np.random.seed(42)
        # Create data with high-cardinality numerical feature
        X = pd.DataFrame(
            {
                "age": np.random.randint(18, 80, 200),  # High cardinality numerical
                "score": np.random.randint(300, 850, 200),  # Another numerical
                "category": np.random.choice(
                    ["A", "B", "C"], 200
                ),  # Low cardinality categorical
            }
        )
        y = np.random.binomial(1, 0.3, 200)

        woe = FastWoe(numerical_threshold=10, warn_on_numerical=True)

        # Should trigger numerical detection and binning
        with pytest.warns(UserWarning, match="Detected numerical features"):
            woe.fit(X, y)

        # Check that numerical features were detected
        binning_summary = woe.get_binning_summary()
        assert len(binning_summary) > 0
        assert "age" in binning_summary["feature"].values
        assert "score" in binning_summary["feature"].values
        assert (
            "category" not in binning_summary["feature"].values
        )  # Should not be binned

    def test_numerical_threshold_parameter(self):
        """Test numerical_threshold parameter controls binning."""
        np.random.seed(42)
        # Create feature with exactly 15 unique values
        X = pd.DataFrame(
            {
                "feature": np.random.randint(1, 16, 100)  # 15 unique values
            }
        )
        y = np.random.binomial(1, 0.3, 100)

        # With threshold=10, should be binned (15 >= 10)
        woe_bin = FastWoe(numerical_threshold=10, warn_on_numerical=True)
        with pytest.warns(UserWarning):
            woe_bin.fit(X, y)
        binning_summary = woe_bin.get_binning_summary()
        assert len(binning_summary) == 1
        assert binning_summary.iloc[0]["feature"] == "feature"

        # With threshold=20, should not be binned (15 < 20)
        woe_no_bin = FastWoe(numerical_threshold=20, warn_on_numerical=True)
        woe_no_bin.fit(X, y)  # No warning expected
        binning_summary = woe_no_bin.get_binning_summary()
        assert len(binning_summary) == 0

    def test_numerical_binning_transform_consistency(self):
        """Test that binned numerical features transform consistently."""
        np.random.seed(42)
        X = pd.DataFrame({"numerical_score": np.random.randint(100, 1000, 150)})
        y = np.random.binomial(1, 0.4, 150)

        woe = FastWoe(numerical_threshold=10, warn_on_numerical=True)
        with pytest.warns(UserWarning):
            woe.fit(X, y)

        # Transform should work without errors
        X_transformed = woe.transform(X)  # pylint: disable=invalid-name
        assert X_transformed.shape == X.shape
        assert not X_transformed["numerical_score"].isna().any()
        assert np.isfinite(X_transformed["numerical_score"]).all()

        # All values should be valid WOE scores
        assert X_transformed["numerical_score"].dtype == "float64"

    # pylint: disable=invalid-name
    def test_mixed_numerical_categorical(self):
        """Test handling of mixed numerical and categorical features."""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "high_card_num": np.random.randint(1, 100, 200),  # Should be binned
                "low_card_num": np.random.choice(
                    [1, 2, 3], 200
                ),  # Should not be binned
                "high_card_cat": np.random.choice(
                    [f"cat_{i}" for i in range(50)], 200
                ),  # Categorical, high card
                "low_card_cat": np.random.choice(
                    ["A", "B", "C"], 200
                ),  # Categorical, low card
            }
        )
        y = np.random.binomial(1, 0.35, 200)

        woe = FastWoe(numerical_threshold=10, warn_on_numerical=True)
        with pytest.warns(UserWarning):
            woe.fit(X, y)

        # Check binning summary
        binning_summary = woe.get_binning_summary()
        binned_features = binning_summary["feature"].tolist()

        # Only high_card_num should be binned (high cardinality numerical)
        assert "high_card_num" in binned_features
        assert "low_card_num" not in binned_features  # Low cardinality
        assert (
            "high_card_cat" not in binned_features
        )  # Categorical (not numerical dtype)
        assert "low_card_cat" not in binned_features  # Low cardinality categorical

        # Transform should work for all features
        X_transformed = woe.transform(X)
        assert X_transformed.shape == X.shape
        assert not X_transformed.isna().any().any()

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with single category
        df_single = pd.DataFrame(
            {"cat": ["A"] * 100, "target": np.random.binomial(1, 0.3, 100)}
        )

        woe = FastWoe()
        woe.fit(df_single[["cat"]], df_single["target"])
        transformed = woe.transform(df_single[["cat"]])

        # Should handle single category gracefully
        assert not np.isnan(transformed["cat"]).any()

    def test_missing_values(self):
        """Test handling of missing values."""
        df_missing = pd.DataFrame(
            {
                "cat": ["A"] * 40 + ["B"] * 40 + [None] * 20,
                "target": np.random.binomial(1, 0.3, 100),
            }
        )

        woe = FastWoe()
        woe.fit(df_missing[["cat"]], df_missing["target"])
        transformed = woe.transform(df_missing[["cat"]])

        # Should handle missing values
        assert transformed.shape[0] == 100

    def test_target_validation_multiclass(self):
        """Test target validation for multiclass targets."""
        X = pd.DataFrame({"cat": ["A", "B", "C"] * 10})
        y_multiclass = pd.Series([0, 1, 2] * 10)  # 3 classes

        woe = FastWoe()
        with pytest.raises(ValueError, match="FastWoe only supports binary targets"):
            woe.fit(X, y_multiclass)

    def test_target_validation_single_class(self):
        """Test target validation for single class targets."""
        X = pd.DataFrame({"cat": ["A", "B", "C"] * 10})
        y_single = pd.Series([1] * 30)  # Only one class

        woe = FastWoe()
        with pytest.raises(
            ValueError, match="Target variable must have exactly 2 classes"
        ):
            woe.fit(X, y_single)

    def test_target_validation_invalid_values(self):
        """Test target validation for non-binary values."""
        X = pd.DataFrame({"cat": ["A", "B", "C"] * 10})
        y_invalid = pd.Series([1, 2] * 15)  # Values not 0 and 1

        woe = FastWoe()
        with pytest.raises(
            ValueError, match="Target variable must contain only 0 and 1"
        ):
            woe.fit(X, y_invalid)

    def test_predict_method(self, sample_data):
        """Test predict method functionality."""
        X = sample_data[["cat1", "cat2"]]
        y = sample_data["target"]

        woe = FastWoe()
        woe.fit(X, y)
        predictions = woe.predict(X)

        # Basic checks
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (len(X),)
        assert set(predictions).issubset({0, 1})  # Only 0 and 1 predictions

        # Check that predictions are based on WOE scores
        woe_scores = woe.transform(X).sum(axis=1)
        expected_predictions = (woe_scores > 0).astype(int)
        np.testing.assert_array_equal(predictions, expected_predictions)

    def test_predict_not_fitted(self, sample_data):
        """Test predict raises error when not fitted."""
        X = sample_data[["cat1", "cat2"]]
        woe = FastWoe()

        with pytest.raises(TypeError):  # Will get TypeError because y_prior_ is None
            woe.predict(X)

    def test_predict_vs_predict_proba_difference(self, sample_data):
        """Test that predict (WOE-based) differs from predict_proba > 0.5 threshold."""
        X = sample_data[["cat1", "cat2"]]
        y = sample_data["target"]

        woe = FastWoe()
        woe.fit(X, y)

        # WOE-based predictions (predict method)
        woe_predictions = woe.predict(X)

        # Probability-based predictions (predict_proba > 0.5)
        probabilities = woe.predict_proba(X)
        prob_predictions = (probabilities[:, 1] > 0.5).astype(int)

        # Calculate average prediction rates
        woe_avg_rate = woe_predictions.mean()
        prob_avg_rate = prob_predictions.mean()

        print(f"WOE-based average prediction rate: {woe_avg_rate:.4f}")
        print(f"Probability-based (>0.5) average prediction rate: {prob_avg_rate:.4f}")
        print(f"Difference: {abs(woe_avg_rate - prob_avg_rate):.4f}")

        # Assert that the methods give different results
        # (they should differ because WOE=0 is the center, not 50% probability)
        assert not np.array_equal(woe_predictions, prob_predictions), (
            "WOE-based and probability-based predictions should differ"
        )

        # Assert that average rates are different
        assert abs(woe_avg_rate - prob_avg_rate) > 0.01, (
            "Average prediction rates should differ by more than 1%"
        )

    def test_predict_edge_cases(self):
        """Test predict method with edge cases."""
        # Create simple data with clear separation
        X = pd.DataFrame({"cat": ["low_risk"] * 50 + ["high_risk"] * 50})
        y = pd.Series([0] * 45 + [1] * 5 + [1] * 45 + [0] * 5)  # Clear separation

        predictions = self._predict_edge_cases(X, y)
        # Should predict correctly for clear separation
        assert len(set(predictions)) <= 2  # At most 2 unique predictions

        # Test with single category
        X_single = pd.DataFrame({"cat": ["A"] * 20})
        y_single = pd.Series([0] * 10 + [1] * 10)

        predictions_single = self._predict_edge_cases(X_single, y_single)
        # Should handle single category without errors
        assert len(predictions_single) == 20
        assert set(predictions_single).issubset({0, 1})

    def _predict_edge_cases(self, arg0, arg1):
        """Predict method for edge cases."""
        woe = FastWoe()
        woe.fit(arg0, arg1)
        return woe.predict(arg0)

    def test_get_split_value_histogram(self):
        """Test get_split_value_histogram for a binned numerical feature."""
        np.random.seed(42)
        X = pd.DataFrame({"score": np.random.randint(300, 850, 200)})
        y = np.random.binomial(1, 0.3, 200)
        woe = FastWoe(numerical_threshold=10, warn_on_numerical=True)
        woe.fit(X, y)

        # Test array output
        edges_array = woe.get_split_value_histogram("score", as_array=True)
        assert isinstance(edges_array, np.ndarray)
        assert edges_array.shape[0] > 2  # At least 2 edges
        assert np.isneginf(edges_array[0])  # First edge should be -inf
        assert np.isinf(edges_array[-1])  # Last edge should be inf
        assert np.all(
            np.diff(edges_array[1:-1]) > 0
        )  # Edges should be strictly increasing

        # Test list output
        edges_list = woe.get_split_value_histogram("score", as_array=False)
        assert isinstance(edges_list, list)
        assert len(edges_list) == edges_array.shape[0]
        assert edges_list[0] == float("-inf")
        assert edges_list[-1] == float("inf")

        # Test error cases
        with pytest.raises(ValueError, match="FastWoe must be fitted"):
            FastWoe().get_split_value_histogram("score")

        with pytest.raises(ValueError, match="Feature 'nonexistent' not found"):
            woe.get_split_value_histogram("nonexistent")

        # Test with non-binned feature
        X_cat = pd.DataFrame({"category": ["A", "B", "C"] * 67})  # 201 rows
        y_cat = np.random.binomial(1, 0.3, 201)  # Match X_cat size
        woe_cat = FastWoe()
        woe_cat.fit(X_cat, y_cat)


# pylint: disable=invalid-name
class TestIntegration:
    """Integration tests combining WoePreprocessor and FastWoe."""

    def test_preprocessing_and_woe_pipeline(self):
        """Test full pipeline with preprocessing and WOE encoding."""
        # Create data with high cardinality
        np.random.seed(42)
        categories = [f"cat_{i}" for i in range(50)]  # 50 categories
        counts = np.random.poisson(lam=20, size=50) + 1  # Random counts

        data_rows = []
        # sourcery skip: no-loop-in-tests
        for cat, count in zip(categories, counts):
            data_rows.extend(
                {"high_card_feature": cat, "target": np.random.binomial(1, 0.3)}
                for _ in range(count)
            )
        df = pd.DataFrame(data_rows)
        X = df[["high_card_feature"]]
        y = df["target"]

        # Step 1: Preprocess to reduce cardinality
        preprocessor = WoePreprocessor(
            max_categories=10, top_p=None
        )  # Use max_categories only
        X_preprocessed = preprocessor.fit_transform(X)

        # Step 2: Apply WOE encoding
        woe_encoder = FastWoe()
        X_woe = woe_encoder.fit_transform(X_preprocessed, y)

        # Verify pipeline worked
        assert X_woe.shape[0] == len(df)
        assert X_woe.shape[1] == 1
        assert X_preprocessed["high_card_feature"].nunique() <= 11  # 10 + __other__

        # Check that WOE values are reasonable
        assert not np.isnan(X_woe["high_card_feature"]).any()
        assert np.isfinite(X_woe["high_card_feature"]).all()

    def test_multiclass_target(self):
        """Test multiclass target."""
        X = pd.DataFrame({"cat": ["A", "B", "C"] * 10})
        y_multiclass = [0, 1, 2] * 10
        y_multiclass = [chr(65 + i) for i in y_multiclass]

        woe = FastWoe()
        with pytest.raises(ValueError, match="FastWoe only supports binary targets"):
            woe.fit(X, y_multiclass)

    def test_fastwoe_input_types(self):
        """Test FastWoe behavior with different input types."""

        # Create sample data
        np.random.seed(42)
        X_np = np.random.randn(100, 3)  # 100 samples, 3 features
        y_np = np.random.binomial(1, 0.5, 100)  # Binary target

        # Convert to pandas
        X_pd = pd.DataFrame(X_np, columns=["feature_1", "feature_2", "feature_3"])
        y_pd = pd.Series(y_np)

        # Test with numpy arrays - should work with warning
        fastwoe_np = FastWoe()
        with pytest.warns(UserWarning, match="Input X is a numpy array"):
            fastwoe_np.fit(X_np, y_np)

        # Verify numpy fit worked
        assert len(fastwoe_np.mappings_) == 3  # One mapping per feature
        assert all(f"feature_{i}" in fastwoe_np.mappings_ for i in range(3))

        # Test with pandas - should work without input type warnings
        fastwoe_pd = FastWoe()
        with warnings.catch_warnings(record=True) as w:
            fastwoe_pd.fit(X_pd, y_pd)
            # Filter out the numerical binning warning
            input_type_warnings = [msg for msg in w if "Input" in str(msg.message)]
            assert not input_type_warnings

        # Verify pandas fit worked
        assert len(fastwoe_pd.mappings_) == 3  # One mapping per feature
        assert all(col in fastwoe_pd.mappings_ for col in X_pd.columns)

        # Verify both fits produce similar results
        np_woe_values = [df["woe"].values for df in fastwoe_np.mappings_.values()]
        pd_woe_values = [df["woe"].values for df in fastwoe_pd.mappings_.values()]

        # Compare WOE values (they should be similar but not exactly equal due to binning)
        # sourcery skip: no-loop-in-tests
        for np_woe, pd_woe in zip(np_woe_values, pd_woe_values):
            np.testing.assert_allclose(np_woe, pd_woe, rtol=1e-2)
