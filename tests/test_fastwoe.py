"""Tests for FastWoe library."""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

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
        assert preprocessor.cat_features_ is not None
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
        assert preprocessor.cat_features_ is not None
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
        assert woe.encoder_kwargs == {"smooth": 1e-5}  # target_type set during fit()
        assert not woe.is_fitted_

        # Test custom initialization
        woe = FastWoe(encoder_kwargs={"smooth": 1e-3}, random_state=123)
        assert woe.random_state == 123
        assert woe.encoder_kwargs == {"smooth": 1e-3}  # target_type set during fit()

    def test_init_multiclass_support(self):
        """Test that multiclass is now supported."""
        # Multiclass is now supported, so this should not raise an error
        woe = FastWoe()
        # target_type is set during fit, not during init
        assert "target_type" not in woe.encoder_kwargs  # Not set until fit

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

        with pytest.raises(ValueError, match="Model must be fitted before transforming data"):
            woe.transform(X)

    def test_transform_numpy_array(self, sample_data):
        # sourcery skip: extract-duplicate-method
        """Test transform method with numpy array input."""
        X = sample_data[["cat1", "cat2"]]
        y = sample_data["target"]

        # Test case 1: Fit with numpy array, transform with numpy array (should work perfectly)
        X_numpy = X.values
        woe1 = FastWoe()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            woe1.fit(X_numpy, y.values)
            X_transformed = woe1.transform(X_numpy)  # pylint: disable=invalid-name

        assert X_transformed.shape == X.shape
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.dtypes["feature_0"] == "float64"
        assert X_transformed.dtypes["feature_1"] == "float64"

        # Test case 2: Single feature numpy array
        X_single = X[["cat1"]].values
        woe2 = FastWoe()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            woe2.fit(X_single, y.values)
            X_single_transformed = woe2.transform(X_single)  # pylint: disable=invalid-name
        assert X_single_transformed.shape == (X.shape[0], 1)
        assert isinstance(X_single_transformed, pd.DataFrame)

        # Test case 3: Test that transform no longer crashes with numpy array input
        # (even if column names don't match, it should not raise AttributeError)
        woe3 = FastWoe()
        woe3.fit(X, y)  # Fit with DataFrame

        # This should not crash with AttributeError anymore
        X_test_numpy = np.random.randn(10, 2)  # Different data, different shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # This will fail due to column name mismatch, but should not crash with AttributeError
            try:
                woe3.transform(X_test_numpy)
            except KeyError:
                # Expected due to column name mismatch, but no longer AttributeError
                pass
            except AttributeError as e:
                # This should not happen anymore
                pytest.fail(f"transform() should not raise AttributeError with numpy input: {e}")

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

    def test_iv_standard_errors(self, sample_data):
        """Test IV standard error calculations."""
        X = sample_data[["cat1", "cat2"]]
        y = sample_data["target"]

        woe = FastWoe()
        woe.fit(X, y)

        # Test that feature stats include IV standard errors
        stats = woe.get_feature_stats()
        required_iv_cols = ["iv", "iv_se", "iv_ci_lower", "iv_ci_upper"]
        for col in required_iv_cols:
            assert col in stats.columns, f"Missing column: {col}"

        # Test IV standard errors are non-negative
        assert (stats["iv_se"] >= 0).all(), "IV standard errors should be non-negative"

        # Test confidence intervals are properly ordered
        assert (stats["iv_ci_lower"] <= stats["iv"]).all(), "Lower CI should be <= IV"
        assert (stats["iv"] <= stats["iv_ci_upper"]).all(), "IV should be <= Upper CI"
        assert (stats["iv_ci_lower"] >= 0).all(), "Lower CI should be >= 0"

    def test_get_iv_analysis(self, sample_data):
        """Test get_iv_analysis method."""
        X = sample_data[["cat1", "cat2"]]
        y = sample_data["target"]

        woe = FastWoe()
        woe.fit(X, y)

        # Test IV analysis for all features
        iv_analysis = woe.get_iv_analysis()
        assert isinstance(iv_analysis, pd.DataFrame)
        assert len(iv_analysis) == 2  # Two features

        expected_cols = [
            "feature",
            "iv",
            "iv_se",
            "iv_ci_lower",
            "iv_ci_upper",
            "iv_significance",
        ]
        for col in expected_cols:
            assert col in iv_analysis.columns, f"Missing column: {col}"

        # Test significance classification
        significance_values = iv_analysis["iv_significance"].unique()
        valid_significance = {"Significant", "Not Significant"}
        assert set(significance_values).issubset(valid_significance)

        # Test single feature IV analysis
        single_iv = woe.get_iv_analysis("cat1")
        assert isinstance(single_iv, pd.DataFrame)
        assert len(single_iv) == 1
        assert single_iv["feature"].iloc[0] == "cat1"

        # Test error for non-existent feature
        with pytest.raises(ValueError, match="Feature 'nonexistent' not found"):
            woe.get_iv_analysis("nonexistent")

        # Test error when not fitted
        unfitted_woe = FastWoe()
        with pytest.raises(ValueError, match="FastWoe must be fitted"):
            unfitted_woe.get_iv_analysis()

    def test_iv_confidence_intervals(self):
        """Test IV confidence interval calculations with known data."""
        # Create data with strong predictive feature
        np.random.seed(42)
        n_samples = 1000

        # Strong predictor with clear separation
        X = pd.DataFrame(
            {"strong_feature": np.random.choice(["Low", "High"], n_samples, p=[0.6, 0.4])}
        )

        # Create target with strong correlation
        y = np.zeros(n_samples)
        high_mask = X["strong_feature"] == "High"
        y[high_mask] = np.random.choice([0, 1], high_mask.sum(), p=[0.2, 0.8])  # 80% positive
        y[~high_mask] = np.random.choice([0, 1], (~high_mask).sum(), p=[0.8, 0.2])  # 20% positive
        y = pd.Series(y, dtype=int)

        woe = FastWoe()
        woe.fit(X, y)

        # Get IV analysis
        iv_analysis = woe.get_iv_analysis("strong_feature")

        # With strong correlation, IV should be significant
        assert iv_analysis["iv_significance"].iloc[0] == "Significant"
        assert iv_analysis["iv"].iloc[0] > 0.1  # Should have substantial IV
        assert iv_analysis["iv_se"].iloc[0] > 0  # Should have positive standard error
        assert iv_analysis["iv_ci_lower"].iloc[0] > 0  # Lower bound should be positive

    def test_iv_mathematical_properties(self):
        """Test mathematical properties of IV standard errors."""
        # Create simple test case
        np.random.seed(123)
        X = pd.DataFrame({"feature": ["A", "B"] * 100})
        y = pd.Series([0, 1] * 100)  # Perfect separation

        woe = FastWoe()
        woe.fit(X, y)

        # Get mapping to check WOE standard errors
        mapping = woe.get_mapping("feature")

        # Check that WOE standard errors exist and are reasonable
        assert "woe_se" in mapping.columns
        assert (mapping["woe_se"] > 0).all()  # Should be positive

        # Get IV analysis
        iv_stats = woe.get_iv_analysis("feature")
        iv_se = iv_stats["iv_se"].iloc[0]

        # IV standard error should be positive for non-trivial cases
        assert iv_se >= 0

        # Test confidence interval width is reasonable (2 * 1.96 * SE for 95% CI)
        ci_width = iv_stats["iv_ci_upper"].iloc[0] - iv_stats["iv_ci_lower"].iloc[0]
        expected_width = 2 * 1.96 * iv_se
        # Allow for small numerical differences and lower bound truncation at 0
        # The difference can be larger due to lower bound truncation at 0
        assert (
            abs(ci_width - expected_width) < 2.0
        )  # Should be reasonably close to theoretical width

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
        ci_array = woe.predict_ci(X)

        # Should return numpy array with shape (n_samples, 2)
        assert isinstance(ci_array, np.ndarray)
        assert ci_array.shape == (len(X), 2)

        # Extract columns: [ci_lower, ci_upper]
        ci_lower = ci_array[:, 0]
        ci_upper = ci_array[:, 1]

        # All values should be valid probabilities (0-1)
        assert np.all((ci_lower >= 0) & (ci_lower <= 1))
        assert np.all((ci_upper >= 0) & (ci_upper <= 1))

        # Lower bound should be <= upper bound
        assert np.all(ci_lower <= ci_upper)

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

    def test_calculate_gini_with_proportions(self):
        """Test Gini calculation with proportion targets using somersd_yx."""
        woe = FastWoe()

        # Create test data with proportions as target (0.0 to 1.0)
        # This should trigger the ValueError in roc_auc_score and fall back to somersd_yx
        y_true = np.array([0.2, 0.3, 0.7, 0.8, 0.1, 0.9, 0.4, 0.6])
        y_pred = np.array([0.1, 0.4, 0.6, 0.9, 0.2, 0.8, 0.3, 0.7])

        # Calculate Gini using the method
        gini = woe._calculate_gini(y_true, y_pred)

        # Should be a valid float value
        assert isinstance(gini, float)
        assert not np.isnan(gini)
        assert not np.isinf(gini)

        # Gini should be between -1 and 1
        assert -1.0 <= gini <= 1.0

        # Test with perfect correlation (should give high positive Gini)
        y_perfect = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        y_pred_perfect = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        gini_perfect = woe._calculate_gini(y_perfect, y_pred_perfect)
        assert gini_perfect > 0.8  # Should be high for perfect correlation

        # Test with negative correlation (should give negative Gini)
        y_neg = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        y_pred_neg = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        gini_neg = woe._calculate_gini(y_neg, y_pred_neg)
        assert gini_neg < 0  # Should be negative for negative correlation

        # Test with constant values (should return NaN when no variation)
        y_constant = np.array([0.5, 0.5, 0.5, 0.5])
        y_pred_constant = np.array([0.3, 0.3, 0.3, 0.3])
        gini_constant = woe._calculate_gini(y_constant, y_pred_constant)
        assert isinstance(gini_constant, float)
        # With constant values, somersd_yx returns NaN, which is expected
        assert np.isnan(gini_constant)

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
                "category": np.random.choice(["A", "B", "C"], 200),  # Low cardinality categorical
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
        assert "category" not in binning_summary["feature"].values  # Should not be binned

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
                "low_card_num": np.random.choice([1, 2, 3], 200),  # Should not be binned
                "high_card_cat": np.random.choice(
                    [f"cat_{i}" for i in range(50)], 200
                ),  # Categorical, high card
                "low_card_cat": np.random.choice(["A", "B", "C"], 200),  # Categorical, low card
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
        assert "high_card_cat" not in binned_features  # Categorical (not numerical dtype)
        assert "low_card_cat" not in binned_features  # Low cardinality categorical

        # Transform should work for all features
        X_transformed = woe.transform(X)
        assert X_transformed.shape == X.shape
        assert not X_transformed.isna().any().any()

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with single category
        df_single = pd.DataFrame({"cat": ["A"] * 100, "target": np.random.binomial(1, 0.3, 100)})

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
        # Multiclass is now supported
        woe.fit(X, y_multiclass)
        assert woe.is_multiclass_target

    def test_target_validation_single_class(self):
        """Test target validation for single class targets."""
        X = pd.DataFrame({"cat": ["A", "B", "C"] * 10})
        y_single = pd.Series([1] * 30)  # Only one class

        woe = FastWoe()
        with pytest.raises(ValueError, match="Target variable must have at least 2 unique values"):
            woe.fit(X, y_single)

    def test_target_validation_invalid_values(self):
        """Test target validation for non-binary values."""
        X = pd.DataFrame({"cat": ["A", "B", "C"] * 10})
        y_invalid = pd.Series([1, 2] * 15)  # Values not 0 and 1

        woe = FastWoe()
        with pytest.raises(
            ValueError,
            match="Target variable must be binary \\(0/1\\), multiclass, or continuous proportions \\(0-1\\)",
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

        with pytest.raises(ValueError, match="Model must be fitted before transforming data"):
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
        assert edges_array.shape[0] > 2  # At least 2 edges  # type: ignore
        assert np.isneginf(edges_array[0])  # First edge should be -inf
        assert np.isinf(edges_array[-1])  # Last edge should be inf
        assert np.all(np.diff(edges_array[1:-1]) > 0)  # Edges should be strictly increasing

        # Test list output
        edges_list = woe.get_split_value_histogram("score", as_array=False)
        assert isinstance(edges_list, list)
        assert len(edges_list) == edges_array.shape[0]  # type: ignore
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
        preprocessor = WoePreprocessor(max_categories=10, top_p=None)  # Use max_categories only
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
        y_multiclass = np.array(y_multiclass)

        woe = FastWoe()
        # Multiclass is now supported
        woe.fit(X, y_multiclass)
        assert woe.is_multiclass_target

    def test_fastwoe_input_types(self):
        """Test FastWoe behavior with different input types."""

        # Create sample data
        np.random.seed(42)
        X_np = np.random.randn(100, 3)  # 100 samples, 3 features
        y_np = np.random.binomial(1, 0.5, 100)  # Binary target

        # Convert to pandas
        X_pd = pd.DataFrame(X_np, columns=["feature_1", "feature_2", "feature_3"])
        y_pd = pd.Series(y_np)

        # Test with numpy arrays - should work cleanly without warnings (improved UX)
        fastwoe_np = FastWoe()
        fastwoe_np.fit(X_np, y_np)  # No warnings expected with clean implementation

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

    def test_sklearn_version_compatibility(self):
        """Test that FastWoe works with different sklearn versions."""

        # Generate synthetic data with numerical features
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=5,
            n_redundant=5,
            random_state=42,
        )

        # Convert to DataFrame
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        # Test that FastWoe works without quantile_method errors
        fastwoe = FastWoe(numerical_threshold=10)

        # This should work regardless of sklearn version
        fastwoe.fit(X_df, y)

        # Verify the fit worked
        assert fastwoe.is_fitted_
        assert len(fastwoe.mappings_) > 0

        # Test transform
        X_transformed = fastwoe.transform(X_df)
        assert X_transformed.shape == X_df.shape
        assert not X_transformed.isna().any().any()

    def test_continuous_target_tree_binning(self):
        """Test tree binning with continuous target values (proportions)."""
        from scipy.special import expit as sigmoid  # pylint: disable=no-name-in-module

        # Create synthetic aggregated-binomial data
        np.random.seed(42)
        n_samples = 1000
        n_features = 5

        # Generate features
        X = np.random.normal(size=(n_samples, n_features)).astype(np.float32)

        # Generate continuous target (proportions between 0 and 1)
        beta = np.random.normal(size=(n_features,)).astype(np.float32)
        logit = X @ beta + np.random.normal(scale=0.5, size=n_samples).astype(np.float32)
        p_true = sigmoid(logit)

        # Create DataFrame
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(n_features)])
        df["p_true"] = p_true

        # Test with tree binning
        # sourcery skip: extract-duplicate-method
        fw = FastWoe(binning_method="tree", numerical_threshold=10, warn_on_numerical=False)
        fw.fit(df[["x0"]], df["p_true"])

        # Verify the fit worked
        assert fw.is_fitted_
        assert "x0" in fw.mappings_

        # Test transform
        X_transformed = fw.transform(df[["x0"]])
        assert X_transformed.shape == (n_samples, 1)
        assert not X_transformed.isna().any().any()

        # Test binning summary
        summary = fw.get_binning_summary()
        assert len(summary) == 1
        assert summary.iloc[0]["feature"] == "x0"
        assert summary.iloc[0]["method"] == "tree"
        assert summary.iloc[0]["n_bins"] > 1

        # Test split values
        splits = fw.get_split_value_histogram("x0")
        assert len(splits) > 2  # Should have at least 3 values (start, splits, end)
        assert splits[0] == -np.inf
        assert splits[-1] == np.inf

        # Test that it works with binary targets too
        df_binary = df.copy()
        df_binary["p_true"] = (df["p_true"] > 0.5).astype(int)

        fw_binary = FastWoe(binning_method="tree", numerical_threshold=10, warn_on_numerical=False)
        fw_binary.fit(df_binary[["x0"]], df_binary["p_true"])

        assert fw_binary.is_fitted_
        assert "x0" in fw_binary.mappings_

    def test_faiss_kmeans_binning_basic(self):
        """Test basic FAISS KMeans binning functionality."""
        import importlib.util

        # sourcery skip: no-conditionals-in-tests
        if importlib.util.find_spec("faiss") is None:
            pytest.skip("FAISS not available, skipping FAISS KMeans tests")

        np.random.seed(42)
        # Create data with high-cardinality numerical feature
        X = pd.DataFrame(
            {
                "score": np.random.randint(300, 850, 200),  # High cardinality numerical
                "age": np.random.randint(18, 80, 200),  # Another numerical
                "category": np.random.choice(["A", "B", "C"], 200),  # Low cardinality categorical
            }
        )
        y = np.random.binomial(1, 0.3, 200)

        woe = FastWoe(
            binning_method="faiss_kmeans",
            faiss_kwargs={"k": 5, "niter": 10, "verbose": False, "gpu": False},
            numerical_threshold=10,
            warn_on_numerical=True,
        )

        # Should trigger numerical detection and binning
        with pytest.warns(UserWarning, match="Detected numerical features"):
            woe.fit(X, y)

        # Check that numerical features were detected
        binning_summary = woe.get_binning_summary()
        assert len(binning_summary) > 0
        assert "score" in binning_summary["feature"].values
        assert "age" in binning_summary["feature"].values
        assert "category" not in binning_summary["feature"].values  # Should not be binned

        # Check that FAISS KMeans method was used
        faiss_features = binning_summary[binning_summary["method"] == "faiss_kmeans"]
        assert len(faiss_features) == 2  # score and age

    def test_faiss_kmeans_binning_transform_consistency(self):
        """Test that FAISS KMeans binned features transform consistently."""
        import importlib.util

        if importlib.util.find_spec("faiss") is None:
            pytest.skip("FAISS not available, skipping FAISS KMeans tests")

        np.random.seed(42)
        X = pd.DataFrame({"numerical_score": np.random.randint(100, 1000, 150)})
        y = np.random.binomial(1, 0.4, 150)

        woe = FastWoe(
            binning_method="faiss_kmeans",
            faiss_kwargs={"k": 4, "niter": 10, "verbose": False, "gpu": False},
            numerical_threshold=10,
            warn_on_numerical=True,
        )
        with pytest.warns(UserWarning):
            woe.fit(X, y)

        # Transform should work without errors
        X_transformed = woe.transform(X)  # pylint: disable=invalid-name
        assert X_transformed.shape == X.shape
        assert not X_transformed["numerical_score"].isna().any()
        assert np.isfinite(X_transformed["numerical_score"]).all()

        # All values should be valid WOE scores
        assert X_transformed["numerical_score"].dtype == "float64"

    def test_faiss_kmeans_binning_parameters(self):
        """Test FAISS KMeans binning with different parameters."""
        import importlib.util

        if importlib.util.find_spec("faiss") is None:
            pytest.skip("FAISS not available, skipping FAISS KMeans tests")

        np.random.seed(42)
        X = pd.DataFrame({"feature": np.random.randint(1, 100, 100)})
        y = np.random.binomial(1, 0.3, 100)

        # Test with different k values
        # sourcery skip: no-loop-in-tests
        for k in [3, 5, 7]:
            woe = FastWoe(
                binning_method="faiss_kmeans",
                faiss_kwargs={"k": k, "niter": 5, "verbose": False, "gpu": False},
                numerical_threshold=10,
                warn_on_numerical=False,
            )
            woe.fit(X, y)

            # Check that correct number of bins was created
            binning_summary = woe.get_binning_summary()
            assert binning_summary.iloc[0]["n_bins"] == k

    def test_faiss_kmeans_binning_missing_values(self):
        """Test FAISS KMeans binning with missing values."""
        import importlib.util

        # sourcery skip: no-conditionals-in-tests
        if importlib.util.find_spec("faiss") is None:
            pytest.skip("FAISS not available, skipping FAISS KMeans tests")

        np.random.seed(42)
        X = pd.DataFrame(
            {"feature": [1, 2, 3, 4, 5, None, 6, 7, 8, 9, 10] * 10}  # Include missing values
        )
        y = np.random.binomial(1, 0.3, 110)

        woe = FastWoe(
            binning_method="faiss_kmeans",
            faiss_kwargs={"k": 3, "niter": 5, "verbose": False, "gpu": False},
            numerical_threshold=5,
            warn_on_numerical=False,
        )
        woe.fit(X, y)

        # Transform should handle missing values
        X_transformed = woe.transform(X)
        assert X_transformed.shape == X.shape
        assert not X_transformed["feature"].isna().any()

    def test_faiss_kmeans_binning_get_mapping(self):
        """Test getting mapping for FAISS KMeans binned feature."""
        import importlib.util

        # sourcery skip: no-conditionals-in-tests
        if importlib.util.find_spec("faiss") is None:
            pytest.skip("FAISS not available, skipping FAISS KMeans tests")

        np.random.seed(42)
        X = pd.DataFrame({"score": np.random.randint(300, 850, 200)})
        y = np.random.binomial(1, 0.3, 200)

        woe = FastWoe(
            binning_method="faiss_kmeans",
            faiss_kwargs={"k": 4, "niter": 10, "verbose": False, "gpu": False},
            numerical_threshold=10,
            warn_on_numerical=False,
        )
        woe.fit(X, y)

        # Test get_mapping
        mapping = woe.get_mapping("score")
        assert isinstance(mapping, pd.DataFrame)
        assert "category" in mapping.columns
        assert "count" in mapping.columns
        assert "event_rate" in mapping.columns
        assert "woe" in mapping.columns
        assert len(mapping) == 4  # Should have 4 bins

    def test_faiss_kmeans_binning_get_split_value_histogram(self):
        """Test get_split_value_histogram for FAISS KMeans binned feature."""
        import importlib.util

        if importlib.util.find_spec("faiss") is None:
            pytest.skip("FAISS not available, skipping FAISS KMeans tests")

        np.random.seed(42)
        X = pd.DataFrame({"score": np.random.randint(300, 850, 200)})
        y = np.random.binomial(1, 0.3, 200)

        woe = FastWoe(
            binning_method="faiss_kmeans",
            faiss_kwargs={"k": 5, "niter": 10, "verbose": False, "gpu": False},
            numerical_threshold=10,
            warn_on_numerical=False,
        )
        woe.fit(X, y)

        # Test array output
        edges_array = woe.get_split_value_histogram("score", as_array=True)
        assert isinstance(edges_array, np.ndarray)
        assert edges_array.shape[0] == 6  # k+1 edges  # type: ignore
        assert np.isneginf(edges_array[0])  # First edge should be -inf
        assert np.isinf(edges_array[-1])  # Last edge should be inf
        assert np.all(np.diff(edges_array[1:-1]) > 0)  # Edges should be strictly increasing

        # Test list output
        edges_list = woe.get_split_value_histogram("score", as_array=False)
        assert isinstance(edges_list, list)
        assert len(edges_list) == 6
        assert edges_list[0] == float("-inf")
        assert edges_list[-1] == float("inf")

    def test_faiss_kmeans_binning_import_error(self):
        """Test that FAISS KMeans raises ImportError when FAISS is not available."""
        # Mock the import to simulate FAISS not being available
        import sys
        from unittest.mock import patch

        with patch.dict(sys.modules, {"faiss": None}):
            woe = FastWoe(
                binning_method="faiss_kmeans",
                numerical_threshold=10,
                warn_on_numerical=False,
            )
            X = pd.DataFrame({"feature": np.random.randint(1, 100, 50)})
            y = np.random.binomial(1, 0.3, 50)

            with pytest.raises(ImportError, match="FAISS is required for faiss_kmeans"):
                woe.fit(X, y)

    def test_faiss_kmeans_binning_invalid_method(self):
        """Test that invalid binning method raises ValueError."""
        with pytest.raises(
            ValueError,
            match="binning_method must be 'kbins', 'tree', or 'faiss_kmeans'",
        ):
            FastWoe(binning_method="invalid_method")

    def test_faiss_kmeans_binning_continuous_target(self):
        """Test FAISS KMeans binning with continuous target values (proportions)."""
        import importlib.util

        if importlib.util.find_spec("faiss") is None:
            pytest.skip("FAISS not available, skipping FAISS KMeans tests")

        from scipy.special import expit as sigmoid  # pylint: disable=no-name-in-module

        # Create synthetic aggregated-binomial data
        np.random.seed(42)
        n_samples = 1000

        # Generate features
        X = np.random.normal(size=(n_samples, 1)).astype(np.float32)

        # Generate continuous target (proportions between 0 and 1)
        beta = np.random.normal(size=(1,)).astype(np.float32)
        logit = X @ beta + np.random.normal(scale=0.5, size=n_samples).astype(np.float32)
        p_true = sigmoid(logit)

        # Create DataFrame
        df = pd.DataFrame(X, columns=["x0"])
        df["p_true"] = p_true

        # Test with FAISS KMeans binning
        fw = FastWoe(
            binning_method="faiss_kmeans",
            faiss_kwargs={"k": 4, "niter": 10, "verbose": False, "gpu": False},
            numerical_threshold=10,
            warn_on_numerical=False,
        )
        fw.fit(df[["x0"]], df["p_true"])

        # Verify the fit worked
        assert fw.is_fitted_
        assert "x0" in fw.mappings_

        # Test transform
        X_transformed = fw.transform(df[["x0"]])
        assert X_transformed.shape == (n_samples, 1)
        assert not X_transformed.isna().any().any()

        # Test binning summary
        summary = fw.get_binning_summary()
        assert len(summary) == 1
        assert summary.iloc[0]["feature"] == "x0"
        assert summary.iloc[0]["method"] == "faiss_kmeans"
        assert summary.iloc[0]["n_bins"] == 4

        # Test split values
        splits = fw.get_split_value_histogram("x0")
        assert len(splits) == 5  # k+1 edges
        assert splits[0] == -np.inf
        assert splits[-1] == np.inf


class TestTreeBinning:
    """Test tree-based binning functionality."""

    def setup_method(self):
        """Set up test data."""
        # Create test data
        X, y = make_classification(
            n_samples=1000,
            n_features=5,
            n_informative=3,
            n_redundant=0,
            n_clusters_per_class=1,
            flip_y=0.03,
            weights=[0.95, 0.05],
            random_state=42,
        )
        self.X = pd.DataFrame(X, columns=[f"x_{i}" for i in range(X.shape[1])])
        self.y = pd.Series(y, name="y")

    def test_tree_binning_basic(self):
        """Test basic tree binning functionality."""
        binning_params = {
            "binning_method": "tree",
            "binner_kwargs": {"max_depth": 2, "min_samples_leaf": 10},
        }
        woe_model = FastWoe(**binning_params)
        woe_model.fit(self.X, self.y)

        # Check that all features are binned
        assert len(woe_model.get_binning_summary()) == 5
        assert all(woe_model.get_binning_summary()["method"] == "tree")

        # Check that mappings don't contain NaN values
        for feature in self.X.columns:
            mapping = woe_model.get_mapping(feature)
            assert not mapping["count"].isna().any()
            assert not mapping["woe"].isna().any()

    def test_tree_binning_max_depth_1(self):
        """Test tree binning with max_depth=1."""
        binning_params = {"binning_method": "tree", "binner_kwargs": {"max_depth": 1}}
        woe_model = FastWoe(**binning_params)
        woe_model.fit(self.X, self.y)

        # Check that all features have 2 bins (max_depth=1 creates 2 leaves)
        summary = woe_model.get_binning_summary()
        assert all(summary["n_bins"] == 2)

        # Check tree access
        for feature in self.X.columns:
            tree = woe_model.get_tree_estimator(feature)
            assert tree.get_depth() == 1
            assert tree.get_n_leaves() == 2

    def test_tree_binning_max_depth_3(self):
        """Test tree binning with max_depth=3."""
        binning_params = {"binning_method": "tree", "binner_kwargs": {"max_depth": 3}}
        woe_model = FastWoe(**binning_params)
        woe_model.fit(self.X, self.y)

        # Check that trees have appropriate depth
        for feature in self.X.columns:
            tree = woe_model.get_tree_estimator(feature)
            assert tree.get_depth() <= 3

    def test_tree_binning_restrictive_leaf_size(self):
        """Test tree binning with very restrictive min_samples_leaf."""
        binning_params = {
            "binning_method": "tree",
            "binner_kwargs": {"max_depth": 5, "min_samples_leaf": 500},
        }
        woe_model = FastWoe(**binning_params)
        woe_model.fit(self.X, self.y)

        # Should create fewer bins due to restrictive leaf size
        summary = woe_model.get_binning_summary()
        # With min_samples_leaf=500 and 1000 samples, we should get at most 2 bins
        assert all(summary["n_bins"] <= 2)

    def test_tree_binning_custom_estimator(self):
        """Test tree binning with custom tree estimator."""
        from sklearn.tree import DecisionTreeRegressor

        binning_params = {
            "binning_method": "tree",
            "tree_estimator": DecisionTreeRegressor(max_depth=2, min_samples_leaf=20),
            "binner_kwargs": {"max_depth": 2, "min_samples_leaf": 20},
        }
        woe_model = FastWoe(**binning_params)
        woe_model.fit(self.X, self.y)

        # Check that custom estimator is used
        # sourcery skip: no-loop-in-tests
        for feature in self.X.columns:
            tree = woe_model.get_tree_estimator(feature)
            assert isinstance(tree, DecisionTreeRegressor)

    def test_tree_binning_backward_compatibility(self):
        """Test backward compatibility with tree_kwargs."""
        binning_params = {
            "binning_method": "tree",
            "tree_kwargs": {"max_depth": 2, "min_samples_leaf": 10},
        }
        woe_model = FastWoe(**binning_params)
        woe_model.fit(self.X, self.y)

        # Should work the same as binner_kwargs
        summary = woe_model.get_binning_summary()
        assert len(summary) == 5
        assert all(summary["method"] == "tree")

    def test_tree_binning_unified_kwargs(self):
        """Test unified binner_kwargs approach."""
        # Test tree method with binner_kwargs
        binning_params = {
            "binning_method": "tree",
            "binner_kwargs": {"max_depth": 2, "min_samples_leaf": 10},
        }
        woe_model = FastWoe(**binning_params)
        woe_model.fit(self.X, self.y)

        summary = woe_model.get_binning_summary()
        assert len(summary) == 5
        assert all(summary["method"] == "tree")

    def test_tree_binning_edge_cases(self):
        """Test tree binning with edge cases."""
        # Test with very small dataset
        X_small, y_small = make_classification(
            n_samples=20,
            n_features=3,
            n_informative=1,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=42,
        )
        X_small = pd.DataFrame(X_small, columns=[f"x_{i}" for i in range(X_small.shape[1])])
        y_small = pd.Series(y_small, name="y")

        binning_params = {
            "binning_method": "tree",
            "binner_kwargs": {"max_depth": 2, "min_samples_leaf": 5},
        }
        woe_model = FastWoe(**binning_params)
        woe_model.fit(X_small, y_small)

        # Should still work
        summary = woe_model.get_binning_summary()
        assert len(summary) == 3
        assert all(summary["method"] == "tree")

    def test_tree_binning_no_splits(self):
        """Test tree binning when no splits are possible."""
        # Create data where no splits are meaningful
        X_no_splits = pd.DataFrame({"x_0": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})
        y_no_splits = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        binning_params = {
            "binning_method": "tree",
            "binner_kwargs": {"max_depth": 2, "min_samples_leaf": 2},
        }
        woe_model = FastWoe(**binning_params)
        woe_model.fit(X_no_splits, y_no_splits)

        # Should create single bin
        summary = woe_model.get_binning_summary()
        if len(summary) > 0:  # Check if any features were binned
            assert summary["n_bins"].iloc[0] == 1

    def test_tree_binning_error_handling(self):
        """Test error handling for tree binning."""
        woe_model = FastWoe(binning_method="tree")

        # Test accessing tree before fitting
        with pytest.raises(ValueError, match="FastWoe must be fitted"):
            woe_model.get_tree_estimator("x_0")

        woe_model.fit(self.X, self.y)

        # Test accessing tree for non-existent feature
        with pytest.raises(ValueError, match="Feature 'non_existent' not found"):
            woe_model.get_tree_estimator("non_existent")

        # Test accessing tree for non-tree binned feature
        # First create a model with kbins method
        woe_model_kbins = FastWoe(binning_method="kbins")
        woe_model_kbins.fit(self.X, self.y)

        with pytest.raises(ValueError, match="was not binned using tree method"):
            woe_model_kbins.get_tree_estimator("x_0")

    def test_tree_binning_continuous_target(self):
        """Test tree binning with continuous target (proportions)."""
        from sklearn.tree import DecisionTreeRegressor

        # Create continuous target (proportions between 0 and 1)
        y_continuous = np.random.beta(2, 5, size=len(self.y))

        binning_params = {
            "binning_method": "tree",
            "binner_kwargs": {"max_depth": 2, "min_samples_leaf": 10},
        }
        woe_model = FastWoe(**binning_params)
        woe_model.fit(self.X, y_continuous)

        # Should work with continuous target
        summary = woe_model.get_binning_summary()
        assert len(summary) == 5
        assert all(summary["method"] == "tree")

        # Check that trees are regressors for continuous targets
        # sourcery skip: no-loop-in-tests
        for feature in self.X.columns:
            tree = woe_model.get_tree_estimator(feature)
            assert isinstance(tree, DecisionTreeRegressor)

    def test_tree_binning_mapping_consistency(self):
        """Test that tree binning creates consistent mappings."""
        binning_params = {
            "binning_method": "tree",
            "binner_kwargs": {"max_depth": 2, "min_samples_leaf": 10},
        }
        woe_model = FastWoe(**binning_params)
        woe_model.fit(self.X, self.y)

        # Check that all bins have valid counts and WOE values
        for feature in self.X.columns:
            mapping = woe_model.get_mapping(feature)

            # All counts should be non-negative
            assert (mapping["count"] >= 0).all()

            # All WOE values should be finite
            assert mapping["woe"].notna().all()
            assert np.isfinite(mapping["woe"]).all()

            # All event rates should be between 0 and 1
            assert (mapping["event_rate"] >= 0).all()
            assert (mapping["event_rate"] <= 1).all()

    def test_tree_binning_reproducibility(self):
        """Test that tree binning is reproducible with same random_state."""
        binning_params = {
            "binning_method": "tree",
            "binner_kwargs": {
                "max_depth": 2,
                "min_samples_leaf": 10,
                "random_state": 42,
            },
        }

        # Fit two models with same parameters
        woe_model1 = FastWoe(**binning_params)
        woe_model1.fit(self.X, self.y)

        woe_model2 = FastWoe(**binning_params)
        woe_model2.fit(self.X, self.y)

        # Results should be identical
        for feature in self.X.columns:
            mapping1 = woe_model1.get_mapping(feature)
            mapping2 = woe_model2.get_mapping(feature)

            # Compare bin edges (should be identical)
            tree1 = woe_model1.get_tree_estimator(feature)
            tree2 = woe_model2.get_tree_estimator(feature)

            # Compare split points
            splits1 = woe_model1._extract_tree_splits(tree1, feature)
            splits2 = woe_model2._extract_tree_splits(tree2, feature)

            np.testing.assert_array_almost_equal(splits1, splits2, decimal=10)


class TestDefaultTreeBinning:
    """Test that FastWoe defaults to tree binning in version 0.1.4."""

    def test_default_binning_method(self):
        """Test that FastWoe defaults to tree binning method."""
        # Create test data
        X, y = make_classification(
            n_samples=100,
            n_features=4,
            n_informative=2,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=42,
        )
        X = pd.DataFrame(X, columns=[f"x_{i}" for i in range(X.shape[1])])
        y = pd.Series(y, name="y")

        # Test default behavior
        woe_model = FastWoe()
        assert woe_model.binning_method == "tree"
        assert woe_model.tree_kwargs["max_depth"] == 3
        assert woe_model.tree_kwargs["random_state"] == 42

        # Test that it works
        woe_model.fit(X, y)
        summary = woe_model.get_binning_summary()
        assert len(summary) == 4
        assert all(summary["method"] == "tree")

    def test_default_tree_parameters_optimized_for_credit_scoring(self):
        """Test that default tree parameters are optimized for credit scoring."""
        woe_model = FastWoe()

        # Check that parameters are optimized for credit scoring
        assert woe_model.tree_kwargs["max_depth"] == 3  # 4-8 bins per feature
        assert woe_model.tree_kwargs["random_state"] == 42  # Reproducible results

    def test_backward_compatibility_with_explicit_kbins(self):
        """Test that explicit kbins method still works."""
        X, y = make_classification(
            n_samples=100,
            n_features=4,
            n_informative=2,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=42,
        )
        X = pd.DataFrame(X, columns=[f"x_{i}" for i in range(X.shape[1])])
        y = pd.Series(y, name="y")

        # Test explicit kbins
        woe_model = FastWoe(binning_method="kbins")
        assert woe_model.binning_method == "kbins"

        woe_model.fit(X, y)
        summary = woe_model.get_binning_summary()
        assert all(summary["method"] == "kbins")

    def test_default_creates_interpretable_bins(self):
        """Test that default settings create interpretable number of bins."""
        X, y = make_classification(
            n_samples=1000,
            n_features=5,
            n_informative=3,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=42,
        )
        X = pd.DataFrame(X, columns=[f"x_{i}" for i in range(X.shape[1])])
        y = pd.Series(y, name="y")

        woe_model = FastWoe()  # Use defaults
        woe_model.fit(X, y)

        summary = woe_model.get_binning_summary()

        # Check that number of bins is reasonable for credit scoring
        for n_bins in summary["n_bins"]:
            assert 2 <= n_bins <= 15  # Reasonable range for credit scoring
            # Most features should have 4-8 bins (typical for credit scoring)
            assert n_bins >= 2  # At least 2 bins
            assert n_bins <= 15  # Not too many bins (interpretability)


class TestMulticlassWoe:
    """Test cases for multiclass WOE encoding."""

    def test_multiclass_detection(self):
        """Test that multiclass targets are correctly detected."""
        # Create multiclass data
        X = pd.DataFrame(
            {
                "feature1": ["A", "B", "C"] * 20,
                "feature2": [1, 2, 3] * 20,
            }
        )
        y = pd.Series([0, 1, 2] * 20)  # 3 classes

        woe = FastWoe()
        woe.fit(X, y)

        assert woe.is_multiclass_target
        assert not woe.is_binary_target
        assert not woe.is_continuous_target
        assert woe.classes_ == [0, 1, 2]
        assert woe.n_classes_ == 3
        assert woe.encoder_kwargs["target_type"] == "multiclass"

    def test_multiclass_priors(self):
        """Test that class priors are correctly calculated."""
        X = pd.DataFrame(
            {
                "feature1": ["A", "B", "C"] * 20,
            }
        )
        y = pd.Series([0, 1, 2] * 20)  # Balanced classes

        woe = FastWoe()
        woe.fit(X, y)

        # Check that y_prior_ is a dictionary
        assert isinstance(woe.y_prior_, dict)
        assert len(woe.y_prior_) == 3
        assert all(abs(woe.y_prior_[i] - 1 / 3) < 1e-10 for i in range(3))

        # Check odds_prior_per_class_
        assert len(woe.odds_prior_per_class_) == 3
        for i in range(3):
            expected_odds = (1 / 3) / (2 / 3)  # class_prior / (1 - class_prior)
            assert abs(woe.odds_prior_per_class_[i] - expected_odds) < 1e-10

    def test_multiclass_encoding_structure(self):
        """Test that multiclass encoding creates correct structure."""
        X = pd.DataFrame(
            {
                "feature1": ["A", "B", "C"] * 20,
                "feature2": [1, 2, 3] * 20,
            }
        )
        y = pd.Series([0, 1, 2] * 20)

        woe = FastWoe()
        woe.fit(X, y)

        # Check encoder structure
        assert len(woe.encoders_) == 2  # 2 features
        for feature in ["feature1", "feature2"]:
            assert isinstance(woe.encoders_[feature], dict)
            assert len(woe.encoders_[feature]) == 3  # 3 classes
            for class_label in [0, 1, 2]:
                assert class_label in woe.encoders_[feature]

        # Check mapping structure
        assert len(woe.mappings_) == 2
        for feature in ["feature1", "feature2"]:
            assert isinstance(woe.mappings_[feature], dict)
            assert len(woe.mappings_[feature]) == 3
            for class_label in [0, 1, 2]:
                assert class_label in woe.mappings_[feature]

    def test_multiclass_transform(self):
        """Test multiclass transform produces correct output."""
        X = pd.DataFrame(
            {
                "feature1": ["A", "B", "C"] * 20,
                "feature2": [1, 2, 3] * 20,
            }
        )
        y = pd.Series([0, 1, 2] * 20)

        woe = FastWoe()
        woe.fit(X, y)
        X_transformed = woe.transform(X)

        # Check output shape
        expected_cols = 2 * 3  # 2 features * 3 classes
        assert X_transformed.shape == (60, expected_cols)

        # Check column names
        expected_columns = [
            "feature1_class_0",
            "feature1_class_1",
            "feature1_class_2",
            "feature2_class_0",
            "feature2_class_1",
            "feature2_class_2",
        ]
        assert list(X_transformed.columns) == expected_columns

        # Check that all values are finite
        assert X_transformed.isna().sum().sum() == 0
        assert np.isfinite(X_transformed).all().all()

    def test_multiclass_with_imbalanced_classes(self):
        """Test multiclass WOE with imbalanced classes."""
        X = pd.DataFrame(
            {
                "feature1": ["A", "B", "C"] * 30,
            }
        )
        y = pd.Series([0] * 50 + [1] * 30 + [2] * 10)  # Imbalanced

        woe = FastWoe()
        woe.fit(X, y)

        # Check priors
        assert isinstance(woe.y_prior_, dict)
        assert abs(woe.y_prior_[0] - 50 / 90) < 1e-10
        assert abs(woe.y_prior_[1] - 30 / 90) < 1e-10
        assert abs(woe.y_prior_[2] - 10 / 90) < 1e-10

        # Transform should work
        X_transformed = woe.transform(X)
        assert X_transformed.shape == (90, 3)  # 1 feature * 3 classes

    def test_multiclass_with_numerical_binning(self):
        """Test multiclass WOE with numerical features that need binning."""
        X = pd.DataFrame(
            {
                "categorical": ["A", "B", "C"] * 30,
                "numerical": np.random.normal(0, 1, 90),
            }
        )
        y = pd.Series([0, 1, 2] * 30)

        woe = FastWoe(binning_method="tree", tree_kwargs={"max_depth": 2})
        woe.fit(X, y)

        # Should work with binning
        X_transformed = woe.transform(X)
        assert X_transformed.shape == (90, 6)  # 2 features * 3 classes

        # Check that numerical feature was binned
        assert "numerical" in woe.binners_

    def test_multiclass_error_handling(self):
        """Test error handling for multiclass cases."""
        X = pd.DataFrame(
            {
                "feature1": ["A", "B", "C"] * 20,
            }
        )

        # Test with single class (should raise error)
        y_single = pd.Series([0] * 60)
        woe = FastWoe()
        with pytest.raises(ValueError, match="Target variable must have at least 2 unique values"):
            woe.fit(X, y_single)

        # Test with two classes (should work as binary)
        y_binary = pd.Series([0, 1] * 30)
        woe = FastWoe()
        woe.fit(X, y_binary)
        assert not woe.is_multiclass_target
        assert woe.is_binary_target

    def test_multiclass_consistency(self):
        """Test that multiclass WOE is consistent across multiple fits."""
        X = pd.DataFrame(
            {
                "feature1": ["A", "B", "C"] * 20,
            }
        )
        y = pd.Series([0, 1, 2] * 20)

        woe1 = FastWoe(random_state=42)
        woe1.fit(X, y)

        woe2 = FastWoe(random_state=42)
        woe2.fit(X, y)

        # Should produce identical results
        X_trans1 = woe1.transform(X)
        X_trans2 = woe2.transform(X)

        pd.testing.assert_frame_equal(X_trans1, X_trans2)

    def test_multiclass_with_string_labels(self):
        """Test multiclass WOE with string class labels."""
        X = pd.DataFrame(
            {
                "feature1": ["A", "B", "C"] * 20,
            }
        )
        y = pd.Series(["class_0", "class_1", "class_2"] * 20)

        woe = FastWoe()
        woe.fit(X, y)

        assert woe.is_multiclass_target
        assert woe.classes_ == ["class_0", "class_1", "class_2"]

        X_transformed = woe.transform(X)
        expected_columns = [
            "feature1_class_class_0",
            "feature1_class_class_1",
            "feature1_class_class_2",
        ]
        assert list(X_transformed.columns) == expected_columns


class TestMonotonicConstraints:
    """Test cases for monotonic constraints functionality."""

    def test_init_with_monotonic_constraints(self):
        """Test initialization with monotonic constraints."""
        # Test valid constraints
        woe = FastWoe(monotonic_cst={"income": 1, "age": -1, "score": 0})
        assert woe.monotonic_cst == {"income": 1, "age": -1, "score": 0}

        # Test empty constraints
        woe = FastWoe(monotonic_cst={})
        assert woe.monotonic_cst == {}

        # Test None constraints
        woe = FastWoe(monotonic_cst=None)
        assert woe.monotonic_cst == {}

    def test_monotonic_constraints_validation(self):
        """Test validation of monotonic constraints."""
        # Test invalid constraint values
        with pytest.raises(ValueError, match="Constraint value must be -1, 0, or 1"):
            FastWoe(monotonic_cst={"income": 2})

        with pytest.raises(ValueError, match="Constraint value must be -1, 0, or 1"):
            FastWoe(monotonic_cst={"income": -2})

        # Test invalid constraint types
        with pytest.raises(TypeError, match="Constraint values must be integers"):
            FastWoe(monotonic_cst={"income": "1"})

        with pytest.raises(TypeError, match="Constraint values must be integers"):
            FastWoe(monotonic_cst={"income": 1.0})

        # Test invalid feature name types
        with pytest.raises(TypeError, match="Feature names in monotonic_cst must be strings"):
            FastWoe(monotonic_cst={123: 1})

        # Test invalid constraint dict type
        with pytest.raises(TypeError, match="monotonic_cst must be a dictionary"):
            FastWoe(monotonic_cst="invalid")

    def test_monotonic_constraints_warning_for_unsupported_methods(self):
        """Test warning when using monotonic constraints with unsupported binning methods."""
        # Test with an invalid binning method (should raise ValueError)
        with pytest.raises(ValueError, match="binning_method must be"):
            FastWoe(binning_method="invalid_method", monotonic_cst={"income": 1})

    def test_monotonic_constraints_tree_binning(self):
        """Test monotonic constraints with tree binning."""
        # Create synthetic data with clear monotonic relationship
        np.random.seed(42)
        n_samples = 1000

        # Income: higher income -> lower risk (decreasing)
        income = np.random.normal(50, 15, n_samples)
        income_risk = 1 / (1 + np.exp((income - 50) / 10))  # Decreasing relationship

        # Age: higher age -> higher risk (increasing)
        age = np.random.normal(35, 10, n_samples)
        age_risk = 1 / (1 + np.exp(-(age - 35) / 5))  # Increasing relationship

        # Combine risks
        combined_risk = (income_risk + age_risk) / 2
        y = (combined_risk > 0.5).astype(int)

        X_df = pd.DataFrame({"income": income, "age": age})

        # Test with monotonic constraints
        woe = FastWoe(
            binning_method="tree",
            monotonic_cst={
                "income": -1,
                "age": 1,
            },  # income decreases risk, age increases risk
            numerical_threshold=10,
        )

        woe.fit(X_df, y)

        # Check that constraints were applied
        assert woe.binning_info_["income"]["monotonic_constraint"] == -1
        assert woe.binning_info_["age"]["monotonic_constraint"] == 1

        # Check that binning summary includes monotonic constraints
        summary = woe.get_binning_summary()
        assert "monotonic_constraint" in summary.columns
        assert summary[summary["feature"] == "income"]["monotonic_constraint"].iloc[0] == -1
        assert summary[summary["feature"] == "age"]["monotonic_constraint"].iloc[0] == 1

    def test_monotonic_constraints_no_constraint(self):
        """Test that constraint value 0 means no constraint."""
        np.random.seed(42)
        n_samples = 500

        # Create data without clear monotonic relationship
        X = np.random.normal(0, 1, (n_samples, 1))
        y = np.random.binomial(1, 0.5, n_samples)

        X_df = pd.DataFrame(X, columns=["feature"])

        woe = FastWoe(
            binning_method="tree",
            monotonic_cst={"feature": 0},  # No constraint
            numerical_threshold=10,
        )

        woe.fit(X_df, y)

        # Check that no constraint was applied
        assert woe.binning_info_["feature"]["monotonic_constraint"] == 0

    def test_monotonic_constraints_partial_features(self):
        """Test monotonic constraints applied only to specified features."""
        np.random.seed(42)
        n_samples = 500

        X_df = pd.DataFrame(
            {
                "income": np.random.normal(50, 15, n_samples),
                "age": np.random.normal(35, 10, n_samples),
                "score": np.random.normal(0, 1, n_samples),
            }
        )
        y = np.random.binomial(1, 0.5, n_samples)

        woe = FastWoe(
            binning_method="tree",
            monotonic_cst={"income": 1},  # Only income has constraint
            numerical_threshold=10,
        )

        woe.fit(X_df, y)

        # Check constraints
        assert woe.binning_info_["income"]["monotonic_constraint"] == 1
        assert woe.binning_info_["age"]["monotonic_constraint"] == 0  # No constraint
        assert woe.binning_info_["score"]["monotonic_constraint"] == 0  # No constraint

    def test_monotonic_constraints_functional_validation_tree(self):
        """Test that tree monotonic constraints actually produce monotonic WOE values."""
        # Create synthetic data with clear monotonic relationship
        np.random.seed(42)
        n_samples = 1000

        # Create feature with clear relationship to target
        x = np.random.uniform(0, 10, n_samples)
        # Higher x should lead to higher probability (increasing relationship)
        prob = 1 / (1 + np.exp(-(x - 5) / 2))
        y = np.random.binomial(1, prob, n_samples)

        X_df = pd.DataFrame(x.reshape(-1, 1), columns=["feature"])

        # Test increasing constraint
        woe_inc = FastWoe(binning_method="tree", monotonic_cst={"feature": 1})
        woe_inc.fit(X_df, y)

        # Transform test data in sorted order
        x_test = np.linspace(0, 10, 20)
        X_test = pd.DataFrame(x_test.reshape(-1, 1), columns=["feature"])
        woe_values_inc = woe_inc.transform(X_test).values.flatten()

        # Check that WOE is increasing (allowing small numerical errors)
        # sourcery skip: no-loop-in-tests
        for i in range(len(woe_values_inc) - 1):
            assert woe_values_inc[i] <= woe_values_inc[i + 1] + 1e-10, (
                f"WOE not increasing at position {i}: {woe_values_inc[i]:.6f} > {woe_values_inc[i + 1]:.6f}"
            )

        # Test decreasing constraint
        woe_dec = FastWoe(binning_method="tree", monotonic_cst={"feature": -1})
        woe_dec.fit(X_df, y)
        woe_values_dec = woe_dec.transform(X_test).values.flatten()

        # Check that WOE is decreasing
        for i in range(len(woe_values_dec) - 1):
            assert woe_values_dec[i] >= woe_values_dec[i + 1] - 1e-10, (
                f"WOE not decreasing at position {i}: {woe_values_dec[i]:.6f} < {woe_values_dec[i + 1]:.6f}"
            )

    def test_monotonic_constraints_functional_validation_faiss(self):
        # sourcery skip: extract-duplicate-method
        """Test that FAISS monotonic constraints are properly ignored and warnings are shown."""
        # Create synthetic data with clear monotonic relationship
        np.random.seed(42)
        n_samples = 1000

        # Create feature with clear relationship to target
        x = np.random.uniform(0, 10, n_samples)
        # Higher x should lead to higher probability (increasing relationship)
        prob = 1 / (1 + np.exp(-(x - 5) / 2))
        y = np.random.binomial(1, prob, n_samples)

        X_df = pd.DataFrame(x.reshape(-1, 1), columns=["feature"])

        # Test that constraints are ignored and warnings are shown
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            woe_inc = FastWoe(binning_method="faiss_kmeans", monotonic_cst={"feature": 1})
            woe_inc.fit(X_df, y)

            # Check that warnings were raised
            warning_messages = [str(warning.message) for warning in w]
            assert any(
                "Monotonic constraints specified but binning_method='faiss_kmeans'" in msg
                for msg in warning_messages
            )
            assert any(
                "Monotonic constraints for feature 'feature' are ignored" in msg
                for msg in warning_messages
            )

        # Test that the same warnings occur for decreasing constraints
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            woe_dec = FastWoe(binning_method="faiss_kmeans", monotonic_cst={"feature": -1})
            woe_dec.fit(X_df, y)

            # Check that warnings were raised
            warning_messages = [str(warning.message) for warning in w]
            assert any(
                "Monotonic constraints specified but binning_method='faiss_kmeans'" in msg
                for msg in warning_messages
            )
            assert any(
                "Monotonic constraints for feature 'feature' are ignored" in msg
                for msg in warning_messages
            )

    def test_monotonic_constraints_functional_validation_kbins(self):
        # sourcery skip: extract-duplicate-method
        """Test that kbins monotonic constraints are properly ignored and warnings are shown."""
        # Create synthetic data with clear monotonic relationship
        np.random.seed(42)
        n_samples = 1000

        # Create feature with clear relationship to target
        x = np.random.uniform(0, 10, n_samples)
        # Higher x should lead to higher probability (increasing relationship)
        prob = 1 / (1 + np.exp(-(x - 5) / 2))
        y = np.random.binomial(1, prob, n_samples)

        X_df = pd.DataFrame(x.reshape(-1, 1), columns=["feature"])

        # Test that constraints are ignored and warnings are shown
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            woe_inc = FastWoe(
                binning_method="kbins",
                binner_kwargs={"n_bins": 10, "strategy": "uniform"},
                monotonic_cst={"feature": 1},
            )
            woe_inc.fit(X_df, y)

            # Check that warnings were raised
            warning_messages = [str(warning.message) for warning in w]
            assert any(
                "Monotonic constraints specified but binning_method='kbins'" in msg
                for msg in warning_messages
            )
            assert any(
                "Monotonic constraints for feature 'feature' are ignored" in msg
                for msg in warning_messages
            )

        # Test that the same warnings occur for decreasing constraints
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            woe_dec = FastWoe(
                binning_method="kbins",
                binner_kwargs={"n_bins": 10, "strategy": "uniform"},
                monotonic_cst={"feature": -1},
            )
            woe_dec.fit(X_df, y)

            # Check that warnings were raised
            warning_messages = [str(warning.message) for warning in w]
            assert any(
                "Monotonic constraints specified but binning_method='kbins'" in msg
                for msg in warning_messages
            )
            assert any(
                "Monotonic constraints for feature 'feature' are ignored" in msg
                for msg in warning_messages
            )

    def test_monotonic_vs_non_monotonic_difference(self):
        """Test that monotonic constraints actually change the WOE values."""
        # Create data that would naturally violate monotonicity
        np.random.seed(42)
        n_samples = 1000

        # Create feature where middle values have different target distribution
        x = np.random.uniform(0, 10, n_samples)
        # Create non-monotonic relationship: high probability at ends, low in middle
        prob = np.where((x < 3) | (x > 7), 0.8, 0.2)
        y = np.random.binomial(1, prob, n_samples)

        X_df = pd.DataFrame(x.reshape(-1, 1), columns=["feature"])
        x_test = np.linspace(0, 10, 10)
        X_test = pd.DataFrame(x_test.reshape(-1, 1), columns=["feature"])

        # Test tree method (always available)
        method = "tree"
        # Without monotonic constraint
        woe_normal = FastWoe(binning_method=method)
        woe_normal.fit(X_df, y)
        woe_values_normal = woe_normal.transform(X_test).values.flatten()

        # With monotonic constraint (decreasing)
        woe_mono = FastWoe(binning_method=method, monotonic_cst={"feature": -1})
        woe_mono.fit(X_df, y)
        woe_values_mono = woe_mono.transform(X_test).values.flatten()

        # They should be different (monotonic constraint should change the values)
        assert not np.allclose(woe_values_normal, woe_values_mono, atol=1e-6), (
            f"Monotonic constraint had no effect for {method}"
        )

        # Monotonic values should be decreasing
        # sourcery skip: no-loop-in-tests
        for i in range(len(woe_values_mono) - 1):
            assert woe_values_mono[i] >= woe_values_mono[i + 1] - 1e-10, (
                f"{method} monotonic WOE not decreasing: {woe_values_mono[i]:.6f} < {woe_values_mono[i + 1]:.6f}"
            )

    def test_monotonic_vs_non_monotonic_difference_faiss(self):
        """Test that FAISS monotonic constraints are ignored and warnings are shown."""
        # Skip if FAISS not available
        faiss = pytest.importorskip("faiss")

        # Create data that would naturally violate monotonicity
        np.random.seed(42)
        n_samples = 1000

        # Create feature where middle values have different target distribution
        x = np.random.uniform(0, 10, n_samples)
        # Create non-monotonic relationship: high probability at ends, low in middle
        prob = np.where((x < 3) | (x > 7), 0.8, 0.2)
        y = np.random.binomial(1, prob, n_samples)

        X_df = pd.DataFrame(x.reshape(-1, 1), columns=["feature"])
        x_test = np.linspace(0, 10, 10)
        X_test = pd.DataFrame(x_test.reshape(-1, 1), columns=["feature"])

        method = "faiss_kmeans"
        # Without monotonic constraint
        woe_normal = FastWoe(binning_method=method)
        woe_normal.fit(X_df, y)
        woe_values_normal = woe_normal.transform(X_test).values.flatten()

        # With monotonic constraint (should be ignored)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            woe_mono = FastWoe(binning_method=method, monotonic_cst={"feature": -1})
            woe_mono.fit(X_df, y)
            woe_values_mono = woe_mono.transform(X_test).values.flatten()

            # Check that warnings were raised
            warning_messages = [str(warning.message) for warning in w]
            assert any(
                "Monotonic constraints specified but binning_method='faiss_kmeans'" in msg
                for msg in warning_messages
            )
            assert any(
                "Monotonic constraints for feature 'feature' are ignored" in msg
                for msg in warning_messages
            )

        # They should be identical (monotonic constraint should have no effect)
        assert np.allclose(woe_values_normal, woe_values_mono, atol=1e-6), (
            f"Monotonic constraint incorrectly affected values for {method}"
        )

    def test_monotonic_constraints_with_custom_tree_estimator(self):
        """Test monotonic constraints with custom tree estimator."""
        from sklearn.tree import DecisionTreeClassifier

        np.random.seed(42)
        n_samples = 500

        X_df = pd.DataFrame({"feature": np.random.normal(0, 1, n_samples)})
        y = np.random.binomial(1, 0.5, n_samples)

        # Custom tree estimator
        custom_tree = DecisionTreeClassifier(max_depth=2, random_state=42)

        woe = FastWoe(
            binning_method="tree",
            tree_estimator=custom_tree,
            monotonic_cst={"feature": 1},
            numerical_threshold=10,
        )

        woe.fit(X_df, y)

        # Check that constraint was applied
        assert woe.binning_info_["feature"]["monotonic_constraint"] == 1

    def test_monotonic_constraints_continuous_target(self):
        """Test monotonic constraints with continuous target."""
        np.random.seed(42)
        n_samples = 500

        # Create continuous target with monotonic relationship
        income = np.random.normal(50, 15, n_samples)
        y_continuous = np.clip(1 / (1 + np.exp((income - 50) / 10)), 0, 1)  # Decreasing

        X_df = pd.DataFrame({"income": income})

        woe = FastWoe(
            binning_method="tree",
            monotonic_cst={"income": -1},  # Decreasing constraint
            numerical_threshold=10,
        )

        woe.fit(X_df, y_continuous)

        # Check that constraint was applied
        assert woe.binning_info_["income"]["monotonic_constraint"] == -1

    def test_monotonic_constraints_backward_compatibility(self):
        """Test that existing code without monotonic constraints still works."""
        np.random.seed(42)
        n_samples = 500

        X_df = pd.DataFrame({"feature": np.random.normal(0, 1, n_samples)})
        y = np.random.binomial(1, 0.5, n_samples)

        # Old code without monotonic_cst parameter
        woe = FastWoe(binning_method="tree", numerical_threshold=10)
        woe.fit(X_df, y)

        # Should work without issues
        assert woe.is_fitted_
        assert "feature" in woe.binning_info_
        assert woe.binning_info_["feature"]["monotonic_constraint"] == 0

    def test_monotonic_constraints_kbins_warning(self):
        """Test that monotonic constraints with KBins show appropriate warnings."""
        np.random.seed(42)
        n_samples = 1000

        # Create data with clear monotonic relationship
        income = np.random.lognormal(mean=10, sigma=0.5, size=n_samples)
        income_risk = 1 / (1 + np.exp((income - np.median(income)) / 20))  # Decreasing
        y = (income_risk > 0.5).astype(int)

        X_df = pd.DataFrame({"income": income})

        # Test that warnings are issued for monotonic constraints with kbins
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            woe = FastWoe(
                binning_method="kbins",
                monotonic_cst={"income": -1},  # Decreasing constraint
                numerical_threshold=10,
                binner_kwargs={"n_bins": 5, "strategy": "quantile"},
            )

            # Should warn during initialization
            assert len(w) > 0
            assert any("Monotonic constraints" in str(warning.message) for warning in w)

            woe.fit(X_df, y)

            # Should warn during fitting as well
            assert any(
                "ignored for binning method 'kbins'" in str(warning.message) for warning in w
            )

        # Verify that the monotonic constraints are actually ignored (no isotonic adjustment)
        mapping = woe.mappings_["income"]

        # Check that constraint information is stored but ignored for kbins
        assert woe.binning_info_["income"]["monotonic_constraint"] == -1
        assert woe.binning_info_["income"]["method"] == "kbins"

    def test_monotonic_constraints_faiss_warning(self):
        """Test that monotonic constraints with FAISS show appropriate warnings."""
        np.random.seed(42)
        n_samples = 500

        # Create data with clear monotonic relationship
        age = np.random.normal(35, 10, n_samples)
        age_risk = 1 / (1 + np.exp(-(age - 35) / 8))  # Increasing
        y = (age_risk > 0.5).astype(int)

        X_df = pd.DataFrame({"age": age})

        # Test that warnings are issued for monotonic constraints with FAISS
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            woe = FastWoe(
                binning_method="faiss_kmeans",
                monotonic_cst={"age": 1},  # Increasing constraint
                numerical_threshold=10,
                faiss_kwargs={"k": 4, "niter": 10},
            )

            # Should warn during initialization
            assert len(w) > 0
            assert any("Monotonic constraints" in str(warning.message) for warning in w)

            try:
                woe.fit(X_df, y)

                # Should warn during fitting as well
                assert any(
                    "ignored for binning method 'faiss_kmeans'" in str(warning.message)
                    for warning in w
                )

                # Check that constraint information is stored but ignored for FAISS
                assert woe.binning_info_["age"]["monotonic_constraint"] == 1
                assert woe.binning_info_["age"]["method"] == "faiss_kmeans"

            except ImportError:
                pytest.skip("FAISS not available, skipping FAISS KMeans test")

    def test_monotonic_constraints_multiclass_kbins(self):
        """Test that monotonic constraints with multiclass targets and KBins show warnings."""
        np.random.seed(42)
        n_samples = 500

        # Create multiclass data
        income = np.random.lognormal(mean=10, sigma=0.5, size=n_samples)
        y_multiclass = np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.4, 0.2])

        X_df = pd.DataFrame({"income": income})

        # Test that warnings are issued for monotonic constraints with kbins
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            woe = FastWoe(
                binning_method="kbins",
                monotonic_cst={"income": -1},  # Decreasing constraint
                numerical_threshold=10,
                binner_kwargs={"n_bins": 4, "strategy": "quantile"},
            )

            woe.fit(X_df, y_multiclass)

            # Should warn about ignored monotonic constraints
            assert any("Monotonic constraints" in str(warning.message) for warning in w)
            assert any(
                "ignored in multiclass WOE encoding" in str(warning.message) for warning in w
            )

        # Check that constraint information is stored but ignored for all classes
        assert woe.binning_info_["income"]["monotonic_constraint"] == -1

    def test_monotonic_constraints_continuous_target_kbins(self):
        """Test that monotonic constraints with continuous target and KBins show warnings."""
        np.random.seed(42)
        n_samples = 500

        # Create continuous target with monotonic relationship
        income = np.random.lognormal(mean=10, sigma=0.5, size=n_samples)
        y_continuous = np.clip(1 / (1 + np.exp((income - np.median(income)) / 20)), 0, 1)

        X_df = pd.DataFrame({"income": income})

        # Test that warnings are issued for monotonic constraints with kbins
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            woe = FastWoe(
                binning_method="kbins",
                monotonic_cst={"income": -1},  # Decreasing constraint
                numerical_threshold=10,
                binner_kwargs={"n_bins": 4, "strategy": "quantile"},
            )

            woe.fit(X_df, y_continuous)

            # Should warn about ignored monotonic constraints
            assert any("Monotonic constraints" in str(warning.message) for warning in w)
            assert any(
                "ignored for binning method 'kbins'" in str(warning.message) for warning in w
            )

        # Check that constraint information is stored but ignored
        assert woe.binning_info_["income"]["monotonic_constraint"] == -1

    def test_monotonic_constraints_mixed_methods(self):
        """Test monotonic constraints with different binning methods."""
        np.random.seed(42)
        n_samples = 500

        X_df = pd.DataFrame(
            {
                "income": np.random.lognormal(mean=10, sigma=0.5, size=n_samples),
                "age": np.random.normal(35, 10, n_samples),
                "score": np.random.normal(0, 1, n_samples),
            }
        )
        y = np.random.binomial(1, 0.5, n_samples)

        # Test tree method (native constraints)
        woe_tree = FastWoe(
            binning_method="tree",
            monotonic_cst={"income": -1, "age": 1},
            numerical_threshold=10,
        )
        woe_tree.fit(X_df, y)

        # Test KBins method (isotonic regression)
        woe_kbins = FastWoe(
            binning_method="kbins",
            monotonic_cst={"income": -1, "age": 1},
            numerical_threshold=10,
            binner_kwargs={"n_bins": 4, "strategy": "quantile"},
        )
        woe_kbins.fit(X_df, y)

        # Both should have constraints applied
        assert woe_tree.binning_info_["income"]["monotonic_constraint"] == -1
        assert woe_tree.binning_info_["age"]["monotonic_constraint"] == 1
        assert woe_kbins.binning_info_["income"]["monotonic_constraint"] == -1
        assert woe_kbins.binning_info_["age"]["monotonic_constraint"] == 1

        # Both should have different methods
        assert woe_tree.binning_info_["income"]["method"] == "tree"
        assert woe_kbins.binning_info_["income"]["method"] == "kbins"

    def test_monotonic_constraints_edge_cases(self):
        # sourcery skip: extract-duplicate-method
        """Test edge cases for monotonic constraints."""
        np.random.seed(42)
        n_samples = 100

        # Test with very few bins
        X_df = pd.DataFrame({"feature": np.random.normal(0, 1, n_samples)})
        y = np.random.binomial(1, 0.5, n_samples)

        woe = FastWoe(
            binning_method="kbins",
            monotonic_cst={"feature": 1},
            numerical_threshold=10,
            binner_kwargs={"n_bins": 2, "strategy": "quantile"},  # Only 2 bins
        )

        woe.fit(X_df, y)

        # Should work even with few bins
        assert woe.is_fitted_
        assert woe.binning_info_["feature"]["monotonic_constraint"] == 1

        # Test with constraint value 0 (no constraint)
        woe_no_constraint = FastWoe(
            binning_method="kbins",
            monotonic_cst={"feature": 0},  # No constraint
            numerical_threshold=10,
            binner_kwargs={"n_bins": 3, "strategy": "quantile"},
        )

        woe_no_constraint.fit(X_df, y)

        # Should work without applying constraints
        assert woe_no_constraint.is_fitted_
        assert woe_no_constraint.binning_info_["feature"]["monotonic_constraint"] == 0
