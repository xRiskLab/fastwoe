"""Tests for FastWoe library."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

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
        df = pd.DataFrame({
            'cat1': ['A'] * 50 + ['B'] * 30 + ['C'] * 5,
            'cat2': ['X'] * 40 + ['Y'] * 30 + ['Z'] * 15,
            'num': range(85)
        })

        preprocessor = WoePreprocessor(min_count=10)
        preprocessor.fit(df)

        assert 'cat1' in preprocessor.category_maps
        assert 'cat2' in preprocessor.category_maps
        assert len(preprocessor.cat_features_) == 2

    def test_fit_with_cat_features(self):
        """Test fitting with specified categorical features."""
        df = pd.DataFrame({
            'cat1': ['A'] * 50 + ['B'] * 30,
            'cat2': ['X'] * 40 + ['Y'] * 40,
            'num': range(80)
        })

        preprocessor = WoePreprocessor()
        preprocessor.fit(df, cat_features=['cat1'])

        assert 'cat1' in preprocessor.category_maps
        assert 'cat2' not in preprocessor.category_maps
        assert len(preprocessor.cat_features_) == 1

    def test_transform(self):
        """Test transformation functionality."""
        df = pd.DataFrame({
            'cat1': ['A'] * 50 + ['B'] * 30 + ['C'] * 5,
        })

        preprocessor = WoePreprocessor(min_count=10)
        preprocessor.fit(df)
        df_transformed = preprocessor.transform(df)

        # Check that rare category 'C' was replaced
        assert '__other__' in df_transformed['cat1'].values
        assert 'C' not in df_transformed['cat1'].values

    def test_fit_transform(self):
        """Test fit_transform method."""
        df = pd.DataFrame({
            'cat1': ['A'] * 50 + ['B'] * 30 + ['C'] * 5,
        })

        preprocessor = WoePreprocessor(min_count=10)
        df_transformed = preprocessor.fit_transform(df)

        assert '__other__' in df_transformed['cat1'].values
        assert 'C' not in df_transformed['cat1'].values

    def test_binary_categories(self):
        """Test handling of binary categories."""
        df = pd.DataFrame({
            'binary_cat': ['Yes', 'No'] * 25,
        })

        preprocessor = WoePreprocessor(min_count=10)
        preprocessor.fit(df)
        
        # Binary categories should be kept as-is
        assert len(preprocessor.category_maps['binary_cat']) == 2
        assert 'Yes' in preprocessor.category_maps['binary_cat']
        assert 'No' in preprocessor.category_maps['binary_cat']

    def test_get_reduction_summary(self):
        """Test cardinality reduction summary."""
        df = pd.DataFrame({
            'cat1': ['A'] * 50 + ['B'] * 30 + ['C'] * 5 + ['D'] * 3,
        })

        preprocessor = WoePreprocessor(min_count=10)
        preprocessor.fit(df)
        summary = preprocessor.get_reduction_summary(df)

        assert isinstance(summary, pd.DataFrame)
        assert 'feature' in summary.columns
        assert 'original_categories' in summary.columns
        assert 'kept_categories' in summary.columns
        assert 'reduction_pct' in summary.columns


class TestFastWoe:
    """Test cases for FastWoe class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        data = pd.DataFrame({
            'cat1': ['A'] * 50 + ['B'] * 30 + ['C'] * 20,
            'cat2': ['X'] * 40 + ['Y'] * 35 + ['Z'] * 25,
            'target': np.random.binomial(1, 0.3, 100)
        })
        # Create some correlation with target
        data.loc[data['cat1'] == 'A', 'target'] = np.random.binomial(1, 0.5, 50)
        data.loc[data['cat1'] == 'B', 'target'] = np.random.binomial(1, 0.2, 30)
        data.loc[data['cat1'] == 'C', 'target'] = np.random.binomial(1, 0.1, 20)
        
        return data

    def test_init(self):
        """Test initialization of FastWoe."""
        woe = FastWoe()
        assert woe.random_state == 42
        assert woe.encoder_kwargs == {"smooth": 1e-5}
        assert not woe.is_fitted_

        # Test custom initialization
        woe = FastWoe(encoder_kwargs={"smooth": 1e-3}, random_state=123)
        assert woe.random_state == 123
        assert woe.encoder_kwargs == {"smooth": 1e-3}

    def test_fit(self, sample_data):
        """Test fitting functionality."""
        X = sample_data[['cat1', 'cat2']]
        y = sample_data['target']

        woe = FastWoe()
        woe.fit(X, y)

        assert woe.is_fitted_
        assert 'cat1' in woe.mappings_
        assert 'cat2' in woe.mappings_
        assert woe.y_prior_ == y.mean()

    def test_transform(self, sample_data):
        """Test transformation functionality."""
        X = sample_data[['cat1', 'cat2']]
        y = sample_data['target']

        woe = FastWoe()
        woe.fit(X, y)
        X_transformed = woe.transform(X)

        assert X_transformed.shape == X.shape
        assert isinstance(X_transformed, pd.DataFrame)
        # Check that values are WOE scores (floats)
        assert X_transformed.dtypes['cat1'] == 'float64'
        assert X_transformed.dtypes['cat2'] == 'float64'

    def test_transform_not_fitted(self, sample_data):
        """Test transform raises error when not fitted."""
        X = sample_data[['cat1', 'cat2']]
        woe = FastWoe()
        
        with pytest.raises(TypeError):  # Will get TypeError because y_prior_ is None
            woe.transform(X)

    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        X = sample_data[['cat1', 'cat2']]
        y = sample_data['target']

        woe = FastWoe()
        X_transformed = woe.fit_transform(X, y)

        assert X_transformed.shape == X.shape
        assert isinstance(X_transformed, pd.DataFrame)
        assert woe.is_fitted_

    def test_get_mapping(self, sample_data):
        """Test getting mapping for specific column."""
        X = sample_data[['cat1', 'cat2']]
        y = sample_data['target']

        woe = FastWoe()
        woe.fit(X, y)
        mapping = woe.get_mapping('cat1')

        assert isinstance(mapping, pd.DataFrame)
        assert 'category' in mapping.columns
        assert 'count' in mapping.columns
        assert 'event_rate' in mapping.columns
        assert 'woe' in mapping.columns

    def test_get_all_mappings(self, sample_data):
        """Test getting all mappings."""
        X = sample_data[['cat1', 'cat2']]
        y = sample_data['target']

        woe = FastWoe()
        woe.fit(X, y)
        all_mappings = woe.get_all_mappings()

        assert isinstance(all_mappings, dict)
        assert 'cat1' in all_mappings
        assert 'cat2' in all_mappings

    def test_get_feature_stats(self, sample_data):
        """Test getting feature statistics."""
        X = sample_data[['cat1', 'cat2']]
        y = sample_data['target']

        woe = FastWoe()
        woe.fit(X, y)
        
        # Test single feature stats
        stats_cat1 = woe.get_feature_stats('cat1')
        assert isinstance(stats_cat1, pd.DataFrame)
        assert 'feature' in stats_cat1.columns
        
        # Test all features stats
        all_stats = woe.get_feature_stats()
        assert len(all_stats) == 2  # Two features

    def test_predict_proba(self, sample_data):
        """Test predict_proba method."""
        X = sample_data[['cat1', 'cat2']]
        y = sample_data['target']

        woe = FastWoe()
        woe.fit(X, y)
        probas = woe.predict_proba(X)

        assert isinstance(probas, np.ndarray)
        assert probas.shape == (len(X), 2)  # Returns [neg_prob, pos_prob]
        assert np.all((probas >= 0) & (probas <= 1))
        assert np.allclose(probas.sum(axis=1), 1.0)  # Probabilities sum to 1

    def test_predict_ci(self, sample_data):
        """Test confidence interval prediction."""
        X = sample_data[['cat1', 'cat2']]
        y = sample_data['target']

        woe = FastWoe()
        woe.fit(X, y)
        ci_df = woe.predict_ci(X)

        assert isinstance(ci_df, pd.DataFrame)
        assert 'ci_lower' in ci_df.columns
        assert 'ci_upper' in ci_df.columns
        assert len(ci_df) == len(X)
        assert np.all((ci_df['ci_lower'] >= 0) & (ci_df['ci_lower'] <= 1))
        assert np.all((ci_df['ci_upper'] >= 0) & (ci_df['ci_upper'] <= 1))

    def test_calculate_woe_se(self):
        """Test WOE standard error calculation."""
        woe = FastWoe()
        
        # Normal case
        se = woe._calculate_woe_se(good_count=20, bad_count=30)
        expected_se = np.sqrt(1/20 + 1/30)
        assert np.isclose(se, expected_se)
        
        # Edge cases
        se_zero_good = woe._calculate_woe_se(good_count=0, bad_count=30)
        assert se_zero_good == np.inf
        
        se_zero_bad = woe._calculate_woe_se(good_count=20, bad_count=0)
        assert se_zero_bad == np.inf

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
        X = sample_data[['cat1', 'cat2']]
        y = sample_data['target']

        woe = FastWoe()
        woe.fit(X, y)
        
        # Test WOE output - returns original column names
        woe_result = woe.transform_standardized(X, output="woe")
        assert 'cat1' in woe_result.columns or 'cat2' in woe_result.columns
        
        # Test wald output 
        wald_result = woe.transform_standardized(X, output="wald")
        assert 'cat1' in wald_result.columns or 'cat2' in wald_result.columns

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with single category
        df_single = pd.DataFrame({
            'cat': ['A'] * 100,
            'target': np.random.binomial(1, 0.3, 100)
        })
        
        woe = FastWoe()
        woe.fit(df_single[['cat']], df_single['target'])
        transformed = woe.transform(df_single[['cat']])
        
        # Should handle single category gracefully
        assert not np.isnan(transformed['cat']).any()

    def test_missing_values(self):
        """Test handling of missing values."""
        df_missing = pd.DataFrame({
            'cat': ['A'] * 40 + ['B'] * 40 + [None] * 20,
            'target': np.random.binomial(1, 0.3, 100)
        })
        
        woe = FastWoe()
        woe.fit(df_missing[['cat']], df_missing['target'])
        transformed = woe.transform(df_missing[['cat']])
        
        # Should handle missing values
        assert transformed.shape[0] == 100


class TestIntegration:
    """Integration tests combining WoePreprocessor and FastWoe."""

    def test_preprocessing_and_woe_pipeline(self):
        """Test full pipeline with preprocessing and WOE encoding."""
        # Create data with high cardinality
        np.random.seed(42)
        categories = [f'cat_{i}' for i in range(50)]  # 50 categories
        counts = np.random.poisson(lam=20, size=50) + 1  # Random counts
        
        data_rows = []
        for cat, count in zip(categories, counts):
            for _ in range(count):
                data_rows.append({
                    'high_card_feature': cat,
                    'target': np.random.binomial(1, 0.3)
                })
        
        df = pd.DataFrame(data_rows)
        X = df[['high_card_feature']]
        y = df['target']
        
        # Step 1: Preprocess to reduce cardinality
        preprocessor = WoePreprocessor(max_categories=10, top_p=None)  # Use max_categories only
        X_preprocessed = preprocessor.fit_transform(X)
        
        # Step 2: Apply WOE encoding
        woe_encoder = FastWoe()
        X_woe = woe_encoder.fit_transform(X_preprocessed, y)
        
        # Verify pipeline worked
        assert X_woe.shape[0] == len(df)
        assert X_woe.shape[1] == 1
        assert X_preprocessed['high_card_feature'].nunique() <= 11  # 10 + __other__
        
        # Check that WOE values are reasonable
        assert not np.isnan(X_woe['high_card_feature']).any()
        assert np.isfinite(X_woe['high_card_feature']).all() 