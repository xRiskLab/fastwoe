"""Tests for modeling.py module."""

import numpy as np
import pandas as pd
import pytest

from fastwoe import FastWoe
from fastwoe.modeling import marginal_somersd_selection, somersd_shapley


class TestMarginalSomersdSelection:
    """Test cases for marginal_somersd_selection function."""

    @pytest.fixture
    def binary_data(self):
        """Create sample binary classification data."""
        np.random.seed(42)
        n_samples = 200
        X = pd.DataFrame(
            {
                "feature1": np.random.choice(["A", "B", "C"], n_samples),
                "feature2": np.random.choice(["X", "Y"], n_samples),
                "feature3": np.random.choice(["P", "Q", "R"], n_samples),
                "feature4": np.random.choice(["M", "N"], n_samples),
            }
        )
        # Create target with correlation to feature1
        y = (X["feature1"] == "A").astype(int)
        y = y + np.random.binomial(1, 0.1, n_samples)  # Add noise
        y = np.clip(y, 0, 1)
        return X, y

    @pytest.fixture
    def continuous_data(self):
        """Create sample continuous target data."""
        np.random.seed(42)
        n_samples = 200
        X = pd.DataFrame(
            {
                "feature1": np.random.choice(["A", "B", "C"], n_samples),
                "feature2": np.random.choice(["X", "Y"], n_samples),
                "feature3": np.random.choice(["P", "Q"], n_samples),
            }
        )
        # Create continuous target with correlation to feature1
        y = (X["feature1"] == "A").astype(float) * 10.0 + np.random.normal(0, 1, n_samples)
        return X, y

    def test_basic_binary_selection(self, binary_data):
        """Test basic feature selection with binary target."""
        X, y = binary_data
        result = marginal_somersd_selection(X, y, min_msd=0.01, max_features=3)

        assert isinstance(result, dict)
        assert "selected_features" in result
        assert "msd_history" in result
        assert "univariate_somersd" in result
        assert "model" in result
        assert "test_performance" in result
        assert "correlation_matrix" in result

        assert len(result["selected_features"]) > 0
        assert len(result["selected_features"]) <= 3
        assert len(result["msd_history"]) == len(result["selected_features"])
        assert all(f in X.columns for f in result["selected_features"])

    def test_basic_continuous_selection(self, continuous_data):
        """Test basic feature selection with continuous target."""
        X, y = continuous_data
        result = marginal_somersd_selection(X, y, min_msd=0.01, max_features=2)

        assert isinstance(result, dict)
        assert len(result["selected_features"]) > 0
        assert len(result["selected_features"]) <= 2
        assert isinstance(result["model"], FastWoe)
        assert result["model"].is_fitted_

    def test_with_test_set(self, binary_data):
        """Test feature selection with separate test set."""
        X, y = binary_data
        X_train, X_test = X.iloc[:150], X.iloc[150:]
        y_train, y_test = y[:150], y[150:]

        result = marginal_somersd_selection(
            X_train, y_train, X_test=X_test, y_test=y_test, max_features=2
        )

        # test_performance is computed at the start of each iteration (after first feature)
        # So it has length = len(selected_features) - 1
        assert len(result["test_performance"]) == len(result["selected_features"]) - 1
        assert len(result["test_performance"]) > 0
        assert all(isinstance(p, (int, float)) for p in result["test_performance"])

    def test_max_features_limit(self, binary_data):
        """Test that max_features limit is respected."""
        X, y = binary_data
        result = marginal_somersd_selection(X, y, max_features=2)

        assert len(result["selected_features"]) <= 2

    def test_min_msd_threshold(self, binary_data):
        """Test that selection stops when MSD falls below threshold."""
        X, y = binary_data
        # Use a high threshold to force early stopping
        result = marginal_somersd_selection(X, y, min_msd=0.5, max_features=10)

        # Should stop early if MSD drops below threshold
        assert len(result["selected_features"]) >= 1

    def test_correlation_threshold(self, binary_data):
        """Test that highly correlated features are filtered out."""
        X, y = binary_data
        # Add a highly correlated duplicate feature
        X["feature1_dup"] = X["feature1"]

        result = marginal_somersd_selection(X, y, correlation_threshold=0.9, max_features=5)

        # Should not select both feature1 and feature1_dup if correlation is high
        selected = result["selected_features"]
        if "feature1" in selected and "feature1_dup" in selected:
            # If both selected, check correlation is below threshold
            corr_matrix = result["correlation_matrix"]
            if corr_matrix is not None:
                corr = corr_matrix.loc["feature1", "feature1_dup"]
                assert abs(corr) < 0.9 or np.isnan(corr)

    def test_woe_model_template(self, binary_data):
        """Test that woe_model template is used correctly."""
        X, y = binary_data
        template = FastWoe(random_state=123)

        result = marginal_somersd_selection(X, y, woe_model=template, max_features=2)

        assert result["model"].is_fitted_
        assert len(result["selected_features"]) > 0

    def test_verbose_mode(self, binary_data):
        """Test that verbose mode doesn't crash."""
        X, y = binary_data
        # Should not raise any errors
        result = marginal_somersd_selection(X, y, verbose=True, max_features=2)

        assert len(result["selected_features"]) > 0

    def test_random_state(self, binary_data):
        """Test that random_state provides reproducibility."""
        X, y = binary_data

        result1 = marginal_somersd_selection(X, y, random_state=42, max_features=3)
        result2 = marginal_somersd_selection(X, y, random_state=42, max_features=3)

        assert result1["selected_features"] == result2["selected_features"]

    def test_ties_parameter(self, binary_data):
        """Test that ties parameter works correctly."""
        X, y = binary_data

        result_y = marginal_somersd_selection(X, y, ties="y", max_features=2)
        result_x = marginal_somersd_selection(X, y, ties="x", max_features=2)

        assert len(result_y["selected_features"]) > 0
        assert len(result_x["selected_features"]) > 0

    def test_univariate_somersd_output(self, binary_data):
        """Test that univariate Somers' D is computed for all features."""
        X, y = binary_data
        result = marginal_somersd_selection(X, y, max_features=2)

        assert len(result["univariate_somersd"]) == len(X.columns)
        assert all(f in result["univariate_somersd"] for f in X.columns)
        assert all(isinstance(v, (int, float)) for v in result["univariate_somersd"].values())

    def test_empty_features_error(self):
        """Test that empty feature set raises appropriate error."""
        X = pd.DataFrame()
        y = np.array([0, 1, 0, 1])

        with pytest.raises((ValueError, IndexError, KeyError)):
            marginal_somersd_selection(X, y)

    def test_single_feature(self, binary_data):
        """Test selection with single feature."""
        X, y = binary_data
        X_single = X[["feature1"]]

        result = marginal_somersd_selection(X_single, y, max_features=1)

        assert len(result["selected_features"]) == 1
        assert result["selected_features"][0] == "feature1"


class TestSomersdShapley:
    """Test cases for somersd_shapley function."""

    @pytest.fixture
    def binary_scores(self):
        """Create sample binary target with multiple scores."""
        np.random.seed(42)
        n_samples = 100
        y = np.random.binomial(1, 0.3, n_samples)

        score_dict = {
            "score1": np.random.uniform(0, 1, n_samples),
            "score2": np.random.uniform(0, 1, n_samples),
            "score3": np.random.uniform(0, 1, n_samples),
        }

        # Make scores somewhat predictive
        score_dict["score1"] = y * 0.7 + np.random.uniform(0, 0.3, n_samples)
        score_dict["score2"] = y * 0.5 + np.random.uniform(0, 0.5, n_samples)

        return score_dict, y

    @pytest.fixture
    def continuous_scores(self):
        """Create sample continuous target with multiple scores."""
        np.random.seed(42)
        n_samples = 100
        y = np.random.normal(0, 1, n_samples)

        score_dict = {
            "score1": y + np.random.normal(0, 0.5, n_samples),
            "score2": y * 0.8 + np.random.normal(0, 0.6, n_samples),
            "score3": np.random.normal(0, 1, n_samples),  # Less predictive
        }

        return score_dict, y

    def test_basic_binary_shapley(self, binary_scores):
        """Test basic Shapley computation with binary target."""
        score_dict, y = binary_scores
        result = somersd_shapley(score_dict, y)

        assert isinstance(result, pd.DataFrame)
        assert "score_name" in result.columns
        assert "shapley_value" in result.columns
        assert "total_somersd" in result.columns
        assert "n_scores" in result.columns
        assert "n_subsets" in result.columns

        assert len(result) == len(score_dict)
        assert set(result["score_name"]) == set(score_dict.keys())
        assert result["n_scores"].iloc[0] == len(score_dict)
        assert result["n_subsets"].iloc[0] == 2 ** len(score_dict)

    def test_basic_continuous_shapley(self, continuous_scores):
        """Test basic Shapley computation with continuous target."""
        score_dict, y = continuous_scores
        result = somersd_shapley(score_dict, y)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(score_dict)
        assert all(isinstance(v, (int, float)) for v in result["shapley_value"])

    def test_shapley_with_base_score(self, binary_scores):
        """Test Shapley computation with base score."""
        score_dict, y = binary_scores
        result = somersd_shapley(score_dict, y, base_score_name="score1")

        assert isinstance(result, pd.DataFrame)
        assert "component" in result.columns
        assert "effect_on_somersd" in result.columns
        assert "role" in result.columns
        assert "base_only_somersd" in result.columns
        assert "final_system_somersd" in result.columns

        # Base score should be in results
        base_row = result[result["component"] == "score1"]
        assert len(base_row) == 1
        assert base_row["role"].iloc[0] == "base_score_only"

        # Other scores should have increment_over_base role
        other_rows = result[result["component"] != "score1"]
        assert all(other_rows["role"] == "increment_over_base")

    def test_shapley_with_availability_mask(self, binary_scores):
        """Test Shapley computation with availability masks."""
        score_dict, y = binary_scores
        n_samples = len(y)

        # Create availability mask where score3 is missing for some samples
        availability_mask = {
            "score1": np.ones(n_samples, dtype=bool),
            "score2": np.ones(n_samples, dtype=bool),
            "score3": np.random.binomial(1, 0.7, n_samples).astype(bool),
        }

        result = somersd_shapley(score_dict, y, availability_mask=availability_mask)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(score_dict)

    def test_shapley_with_base_and_availability(self, binary_scores):
        """Test Shapley with both base score and availability masks."""
        score_dict, y = binary_scores
        n_samples = len(y)

        availability_mask = {
            "score1": np.ones(n_samples, dtype=bool),
            "score2": np.random.binomial(1, 0.8, n_samples).astype(bool),
            "score3": np.random.binomial(1, 0.7, n_samples).astype(bool),
        }

        result = somersd_shapley(
            score_dict, y, base_score_name="score1", availability_mask=availability_mask
        )

        assert isinstance(result, pd.DataFrame)
        assert "base_only_somersd" in result.columns

    def test_shapley_values_sum(self, binary_scores):
        """Test that Shapley values sum to total (without base score)."""
        score_dict, y = binary_scores
        result = somersd_shapley(score_dict, y)

        total_shapley = result["shapley_value"].sum()
        total_somersd = result["total_somersd"].iloc[0]

        # Shapley values should sum to total Somers' D (within numerical precision)
        assert abs(total_shapley - total_somersd) < 1e-6

    def test_shapley_ties_parameter(self, binary_scores):
        """Test that ties parameter works correctly."""
        score_dict, y = binary_scores

        result_y = somersd_shapley(score_dict, y, ties="y")
        result_x = somersd_shapley(score_dict, y, ties="x")

        assert len(result_y) == len(result_x)
        assert len(result_y) == len(score_dict)

    def test_shapley_single_score(self, binary_scores):
        """Test Shapley computation with single score."""
        score_dict, y = binary_scores
        single_score = {"score1": score_dict["score1"]}

        result = somersd_shapley(single_score, y)

        assert len(result) == 1
        assert result["score_name"].iloc[0] == "score1"
        assert result["n_scores"].iloc[0] == 1
        assert result["n_subsets"].iloc[0] == 2

    def test_shapley_two_scores(self, binary_scores):
        """Test Shapley computation with two scores."""
        score_dict, y = binary_scores
        two_scores = {"score1": score_dict["score1"], "score2": score_dict["score2"]}

        result = somersd_shapley(two_scores, y)

        assert len(result) == 2
        assert result["n_subsets"].iloc[0] == 4  # 2^2 = 4

    def test_shapley_invalid_base_score(self, binary_scores):
        """Test that invalid base_score_name raises error."""
        score_dict, y = binary_scores

        with pytest.raises(ValueError, match="not found in score_dict"):
            somersd_shapley(score_dict, y, base_score_name="nonexistent")

    def test_shapley_empty_availability(self, binary_scores):
        """Test that empty availability mask raises error."""
        score_dict, y = binary_scores
        n_samples = len(y)

        # All scores unavailable
        availability_mask = {
            "score1": np.zeros(n_samples, dtype=bool),
            "score2": np.zeros(n_samples, dtype=bool),
            "score3": np.zeros(n_samples, dtype=bool),
        }

        with pytest.raises(ValueError, match="No samples available"):
            somersd_shapley(score_dict, y, availability_mask=availability_mask)

    def test_shapley_no_variation_error(self):
        """Test that target with no variation raises error."""
        score_dict = {
            "score1": np.array([0.5, 0.6, 0.7]),
            "score2": np.array([0.3, 0.4, 0.5]),
        }
        y = np.array([1, 1, 1])  # No variation

        with pytest.raises(ValueError, match="no variation"):
            somersd_shapley(score_dict, y)

    def test_shapley_sorted_by_value(self, binary_scores):
        """Test that results are sorted by Shapley value."""
        score_dict, y = binary_scores
        result = somersd_shapley(score_dict, y)

        shapley_values = result["shapley_value"].values
        assert all(
            shapley_values[i] >= shapley_values[i + 1] for i in range(len(shapley_values) - 1)
        )
