"""Tests for conditional WOE (Good's chain rule)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from fastwoe import FastWoe

# Two correlated features over 10,000 applicants: E2 depends on E1.
# Cells are (bad, delinquent, high_util, count).
CELLS = [
    (1, 1, 1, 120), (1, 1, 0, 80), (1, 0, 1, 90), (1, 0, 0, 210),
    (0, 1, 1, 304), (0, 1, 0, 456), (0, 0, 1, 874), (0, 0, 0, 7866),
]


@pytest.fixture
def correlated():
    rows = [(b, d, u) for b, d, u, n in CELLS for _ in range(n)]
    df = pd.DataFrame(rows, columns=["bad", "delinquent", "high_util"])
    return df[["delinquent", "high_util"]].astype(str), df["bad"]


@pytest.fixture
def fitted(correlated):
    X, y = correlated
    woe = FastWoe().fit(X, y)
    woe.fit_conditional(X, y, order=["delinquent", "high_util"])
    return woe, X, y


class TestConditionalWeights:
    def test_first_step_equals_marginal(self, fitted):
        """With nothing to condition on, step one is the ordinary WOE."""
        woe, _, _ = fitted
        table = woe.conditional_weights_[("delinquent", ())]
        assert table.loc["1", "woe"] == pytest.approx(
            woe.mappings_["delinquent"].loc["1", "woe"]
        )

    def test_conditioning_shrinks_a_correlated_weight(self, fitted):
        """high_util is worth less once delinquency is known."""
        woe, _, _ = fitted
        marginal = woe.mappings_["high_util"].loc["1", "woe"]
        conditional = woe.conditional_weights_[("high_util", ("1",))].loc["1", "woe"]
        assert marginal == pytest.approx(np.log(3.387096), abs=1e-4)
        assert conditional == pytest.approx(np.log(1.5), abs=1e-9)
        assert conditional < marginal

    def test_weight_differs_by_branch(self, fitted):
        """The conditional weight depends on which cell you are in."""
        woe, _, _ = fitted
        given_yes = woe.conditional_weights_[("high_util", ("1",))].loc["1", "woe"]
        given_no = woe.conditional_weights_[("high_util", ("0",))].loc["1", "woe"]
        assert given_yes != pytest.approx(given_no)

    def test_standard_errors_are_reported(self, fitted):
        woe, _, _ = fitted
        table = woe.conditional_weights_[("high_util", ("1",))]
        assert table.loc["1", "woe_se"] > 0
        assert np.isfinite(table.loc["1", "woe_se"])


class TestChainRule:
    def test_weights_sum_to_the_joint(self, fitted):
        """The decomposition is exact, not an approximation."""
        woe, X, y = fitted
        report = woe.check_chain_rule(X, y)
        assert report["exact"]
        assert report["paths_checked"] == 4
        assert report["max_abs_error"] < 1e-9

    def test_score_matches_the_hand_computation(self, fitted):
        woe, _, _ = fitted
        one = pd.DataFrame({"delinquent": ["1"], "high_util": ["1"]})
        parts = woe.transform_conditional(one).iloc[0]
        assert parts["delinquent"] == pytest.approx(np.log(5.0), abs=1e-9)
        assert parts["high_util"] == pytest.approx(np.log(1.5), abs=1e-9)
        log_odds = woe.predict_conditional_log_odds(one)[0]
        assert log_odds == pytest.approx(-0.929536, abs=1e-5)
        assert 1 / (1 + np.exp(-log_odds)) == pytest.approx(0.2830, abs=1e-4)

    def test_order_changes_attribution_not_the_total(self, correlated):
        X, y = correlated
        one = pd.DataFrame({"delinquent": ["1"], "high_util": ["1"]})
        totals, splits = [], []
        for order in (["delinquent", "high_util"], ["high_util", "delinquent"]):
            woe = FastWoe().fit(X, y)
            woe.fit_conditional(X, y, order=order)
            parts = woe.transform_conditional(one).iloc[0]
            totals.append(parts.sum())
            splits.append(parts["delinquent"])
        assert totals[0] == pytest.approx(totals[1], abs=1e-9)
        assert splits[0] != pytest.approx(splits[1], abs=1e-3)


class TestSparsity:
    def test_thin_cells_fall_back_and_warn(self, correlated):
        X, y = correlated
        woe = FastWoe().fit(X, y)
        with pytest.warns(UserWarning, match="fell back to the marginal"):
            woe.fit_conditional(X, y, order=["delinquent", "high_util"],
                                min_cell_count=5000)
        assert woe.conditional_fallbacks_
        assert all(
            f["reason"] in ("thin cell", "empty class")
            for f in woe.conditional_fallbacks_
        )

    def test_fallback_uses_the_marginal_weight(self, correlated):
        X, y = correlated
        woe = FastWoe().fit(X, y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            woe.fit_conditional(X, y, order=["delinquent", "high_util"],
                                min_cell_count=5000)
        table = woe.conditional_weights_[("high_util", ("1",))]
        assert table.loc["1", "fallback"]
        assert table.loc["1", "woe"] == pytest.approx(
            woe.mappings_["high_util"].loc["1", "woe"]
        )

    def test_no_fallback_at_a_sane_threshold(self, fitted):
        woe, _, _ = fitted
        assert woe.conditional_fallbacks_ == []

    def test_exactness_is_not_claimed_vacuously(self, correlated):
        """With every path fallen back, nothing is verified - so not 'exact'."""
        X, y = correlated
        woe = FastWoe().fit(X, y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            woe.fit_conditional(X, y, order=["delinquent", "high_util"],
                                min_cell_count=5000)
        report = woe.check_chain_rule(X, y)
        assert report["paths_checked"] == 0
        assert not report["exact"]


class TestApi:
    def test_requires_fit_first(self, correlated):
        X, y = correlated
        with pytest.raises(ValueError, match="call fit"):
            FastWoe().fit_conditional(X, y, order=["delinquent"])

    def test_rejects_unknown_feature(self, fitted):
        woe, X, y = fitted
        with pytest.raises(ValueError, match="not in X"):
            woe.fit_conditional(X, y, order=["nope"])

    def test_rejects_repeated_feature(self, fitted):
        woe, X, y = fitted
        with pytest.raises(ValueError, match="must not repeat"):
            woe.fit_conditional(X, y, order=["delinquent", "delinquent"])

    def test_transform_requires_fit_conditional(self, correlated):
        X, y = correlated
        woe = FastWoe().fit(X, y)
        with pytest.raises(ValueError, match="fit_conditional"):
            woe.transform_conditional(X)

    def test_summary_is_tidy(self, fitted):
        woe, _, _ = fitted
        summary = woe.conditional_summary()
        for col in ("feature", "given", "category", "woe", "woe_se", "fallback"):
            assert col in summary.columns
        assert set(summary["feature"]) == {"delinquent", "high_util"}
