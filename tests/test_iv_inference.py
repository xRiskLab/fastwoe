"""Information Value inference: delta-method standard errors and the chi-square test.

Each claim is checked against a simulation: many draws of bads and goods as two
independent multinomial samples, which is how the IV is estimated.
"""

import numpy as np
import pandas as pd
import pytest

from fastwoe import FastWoe
from fastwoe.metrics import _iv_chi2_test, _iv_standard_error

STRONG_B = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
STRONG_G = STRONG_B[::-1]


def iv(bad, good, parent=None):
    """Plug-in IV, or IV(cells) - IV(parents) with a parent id per cell."""

    def j(bc, gc):
        b, g = bc / bc.sum(), gc / gc.sum()
        m = (b > 0) & (g > 0)
        return float(((b - g)[m] * np.log(b[m] / g[m])).sum())

    if parent is None:
        return j(bad, good)
    return j(bad, good) - j(np.bincount(parent, bad), np.bincount(parent, good))


def draw(rng, b, g, n_bad, n_good):
    return rng.multinomial(n_bad, b).astype(float), rng.multinomial(n_good, g).astype(float)


class TestStandardError:
    @pytest.mark.parametrize("n_bad, n_good", [(2000, 18000), (300, 2700)])
    def test_matches_simulated_spread(self, n_bad, n_good):
        rng = np.random.default_rng(0)
        spread = np.std([iv(*draw(rng, STRONG_B, STRONG_G, n_bad, n_good)) for _ in range(4000)])
        se = np.median(
            [_iv_standard_error(*draw(rng, STRONG_B, STRONG_G, n_bad, n_good)) for _ in range(200)]
        )
        assert se == pytest.approx(spread, rel=0.1)

    def test_fixed_weight_formula_understates(self):
        """sqrt(sum (b-g)^2 (1/bad + 1/good)) holds the weights fixed: about half."""
        rng = np.random.default_rng(1)
        bad, good = draw(rng, STRONG_B, STRONG_G, 2000, 18000)
        b, g = bad / bad.sum(), good / good.sum()
        fixed_weights = np.sqrt(((b - g) ** 2 * (1 / bad + 1 / good)).sum())
        assert _iv_standard_error(bad, good) > 1.6 * fixed_weights

    def test_conditional_matches_simulated_spread(self):
        rng = np.random.default_rng(2)
        parent = np.array([0, 0, 0, 1, 1, 1])
        b = np.array([0.10, 0.20, 0.30, 0.05, 0.15, 0.20])
        g = np.array([0.35, 0.25, 0.15, 0.15, 0.07, 0.03])
        spread = np.std([iv(*draw(rng, b, g, 1500, 13500), parent) for _ in range(4000)])
        se = np.median(
            [_iv_standard_error(*draw(rng, b, g, 1500, 13500), parent) for _ in range(200)]
        )
        assert se == pytest.approx(spread, rel=0.1)

    def test_one_parent_is_the_marginal_se(self):
        rng = np.random.default_rng(3)
        bad, good = draw(rng, STRONG_B, STRONG_G, 500, 4500)
        assert _iv_standard_error(bad, good, np.zeros(5, dtype=int)) == pytest.approx(
            _iv_standard_error(bad, good)
        )

    def test_undefined_without_both_classes(self):
        assert np.isnan(_iv_standard_error(np.array([10.0, 0.0]), np.array([0.0, 10.0])))


class TestChiSquareTest:
    @pytest.mark.parametrize("k, n_bad, n_good", [(5, 1000, 9000), (10, 200, 3800)])
    def test_calibrated_under_h0(self, k, n_bad, n_good):
        rng = np.random.default_rng(k)
        p = rng.dirichlet(np.ones(k) * 3)
        pvalues = np.array([_iv_chi2_test(*draw(rng, p, p, n_bad, n_good))[2] for _ in range(3000)])
        assert 0.03 < np.mean(pvalues < 0.05) < 0.075

    def test_bias_under_h0(self):
        """E[IV | H0] is about (k - 1) / n_eff: pure noise still has positive IV."""
        rng = np.random.default_rng(5)
        k, n_bad, n_good = 5, 1000, 9000
        p = np.full(k, 1 / k)
        mean_iv = np.mean([iv(*draw(rng, p, p, n_bad, n_good)) for _ in range(4000)])
        n_eff = n_bad * n_good / (n_bad + n_good)
        assert mean_iv == pytest.approx((k - 1) / n_eff, rel=0.1)

    def test_stratified_calibrated_under_h0(self):
        """H0: the second feature adds nothing within the first feature's groups."""
        rng = np.random.default_rng(6)
        parent = np.array([0, 0, 0, 1, 1, 1])
        b = np.array([0.2, 0.2, 0.2, 0.1, 0.15, 0.15])
        g = np.empty(6)
        for group, share in ((0, 0.3), (1, 0.7)):
            m = parent == group
            g[m] = b[m] / b[m].sum() * share
        pvalues = np.array(
            [_iv_chi2_test(*draw(rng, b, g, 1500, 13500), parent)[2] for _ in range(3000)]
        )
        assert 0.03 < np.mean(pvalues < 0.05) < 0.075

    def test_perfect_separation_rejects(self):
        _, dof, pvalue = _iv_chi2_test(np.array([100.0, 0.0]), np.array([0.0, 100.0]))
        assert dof == 1 and pvalue < 1e-10

    def test_nothing_to_test(self):
        assert np.isnan(_iv_chi2_test(np.array([5.0]), np.array([7.0]))[2])


class TestFastWoeReporting:
    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(7)
        n = 5000
        strong = rng.choice(list("abcd"), n)
        noise = rng.choice(list("wxyz"), n)
        logit = -1.5 + np.select([strong == "a", strong == "b"], [1.0, 0.5], 0.0)
        y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
        return pd.DataFrame({"strong": strong, "noise": noise}), y

    def test_pvalue_and_significance(self, data):
        X, y = data
        analysis = FastWoe().fit(X, y).get_iv_analysis().set_index("feature")
        assert analysis.loc["strong", "iv_pvalue"] < 1e-6
        assert analysis.loc["strong", "iv_significance"] == "Significant"
        assert analysis.loc["noise", "iv_pvalue"] > 0.01
        assert analysis.loc["noise", "iv_significance"] == "Not Significant"
        assert analysis.loc["noise", "iv"] > 0  # plug-in IV of pure noise is still positive

    def test_alpha_sets_the_label(self, data):
        X, y = data
        woe = FastWoe().fit(X, y)
        p = woe.get_iv_analysis("noise")["iv_pvalue"].iloc[0]
        loose = woe.get_iv_analysis("noise", alpha=min(1.0, p * 1.01))
        assert loose["iv_significance"].iloc[0] == "Significant"

    def test_se_uses_the_delta_method(self, data):
        X, y = data
        woe = FastWoe().fit(X, y)
        mapping = woe.mappings_["strong"]
        bad, good = mapping["bad_count"].to_numpy(), mapping["good_count"].to_numpy()
        assert woe.feature_stats_["strong"]["iv_se"] == pytest.approx(_iv_standard_error(bad, good))

    def test_conditional_iv_inference(self, data):
        X, y = data
        woe = FastWoe(conditional=True).fit(X, y)
        analysis = woe.get_iv_analysis().set_index("feature")
        # first feature: nothing to condition on, so the marginal test and SE
        assert analysis.loc["strong", "iv_conditional_pvalue"] == pytest.approx(
            analysis.loc["strong", "iv_pvalue"]
        )
        assert woe.feature_stats_["strong"]["iv_conditional_se"] == pytest.approx(
            woe.feature_stats_["strong"]["iv_se"], rel=1e-5
        )
        # noise given strong: tested within strong's groups, 4 x (4 - 1) = 12 dof
        assert analysis.loc["noise", "iv_conditional_significance"] == "Not Significant"
        assert "iv_conditional_pvalue" in analysis.columns

    def test_single_class_bins_do_not_inflate_the_se(self):
        """A tiny edge bin with no goods is skipped, not read as 0.00003 goods."""
        rng = np.random.default_rng(0)
        n = 20_000
        income = rng.normal(size=n)
        y = pd.Series((rng.random(n) < 1 / (1 + np.exp(2.0 + 0.3 * income))).astype(int))
        woe = FastWoe().fit(pd.DataFrame({"income": income}), y)
        mapping = woe.mappings_["income"]
        assert ((mapping["good_count"] == 0) | (mapping["bad_count"] == 0)).any()
        stats = woe.feature_stats_["income"]
        assert stats["iv_se"] < stats["iv"]
