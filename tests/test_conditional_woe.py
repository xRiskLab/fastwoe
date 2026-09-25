"""Tests for conditional WOE (Good's chain rule), FastWoe(conditional=True)."""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from fastwoe import FastWoe

# Two correlated features over 10,000 applicants: E2 depends on E1.
# Cells are (bad, delinquent, high_util, count).
CELLS = [
    (1, 1, 1, 120),
    (1, 1, 0, 80),
    (1, 0, 1, 90),
    (1, 0, 0, 210),
    (0, 1, 1, 304),
    (0, 1, 0, 456),
    (0, 0, 1, 874),
    (0, 0, 0, 7866),
]
ONE = pd.DataFrame({"delinquent": ["1"], "high_util": ["1"]})


@pytest.fixture
def correlated():
    rows = [(b, d, u) for b, d, u, n in CELLS for _ in range(n)]
    df = pd.DataFrame(rows, columns=["bad", "delinquent", "high_util"])
    return df[["delinquent", "high_util"]].astype(str), df["bad"]


@pytest.fixture
def fitted(correlated):
    X, y = correlated
    return FastWoe(conditional=True).fit(X, y), X, y


def fit_quietly(X, y, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FastWoe(conditional=True, **kwargs).fit(X, y)


class TestConditionalWeights:
    def test_first_step_equals_marginal(self, fitted):
        """With nothing to condition on, step one is the ordinary WOE."""
        woe, _, _ = fitted
        table = woe.conditional_weights_[("delinquent", ())]
        assert table.loc["1", "woe"] == pytest.approx(woe.mappings_["delinquent"].loc["1", "woe"])

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


class TestPrediction:
    def test_transform_returns_conditional_weights(self, fitted):
        woe, _, _ = fitted
        parts = woe.transform(ONE).iloc[0]
        assert parts["delinquent"] == pytest.approx(np.log(5.0), abs=1e-9)
        assert parts["high_util"] == pytest.approx(np.log(1.5), abs=1e-9)

    def test_predict_proba_matches_the_hand_computation(self, fitted):
        woe, _, _ = fitted
        assert woe.predict_proba(ONE)[0, 1] == pytest.approx(0.2830, abs=1e-4)

    def test_predict_proba_is_the_observed_cell_rate(self, fitted):
        """With no fallback the chain reproduces each cell's event rate."""
        woe, X, y = fitted
        proba = woe.predict_proba(X)[:, 1]
        observed = y.groupby([X["delinquent"], X["high_util"]]).transform("mean")
        np.testing.assert_allclose(proba, observed, atol=1e-9)

    def test_marginal_model_double_counts(self, correlated, fitted):
        X, y = correlated
        woe, _, _ = fitted
        marginal = FastWoe().fit(X, y).predict_proba(ONE)[0, 1]
        assert marginal > woe.predict_proba(ONE)[0, 1] + 0.1

    def test_predict_uses_conditional_score(self, fitted):
        woe, X, _ = fitted
        expected = (woe.transform(X).sum(axis=1) > 0).astype(int).to_numpy()
        np.testing.assert_array_equal(woe.predict(X), expected)

    def test_default_order_is_column_order(self, correlated):
        X, y = correlated
        woe = FastWoe(conditional=True).fit(X[["high_util", "delinquent"]], y)
        assert woe.conditional_order_ == ["high_util", "delinquent"]

    def test_transform_keeps_x_column_order(self, correlated):
        X, y = correlated
        woe = FastWoe(conditional=True, conditional_order=["high_util", "delinquent"]).fit(X, y)
        assert list(woe.transform(X).columns) == list(X.columns)

    def test_order_changes_attribution_not_the_total(self, correlated):
        X, y = correlated
        totals, splits = [], []
        for order in (["delinquent", "high_util"], ["high_util", "delinquent"]):
            woe = FastWoe(conditional=True, conditional_order=order).fit(X, y)
            parts = woe.transform(ONE).iloc[0]
            totals.append(parts.sum())
            splits.append(parts["delinquent"])
        assert totals[0] == pytest.approx(totals[1], abs=1e-9)
        assert splits[0] != pytest.approx(splits[1], abs=1e-3)


class TestOutputsAndIntervals:
    def test_ci_outputs_bracket_the_weight(self, fitted):
        woe, X, _ = fitted
        w = woe.transform(X)
        assert (woe.transform(X, output="woe_lower_ci") < w).all().all()
        assert (woe.transform(X, output="woe_upper_ci") > w).all().all()

    def test_woe_norm_is_weight_over_se(self, fitted):
        woe, _, _ = fitted
        se = woe.conditional_weights_[("high_util", ("1",))].loc["1", "woe_se"]
        norm = woe.transform(ONE, output="woe_norm").iloc[0]
        assert norm["high_util"] == pytest.approx(np.log(1.5) / se)

    @pytest.mark.parametrize("output", ["wald", "piecewise"])
    def test_unsupported_outputs_raise(self, fitted, output):
        woe, X, _ = fitted
        with pytest.raises(ValueError, match="not supported with conditional=True"):
            woe.transform(X, output=output)

    def test_predict_ci_uses_the_joint_cell_se(self, fitted):
        """Chained weights telescope, so the total's SE is the full cell's SE."""
        woe, _, _ = fitted
        se = woe._calculate_woe_se(304, 120)  # good, bad in the (1, 1) cell
        score = np.log(woe.odds_prior_) + np.log(5.0) + np.log(1.5)
        lower, upper = woe.predict_ci(ONE)[0]
        assert lower == pytest.approx(expit(score - 1.959964 * se), abs=1e-6)
        assert upper == pytest.approx(expit(score + 1.959964 * se), abs=1e-6)

    def test_all_fallback_reduces_to_the_marginal_model(self, correlated):
        X, y = correlated
        conditional = fit_quietly(X, y, conditional_min_count=5000)
        marginal = FastWoe().fit(X, y)
        np.testing.assert_allclose(conditional.predict_proba(X), marginal.predict_proba(X))
        np.testing.assert_allclose(conditional.predict_ci(X), marginal.predict_ci(X))


class TestChainRule:
    def test_weights_sum_to_the_joint(self, fitted):
        """The decomposition is exact, not an approximation."""
        woe, X, y = fitted
        report = woe._check_chain_rule(X, y)
        assert report["exact"]
        assert report["paths_checked"] == 4
        assert report["max_abs_error"] < 1e-9


class TestSparsity:
    def test_thin_cells_fall_back_and_warn(self, correlated):
        X, y = correlated
        with pytest.warns(UserWarning, match="fell back to the marginal"):
            woe = FastWoe(conditional=True, conditional_min_count=5000).fit(X, y)
        assert woe.conditional_fallbacks_
        assert all(
            f["reason"] in ("thin parent", "thin cell", "empty class")
            for f in woe.conditional_fallbacks_
        )

    def test_fallback_uses_the_marginal_weight(self, correlated):
        X, y = correlated
        woe = fit_quietly(X, y, conditional_min_count=5000)
        table = woe.conditional_weights_[("high_util", ("1",))]
        assert table.loc["1", "fallback"]
        assert table.loc["1", "woe"] == pytest.approx(woe.mappings_["high_util"].loc["1", "woe"])

    def test_no_fallback_at_a_sane_threshold(self, fitted):
        woe, _, _ = fitted
        assert woe.conditional_fallbacks_ == []

    def test_exactness_is_not_claimed_vacuously(self, correlated):
        """With every path fallen back, nothing is verified - so not 'exact'."""
        X, y = correlated
        woe = fit_quietly(X, y, conditional_min_count=5000)
        report = woe._check_chain_rule(X, y)
        assert report["paths_checked"] == 0
        assert not report["exact"]


class TestApi:
    def test_off_by_default(self, correlated):
        X, y = correlated
        woe = FastWoe().fit(X, y)
        assert not hasattr(woe, "conditional_weights_")
        assert not hasattr(woe, "fit_conditional")

    def test_rejects_unknown_feature_in_order(self, correlated):
        X, y = correlated
        with pytest.raises(ValueError, match="unknown"):
            FastWoe(conditional=True, conditional_order=["delinquent", "nope"]).fit(X, y)

    def test_rejects_incomplete_order(self, correlated):
        X, y = correlated
        with pytest.raises(ValueError, match="every column"):
            FastWoe(conditional=True, conditional_order=["delinquent"]).fit(X, y)

    def test_rejects_repeated_feature(self, correlated):
        X, y = correlated
        with pytest.raises(ValueError, match="must not repeat"):
            FastWoe(conditional=True, conditional_order=["delinquent", "delinquent"]).fit(X, y)

    def test_rejects_bad_min_count(self):
        with pytest.raises(ValueError, match="conditional_min_count"):
            FastWoe(conditional=True, conditional_min_count=0)

    def test_rejects_multiclass_target(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.choice(["x", "y"], 600)})
        y = pd.Series(rng.integers(0, 3, 600))
        with pytest.raises(ValueError, match="binary"):
            FastWoe(conditional=True).fit(X, y)

    def test_transform_requires_fitted_features(self, fitted):
        woe, X, _ = fitted
        with pytest.raises(ValueError, match="missing from X"):
            woe.transform(X[["delinquent"]])

    def test_finetune_is_refused(self, fitted):
        woe, X, y = fitted
        with pytest.raises(NotImplementedError, match="conditional=True"):
            woe.finetune(X, y)

    def test_get_mapping_is_the_conditional_table(self, fitted):
        woe, _, _ = fitted
        table = woe.get_mapping("high_util")
        for col in ("given", "category", "woe", "woe_se", "fallback"):
            assert col in table.columns
        assert set(table["given"]) == {"delinquent=0", "delinquent=1"}
        first = woe.get_mapping("delinquent")
        assert set(first["given"]) == {"-"}
        assert set(woe.get_all_mappings()) == {"delinquent", "high_util"}


class TestUnseenConditional:
    """Conditional transform honours the same `unseen` policy as the marginal one."""

    NEW = pd.DataFrame({"delinquent": ["2"], "high_util": ["1"]})

    def test_warns_by_default(self, correlated):
        X, y = correlated
        woe = FastWoe(conditional=True).fit(X, y)
        with pytest.warns(UserWarning, match="not seen during fit"):
            out = woe.transform(self.NEW)
        assert out.loc[0, "delinquent"] == 0.0
        assert woe.unseen_counts_ == {"delinquent": {"2": 1}}

    def test_raise_mode(self, correlated):
        X, y = correlated
        woe = FastWoe(conditional=True, unseen="raise").fit(X, y)
        with pytest.raises(ValueError, match="not seen during fit"):
            woe.transform(self.NEW)

    def test_prior_is_silent(self, correlated):
        X, y = correlated
        woe = FastWoe(conditional=True, unseen="prior").fit(X, y)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            woe.transform(self.NEW)

    def test_counts_are_per_row(self, correlated):
        X, y = correlated
        woe = FastWoe(conditional=True, unseen="prior").fit(X, y)
        woe.transform(pd.concat([self.NEW] * 3, ignore_index=True))
        assert woe.unseen_counts_ == {"delinquent": {"2": 3}}

    def test_seen_rows_do_not_warn(self, fitted):
        woe, X, _ = fitted
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            woe.transform(X)
        assert woe.unseen_counts_ == {}


@pytest.fixture
def with_missing():
    """A categorical with a well-populated NaN level and a numeric feature."""
    rng = np.random.default_rng(7)
    n = 20_000
    a = rng.choice(["x", "y", None], n, p=[0.4, 0.4, 0.2]).astype(object)
    b = rng.normal(size=n)
    logit = b + np.where(pd.isna(a), 1.0, 0.0) + (a == "x") * -0.5
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    return pd.DataFrame({"a": a, "b": b}), y


class TestMissingAndNumeric:
    def test_nan_level_gets_its_own_conditional_weights(self, with_missing):
        X, y = with_missing
        woe = fit_quietly(X, y, conditional_min_count=10)
        nan_paths = [p for (col, p) in woe.conditional_weights_ if col == "b" and pd.isna(p[0])]
        assert len(nan_paths) == 1
        table = woe.conditional_weights_[("b", nan_paths[0])]
        assert table["bad_in_cell"].iloc[0] + table["good_in_cell"].iloc[0] == X["a"].isna().sum()
        assert not table["fallback"].all()

    def test_chain_rule_checks_nan_paths(self, with_missing):
        X, y = with_missing
        woe = fit_quietly(X, y, conditional_min_count=10)
        report = woe._check_chain_rule(X, y)
        assert report["exact"]
        # every a-level, NaN included, contributes checked paths
        n_bins = woe.binning_info_["b"]["n_bins"]
        assert report["paths_checked"] + report["paths_skipped"] == 3 * n_bins

    def test_nan_and_none_score_alike(self, with_missing):
        X, y = with_missing
        woe = fit_quietly(X, y, conditional_min_count=10)
        rows = pd.DataFrame({"a": [None, np.nan, float("nan")], "b": [0.1, 0.1, 0.1]})
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # a missing value was seen at fit
            out = woe.transform(rows)
        nan_weight = woe.conditional_weights_[("a", ())]
        nan_weight = nan_weight[nan_weight.index.isna()]["woe"].iloc[0]
        np.testing.assert_allclose(out["a"], nan_weight)
        assert nan_weight != pytest.approx(0.0, abs=0.1)
        assert out["b"].nunique() == 1

    def test_numeric_feature_is_conditioned_on_bins(self, with_missing):
        X, y = with_missing
        woe = fit_quietly(X, y, conditional_order=["b", "a"], conditional_min_count=10)
        first = woe.conditional_weights_[("b", ())]
        assert set(first.index) == set(woe.mappings_["b"].index)
        # first step is the ordinary marginal weight of each bin
        np.testing.assert_allclose(
            first["woe"], woe.mappings_["b"].loc[first.index, "woe"], atol=1e-6
        )
        assert woe._check_chain_rule(X, y)["exact"]

    def test_get_mapping_keeps_bin_order(self, with_missing):
        X, y = with_missing
        woe = fit_quietly(X, y, conditional_order=["b", "a"], conditional_min_count=10)
        marginal_order = list(FastWoe().fit(X, y).get_mapping("b")["category"])
        assert list(woe.get_mapping("b")["category"]) == marginal_order


class TestConditionalIV:
    @staticmethod
    def joint_iv(X, y):
        """IV of the features crossed into one: what the conditional IVs must sum to."""
        crossed = X.astype(str).agg("|".join, axis=1).to_frame("joint")
        return FastWoe().fit(crossed, y).feature_stats_["joint"]["iv"]

    def test_conditional_ivs_sum_to_the_joint_iv(self, fitted):
        woe, X, y = fitted
        total = woe.get_iv_analysis()["iv_conditional"].sum()
        assert total == pytest.approx(self.joint_iv(X, y), abs=1e-3)  # frame is rounded to 4 dp

    def test_first_feature_is_its_marginal_iv(self, fitted):
        woe, _, _ = fitted
        stats = woe.feature_stats_["delinquent"]
        # marginal WOE carries TargetEncoder's smooth=1e-5; conditional uses raw counts
        assert stats["iv_conditional"] == pytest.approx(stats["iv"], rel=1e-5)
        assert stats["iv_conditional_se"] == pytest.approx(stats["iv_se"], rel=1e-5)
        assert stats["conditioned_on"] == "-"

    def test_conditioning_shrinks_a_correlated_iv(self, fitted):
        woe, _, _ = fitted
        stats = woe.feature_stats_["high_util"]
        assert stats["iv_conditional"] < stats["iv"]
        assert stats["conditioned_on"] == "delinquent"

    def test_order_changes_the_split_not_the_total(self, correlated):
        X, y = correlated
        totals, first = [], []
        for order in (["delinquent", "high_util"], ["high_util", "delinquent"]):
            woe = FastWoe(conditional=True, conditional_order=order).fit(X, y)
            ivs = {c: woe.feature_stats_[c]["iv_conditional"] for c in order}
            totals.append(sum(ivs.values()))
            first.append(ivs["delinquent"])
        assert totals[0] == pytest.approx(totals[1], abs=1e-9)
        assert first[0] != pytest.approx(first[1], abs=1e-3)

    def test_standard_error_and_interval(self, fitted):
        woe, _, _ = fitted
        stats = woe.feature_stats_["high_util"]
        assert stats["iv_conditional_se"] > 0
        assert stats["iv_conditional_ci_lower"] <= stats["iv_conditional"]
        assert stats["iv_conditional"] <= stats["iv_conditional_ci_upper"]

    def test_reported_in_analysis_and_summary(self, fitted):
        woe, _, _ = fitted
        analysis = woe.get_iv_analysis()
        for col in (
            "iv",
            "iv_conditional",
            "iv_conditional_se",
            "iv_conditional_ci_lower",
            "iv_conditional_ci_upper",
            "iv_conditional_significance",
            "conditioned_on",
        ):
            assert col in analysis.columns
        assert "iv_conditional" in woe.get_iv_analysis("high_util").columns
        assert "iv_conditional" in woe.get_feature_summary().columns

    def test_marginal_model_has_no_conditional_columns(self, correlated):
        X, y = correlated
        woe = FastWoe().fit(X, y)
        assert not any("conditional" in c for c in woe.get_iv_analysis().columns)
        assert "iv_conditional" not in woe.get_feature_summary().columns

    def test_numeric_feature_and_nan_level(self, with_missing):
        X, y = with_missing
        woe = fit_quietly(X, y, conditional_min_count=10)
        assert woe._check_chain_rule(X, y)["exact"]
        binned = X.assign(b=woe._apply_binning_to_column(X, "b"))
        total = sum(woe.feature_stats_[c]["iv_conditional"] for c in ("a", "b"))
        # joint IV comes from a marginal (smooth=1e-5) fit, so not bit-exact
        assert total == pytest.approx(self.joint_iv(binned.fillna("NA"), y), rel=1e-5)


class TestExportText:
    @staticmethod
    def body(text):
        """Tree rows: everything after the blank line and the column header."""
        lines = text.splitlines()
        return lines[lines.index("") + 2 :]

    def test_header_names_target_and_order(self, correlated):
        X, y = correlated
        text = FastWoe(conditional=True).fit(X, y).export_text()
        assert text.splitlines()[0] == (
            "Conditional WOE tree  ·  target: bad  ·  event rate = share of rows with bad = 1"
        )
        assert "Conditioning order: delinquent → high_util" in text.splitlines()[1]
        assert "event rate" in text.splitlines()[3]
        assert "bad rate" not in text

    def test_unnamed_target(self, correlated):
        X, y = correlated
        text = FastWoe(conditional=True).fit(X, y.to_numpy()).export_text()
        assert "target: target" in text.splitlines()[0]

    def test_tree_rows_and_values(self, fitted):
        woe, X, y = fitted
        rows = self.body(woe.export_text())
        assert len(rows) == 1 + 2 + 4  # root, two delinquent branches, four leaves
        assert rows[0].startswith("root") and f"{len(X):,}" in rows[0]
        assert f"{y.mean():.1%}" in rows[0]
        assert rows[1].startswith("├── delinquent = 0")
        assert rows[4].startswith("└── delinquent = 1") and f"{np.log(5.0):+.3f}" in rows[4]
        assert rows[6].startswith("    └── high_util = 1") and f"{np.log(1.5):+.3f}" in rows[6]
        assert "[fallback]" not in woe.export_text()

    def test_interval_is_1_96_standard_errors(self, fitted):
        woe, _, _ = fitted
        se = woe.conditional_weights_[("high_util", ("1",))].loc["1", "woe_se"]
        lo, hi = np.log(1.5) - 1.959964 * se, np.log(1.5) + 1.959964 * se
        assert f"[{lo:+.3f}, {hi:+.3f}]" in self.body(woe.export_text())[6]

    def test_columns_line_up(self, fitted):
        woe, _, _ = fitted
        text = woe.export_text()
        rows = self.body(text)
        header = text.splitlines()[3]
        col = header.index("event rate") + len("event rate")
        assert all(r[col - 1] == "%" for r in rows)

    def test_bars_share_one_scale(self, fitted):
        """Bars are on a common axis: a larger weight sits further right."""
        woe, _, _ = fitted
        rows = self.body(woe.export_text())
        dots = {
            r.split(" = ")[0].split()[-1] + r.split(" = ")[1][:1]: r.index("●") for r in rows[1:]
        }
        assert dots["delinquent1"] > dots["delinquent0"]
        assert all("┊" in r or "┼" in r or "●" in r for r in rows[1:])

    def test_zero_crossing_marked_only_when_covered(self, with_missing):
        X, y = with_missing
        woe = fit_quietly(X, y, conditional_min_count=10)
        crossings = 0
        for row in self.body(woe.export_text()):
            if "[" not in row:
                continue
            lo, hi = (float(v) for v in row.split("[")[1].split("]")[0].split(", "))
            if not lo <= 0 <= hi:
                assert "┼" not in row
            crossings += "┼" in row
        assert crossings > 0

    def test_bar_width_zero_hides_bars(self, fitted):
        woe, _, _ = fitted
        text = woe.export_text(bar_width=0)
        assert "●" not in text and "┊" not in text

    def test_max_depth_truncates(self, fitted):
        woe, _, _ = fitted
        rows = self.body(woe.export_text(max_depth=1))
        assert len(rows) == 1 + 2 * 2
        assert rows[2] == "│   └── ... 1 more level(s), 2 leaves"
        assert woe.export_text(max_depth=5) == woe.export_text()

    def test_marks_fallbacks(self, correlated):
        X, y = correlated
        text = fit_quietly(X, y, conditional_min_count=5000).export_text()
        assert "[fallback]" in text and "○" in text

    def test_bins_in_order_and_nan_label(self, with_missing):
        X, y = with_missing
        woe = fit_quietly(X, y, conditional_order=["b", "a"], conditional_min_count=10)
        rows = self.body(woe.export_text(max_depth=1))
        bins = [r.split("b = ")[1].split("  ")[0] for r in rows if "b = " in r]
        assert bins == woe._category_order("b")
        assert "a = NaN" in fit_quietly(X, y, conditional_min_count=10).export_text()

    def test_requires_conditional_and_fit(self, correlated):
        X, y = correlated
        with pytest.raises(ValueError, match="conditional=True"):
            FastWoe().fit(X, y).export_text()
        with pytest.raises(ValueError, match="fitted"):
            FastWoe(conditional=True).export_text()
        with pytest.raises(ValueError, match="max_depth"):
            FastWoe(conditional=True).fit(X, y).export_text(max_depth=0)


class TestThinCells:
    def test_a_thin_cell_under_a_large_parent_falls_back(self, correlated):
        """(delinquent=1, high_util=0) has 80 bads: enough at 30, too few at 100,
        although its parent (delinquent=1, 200 bads) clears 100."""
        X, y = correlated
        woe = fit_quietly(X, y, conditional_min_count=100)
        table = woe.conditional_weights_[("high_util", ("1",))]
        assert table.loc["0", "fallback"]
        assert table.loc["1", "fallback"] == False  # noqa: E712 - 120 bads, 304 goods
        reasons = {(f["feature"], f["category"]): f["reason"] for f in woe.conditional_fallbacks_}
        assert reasons[("high_util", "0")] == "thin cell"

    def test_fallback_is_inherited_by_deeper_cells(self, with_missing):
        """A thin cell is a thin parent for everything below it."""
        X, y = with_missing
        woe = fit_quietly(X, y, conditional_min_count=400)
        for (_col, path), table in woe.conditional_weights_.items():
            if not path:
                continue
            parent = woe.conditional_weights_[(woe.conditional_order_[len(path) - 1], path[:-1])]
            parent_row = (
                parent[parent.index.isna()] if pd.isna(path[-1]) else parent.loc[[path[-1]]]
            )
            if parent_row["fallback"].iloc[0]:
                assert table["fallback"].all()
