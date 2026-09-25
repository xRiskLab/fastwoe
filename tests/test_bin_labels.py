"""Bin labels are the categories WOE is computed on: two bins must never share one."""

import importlib.util
import warnings

import numpy as np
import pandas as pd
import pytest

from fastwoe import FastWoe
from fastwoe.fastwoe import _bin_labels


class TestBinLabels:
    def test_ordinary_edges_keep_one_decimal(self):
        edges = [-np.inf, -1.23, 0.5, 2.71, np.inf]
        assert _bin_labels(edges) == ["(-∞, -1.2]", "(-1.2, 0.5]", "(0.5, 2.7]", "(2.7, ∞)"]

    def test_close_edges_get_more_decimals(self):
        edges = [-np.inf, 0.061, 0.068, 0.099, np.inf]  # one decimal: (0.1, 0.1] twice
        labels = _bin_labels(edges)
        assert len(set(labels)) == len(labels)
        assert labels == ["(-∞, 0.06]", "(0.06, 0.07]", "(0.07, 0.10]", "(0.10, ∞)"]

    def test_fewest_decimals_that_separate_the_bins(self):
        """Decimals stop at the first precision where the labels differ."""
        edges = [-np.inf, 0.1231, 0.1234, 0.1239, np.inf]
        assert _bin_labels(edges) == [
            "(-∞, 0.123]",
            "(0.123, 0.123]",
            "(0.123, 0.124]",
            "(0.124, ∞)",
        ]

    def test_repeated_edges_stay_distinct(self):
        labels = _bin_labels([-np.inf, 1.0, 1.0, 1.0, np.inf])
        assert len(set(labels)) == len(labels)

    def test_single_bin(self):
        assert _bin_labels([-np.inf, np.inf]) == ["(-∞, inf]"]


def small_decimal_data(n=20_000):
    """A feature in small decimals whose tree and quantile splits fall close together."""
    rng = np.random.default_rng(0)
    x = rng.normal(scale=0.05, size=n)
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(2 - 20 * x))).astype(int))
    return pd.DataFrame({"x": x}), y


METHODS = [("tree", {}), ("kbins", {"binner_kwargs": {"n_bins": 8}})]
if importlib.util.find_spec("faiss") is not None:
    METHODS.append(("faiss_kmeans", {"faiss_kwargs": {"k": 8}}))


@pytest.mark.parametrize("method, kwargs", METHODS, ids=[m for m, _ in METHODS])
class TestNoBinsMerged:
    def fit(self, method, kwargs):
        X, y = small_decimal_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return FastWoe(binning_method=method, **kwargs).fit(X, y), X, y

    def test_every_bin_is_its_own_category(self, method, kwargs):
        woe, X, _ = self.fit(method, kwargs)
        n_bins = len(woe.binning_info_["x"]["bin_edges"]) - 1
        mapping = woe.mappings_["x"]
        assert len(mapping) == n_bins
        assert mapping["count"].sum() == len(X)
        rows = woe.get_mapping("x")["category"]
        assert rows.is_unique and len(rows) == n_bins

    def test_fit_and_transform_agree(self, method, kwargs):
        woe, X, _ = self.fit(method, kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no category may be unseen at transform
            scored = woe.transform(X)["x"]
        binned = woe._apply_binning_to_column(X, "x")
        expected = binned.map(woe.mappings_["x"]["woe"])
        np.testing.assert_allclose(scored.to_numpy(), expected.to_numpy())


def test_merged_bins_would_have_hidden_a_real_difference():
    """The two tree bins that shared a label at one decimal differ in risk."""
    X, y = small_decimal_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        woe = FastWoe().fit(X, y)
    edges = np.asarray(woe.binning_info_["x"]["bin_edges"])
    one_decimal = [f"({a:.1f}, {b:.1f}]" for a, b in zip(edges[1:-2], edges[2:-1])]
    shared = [i for i in range(len(one_decimal)) if one_decimal.count(one_decimal[i]) > 1]
    assert shared, "the data should produce a one-decimal collision"
    rates = woe.get_mapping("x")["event_rate"].to_numpy()[[i + 1 for i in shared]]
    assert np.ptp(rates) > 0.02
