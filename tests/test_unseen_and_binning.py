"""Tests for unseen-category handling and fit/transform bin consistency."""

import warnings

import numpy as np
import pandas as pd
import pytest

from fastwoe import FastWoe


@pytest.fixture
def cat_data():
    """A categorical feature with three levels and a clear signal."""
    rng = np.random.default_rng(1)
    n = 1500
    x = rng.choice(list("abc"), n)
    y = (rng.random(n) < np.where(x == "a", 0.7, 0.2)).astype(int)
    return pd.DataFrame({"c": x}), y


@pytest.fixture
def num_data():
    """A numeric feature with a monotone signal and no missing values."""
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-x))).astype(int)
    return pd.DataFrame({"num": x}), y


class TestUnseenCategories:
    """A category absent from the fitted mapping falls back to WOE 0.

    That is the prior, not "no information", so it must not happen silently.
    """

    def test_warns_by_default(self, cat_data):
        X, y = cat_data
        woe = FastWoe().fit(X, y)
        with pytest.warns(UserWarning, match="not seen during fit"):
            woe.transform(pd.DataFrame({"c": ["a", "z"]}))

    def test_prior_is_silent(self, cat_data):
        X, y = cat_data
        woe = FastWoe(unseen="prior").fit(X, y)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = woe.transform(pd.DataFrame({"c": ["a", "z"]}))
        assert out["c"].iloc[1] == 0.0

    def test_raise_mode(self, cat_data):
        X, y = cat_data
        woe = FastWoe(unseen="raise").fit(X, y)
        with pytest.raises(ValueError, match="not seen during fit"):
            woe.transform(pd.DataFrame({"c": ["a", "z"]}))

    def test_no_warning_when_all_seen(self, cat_data):
        X, y = cat_data
        woe = FastWoe().fit(X, y)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            woe.transform(pd.DataFrame({"c": ["a", "b", "c"]}))
        assert woe.unseen_counts_ == {}

    def test_counts_are_recorded(self, cat_data):
        X, y = cat_data
        woe = FastWoe(unseen="prior").fit(X, y)
        woe.transform(pd.DataFrame({"c": ["a", "z", "z", None]}))
        assert woe.unseen_counts_["c"]["z"] == 2

    def test_counts_reset_between_transforms(self, cat_data):
        X, y = cat_data
        woe = FastWoe(unseen="prior").fit(X, y)
        woe.transform(pd.DataFrame({"c": ["z"]}))
        woe.transform(pd.DataFrame({"c": ["a"]}))
        assert woe.unseen_counts_ == {}

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="unseen must be"):
            FastWoe(unseen="nonsense")


class TestMissingBucket:
    """Missing values are a learned bin when present at fit, and unseen when not."""

    def test_missing_bin_is_learned_when_present(self):
        rng = np.random.default_rng(0)
        n = 2000
        x = rng.normal(size=n)
        x[:200] = np.nan
        y = (rng.random(n) < 1 / (1 + np.exp(-np.nan_to_num(x, nan=1.5)))).astype(int)
        woe = FastWoe().fit(pd.DataFrame({"num": x}), y)
        assert "Missing" in [str(c) for c in woe.mappings_["num"].index]
        assert woe.mappings_["num"].loc["Missing", "count"] == 200

    def test_missing_bin_scores_without_warning(self):
        rng = np.random.default_rng(0)
        n = 2000
        x = rng.normal(size=n)
        x[:200] = np.nan
        y = (rng.random(n) < 1 / (1 + np.exp(-np.nan_to_num(x, nan=1.5)))).astype(int)
        woe = FastWoe().fit(pd.DataFrame({"num": x}), y)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = woe.transform(pd.DataFrame({"num": [np.nan]}))
        assert out["num"].iloc[0] == pytest.approx(woe.mappings_["num"].loc["Missing", "woe"])

    def test_nan_unseen_at_fit_is_flagged(self, num_data):
        """The gap: fitted without NaN, scored with NaN, silently the prior."""
        X, y = num_data
        woe = FastWoe().fit(X, y)
        with pytest.warns(UserWarning, match="Missing"):
            out = woe.transform(pd.DataFrame({"num": [0.5, np.nan]}))
        assert out["num"].iloc[1] == 0.0
        assert woe.unseen_counts_["num"]["Missing"] == 1


class TestBinConsistency:
    """transform() must place a value in the bin fit() counted it in.

    A mismatch scores every numeric value with a neighbouring bin's weight.
    """

    def test_fit_and_transform_agree_on_every_bin(self, num_data):
        X, y = num_data
        woe = FastWoe().fit(X, y)
        mapping = woe.mappings_["num"]
        assigned = woe._apply_binning_to_column(X, "num").value_counts()
        for cat in mapping.index:
            assert int(assigned.get(cat, 0)) == int(mapping.loc[cat, "count"]), (
                f"bin {cat!r}: fit counted {mapping.loc[cat, 'count']}, "
                f"transform assigned {assigned.get(cat, 0)}"
            )

    def test_score_matches_the_local_event_rate(self, num_data):
        """The weight a value receives must reflect the data around that value.

        Compared against the empirical rate in a window of the training data,
        so it cannot pass by agreeing with whichever bin transform picked.
        """
        X, y = num_data
        woe = FastWoe().fit(X, y)
        x = X["num"].to_numpy()
        prior = y.mean()
        odds_prior = prior / (1 - prior)
        for value in (-1.5, -0.5, 0.5, 1.5):
            window = (x > value - 0.15) & (x < value + 0.15)
            local_rate = y[window].mean()
            scored = woe.transform(pd.DataFrame({"num": [value]}))["num"].iloc[0]
            odds = odds_prior * np.exp(scored)
            implied_rate = odds / (1 + odds)
            assert implied_rate == pytest.approx(local_rate, abs=0.10), (
                f"x={value}: scored weight implies P={implied_rate:.3f} "
                f"but the local event rate is {local_rate:.3f}"
            )

    def test_monotone_signal_gives_monotone_scores(self, num_data):
        """With a monotone generator, higher x must not score lower."""
        X, y = num_data
        woe = FastWoe().fit(X, y)
        grid = pd.DataFrame({"num": [-2.0, -1.0, 0.0, 1.0, 2.0]})
        scores = woe.transform(grid)["num"].tolist()
        assert scores == sorted(scores), f"non-monotone scores: {scores}"
