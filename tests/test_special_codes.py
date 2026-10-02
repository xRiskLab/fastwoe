"""Tests for special codes: values of a numerical feature kept out of the binning."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from fastwoe import FastWoe, SoftmaxWoe


@pytest.fixture
def scored():
    """A score with missing values and a -999 'no record' code that carries its own risk."""
    rng = np.random.default_rng(0)
    n = 6000
    x = rng.normal(600, 50, n)
    y = (rng.random(n) < 1 / (1 + np.exp((x - 600) / 30))).astype(int)
    x[::25] = np.nan
    special = rng.random(n) < 0.05
    x[special] = -999
    y[special] = (rng.random(special.sum()) < 0.6).astype(int)
    return pd.DataFrame({"score": x}), y, special


def test_missing_bin_is_in_get_mapping(scored):
    """get_mapping lists the Missing bin of a binned feature, after the intervals."""
    X, y, special = scored
    X = X[~special]
    fw = FastWoe().fit(X, y[~special])
    mapping = fw.get_mapping("score")
    assert mapping["category"].iloc[-1] == "Missing"
    assert mapping["count"].sum() == len(X)


@pytest.mark.parametrize("method", ["tree", "kbins", "faiss_kmeans"])
def test_special_codes_do_not_move_the_bins(scored, method):
    """Bin edges are those of a fit without the special rows."""
    if method == "faiss_kmeans" and importlib.util.find_spec("faiss") is None:
        pytest.skip("faiss is not installed")
    X, y, special = scored
    kwargs = {"binning_method": method}
    with_codes = FastWoe(special_codes=[-999], **kwargs).fit(X, y)
    without_rows = FastWoe(**kwargs).fit(X[~special], y[~special])
    a = with_codes.transform_bins(X[~special])["score"].astype(str)
    b = without_rows.transform_bins(X[~special])["score"].astype(str)
    assert (a == b).all()
    polluted = FastWoe(**kwargs).fit(X, y).transform_bins(X)["score"].astype(str)
    assert (polluted[special] != "Special").all()  # without special_codes, -999 sits in a bin


def test_special_bin_has_its_own_woe(scored):
    """The special bin gets the empirical WOE of its rows, with a standard error, before Missing."""
    X, y, special = scored
    fw = FastWoe(special_codes=[-999]).fit(X, y)
    mapping = fw.get_mapping("score").set_index("category")
    assert list(mapping.index[-2:]) == ["Special", "Missing"]
    assert mapping.loc["Special", "count"] == special.sum()
    rate, prior = y[special].mean(), y.mean()
    expected = np.log(rate / (1 - rate)) - np.log(prior / (1 - prior))
    assert mapping.loc["Special", "woe"] == pytest.approx(expected, abs=1e-3)
    assert mapping.loc["Special", "woe_se"] > 0
    assert mapping["count"].sum() == len(X)
    woe = fw.transform(X)["score"]
    assert np.allclose(woe[special], mapping.loc["Special", "woe"])
    summary = fw.get_binning_summary().iloc[0]
    assert summary["special"] == special.sum()
    assert summary["missing"] == int(X["score"].isna().sum())


def test_named_groups_and_value_containers(scored):
    """A dict per feature names its bins; lists, tuples, sets and single values all work."""
    X, y, _ = scored
    X = X.assign(score=np.where(np.arange(len(X)) % 97 == 0, -1, X["score"]))
    for codes in ({-999, -1}, (-999, -1), np.array([-999, -1])):
        fw = FastWoe(special_codes=codes).fit(X, y)
        assert "Special" in set(fw.get_mapping("score")["category"])
    named = FastWoe(special_codes={"score": {"no_record": -999, "refused": [-1]}}).fit(X, y)
    categories = list(named.get_mapping("score")["category"])
    assert categories[-3:] == ["Special: no_record", "Special: refused", "Missing"]


def test_special_code_unseen_at_fit(scored):
    """A special code absent at fit is unseen at transform and follows the unseen policy."""
    X, y, special = scored
    fw = FastWoe(special_codes=[-999]).fit(X[~special], y[~special])
    with pytest.warns(UserWarning, match="Special"):
        woe = fw.transform(X.head(3).assign(score=[-999, 600.0, 650.0]))
    assert woe["score"].iloc[0] == 0.0


def test_special_codes_validation(scored):
    """Wrong types, unknown features and overlapping groups are refused; unbinned features warn."""
    X, y, _ = scored
    with pytest.raises(TypeError, match="special_codes"):
        FastWoe(special_codes="-999")
    with pytest.raises(ValueError, match="not in X"):
        FastWoe(special_codes={"nope": [-999]}).fit(X, y)
    with pytest.raises(ValueError, match="in two bins"):
        FastWoe(special_codes={"score": {"a": [-999], "b": [-999, -1]}}).fit(X, y)
    with pytest.warns(UserWarning, match="not binned"):
        FastWoe(special_codes={"card": [-1]}).fit(X.assign(card=np.where(y == 1, "Y", "N")), y)


def test_special_codes_downstream(scored):
    """Conditional WOE, predict_ci and SoftmaxWoe all see the special bin."""
    X, y, special = scored
    X = X.assign(card=np.where(np.random.default_rng(1).random(len(X)) < 0.5, "Y", "N"))
    conditional = FastWoe(special_codes=[-999], conditional=True).fit(X, y)
    assert np.isfinite(conditional.transform(X).to_numpy()).all()
    ci = FastWoe(special_codes=[-999]).fit(X, y).predict_ci(X)
    assert np.isfinite(ci).all()
    softmax = SoftmaxWoe(order=["score", "card"], binning_kwargs={"special_codes": [-999]})
    softmax.fit(X, y)
    assert softmax.levels_["score"][-2:] == ["Special", "Missing"]
