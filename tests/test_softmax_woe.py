"""Tests for SoftmaxWoe, conditional WOE by hierarchical softmax."""

import itertools

import numpy as np
import pandas as pd
import pytest

from fastwoe import FastWoe
from fastwoe.softmax_woe import SoftmaxWoe


@pytest.fixture
def data():
    """Three correlated categorical features; the later ones depend on the first."""
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.binomial(1, 0.2, n)
    a = np.where(rng.random(n) < 0.3 + 0.4 * y, "hi", "lo")
    b = np.where(rng.random(n) < np.where(a == "hi", 0.7, 0.2), "x", "y")
    c = rng.choice(["p", "q", "r"], n, p=[0.5, 0.3, 0.2])
    X = pd.DataFrame({"a": a, "b": b, "c": c})
    return X, pd.Series(y)


def test_weights_add_up_to_proper_chain(data):
    """Each class's chain is a distribution over all profiles, so prior + sum W is exact."""
    X, y = data
    m = SoftmaxWoe(C=1.0).fit(X, y)
    grid = pd.DataFrame(itertools.product(*m.levels_.values()), columns=m.order_)
    n = len(grid)
    codes = {f: m._codes(grid[f], f) for f in m.order_}
    rows = np.arange(n)
    for cls in (1, 0):
        lik = np.ones(n)
        for i, f in enumerate(m.order_):
            lik *= m._node_probs(i, cls, codes, n)[rows, codes[f]]
        assert lik.sum() == pytest.approx(1.0, abs=1e-9)


def test_predict_proba_is_prior_plus_weights(data):
    """predict_proba is the sigmoid of prior_log_odds_ plus the row sum of transform()."""
    X, y = data
    m = SoftmaxWoe().fit(X, y)
    s = m.prior_log_odds_ + m.transform(X).sum(axis=1).to_numpy()
    np.testing.assert_allclose(m.predict_proba(X)[:, 1], 1 / (1 + np.exp(-s)))


def test_strong_penalty_gives_marginal_woe(data):
    """As C goes to 0 every node shrinks to its bin shares: marginal WOE (naive Bayes)."""
    X, y = data
    soft = SoftmaxWoe(C=1e-8).fit(X, y).transform(X)
    marginal = FastWoe().fit(X, y).transform(X)
    np.testing.assert_allclose(soft.to_numpy(), marginal.to_numpy(), atol=2e-3)


def test_first_feature_weight_is_marginal(data):
    """The root node has nothing to condition on, so its weight is the marginal WOE."""
    X, y = data
    soft = SoftmaxWoe(C=10.0).fit(X, y).transform(X)["a"]
    marginal = FastWoe().fit(X, y).transform(X)["a"]
    np.testing.assert_allclose(soft, marginal, atol=2e-3)


def test_conditioning_removes_double_count(data):
    """b mostly restates a, so given a its weight shrinks well below the marginal one."""
    X, y = data
    soft = SoftmaxWoe(C=10.0).fit(X, y).transform(X)["b"].abs().mean()
    marginal = FastWoe().fit(X, y).transform(X)["b"].abs().mean()
    assert soft < 0.5 * marginal


def test_order_changes_attribution(data):
    """Putting b first gives it the marginal weight instead of the conditional one."""
    X, y = data
    fwd = SoftmaxWoe(order=["a", "b", "c"], C=10.0).fit(X, y).transform(X)
    rev = SoftmaxWoe(order=["b", "a", "c"], C=10.0).fit(X, y).transform(X)
    assert list(fwd.columns) == list(rev.columns) == ["a", "b", "c"]
    assert rev["b"].abs().mean() > 2 * fwd["b"].abs().mean()


def test_missing_is_its_own_level(data):
    """NaN is a level of its own, at fit and transform."""
    X, y = data
    X = X.copy()
    X.loc[::7, "c"] = np.nan
    m = SoftmaxWoe().fit(X, y)
    assert "Missing" in m.levels_["c"]
    assert np.isfinite(m.transform(X).to_numpy()).all()


def test_unseen_category(data):
    """An unseen category gets weight 0; warn by default, raise or stay silent on request."""
    X, y = data
    new = X.head(3).copy()
    new.loc[new.index[0], "c"] = "zzz"
    with pytest.warns(UserWarning, match="zzz"):
        w = SoftmaxWoe().fit(X, y).transform(new)
    assert w.loc[new.index[0], "c"] == 0.0
    with pytest.raises(ValueError, match="zzz"):
        SoftmaxWoe(unseen="raise").fit(X, y).transform(new)
    SoftmaxWoe(unseen="prior").fit(X, y).transform(new)


@pytest.mark.parametrize(
    "kwargs, match",
    [({"C": 0}, "C must be positive"), ({"unseen": "x"}, "unseen must be")],
)
def test_bad_parameters(kwargs, match):
    """Invalid constructor arguments raise."""
    with pytest.raises(ValueError, match=match):
        SoftmaxWoe(**kwargs)


def test_bad_order_and_target(data):
    """order must cover every column once; the target must be binary."""
    X, y = data
    with pytest.raises(ValueError, match="order must list"):
        SoftmaxWoe(order=["a", "b"]).fit(X, y)
    with pytest.raises(ValueError, match="binary"):
        SoftmaxWoe().fit(X, y * 2)


def test_node_proba_matches_transform(data):
    """The log ratio of node_proba's two columns is the feature's weight."""
    X, y = data
    m = SoftmaxWoe(C=1.0).fit(X, y)
    w = m.transform(X)
    for f in m.order_:
        p = m.node_proba(X, f)
        np.testing.assert_allclose(np.log(p.p_event / p.p_nonevent), w[f])
    p = m.node_proba(X[["a"]], "a")  # the root needs no earlier features
    assert list(p.columns) == ["p_event", "p_nonevent"]
    new = X.head(1).copy()
    new["b"] = "zzz"
    assert m.node_proba(new, "b").isna().all(axis=None)
    with pytest.raises(ValueError, match="unknown feature"):
        m.node_proba(X, "nope")


@pytest.fixture
def numeric_data():
    """A numeric score with missing values next to a categorical feature."""
    rng = np.random.default_rng(1)
    n = 4000
    y = rng.binomial(1, 0.2, n)
    score = rng.normal(600 - 60 * y, 50)
    X = pd.DataFrame({"score": score, "card": np.where(rng.random(n) < 0.3 + 0.3 * y, "N", "Y")})
    X.loc[::40, "score"] = np.nan
    return X, pd.Series(y)


def test_numeric_features_use_fastwoe_bins(numeric_data):
    """Numeric columns are binned exactly as FastWoe bins them, in numeric order."""
    X, y = numeric_data
    m = SoftmaxWoe().fit(X, y)
    bins = FastWoe().fit(X, y).transform_bins(X)["score"]
    expected = [b for b in bins.cat.categories if b in set(bins.astype(str))]
    assert m.levels_["score"] == expected
    assert expected[-1] == "Missing" and len(expected) > 3
    assert m.levels_["card"] == ["N", "Y"]
    # the root feature's weight is FastWoe's marginal WOE, in every bin holding both
    # classes; a one-class bin is tempered by the root's 0.5 pseudo-count instead
    both = y.groupby(bins.astype(str)).transform(lambda t: 0 < t.mean() < 1).astype(bool)
    np.testing.assert_allclose(
        m.transform(X)["score"][both], m.binner_.transform(X)["score"][both], atol=0.02
    )


def test_binning_kwargs_reach_fastwoe(numeric_data):
    """binning_kwargs configures the FastWoe that bins; conditional options are refused."""
    X, y = numeric_data
    m = SoftmaxWoe(binning_kwargs={"binning_method": "kbins", "binner_kwargs": {"n_bins": 4}})
    m.fit(X, y)
    assert m.binner_.binning_method == "kbins"
    assert len(m.levels_["score"]) == 5  # four bins and Missing
    with pytest.raises(ValueError, match="binning only"):
        SoftmaxWoe(binning_kwargs={"conditional": True})


def test_numpy_input(numeric_data):
    """A numpy array works, with columns named by position."""
    X, y = numeric_data
    A = X.assign(card=(X.card == "Y").astype(float)).to_numpy()
    m = SoftmaxWoe().fit(A, y.to_numpy())
    assert m.order_ == ["0", "1"]
    assert m.predict_proba(A).shape == (len(A), 2)


def test_two_bin_node_matches_symmetric_softmax():
    """A two-bin node is the penalized two-class softmax at the same C, not a harder fit."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(0)
    n, C = 3000, 0.1
    a = rng.choice(["p", "q", "r", "s"], n)
    y = rng.binomial(1, 0.3, n)
    effect = {"p": -1.0, "q": 0.0, "r": 0.5, "s": 1.5}
    b = np.where(rng.random(n) < 1 / (1 + np.exp(-pd.Series(a).map(effect))), "Y", "N")
    X = pd.DataFrame({"a": a, "b": b})
    m = SoftmaxWoe(C=C).fit(X, y)

    rows = y == 1
    A = np.eye(4)[pd.Categorical(a[rows], m.levels_["a"]).codes]
    t = pd.Categorical(b[rows], m.levels_["b"]).codes
    k = A.shape[1]

    def objective(theta):
        W, c = theta[: 2 * k].reshape(2, k), theta[2 * k :]
        Z = A @ W.T + c
        Z -= Z.max(axis=1, keepdims=True)
        logp = Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))
        return 0.5 * (W**2).sum() - C * logp[np.arange(len(t)), t].sum()

    theta = minimize(objective, np.zeros(2 * k + 2), method="L-BFGS-B", options={"gtol": 1e-10}).x
    symmetric = theta[k : 2 * k] - theta[:k]
    np.testing.assert_allclose(m.nodes_[("b", 1)].coef_[0], symmetric, atol=1e-3)


def test_faiss_binning_end_to_end(numeric_data):
    """FAISS k-means bins flow through to the chain, in numeric order."""
    pytest.importorskip("faiss")
    X, y = numeric_data
    m = SoftmaxWoe(binning_kwargs={"binning_method": "faiss_kmeans", "faiss_kwargs": {"k": 5}})
    m.fit(X, y)
    assert len(m.levels_["score"]) == 6 and m.levels_["score"][-1] == "Missing"
    lows = [
        float(b.split(",")[0].strip("(").replace("-∞", "-inf")) for b in m.levels_["score"][:-1]
    ]
    assert lows == sorted(lows)
    bins = m.binner_.transform_bins(X)["score"].astype(str)
    assert (m.transform(X)["score"].groupby(bins).nunique() == 1).all()
    assert np.isfinite(m.predict_proba(X)).all()


def test_monotonic_constraint_holds_for_first_feature(numeric_data):
    """monotonic_cst reaches the binner, so the first feature's weights are monotone in its bins."""
    X, y = numeric_data
    m = SoftmaxWoe(binning_kwargs={"monotonic_cst": {"score": -1}}).fit(X, y)
    assert m.binner_.monotonic_cst == {"score": -1}
    bins = m.binner_.transform_bins(X)["score"].astype(str)
    per_bin = m.transform(X)["score"].groupby(bins).first()
    ordered = per_bin.reindex([b for b in m.levels_["score"] if b != "Missing"])
    assert (np.diff(ordered.to_numpy()) <= 1e-9).all()


def test_low_cardinality_numeric_is_categorical(numeric_data):
    """A numeric column with few values is a category, as in FastWoe; a new value is unseen."""
    X, y = numeric_data
    X = X.assign(kids=np.random.default_rng(2).integers(0, 5, len(X)))
    m = SoftmaxWoe().fit(X, y)
    assert "kids" not in m.binner_.binners_
    assert m.levels_["kids"] == [0, 1, 2, 3, 4]
    new = X.head(2).assign(kids=[7, 2])
    with pytest.warns(UserWarning, match="not seen during fit"):
        w = m.transform(new)
    assert w["kids"].iloc[0] == 0.0 and w["kids"].iloc[1] != 0.0
