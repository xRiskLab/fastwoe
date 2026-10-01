"""Tests for SoftmaxWoe, conditional WOE by hierarchical softmax."""

import itertools

import numpy as np
import pandas as pd
import pytest
import sklearn

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
def test_bad_parameters(data, kwargs, match):
    """Invalid parameters raise at fit; construction stores them as given (scikit-learn style)."""
    X, y = data
    model = SoftmaxWoe(**kwargs)
    with pytest.raises(ValueError, match=match):
        model.fit(X, y)


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
        SoftmaxWoe(binning_kwargs={"conditional": True}).fit(X, y)


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


@pytest.mark.skipif(
    tuple(int(v) for v in sklearn.__version__.split(".")[:2]) < (1, 4),
    reason="tree monotonic constraints need scikit-learn 1.4",
)
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


def test_weight_se_shape_and_root_formula(numeric_data):
    """SEs are positive per feature; the root's is the multinomial (1 - p) / (n p) in each class."""
    X, y = numeric_data
    m = SoftmaxWoe(order=["score", "card"]).fit(X, y)
    se = m.transform(X, output="se")
    assert list(se.columns) == ["score", "card"]
    assert np.isfinite(se.to_numpy()).all() and (se.to_numpy() > 0).all()
    p = m.node_proba(X, "score")
    n1, n0 = m.class_counts_[1], m.class_counts_[0]
    expected = np.sqrt(
        (1 - p.p_event) / (n1 * p.p_event) + (1 - p.p_nonevent) / (n0 * p.p_nonevent)
    )
    np.testing.assert_allclose(se["score"], expected)


def test_weight_se_shrinks_with_sample_size():
    """Four times the data halves the standard errors."""
    rng = np.random.default_rng(3)

    def draw(n):
        y = rng.binomial(1, 0.3, n)
        a = np.where(rng.random(n) < 0.3 + 0.3 * y, "hi", "lo")
        b = np.where(rng.random(n) < np.where(a == "hi", 0.6, 0.3) + 0.1 * y, "x", "y")
        return pd.DataFrame({"a": a, "b": b}), y

    probe = pd.DataFrame({"a": ["hi", "lo"], "b": ["x", "y"]})
    small = SoftmaxWoe(C=100.0).fit(*draw(4_000)).transform(probe, output="se")
    large = SoftmaxWoe(C=100.0).fit(*draw(16_000)).transform(probe, output="se")
    np.testing.assert_allclose(small / large, 2.0, rtol=0.15)


def test_predict_ci(numeric_data):
    """The interval contains the point estimate and widens as alpha falls."""
    X, y = numeric_data
    m = SoftmaxWoe().fit(X, y)
    p = m.predict_proba(X)[:, 1]
    ci95, ci80 = m.predict_ci(X), m.predict_ci(X, alpha=0.2)
    assert ci95.shape == (len(X), 2)
    assert (ci95[:, 0] <= p).all() and (p <= ci95[:, 1]).all()
    assert (ci95[:, 0] <= ci80[:, 0]).all() and (ci80[:, 1] <= ci95[:, 1]).all()
    with pytest.raises(ValueError, match="alpha"):
        m.predict_ci(X, alpha=1.5)
    with pytest.raises(ValueError, match="output must be"):
        m.transform(X, output="wald")


def test_unseen_category_has_zero_se(data):
    """A category unseen at fit has weight 0 and standard error 0."""
    X, y = data
    new = X.head(2).copy()
    new.loc[new.index[0], "c"] = "zzz"
    m = SoftmaxWoe(unseen="prior").fit(X, y)
    se = m.transform(new, output="se")
    assert se.loc[new.index[0], "c"] == 0.0 and se.loc[new.index[1], "c"] > 0


@pytest.mark.slow
def test_intervals_cover_true_weights():
    """With a correctly specified node, 95% intervals cover the true weights about 95% of the time."""
    levels_a, levels_b = ["p", "q", "r"], ["u", "v", "w"]
    share_a = {1: np.array([0.5, 0.3, 0.2]), 0: np.array([0.2, 0.3, 0.5])}
    # P(b | a, class): a full table, which a softmax on one-hot a represents exactly
    table_b = {
        1: np.array([[0.6, 0.3, 0.1], [0.3, 0.4, 0.3], [0.2, 0.2, 0.6]]),
        0: np.array([[0.3, 0.4, 0.3], [0.2, 0.3, 0.5], [0.1, 0.3, 0.6]]),
    }
    probe = pd.DataFrame(list(itertools.product(levels_a, levels_b)), columns=["a", "b"])
    ia = probe.a.map(levels_a.index).to_numpy()
    ib = probe.b.map(levels_b.index).to_numpy()
    true_w = np.column_stack(
        [
            np.log(share_a[1][ia] / share_a[0][ia]),
            np.log(table_b[1][ia, ib] / table_b[0][ia, ib]),
        ]
    )
    rng = np.random.default_rng(0)
    covered, reps = np.zeros_like(true_w), 150
    for _ in range(reps):
        n = 6000
        y = rng.binomial(1, 0.25, n)
        a = np.array([rng.choice(3, p=share_a[c]) for c in y])
        b = np.array([rng.choice(3, p=table_b[c][k]) for c, k in zip(y, a)])
        X = pd.DataFrame({"a": np.array(levels_a)[a], "b": np.array(levels_b)[b]})
        m = SoftmaxWoe(C=1000.0).fit(X, y)
        w, se = m.transform(probe).to_numpy(), m.transform(probe, output="se").to_numpy()
        covered += np.abs(w - true_w) <= 1.96 * se
    coverage = covered.mean(axis=0) / reps
    assert ((coverage > 0.92) & (coverage < 0.98)).all(), coverage


class EdgesBinner:
    """A user-side binner: fixed cut points per column, as xgboost or a scorecard would give.

    fit() returns None on purpose: the contract only needs fit to be callable.
    """

    def __init__(self, edges):
        self.edges = edges

    def fit(self, X, y):
        self.fitted_ = True

    def transform(self, X):
        out = X.copy()
        for col, cuts in self.edges.items():
            out[col] = pd.cut(X[col], [-np.inf, *cuts, np.inf], right=False)
        return out


def test_custom_binner_contract(numeric_data):
    """A binner with fit/transform supplies the bins; interval order is kept."""
    X, y = numeric_data
    binner = EdgesBinner({"score": [520, 560, 600]})
    m = SoftmaxWoe(order=["score", "card"], binner=binner).fit(X, y)
    assert not hasattr(binner, "fitted_")  # the caller's object is copied, not fitted
    assert m.binner_.fitted_
    levels = m.levels_["score"]
    assert [str(b) for b in levels[:-1]] == [
        "[-inf, 520.0)",
        "[520.0, 560.0)",
        "[560.0, 600.0)",
        "[600.0, inf)",
    ]
    assert levels[-1] == "Missing"
    assert np.isfinite(m.predict_proba(X)).all()
    # node_proba with only the root column: the binner still sees every fitted column
    p = m.node_proba(X[["score"]].head(3), "score")
    assert np.isfinite(p.to_numpy()).all()


def test_sklearn_binner_returning_array():
    """A scikit-learn transformer returning a numpy array of codes works and is cloned."""
    from sklearn.preprocessing import KBinsDiscretizer

    rng = np.random.default_rng(4)
    n = 3000
    y = rng.binomial(1, 0.3, n)
    X = pd.DataFrame({"a": rng.normal(y, 1, n), "b": rng.normal(-y, 1, n)})
    kb = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="uniform")
    m = SoftmaxWoe(binner=kb).fit(X, y)
    assert not hasattr(kb, "bin_edges_")
    assert m.levels_["a"] == [0.0, 1.0, 2.0, 3.0]
    assert m.predict_proba(X).shape == (n, 2)


def test_binner_contract_errors(numeric_data):
    """Both binning options, a binner without transform, or a wrong shape are refused."""
    X, y = numeric_data
    with pytest.raises(ValueError, match="not both"):
        SoftmaxWoe(binner=EdgesBinner({}), binning_kwargs={"binning_method": "kbins"}).fit(X, y)

    class NoTransform:
        def fit(self, X, y):
            pass

    with pytest.raises(TypeError, match="fit\\(X, y\\) and transform\\(X\\)"):
        SoftmaxWoe(binner=NoTransform()).fit(X, y)

    class WrongShape(EdgesBinner):
        def transform(self, X):
            return np.zeros((len(X), 1))

    with pytest.raises(ValueError, match="shape"):
        SoftmaxWoe(binner=WrongShape({})).fit(X, y)


def test_unbinned_continuous_column_warns(numeric_data):
    """A binner that leaves a continuous column as is triggers a warning."""
    X, y = numeric_data
    X, y = X.head(300), y.head(300)  # small: every distinct value becomes a level
    with pytest.warns(UserWarning, match="left unbinned"):
        SoftmaxWoe(binner=EdgesBinner({})).fit(X, y)


def test_binner_protocol():
    """Binner is a structural type: any object with fit and transform satisfies it."""
    from fastwoe.softmax_woe import Binner

    assert isinstance(EdgesBinner({}), Binner)
    assert isinstance(FastWoe(), Binner)
    assert not isinstance(object(), Binner)


def test_custom_binner_unseen_label(numeric_data):
    """A label the binner never produced at fit follows the unseen policy."""
    X, y = numeric_data

    class CardLabels(EdgesBinner):
        def transform(self, X):
            out = super().transform(X)
            out["card"] = X["card"].map({"N": "no card", "Y": "card"}).fillna("other")
            return out

    m = SoftmaxWoe(order=["score", "card"], binner=CardLabels({"score": [560]})).fit(X, y)
    new = X.head(2).assign(card=["Z", "Y"])  # "Z" becomes "other", never seen at fit
    with pytest.warns(UserWarning, match="other"):
        w = m.transform(new)
    assert w["card"].iloc[0] == 0.0 and w["card"].iloc[1] != 0.0
    with pytest.raises(ValueError, match="other"):
        SoftmaxWoe(
            order=["score", "card"], binner=CardLabels({"score": [560]}), unseen="raise"
        ).fit(X, y).transform(new)


def test_custom_binner_column_order_is_matched_by_name(numeric_data):
    """A binner may return its columns in any order; they are matched by name."""
    X, y = numeric_data

    class Reversed(EdgesBinner):
        def transform(self, X):
            return super().transform(X)[list(X.columns)[::-1]]

    edges = {"score": [520, 560, 600]}
    a = SoftmaxWoe(order=["score", "card"], binner=EdgesBinner(edges)).fit(X, y)
    b = SoftmaxWoe(order=["score", "card"], binner=Reversed(edges)).fit(X, y)
    assert a.levels_ == b.levels_
    pd.testing.assert_frame_equal(a.transform(X), b.transform(X))


def test_scikit_learn_estimator(data):
    """clone, get/set_params, predict, fit_transform, cross-validation and grid search work."""
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV, cross_val_score

    X, y = data
    model = SoftmaxWoe(order=["a", "b", "c"], C=0.5)
    copy_ = clone(model)
    assert copy_.get_params() == model.get_params()
    assert copy_.set_params(C=2.0).C == 2.0 and model.C == 0.5
    model.fit(X, y)
    assert list(model.classes_) == [0, 1] and model.n_features_in_ == 3
    np.testing.assert_array_equal(
        model.predict(X), (model.predict_proba(X)[:, 1] >= 0.5).astype(int)
    )
    pd.testing.assert_frame_equal(
        SoftmaxWoe(order=["a", "b", "c"]).fit_transform(X, y),
        SoftmaxWoe(order=["a", "b", "c"]).fit(X, y).transform(X),
    )
    scores = cross_val_score(SoftmaxWoe(), X, y, cv=3, scoring="neg_log_loss")
    assert np.isfinite(scores).all()
    search = GridSearchCV(SoftmaxWoe(), {"C": [0.01, 1.0]}, cv=3, scoring="neg_log_loss").fit(X, y)
    assert search.best_params_["C"] in (0.01, 1.0)


def test_covariances_computed_on_demand(data):
    """fit computes no covariances; the first standard error does, and caches them."""
    X, y = data
    m = SoftmaxWoe().fit(X, y)
    assert m._node_cov == {}
    se = m.transform(X, output="se")
    assert len(m._node_cov) == len(m._node_C) > 0
    pd.testing.assert_frame_equal(se, m.transform(X, output="se"))
