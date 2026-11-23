"""Tests for JointWoe: WOE tables fitted jointly (L2 logistic regression on the bins)."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder

from fastwoe import JointWoe


def correlated(n=6000, seed=0):
    """Two correlated numeric features with missing values, a categorical one, and a 0/1 target."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "a": z + 0.5 * rng.normal(size=n),
            "b": z + 0.5 * rng.normal(size=n),
            "c": rng.choice(["x", "y", "z"], n),
        }
    )
    X.loc[rng.random(n) < 0.05, "a"] = np.nan
    y = rng.binomial(1, expit(-1 + 1.2 * z + 0.4 * (X.c == "x")))
    return X, y


def cells(n=4000, seed=1):
    """Two categorical features (no binning) and a 0/1 target."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"u": rng.choice(list("pqrs"), n), "v": rng.choice(list("klm"), n)})
    eta = (
        -0.5
        + X.u.map({"p": -1.0, "q": 0.0, "r": 0.5, "s": 1.2})
        + X.v.map({"k": 0.4, "l": 0, "m": -0.6})
    )
    return X, rng.binomial(1, expit(eta.to_numpy()))


def test_equals_logistic_regression_on_one_hot_bins():
    X, y = correlated()
    m = JointWoe(C=0.5).fit(X, y)
    bins = m.binner_.transform_bins(X).astype(object).fillna("Missing").astype(str)
    onehot = OneHotEncoder().fit_transform(bins)
    lr = LogisticRegression(C=0.5, tol=1e-10, max_iter=10_000).fit(onehot, y)
    np.testing.assert_allclose(m.decision_function(X), lr.decision_function(onehot), atol=1e-4)
    assert "Missing" in m.levels_["a"]


def test_tables_sum_to_zero():
    X, y = correlated()
    m = JointWoe(C=0.3).fit(X, y)
    for table in m.tables_.values():
        assert abs(table.sum()) < 1e-4


def test_duplicated_feature_splits_evidence_evenly():
    X, y = cells()
    single = JointWoe(C=10.0).fit(X, y)
    double = JointWoe(C=10.0).fit(X.assign(u2=X.u), y)
    np.testing.assert_allclose(double.tables_["u"], double.tables_["u2"], atol=1e-3)
    np.testing.assert_allclose(
        double.tables_["u"] + double.tables_["u2"], single.tables_["u"], atol=0.02
    )
    np.testing.assert_allclose(
        double.decision_function(X.assign(u2=X.u)), single.decision_function(X), atol=0.02
    )


def test_soft_target_equals_grouped_labels():
    """A soft target per cell with counts as weights is the same likelihood as the 0/1 rows."""
    X, y = cells()
    rows = JointWoe(C=1.0, binner=None).fit(X, y)
    grouped = (
        pd.DataFrame({"u": X.u, "v": X.v, "y": y})
        .groupby(["u", "v"])
        .y.agg(["mean", "size"])
        .reset_index()
    )
    soft = JointWoe(C=1.0).fit(
        grouped[["u", "v"]], grouped["mean"], sample_weight=grouped["size"].to_numpy()
    )
    np.testing.assert_allclose(soft.decision_function(X), rows.decision_function(X), atol=1e-4)


def test_sample_weight_two_equals_duplicated_rows():
    X, y = cells()
    w = np.where(np.arange(len(y)) % 3 == 0, 2.0, 1.0)
    weighted = JointWoe().fit(X, y, sample_weight=w)
    doubled = JointWoe().fit(pd.concat([X, X[w == 2]]), np.r_[y, y[w == 2]])
    np.testing.assert_allclose(
        weighted.decision_function(X), doubled.decision_function(X), atol=1e-4
    )


def test_predict_ci_matches_the_exact_single_feature_interval():
    """One feature, almost no penalty: each bin's log-odds is logit(event rate), SE sqrt(1/events + 1/non-events)."""
    X, y = cells()
    m = JointWoe(C=1e8).fit(X[["u"]], y)
    lo, hi = m.predict_ci(X[["u"]], alpha=0.1).T
    stats = pd.DataFrame({"u": X.u, "y": y}).groupby("u").y.agg(["sum", "size"])
    events, nonevents = stats["sum"].to_numpy(), (stats["size"] - stats["sum"]).to_numpy()
    center = logit(events / stats["size"].to_numpy())
    half = norm.ppf(0.95) * np.sqrt(1 / events + 1 / nonevents)
    first = X.u.reset_index(drop=True).drop_duplicates().sort_values()
    np.testing.assert_allclose(lo[first.index], expit(center - half), rtol=1e-3)
    np.testing.assert_allclose(hi[first.index], expit(center + half), rtol=1e-3)


def test_standard_errors_are_positive_and_shrink_with_data():
    X, y = correlated()
    se_small = JointWoe().fit(X[:1500], y[:1500]).transform(X, output="se")
    se_large = JointWoe().fit(X, y).transform(X, output="se")
    assert (se_large.to_numpy() > 0).all()
    assert se_large.to_numpy().mean() < se_small.to_numpy().mean()


def test_unseen_bins_follow_the_policy():
    X, y = cells()
    new = pd.DataFrame({"u": ["p", "t"], "v": ["k", "k"]})
    with pytest.warns(UserWarning, match="not seen during fit"):
        w = JointWoe().fit(X, y).transform(new)
    assert w.loc[1, "u"] == 0.0
    assert JointWoe(unseen="prior").fit(X, y).transform(new).loc[1, "u"] == 0.0
    with pytest.raises(ValueError, match="not seen during fit"):
        JointWoe(unseen="raise").fit(X, y).transform(new)


def test_iv_and_mapping():
    X, y = correlated()
    m = JointWoe().fit(X.assign(noise=np.random.default_rng(5).normal(size=len(y))), y)
    iv = m.iv(X.assign(noise=0.0), y)
    assert iv["a"] > 0 and iv["b"] > 0
    table = m.get_mapping("c")
    assert table["count"].sum() == len(y)
    np.testing.assert_allclose(
        table.set_index("category").event_rate, pd.Series(y).groupby(X.c.to_numpy()).mean()
    )
    assert {"marginal_woe", "woe", "woe_se"} <= set(table.columns)


def test_scikit_learn_tooling():
    X, y = correlated(n=3000)
    assert clone(JointWoe(C=0.2)).get_params()["C"] == 0.2
    scores = cross_val_score(JointWoe(), X, y, cv=3, scoring="roc_auc")
    assert scores.mean() > 0.7
    search = GridSearchCV(JointWoe(), {"C": [0.1, 1.0]}, cv=3, scoring="neg_log_loss").fit(X, y)
    assert search.best_params_["C"] in (0.1, 1.0)


def test_custom_binner():
    class Quartiles:
        def fit(self, X, y):
            self.edges = {c: np.nanquantile(X[c], [0.25, 0.5, 0.75]) for c in X.columns}

        def transform(self, X):
            return pd.DataFrame(
                {c: np.digitize(X[c], self.edges[c]) for c in X.columns}, index=X.index
            )

    X, y = correlated()
    m = JointWoe(binner=Quartiles()).fit(X[["a", "b"]].fillna(0), y)
    assert [len(m.levels_[f]) for f in ("a", "b")] == [4, 4]


@pytest.mark.parametrize(
    "kwargs, y, match",
    [
        ({"C": 0}, None, "C must be positive"),
        ({}, "bad", r"0/1 labels or proportions"),
        ({}, "one class", "both events and non-events"),
    ],
)
def test_invalid_input(kwargs, y, match):
    X, labels = cells()
    target = {None: labels, "bad": labels * 2, "one class": np.zeros(len(labels))}[y]
    with pytest.raises(ValueError, match=match):
        JointWoe(**kwargs).fit(X, target)
