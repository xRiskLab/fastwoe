"""Multiclass predict_proba and predict_ci: normalized probabilities, intervals by bin."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from scipy.stats import norm

from fastwoe import FastWoe


@pytest.fixture
def three_classes():
    """A numerical and a categorical feature against a three-class target."""
    rng = np.random.default_rng(0)
    n = 6000
    y = rng.choice([0, 1, 2], n, p=[0.6, 0.3, 0.1])
    X = pd.DataFrame(
        {
            "score": rng.normal(600 - 40 * y, 50),
            "card": np.where(rng.random(n) < np.array([0.2, 0.5, 0.8])[y], "hi", "lo"),
        }
    )
    return X, y


def one_vs_rest(fw, X):
    """Each class's one-vs-rest probability: sigmoid of its summed WOE plus its prior log-odds."""
    woe = fw.transform(X)
    columns = []
    for label in fw.classes_:
        score = woe[[c for c in woe.columns if c.endswith(f"_class_{label}")]].sum(axis=1)
        prior = fw.y_prior_[label]
        columns.append(expit(score + np.log(prior / (1 - prior))))
    return np.column_stack(columns)


def test_probabilities_sum_to_one(three_classes):
    """predict_proba normalizes the one-vs-rest probabilities, so rows sum to 1."""
    X, y = three_classes
    fw = FastWoe().fit(X, y)
    proba = fw.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    raw = one_vs_rest(fw, X)
    np.testing.assert_allclose(proba, raw / raw.sum(axis=1, keepdims=True))
    # normalizing a row keeps its ranking, so predict() is unchanged
    assert (np.asarray(fw.classes_)[raw.argmax(axis=1)] == fw.predict(X)).all()
    np.testing.assert_allclose(fw.predict_proba_class(X, 1), proba[:, 1])


def test_intervals_use_each_rows_bin(three_classes):
    """predict_ci looks up standard errors by bin, so interval widths follow the bins."""
    X, y = three_classes
    fw = FastWoe().fit(X, y)
    ci = fw.predict_ci(X)
    bins = fw.transform_bins(X)
    z = norm.ppf(0.975)
    for k, label in enumerate(fw.classes_):
        width = logit(ci[:, 2 * k + 1]) - logit(ci[:, 2 * k])
        se_score = bins["score"].astype(str).map(fw.mappings_["score"][label]["woe_se"].to_dict())
        se_card = X["card"].map(fw.mappings_["card"][label]["woe_se"].to_dict())
        expected = 2 * z * np.sqrt(se_score.to_numpy() ** 2 + se_card.to_numpy() ** 2)
        np.testing.assert_allclose(width, expected)
    assert len(np.unique(np.round(width, 6))) > 1


def test_numpy_input(three_classes):
    """A model fitted on a numpy array gives the same multiclass outputs as on a DataFrame."""
    X, y = three_classes
    A = X.assign(card=(X["card"] == "hi").astype(float)).to_numpy()
    fw = FastWoe().fit(A, y)
    proba, ci = fw.predict_proba(A), fw.predict_ci(A)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert ci.shape == (len(A), 2 * len(fw.classes_)) and np.isfinite(ci).all()
    named = pd.DataFrame(A, columns=["feature_0", "feature_1"])
    np.testing.assert_allclose(fw.predict_ci(named), ci)
