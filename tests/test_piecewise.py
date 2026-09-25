"""Piecewise WOE (Anderson, 2015): assign_pieces and transform(output="piecewise")."""

import numpy as np
import pandas as pd
import pytest

from fastwoe import FastWoe


@pytest.fixture
def fitted():
    rng = np.random.default_rng(0)
    n = 4000
    grade = rng.choice(list("ABCD"), n)
    logit = -1.5 + np.select([grade == "A", grade == "D"], [-1.0, 1.0], 0.0)
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    X = pd.DataFrame({"grade": grade})
    return FastWoe().fit(X, y), X


def test_assign_pieces_updates_in_place_and_returns_none(fitted):
    """assign_pieces() mutates the encoder, so it returns None rather than self."""
    woe, _ = fitted
    assert "piece" not in woe.mappings_["grade"].columns
    assert woe.assign_pieces(strategy="sign") is None
    mapping = woe.mappings_["grade"]
    assert set(mapping["piece"]) == {0, 1}
    assert (mapping.loc[mapping["woe"] < 0, "piece"] == 0).all()
    assert (mapping.loc[mapping["woe"] >= 0, "piece"] == 1).all()


def test_piece_map_overrides_strategy(fitted):
    woe, _ = fitted
    woe.assign_pieces(piece_map={"grade": {"A": 0, "B": 0, "C": 1, "D": 2}})
    assert woe.mappings_["grade"]["piece"].to_dict() == {"A": 0, "B": 0, "C": 1, "D": 2}


def test_piecewise_columns_split_the_weight(fitted):
    """Each row's WOE lands in exactly one piece column; the others are 0."""
    woe, X = fitted
    woe.assign_pieces(strategy="sign")
    pieces = woe.transform(X, output="piecewise")
    assert list(pieces.columns) == ["grade__piece_0", "grade__piece_1"]
    np.testing.assert_allclose(pieces.sum(axis=1), woe.transform(X)["grade"])
    assert ((pieces != 0).sum(axis=1) <= 1).all()


def test_positional_piece_map_follows_get_mapping_rows():
    """Integer keys index the rows of get_mapping(), which are in bin order.

    For a binned feature the internal order sorts bin labels as strings, so
    translating through it assigned pieces to the wrong bins.
    """
    rng = np.random.default_rng(0)
    n = 5000
    x = rng.normal(size=n)
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(2 - x))).astype(int))
    woe = FastWoe().fit(pd.DataFrame({"score": x}), y)
    shown = woe.get_mapping("score")["category"].tolist()
    assert shown != woe.mappings_["score"].index.tolist()  # the orders do differ

    woe.assign_pieces(piece_map={"score": {i: int(i == 0) for i in range(len(shown))}})
    pieces = woe.mappings_["score"]["piece"]
    assert pieces[pieces == 1].index.tolist() == [shown[0]]
