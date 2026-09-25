"""The metrics and plots modules import without the WOE encoder's dependencies."""

import subprocess
import sys

import pytest

import fastwoe

HEAVY = ("sklearn", "loguru", "rich")


def _loaded_after(statement: str) -> set:
    """Top-level modules from HEAVY loaded by `statement` in a fresh interpreter."""
    code = f"import sys; {statement}; print(','.join(m for m in {HEAVY!r} if m in sys.modules))"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    return set(filter(None, out.split(",")))


@pytest.mark.parametrize(
    "statement",
    [
        "import fastwoe",
        "import fastwoe.metrics",
        "from fastwoe.metrics import somersd_yx, gini_contributions",
        "from fastwoe import gini_contributions",
        "import fastwoe.plots",
        "from fastwoe import plot_performance, visualize_woe",
    ],
)
def test_light_imports_skip_heavy_dependencies(statement):
    assert _loaded_after(statement) == set()


def test_encoder_import_still_loads_sklearn():
    assert "sklearn" in _loaded_after("from fastwoe import FastWoe")


@pytest.mark.parametrize("name", fastwoe.__all__)
def test_every_public_name_resolves(name):
    assert getattr(fastwoe, name) is not None


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        _ = fastwoe.nope


def test_dir_lists_public_names():
    assert set(fastwoe.__all__) <= set(dir(fastwoe))
