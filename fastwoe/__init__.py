"""
FastWoe: Fast Weight of Evidence encoding for categorical features.

This package provides efficient Weight of Evidence (WOE) encoding for categorical
features with statistical confidence intervals and cardinality preprocessing.

Features:
- FastWoe: Fast WOE encoding with confidence intervals
- WoePreprocessor: Cardinality reduction for high-cardinality features
- WeightOfEvidence: Model interpretability tool with FastWoe
- plot_performance: CAP/Power curve visualization for binary and continuous targets
- visualize_woe: WOE feature visualization
- StyledDataFrame: Rich HTML rendering for Jupyter notebooks
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .display import StyledDataFrame, iv_styled, style_iv_analysis, style_woe_mapping, styled
    from .fastwoe import FastWoe, WoePreprocessor
    from .interpret_fastwoe import WeightOfEvidence
    from .metrics import gini_contributions
    from .plots import plot_performance, visualize_woe

__version__ = "0.1.9"
__author__ = "xRiskLab"
__email__ = "contact@xrisklab.ai"

__all__ = [
    "FastWoe",
    "WoePreprocessor",
    "WeightOfEvidence",
    "plot_performance",
    "visualize_woe",
    "StyledDataFrame",
    "style_iv_analysis",
    "style_woe_mapping",
    "styled",
    "iv_styled",
    "gini_contributions",
]

# Submodules load on first use, so `from fastwoe.metrics import ...` or
# `from fastwoe.plots import ...` does not pull in scikit-learn.
_LAZY = {
    "FastWoe": "fastwoe",
    "WoePreprocessor": "fastwoe",
    "WeightOfEvidence": "interpret_fastwoe",
    "plot_performance": "plots",
    "visualize_woe": "plots",
    "StyledDataFrame": "display",
    "style_iv_analysis": "display",
    "style_woe_mapping": "display",
    "styled": "display",
    "iv_styled": "display",
    "gini_contributions": "metrics",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        value = getattr(import_module(f".{_LAZY[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
