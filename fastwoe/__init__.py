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

from .display import StyledDataFrame, iv_styled, style_iv_analysis, style_woe_mapping, styled
from .fastwoe import FastWoe, WoePreprocessor
from .interpret_fastwoe import WeightOfEvidence

# Optional plotting functionality
try:
    from .plots import plot_performance, visualize_woe

    _HAS_PLOTTING = True
except ImportError:
    _HAS_PLOTTING = False

    def _plotting_not_available(*args, **kwargs):
        """Raise an ImportError if plotting functionality is not available."""
        raise ImportError(
            "Plotting functionality requires matplotlib. "
            "Install it with: pip install fastwoe[plotting]"
        )

    plot_performance = _plotting_not_available
    visualize_woe = _plotting_not_available

__version__ = "0.1.6"
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
]
