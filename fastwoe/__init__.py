"""
FastWoe: Fast Weight of Evidence encoding for categorical features.

This package provides efficient Weight of Evidence (WOE) encoding for categorical
features with statistical confidence intervals and cardinality preprocessing.

Features:
- FastWoe: Fast WOE encoding with confidence intervals
- WoePreprocessor: Cardinality reduction for high-cardinality features
- WeightOfEvidence: Model interpretability tool with FastWoe
"""

from .fastwoe import FastWoe, WoePreprocessor
from .interpret_fastwoe import WeightOfEvidence

__version__ = "0.1.1.post2"
__author__ = "xRiskLab"
__email__ = "contact@xrisklab.ai"

__all__ = ["FastWoe", "WoePreprocessor", "WeightOfEvidence"]
