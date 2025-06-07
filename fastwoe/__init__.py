"""FastWoe: Fast Weight of Evidence encoding for categorical features.

This package provides efficient Weight of Evidence (WOE) encoding for categorical 
features with statistical confidence intervals and cardinality preprocessing.
"""

from .fastwoe import FastWoe, WoePreprocessor

__version__ = "0.1.0"
__author__ = "Denis Burakov"
__email__ = "your.email@example.com"

__all__ = ["FastWoe", "WoePreprocessor"] 