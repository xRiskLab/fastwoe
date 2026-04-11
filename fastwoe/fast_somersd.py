"""Backward compatibility module for fast_somersd.

.. deprecated:: 0.1.7
    This module is maintained for backward compatibility only.
    All functionality has been moved to :mod:`fastwoe.metrics`.
    Import from ``fastwoe.metrics`` instead.
"""

import warnings

from .metrics import SomersDResult, somersd_pairwise, somersd_xy, somersd_yx

warnings.warn(
    "fastwoe.fast_somersd is deprecated. Import from fastwoe.metrics instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "SomersDResult",
    "somersd_yx",
    "somersd_xy",
    "somersd_pairwise",
]
