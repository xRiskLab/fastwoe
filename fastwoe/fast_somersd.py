"""Backward compatibility module for fast_somersd.

This module is maintained for backward compatibility. All functionality
has been moved to metrics.py. Please import from metrics instead.
"""

from .metrics import SomersDResult, somersd_pairwise, somersd_xy, somersd_yx

__all__ = [
    "SomersDResult",
    "somersd_yx",
    "somersd_xy",
    "somersd_pairwise",
]
