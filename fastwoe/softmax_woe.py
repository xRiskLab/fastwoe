"""Backward compatibility module for softmax_woe.

.. deprecated:: 0.3.2
    SoftmaxWoe moved to :mod:`fastwoe.fastwoe_softmax`. Import it from
    ``fastwoe`` (``from fastwoe import SoftmaxWoe``) or from
    ``fastwoe.fastwoe_softmax`` instead.
"""

import warnings

from ._binned import MISSING, Binner
from .fastwoe_softmax import SoftmaxWoe

warnings.warn(
    "fastwoe.softmax_woe is deprecated. Import from fastwoe or fastwoe.fastwoe_softmax instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Binner", "MISSING", "SoftmaxWoe"]
