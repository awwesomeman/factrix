"""Preprocessing helpers for shaping a raw panel before :func:`factrix.evaluate`.

The canonical entry point is :func:`compute_forward_return`: given a raw
``(date, asset_id, price)`` panel, it returns the same panel with a
``forward_return`` column attached, which is the canonical input to
:func:`factrix.evaluate`.

The surrounding helpers cover the rest of the documented preprocessing
pipeline and are independently usable on a canonical panel:

- :func:`winsorize_forward_return` — per-date percentile clip of returns.
- :func:`compute_abnormal_return` — cross-sectional de-meaning of returns.
- :func:`mad_winsorize` — per-date MAD-based factor outlier clipping.
- :func:`cross_sectional_zscore` — MAD-robust factor standardization.
- :func:`orthogonalize_factor` — residualize a factor against base factors.
"""

from factrix.preprocess.normalize import cross_sectional_zscore, mad_winsorize
from factrix.preprocess.orthogonalize import OrthogonalizeResult, orthogonalize_factor
from factrix.preprocess.returns import (
    compute_abnormal_return,
    compute_forward_return,
    winsorize_forward_return,
)

__all__ = [
    "OrthogonalizeResult",
    "compute_abnormal_return",
    "compute_forward_return",
    "cross_sectional_zscore",
    "mad_winsorize",
    "orthogonalize_factor",
    "winsorize_forward_return",
]
