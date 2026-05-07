"""Preprocessing helpers for attaching ``forward_return`` to a raw panel.

The public entry point is :func:`compute_forward_return`: given a raw
``(date, asset_id, price)`` panel, it returns the same panel with a
``forward_return`` column attached, which is the canonical input to
:func:`factrix.evaluate`.
"""

from factrix.preprocess.returns import compute_forward_return

__all__ = ["compute_forward_return"]
