"""factrix.slicing — cross-slice dispatcher and inference verbs.

Public surface:

- :func:`by_slice` — partition a metric input by an existing column
  and apply the metric per slice (no cross-slice inference).
- :func:`slice_pairwise_test` — K(K-1)/2 cross-slice Wald contrasts
  with Holm / Romano-Wolf / Bonferroni adjusted p-values.
- :func:`slice_joint_test` — single-row omnibus Wald χ² that all
  slice means are equal.

These verbs are intentionally *not* hosted under ``factrix.metrics``:
that package is a structural registry where every public ``*.py`` is
a per-(scope, signal) cell metric. Slicing verbs are infrastructure
that consumes a metric callable, so they live in their own package.
"""

from __future__ import annotations

from factrix.slicing.dispatcher import by_slice
from factrix.slicing.inference import slice_joint_test, slice_pairwise_test
from factrix.slicing.result import SliceResult

__all__ = [
    "SliceResult",
    "by_slice",
    "slice_joint_test",
    "slice_pairwise_test",
]
