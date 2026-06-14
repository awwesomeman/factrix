"""Public ``multi_factor`` namespace — collection-level FDR control.

Three procedure-paired result dataclasses (one per function) replace
the v0.13 unified ``Survivors``; each function returns
``dict[metric_name, *Result]`` keyed by ``MetricSpec.name``.
"""

from factrix._multi_factor import (
    BhyResult,
    HierarchicalBhyResult,
    PartialConjunctionResult,
    bhy,
    bhy_hierarchical,
    partial_conjunction,
)

__all__ = [
    "BhyResult",
    "HierarchicalBhyResult",
    "PartialConjunctionResult",
    "bhy",
    "bhy_hierarchical",
    "partial_conjunction",
]
