"""Public ``multi_factor`` namespace — collection-level FDR control.

One procedure-paired result dataclass per function
(:class:`BhyResult`, :class:`PartialConjunctionResult`,
:class:`HierarchicalBhyResult`); each function returns
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
