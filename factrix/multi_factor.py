"""Public ``multi_factor`` namespace — collection-level FDR control.

The original per-metric procedures return one result container per metric
label. Cross-metric procedures return one container whose survivor unit is
explicitly either a factor-by-metric hypothesis or a factor identity.

Cross-metric procedures return one result whose survivor unit is explicitly
either a factor-by-metric hypothesis or a factor identity.
"""

from factrix._multi_factor import (
    BhyResult,
    CrossMetricBhyResult,
    CrossMetricPartialConjunctionResult,
    HierarchicalBhyResult,
    MetricHypothesis,
    PartialConjunctionResult,
    bhy,
    bhy_across_metrics,
    bhy_hierarchical,
    partial_conjunction,
    partial_conjunction_across_metrics,
)

__all__ = [
    "BhyResult",
    "CrossMetricBhyResult",
    "CrossMetricPartialConjunctionResult",
    "HierarchicalBhyResult",
    "MetricHypothesis",
    "PartialConjunctionResult",
    "bhy",
    "bhy_across_metrics",
    "bhy_hierarchical",
    "partial_conjunction",
    "partial_conjunction_across_metrics",
]
