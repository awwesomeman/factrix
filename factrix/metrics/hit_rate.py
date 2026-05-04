"""Hit rate computation for any time-indexed series.

Aggregation: time-series only, sampled non-overlapping on a 1-D
series; binomial test against `p = 0.5`.

Input: DataFrame with ``date, value`` or a 1-D array.
Output: proportion of periods where the value satisfies a condition
(default: value > 0).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factrix._types import MIN_ASSETS_PER_DATE_IC, MetricOutput
from factrix.metrics._helpers import _sample_non_overlapping, _short_circuit_output
from factrix._stats import (
    _BINOMIAL_EXACT_CUTOFF,
    _binomial_test_method_name,
    _binomial_two_sided_p,
    _significance_marker,
)


def hit_rate(
    series: pl.DataFrame,
    value_col: str = "value",
    forward_periods: int = 5,
) -> MetricOutput:
    """Hit rate = proportion of periods where value > 0.

    Uses non-overlapping sampling to avoid autocorrelation.
    t-stat is from a binomial test: ``(p - 0.5) / sqrt(0.25 / n)``.

    Args:
        series: DataFrame with ``date`` and ``value_col``.
        forward_periods: Sampling interval for non-overlapping dates.

    Returns:
        MetricOutput with value = hit rate (0.0-1.0).
    """
    sampled = _sample_non_overlapping(series, forward_periods)
    vals = sampled[value_col].drop_nulls()

    n = len(vals)
    if n < MIN_ASSETS_PER_DATE_IC:
        return _short_circuit_output(
            "hit_rate", "insufficient_hit_rate_samples",
            n_observed=n, min_required=MIN_ASSETS_PER_DATE_IC,
        )

    hits = int((vals > 0).sum())
    rate = hits / n
    p = _binomial_two_sided_p(hits, n, p0=0.5)

    # stat / stat_type must reflect the test actually run, so a reader
    # never sees stat=z paired with an exact-binomial p (the z↔p normal
    # identity would silently break). Under the exact branch we publish
    # the hit count as the statistic and flag stat_type accordingly.
    if n < _BINOMIAL_EXACT_CUTOFF:
        stat: float = float(hits)
        stat_type = "binomial_hits"
    else:
        stat = float((rate - 0.5) * np.sqrt(n) / 0.5)
        stat_type = "z"

    return MetricOutput(
        name="hit_rate",
        value=rate,
        stat=stat,
        significance=_significance_marker(p),
        metadata={
            "n_hits": hits,
            "n_total": n,
            "p_value": p,
            "stat_type": stat_type,
            "h0": "p=0.5",
            "method": _binomial_test_method_name(n),
        },
    )
