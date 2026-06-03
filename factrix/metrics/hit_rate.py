"""Hit rate computation for any time-indexed series.

Notes:
    **Pipeline.** Time-series only, sampled non-overlapping on a 1-D
    series; binomial test against `p = 0.5`.

    **Input.** DataFrame with ``date, value`` or a 1-D array.

    **Output.** Proportion of periods where the value satisfies a
    condition (default: value > 0).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    SEMethod,
    TestMethod,
)
from factrix._metric_index import MetricSpec, cell
from factrix._results import MetricResult
from factrix._stats import (
    _BINOMIAL_EXACT_CUTOFF,
    _binomial_test_method_name,
    _binomial_two_sided_p,
)
from factrix._types import MIN_ASSETS_PER_DATE_IC
from factrix.metrics._helpers import _sample_non_overlapping, _short_circuit_output

__metric_specs__ = (
    MetricSpec(
        name="hit_rate",
        cell=cell(None, FactorDensity.DENSE, structure=DataStructure.TIMESERIES),
        aggregation=Aggregation.TS_ONLY,
        test_method=TestMethod.BINOMIAL,
        se_method=SEMethod.BUILT_IN,
    ),
)

__all__ = [
    "hit_rate",
]

# Slice-test contract (#153 §5): hit_rate operates on a
# pre-aggregated per-date series (no cross-section bucket pass), so
# slice tests skip the `n_groups` downscale step. Per-date minimum
# (if any) is the responsibility of the upstream metric that produced
# the series.
min_assets_per_group: int | None = None


def per_date_series(series: pl.DataFrame) -> pl.DataFrame:
    """Return ``(date, value)`` per-date hit indicator series.

    Casts ``value > 0`` to ``Float64`` per date. Caller is expected to
    rename a non-default value column upstream; the slice-test
    capability contract takes no kwargs. Consumed by
    ``slice_pairwise_test`` / ``slice_joint_test`` (#176) via
    ``factrix.metrics._metric_capabilities.resolve_per_date_series``.
    """
    return series.select(
        [
            pl.col("date"),
            (pl.col("value") > 0).cast(pl.Float64).alias("value"),
        ]
    ).drop_nulls()


def hit_rate(
    series: pl.DataFrame,
    value_col: str = "value",
    forward_periods: int = 5,
) -> MetricResult:
    """Hit rate = proportion of periods where value > 0.

    Args:
        series: DataFrame with ``date`` and ``value_col``.
        forward_periods: Sampling interval for non-overlapping dates.

    Returns:
        MetricResult with value = hit rate (0.0-1.0).

    Notes:
        ``rate = (#{t : value_t > 0}) / n`` on a non-overlapping subsample
        at stride ``forward_periods``. Two-sided binomial test against
        ``H0: p = 0.5``: exact binomial below ``_BINOMIAL_EXACT_CUTOFF``,
        normal-approximation z-test ``(rate - 0.5) sqrt(n) / 0.5`` above.

        factrix reports the actual statistic (hits or z) consistent with
        the test branch taken, so a reader cannot mistake an exact-binomial
        p for a Gaussian z. Non-overlap stride mirrors the information coefficient (IC) pipeline so
        autocorrelation from overlapping forward returns does not leak in.

    References:
        [Hansen-Hodrick 1980][hansen-hodrick-1980]: overlapping-return
        autocorrelation horizon motivating the non-overlap stride.

    Examples:
        Hit rate of a per-date IC series produced by
        :func:`~factrix.metrics.ic.compute_ic`:

        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.ic import compute_ic
        >>> from factrix.metrics.hit_rate import hit_rate
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, seed=0),
        ...     forward_periods=5,
        ... )
        >>> series = compute_ic(panel)["factor"].rename({"ic": "value"}).select("date", "value")
        >>> result = hit_rate(series, forward_periods=5)
        >>> result.spec is None
        True
    """
    sampled = _sample_non_overlapping(series, forward_periods)
    vals = sampled[value_col].drop_nulls()

    n = len(vals)
    if n < MIN_ASSETS_PER_DATE_IC:
        return _short_circuit_output(
            "hit_rate",
            "insufficient_hit_rate_samples",
            n_obs=n,
            min_required=MIN_ASSETS_PER_DATE_IC,
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

    return MetricResult(
        p=p,
        value=rate,
        stat=stat,
        metadata={
            "n_hits": hits,
            "n_total": n,
            "p_value": p,
            "stat_type": stat_type,
            "h0": "p=0.5",
            "method": _binomial_test_method_name(n),
        },
    )
