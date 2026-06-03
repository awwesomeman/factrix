"""Out-of-sample (OOS) persistence analysis for any time-indexed series.

This tool is agnostic to what the series represents — it only knows
about a single IS/OOS split on a time-indexed numeric sequence.

Notes:
    **Pipeline.** Time-series only, single IS/OOS window split on a 1-D
    series; descriptive decay diagnostic (no formal H_0).

    **Input.** DataFrame with ``date, value`` (IC series, CAAR series,
    spread series).

    **Output.** MetricResult with ``value`` = survival ratio +
    sign-flip / status detail in ``metadata``.
"""

from __future__ import annotations

from typing import Literal

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
from factrix._types import EPSILON, MIN_OOS_PERIODS
from factrix.metrics._helpers import _short_circuit_output

__metric_specs__ = (
    MetricSpec(
        name="oos_decay",
        cell=cell(None, FactorDensity.DENSE, structure=DataStructure.TIMESERIES),
        aggregation=Aggregation.TS_ONLY,
        test_method=TestMethod.DESCRIPTIVE,
        se_method=SEMethod.NONE,
    ),
)

__all__ = [
    "oos_decay",
]

GateStatus = Literal["PASS", "VETOED"]


def oos_decay(
    series: pl.DataFrame,
    value_col: str = "value",
    is_ratio: float = 0.7,
    survival_threshold: float = 0.5,
) -> MetricResult:
    """Single-split out-of-sample (OOS) survival ratio with sign-flip detection.

    Splits the sorted series at ``is_ratio`` (IS = first ``is_ratio * n``
    rows, OOS = remainder), computes ``|mean_OOS| / |mean_IS|`` (the
    survival ratio), and checks for an IS/OOS sign flip.

    Args:
        series: DataFrame with ``date`` and ``value_col``, sorted by date.
        value_col: Numeric column to evaluate.
        is_ratio: Fraction of the series allocated to IS (default ``0.7``).
        survival_threshold: Minimum survival ratio for ``status="PASS"``
            (default ``0.5``).

    Returns:
        MetricResult with:

        - ``value``: survival ratio (NaN on short-circuit)
        - ``stat``: ``None`` — descriptive only (no hypothesis test
          attached; a t-stat at ``MIN_OOS_PERIODS = 5`` would have power
          ~ 0 and would invite mis-reading the diagnostic as a
          significance test)
        - ``metadata``:

            - ``sign_flipped`` (bool)
            - ``status`` (``"PASS"`` | ``"VETOED"``)
            - ``is_ratio`` (float)
            - ``mean_is`` (float)
            - ``mean_oos`` (float)
            - ``survival_threshold`` (float)
            - ``reason`` (str, short-circuit only):
              ``"insufficient_oos_periods"``

    Notes:
        For multi-fraction sweeps, call ``oos_decay`` per fraction and
        aggregate on the caller side::

            results = {f: oos_decay(series, is_ratio=f) for f in (0.6, 0.7, 0.8)}
            median = statistics.median(r.value for r in results.values())

        Descriptive only — no ``p_value`` is emitted.

    References:
        - [McLean-Pontiff (2016)][mclean-pontiff-2016]: post-publication
          returns ~58% lower than in-sample, with ~32% of that drop
          attributable to publication itself (the remaining ~26% is the
          pure out-of-sample decay).
        - [Lopez-de-Prado (2018)][lopez-de-prado-2018]: CPCV for robust
          train/test split.

    Examples:
        Survival on a per-date information coefficient (IC) series from
        :func:`~factrix.metrics.ic.compute_ic`:

        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.ic import compute_ic
        >>> from factrix.metrics.oos_decay import oos_decay
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=240, seed=0),
        ...     forward_periods=5,
        ... )
        >>> series = compute_ic(panel)["factor"].rename({"ic": "value"}).select("date", "value")
        >>> result = oos_decay(series)
        >>> result.spec is None
        True
    """
    sorted_series = series.sort("date")
    vals = sorted_series[value_col].drop_nulls()
    n = len(vals)

    # Need MIN_OOS_PERIODS in both IS and OOS halves for a meaningful ratio.
    if n < MIN_OOS_PERIODS * 2:
        return _short_circuit_output(
            "oos_decay",
            "insufficient_oos_periods",
            n_obs=n,
            descriptive=True,
            min_required=MIN_OOS_PERIODS * 2,
            sign_flipped=False,
            status="VETOED",
            is_ratio=is_ratio,
            survival_threshold=survival_threshold,
        )

    split_idx = int(n * is_ratio)
    is_vals = vals[:split_idx]
    oos_vals = vals[split_idx:]

    # `n >= MIN_OOS_PERIODS * 2` and `0 < is_ratio < 1` guarantee both
    # slices are non-empty, so polars mean() returns a numeric.
    mean_is = float(is_vals.mean())  # type: ignore[arg-type]
    mean_oos = float(oos_vals.mean())  # type: ignore[arg-type]

    sign_flipped = (mean_is > 0 and mean_oos < 0) or (mean_is < 0 and mean_oos > 0)
    survival = 0.0 if abs(mean_is) < EPSILON else abs(mean_oos) / abs(mean_is)

    if sign_flipped:
        status: GateStatus = "VETOED"
    elif survival >= survival_threshold:
        status = "PASS"
    else:
        status = "VETOED"

    return MetricResult(
        value=survival,
        stat=None,
        metadata={
            "sign_flipped": sign_flipped,
            "status": status,
            "is_ratio": is_ratio,
            "mean_is": mean_is,
            "mean_oos": mean_oos,
            "survival_threshold": survival_threshold,
        },
    )
