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
    FactorScope,
    InputShape,
)
from factrix._metric_index import SampleThreshold, cell
from factrix._results import MetricResult
from factrix._types import EPSILON, MIN_OOS_PERIODS_HARD
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _enforce_min_floor,
    _resolve_series_value_col,
    _short_circuit_output,
    _surface_null_drop,
)
from factrix.metrics.ic import compute_ic

__all__ = [
    "oos_decay",
]

GateStatus = Literal["PASS", "VETOED"]

# Minimum observations each side of the split must carry for its mean to be
# a window statistic rather than a single point.
_MIN_SPLIT_OBS = 2


@metric(
    cell=cell(
        FactorScope.INDIVIDUAL, FactorDensity.DENSE, structure=DataStructure.PANEL
    ),
    aggregation=Aggregation.TS_ONLY,
    slice_boundary_sensitive=True,
    input_shape=InputShape.SERIES,
    requires={"series": compute_ic},
    sample_threshold=SampleThreshold(min_periods=MIN_OOS_PERIODS_HARD * 2),
)
def oos_decay(
    series: pl.DataFrame,
    value_col: str = "value",
    is_ratio: float = 0.7,
    survival_threshold: float = 0.5,
    expected_warnings: tuple[str, ...] = (),
) -> MetricResult:
    """Single-split out-of-sample (OOS) survival ratio with sign-flip detection.

    Splits the sorted series at ``is_ratio`` (IS = first ``is_ratio * n``
    rows, OOS = remainder), computes ``|mean_OOS| / |mean_IS|`` (the
    survival ratio), and checks for an IS/OOS sign flip.

    Args:
        series: DataFrame with ``date`` and ``value_col``, sorted by date.
        value_col: Numeric column to evaluate.
        is_ratio: Fraction of the series allocated to IS (default ``0.7``).
            Must lie strictly inside ``(0, 1)``.
        survival_threshold: Minimum survival ratio for ``status="PASS"``
            (default ``0.5``).

    Raises:
        ValueError: ``is_ratio`` is not strictly inside ``(0, 1)``.

    Returns:
        MetricResult with:

        - ``value``: survival ratio (NaN on short-circuit)
        - ``stat``: ``None`` — descriptive only (no hypothesis test
        attached; a t-stat at ``MIN_OOS_PERIODS_HARD = 5`` would have power
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

        **Split validity.** ``is_ratio`` must be strictly inside ``(0, 1)``
        and is validated up front: ``is_ratio=1.0`` used to produce an empty
        OOS slice whose polars ``mean()`` is ``None``, and ``float(None)``
        then raised a bare ``TypeError`` from deep inside the metric. Beyond
        that, the ``min_periods`` floor bounds the *series length*, not the
        split, so an extreme ratio can still leave one side with fewer than
        two observations; that short-circuits with the usual
        ``reason="insufficient_oos_periods"`` rather than reporting a
        survival ratio computed from a single point.

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
        >>> result.name == ""
        True
    """
    if not 0.0 < is_ratio < 1.0:
        raise ValueError(
            f"is_ratio must be a fraction strictly inside (0, 1), got {is_ratio!r}. "
            "0 leaves no in-sample window and 1 leaves no out-of-sample window, "
            "so no survival ratio is defined."
        )

    value_col = _resolve_series_value_col(series, value_col)
    sorted_series = series.sort("date")
    vals = sorted_series[value_col].drop_nulls().drop_nans()
    n = len(vals)

    sc = _enforce_min_floor(
        oos_decay,
        "oos_decay",
        n,
        "insufficient_oos_periods",
        descriptive=True,
        sign_flipped=False,
        status="VETOED",
        is_ratio=is_ratio,
        survival_threshold=survival_threshold,
    )
    if sc is not None:
        return sc

    split_idx = int(n * is_ratio)

    # WHY: the min-periods floor bounds `n`, not the *split*. A lopsided
    # `is_ratio` (0.05, 0.98, ...) can still leave one side with 0 or 1
    # observation, where a mean is either undefined (polars returns None →
    # `float(None)` TypeError) or a single point masquerading as a window.
    if split_idx < _MIN_SPLIT_OBS or n - split_idx < _MIN_SPLIT_OBS:
        return _short_circuit_output(
            "oos_decay",
            "insufficient_oos_periods",
            n_obs=n,
            n_obs_axis="periods",
            descriptive=True,
            sign_flipped=False,
            status="VETOED",
            is_ratio=is_ratio,
            survival_threshold=survival_threshold,
        )

    is_vals = vals[:split_idx]
    oos_vals = vals[split_idx:]

    # Both slices carry >= _MIN_SPLIT_OBS observations, so polars mean()
    # returns a numeric.
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

    metadata: dict[str, object] = {
        "sign_flipped": sign_flipped,
        "status": status,
        "is_ratio": is_ratio,
        "mean_is": mean_is,
        "mean_oos": mean_oos,
        "survival_threshold": survival_threshold,
    }
    warning_codes: list[str] = []
    _surface_null_drop(
        n_periods_in=sorted_series.height,
        n_periods_out=n,
        drop_reason="null / NaN value observations in the series",
        metric_name="oos_decay",
        metadata=metadata,
        warning_codes=warning_codes,
        expected_warnings=expected_warnings,
    )
    return MetricResult(
        value=survival,
        n_obs=n,
        n_obs_axis="periods",
        stat=None,
        metadata=metadata,
        warning_codes=tuple(warning_codes),
    )
