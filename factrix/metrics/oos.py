"""Out-of-sample (OOS) persistence analysis for any time-indexed series.

Aggregation: time-series only, IS/OOS window split on a 1-D series;
descriptive decay diagnostic (no formal H₀).

Input: DataFrame with ``date, value`` (IC series, CAAR series, spread series).
Output: MetricOutput with ``value`` = median survival ratio + sign-flip / status
detail in ``metadata``.

This tool is agnostic to what the series represents — it only knows
about IS/OOS splits on a time-indexed numeric sequence.

Matrix-row: multi_split_oos_decay | (*, CONTINUOUS, *, TIMESERIES) | TS-only | no formal H₀ | _short_circuit_output
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from factrix._types import EPSILON, MIN_OOS_PERIODS, MetricOutput
from factrix.metrics._helpers import _short_circuit_output

GateStatus = Literal["PASS", "VETOED"]


@dataclass
class SplitDetail:
    """Per-split IS/OOS calculation intermediate.

    Not part of the public return API (which is ``MetricOutput``); surfaced
    as a typed helper for callers that want typed access to individual splits
    without round-tripping through ``metadata["per_split"]`` dicts.

    ``survival_ratio = |mean_OOS| / |mean_IS|`` — 1.0 = OOS matches IS,
    0.0 = signal vanished out of sample. Higher is better.
    """

    is_ratio: float
    mean_is: float
    mean_oos: float
    survival_ratio: float
    sign_flipped: bool

    @property
    def oos_ratio(self) -> float:
        return 1 - self.is_ratio

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "is_ratio": self.is_ratio,
            "mean_is": self.mean_is,
            "mean_oos": self.mean_oos,
            "survival_ratio": self.survival_ratio,
            "sign_flipped": self.sign_flipped,
        }


def multi_split_oos_decay(
    series: pl.DataFrame,
    value_col: str = "value",
    splits: list[tuple[float, float]] | None = None,
    survival_threshold: float = 0.5,
) -> MetricOutput:
    """Multi-split OOS survival analysis with sign-flip detection.

    For each split point, divides the series into IS and OOS portions,
    computes ``|mean_OOS| / |mean_IS|`` (the survival ratio), and checks
    for sign flips. The reported ratio is the **median** across splits.

    Args:
        series: DataFrame with ``date`` and ``value_col``, sorted by date.
        splits: List of (IS_fraction, OOS_fraction) tuples.
            Default: ``[(0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]``.
        survival_threshold: Minimum survival ratio for PASS (default 0.5).

    Returns:
        MetricOutput with:

        - ``name``: "oos_decay"
        - ``value``: median survival ratio across splits (0.0 on short-circuit)
        - ``stat``: None (descriptive statistic, not a hypothesis test)
        - ``metadata``:

            - ``sign_flipped`` (bool): any split had sign flip
            - ``status`` ("PASS" | "VETOED")
            - ``per_split`` (list[dict]): see ``SplitDetail.to_dict``
            - ``p_value`` (float): 1.0 (not a hypothesis test; conservative
              default so downstream BHY doesn't treat descriptive stats as
              significant by omission)
            - ``method`` (str): "multi-split OOS decay"
            - ``survival_threshold`` (float)
            - ``reason`` (str, short-circuit only): "insufficient_oos_periods"
              or "no_valid_splits"

    Notes:
        For each split fraction ``f``, partition the sorted series into
        IS (first ``f·n``) and OOS (remainder). Per-split survival ratio
        is ``s_f = |mean_OOS| / |mean_IS|``; reported headline is
        ``median_f s_f``. A split is flagged as ``sign_flipped`` when
        ``mean_IS`` and ``mean_OOS`` have opposite signs — any such split
        sets ``status = VETOED``.

        factrix reports the **median** across splits rather than mean:
        a single regime change landing inside one split distorts the
        mean disproportionately. Descriptive only — no formal H0 is
        attached and ``p_value`` is set to 1.0 so downstream BHY does
        not treat the diagnostic as a significant test.

    References:
        - McLean & Pontiff (2016): average OOS decay ~32%.
        - de Prado (2018): CPCV for robust train/test split.
    """
    if splits is None:
        splits = [(0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]

    sorted_series = series.sort("date")
    vals = sorted_series[value_col].drop_nulls()
    n = len(vals)

    if n < MIN_OOS_PERIODS * 2:
        return _short_circuit_output(
            "oos_decay",
            "insufficient_oos_periods",
            n_observed=n,
            min_required=MIN_OOS_PERIODS * 2,
            sign_flipped=False,
            status="VETOED",
            per_split=[],
            method="multi-split OOS decay",
            survival_threshold=survival_threshold,
        )

    split_details: list[SplitDetail] = []
    any_sign_flip = False

    for is_frac, _ in splits:
        split_idx = int(n * is_frac)
        if split_idx < MIN_OOS_PERIODS or (n - split_idx) < MIN_OOS_PERIODS:
            continue

        is_vals = vals[:split_idx]
        oos_vals = vals[split_idx:]

        mean_is = float(is_vals.mean())
        mean_oos = float(oos_vals.mean())

        sign_flip = (mean_is > 0 and mean_oos < 0) or (mean_is < 0 and mean_oos > 0)
        if sign_flip:
            any_sign_flip = True

        if abs(mean_is) < EPSILON:
            survival = 0.0
        else:
            survival = abs(mean_oos) / abs(mean_is)

        split_details.append(
            SplitDetail(
                is_ratio=is_frac,
                mean_is=mean_is,
                mean_oos=mean_oos,
                survival_ratio=survival,
                sign_flipped=sign_flip,
            )
        )

    if not split_details:
        return _short_circuit_output(
            "oos_decay",
            "no_valid_splits",
            sign_flipped=False,
            status="VETOED",
            per_split=[],
            method="multi-split OOS decay",
            survival_threshold=survival_threshold,
        )

    # WHY: 取中位數而非均值，對單一 regime change 落在某 split 點更穩健
    median_survival = float(np.median([d.survival_ratio for d in split_details]))

    # WHY: sign flip 任一 split 發生即 VETOED — IC 翻轉代表因子在 OOS 預測反了
    if any_sign_flip:
        status: GateStatus = "VETOED"
    elif median_survival >= survival_threshold:
        status = "PASS"
    else:
        status = "VETOED"

    return MetricOutput(
        name="oos_decay",
        value=median_survival,
        stat=None,
        significance="",
        metadata={
            "sign_flipped": any_sign_flip,
            "status": status,
            "per_split": [sd.to_dict() for sd in split_details],
            "p_value": 1.0,
            "method": "multi-split OOS decay",
            "survival_threshold": survival_threshold,
            "n_splits": len(split_details),
        },
    )
