"""Out-of-sample (OOS) persistence analysis for any time-indexed series.

Input: DataFrame with ``date, value`` (IC series, CAAR series, spread series).
Output: multi-split decay ratio + sign flip detection.

This tool is agnostic to what the series represents — it only knows
about IS/OOS splits on a time-indexed numeric sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import polars as pl

from factorlib._types import EPSILON, MIN_OOS_PERIODS

GateStatus = Literal["PASS", "VETOED"]


@dataclass
class SplitDetail:
    """Detail for a single IS/OOS split.

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


@dataclass
class OOSResult:
    """Aggregated multi-split OOS analysis result.

    Attributes:
        survival_ratio: Median of per-split survival ratios
            (``|mean_OOS| / |mean_IS|``). Higher is better — 1.0 means
            OOS signal is as strong as IS.
        sign_flipped: True if any split has sign flip → VETOED.
        per_split: Per-split details.
        status: "PASS" or "VETOED".
    """

    survival_ratio: float
    sign_flipped: bool
    per_split: list[SplitDetail] = field(default_factory=list)
    status: GateStatus = "PASS"


def multi_split_oos_decay(
    series: pl.DataFrame,
    value_col: str = "value",
    splits: list[tuple[float, float]] | None = None,
    survival_threshold: float = 0.5,
) -> OOSResult:
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
        OOSResult with aggregated status.

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
        return OOSResult(
            survival_ratio=0.0,
            sign_flipped=False,
            status="VETOED",
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

        split_details.append(SplitDetail(
            is_ratio=is_frac,
            mean_is=mean_is,
            mean_oos=mean_oos,
            survival_ratio=survival,
            sign_flipped=sign_flip,
        ))

    if not split_details:
        return OOSResult(survival_ratio=0.0, sign_flipped=False, status="VETOED")

    # WHY: 取中位數而非均值，對單一 regime change 落在某 split 點更穩健
    median_survival = float(np.median([d.survival_ratio for d in split_details]))

    # WHY: sign flip 任一 split 發生即 VETOED — IC 翻轉代表因子在 OOS 預測反了
    if any_sign_flip:
        status = "VETOED"
    elif median_survival >= survival_threshold:
        status = "PASS"
    else:
        status = "VETOED"

    return OOSResult(
        survival_ratio=median_survival,
        sign_flipped=any_sign_flip,
        per_split=split_details,
        status=status,
    )
