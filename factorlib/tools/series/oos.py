"""Out-of-sample (OOS) persistence analysis for any time-indexed series.

Input: DataFrame with ``date, value`` (IC series, CAAR series, spread series).
Output: multi-split decay ratio + sign flip detection.

This tool is agnostic to what the series represents — it only knows
about IS/OOS splits on a time-indexed numeric sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import polars as pl

from factorlib.tools._typing import EPSILON, MIN_OOS_PERIODS

GateStatus = Literal["PASS", "VETOED"]


@dataclass
class SplitDetail:
    """Detail for a single IS/OOS split."""

    is_ratio: float
    mean_is: float
    mean_oos: float
    decay_ratio: float
    sign_flipped: bool

    @property
    def oos_ratio(self) -> float:
        return 1 - self.is_ratio


@dataclass
class OOSResult:
    """Aggregated multi-split OOS analysis result.

    Attributes:
        decay_ratio: Median of per-split decay ratios.
        sign_flipped: True if any split has sign flip → VETOED.
        per_split: Per-split details.
        status: "PASS" or "VETOED".
    """

    decay_ratio: float
    sign_flipped: bool
    per_split: list[SplitDetail] = field(default_factory=list)
    status: GateStatus = "PASS"


def multi_split_oos_decay(
    series: pl.DataFrame,
    value_col: str = "value",
    splits: list[tuple[float, float]] | None = None,
    decay_threshold: float = 0.5,
) -> OOSResult:
    """Multi-split OOS decay analysis with sign flip detection.

    For each split point, divides the series into IS and OOS portions,
    computes ``|mean_OOS| / |mean_IS|``, and checks for sign flips.
    The final decay ratio is the **median** across splits.

    Args:
        series: DataFrame with ``date`` and ``value_col``, sorted by date.
        splits: List of (IS_fraction, OOS_fraction) tuples.
            Default: ``[(0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]``.
        decay_threshold: Minimum decay ratio for PASS (default 0.5).

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
            decay_ratio=0.0,
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

        # Sign flip detection
        sign_flip = (mean_is > 0 and mean_oos < 0) or (mean_is < 0 and mean_oos > 0)
        if sign_flip:
            any_sign_flip = True

        # Decay ratio
        if abs(mean_is) < EPSILON:
            decay = 0.0
        else:
            decay = abs(mean_oos) / abs(mean_is)

        split_details.append(SplitDetail(
            is_ratio=is_frac,
            mean_is=mean_is,
            mean_oos=mean_oos,
            decay_ratio=decay,
            sign_flipped=sign_flip,
        ))

    if not split_details:
        return OOSResult(decay_ratio=0.0, sign_flipped=False, status="VETOED")

    # WHY: 取中位數而非均值，對單一 regime change 落在某 split 點更穩健
    decay_ratios = sorted(d.decay_ratio for d in split_details)
    median_decay = decay_ratios[len(decay_ratios) // 2]

    # WHY: sign flip 任一 split 發生即 VETOED — IC 翻轉代表因子在 OOS 預測反了
    if any_sign_flip:
        status = "VETOED"
    elif median_decay >= decay_threshold:
        status = "PASS"
    else:
        status = "VETOED"

    return OOSResult(
        decay_ratio=median_decay,
        sign_flipped=any_sign_flip,
        per_split=split_details,
        status=status,
    )
