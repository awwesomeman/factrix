"""Small-N cross-sectional pair-ordering accuracy.

For allocation-style panels with only a handful of assets, quantile buckets can
be too coarse to answer the most basic ordering question: did the higher-scored
asset outperform the lower-scored asset on the same date? This module reports a
descriptive pooled pairwise ordering accuracy and deliberately does not attach a
naive binomial p-value over same-date pairs.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import polars as pl

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    FactorScope,
    InputShape,
)
from factrix._codes import WarningCode
from factrix._metric_index import SampleThreshold, cell
from factrix._results import MetricResult
from factrix._types import (
    EPSILON,
    MIN_PAIR_ACCURACY_PAIRS_HARD,
    MIN_PAIR_ACCURACY_PAIRS_WARN,
)
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _sample_non_overlapping,
    _short_circuit_output,
)

__all__ = [
    "directional_pair_accuracy",
]


@metric(
    cell=cell(
        FactorScope.INDIVIDUAL,
        FactorDensity.DENSE,
        structure=DataStructure.PANEL,
    ),
    aggregation=Aggregation.CS_THEN_TS,
    input_shape=InputShape.PANEL,
    # DataProperties.n_pairs is row count, not within-date asset-pair
    # combinations. Runtime guard below enforces the true comparable-pair floor.
    sample_threshold=SampleThreshold(),
)
def directional_pair_accuracy(
    data: pl.DataFrame,
    forward_periods: int = 5,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> MetricResult:
    """Pairwise ordering accuracy for small allocation universes.

    For each non-overlapping date, compare every pair of assets with
    pairwise-complete ``factor_col`` and ``return_col`` values. A pair is
    correct when the asset with the higher factor value also has the higher
    forward return. Factor ties and return ties are excluded from the accuracy
    denominator and counted in metadata.

    Args:
        data: Long panel with ``date``, ``asset_id``, ``factor_col`` and
            ``return_col``.
        forward_periods: Sampling stride for non-overlapping dates; match the
            forward-return horizon so overlapping windows do not dominate the
            per-date series.
        factor_col: Ranking column.
        return_col: Forward-return column.

    Returns:
        MetricResult with ``value`` equal to pooled correct comparable pairs
        divided by pooled comparable pairs. The unweighted mean of per-date
        accuracies is reported in metadata. ``p_value`` and ``stat`` are
        ``None``: same-date asset pairs share shocks and are not treated as
        independent Bernoulli trials.
    """
    if factor_col not in data.columns:
        return _short_circuit_output(
            "directional_pair_accuracy",
            "no_factor_column",
            missing_column=factor_col,
            descriptive=True,
        )
    if return_col not in data.columns:
        return _short_circuit_output(
            "directional_pair_accuracy",
            "no_return_column",
            missing_column=return_col,
            descriptive=True,
        )

    sampled = _sample_non_overlapping(data, forward_periods)
    rows_in = sampled.height
    paired = sampled.select(
        "date",
        "asset_id",
        pl.col(factor_col).alias("_factor"),
        pl.col(return_col).alias("_return"),
    ).filter(pl.col("_factor").is_not_null() & pl.col("_return").is_not_null())

    per_date_accuracy: list[float] = []
    pairs_per_period: list[int] = []
    n_raw_pairs = 0
    n_usable_pairs = 0
    n_correct_pairs = 0
    n_incorrect_pairs = 0
    factor_tie_pairs = 0
    return_tie_pairs = 0
    both_tie_pairs = 0

    for date_df in paired.sort("date").partition_by("date", maintain_order=True):
        n_assets = date_df.height
        if n_assets < 2:
            continue

        factor = date_df["_factor"].to_numpy().astype(np.float64)
        forward_return = date_df["_return"].to_numpy().astype(np.float64)
        i, j = np.triu_indices(n_assets, k=1)
        factor_diff = factor[i] - factor[j]
        return_diff = forward_return[i] - forward_return[j]

        factor_tie = np.isclose(factor_diff, 0.0, rtol=0.0, atol=EPSILON)
        return_tie = np.isclose(return_diff, 0.0, rtol=0.0, atol=EPSILON)
        usable = ~(factor_tie | return_tie)
        products = factor_diff[usable] * return_diff[usable]

        raw_pairs = len(factor_diff)
        correct = int(np.sum(products > 0.0))
        incorrect = int(np.sum(products < 0.0))
        usable_pairs = correct + incorrect

        n_raw_pairs += raw_pairs
        n_usable_pairs += usable_pairs
        n_correct_pairs += correct
        n_incorrect_pairs += incorrect
        factor_tie_pairs += int(np.sum(factor_tie))
        return_tie_pairs += int(np.sum(return_tie))
        both_tie_pairs += int(np.sum(factor_tie & return_tie))

        if usable_pairs > 0:
            per_date_accuracy.append(correct / usable_pairs)
            pairs_per_period.append(usable_pairs)

    if n_usable_pairs < MIN_PAIR_ACCURACY_PAIRS_HARD:
        return _short_circuit_output(
            "directional_pair_accuracy",
            "insufficient_ordering_pairs",
            n_obs=n_usable_pairs,
            n_obs_axis="pairs",
            descriptive=True,
            min_required=MIN_PAIR_ACCURACY_PAIRS_HARD,
            n_periods=len(per_date_accuracy),
            n_raw_pairs=n_raw_pairs,
            factor_tie_pairs=factor_tie_pairs,
            return_tie_pairs=return_tie_pairs,
            both_tie_pairs=both_tie_pairs,
            dropped_pairs=n_raw_pairs - n_usable_pairs,
            dropped_rows_null=rows_in - paired.height,
        )

    warning_codes: list[str] = []
    if n_usable_pairs < MIN_PAIR_ACCURACY_PAIRS_WARN:
        warnings.warn(
            f"directional_pair_accuracy: n_pairs={n_usable_pairs} below "
            f"MIN_PAIR_ACCURACY_PAIRS_WARN={MIN_PAIR_ACCURACY_PAIRS_WARN}; the "
            f"descriptive ordering accuracy is returned, but the comparable-pair "
            f"sample is thin. Read it as a fragile small-N diagnostic.",
            UserWarning,
            stacklevel=3,
        )
        warning_codes.append(WarningCode.FEW_ORDERING_PAIRS.value)

    pooled_accuracy = n_correct_pairs / n_usable_pairs
    mean_per_date_accuracy = float(np.mean(per_date_accuracy))
    min_pairs = min(pairs_per_period) if pairs_per_period else 0
    max_pairs = max(pairs_per_period) if pairs_per_period else 0
    mean_pairs = float(np.mean(pairs_per_period)) if pairs_per_period else math.nan

    metadata: dict[str, object] = {
        "method": "pooled pairwise ordering accuracy (descriptive; no p-value)",
        "n_pairs": n_usable_pairs,
        "n_raw_pairs": n_raw_pairs,
        "n_periods": len(per_date_accuracy),
        "n_correct_pairs": n_correct_pairs,
        "n_incorrect_pairs": n_incorrect_pairs,
        "factor_tie_pairs": factor_tie_pairs,
        "return_tie_pairs": return_tie_pairs,
        "both_tie_pairs": both_tie_pairs,
        "dropped_pairs": n_raw_pairs - n_usable_pairs,
        "dropped_rows_null": rows_in - paired.height,
        "pooled_accuracy": float(pooled_accuracy),
        "mean_per_date_accuracy": mean_per_date_accuracy,
        "mean_pairs_per_period": mean_pairs,
        "min_pairs_per_period": min_pairs,
        "max_pairs_per_period": max_pairs,
        "tie_epsilon": EPSILON,
    }
    return MetricResult(
        value=float(pooled_accuracy),
        p_value=None,
        n_obs=n_usable_pairs,
        n_obs_axis="pairs",
        stat=None,
        metadata=metadata,
        warning_codes=tuple(warning_codes),
    )
