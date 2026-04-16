"""Group splitting utility for sub-universe / time-period / factor comparison.

Splits a DataFrame by arbitrary filter conditions and adjusts PipelineConfig
(n_groups) based on each group's universe size.

When ``unify_n_groups=True`` (default), all groups use the same n_groups
(the minimum across groups), ensuring apple-to-apple quantile comparisons.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import polars as pl

from factorlib.config import PipelineConfig
from factorlib.metrics._helpers import _median_universe_size

logger = logging.getLogger(__name__)

# N-aware defaults: (min_N, recommended_n_groups)
_N_GROUP_TIERS = [
    (1000, 10),
    (200, 5),
    (0, 3),
]


def _recommend_n_groups(median_n: int) -> int:
    for min_n, groups in _N_GROUP_TIERS:
        if median_n >= min_n:
            return groups
    return 3


def split_by_group(
    df: pl.DataFrame,
    definitions: dict[str, pl.Expr],
    base_config: PipelineConfig,
    auto_n_groups: bool = True,
    unify_n_groups: bool = True,
) -> dict[str, tuple[pl.DataFrame, PipelineConfig]]:
    """Split DataFrame by filter conditions with N-aware config adjustment.

    Args:
        df: Preprocessed panel with ``date, asset_id, ...``.
        definitions: Mapping of group name → Polars filter expression.
        base_config: Base pipeline configuration.
        auto_n_groups: If True (default), adjust ``n_groups`` per group
            based on median assets per date.
        unify_n_groups: If True (default), all groups use the same n_groups
            (the minimum recommended across all groups). This ensures
            quantile comparisons are apple-to-apple. Set to False to let
            each group use its own optimal n_groups independently.

    Returns:
        Mapping of group name → (filtered DataFrame, adjusted PipelineConfig).
    """
    # WHY: first pass collects per-group data and recommended n_groups;
    # second pass applies the unified minimum (if unify_n_groups=True).
    per_group: dict[str, tuple[pl.DataFrame, int, int]] = {}

    for name, expr in definitions.items():
        filtered = df.filter(expr)

        if filtered.is_empty():
            logger.warning("split_by_group: group '%s' is empty after filtering", name)
            continue

        median_n = _median_universe_size(filtered)
        recommended = _recommend_n_groups(median_n)
        per_group[name] = (filtered, median_n, recommended)

    if not per_group:
        return {}

    # Determine the n_groups each group will use
    if auto_n_groups:
        if unify_n_groups:
            per_group_rec = {n: rec for n, (_, _, rec) in per_group.items()}
            unified = min(per_group_rec.values())
            if unified != base_config.n_groups:
                bottleneck = [n for n, r in per_group_rec.items() if r == unified]
                logger.warning(
                    "split_by_group: n_groups reduced %d → %d to unify "
                    "across groups (bottleneck: %s, per-group recommended: %s). "
                    "Pass unify_n_groups=False to keep independent n_groups.",
                    base_config.n_groups, unified,
                    bottleneck, per_group_rec,
                )
        else:
            unified = None  # each group uses its own
    else:
        unified = None

    # Warn about groups too small for stable quantile analysis
    for name, (_, median_n, _) in per_group.items():
        if median_n < 50:
            logger.warning(
                "split_by_group: group '%s' has median N=%d (< 50) "
                "— quantile results may be unstable",
                name, median_n,
            )

    result: dict[str, tuple[pl.DataFrame, PipelineConfig]] = {}
    for name, (filtered, median_n, recommended) in per_group.items():
        if auto_n_groups:
            target = unified if unified is not None else recommended
            if target != base_config.n_groups:
                config = replace(base_config, n_groups=target)
            else:
                config = base_config
        else:
            config = base_config
            if base_config.n_groups > recommended:
                logger.warning(
                    "split_by_group: group '%s' median N=%d, "
                    "n_groups=%d may be too high (recommended: %d)",
                    name, median_n, base_config.n_groups, recommended,
                )

        result[name] = (filtered, config)

    return result
