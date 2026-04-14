"""Group splitting utility for sub-universe / time-period / factor comparison.

Splits a DataFrame by arbitrary filter conditions and adjusts PipelineConfig
(n_groups) based on each group's universe size.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import polars as pl

from factorlib.gates.config import PipelineConfig
from factorlib.tools._helpers import median_universe_size

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
) -> dict[str, tuple[pl.DataFrame, PipelineConfig]]:
    """Split DataFrame by filter conditions with N-aware config adjustment.

    Args:
        df: Preprocessed panel with ``date, asset_id, ...``.
        definitions: Mapping of group name → Polars filter expression.
            Supports arbitrary combinations, e.g.
            ``(pl.col("market") == "上市") & (pl.col("industry") == "半導體")``.
        base_config: Base pipeline configuration. ``n_groups`` may be adjusted
            per group based on universe size.
        auto_n_groups: If True (default), automatically adjust ``n_groups``
            based on median assets per date. If False, use base_config as-is.

    Returns:
        Mapping of group name → (filtered DataFrame, adjusted PipelineConfig).
    """
    result: dict[str, tuple[pl.DataFrame, PipelineConfig]] = {}

    for name, expr in definitions.items():
        filtered = df.filter(expr)

        if filtered.is_empty():
            logger.warning("split_by_group: group '%s' is empty after filtering", name)
            continue

        median_n = median_universe_size(filtered)

        config = base_config
        recommended = _recommend_n_groups(median_n)

        if auto_n_groups:
            if recommended != base_config.n_groups:
                logger.info(
                    "split_by_group: group '%s' median N=%d, "
                    "adjusting n_groups %d → %d",
                    name, median_n, base_config.n_groups, recommended,
                )
                config = replace(base_config, n_groups=recommended)
        else:
            if base_config.n_groups > recommended:
                logger.warning(
                    "split_by_group: group '%s' median N=%d, "
                    "n_groups=%d may be too high (recommended: %d)",
                    name, median_n, base_config.n_groups, recommended,
                )

        result[name] = (filtered, config)

    return result
