"""Timing-facet metrics (placeholder for future implementation)."""

import logging

import polars as pl

from factorlib.scoring.registry import register

logger = logging.getLogger(__name__)


@register("Hit_Rate")
def calc_hit_rate(df: pl.DataFrame, **kwargs) -> float | None:
    """Placeholder: fraction of periods with correct directional prediction."""
    return None


@register("Profit_Factor")
def calc_profit_factor(df: pl.DataFrame, **kwargs) -> float | None:
    """Placeholder: sum of gains / sum of losses."""
    return None


@register("CAR")
def calc_car(df: pl.DataFrame, **kwargs) -> float | None:
    """Placeholder: cumulative abnormal return in event window."""
    return None


@register("Strategy_Volatility")
def calc_strategy_volatility(df: pl.DataFrame, **kwargs) -> float | None:
    """Placeholder: annualized strategy return volatility."""
    return None
