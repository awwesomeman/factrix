"""Metric registry and scoring utilities (map_linear, adaptive_weight)."""

import math
from typing import Callable

# ---------------------------------------------------------------------------
# Metric Registry
# ---------------------------------------------------------------------------

METRIC_REGISTRY: dict[str, Callable] = {}


def register(name: str):
    def decorator(fn: Callable):
        METRIC_REGISTRY[name] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Scoring Utilities
# ---------------------------------------------------------------------------

def map_linear(value: float, min_val: float, max_val: float) -> float:
    if max_val == min_val:
        return 50.0
    score = (value - min_val) / (max_val - min_val) * 100
    return float(max(0.0, min(100.0, score)))


def adaptive_weight(w_base: float, t_stat: float | None, tau: float = 2.0, k: float = 2.0) -> float:
    """Scale metric weight by statistical significance via sigmoid on t-stat.

    w_adjusted = w_base * sigmoid(k * (|t| - tau))
    Returns w_base unchanged if t_stat is None.
    """
    if t_stat is None:
        return w_base
    return w_base * (1.0 / (1.0 + math.exp(-k * (abs(t_stat) - tau))))
