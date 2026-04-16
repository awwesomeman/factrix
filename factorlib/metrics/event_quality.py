"""Per-event quality descriptive statistics for event signals.

All metrics operate on the ``signed_car`` (return x sign(factor)) of
individual events. They describe the quality and shape of per-event
outcomes — distinct from significance testing (caar.py) and path
analysis (mfe_mae.py).

Metrics:
    event_hit_rate — fraction of correct-direction events (binomial test)
    event_ic       — signal strength → return correlation (Spearman)
    profit_factor  — sum(gains) / sum(losses)
    event_skewness — skewness of signed_car distribution
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factorlib._types import EPSILON, MIN_EVENTS, MetricOutput
from factorlib._stats import (
    _p_value_from_z,
    _significance_marker,
)
from factorlib.metrics._helpers import _signed_car


def event_hit_rate(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> MetricOutput:
    """Fraction of events where signed abnormal return > 0.

    Uses binomial score test: H₀: p = 0.5 (random direction).
    z = (hits - n*p0) / sqrt(n*p0*(1-p0))

    Args:
        df: Panel with event signal and forward return.

    Returns:
        MetricOutput with value=hit_rate, stat=z from binomial test.
    """
    events = df.filter(pl.col(factor_col) != 0)

    n = len(events)
    if n < MIN_EVENTS:
        return MetricOutput(
            name="event_hit_rate", value=0.0, stat=0.0, significance="",
        )

    signed = _signed_car(events, factor_col, return_col)
    hits = int(np.sum(signed > 0))
    rate = hits / n

    z = (hits - n * 0.5) / (np.sqrt(n) * 0.5)
    p = _p_value_from_z(z)

    return MetricOutput(
        name="event_hit_rate",
        value=rate,
        stat=z,
        significance=_significance_marker(p),
        metadata={
            "n_events": n,
            "n_hits": hits,
            "p_value": p,
            "stat_type": "z",
            "h0": "p=0.5",
            "method": "binomial score test",
        },
    )


def event_ic(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> MetricOutput:
    """Signal strength → directional return correlation among events.

    Spearman correlation between ``|factor|`` and ``signed_car``
    (``return × sign(factor)``), computed only on event rows.

    Unlike standard IC (full cross-section per date), this measures
    whether signal **magnitude** predicts return magnitude among
    triggered events. Direction is already accounted for via sign().

    Only meaningful when signal values have magnitude variance
    (not all ±1). Profile auto-skips when variance is absent.

    Args:
        df: Panel with event signal and forward return.

    Returns:
        MetricOutput with value=Spearman rho, stat=z from Fisher transform.
    """
    from scipy import stats as sp_stats

    events = df.filter(pl.col(factor_col) != 0)
    n = len(events)

    if n < MIN_EVENTS:
        return MetricOutput(name="event_ic", value=0.0, stat=0.0, significance="")

    abs_signal = np.abs(events[factor_col].to_numpy())

    if np.ptp(abs_signal) < EPSILON:
        return MetricOutput(name="event_ic", value=0.0, stat=0.0, significance="")

    signed = _signed_car(events, factor_col, return_col)

    rho, p = sp_stats.spearmanr(abs_signal, signed)
    rho = float(rho)
    p = float(p)

    if abs(rho) < 1.0 - 1e-10 and n > 3:
        z = np.arctanh(rho) * np.sqrt(n - 3)
    else:
        z = 0.0

    return MetricOutput(
        name="event_ic",
        value=rho,
        stat=z,
        significance=_significance_marker(p),
        metadata={
            "n_events": n,
            "p_value": p,
            "stat_type": "z",
            "h0": "rho=0",
            "method": "Spearman rank correlation (|signal| vs signed_car)",
        },
    )


def profit_factor(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> MetricOutput:
    """sum(positive signed_car) / sum(negative signed_car).

    Per-event aggregate — no strategy assumptions. A profit factor > 1
    means gross gains exceed gross losses across all events.

    Args:
        df: Panel with event signal and forward return.

    Returns:
        MetricOutput with value=profit_factor.
    """
    events = df.filter(pl.col(factor_col) != 0)
    n = len(events)

    if n < MIN_EVENTS:
        return MetricOutput(name="profit_factor", value=0.0)

    signed = _signed_car(events, factor_col, return_col)

    gains = float(np.sum(signed[signed > 0]))
    losses = float(np.abs(np.sum(signed[signed < 0])))

    pf = gains / losses if losses > EPSILON else 0.0

    return MetricOutput(
        name="profit_factor",
        value=pf,
        metadata={
            "total_gains": gains,
            "total_losses": losses,
            "n_events": n,
            "n_wins": int(np.sum(signed > 0)),
            "n_losses": int(np.sum(signed < 0)),
        },
    )


def event_skewness(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> MetricOutput:
    """Skewness of signed_car distribution.

    Positive skew = occasional large gains, frequent small losses
    (desirable for event strategies). Uses scipy's Fisher skewness
    (bias-corrected).

    Also tests H₀: skewness = 0 via D'Agostino's skew test.

    Args:
        df: Panel with event signal and forward return.

    Returns:
        MetricOutput with value=skewness, stat=z from D'Agostino test.
    """
    from scipy import stats as sp_stats

    events = df.filter(pl.col(factor_col) != 0)
    n = len(events)

    if n < MIN_EVENTS:
        return MetricOutput(name="event_skewness", value=0.0)

    signed = _signed_car(events, factor_col, return_col)

    skew = float(sp_stats.skew(signed, bias=False))

    if n >= 20:
        z, p = sp_stats.skewtest(signed)
        z = float(z)
        p = float(p)
    else:
        z = None
        p = None

    return MetricOutput(
        name="event_skewness",
        value=skew,
        stat=z,
        significance=_significance_marker(p) if p is not None else None,
        metadata={
            "n_events": n,
            **({"p_value": p, "stat_type": "z", "h0": "skew=0",
                "method": "D'Agostino skew test"} if p is not None else {}),
        },
    )
