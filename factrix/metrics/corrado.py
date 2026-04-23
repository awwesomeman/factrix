"""Corrado (1989) nonparametric rank test for event signals.

A non-parametric alternative to the CAAR t-test. Ranks abnormal returns
across the full sample (event + non-event periods) for each asset, then
tests whether event-period ranks deviate from their expected value.

Robust to extreme returns, non-normal distributions, and cross-asset
heteroscedasticity. Direction-adjusted for two-sided signals (extension
of the original one-directional test).

Standalone metric — not in the default profile. Available via:
    ``from factrix.metrics import corrado_rank_test``

References:
    Corrado (1989), "A nonparametric test for abnormal security-price
        performance in event studies"
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factrix._types import EPSILON, MIN_EVENTS, MetricOutput
from factrix.metrics._helpers import _short_circuit_output
from factrix._stats import _calc_t_stat, _p_value_from_z, _significance_marker


def corrado_rank_test(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> MetricOutput:
    """Corrado nonparametric rank test for event abnormal returns.

    For each asset:
        1. Rank ``return_col`` across the full time series (event + non-event).
        2. Transform ranks to U_it = rank / (T+1) - 0.5 (centered at 0).
        3. Extract U values at event dates.
    Across all event observations:
        4. z = mean(U_event × sign(factor)) / (std(U_all) / √N_events)

    Deviation from Corrado (1989) eq.(5):
        The paper computes the denominator as the time-series std of the
        **cross-sectional mean** of rank deviations across the combined
        estimation + event window. We use the **pooled std of U_all**
        across all (asset, date) cells instead — a simpler estimator
        that conflates asset-level and time-level dispersion. The two
        coincide under iid ranks but diverge when event-date clustering
        is present. Adequate for a robustness screen against parametric
        BMP / CAAR; not a substitute for a reference event-study package
        if strict size control matters.

    Args:
        df: Full panel with ``date, asset_id, factor, forward_return``.
            Must include non-event rows for ranking.

    Returns:
        MetricOutput with value=mean rank deviation, stat=z.

    References:
        Corrado (1989), "A Nonparametric Test for Abnormal Security-
        Price Performance in Event Studies."
    """
    ranked = df.with_columns(
        (
            pl.col(return_col).rank(method="average").over("asset_id")
            / (pl.col(return_col).count().over("asset_id") + 1)
            - 0.5
        ).alias("_rank_u")
    )

    events = ranked.filter(pl.col(factor_col) != 0)
    n_events = len(events)

    if n_events < MIN_EVENTS:
        return _short_circuit_output(
            "corrado_rank", "insufficient_events",
            n_observed=n_events, min_required=MIN_EVENTS,
        )

    u_event = (
        events["_rank_u"].to_numpy() * np.sign(events[factor_col].to_numpy())
    )

    u_all = ranked["_rank_u"].to_numpy()
    std_u = float(np.std(u_all))

    if std_u < EPSILON:
        return _short_circuit_output(
            "corrado_rank", "degenerate_rank_variance",
            std_u=std_u,
        )

    mean_u = float(np.mean(u_event))
    z = _calc_t_stat(mean_u, std_u, n_events)
    p = _p_value_from_z(z)

    return MetricOutput(
        name="corrado_rank",
        value=mean_u,
        stat=z,
        significance=_significance_marker(p),
        metadata={
            "n_events": n_events,
            "n_total_obs": len(ranked),
            "p_value": p,
            "stat_type": "z",
            "h0": "mu_rank=0",
            "method": "Corrado (1989) rank test",
        },
    )
