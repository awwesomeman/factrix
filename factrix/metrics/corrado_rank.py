"""Corrado nonparametric rank test on event abnormal returns.

Standalone metric in cell ``(*, SPARSE, *, PANEL)`` — not part of the
default profile. Available via
``from factrix.metrics import corrado_rank``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from factrix._axis import (
    Aggregation,
    FactorDensity,
)
from factrix._metric_index import SampleThreshold, cell
from factrix._results import MetricResult
from factrix._stats import _calc_t_stat
from factrix._types import EPSILON, MIN_EVENTS_HARD
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import _enforce_min_floor, _short_circuit_output

__all__ = [
    "corrado_rank",
]


@metric(
    # structure=None (event-axis): the rank test runs across events, so a single
    # name with enough events is a valid sample. Density stays SPARSE; the event
    # floor guards thin samples.
    cell=cell(None, FactorDensity.SPARSE, structure=None),
    aggregation=Aggregation.EVENT_TIME,
    sample_threshold=SampleThreshold(min_events=MIN_EVENTS_HARD),
)
def corrado_rank(
    data: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> MetricResult:
    r"""Corrado nonparametric rank test for event abnormal returns.

    The static event floor (sample_threshold=SampleThreshold(min_events=MIN_EVENTS_HARD)) gates the rank test on the count of non-zero (event) observations.

    A non-parametric alternative to the CAAR t-test. Robust to extreme
    returns, non-normal distributions, and cross-asset
    heteroscedasticity. Direction-adjusted for two-sided signals
    (extension of the original one-directional test).

    Formula:
        For each asset $i$, rank ``return`` across the full sample
        (event + non-event), transform to
        $U_{i,t} = \mathrm{rank} / (T+1) - 0.5$, and on event rows
        form $U_{\text{event,signed}} = U_{\text{event}} \cdot \mathrm{sign}(\text{factor})$.
        Test statistic
        $z = \mathrm{mean}(U_{\text{event,signed}}) / (\mathrm{std}(U_{\text{all}}) / \sqrt{N_{\text{events}}})$.
        ``p_value`` is **one-sided** (``H0: mean_u <= 0``): the direction
        adjustment already folds sign into $z$, so a factor that
        anti-predicts returns a negative $z$ rather than a small two-sided
        $p$ — mirroring :func:`~factrix.metrics.directional_hit_rate.directional_hit_rate`.

    Args:
        data: Full panel with ``date, asset_id, factor, forward_return``.
            Must include non-event rows for ranking.

    Returns:
        MetricResult with value=mean rank deviation, stat=z.

    Notes:
        factrix uses the **pooled** std of ``U_all`` across all
        ``(asset, date)`` cells in the denominator, instead of the
        time-series std of the cross-sectional mean used by Corrado
        (1989) eq. (5). The two coincide under iid ranks but diverge
        when event-date clustering is present; treat this as a
        robustness screen against parametric BMP / CAAR rather than a
        substitute for a reference event-study package when strict
        size control matters.

        Short-circuits to ``MetricResult`` with
        ``metadata["reason"]="insufficient_events"`` when
        ``N_events < MIN_EVENTS_HARD``, and
        ``"degenerate_rank_variance"`` when ``std(U_all) < EPSILON``.

    References:
        - [Corrado (1989)][corrado-1989]. "A Nonparametric Test for
          Abnormal Security-price Performance in Event Studies."
          *Journal of Financial Economics* 23(2), 385–395. The
          nonparametric rank test factrix implements with a pooled-std
          denominator simplification.
        - [Corrado & Zivney (1992)][corrado-zivney-1992]. "The
          Specification and Power of the Sign Test in Event Study
          Hypothesis Tests Using Daily Stock Returns." *Journal of
          Financial and Quantitative Analysis* 27(3), 465–478. Source
          of the direction-adjustment idea applied to two-sided signals.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.corrado_rank import corrado_rank
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_event_panel(n_assets=50, n_dates=400, seed=0),
        ...     forward_periods=5,
        ... )
        >>> result = corrado_rank(panel)
        >>> result.name == ""
        True
    """
    ranked = data.with_columns(
        (
            pl.col(return_col).rank(method="average").over("asset_id")
            / (pl.col(return_col).count().over("asset_id") + 1)
            - 0.5
        ).alias("_rank_u")
    )

    events = ranked.filter(pl.col(factor_col) != 0)
    n_events = len(events)

    sc = _enforce_min_floor(
        corrado_rank, "corrado_rank", n_events, "insufficient_events", axis="events"
    )
    if sc is not None:
        return sc

    u_event = events["_rank_u"].to_numpy() * np.sign(events[factor_col].to_numpy())

    u_all = ranked["_rank_u"].to_numpy()
    std_u = float(np.std(u_all))

    if std_u < EPSILON:
        return _short_circuit_output(
            "corrado_rank",
            "degenerate_rank_variance",
            std_u=std_u,
        )

    mean_u = float(np.mean(u_event))
    z = _calc_t_stat(mean_u, std_u, n_events)
    # One-sided: u_event is already direction-adjusted by sign(factor), so
    # z > 0 signals genuine directional skill and z < 0 signals a factor that
    # anti-predicts — a two-sided p would read the latter as "significant".
    p = float(sp_stats.norm.sf(z))

    return MetricResult(
        p_value=p,
        value=mean_u,
        n_obs=n_events,
        n_obs_axis="events",
        stat=z,
        metadata={
            "n_events": n_events,
            "n_total_obs": len(ranked),
            "stat_type": "z",
            "h0": "mu_rank<=0",
            "method": "Corrado (1989) rank test",
        },
    )
