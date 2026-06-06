"""Corrado nonparametric rank test on event abnormal returns.

Standalone metric in cell ``(*, SPARSE, *, PANEL)`` — not part of the
default profile. Available via
``from factrix.metrics import corrado_rank``.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    SEMethod,
    TestMethod,
)
from factrix._metric_index import SampleThreshold, cell
from factrix._results import MetricResult
from factrix._stats import _calc_t_stat, _p_value_from_z
from factrix._types import EPSILON, MIN_EVENTS_HARD
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import _short_circuit_output

__all__ = [
    "corrado_rank",
]


@metric(
    cell=cell(None, FactorDensity.SPARSE, structure=DataStructure.PANEL),
    aggregation=Aggregation.EVENT_TIME,
    test_method=TestMethod.RANK,
    se_method=SEMethod.BUILT_IN,
    sample_threshold=SampleThreshold(),
)
def corrado_rank(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> MetricResult:
    r"""Corrado nonparametric rank test for event abnormal returns.

    No static panel-shape thresholds are declared (sample_threshold=SampleThreshold()) because the minimum required periods depend dynamically on event occurrence count (which is factor-context-dependent).

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

    Args:
        df: Full panel with ``date, asset_id, factor, forward_return``.
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
    ranked = df.with_columns(
        (
            pl.col(return_col).rank(method="average").over("asset_id")
            / (pl.col(return_col).count().over("asset_id") + 1)
            - 0.5
        ).alias("_rank_u")
    )

    events = ranked.filter(pl.col(factor_col) != 0)
    n_events = len(events)

    if n_events < MIN_EVENTS_HARD:
        return _short_circuit_output(
            "corrado_rank",
            "insufficient_events",
            n_obs=n_events,
            min_required=MIN_EVENTS_HARD,
        )

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
    p = _p_value_from_z(z)

    return MetricResult(
        p_value=p,
        value=mean_u,
        stat=z,
        metadata={
            "n_events": n_events,
            "n_total_obs": len(ranked),
            "p_value": p,
            "stat_type": "z",
            "h0": "mu_rank=0",
            "method": "Corrado (1989) rank test",
        },
    )
