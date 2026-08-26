"""Tradability metrics: Rank Turnover, Breakeven Cost, Net Spread.

Two flavours of turnover co-exist here, measuring different things:

- ``rank_turnover()`` — ``1 − mean(rank autocorrelation)``. Rank-stability
  diagnostic; responds to mid-rank reshuffling. **Not** a notional
  trading-fraction and should **not** be fed into ``breakeven_cost`` /
  ``net_spread``.
- ``notional_turnover()`` — fraction of top-and-bottom quantile members
  replaced per rebalance. Matches [Novy-Marx-Velikov (2016)][novy-marx-velikov-2016] τ; this is
  the quantity that drives bps trading cost for an equal-weight Q1/Qn
  long-short portfolio.

These are implementation-feasibility indicators, not factor quality
or significance tests.

Input for Rank Turnover: DataFrame with ``date, asset_id, factor``.
Input for Breakeven/Net Spread: pre-computed spread and turnover values.

Notes:
    **Pipeline.** Per-date turnover / cost diagnostics on
    quantile-group membership (cross-section step), then time-series
    mean; descriptive (no formal H₀).
"""

from __future__ import annotations

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
from factrix._errors import UserInputError
from factrix._metric_index import SampleThreshold, cell
from factrix._results import MetricResult
from factrix._types import (
    DDOF,
    DEFAULT_FORWARD_PERIODS,
    DEFAULT_N_GROUPS,
    EPSILON,
)
from factrix.metrics._base import MetricBase
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _assign_quantile_groups,
    _enforce_min_floor,
    _finite_expr,
    _median_universe_size,
    _sample_non_overlapping,
    _short_circuit_output,
)

_DOCS_TRADABILITY = "api/metrics/tradability"

_TR_CELL = cell(
    FactorScope.INDIVIDUAL, FactorDensity.DENSE, structure=DataStructure.PANEL
)
_TR_CS_PRIMITIVES = (
    "_sample_non_overlapping",
    "_short_circuit_output",
    "_assign_quantile_groups",
)

__all__ = [  # noqa: RUF022 (teaching order, see SSOT note)
    "notional_turnover",
    "rank_turnover",
    "breakeven_cost",
    "net_spread",
]


def _rank_turnover_min_dates(forward_periods: int) -> int:
    """Raw-date floor for ``rank_turnover``: the non-overlap pair stride ``h`` needs
    >= 3 sampled dates (>= 2 non-overlapping pairs so ``std(rho)`` is defined),
    i.e. >= ``2*h + 1`` raw dates (Hansen & Hodrick 1980).
    """
    return 2 * forward_periods + 1


def _rank_turnover_sample_threshold(self: MetricBase) -> SampleThreshold:
    """Dynamic periods floor for ``rank_turnover``, scaling with ``forward_periods``.
    Delegates to the same ``_rank_turnover_min_dates`` the in-body short-circuit
    reads, so the pre-flight and run-time floors agree.
    """
    return SampleThreshold(min_periods=_rank_turnover_min_dates(self.forward_periods))


@metric(
    cell=_TR_CELL,
    aggregation=Aggregation.TS_ONLY,
    slice_boundary_sensitive=True,
    sample_threshold=_rank_turnover_sample_threshold,
)
def rank_turnover(
    data: pl.DataFrame,
    factor_col: str = "factor",
    forward_periods: int = 1,
    quantile: float | None = None,
) -> MetricResult:
    r"""Factor rank-stability via non-overlapping rank autocorrelation.

    The periods floor is dynamic — the minimum date count is ``2*forward_periods + 1`` — so it is declared as a resolver (a callable sample_threshold) rather than a constant, letting inspect_data pre-flight it.

    $\text{rank_turnover} = 1 - \mathrm{mean}(\bar\rho)$ where $\bar\rho$ is the mean rank autocorrelation

    **What this measures.** Sensitivity of the *full* cross-section rank
    vector to reshuffling between ``t`` and ``t + forward_periods``. Mid-rank
    churn (names moving between e.g. Q4 ↔ Q5 in a 10-group split) counts
    even though those names carry zero weight in a Q1/Qn long-short
    portfolio. So this is a **rank-stability diagnostic**, *not* a notional
    trading-fraction.

    **When to use this vs ``notional_turnover``.** Feed a strategy-cost
    formula (breakeven_cost / net_spread) with ``notional_turnover``, not
    this function — the bps coefficients there assume ``turnover`` is the
    fraction of Q1/Qn positions replaced per rebalance, which ``1 − ρ``
    does not provide. Keep this metric for ranking-stability comparisons
    across factors.

    Rank autocorrelation is measured between dates ``t`` and ``t +
    forward_periods``, sub-sampled at stride ``forward_periods`` (phase-0)
    so each transition is a non-overlapping snapshot. This aligns the stability
    window with the forward-return horizon used elsewhere in the profile.

    Args:
        data: Panel with ``date, asset_id, factor``.
        factor_col: Name of the factor column. Defaults to ``"factor"``.
        forward_periods: Sampling stride in periods — should match the
            forward-return horizon the factor is being evaluated against.
            ``1`` reproduces the lag-1 behaviour.
        quantile: Optional tail filter in ``(0, 0.5)``. When set, restrict
            the Spearman ρ at each pair to assets whose rank at *either*
            endpoint lies in the top-q or bottom-q of that date's cross-
            section — i.e. the statistical region where the long-short
            spread is actually measured. Union (not intersection) so names
            entering or leaving the tail both register as turnover.

            Caveat: ρ on the tail union is NOT comparable to the
            unfiltered ρ — tail names are more persistent by construction,
            so the resulting turnover will typically be lower. Compare
            only against other tail-filtered estimates at the same q.

    Returns:
        MetricResult with ``value = 1 − mean(ρ)`` and metadata
        carrying ``mean_rank_autocorrelation``, ``std_rank_autocorrelation``,
        ``n_periods``, ``forward_periods``, ``quantile``, and
        ``n_cross_section_mean`` (mean assets-per-transition post-filter).

        The value lies in ``[0, 2]``, not ``[0, 1]``: ρ ranges over
        ``[-1, +1]``, so a perfectly stable ranking gives 0, an independent
        re-draw gives ≈1 (a white-noise factor measures just above or below
        it by sampling error), and a systematically *reversed* ranking gives
        up to 2. It is a rank-stability index, not a traded fraction — see
        ``notional_turnover`` for the fraction of positions replaced.

        ``n_obs`` counts the adjacent-period transitions the ρ's were
        measured over (``n_obs_axis="periods"``), not ``(date, asset)``
        pairs.

        ``std_rank_autocorrelation`` is the cross-transition sample std. Using
        ``std/√n_periods`` as an SE is a *lower bound*: consecutive transitions
        share one rank-vector endpoint (transition k and k+1 both involve
        ``rank @ t_{k·h}``), so the per-transition ρ's have weak positive
        dependence and the true SE is marginally larger. For publication
        grade inference, use a heteroskedasticity-and-autocorrelation-consistent (HAC) variance estimator.

    Notes:
        For each non-overlap pair $(t, t+h)$, compute
        $\rho_t = \mathrm{Spearman}(\mathrm{rank}_t, \mathrm{rank}_{t+h})$
        over assets present in both cross-sections;
        $\text{rank_turnover} = 1 - \mathrm{mean}_t \rho_t$. With the optional
        tail filter, $\rho_t$ is restricted to the union of top- and
        bottom-q assets at either endpoint.

        factrix exposes this metric as a **rank-stability diagnostic**
        only — it is not a notional turnover and should not be fed into
        ``breakeven_cost`` / ``net_spread``. Use ``notional_turnover()``
        for those.

    References:
        [Hansen-Hodrick 1980][hansen-hodrick-1980]: justifies the
        ``2h + 1`` minimum-date floor for non-overlap pair stride ``h``.

    Examples:
        >>> import factrix as fx
        >>> from factrix.metrics.tradability import rank_turnover
        >>> panel = fx.datasets.make_cs_panel(n_assets=80, n_dates=180, seed=0)
        >>> result = rank_turnover(panel, forward_periods=5)
        >>> result.name == ""
        True
    """
    if quantile is not None and not 0.0 < quantile < 0.5:
        raise ValueError(f"quantile must be in (0, 0.5), got {quantile!r}")
    if forward_periods < 1:
        raise ValueError(f"forward_periods must be ≥ 1, got {forward_periods!r}")

    all_dates = data["date"].unique().sort()
    # Need ≥ 2 non-overlapping pairs so std(ρ) is defined; that requires
    # ≥ 3 sampled dates (Hansen & Hodrick 1980), i.e. ≥ 2·h + 1 raw dates.
    min_required = _rank_turnover_min_dates(forward_periods)
    if len(all_dates) < min_required:
        return _short_circuit_output(
            "rank_turnover",
            "insufficient_dates",
            n_obs=len(all_dates),
            min_required=min_required,
            forward_periods=forward_periods,
        )

    # WHY: polars ``rank()`` sorts NaN *last*, i.e. treats it as larger than
    # every real value, so a missing factor landed in the top tail; and
    # ``pl.len()`` counted those rows, giving the wrong denominator for the
    # tail cutoffs below. Together they pushed rank turnover to 1.0047 —
    # outside the metric's own [0, 1] range — on a panel with 20% NaN factor
    # cells. Every other ranking site in the library filters first; this was
    # the last one that did not.
    sampled_df = _sample_non_overlapping(data, forward_periods).filter(
        _finite_expr(factor_col)
    )

    ranked = sampled_df.select(
        "date",
        "asset_id",
        pl.col(factor_col).rank(method="average").over("date").alias("rank_curr"),
        pl.len().over("date").alias("n_curr"),
    ).sort("asset_id", "date")

    # Lag-within-asset avoids a self-join on (prev_date, asset_id):
    # rank_prev at date_k is this asset's rank at the previous *sampled*
    # date, which is the prior row in each asset's sorted group.
    paired = ranked.with_columns(
        pl.col("rank_curr").shift(1).over("asset_id").alias("rank_prev"),
        pl.col("n_curr").shift(1).over("asset_id").alias("n_prev"),
    ).drop_nulls(["rank_prev"])

    if quantile is not None:
        in_tail = (
            (pl.col("rank_curr") <= pl.col("n_curr") * quantile)
            | (pl.col("rank_curr") > pl.col("n_curr") * (1.0 - quantile))
            | (pl.col("rank_prev") <= pl.col("n_prev") * quantile)
            | (pl.col("rank_prev") > pl.col("n_prev") * (1.0 - quantile))
        )
        paired = paired.filter(in_tail)

    rc_per_date = (
        paired.group_by("date")
        .agg(
            pl.corr("rank_curr", "rank_prev").alias("rc"),
            pl.len().alias("n_pair"),
        )
        .filter(pl.col("rc").is_not_null() & pl.col("rc").is_not_nan())
        .sort("date")
    )

    if rc_per_date.height < 2:
        return _short_circuit_output(
            "rank_turnover",
            "insufficient_periods",
            n_obs=rc_per_date.height,
            n_obs_axis="periods",
            min_required=2,
            forward_periods=forward_periods,
            quantile=quantile,
        )

    rc_arr = rc_per_date["rc"].to_numpy()
    mean_rc = float(np.mean(rc_arr))
    std_rc = float(np.std(rc_arr, ddof=DDOF))
    n_cs_mean = float(rc_per_date["n_pair"].mean())  # type: ignore[arg-type]

    return MetricResult(
        value=1.0 - mean_rc,
        # Adjacent-period transitions (T-1 of them), not (date, asset) pairs:
        # the count moves with the calendar, so the axis is periods.
        n_obs=rc_per_date.height,
        n_obs_axis="periods",
        metadata={
            "mean_rank_autocorrelation": mean_rc,
            "std_rank_autocorrelation": std_rc,
            "n_periods": rc_per_date.height,
            "forward_periods": forward_periods,
            "quantile": quantile,
            "n_cross_section_mean": n_cs_mean,
        },
    )


def _notional_turnover_sample_threshold(self) -> SampleThreshold:
    """Static periods floor plus an instance-derived ``min_assets = n_groups``.

    ``notional_turnover`` buckets each date into ``n_groups`` quantiles and
    reads the top and bottom ones, so a date with fewer than ``n_groups``
    valid names cannot fill both legs and is dropped — with the default
    ``n_groups=10`` an 8-name universe drops *every* date, however long the
    panel. The floor is a function of a constructor argument, so it is
    declared as a resolver (a callable sample_threshold): the default-config
    value is what ``inspect_data`` pre-flights, and the in-body gate below
    re-derives the same ``n_groups`` at run time. Same shape as
    ``quantile._quantile_groups_threshold`` and ``k_spread._k_spread_threshold``.
    """
    return SampleThreshold(min_periods=2, min_assets=self.n_groups)


@metric(
    cell=_TR_CELL,
    aggregation=Aggregation.CS_THEN_TS,
    sample_threshold=_notional_turnover_sample_threshold,
)
def notional_turnover(
    data: pl.DataFrame,
    factor_col: str = "factor",
    *,
    n_groups: int = DEFAULT_N_GROUPS,
    forward_periods: int = DEFAULT_FORWARD_PERIODS,
) -> MetricResult:
    """Portfolio notional turnover via top/bottom quantile membership churn.

    For an equal-weight Q1/Q_n long-short portfolio the only trades that
    incur cost are changes in top-quantile and bottom-quantile membership
    — reshuffling within the middle deciles triggers no rebalancing and
    should not be counted. This is the metric whose units are directly
    compatible with ``breakeven_cost`` / ``net_spread``:
    [Novy-Marx-Velikov (2016)][novy-marx-velikov-2016] τ = fraction of
    portfolio value replaced per rebalance.

    Per-rebalance turnover is the mean of two one-sided overlap losses::

        top_churn = 1 - |Q_top_t ∩ Q_top_{t-1}| / |Q_top_t|
        bot_churn = 1 - |Q_bot_t ∩ Q_bot_{t-1}| / |Q_bot_t|
        turnover  = (top_churn + bot_churn) / 2

    ``(k − m) / k`` for ``k`` names in today's tail and ``m`` carry-overs
    equals the fraction of that leg that must be traded under equal
    weighting. Averaging the two legs (rather than summing) is what makes
    τ a **per-leg** replaced fraction, which is the input
    ``breakeven_cost`` / ``net_spread`` expect: they multiply it back up
    by ``4`` (2 legs × 2 trades per replacement).

    Args:
        data: Panel with ``date, asset_id, factor``.
        factor_col: Name of the factor column.
        n_groups: Number of quantile groups (default
            :data:`~factrix._types.DEFAULT_N_GROUPS` = 5 = quintiles, the
            same constant ``quantile_spread`` defaults to). Must be ≥ 3 so
            top and bottom are distinct buckets.
        forward_periods: Rebalance stride (default
            :data:`~factrix._types.DEFAULT_FORWARD_PERIODS`). When ``> 1``,
            sub-samples at stride ``h`` before pairing consecutive dates —
            matches a holding-period-aligned rebalance schedule.

    Warning:
        ``n_groups`` and ``forward_periods`` must match the ``quantile_spread``
        run whose spread is paired with this turnover. The cost algebra in
        ``breakeven_cost`` / ``net_spread`` is a statement about *one*
        portfolio, so a τ measured on decile membership churn does not price a
        quintile spread, and a τ per bar does not price a 5-period holding.
        The two used to ship incompatible defaults (10 / 1 here against 5 / 5
        there); on a 60-name, 400-period panel at ``gross_spread = 0.001``, the
        matched pair gave breakeven 15.7 bps and net −9.14 bps while each
        function at its own default gave 2.8 bps and −98.02 bps — breakeven
        understated 5.6x, drag overstated 10.7x. They now share one constant,
        and the consumers cross-check the pairing when handed ``MetricResult``s
        instead of bare floats.

    Returns:
        MetricResult with ``value`` = mean per-rebalance turnover ∈ [0, 1].
        ``0`` = identical tail sets every rebalance; ``1`` = full rotation.
        Metadata: ``n_rebalances``, ``n_groups``, ``forward_periods``,
        ``mean_tail_size`` (per-date average of ``(|Q_top| + |Q_bot|)/2``;
        ≠ ``n_assets / n_groups`` signals unbalanced buckets from ties or
        a short universe), ``method``.

    Notes:
        Per rebalance date ``t``::

            top_churn = 1 - |Q_top(t) ∩ Q_top(t-1)| / |Q_top(t)|
            bot_churn = 1 - |Q_bot(t) ∩ Q_bot(t-1)| / |Q_bot(t)|
            turnover_t = (top_churn + bot_churn) / 2
            value = mean_t turnover_t

        factrix averages the two legs (rather than summing) so that ``value``
        is a **per-leg** replaced fraction in [0, 1]. The consumers restore
        the missing factors: ``breakeven_cost`` and ``net_spread`` both use
        ``4 × turnover`` = 2 legs × 2 trades (sell the leaver, buy the
        joiner) with a one-way cost per trade. Summing the legs here
        instead would double-count against those coefficients.
        Names dropped from ``Q_top(t-1)`` / ``Q_bot(t-1)`` by
        delisting before ``t`` are silently missed on the sell side — a
        real portfolio would still book that liquidation cost.

    References:
        [Novy-Marx-Velikov (2016)][novy-marx-velikov-2016], "A Taxonomy of
        Anomalies and Their Trading Costs."

    Examples:
        >>> import factrix as fx
        >>> from factrix.metrics.tradability import notional_turnover
        >>> panel = fx.datasets.make_cs_panel(n_assets=80, n_dates=180, seed=0)
        >>> result = notional_turnover(panel, n_groups=10, forward_periods=5)
        >>> result.name == ""
        True
    """
    if forward_periods < 1:
        raise ValueError(f"forward_periods must be ≥ 1, got {forward_periods!r}")
    if n_groups < 3:
        raise ValueError(
            f"n_groups must be ≥ 3 (need distinct top/bottom buckets), got {n_groups!r}"
        )

    if forward_periods > 1:
        data = _sample_non_overlapping(data, forward_periods)

    dates = data["date"].unique().sort()
    sc = _enforce_min_floor(
        notional_turnover,
        "notional_turnover",
        len(dates),
        "insufficient_dates",
        forward_periods=forward_periods,
    )
    if sc is not None:
        return sc

    top_g = n_groups - 1
    bot_g = 0
    grouped = _assign_quantile_groups(data, factor_col, n_groups).select(
        "date",
        "asset_id",
        (pl.col("_group") == top_g).alias("is_top"),
        (pl.col("_group") == bot_g).alias("is_bot"),
    )

    date_map = pl.DataFrame({"date": dates[1:], "prev_date": dates[:-1]})
    prev = grouped.select(
        pl.col("date").alias("prev_date"),
        "asset_id",
        pl.col("is_top").alias("was_top"),
        pl.col("is_bot").alias("was_bot"),
    )

    # WHY: fill_null(False) treats names absent at t-1 as "new top/bot" —
    # this matches a live portfolio that has to buy into them at t.
    paired = (
        grouped.join(date_map, on="date")
        .join(prev, on=["prev_date", "asset_id"], how="left")
        .with_columns(
            pl.col("was_top").fill_null(False),
            pl.col("was_bot").fill_null(False),
        )
    )

    per_date = (
        paired.group_by("date")
        .agg(
            pl.col("is_top").sum().alias("n_top"),
            (pl.col("is_top") & pl.col("was_top")).sum().alias("n_top_kept"),
            pl.col("is_bot").sum().alias("n_bot"),
            (pl.col("is_bot") & pl.col("was_bot")).sum().alias("n_bot_kept"),
        )
        .filter((pl.col("n_top") > 0) & (pl.col("n_bot") > 0))
        .with_columns(
            (
                (1 - pl.col("n_top_kept") / pl.col("n_top"))
                + (1 - pl.col("n_bot_kept") / pl.col("n_bot"))
            )
            .truediv(2)
            .alias("turnover")
        )
        .sort("date")
    )

    if per_date.is_empty():
        # Name the binding axis. The overwhelmingly common cause is a
        # cross-section too thin to fill ``n_groups`` buckets (the default
        # ``n_groups=10`` empties every date on an allocation-sized universe),
        # so report the assets axis and the floor that was missed rather than a
        # generic "no pairs". A wide-enough panel that still lands here had
        # every date emptied by nulls, and the same reason reads correctly.
        median_assets = _median_universe_size(data)
        return _short_circuit_output(
            "notional_turnover",
            "insufficient_assets_for_quantile_groups",
            n_obs=median_assets,
            n_obs_axis="assets",
            min_required=n_groups,
            warning_codes=(WarningCode.THIN_QUANTILE_GROUPS.value,),
            forward_periods=forward_periods,
            n_groups=n_groups,
        )

    turnover_arr = per_date["turnover"].to_numpy()
    mean_turnover = float(np.mean(turnover_arr))
    tail_pct = 1.0 / n_groups

    mean_tail_size = float(
        per_date.select(((pl.col("n_top") + pl.col("n_bot")) / 2).mean()).item()
    )
    return MetricResult(
        value=mean_turnover,
        # Rebalances — one per adjacent-period transition, not (date, asset)
        # pairs; the axis is periods.
        n_obs=int(per_date.height),
        n_obs_axis="periods",
        metadata={
            "n_rebalances": int(per_date.height),
            "n_groups": n_groups,
            "forward_periods": forward_periods,
            "mean_tail_size": mean_tail_size,
            "method": (
                f"one-sided overlap on top/bottom {tail_pct:.0%} "
                f"quantile, top/bot averaged"
            ),
        },
    )


def _unpack_cost_inputs(
    gross_spread: float | MetricResult,
    turnover: float | MetricResult,
    forward_periods: int,
) -> tuple[float, float, dict[str, object]]:
    """Resolve the two cost inputs and cross-check that they describe one book.

    ``breakeven_cost`` / ``net_spread`` take a spread and a turnover and solve
    a single portfolio's cost algebra. That algebra is only meaningful when the
    two were computed on the *same* bucketing and the *same* rebalance stride —
    a tau measured on decile membership churn does not price a quintile spread.
    Bare floats carry no provenance, so nothing could be checked; when the
    caller passes the producing ``MetricResult``s instead, the pairing is
    verified here and recorded in the consumer's metadata.

    Raises:
        UserInputError: the two results disagree on ``n_groups``, or either
            disagrees with the ``forward_periods`` this call was given.
    """
    checked: dict[str, object] = {}
    spread_meta = (
        gross_spread.metadata if isinstance(gross_spread, MetricResult) else {}
    )
    turnover_meta = turnover.metadata if isinstance(turnover, MetricResult) else {}

    def _mismatch(field: str, left: object, right: object, detail: str) -> None:
        raise UserInputError(
            func_name="breakeven_cost / net_spread",
            field=field,
            value={"gross_spread": left, "turnover": right},
            expected=detail,
            docs_path=_DOCS_TRADABILITY,
        )

    spread_groups = spread_meta.get("n_groups")
    turnover_groups = turnover_meta.get("n_groups")
    if (
        spread_groups is not None
        and turnover_groups is not None
        and spread_groups != turnover_groups
    ):
        _mismatch(
            "n_groups",
            spread_groups,
            turnover_groups,
            "the spread and the turnover to be computed on the same bucketing; "
            "recompute one of them so both use the same n_groups (the shared "
            "default is DEFAULT_N_GROUPS)",
        )
    if spread_groups is not None or turnover_groups is not None:
        checked["n_groups"] = (
            spread_groups if spread_groups is not None else turnover_groups
        )

    for label, meta in (("gross_spread", spread_meta), ("turnover", turnover_meta)):
        upstream_h = meta.get("forward_periods")
        if upstream_h is not None and upstream_h != forward_periods:
            _mismatch(
                "forward_periods",
                upstream_h if label == "gross_spread" else None,
                upstream_h if label == "turnover" else None,
                f"the {label} to have been computed at the same rebalance "
                f"stride as this call's forward_periods={forward_periods}; got "
                f"{upstream_h} upstream",
            )
    if spread_meta or turnover_meta:
        checked["forward_periods"] = forward_periods
        checked["pairing_checked"] = True

    spread_value = (
        gross_spread.value
        if isinstance(gross_spread, MetricResult)
        else float(gross_spread)
    )
    turnover_value = (
        turnover.value if isinstance(turnover, MetricResult) else float(turnover)
    )
    return float(spread_value), float(turnover_value), checked


@metric(
    cell=_TR_CELL,
    aggregation=Aggregation.CS_THEN_TS,
    input_shape=InputShape.SCALAR,
    sample_threshold=SampleThreshold(),
)
def breakeven_cost(
    gross_spread: float | MetricResult,
    turnover: float | MetricResult,
    *,
    forward_periods: int = DEFAULT_FORWARD_PERIODS,
) -> MetricResult:
    """Breakeven single-leg trading cost in bps.

    No static panel-shape thresholds are declared (sample_threshold=SampleThreshold()) because this is a scalar diagnostic function rather than a panel-based metric.

    ``Breakeven = Gross_Spread × forward_periods / (4 × Turnover)``

    If the actual **one-way** trading cost is below this, the factor's
    alpha survives.

    Expects ``turnover`` to be a **notional** fraction ∈ [0, 1] — the
    share of the equal-weight Q1/Q_n portfolio replaced per rebalance.
    Use ``notional_turnover()``; do **not** feed in ``rank_turnover()``
    (which is rank-stability, not position-change).

    Time-scale alignment: ``gross_spread`` from ``quantile_spread`` is
    per-period (forward_return is divided by ``forward_periods`` upstream), but ``turnover``
    is per-rebalance (one rotation every ``forward_periods`` periods). Multiplying spread
    by ``forward_periods`` puts both sides on the per-rebalance scale
    before solving net=0 — without it, breakeven is understated by N×.

    Args:
        gross_spread: Per-period mean long-short spread. This is the
            metric's *data* argument, so it is passed positionally and does
            not appear on the generated ``__init__`` — the call shape is
            ``breakeven_cost(gross_spread, turnover=..., forward_periods=...)``
            and a second positional argument raises ``TypeError``.
        turnover: Notional turnover ∈ [0, 1] from ``notional_turnover()``.
        forward_periods: Holding period matching the upstream
            ``compute_forward_return`` and ``notional_turnover`` stride.

    Note:
        **Pass the ``MetricResult``s, not their ``.value``.** Both data
        arguments accept either a bare ``float`` or the producing
        ``MetricResult``. Given the results, this function verifies that the
        spread and the turnover describe the *same* portfolio — same
        ``n_groups``, same ``forward_periods`` — and raises ``UserInputError``
        when they do not, recording ``pairing_checked`` in metadata when they
        do. Bare floats carry no provenance, so nothing can be checked: the
        cost algebra silently prices a quintile spread with decile turnover if
        that is what it is handed. With the pre-unification defaults that was
        the *likely* outcome, not a corner case — breakeven came out 5.6x too
        low and cost drag 10.7x too high.

    Returns:
        MetricResult with value = breakeven cost in bps.

    Notes:
        ``breakeven_bps = (gross_spread × forward_periods) /
        (4 × turnover) × 1e4``. Multiplying spread by ``forward_periods``
        lifts the per-period spread to the per-rebalance scale matching
        ``turnover``; ``× 1e4`` converts to bps.

        **Where the 4 comes from** (the mirror of ``net_spread``'s cost
        drag — the two must use the same coefficient or breakeven does not
        solve ``net = 0``):

        1. ``turnover`` τ from ``notional_turnover`` is the mean **per-leg**
           fraction of the portfolio replaced per rebalance — the two legs
           are averaged, not summed.
        2. Replacing a fraction τ of a leg is a sell of the leaver plus a
           buy of the joiner: **2τ** of traded notional per leg.
        3. A $1 long / $1 short spread holds two legs: **4τ** of traded
           notional per rebalance per $1 of gross exposure.
        4. The returned cost is therefore the **one-way (per-trade)** cost
           each of those trades may bear.

        The alternative convention states costs per **round trip** (buy +
        sell as one number), under which the coefficient is ``2`` and the
        breakeven figure doubles. factrix reports one-way because that is
        how the half-spread-plus-impact estimates this function is fed
        (and ``net_spread``'s ``estimated_cost_bps`` default) are quoted;
        halve a round-trip quote before comparing it to this number.

        factrix expects ``turnover`` to be a notional fraction in [0, 1]
        (output of ``notional_turnover``); feeding the rank-stability
        ``rank_turnover()`` value will mis-state breakeven by a factor that
        grows with mid-rank churn.

    References:
        [Novy-Marx-Velikov (2016)][novy-marx-velikov-2016], "A Taxonomy of
        Anomalies and Their Trading Costs."

    Examples:
        >>> from factrix.metrics.tradability import breakeven_cost
        >>> result = breakeven_cost(
        ...     gross_spread=0.001, turnover=0.2, forward_periods=5,
        ... )
        >>> result.name == ""
        True
    """
    if forward_periods < 1:
        raise ValueError(f"forward_periods must be ≥ 1, got {forward_periods!r}")
    gross_spread, turnover, checked = _unpack_cost_inputs(
        gross_spread, turnover, forward_periods
    )
    if turnover < EPSILON:
        return MetricResult(
            value=float("inf"),
            metadata={
                "gross_spread": gross_spread,
                "turnover": turnover,
                "forward_periods": forward_periods,
                **checked,
            },
        )

    # WHY: ×4 = 2 legs × 2 trades (sell the leaver, buy the joiner) per unit of
    # per-leg turnover; ×forward_periods lifts the per-period spread to
    # per-rebalance to align with turnover; ×10000 → bps. See Notes for the
    # full derivation.
    be_bps = (gross_spread * forward_periods / (4 * turnover)) * 10000

    return MetricResult(
        value=be_bps,
        metadata={
            "gross_spread": gross_spread,
            "turnover": turnover,
            "forward_periods": forward_periods,
            **checked,
        },
    )


@metric(
    cell=_TR_CELL,
    aggregation=Aggregation.CS_THEN_TS,
    input_shape=InputShape.SCALAR,
    sample_threshold=SampleThreshold(),
)
def net_spread(
    gross_spread: float | MetricResult,
    turnover: float | MetricResult,
    estimated_cost_bps: float = 30.0,
    *,
    forward_periods: int = DEFAULT_FORWARD_PERIODS,
) -> MetricResult:
    """Net spread after estimated trading costs (per-period).

    No static panel-shape thresholds are declared (sample_threshold=SampleThreshold()) because this is a scalar diagnostic function rather than a panel-based metric.

    ``Net = Gross_Spread - 4 × cost_bps × Turnover / forward_periods``

    The ``4 ×`` is ``2 legs × 2 trades``: ``turnover`` is the mean
    **per-leg** fraction replaced per rebalance, each replacement is a
    sell plus a buy, and a $1/$1 long-short holds two legs (see Notes).
    Default ``estimated_cost_bps=30`` is a conservative **one-way**
    mid-cap US equity estimate (half-spread + impact) sized to give a
    useful headline number; override with a venue-specific estimate
    when available.

    Time-scale alignment: ``gross_spread`` is per-period (forward_return
    is divided by ``forward_periods`` upstream) but ``4 × cost × turnover`` is the cost paid
    once per N-period rebalance. Dividing by ``forward_periods`` amortises
    that cost back to per-period. Without it, net is over-charged by N×
    and any factor with h ≥ 2 is artificially killed.

    Expects ``turnover`` to be a **notional** fraction ∈ [0, 1] — the
    share of the equal-weight Q1/Q_n portfolio replaced per rebalance.
    Use ``notional_turnover()``; do **not** feed in ``rank_turnover()``
    (which is rank-stability, not position-change).

    Args:
        gross_spread: Per-period mean long-short spread. This is the
            metric's *data* argument, so it is passed positionally and does
            not appear on the generated ``__init__`` — the call shape is
            ``net_spread(gross_spread, turnover=..., estimated_cost_bps=...,
            forward_periods=...)`` and a second positional argument raises
            ``TypeError``.
        turnover: Notional turnover ∈ [0, 1] from ``notional_turnover()``.
        estimated_cost_bps: Estimated **one-way** (per-trade) trading cost
            in bps — what a single buy or a single sell costs, e.g.
            half-spread + impact. Halve a round-trip quote before passing
            it here.
        forward_periods: Holding period matching the upstream
            ``compute_forward_return`` and ``notional_turnover`` stride.

    Note:
        ``gross_spread`` and ``turnover`` accept the producing
        ``MetricResult``s as well as bare floats; passing the results lets this
        function verify that the two were computed on the same bucketing and
        the same stride. See :func:`breakeven_cost`.

    Returns:
        MetricResult with value = net spread (per-period).

    Notes:
        ``net = gross_spread - 4 × (cost_bps / 1e4) × turnover /
        forward_periods``. Cost is incurred once per ``forward_periods``-
        period rebalance, so dividing by ``forward_periods`` amortises it
        back to the per-period scale of ``gross_spread``. Without the
        amortisation any factor with ``h >= 2`` is over-charged by ``h x``.

        **Where the 4 comes from.** Each step of the accounting, in order:

        1. **Turnover definition.** τ from ``notional_turnover`` is the
           mean **per-leg** fraction of the portfolio replaced per
           rebalance — ``(top_churn + bot_churn) / 2``, an average of the
           two legs, not their sum.
        2. **Trades per rebalance.** Replacing a fraction τ of a leg means
           selling the names that left and buying the names that joined:
           two trades, not one.
        3. **Traded notional.** So one leg trades ``2τ`` of its notional
           per rebalance, and the $1 long / $1 short spread — two legs —
           trades ``4τ`` per $1 of gross spread exposure.
        4. **Cost.** ``estimated_cost_bps`` is the **one-way** cost of a
           single trade, so the drag is ``4 τ c``.

        The alternative convention quotes ``estimated_cost_bps`` as a
        **round-trip** cost (one buy *and* its later sell priced together),
        under which the coefficient is ``2 τ c`` — the pre-0.20 factrix
        behaviour. factrix picks the one-way convention because the
        estimates practitioners have to hand (half-spread, per-order
        impact, per-share commission, and the default ``30`` bps here) are
        one-way quantities; charging them ``2 τ`` under-states the drag by
        exactly half. ``breakeven_cost`` inverts the *same* coefficient, so
        the two remain consistent: the breakeven bps it returns is the
        one-way cost at which this function's ``net`` reaches zero.

        factrix expects ``turnover`` to be a notional fraction (output of
        ``notional_turnover``); rank-stability ``rank_turnover()`` over-states
        the cost drag.

    References:
        [DeMiguel-Martin-Utrera-Nogales-Uppal (2020)][demiguel-martin-utrera-nogales-uppal-2020],
        "A Transaction-Cost Perspective on the Multitude of Firm
        Characteristics." *Review of Financial Studies* 33(5).

    Examples:
        >>> from factrix.metrics.tradability import net_spread
        >>> result = net_spread(
        ...     gross_spread=0.001, turnover=0.2,
        ...     estimated_cost_bps=30.0, forward_periods=5,
        ... )
        >>> result.name == ""
        True
    """
    if forward_periods < 1:
        raise ValueError(f"forward_periods must be ≥ 1, got {forward_periods!r}")
    gross_spread, turnover, checked = _unpack_cost_inputs(
        gross_spread, turnover, forward_periods
    )
    # 4 × τ × c: τ is the mean per-leg replaced fraction, each replacement is a
    # sell plus a buy (2τ traded notional per leg) and the $1/$1 long-short
    # holds two legs. See Notes for the derivation.
    cost_drag = 4 * (estimated_cost_bps / 10000) * turnover / forward_periods
    net = gross_spread - cost_drag

    return MetricResult(
        value=net,
        metadata={
            "gross_spread": gross_spread,
            "cost_drag": cost_drag,
            "estimated_cost_bps": estimated_cost_bps,
            "turnover": turnover,
            "forward_periods": forward_periods,
            **checked,
        },
    )
