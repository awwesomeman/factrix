"""Top-bucket concentration analysis for cross-sectional panels.

Measures whether top-bucket (long-leg) alpha is concentrated in a few
stocks or broadly distributed, using Herfindahl-Hirschman index (HHI)
inverse.

Notes:
    **Pipeline.** Per-date HHI inverse on top-bucket weights
    (cross-section step) → per-date ratio series, then non-overlapping
    sample; across-time t against ``H₀: ratio ≥ 0.5``.

    **Input.** DataFrame with ``date, asset_id, factor, forward_return``.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    FactorScope,
)
from factrix._codes import WarningCode
from factrix._metric_index import cell
from factrix._results import MetricResult
from factrix._stats import _calc_t_stat, _p_value_from_t
from factrix._types import (
    DDOF,
    EPSILON,
    MIN_PORTFOLIO_PERIODS_HARD,
    MIN_PORTFOLIO_PERIODS_WARN,
    ConcentrationWeight,
)
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _compute_tie_ratio,
    _enforce_scaled_floor,
    _sample_non_overlapping,
    _scaled_periods_threshold,
    _short_circuit_output,
    _warn_below_scaled_floor,
)

__all__ = [
    "top_concentration",
]


@metric(
    cell=cell(
        FactorScope.INDIVIDUAL, FactorDensity.DENSE, structure=DataStructure.PANEL
    ),
    aggregation=Aggregation.CS_THEN_TS,
    # Periods floor scales with the non-overlap stride (see ``quantile``): the
    # per-date HHI series is sub-sampled at ``forward_periods``, so the HARD and
    # WARN floors and their in-body gates share ``MIN_PORTFOLIO_PERIODS_*`` +
    # ``_scaled_min_periods``.
    sample_threshold=_scaled_periods_threshold(
        MIN_PORTFOLIO_PERIODS_HARD, warn=MIN_PORTFOLIO_PERIODS_WARN
    ),
)
def top_concentration(
    data: pl.DataFrame,
    forward_periods: int = 5,
    q_top: float = 0.2,
    factor_col: str = "factor",
    return_col: str = "forward_return",
    weight_by: ConcentrationWeight = "abs_factor",
) -> MetricResult:
    r"""Top-bucket concentration via Herfindahl-Hirschman index (HHI) inverse.

    Per date, selects top ``q_top`` stocks by factor rank, computes
    HHI of their weights, and returns 1/HHI as the effective number of
    independent bets.

    Args:
        data: Panel with ``date, asset_id, factor`` (and ``forward_return``
            if ``weight_by="alpha_contribution"``).
        q_top: Fraction of top-ranked stocks to include (default 0.2 =
            top 20%).
        weight_by: HHI weight convention.
            - ``"abs_factor"`` (default): weight by ``|factor|``. Answers
              "how concentrated is the density itself in the top bucket".
              Conservative, density-level.
            - ``"alpha_contribution"``: weight by the magnitude of each
              name's realised contribution ``|sign(factor) · forward_return|``.
              Captures **risk-concentration**: the top bucket's realised
              return is dominated by a few outliers. Note the absolute
              value — a single big *winner* and a single big *loser*
              both register as concentration, which is the right
              framing for risk but NOT for signed-alpha attribution.
              If you need the latter, apply HHI downstream on the
              signed ``sign(factor) · forward_return`` series yourself.

    Returns:
        MetricResult with value = mean(1/HHI) across dates.
        Higher = more diversified top bucket.

    Notes:
        Per non-overlap date $t$ with top-bucket members $Q^{\mathrm{top}}(t)$
        (size $n^{\mathrm{top}}$), define weights $w_i$ by ``weight_by``
        and form the Herfindahl
        $\mathrm{HHI}_t = \sum_i (w_i / \sum_j w_j)^2$. Effective
        independent bets $n^{\mathrm{eff}}_t = 1 / \mathrm{HHI}_t$.
        Per-date diversification ratio
        $r_t = n^{\mathrm{eff}}_t / n^{\mathrm{top}}$ is averaged and tested
        one-sided against $H_0: \mathbb{E}[r] \geq 0.5$: rejecting flags
        concentration.

        factrix uses ``rank(method="average")`` for the top-bucket cutoff
        — ``tie_policy`` from Config does not apply because HHI measures
        concentration *among* the selected stocks, not their bucketing.
        ``tie_ratio`` is still recorded in metadata as a data-quality
        diagnostic (high tie_ratio → unstable membership across
        re-rankings).

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.concentration import top_concentration
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, seed=0),
        ...     forward_periods=5,
        ... )
        >>> result = top_concentration(panel, forward_periods=5, q_top=0.2)
        >>> result.name == ""
        True
    """
    if weight_by == "alpha_contribution" and return_col not in data.columns:
        return _short_circuit_output(
            "top_concentration",
            "no_return_column",
            alternative="less",
            missing_column=return_col,
            weight_by=weight_by,
        )

    filtered = _sample_non_overlapping(data, forward_periods)
    tie_ratio = _compute_tie_ratio(filtered, factor_col)

    q1 = filtered.with_columns(
        (
            pl.col(factor_col).rank(method="average").over("date")
            / pl.len().over("date")
        ).alias("_pct_rank")
    ).filter(pl.col("_pct_rank") >= (1 - q_top))

    if weight_by == "alpha_contribution":
        weighted = q1.with_columns(
            (pl.col(factor_col).sign() * pl.col(return_col)).abs().alias("_raw_weight")
        )
    else:
        weighted = q1.with_columns(pl.col(factor_col).abs().alias("_raw_weight"))

    hhi_per_date = (
        weighted.with_columns(
            (pl.col("_raw_weight") / pl.col("_raw_weight").sum().over("date")).alias(
                "_weight"
            )
        )
        .group_by("date")
        .agg(
            (pl.col("_weight") ** 2).sum().alias("hhi"),
            pl.len().alias("n_top"),
        )
        .filter(pl.col("hhi") > EPSILON)
        .with_columns((1.0 / pl.col("hhi")).alias("eff_n"))
        .sort("date")
    )

    # Raw (pre-sampling) date count: the axis the stride-scaled periods floors
    # are calibrated against.
    n_raw_periods = data["date"].n_unique()
    sc = _enforce_scaled_floor(
        "top_concentration",
        n_raw_periods,
        MIN_PORTFOLIO_PERIODS_HARD,
        forward_periods,
        "insufficient_portfolio_periods",
        tie_ratio=tie_ratio,
    )
    if sc is not None:
        return sc

    warning_codes: list[str] = []
    warn_code = _warn_below_scaled_floor(
        n_raw_periods,
        MIN_PORTFOLIO_PERIODS_WARN,
        forward_periods,
        f"top_concentration: {n_raw_periods} raw dates below "
        f"MIN_PORTFOLIO_PERIODS_WARN*forward_periods="
        f"{MIN_PORTFOLIO_PERIODS_WARN * forward_periods}; the one-sided t-test "
        f"on the per-date diversification ratio is returned but df=n-1 inflates "
        f"t_crit relative to the asymptotic cutoff. Read borderline p-values "
        f"cautiously.",
        WarningCode.BORDERLINE_PORTFOLIO_PERIODS,
    )
    if warn_code is not None:
        warning_codes.append(warn_code)

    eff_n_arr = hhi_per_date["eff_n"].to_numpy()
    n_top_arr = hhi_per_date["n_top"].to_numpy()
    mean_eff_n = float(np.mean(eff_n_arr))
    mean_n_top = float(np.mean(n_top_arr))
    ratio = mean_eff_n / max(mean_n_top, 1)

    # WHY: t-stat tests H₀: ratio ≥ 0.5 (well-diversified).
    # Per-date ratio = eff_n / n_top; if mean ratio < 0.5 with significant t,
    # alpha is concentrated in a few stocks.
    ratio_arr = eff_n_arr / np.maximum(n_top_arr, 1)
    n = len(ratio_arr)
    mean_ratio = float(np.mean(ratio_arr))
    std_ratio = float(np.std(ratio_arr, ddof=DDOF))
    # Test H₀: ratio ≥ 0.5 → shift by 0.5 then use standard t-test
    t = _calc_t_stat(mean_ratio - 0.5, std_ratio, n)

    # WHY: one-sided test → p = P(T < t), not two-sided
    p = _p_value_from_t(t, n, alternative="less")
    metadata: dict = {
        "stat_type": "t",
        "h0": "ratio>=0.5",
        "method": "one-sided t-test on ratio",
        "mean_n_top": mean_n_top,
        "ratio_eff_to_total": ratio,
        "tie_ratio": tie_ratio,
        "weight_by": weight_by,
    }
    return MetricResult(
        p_value=p,
        alternative="less",
        value=mean_eff_n,
        n_obs=n,
        n_obs_axis="periods",
        stat=t,
        metadata=metadata,
        warning_codes=tuple(warning_codes),
    )
