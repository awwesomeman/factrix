"""Monotonicity test for cross-sectional panels.

Measures whether factor quantile groups exhibit monotonic return ordering.
Per-period: split into n_groups by factor rank, compute mean return per group,
Spearman corr between group index and return.

Notes:
    **Pipeline.** Per-period Spearman corr between quantile index and
    group mean return (cross-section step), then non-overlapping
    cross-asset t on the per-period series.

    **Input.** DataFrame with ``date, asset_id, factor, forward_return``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import polars as pl
import scipy.stats as scipy_stats

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    FactorScope,
)
from factrix._codes import WarningCode
from factrix._metric_index import SampleThreshold, cell
from factrix._results import MetricResult
from factrix._stats import _calc_t_stat, _p_value_from_t
from factrix._stats.bootstrap import (
    Rng,
    _check_n_resamples,
    _empirical_p,
    _resolve_rng,
)
from factrix._types import (
    DDOF,
    DEFAULT_FORWARD_PERIODS,
    MIN_MONOTONICITY_PERIODS_HARD,
    TiePolicy,
)
from factrix.metrics._base import MetricBase
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _assign_quantile_groups_batch,
    _enforce_scaled_floor,
    _finite_expr,
    _median_universe_size,
    _sample_non_overlapping,
    _scaled_periods_threshold,
    _short_circuit_output,
    _validate_factor_cols,
    _validate_n_groups,
    _warn_high_tie_ratio,
)

__all__ = [
    "monotonicity",
]


# Slice-test contract: monotonicity buckets the
# cross-section into `n_groups` (default 10) and computes Spearman ρ
# across per-bucket means. Patton & Timmermann (2010) "Monotonicity
# in Asset Returns" recommend ≥ 50 assets per bucket so the per-period
# bucket means converge to their cross-sectional expectation; below
# this floor individual-asset noise dominates the rank statistic.
# `_downscale_n_groups(base, n_assets, min_assets_per_group=50)` caps
# `n_groups` accordingly inside the slice-test function.
min_assets_per_group: int | None = 50


_MONOTONICITY_PERIODS_FLOOR = _scaled_periods_threshold(MIN_MONOTONICITY_PERIODS_HARD)


def _monotonicity_sample_threshold(self) -> SampleThreshold:
    """Stride-scaled periods floor plus an instance-derived ``min_assets = n_groups``.

    The periods floor scales with the non-overlap stride (see ``quantile``):
    the per-period Spearman series is sub-sampled at ``overlap_periods``, so
    pre-flight and the in-body gate share ``MIN_MONOTONICITY_PERIODS_HARD`` +
    ``_scaled_min_periods``.

    The assets floor is the binding one on a small universe. Each date is split
    into ``n_groups`` buckets and a date with a null bucket mean is dropped, so
    with the default ``n_groups=10`` (calibrated for a ~2000-name equity
    universe) *every* date is dropped when ``n_assets < n_groups`` — however
    long the panel. Declaring it here, in the same resolver shape as
    ``quantile._quantile_groups_threshold`` and ``k_spread._k_spread_threshold``,
    is what stops ``inspect_data`` calling the default config usable on an
    8-name panel that ``evaluate`` then refuses.
    """
    periods = _MONOTONICITY_PERIODS_FLOOR(self)
    return SampleThreshold(
        min_periods=periods.min_periods,
        warn_periods=periods.warn_periods,
        min_assets=self.n_groups,
    )


# Patton-Timmermann (2010) monotonic-relationship (MR) test. ``direction`` is
# declared by the caller, not read off the data: running both and reporting the
# better one is a two-sided search reported at a one-sided level.
MRDirection = Literal["increasing", "decreasing"]


def _mr_test(
    bucket_means: np.ndarray,
    *,
    direction: MRDirection,
    n_resamples: int,
    rng: Rng,
) -> tuple[float, float, dict[str, object]]:
    """Patton-Timmermann (2010) MR test on a ``(n_periods, n_groups)`` block.

    Returns ``(J, p_value, metadata)`` where ``J = min_i mean_t Delta_{i,t}`` is
    the smallest average adjacent bucket-return difference, in return units.

    H0 is "the pattern is **not** monotonically increasing", i.e.
    ``min_i E[Delta_i] <= 0``; H1 is that every adjacent step is positive. The
    null distribution is obtained by recentring the per-period difference matrix
    at zero (the least-favourable configuration under H0) and resampling whole
    blocks of periods with the stationary bootstrap, so cross-bucket dependence
    within a period and serial dependence across periods are both preserved —
    the differences are resampled jointly under one row-index draw, never
    column by column.

    ``p = (1 + #{J* >= J}) / (B + 1)`` — the Davison-Hinkley ``+1`` smoothing
    the rest of the library's empirical-p paths use (:func:`_empirical_p`),
    so the p can never be exactly 0.
    """
    from factrix.stats import stationary_bootstrap_resamples

    diffs = np.diff(bucket_means, axis=1)
    if direction == "decreasing":
        diffs = -diffs
    delta_bar = diffs.mean(axis=0)
    j_stat = float(delta_bar.min())

    # Mirror ``StationaryBootstrap``: resolve the seed and report it, so a run
    # is reproducible after the fact without a mandatory knob.
    rng, seed_used = _resolve_rng(
        rng, func_name="monotonicity", docs_path="api/metrics/monotonicity"
    )
    resamples = stationary_bootstrap_resamples(diffs, n_resamples, rng=rng)
    # (B, T, K-1) -> (B, K-1) bootstrap means, recentred under H0.
    j_star = (resamples.mean(axis=1) - delta_bar).min(axis=1)
    p_value, p_mc_se = _empirical_p(
        int(np.count_nonzero(j_star >= j_stat)), n_resamples
    )

    metadata: dict[str, object] = {
        "mr_direction": direction,
        "mr_min_diff": j_stat,
        "mr_adjacent_diffs": [float(v) for v in delta_bar],
        "n_resamples": n_resamples,
        "seed": seed_used,
        "p_value_mc_se": p_mc_se,
    }
    return j_stat, p_value, metadata


_DOCS_MONOTONICITY = "api/metrics/monotonicity"


def _validate_monotonicity(m: MetricBase) -> None:
    """Numeric knob bounds for ``monotonicity``.

    ``tie_policy`` and ``direction`` are closed sets the constructor reads off
    their ``Literal`` annotations, so only the bucketing floor, the resampling
    count and the batch request are stated here.
    """
    _validate_n_groups(
        m.n_groups,  # type: ignore[attr-defined]
        func_name="monotonicity",
        docs_path=_DOCS_MONOTONICITY,
    )
    _check_n_resamples(
        m.n_resamples,  # type: ignore[attr-defined]
        func_name="monotonicity",
        docs_path=_DOCS_MONOTONICITY,
    )
    _validate_factor_cols(
        m.factor_cols,  # type: ignore[attr-defined]
        func_name="monotonicity",
        docs_path=_DOCS_MONOTONICITY,
    )


@metric(
    cell=cell(
        FactorScope.INDIVIDUAL, FactorDensity.DENSE, structure=DataStructure.PANEL
    ),
    aggregation=Aggregation.CS_THEN_TS,
    batchable=True,
    sample_threshold=_monotonicity_sample_threshold,
    validate=_validate_monotonicity,
)
def monotonicity(
    data: pl.DataFrame,
    overlap_periods: int = DEFAULT_FORWARD_PERIODS,
    n_groups: int = 10,
    factor_cols: Sequence[str] = ("factor",),
    return_col: str = "forward_return",
    tie_policy: TiePolicy = "ordinal",
    direction: MRDirection = "increasing",
    n_resamples: int = 999,
    rng: Rng = None,
    expected_warnings: tuple[str, ...] = (),
) -> dict[str, MetricResult]:
    """Quantile return monotonicity — Patton-Timmermann (2010) MR test.

    ``value`` / ``stat`` = the MR statistic ``J = min_i mean_t Delta_{i,t}``,
    the smallest average adjacent bucket-return difference, in return units.
    ``p_value`` is its stationary-bootstrap p.

    Args:
        data: Panel with ``date, asset_id, factor, forward_return``.
        n_groups: Number of quantile groups (default 10 for a ~2000-name
            universe). Use 5 for ``n_assets < 1000``, 3 for ``n_assets < 200``.
        tie_policy: Bucketing tie-break policy — the same knob and values as
            ``quantile_spread`` / ``k_spread``, see ``_assign_quantile_groups``.
            Either ``"ordinal"`` or ``"average"``; anything else raises ``UserInputError``.
            Under ``"ordinal"`` the realised per-period tie ratio is reported
            in ``metadata["tie_ratio"]``; when its median exceeds
            ``TIE_RATIO_WARN_THRESHOLD`` the result carries
            ``WarningCode.HIGH_TIE_RATIO`` and echoes a ``UserWarning`` unless
            the code is declared in ``expected_warnings``.
        direction: Which monotone pattern H1 asserts — ``"increasing"``
            (default) or ``"decreasing"`` in bucket index. Declare it from the
            factor's hypothesis; running both and reporting the smaller p is a
            two-sided search charged at a one-sided level.
        n_resamples: Bootstrap resamples for the MR null distribution. Must
            be at least ``BOOTSTRAP_RESAMPLES_FLOOR`` — the shared floor every
            factrix entry point reporting an inference drawn from resamples
            enforces. Below it the empirical p is dominated by resampling
            noise: the Monte-Carlo SE of the reported p is
            ``sqrt(p(1-p)/n_resamples)``, reported as
            ``metadata["p_value_mc_se"]``. At the default 999 resamples a p
            near 0.05 carries ~0.7pp of MC SE, so 0.043 and 0.058 are one
            draw apart; raise ``n_resamples`` to shrink it, since no amount
            of data does.
        rng: Bootstrap seed — an ``int``, ``None``, or a
            ``numpy.random.Generator``. ``None`` resolves one and reports it
            in ``metadata["seed"]``, so a run stays reproducible after the
            fact; a ``Generator`` is used as-is and advanced, and
            ``metadata["seed"]`` is ``None`` because only its owner can
            reproduce the draw. An unseeded re-run redraws the null, moving
            the p by roughly ``metadata["p_value_mc_se"]``.

    Returns:
        MetricResult with ``value`` = ``stat`` = the MR statistic and
        ``p_value`` from the bootstrap. The descriptive Spearman summaries
        stay in metadata.

    Notes:
        Per non-overlap date ``t``, assets are bucketed into ``n_groups`` by
        factor rank and each bucket's mean return is taken. The MR test then
        works on the adjacent differences ``Delta_{i,t} = mu_{i,t} -
        mu_{i-1,t}``: ``J = min_i mean_t Delta_{i,t}``, with
        ``H0: min_i E[Delta_i] <= 0`` ("the relation is *not* monotonically
        increasing") against ``H1: min_i E[Delta_i] > 0``. The null
        distribution comes from recentring the per-period difference matrix at
        zero — the least-favourable configuration under H0 — and resampling
        whole blocks of periods with the stationary bootstrap, which keeps both
        the within-period cross-bucket dependence and the serial dependence
        across periods. This is the test Patton-Timmermann (2010) actually
        propose, and the one this metric previously cited without implementing.

        **Why the headline is no longer mean |Spearman|.** ``mean_t |rho_t|``
        has a large null floor that depends on ``n_groups``: on a factor drawn
        independently of returns (T=400, N=100) it reads 0.66 at
        ``n_groups=3``, 0.43 at 5 and 0.27 at 10, because ``E|rho| > 0`` under
        H0 by Jensen's inequality. A reader seeing "value = 0.43,
        Patton-Timmermann (2010)" took it as MR evidence when it was the noise
        floor for five buckets. The Spearman summaries remain in metadata as
        descriptive shape statistics —
        ``mean_abs_spearman`` (magnitude, always >= 0) and ``mean_signed``
        (direction consistency) — where a high magnitude with a near-zero
        signed mean still tells the useful story that the factor sorts returns
        but flips sign across dates.

        **Deviation from the paper.** Patton-Timmermann bootstrap the raw
        (unstudentised) differences, which is what runs here. Their studentised
        variant is not implemented.

    Examples:
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.monotonicity import monotonicity
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=200, n_dates=180, rng=0),
        ...     forward_periods=5,
        ... )
        >>> result = monotonicity(
        ...     panel, overlap_periods=5, n_groups=5, n_resamples=199, rng=0
        ... )
        >>> result["factor"].name == ""
        True
    """
    cols = list(factor_cols)

    # Raw (pre-sampling) date count: the axis the stride-scaled periods floor is
    # calibrated against, shared across all factors.
    n_raw_periods = data["date"].n_unique()

    # Sample non-overlapping once — shared across all factors on the
    # same panel (depends only on `date` + `overlap_periods`).
    filtered = _sample_non_overlapping(data, overlap_periods)
    tie_ratios = _compute_tie_ratios_batch(filtered, cols)
    # Echo per factor (the helper gates on ``expected_warnings``); the
    # predicate is kept so each factor's result carries the structured code.
    high_tie_ratio = {
        f: _warn_high_tie_ratio(
            tie_ratios[f],
            "monotonicity",
            tie_policy,
            expected_warnings=expected_warnings,
        )
        for f in cols
    }

    grouped = _assign_quantile_groups_batch(filtered, cols, n_groups, tie_policy)

    # Stage 1: per-(date, factor, group) mean return, expressed as one
    # ``group_by("date").agg(...)`` carrying N × n_groups filter+mean
    # expressions. Wide output: (n_dates, 1 + N * n_groups). For large
    # F × n_groups this is a tall ask of the planner, but a single
    # ``collect()`` beats N separate per-factor queries because the
    # rank / group columns get computed once and the panel is scanned
    # only once.
    agg_exprs: list[pl.Expr] = [
        pl.col(return_col)
        .filter(pl.col(f"_group__{f}") == g)
        .mean()
        .alias(f"_gr__{f}__{g}")
        for f in cols
        for g in range(n_groups)
    ]
    group_returns_wide = (
        grouped.lazy().group_by("date").agg(agg_exprs).sort("date").collect()
    )

    # Stage 2: per-(factor, date) Spearman ρ between group index and
    # the rank of the per-group mean return. Done in numpy because the
    # inner shape (n_dates rows × n_groups cells per factor) is small
    # and the operation is uniform — vectorising in numpy beats the
    # polars-side rank+corr pipeline at this size. Materialise the
    # full (n_dates, N * n_groups) block once (zero-copy via Arrow)
    # and slice per factor — saves N * n_groups individual ``.to_numpy``
    # calls.
    gr_col_names = [f"_gr__{f}__{g}" for f in cols for g in range(n_groups)]
    all_means = group_returns_wide.select(gr_col_names).to_numpy()
    group_idx = np.arange(n_groups, dtype=np.float64)
    group_idx_centered = group_idx - group_idx.mean()
    group_idx_norm = float(np.sqrt(np.sum(group_idx_centered**2)))
    results: dict[str, MetricResult] = {}
    for i, f in enumerate(cols):
        mat = all_means[:, i * n_groups : (i + 1) * n_groups]
        # Drop dates with any null/nan bucket mean (matches the
        # original filter `n == n_groups` and `mono.is_not_null`).
        mat = mat[np.all(np.isfinite(mat), axis=1)]
        if mat.shape[0] == 0:
            mono_arr = np.empty(0)
        else:
            # Spearman = Pearson(group_idx, rank(group_mean)).
            ranks = scipy_stats.rankdata(mat, axis=1, method="average")
            ranks_centered = ranks - ranks.mean(axis=1, keepdims=True)
            ranks_norm = np.sqrt(np.sum(ranks_centered**2, axis=1))
            with np.errstate(invalid="ignore", divide="ignore"):
                mono_arr = (ranks_centered @ group_idx_centered) / (
                    ranks_norm * group_idx_norm
                )
            mono_arr = mono_arr[np.isfinite(mono_arr)]

        sc = _enforce_scaled_floor(
            "monotonicity",
            n_raw_periods,
            MIN_MONOTONICITY_PERIODS_HARD,
            overlap_periods,
            "insufficient_monotonicity_periods",
            n_groups=n_groups,
            tie_ratio=tie_ratios[f],
            tie_policy=tie_policy,
        )
        if sc is not None:
            results[f] = sc
            continue
        if len(mono_arr) == 0:
            # n_raw_periods cleared the scaled floor above, yet every sampled
            # date had a null bucket mean. Name the axis that actually binds:
            # a cross-section too thin to populate ``n_groups`` buckets empties
            # every date regardless of how many periods there are, so calling
            # that "insufficient periods" sent the reader to the wrong axis.
            # A wide-enough cross-section that still lands here was emptied by
            # nulls (e.g. a sparse column) and reads correctly under the same
            # reason: the buckets could not be filled.
            median_assets = _median_universe_size(data)
            results[f] = _short_circuit_output(
                "monotonicity",
                "insufficient_assets_for_quantile_groups",
                n_obs=median_assets,
                n_obs_axis="assets",
                min_required=n_groups,
                warning_codes=(WarningCode.THIN_QUANTILE_GROUPS.value,),
                n_groups=n_groups,
                tie_ratio=tie_ratios[f],
                tie_policy=tie_policy,
            )
            continue
        # Headline: the MR test on the same bucket means the Spearman
        # summaries describe. ``mat`` is already restricted to periods with a
        # finite mean in every bucket, so the difference matrix is finite.
        j_stat, p_mr, mr_metadata = _mr_test(
            mat,
            direction=direction,
            n_resamples=n_resamples,
            rng=rng,
        )
        # Descriptive shape statistics, kept because magnitude and direction
        # consistency read separately (see Notes) — not the headline.
        avg_mono = float(np.mean(np.abs(mono_arr)))
        mean_mono = float(np.mean(mono_arr))
        std_mono = float(np.std(mono_arr, ddof=DDOF))
        t_signed = _calc_t_stat(mean_mono, std_mono, len(mono_arr))
        metadata: dict[str, object] = {
            "method": (
                "Patton-Timmermann (2010) MR test; stationary-bootstrap "
                "empirical p on the min adjacent bucket-return difference"
            ),
            "stat_type": "mr",
            "h0": f"min_i E[delta_i] <= 0 ({direction})",
            "mean_abs_spearman": avg_mono,
            "mean_signed": mean_mono,
            "signed_spearman_t": t_signed,
            "signed_spearman_p_value": _p_value_from_t(t_signed, len(mono_arr)),
            "n_valid_periods": len(mono_arr),
            "n_groups": n_groups,
            "tie_ratio": tie_ratios[f],
            "tie_policy": tie_policy,
            **mr_metadata,
        }
        results[f] = MetricResult(
            p_value=p_mr,
            alternative="greater",
            value=j_stat,
            n_obs=mat.shape[0],
            n_obs_axis="periods",
            stat=j_stat,
            metadata=metadata,
            warning_codes=(
                (WarningCode.HIGH_TIE_RATIO.value,) if high_tie_ratio[f] else ()
            ),
        )

    return results


def _compute_tie_ratios_batch(
    data: pl.DataFrame, factor_cols: list[str]
) -> dict[str, float]:
    """Median-across-dates tie ratio (``1 - n_unique / n``) for many factors.

    The single-factor :func:`_compute_tie_ratio` runs a separate polars
    aggregation per factor; this batches them into one ``group_by("date")`` so
    the sampled panel is scanned once for any number of factors. The tie ratio
    is **per period** over the *finite* values then median-reduced — the same
    statistic the single-factor helper returns. Computing it globally (``n_unique`` / ``len`` over the whole
    frame) would conflate cross-sectional ties with values merely repeating
    across dates, inflating the ratio toward 1 and tripping spurious
    high-tie-ratio warnings on a continuous factor.
    """
    if not factor_cols:
        return {}
    # Count only finite values, exactly as the single-factor
    # :func:`_compute_tie_ratio` does. A bare ``pl.len()`` / ``n_unique()``
    # counts nulls and NaNs as a tied level, so a cross-regional panel with 5
    # of 10 names missing on a date read 0.4 on a factor with **zero** ties —
    # over the 0.3 threshold, so the metric emitted a spurious
    # "consider tie_policy='average'" advisory and stamped the wrong tie_ratio
    # into metadata, while quantile_spread on the same panel reported 0.0.
    per_period = data.group_by("date").agg(
        *[_finite_expr(f).sum().alias(f"_n__{f}") for f in factor_cols],
        *[
            pl.col(f).filter(_finite_expr(f)).n_unique().alias(f"_u__{f}")
            for f in factor_cols
        ],
    )
    # ``median`` over the (possibly empty) per-period ratio yields ``None`` on an
    # empty frame, which maps to ``nan`` below — the same empty-panel contract as
    # the single-factor :func:`_compute_tie_ratio`, no separate guard needed.
    medians = per_period.select(
        (1.0 - pl.col(f"_u__{f}") / pl.col(f"_n__{f}")).median().alias(f"_tr__{f}")
        for f in factor_cols
    ).row(0, named=True)
    return {
        f: float("nan") if medians[f"_tr__{f}"] is None else float(medians[f"_tr__{f}"])
        for f in factor_cols
    }
