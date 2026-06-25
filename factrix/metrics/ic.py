"""IC (Information Coefficient) computation for cross-sectional panels.

Notes:
    **Pipeline.** Per-date Spearman rank IC (cross-section step) → IC
    time series, then non-overlapping cross-asset t or Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent (HAC) t on its
    mean; the regime variant slices the same pipeline.

    **Input.** DataFrame with ``date, asset_id, factor, forward_return``.

    **Output.** Time-indexed IC series (``date, ic``) that can be fed
    into any ``series/`` tool (oos, trend, significance, hit_rate).
"""

from __future__ import annotations

import math
import warnings as _warnings

import polars as pl

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    FactorScope,
    InputShape,
)
from factrix._codes import WarningCode
from factrix._metric_index import SampleThreshold, cell
from factrix._results import MetricResult
from factrix._stats.constants import MIN_PERIODS_HARD, MIN_PERIODS_WARN
from factrix._types import (
    EPSILON,
    MIN_IC_PERIODS,
)
from factrix.inference import NEWEY_WEST, NON_OVERLAPPING, NeweyWest, NonOverlapping
from factrix.metrics._base import MetricBase
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    TIE_RATIO_WARN_THRESHOLD,
    _check_applicable_inference,
    _enforce_min_floor,
    _short_circuit_output,
    _surface_drop_stats,
    _warn_below_floor,
)
from factrix.metrics._metric_capabilities import per_date_series_rename
from factrix.metrics._primitives import compute_ic

__all__ = [
    "ic",
    "ic_ir",
]

_IC_CELL = cell(
    FactorScope.INDIVIDUAL,
    FactorDensity.DENSE,
    structure=DataStructure.PANEL,
)

# Slice-test contract: IC is per-date Spearman rank
# correlation, not a bucketed metric — slice tests never need to
# downscale `n_groups`. The min-cross-section-per-date constraint
# (Spearman ρ asymptotic distribution requires ≥ 30 obs per date,
# Hollander-Wolfe-Chicken §8.6) lives in the procedure as
# `MIN_IC_ASSETS` short-circuit, parallel to (not exposed
# via) this attribute.
min_assets_per_group: int | None = None
per_date_series = per_date_series_rename("ic")

# Inference allowlist: ``ic`` dispatches an ``Inference.compute`` polymorphically,
# so it *could* run any series-mean member, but the vetted pair is the
# non-overlap t-test and the Bartlett-kernel Newey-West HAC. ``HansenHodrick``
# (rectangular kernel, no PSD guarantee) is deliberately excluded.
applicable_inference: frozenset[NonOverlapping | NeweyWest] = frozenset(
    {NON_OVERLAPPING, NEWEY_WEST}
)


def _median_tie_ratio(ic_df: pl.DataFrame) -> float:
    """Median of the per-date ``tie_ratio`` column, or ``nan`` if absent/empty."""
    if "tie_ratio" not in ic_df.columns:
        return float("nan")
    med = ic_df["tie_ratio"].median()
    return float("nan") if med is None else float(med)  # type: ignore[arg-type]


def _warn_if_high_ic_tie_ratio(ic_df: pl.DataFrame, metric_name: str) -> float:
    """Emit ``UserWarning`` when median tie_ratio exceeds the global threshold.

    Returns the median for caller to stash in metadata. The Spearman ρ on
    average ranks is biased relative to the tie-corrected formula
    (Kendall-Stuart §31) at high tie densities — a bucketed / categorical
    factor will look like it has IC ≈ 0 even if the bucketing is
    informative. Threshold reuses the global ``TIE_RATIO_WARN_THRESHOLD``
    (0.3) shared with the quantile-bucketing diagnostics.
    """
    med = _median_tie_ratio(ic_df)
    if not math.isnan(med) and med > TIE_RATIO_WARN_THRESHOLD:
        _warnings.warn(
            f"{metric_name}: median tie_ratio={med:.3f} exceeds "
            f"{TIE_RATIO_WARN_THRESHOLD:.2f}. Spearman ρ on average ranks is "
            f"biased on bucketed / categorical factors; treat the IC "
            f"magnitude as a lower bound and consider a tie-corrected "
            f"correlation or a continuous transform of the factor.",
            UserWarning,
            stacklevel=2,
        )
    return med


def _ic_sample_threshold(self: MetricBase) -> SampleThreshold:
    """Dynamic periods floor for ``ic``: the inference method's minimum input
    length, which scales with ``forward_periods`` (non-overlapping stride) or
    is a fixed HAC bound. Delegates to the same ``min_input_periods`` the
    in-body short-circuit reads, so the pre-flight and run-time floors agree.
    """
    # ``inference`` is an ``ic``-specific field; the resolver is only ever bound
    # to ``ic``, but its declared param type is the ``MetricBase`` contract.
    inference = self.inference  # type: ignore[attr-defined]
    return SampleThreshold(
        min_periods=inference.min_input_periods(self.forward_periods)
    )


@metric(
    cell=_IC_CELL,
    aggregation=Aggregation.CS_THEN_TS,
    input_shape=InputShape.SERIES,
    requires={"ic_df": compute_ic},
    sample_threshold=_ic_sample_threshold,
)
def ic(
    ic_df: pl.DataFrame,
    forward_periods: int = 5,
    inference: NonOverlapping | NeweyWest = NON_OVERLAPPING,
) -> MetricResult:
    r"""Information coefficient (IC) mean significance: is mean IC significantly different from zero?

    The periods floor is dynamic — the minimum input length scales with the
    forward_periods parameter and the inference method — so it is declared as a
    resolver (a callable sample_threshold) rather than a constant.

    Args:
        ic_df: Output of ``compute_ic()``.
        forward_periods: Overlap horizon of the forward returns; the
            non-overlapping stride and the HAC bandwidth floor both key
            off it.
        inference: Significance-test method. ``fx.inference.NON_OVERLAPPING``
            (default) runs an OLS t-test on a non-overlapping stride
            subsample; ``fx.inference.NEWEY_WEST`` keeps every observation
            and uses a Newey-West HAC standard error. Both test the same
            $H_0: \mathbb{E}[\mathrm{IC}] = 0$.

    Returns:
        MetricResult with value=mean IC and the inference method's t/p.

    Notes:
        Given the per-date IC series $\mathrm{IC}_t$, $H_0:
        \mathbb{E}[\mathrm{IC}] = 0$. The non-overlapping path strides the
        series at ``forward_periods`` (discarding $h-1$ of every $h$
        observations) to avoid the lag floor implied by overlapping
        forward returns; the Newey-West path keeps every observation and
        absorbs the induced MA($h-1$) autocorrelation through HAC standard
        errors (Bartlett kernel, NW1994 auto-bandwidth floored at
        $h - 1$).

    References:
        [Grinold 1989][grinold-1989]: IC as the canonical density-quality
        measure under the Fundamental Law of Active Management.
        [Hansen-Hodrick 1980][hansen-hodrick-1980]: K-period overlapping
        returns carry MA(K-1) autocorrelation — the motivation for the
        non-overlap stride used here.

    Method selection:
        The default ``NON_OVERLAPPING`` path tests on roughly
        ``n / forward_periods`` effective observations; when that
        post-stride sample is thin it emits
        ``WarningCode.UNRELIABLE_SE_SHORT_PERIODS`` (now surfaced on the
        returned result's ``warning_codes``). ``NEWEY_WEST`` keeps every
        observation and absorbs the overlap-induced autocorrelation in the
        HAC standard error, so on a thin series it retains more test power.
        The guidance is one-directional: prefer ``NEWEY_WEST`` when the
        non-overlapping effective sample is too thin; there is no symmetric
        reason to switch back to non-overlapping once the sample is ample.
        ``ic`` never changes ``inference`` for you — the choice stays
        explicit.

    Examples:
        Chain from :func:`compute_ic` output:

        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.ic import compute_ic, ic
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, seed=0),
        ...     forward_periods=5,
        ... )
        >>> ic_df = compute_ic(panel)["factor"]
        >>> result = ic(ic_df, forward_periods=5)
        >>> result.name == ""
        True
    """
    _check_applicable_inference(inference, applicable_inference, func_name="ic")
    median_tie = _warn_if_high_ic_tie_ratio(ic_df, "ic")
    # Mean is order-invariant; the inference method owns date-ordering for
    # its stride / lag math.
    ic_vals = ic_df["ic"].drop_nulls()
    n = len(ic_vals)
    raw_min = inference.min_input_periods(forward_periods)
    if n < raw_min:
        return _short_circuit_output(
            "ic",
            "insufficient_ic_periods",
            n_obs=n,
            n_obs_axis="periods",
            min_required=raw_min,
            forward_periods=forward_periods,
        )

    result = inference.compute(ic_df, value_col="ic", forward_periods=forward_periods)

    # Stride-based methods report a post-sampling count; guard on the
    # effective sample so a coarse stride cannot silently test ~nothing.
    n_sampled = result.metadata.get("n_obs_sampled")
    if n_sampled is not None and n_sampled < MIN_IC_PERIODS:
        return _short_circuit_output(
            "ic",
            "insufficient_sampled_ic_periods",
            n_obs=int(n_sampled),
            n_obs_axis="periods",
            min_required=MIN_IC_PERIODS,
            forward_periods=forward_periods,
        )

    mean_ic = float(ic_vals.mean())  # type: ignore[arg-type]
    metadata: dict[str, object] = {
        "n_periods": n,
        "forward_periods": forward_periods,
        "stat_type": "t",
        "h0": "mu=0",
        "method": inference.summary,
        "tie_ratio": median_tie,
    }
    warning_codes: list[str] = []
    _surface_drop_stats(ic_df, "ic", metadata, warning_codes)
    # Surface the inference method's own soft-floor signals (e.g. a thin
    # post-stride sample tripping UNRELIABLE_SE_SHORT_PERIODS); de-dup so a
    # code already raised by the drop-stats pass is not repeated.
    for code in result.warnings:
        if code.value not in warning_codes:
            warning_codes.append(code.value)
    return MetricResult(
        p_value=result.p_value,
        value=mean_ic,
        n_obs=n,
        n_obs_axis="periods",
        stat=result.stat,
        metadata=metadata,
        warning_codes=tuple(warning_codes),
    )


@metric(
    cell=_IC_CELL,
    aggregation=Aggregation.CS_THEN_TS,
    input_shape=InputShape.SERIES,
    requires={"ic_df": compute_ic},
    sample_threshold=SampleThreshold(
        min_periods=MIN_PERIODS_HARD,
        warn_periods=MIN_PERIODS_WARN,
    ),
)
def ic_ir(
    ic_df: pl.DataFrame,
) -> MetricResult:
    r"""$\mathrm{ICIR} = \mathrm{mean}(\mathrm{IC}) / \mathrm{std}(\mathrm{IC})$.

    Signed ratio — positive when information coefficient (IC) is consistently positive, negative
    when consistently negative.  Analogous to a Sharpe ratio for the
    factor density.

    This is a **descriptive statistic**, not a hypothesis test (t_stat=None).
    For significance testing, use ``ic()``.

    Args:
        ic_df: Output of ``compute_ic()``.

    Returns:
        MetricResult with value=IC_IR (signed), t_stat=None.

    Notes:
        $\mathrm{ICIR} = \mathrm{mean}(\mathrm{IC}) / \mathrm{std}(\mathrm{IC})$
        over the per-date IC series — a Sharpe-style ratio describing
        time-series stability of the density. Reported as a descriptive
        statistic; no inference is attached because the significance test
        on $\mathrm{mean}(\mathrm{IC})$ lives in ``ic`` (optionally with
        ``inference=fx.inference.NEWEY_WEST`` for the HAC-corrected SE).

    References:
        [Grinold 1989][grinold-1989]: ICIR is the time-stability
        normalisation that completes the IR decomposition.

    Examples:
        Chain from :func:`compute_ic` output:

        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.ic import compute_ic, ic_ir
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, seed=0),
        ...     forward_periods=5,
        ... )
        >>> ic_df = compute_ic(panel)["factor"]
        >>> result = ic_ir(ic_df)
        >>> result.name == ""
        True
    """
    median_tie = _warn_if_high_ic_tie_ratio(ic_df, "ic_ir")
    ic_vals = ic_df["ic"].drop_nulls()
    n = len(ic_vals)
    sc = _enforce_min_floor(ic_ir, "ic_ir", n, "insufficient_ic_periods")
    if sc is not None:
        return sc

    mean_ic = float(ic_vals.mean())  # type: ignore[arg-type]
    std_ic = float(ic_vals.std())  # type: ignore[arg-type]

    if std_ic < EPSILON:
        return _short_circuit_output(
            "ic_ir",
            "degenerate_ic_variance",
            std_ic=std_ic,
        )

    ratio = mean_ic / std_ic

    warning_codes: list[str] = []
    warn_code = _warn_below_floor(
        ic_ir,
        n,
        f"ic_ir: n_periods={n} below MIN_PERIODS_WARN={MIN_PERIODS_WARN}; "
        f"the IC information ratio on a short series is unstable. value is "
        f"returned but read it cautiously.",
        WarningCode.UNRELIABLE_SE_SHORT_PERIODS,
    )
    if warn_code is not None:
        warning_codes.append(warn_code)

    metadata: dict[str, object] = {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "n_periods": n,
        "tie_ratio": median_tie,
    }
    _surface_drop_stats(ic_df, "ic_ir", metadata, warning_codes)
    return MetricResult(
        value=ratio,
        n_obs=n,
        n_obs_axis="periods",
        warning_codes=tuple(warning_codes),
        metadata=metadata,
    )
