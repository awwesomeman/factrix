"""Spanning regression — single-factor test and multi-factor selection.

``spanning_alpha``: does a single factor have alpha after controlling for
base factors? Standard factor research tool ([Barillas-Shanken (2017)][barillas-shanken-2017]).

``greedy_forward_selection``: given a pool of PASS factors, iteratively
select those with incremental alpha (Stage 2).

Both use factor return time series (quantile spread series), not information coefficient (IC).
This is a single-factor artifact, not a jointly-estimated Barra/APT factor return.

Notes:
    **Pipeline.** Regression of factor return time-series on
    base-factor returns (time-series step); Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent (HAC) t on alpha. The
    greedy stepwise selection variant inflates t-stats and is not for
    inference.

References:
    - [Barillas & Shanken (2017)][barillas-shanken-2017], "Which Alpha?"
    - [Feng, Giglio & Xiu (2020)][feng-giglio-xiu-2020], "Taming the
      Factor Zoo: A Test of New Factors."
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from factrix._axis import (
    Aggregation,
    FactorDensity,
    FactorScope,
    InputShape,
)
from factrix._metric_index import SampleThreshold, cell
from factrix._ols import ols_alpha as _ols_alpha
from factrix._results import MetricResult
from factrix._stats import _p_value_from_t
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _degenerate_test_fields,
    _enforce_min_floor,
    _short_circuit_output,
)
from factrix.metrics.quantile import compute_spread_series

__all__ = [  # noqa: RUF022 (teaching order, see SSOT note)
    "spanning_alpha",
    "greedy_forward_selection",
    "SpanningResult",
]

_SPANNING_CELL = cell(
    FactorScope.INDIVIDUAL,
    FactorDensity.DENSE,
    raw="factor-return-series consumer (post-PANEL pipeline)",
)

logger = logging.getLogger(__name__)


@dataclass
class SpanningResult:
    """Result of a single spanning regression for one candidate factor."""

    factor_name: str
    alpha: float
    t_stat: float
    selected: bool


@dataclass
class _ForwardSelection:
    """Internal mutable accumulator for greedy forward selection.

    Holds the ordered factor lists while the algorithm runs;
    :func:`_selection_to_result` folds it into the public ``MetricResult``
    (lists land in ``metadata``). Not a public type — the metric returns a
    standard ``MetricResult`` like every other ``factrix.metrics.*``
    primitive, so ``EvaluationResult.metrics`` stays uniformly
    ``label -> MetricResult``.
    """

    selected_factors: list[SpanningResult] = field(default_factory=list)
    eliminated_factors: list[SpanningResult] = field(default_factory=list)
    all_candidates: list[SpanningResult] = field(default_factory=list)


def _selection_to_result(acc: _ForwardSelection, n_obs: int) -> MetricResult:
    """Fold a :class:`_ForwardSelection` accumulator into a ``MetricResult``.

    ``value`` is the count of surviving (selected) factors; the metric is
    descriptive (``p_value=None``) — no single hypothesis test is emitted.
    The ordered factor lists ride in ``metadata`` alongside
    ``t_stats_inference_invalid=True``: stepwise selection searches over the
    candidate pool and picks by |alpha|, so the t-statistics on
    ``selected_factors`` / ``eliminated_factors`` are conditioned on having
    been chosen. They do not have a valid t-distribution null and must not
    be used for inference ([White (2000)][white-2000];
    [Harvey-Liu-Zhu (2016)][harvey-liu-zhu-2016]). For post-selection
    significance, re-evaluate survivors on a held-out sample or with a
    bootstrap.
    """
    return MetricResult(
        value=float(len(acc.selected_factors)),
        p_value=None,
        n_obs=n_obs,
        n_obs_axis="periods",
        stat=None,
        metadata={
            "method": "greedy forward selection",
            "selected_factors": acc.selected_factors,
            "eliminated_factors": acc.eliminated_factors,
            "all_candidates": acc.all_candidates,
            "t_stats_inference_invalid": True,
        },
    )


# ---------------------------------------------------------------------------
# Date alignment helper
# ---------------------------------------------------------------------------


def _align_spread_series(
    series_dict: dict[str, pl.DataFrame],
) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Align multiple spread series to common dates with finite spreads.

    Only dates where ALL series carry a finite (non-null, non-NaN) spread are
    kept. This avoids biasing the regression by filling missing data with zeros.

    Every returned array is materialised by sorting the per-series join result
    on ``date``, so position ``i`` of every array is the value observed on
    ``common_dates[i]`` regardless of the input frames' row order.

    Raises:
        ValueError: if any input series has duplicate ``date`` rows. A
            duplicated date makes the date → value map ambiguous and silently
            produces arrays of unequal length (``np.column_stack`` then raises
            a shape error far from the cause).

    Returns:
        (common_dates DataFrame, dict of name → aligned numpy array).

    Notes:
        polars joins carry no row-order guarantee (``maintain_order`` defaults
        to ``None``), so the previous ``common_dates.join(...)`` ordering was
        incidental. Under a shuffled input frame the arrays came back permuted
        relative to one another, which silently destroys every downstream
        regression (a true 0.5 slope collapsed to ~0.001). The explicit
        ``.sort("date")`` below pins the contract to the date key rather than
        to join internals. The alternative — passing ``maintain_order="left"``
        — is engine-version-dependent and still gives no protection against
        duplicate keys, so factrix sorts and rejects duplicates instead.
    """
    # WHY: inner join on dates ensures only dates present in ALL series;
    # then filter out any date where any series has a non-finite spread.
    all_dates = None
    for name, data in series_dict.items():
        if data["date"].n_unique() != data.height:
            raise ValueError(
                f"_align_spread_series: series {name!r} has duplicate dates "
                f"({data.height} rows, {data['date'].n_unique()} unique); "
                f"date-keyed alignment is ambiguous. De-duplicate first."
            )
        valid = (
            data.filter(pl.col("spread").is_not_null() & pl.col("spread").is_not_nan())
            .select("date")
            .unique()
        )
        if all_dates is None:
            all_dates = valid
        else:
            all_dates = all_dates.join(valid, on="date", how="inner")

    if all_dates is None or len(all_dates) < 10:
        return pl.DataFrame({"date": []}), {}

    common_dates = all_dates.sort("date")
    arrays = {}
    for name, data in series_dict.items():
        arrays[name] = (
            common_dates.join(data.select("date", "spread"), on="date", how="inner")
            .sort("date")["spread"]
            .to_numpy()
        )
    return common_dates, arrays


# ---------------------------------------------------------------------------
# Public API: single-factor spanning test
# ---------------------------------------------------------------------------


@metric(
    cell=_SPANNING_CELL,
    aggregation=Aggregation.TS_ONLY,
    slice_boundary_sensitive=True,
    input_shape=InputShape.SERIES,
    requires={"factor_spread": compute_spread_series},
    sample_threshold=SampleThreshold(min_periods=10),
)
def spanning_alpha(
    factor_spread: pl.DataFrame,
    base_spreads: dict[str, pl.DataFrame] | None = None,
) -> MetricResult:
    """Test whether a factor has alpha after controlling for base factors.

    Runs: factor_spread = alpha + beta_1 * base_1 + ... + epsilon
    If alpha is significantly != 0, the factor provides incremental value.

    Args:
        factor_spread: DataFrame with ``date, spread`` for the candidate factor.
        base_spreads: Mapping of base factor name → DataFrame with ``date, spread``.
            Required: with no base factors there is nothing to span against,
            so the metric short-circuits (see Returns).

    Returns:
        MetricResult with value=alpha, t_stat, significance. When
        ``base_spreads`` is missing or empty the alpha is not defined —
        regressing the candidate spread on nothing returns its own mean, not
        an incremental alpha — so the metric short-circuits to the standard
        not-computable result (``value=NaN``, ``metadata["reason"] =
        "no_base_factors"``). ``factrix.evaluate`` has no channel for
        injecting base spreads, so this metric is only computable when called
        directly; through ``evaluate`` it reports itself unavailable via
        :attr:`~factrix.WarningCode.METRIC_UNAVAILABLE` rather than silently
        returning the raw spread mean.

    Notes:
        Run ordinary least squares (OLS) ``r_t = alpha + sum_k beta_k * base_k(t) + eps_t`` on
        common-date intersected spread series. Test ``H0: alpha = 0`` via
        ``t = alpha / SE(alpha)`` from the OLS covariance.

        The alpha t-stat uses a Newey-West HAC standard error (Bartlett
        kernel, automatic bandwidth) — the convention for spanning alphas
        in the factor-model literature. An earlier version used the
        homoskedastic OLS SE on the stated assumption that inputs are
        non-overlap spreads; nothing in a ``(date, spread)`` frame can
        verify that, so the SE is now robust to whatever the caller
        passes. On a genuinely non-overlapping iid series the HAC t costs
        1–2 points of size (6.5% at ``n = 120`` against a nominal 5%); on
        one with AR(0.6) residuals the OLS SE it replaces rejects at 33%
        and does not improve with sample size. See
        :func:`factrix._ols.ols_alpha` for the measured table.

    References:
        - [Barillas & Shanken (2017)][barillas-shanken-2017]. "Which
          Alpha?" Review of Financial Studies, 30(4), 1316–1338.
          Spanning-test framework for nested factor models.

    Examples:
        Build a spread series via
        :func:`~factrix.metrics.quantile.compute_spread_series`, then
        test its alpha standalone:

        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.quantile import compute_spread_series
        >>> from factrix.metrics.spanning import spanning_alpha
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, seed=0),
        ...     forward_periods=5,
        ... )
        >>> spreads = compute_spread_series(panel, factor_cols=["factor", "base"])
        >>> result = spanning_alpha(
        ...     spreads["factor"], base_spreads={"base": spreads["base"]}
        ... )
        >>> result.name == ""
        True
    """
    if not base_spreads:
        # Regressing the candidate spread on an empty design returns its own
        # mean, not an incremental alpha. Reporting that as a spanning alpha
        # (with ``n_base_factors=0``) would read as "survives the base model"
        # for a caller — notably ``evaluate``, which cannot supply base
        # spreads at all — who never supplied a base model.
        return _short_circuit_output(
            "spanning_alpha",
            "no_base_factors",
            n_obs=0,
            n_obs_axis="periods",
        )

    all_series = {"_candidate_": factor_spread, **base_spreads}
    _common_dates, arrays = _align_spread_series(all_series)
    if "_candidate_" not in arrays:
        return _short_circuit_output(
            "spanning_alpha",
            "no_overlapping_dates_with_candidate",
            n_obs=0,
            n_obs_axis="periods",
        )
    candidate_arr = arrays.pop("_candidate_")
    base_arrays = arrays
    base_matrix = np.column_stack(list(base_arrays.values()))

    sc = _enforce_min_floor(
        spanning_alpha,
        "spanning_alpha",
        len(candidate_arr),
        "insufficient_spread_observations",
    )
    if sc is not None:
        return sc

    ols = _ols_alpha(candidate_arr, base_matrix)

    base_names = list(base_arrays.keys())
    beta_dict = dict(zip(base_names, ols.betas, strict=False)) if base_names else {}
    n_obs = len(candidate_arr)
    # Reference the regression residual dof (n - 1 - n_base_factors), not the
    # single-sample n - 1: the alpha t-stat is built on the full design matrix.
    p = _p_value_from_t(ols.alpha_t, n_obs, dof=ols.df_resid)

    metadata: dict = {
        "stat_type": "t",
        "h0": "alpha=0",
        "method": "OLS spanning regression, Newey-West HAC SE",
        "n_base_factors": base_matrix.shape[1],
        "base_factors": base_names,
        "betas": beta_dict,
        "r_squared": ols.r_squared,
    }
    # A rank-deficient design (collinear base factors) leaves no fit, so
    # ``alpha`` and its t are NaN: withhold the test rather than report the
    # zeros that would read as "adds exactly nothing".
    warning_codes: list[str] = []
    stat, p_out, alternative = _degenerate_test_fields(
        ols.alpha_t, p, "two-sided", metadata, warning_codes
    )

    return MetricResult(
        p_value=p_out,
        alternative=alternative,
        value=ols.alpha,
        n_obs=n_obs,
        n_obs_axis="periods",
        stat=stat,
        metadata=metadata,
        warning_codes=tuple(warning_codes),
    )


# ---------------------------------------------------------------------------
# Public API: multi-factor greedy forward selection
# ---------------------------------------------------------------------------


@metric(
    cell=_SPANNING_CELL,
    aggregation=Aggregation.TS_ONLY,
    slice_boundary_sensitive=True,
    input_shape=InputShape.SERIES,
    batchable=True,
    requires={"factor_spreads": compute_spread_series},
    sample_threshold=SampleThreshold(),
)
def greedy_forward_selection(
    factor_spreads: dict[str, pl.DataFrame],
    base_spreads: dict[str, pl.DataFrame] | None = None,
    significance_threshold: float = 2.0,
    max_factors: int = 20,
    suppress_snooping_warning: bool = False,
) -> MetricResult:
    """Greedy forward selection with backward elimination.

    No static panel-shape thresholds are declared (sample_threshold=SampleThreshold()) because the descriptive result carries no single hypothesis test and does not short-circuit like value/p_value metrics.

    WARNING — data snooping / selection bias:
        Stepwise selection over a candidate pool of K factors inflates
        the per-selected-factor t-stat by an order-statistic factor
        (typical estimates 2-4× on K=10-100 pools). The t-stats on
        ``metadata["selected_factors"]`` are NOT valid for hypothesis testing —
        they are conditional on survival, not draws from the t-null.
        Use this function as a **model-construction helper**, not as
        an inference tool. For post-selection significance, re-evaluate
        the surviving set on a held-out window, or use a [White (2000)][white-2000]
        Reality Check / [Hansen (2005)][hansen-2005] SPA procedure on the pre-selection
        stage. The returned ``t_stats_inference_invalid=True`` flag
        encodes this contract.

    Algorithm:
        1. Start with base factor set (e.g., Size, Value, Momentum spreads).
        2. For each candidate PASS factor, compute spanning alpha.
        3. Select the candidate with largest |alpha| if t-stat >= threshold.
        4. After each addition, backward-check all selected factors:
           re-run spanning regression for each against all others.
           Remove any whose alpha becomes insignificant.
        5. Repeat until no candidate has significant alpha.

    Args:
        factor_spreads: Mapping of factor name → DataFrame with ``date, spread``.
        base_spreads: Mapping of base factor name → DataFrame with ``date, spread``.
            If None, starts with an empty base.
        significance_threshold: Minimum |t-stat| for selection (default 2.0).
        max_factors: Maximum number of factors to select.
        suppress_snooping_warning: Silence the one-shot ``UserWarning``.
            Only set when the caller has explicitly acknowledged that
            the returned t-stats are for model-construction, not
            inference.

    Returns:
        A descriptive :class:`~factrix._results.MetricResult`: ``value`` is
        the count of surviving (selected) factors and ``p_value`` is
        ``None``. The ordered factor lists ride in ``metadata`` —
        ``selected_factors`` / ``eliminated_factors`` / ``all_candidates``
        (each a list of :class:`SpanningResult`) plus
        ``t_stats_inference_invalid=True``.

    Notes:
        Iteratively run ``spanning_alpha(candidate, base ∪ selected)``;
        add the candidate with the largest ``|alpha|`` whose ``|t| >=
        threshold``; after each add, re-test all selected factors against
        the others and drop any that lose significance. Repeat until the
        pool dries up or ``max_factors`` is hit.

        factrix flags ``t_stats_inference_invalid = True`` because the
        retained t-stats are conditional on selection — they are
        order-statistic inflated and must not be read as draws from the
        t-null. Use the result as a model-construction helper; verify
        survivors on a held-out window.

    References:
        - [White (2000)][white-2000]. "A Reality Check for Data Snooping."
          Econometrica, 68(5), 1097–1126. Bootstrap reality-check for
          data-snooping bias — the canonical correction this function
          does *not* apply (inflates t-stats by design).
        - [Harvey, Liu & Zhu (2016)][harvey-liu-zhu-2016]. "…and the
          Cross-Section of Expected Returns." Review of Financial
          Studies, 29(1), 5–68. Empirical case for raising t-thresholds;
          section on stepwise-selection bias.

    Examples:
        Greedy step-wise selection across two candidate spread series.
        ``suppress_snooping_warning=True`` acknowledges the inflated-t
        contract carried by the ``t_stats_inference_invalid`` metadata
        flag:

        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.quantile import compute_spread_series
        >>> from factrix.metrics.spanning import greedy_forward_selection
        >>> seeds = [0, 1]
        >>> spreads = {
        ...     f"cand_{s}": compute_spread_series(
        ...         compute_forward_return(
        ...             fx.datasets.make_cs_panel(n_assets=80, n_dates=180, seed=s),
        ...             forward_periods=5,
        ...         ),
        ...         forward_periods=5,
        ...     )["factor"]
        ...     for s in seeds
        ... }
        >>> result = greedy_forward_selection(
        ...     spreads,
        ...     suppress_snooping_warning=True,
        ... )
        >>> result.metadata["t_stats_inference_invalid"]
        True
    """
    if not suppress_snooping_warning:
        warnings.warn(
            "greedy_forward_selection: stepwise selection inflates the "
            "t-stats on selected factors; the returned values are NOT "
            "valid for inference. Use this as a model-construction "
            "helper and re-evaluate survivors on a held-out sample or "
            "with a Hansen (2005) SPA / White (2000) Reality Check "
            "procedure. Pass suppress_snooping_warning=True to silence.",
            UserWarning,
            stacklevel=2,
        )
    else:
        # Audit trail: flip-to-silent is the obvious bypass path; log
        # it at INFO so a research-log reader can still see that the
        # snooping discipline was explicitly acknowledged-and-dismissed.
        logger.info(
            "greedy_forward_selection: snooping warning suppressed by "
            "caller (n_candidates=%d, n_base=%d)",
            len(factor_spreads),
            len(base_spreads or {}),
        )

    if base_spreads is None:
        base_spreads = {}

    all_series = {**base_spreads, **factor_spreads}
    common_dates, all_arrays = _align_spread_series(all_series)

    if not all_arrays:
        return _selection_to_result(_ForwardSelection(), n_obs=0)

    base_arrays = {n: all_arrays[n] for n in base_spreads if n in all_arrays}
    candidate_arrays = {n: all_arrays[n] for n in factor_spreads if n in all_arrays}

    selected_names: list[str] = []
    selected_arrays: dict[str, np.ndarray] = {}
    result = _ForwardSelection()
    remaining = set(candidate_arrays.keys())

    for _step in range(max_factors):
        if not remaining:
            break

        base_cols = list(base_arrays.values()) + [
            selected_arrays[n] for n in selected_names
        ]
        if base_cols:
            base_matrix = np.column_stack(base_cols)
        else:
            base_matrix = np.empty((len(common_dates), 0))

        best_name = None
        best_alpha = 0.0
        best_t = 0.0

        for name in remaining:
            ols = _ols_alpha(candidate_arrays[name], base_matrix)
            result.all_candidates.append(
                SpanningResult(
                    factor_name=name,
                    alpha=ols.alpha,
                    t_stat=ols.alpha_t,
                    selected=False,
                )
            )
            if abs(ols.alpha_t) >= significance_threshold and abs(ols.alpha) > abs(
                best_alpha
            ):
                best_name = name
                best_alpha = ols.alpha
                best_t = ols.alpha_t

        if best_name is None:
            break

        selected_names.append(best_name)
        # ``pop`` so ``selected_arrays`` becomes sole owner of the
        # selected factor's buffer — when ``_backward_eliminate``
        # later drops it from ``selected_arrays`` the buffer is freed
        # immediately rather than lingering until function return.
        selected_arrays[best_name] = candidate_arrays.pop(best_name)
        remaining.remove(best_name)
        result.selected_factors.append(
            SpanningResult(
                factor_name=best_name,
                alpha=best_alpha,
                t_stat=best_t,
                selected=True,
            )
        )

        _backward_eliminate(
            selected_names,
            selected_arrays,
            base_arrays,
            significance_threshold,
            result,
        )

    return _selection_to_result(result, n_obs=len(common_dates))


def _backward_eliminate(
    selected_names: list[str],
    selected_arrays: dict[str, np.ndarray],
    base_arrays: dict[str, np.ndarray],
    threshold: float,
    result: _ForwardSelection,
) -> None:
    """Remove selected factors whose alpha becomes insignificant."""
    changed = True
    while changed:
        changed = False
        for name in list(selected_names):
            others = [selected_arrays[n] for n in selected_names if n != name]
            base_cols = list(base_arrays.values()) + others
            if base_cols:
                base_matrix = np.column_stack(base_cols)
            else:
                base_matrix = np.empty((len(selected_arrays[name]), 0))

            ols = _ols_alpha(selected_arrays[name], base_matrix)

            # NaN t means the fit could not be formed on this subset
            # (rank-deficient once the other selections are in the design).
            # Drop rather than keep: an unevaluable factor has not earned
            # its place, and ``NaN < threshold`` is False, which would.
            if not math.isfinite(ols.alpha_t) or abs(ols.alpha_t) < threshold:
                selected_names.remove(name)
                del selected_arrays[name]
                result.eliminated_factors.append(
                    SpanningResult(
                        factor_name=name,
                        alpha=ols.alpha,
                        t_stat=ols.alpha_t,
                        selected=False,
                    )
                )
                result.selected_factors = [
                    s for s in result.selected_factors if s.factor_name != name
                ]
                changed = True
                logger.info(
                    "backward elimination: removed %s (alpha=%.6f, t=%.2f)",
                    name,
                    ols.alpha,
                    ols.alpha_t,
                )
                break
