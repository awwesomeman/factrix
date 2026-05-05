"""Spanning regression — single-factor test and multi-factor selection.

Aggregation: regression of factor return time-series on base-factor
returns (time-series step); NW HAC t on alpha. The greedy stepwise
selection variant inflates t-stats and is not for inference.

``spanning_alpha``: does a single factor have alpha after controlling for
base factors? Standard factor research tool (Barillas & Shanken 2017).

``greedy_forward_selection``: given a pool of PASS factors, iteratively
select those with incremental alpha (Stage 2).

Both use factor return time series (quantile spread series), not IC.

References:
    Barillas & Shanken (2017), "Which Alpha?"
    Feng, Giglio & Xiu (2020), "Taming the Factor Zoo."

Matrix-row: spanning_alpha, greedy_forward_selection | factor-return-series consumer (post-PANEL pipeline) | TS-only | NW HAC / OLS t | _p_value_from_t, _significance_marker, _short_circuit_output, _ols_alpha
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from factrix._ols import OLSResult, ols_alpha as _ols_alpha
from factrix._types import MetricOutput
from factrix.metrics._helpers import _short_circuit_output
from factrix._stats import _p_value_from_t, _significance_marker

logger = logging.getLogger(__name__)


@dataclass
class SpanningResult:
    """Result of a single spanning regression for one candidate factor."""

    factor_name: str
    alpha: float
    t_stat: float
    selected: bool


@dataclass
class ForwardSelectionResult:
    """Output of greedy forward selection.

    ``t_stats_inference_invalid``: a fixed ``True`` — stepwise selection
    searches over the candidate pool and picks by |alpha|, so the
    t-statistics on ``selected_factors`` and ``eliminated_factors`` are
    conditioned on having been chosen. They do not have a valid
    t-distribution null and must not be used for inference (White 2000;
    Harvey-Liu-Zhu 2016). For post-selection significance, re-evaluate
    survivors on a held-out sample or with a bootstrap.
    """

    selected_factors: list[SpanningResult] = field(default_factory=list)
    eliminated_factors: list[SpanningResult] = field(default_factory=list)
    all_candidates: list[SpanningResult] = field(default_factory=list)
    t_stats_inference_invalid: bool = field(default=True, init=False)


# ---------------------------------------------------------------------------
# Date alignment helper
# ---------------------------------------------------------------------------

def _align_spread_series(
    series_dict: dict[str, pl.DataFrame],
) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Align multiple spread series to common dates with non-null spreads.

    Only dates where ALL series have non-null spread values are kept.
    This avoids biasing regression by filling missing data with zeros.

    Returns:
        (common_dates DataFrame, dict of name → aligned numpy array).
    """
    # WHY: inner join on dates ensures only dates present in ALL series;
    # then filter out any date where any series has null spread
    all_dates = None
    for df in series_dict.values():
        valid = df.filter(pl.col("spread").is_not_null()).select("date").unique()
        if all_dates is None:
            all_dates = valid
        else:
            all_dates = all_dates.join(valid, on="date", how="inner")

    if all_dates is None or len(all_dates) < 10:
        return pl.DataFrame({"date": []}), {}

    common_dates = all_dates.sort("date")
    arrays = {}
    for name, df in series_dict.items():
        arrays[name] = (
            common_dates.join(df.select("date", "spread"), on="date", how="inner")
            ["spread"].to_numpy()
        )
    return common_dates, arrays


# ---------------------------------------------------------------------------
# Public API: single-factor spanning test
# ---------------------------------------------------------------------------

def spanning_alpha(
    factor_spread: pl.DataFrame,
    base_spreads: dict[str, pl.DataFrame] | None = None,
) -> MetricOutput:
    """Test whether a factor has alpha after controlling for base factors.

    Runs: factor_spread = alpha + beta_1 * base_1 + ... + epsilon
    If alpha is significantly != 0, the factor provides incremental value.

    Args:
        factor_spread: DataFrame with ``date, spread`` for the candidate factor.
        base_spreads: Mapping of base factor name → DataFrame with ``date, spread``.
            If None or empty, tests whether the factor has nonzero mean return.

    Returns:
        MetricOutput with value=alpha, t_stat, significance.

    Notes:
        Run OLS ``r_t = alpha + sum_k beta_k * base_k(t) + eps_t`` on
        common-date intersected spread series. Test ``H0: alpha = 0`` via
        ``t = alpha / SE(alpha)`` from the OLS covariance.

        factrix uses plain OLS standard errors here rather than NW HAC:
        the inputs are non-overlap quantile spreads (single-period stride)
        so MA(h-1) overlap is absent. Callers feeding HAC-relevant
        overlapping series should either pre-resample or wrap the call
        with their own HAC SE.

    References:
        Barillas & Shanken (2017), "Which Alpha?"
        [White 1980](../../reference/bibliography.md#white-1980): heteroskedasticity-consistent SE
        ancestor of the HAC variants applicable when overlap is added.
    """
    if base_spreads is None:
        base_spreads = {}

    if base_spreads:
        all_series = {"_candidate_": factor_spread, **base_spreads}
        common_dates, arrays = _align_spread_series(all_series)
        if "_candidate_" not in arrays:
            return _short_circuit_output(
                "spanning_alpha", "no_overlapping_dates_with_candidate",
                n_observed=0,
            )
        candidate_arr = arrays.pop("_candidate_")
        base_arrays = arrays
        base_matrix = np.column_stack(list(base_arrays.values()))
    else:
        vals = factor_spread["spread"].drop_nulls()
        if len(vals) < 10:
            return _short_circuit_output(
                "spanning_alpha", "insufficient_spread_observations",
                n_observed=len(vals), min_required=10,
            )
        candidate_arr = vals.to_numpy()
        base_arrays = {}
        base_matrix = np.empty((len(candidate_arr), 0))

    ols = _ols_alpha(candidate_arr, base_matrix)

    base_names = list(base_arrays.keys())
    beta_dict = dict(zip(base_names, ols.betas)) if base_names else {}
    n_obs = len(candidate_arr)
    p = _p_value_from_t(ols.alpha_t, n_obs)

    return MetricOutput(
        name="spanning_alpha",
        value=ols.alpha,
        stat=ols.alpha_t,
        significance=_significance_marker(p),
        metadata={
            "n_obs": n_obs,
            "p_value": p,
            "stat_type": "t",
            "h0": "alpha=0",
            "method": "OLS spanning regression",
            "n_base_factors": base_matrix.shape[1],
            "base_factors": base_names,
            "betas": beta_dict,
            "r_squared": ols.r_squared,
        },
    )


# ---------------------------------------------------------------------------
# Public API: multi-factor greedy forward selection
# ---------------------------------------------------------------------------

def greedy_forward_selection(
    factor_spreads: dict[str, pl.DataFrame],
    base_spreads: dict[str, pl.DataFrame] | None = None,
    significance_threshold: float = 2.0,
    max_factors: int = 20,
    suppress_snooping_warning: bool = False,
) -> ForwardSelectionResult:
    """Greedy forward selection with backward elimination.

    WARNING — data snooping / selection bias:
        Stepwise selection over a candidate pool of K factors inflates
        the per-selected-factor t-stat by an order-statistic factor
        (typical estimates 2-4× on K=10-100 pools). The t-stats on
        ``selected_factors`` are NOT valid for hypothesis testing —
        they are conditional on survival, not draws from the t-null.
        Use this function as a **model-construction helper**, not as
        an inference tool. For post-selection significance, re-evaluate
        the surviving set on a held-out window, or use a White (2000)
        Reality Check / Hansen (2005) SPA procedure on the pre-selection
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
        ForwardSelectionResult with selected factors in order.

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
        White (2000), "A Reality Check for Data Snooping."
        Harvey, Liu & Zhu (2016), "…and the Cross-Section of Expected
        Returns," Section on stepwise-selection bias.
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
            len(factor_spreads), len(base_spreads or {}),
        )

    if base_spreads is None:
        base_spreads = {}

    all_series = {**base_spreads, **factor_spreads}
    common_dates, all_arrays = _align_spread_series(all_series)

    if not all_arrays:
        return ForwardSelectionResult()

    base_arrays = {n: all_arrays[n] for n in base_spreads if n in all_arrays}
    candidate_arrays = {n: all_arrays[n] for n in factor_spreads if n in all_arrays}

    selected_names: list[str] = []
    selected_arrays: dict[str, np.ndarray] = {}
    result = ForwardSelectionResult()
    remaining = set(candidate_arrays.keys())

    for step in range(max_factors):
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
            result.all_candidates.append(SpanningResult(
                factor_name=name, alpha=ols.alpha, t_stat=ols.alpha_t, selected=False,
            ))
            if abs(ols.alpha_t) >= significance_threshold and abs(ols.alpha) > abs(best_alpha):
                best_name = name
                best_alpha = ols.alpha
                best_t = ols.alpha_t

        if best_name is None:
            break

        selected_names.append(best_name)
        selected_arrays[best_name] = candidate_arrays[best_name]
        remaining.remove(best_name)
        result.selected_factors.append(SpanningResult(
            factor_name=best_name, alpha=best_alpha, t_stat=best_t,
            selected=True,
        ))

        _backward_eliminate(
            selected_names, selected_arrays, base_arrays,
            significance_threshold, result,
        )

    return result


def _backward_eliminate(
    selected_names: list[str],
    selected_arrays: dict[str, np.ndarray],
    base_arrays: dict[str, np.ndarray],
    threshold: float,
    result: ForwardSelectionResult,
) -> None:
    """Remove selected factors whose alpha becomes insignificant."""
    changed = True
    while changed:
        changed = False
        for name in list(selected_names):
            others = [
                selected_arrays[n] for n in selected_names if n != name
            ]
            base_cols = list(base_arrays.values()) + others
            if base_cols:
                base_matrix = np.column_stack(base_cols)
            else:
                base_matrix = np.empty((len(selected_arrays[name]), 0))

            ols = _ols_alpha(selected_arrays[name], base_matrix)

            if abs(ols.alpha_t) < threshold:
                selected_names.remove(name)
                del selected_arrays[name]
                result.eliminated_factors.append(SpanningResult(
                    factor_name=name, alpha=ols.alpha, t_stat=ols.alpha_t, selected=False,
                ))
                result.selected_factors = [
                    s for s in result.selected_factors if s.factor_name != name
                ]
                changed = True
                logger.info(
                    "backward elimination: removed %s (alpha=%.6f, t=%.2f)",
                    name, ols.alpha, ols.alpha_t,
                )
                break
