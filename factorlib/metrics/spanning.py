"""Spanning regression — single-factor test and multi-factor selection.

``spanning_alpha``: does a single factor have alpha after controlling for
base factors? Standard factor research tool (Barillas & Shanken 2017).

``greedy_forward_selection``: given a pool of PASS factors, iteratively
select those with incremental alpha (Stage 2).

Both use factor return time series (quantile spread series), not IC.

References:
    Barillas & Shanken (2017), "Which Alpha?"
    Feng, Giglio & Xiu (2020), "Taming the Factor Zoo."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from factorlib._ols import OLSResult, ols_alpha as _ols_alpha
from factorlib._types import MetricOutput
from factorlib._stats import _p_value_from_t, _significance_marker

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
    """Output of greedy forward selection."""

    selected_factors: list[SpanningResult] = field(default_factory=list)
    eliminated_factors: list[SpanningResult] = field(default_factory=list)
    all_candidates: list[SpanningResult] = field(default_factory=list)


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

    References:
        Barillas & Shanken (2017), "Which Alpha?"
    """
    if base_spreads is None:
        base_spreads = {}

    if base_spreads:
        all_series = {"_candidate_": factor_spread, **base_spreads}
        common_dates, arrays = _align_spread_series(all_series)
        if "_candidate_" not in arrays:
            return MetricOutput(
                name="spanning_alpha", value=float("nan"), stat=None, significance="",
                metadata={
                    "reason": "no_overlapping_dates_with_candidate",
                    "n_observed": 0,
                    "p_value": 1.0,
                },
            )
        candidate_arr = arrays.pop("_candidate_")
        base_arrays = arrays
        base_matrix = np.column_stack(list(base_arrays.values()))
    else:
        vals = factor_spread["spread"].drop_nulls()
        if len(vals) < 10:
            return MetricOutput(
                name="spanning_alpha", value=float("nan"), stat=None, significance="",
                metadata={
                    "reason": "insufficient_spread_observations",
                    "n_observed": len(vals),
                    "min_required": 10,
                    "p_value": 1.0,
                },
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
) -> ForwardSelectionResult:
    """Greedy forward selection with backward elimination.

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

    Returns:
        ForwardSelectionResult with selected factors in order.
    """
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
