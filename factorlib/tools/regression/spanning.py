"""Spanning regression — Greedy Forward Selection with Backward Elimination.

Stage 2: given a pool of PASS factors, determine which have incremental
alpha beyond existing base factors. Uses factor return time series
(quantile spread series), not IC.

Input: dict of factor name → spread series (date, spread).
Output: ordered list of selected factors with spanning alpha and t-stat.

References:
    Barillas & Shanken (2017), "Which Alpha?"
    Feng, Giglio & Xiu (2020), "Taming the Factor Zoo."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from factorlib.tools._typing import EPSILON

logger = logging.getLogger(__name__)


@dataclass
class SpanningResult:
    """Result of a single spanning regression for one candidate factor.

    Attributes:
        factor_name: Name of the candidate factor.
        alpha: Intercept (daily excess return not explained by base factors).
        t_stat: t-statistic of alpha.
        selected: Whether the factor was selected into the set.
    """

    factor_name: str
    alpha: float
    t_stat: float
    selected: bool


@dataclass
class ForwardSelectionResult:
    """Output of greedy forward selection.

    Attributes:
        selected_factors: Factors selected in order, with their spanning stats.
        eliminated_factors: Factors that were eliminated during backward checks.
        all_candidates: Full list of candidate evaluations per step.
    """

    selected_factors: list[SpanningResult] = field(default_factory=list)
    eliminated_factors: list[SpanningResult] = field(default_factory=list)
    all_candidates: list[SpanningResult] = field(default_factory=list)


def _spanning_regression(
    candidate: np.ndarray,
    base_matrix: np.ndarray,
) -> tuple[float, float]:
    """Run spanning regression: candidate = alpha + beta @ base + epsilon.

    Args:
        candidate: 1-D array of candidate factor returns.
        base_matrix: 2-D array (T × K) of base factor returns.

    Returns:
        (alpha, t_stat) tuple.
    """
    n_obs = len(candidate)
    if n_obs < 3:
        return 0.0, 0.0

    ones = np.ones((n_obs, 1))
    X = np.hstack([ones, base_matrix]) if base_matrix.shape[1] > 0 else ones

    try:
        beta, _, _, _ = np.linalg.lstsq(X, candidate, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 0.0

    alpha = float(beta[0])

    resid = candidate - X @ beta
    dof = n_obs - X.shape[1]
    if dof <= 0:
        return alpha, 0.0

    sigma2 = float(np.sum(resid ** 2) / dof)
    if sigma2 < EPSILON:
        return alpha, 0.0

    # WHY: variance of alpha = sigma^2 * (X'X)^-1[0,0]
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
        se_alpha = float(np.sqrt(sigma2 * xtx_inv[0, 0]))
    except np.linalg.LinAlgError:
        return alpha, 0.0

    if se_alpha < EPSILON:
        return alpha, 0.0

    return alpha, alpha / se_alpha


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
            These are the PASS pool candidates.
        base_spreads: Mapping of base factor name → DataFrame with ``date, spread``.
            These are always in the regression (not candidates for selection).
            If None, starts with an empty base.
        significance_threshold: Minimum |t-stat| for selection (default 2.0).
        max_factors: Maximum number of factors to select.

    Returns:
        ForwardSelectionResult with selected factors in order.
    """
    if base_spreads is None:
        base_spreads = {}

    # Align all series to common dates
    all_dates = None
    all_series: dict[str, pl.DataFrame] = {**base_spreads, **factor_spreads}
    for name, df in all_series.items():
        dates = df.select("date").unique()
        if all_dates is None:
            all_dates = dates
        else:
            all_dates = all_dates.join(dates, on="date", how="inner")

    if all_dates is None or len(all_dates) < 10:
        logger.warning("greedy_forward_selection: insufficient common dates (%d)",
                        0 if all_dates is None else len(all_dates))
        return ForwardSelectionResult()

    common_dates = all_dates.sort("date")

    # Build aligned numpy arrays
    def _align(df: pl.DataFrame) -> np.ndarray:
        return (
            common_dates.join(df.select("date", "spread"), on="date", how="left")
            ["spread"].fill_null(0.0).to_numpy()
        )

    base_arrays: dict[str, np.ndarray] = {
        name: _align(df) for name, df in base_spreads.items()
    }
    candidate_arrays: dict[str, np.ndarray] = {
        name: _align(df) for name, df in factor_spreads.items()
    }

    selected_names: list[str] = []
    selected_arrays: dict[str, np.ndarray] = {}
    result = ForwardSelectionResult()
    remaining = set(candidate_arrays.keys())

    for step in range(max_factors):
        if not remaining:
            break

        # Build current base matrix: base_spreads + selected so far
        base_cols = list(base_arrays.values()) + [
            selected_arrays[n] for n in selected_names
        ]
        if base_cols:
            base_matrix = np.column_stack(base_cols)
        else:
            base_matrix = np.empty((len(common_dates), 0))

        # Evaluate all remaining candidates
        best_name = None
        best_alpha = 0.0
        best_t = 0.0

        for name in remaining:
            alpha, t = _spanning_regression(candidate_arrays[name], base_matrix)
            sr = SpanningResult(
                factor_name=name, alpha=alpha, t_stat=t,
                selected=False,
            )
            result.all_candidates.append(sr)

            if abs(t) >= significance_threshold and abs(alpha) > abs(best_alpha):
                best_name = name
                best_alpha = alpha
                best_t = t

        if best_name is None:
            break

        # Select the best candidate
        selected_names.append(best_name)
        selected_arrays[best_name] = candidate_arrays[best_name]
        remaining.remove(best_name)
        result.selected_factors.append(SpanningResult(
            factor_name=best_name, alpha=best_alpha, t_stat=best_t,
            selected=True,
        ))

        # Backward elimination: check all selected factors
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
    """Remove selected factors whose alpha becomes insignificant.

    Modifies selected_names and selected_arrays in place.
    """
    changed = True
    while changed:
        changed = False
        for name in list(selected_names):
            # Build base = base_spreads + all selected EXCEPT this one
            others = [
                selected_arrays[n] for n in selected_names if n != name
            ]
            base_cols = list(base_arrays.values()) + others
            if base_cols:
                base_matrix = np.column_stack(base_cols)
            else:
                base_matrix = np.empty((len(selected_arrays[name]), 0))

            alpha, t = _spanning_regression(selected_arrays[name], base_matrix)

            if abs(t) < threshold:
                selected_names.remove(name)
                del selected_arrays[name]
                result.eliminated_factors.append(SpanningResult(
                    factor_name=name, alpha=alpha, t_stat=t,
                    selected=False,
                ))
                # Also remove from selected_factors list
                result.selected_factors = [
                    s for s in result.selected_factors if s.factor_name != name
                ]
                changed = True
                logger.info(
                    "backward elimination: removed %s (alpha=%.6f, t=%.2f)",
                    name, alpha, t,
                )
                break
