"""Size of the single-restriction Wald consumers under overlapping returns.

``common_quantile_spread`` and ``common_asymmetry`` test one linear
restriction on an OLS fit, so they resolve their HAC reference through
``_resolve_scalar_wald_hac`` (the scalar HAR recipe) rather than through the
``max(auto_bartlett(T), h - 1)`` rule the K-restriction Wald paths keep.

On the common-factor null of ``statistical-methods`` section 6 (300
replications, seed ``20260830 + rep``) the change measures, at a nominal 5%:

| metric | phi | T, h | narrow rule | shipped rule |
|---|---|---|---|---|
| `common_asymmetry` | 0.0 | 60, 5 | 15.3% | 8.0% |
| `common_asymmetry` | 0.0 | 60, 21 | 34.0% | 5.7% |
| `common_quantile_spread` | 0.0 | 60, 21 | 16.3% | 0.0% |
| `common_quantile_spread` | 0.0 | 240, 21 | 7.7% | 2.0% |

This module re-runs the contrast on a cheap single-period-per-date panel at
a cut replication count, so the bounds characterise rather than pin. On that
narrower null at ``T = 120, h = 5``, 1000 replications from the same seed
stream measure ``common_asymmetry`` at 15.3% under the narrow rule against
8.7% shipped, and ``common_quantile_spread`` at 5.4% shipped.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
from factrix._stats import _ols_nw_multivariate, _resolve_nw_lags, _wald_p_linear
from factrix.metrics.common_asymmetry import common_asymmetry
from factrix.metrics.common_quantile import common_quantile_spread

T = 120
H = 5
N_GROUPS = 5
N_REPS = 200
SEED = 20260830
ALPHA = 0.05


def _null_panel(rep: int) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """One period per date: iid factor, returns overlapping ``H``-period sums."""
    rng = np.random.default_rng(SEED + rep)
    factor = rng.standard_normal(T)
    shocks = rng.standard_normal(T + H)
    cumulative = np.cumsum(np.concatenate([[0.0], shocks]))
    ret = (cumulative[H : H + T] - cumulative[:T]) / np.sqrt(H)
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(T)]
    frame = pl.DataFrame(
        {
            "date": dates,
            "asset_id": ["A"] * T,
            "factor": factor,
            "forward_return": ret,
        }
    )
    return frame, factor, ret


def _narrow_rule_bucket_spread_p(factor: np.ndarray, ret: np.ndarray) -> float:
    """The bucket-spread contrast under the retired ``_resolve_nw_lags`` rule.

    Written out here rather than reached through the metric: the metric no
    longer offers the narrow rule, and this is the "before" column.
    """
    ranks = pl.Series(factor).rank(method="ordinal").to_numpy().astype(np.int64)
    bucket = np.minimum(((ranks - 1) * N_GROUPS) // T, N_GROUPS - 1)
    design = np.zeros((T, N_GROUPS))
    design[np.arange(T), bucket] = 1.0
    restriction = np.zeros((1, N_GROUPS))
    restriction[0, N_GROUPS - 1] = 1.0
    restriction[0, 0] = -1.0
    beta, cov, _ = _ols_nw_multivariate(ret, design, lags=_resolve_nw_lags(T, None, H))
    _, p = _wald_p_linear(beta, cov, restriction, q=0.0, df_denom=T - N_GROUPS)
    return float(p)


def _narrow_rule_asymmetry_p(factor: np.ndarray, ret: np.ndarray) -> float:
    """The long-plus-short contrast under the retired ``_resolve_nw_lags`` rule."""
    design = np.column_stack([(factor > 0).astype(float), (factor < 0).astype(float)])
    beta, cov, _ = _ols_nw_multivariate(
        ret, design, lags=_resolve_nw_lags(len(ret), None, H)
    )
    _, p = _wald_p_linear(
        beta, cov, np.array([[1.0, 1.0]]), q=0.0, df_denom=len(ret) - 2
    )
    return float(p)


def _rejection_rate(p_values: list[float]) -> float:
    return float(np.mean([p < ALPHA for p in p_values]))


class TestSingleRestrictionOverlapSize:
    def test_quantile_spread_is_better_sized_than_the_narrow_rule(self):
        shipped, narrow = [], []
        for rep in range(N_REPS):
            frame, factor, ret = _null_panel(rep)
            shipped.append(common_quantile_spread(frame, overlap_periods=H).p_value)
            narrow.append(_narrow_rule_bucket_spread_p(factor, ret))
        shipped_size = _rejection_rate(shipped)
        assert shipped_size < _rejection_rate(narrow)
        assert shipped_size <= 0.10

    def test_asymmetry_is_better_sized_than_the_narrow_rule(self):
        shipped, narrow = [], []
        for rep in range(N_REPS):
            frame, factor, ret = _null_panel(rep)
            shipped.append(common_asymmetry(frame, overlap_periods=H).p_value)
            narrow.append(_narrow_rule_asymmetry_p(factor, ret))
        shipped_size = _rejection_rate(shipped)
        assert shipped_size < _rejection_rate(narrow)
        assert shipped_size <= 0.12


class TestShortEffectiveSampleIsFlagged:
    """The regime the recipe does not fix is reported, not tuned away."""

    def test_long_horizon_flags_the_effective_period_shortage(self):
        # 120 periods at h = 21 leave 5 independent observations, below the
        # library's 10-period floor for a series statistic.
        rng = np.random.default_rng(SEED)
        frame = pl.DataFrame(
            {
                "date": [date(2020, 1, 1) + timedelta(days=i) for i in range(T)],
                "asset_id": ["A"] * T,
                "factor": rng.standard_normal(T),
                "forward_return": rng.standard_normal(T),
            }
        )
        for out in (
            common_quantile_spread(frame, overlap_periods=21),
            common_asymmetry(frame, overlap_periods=21),
        ):
            assert "unreliable_se_short_periods" in out.warning_codes

    def test_persistent_factor_beyond_the_horizon_is_flagged(self):
        rng = np.random.default_rng(SEED)
        noise = rng.standard_normal(T)
        factor = np.empty(T)
        factor[0] = noise[0]
        for i in range(1, T):
            factor[i] = 0.95 * factor[i - 1] + np.sqrt(1 - 0.95**2) * noise[i]
        frame = pl.DataFrame(
            {
                "date": [date(2020, 1, 1) + timedelta(days=i) for i in range(T)],
                "asset_id": ["A"] * T,
                "factor": factor,
                "forward_return": rng.standard_normal(T),
            }
        )
        for out in (
            common_quantile_spread(frame, overlap_periods=H),
            common_asymmetry(frame, overlap_periods=H),
        ):
            assert "serial_correlation_detected" in out.warning_codes
