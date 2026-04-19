"""Macro-common factor profile.

Canonical test: cross-sectional t-test on per-asset TS betas (``ts_beta_p``).

Canonical rationale: a macro-common factor (VIX, gold, USD index) is a
single time series; each asset gets its own time-series β via
R_{i,t} = α_i + β_i·F_t + ε. The canonical hypothesis "the factor has
nonzero average exposure across assets" translates into a
cross-sectional t-test on the β distribution. With N=1 asset the test
degenerates; we fall back to reporting the single-asset TS β directly
and mark it with ``n_assets_used=1`` so diagnose() can warn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Self, TYPE_CHECKING

from factorlib._types import Diagnostic, FactorType, MetricOutput, PValue, Verdict
from factorlib.evaluation.profiles._base import (
    _diagnose,
    _insufficient_metrics,
    _memoized,
    _pv,
    _verdict_from_p,
    register_profile,
)

if TYPE_CHECKING:
    from factorlib.evaluation._protocol import Artifacts


@register_profile(FactorType.MACRO_COMMON)
@dataclass(frozen=True, slots=True)
class MacroCommonProfile:
    """Typed profile for a macro-common factor.

    ``n_periods`` here counts distinct dates in the prepared panel —
    the time dimension that per-asset TS regressions consume. The
    cross-sectional β t-test uses ``n_assets`` instead; both are
    exposed so downstream code can read whichever dimension is
    relevant without unpacking metadata.
    """

    # Identity
    factor_name: str
    n_periods: int
    n_assets: int

    # TS β (canonical)
    ts_beta_mean: float
    ts_beta_tstat: float
    ts_beta_p: PValue

    # Explanatory power
    mean_r_squared: float
    ts_beta_sign_consistency: float

    # Stability
    oos_survival_ratio: float
    oos_sign_flipped: bool
    beta_trend: float
    beta_trend_p: PValue

    insufficient_metrics: tuple[str, ...]  # see _base._insufficient_metrics

    CANONICAL_P_FIELD: ClassVar[str] = "ts_beta_p"
    P_VALUE_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "ts_beta_p", "beta_trend_p",
    })

    @property
    def canonical_p(self) -> PValue:
        return getattr(self, self.CANONICAL_P_FIELD)

    def verdict(self, threshold: float = 2.0) -> Verdict:
        return _verdict_from_p(self.canonical_p, threshold, self.n_periods)

    def diagnose(self) -> list[Diagnostic]:
        return _diagnose(self)

    @classmethod
    def from_artifacts(
        cls, artifacts: "Artifacts",
    ) -> tuple[Self, dict[str, MetricOutput]]:
        from factorlib.config import MacroCommonConfig
        from factorlib.metrics.oos import multi_split_oos_decay
        from factorlib.metrics.trend import ic_trend
        from factorlib.metrics.ts_beta import (
            mean_r_squared,
            ts_beta,
            ts_beta_sign_consistency,
        )

        config = artifacts.config
        if not isinstance(config, MacroCommonConfig):
            raise TypeError(
                f"MacroCommonProfile.from_artifacts expects MacroCommonConfig; "
                f"got {type(config).__name__}."
            )

        outputs: dict[str, MetricOutput] = dict(artifacts.metric_outputs)
        ts_betas_df = artifacts.get("beta_series")
        beta_values = artifacts.get("beta_values")
        n_assets = len(ts_betas_df)
        n_periods = int(artifacts.prepared["date"].n_unique())

        # N=1 degenerate case: cross-sectional t-test needs N>=2.
        # Fall back to the single-asset regression's own t-stat
        # (already computed in compute_ts_betas).
        if n_assets == 1:
            def _build_n1_ts_beta() -> MetricOutput:
                row = ts_betas_df.row(0, named=True)
                return MetricOutput(
                    name="ts_beta",
                    value=float(row["beta"]),
                    stat=float(row["t_stat"]),
                    metadata={
                        "n_assets": 1,
                        "p_value": 1.0,  # single-asset: suppress from BHY
                        "method": "single-asset TS regression (no cross-asset test)",
                    },
                )
            ts_beta_m = _memoized(outputs, "ts_beta", _build_n1_ts_beta)
        else:
            ts_beta_m = _memoized(outputs, "ts_beta", ts_beta, ts_betas_df)

        r2_m = _memoized(outputs, "mean_r_squared", mean_r_squared, ts_betas_df)
        sign_m = _memoized(
            outputs, "ts_beta_sign_consistency", ts_beta_sign_consistency, ts_betas_df,
        )
        oos_m = _memoized(outputs, "oos_decay", multi_split_oos_decay, beta_values)
        trend_m = _memoized(outputs, "ic_trend", ic_trend, beta_values)

        insufficient = _insufficient_metrics({
            "ts_beta_mean": ts_beta_m,
            "mean_r_squared": r2_m,
            "ts_beta_sign_consistency": sign_m,
            "beta_trend": trend_m,
        })

        profile = cls(
            factor_name=artifacts.factor_name,
            n_periods=n_periods,
            n_assets=n_assets,
            ts_beta_mean=float(ts_beta_m.value),
            ts_beta_tstat=float(ts_beta_m.stat or 0.0),
            ts_beta_p=_pv(ts_beta_m),
            mean_r_squared=float(r2_m.value),
            ts_beta_sign_consistency=float(sign_m.value),
            oos_survival_ratio=float(oos_m.value),
            oos_sign_flipped=bool(oos_m.metadata["sign_flipped"]),
            beta_trend=float(trend_m.value),
            beta_trend_p=_pv(trend_m),
            insufficient_metrics=insufficient,
        )
        return profile, outputs
