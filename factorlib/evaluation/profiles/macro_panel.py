"""Macro-panel factor profile.

Canonical test: Fama-MacBeth λ Newey-West t-test (``fm_beta_p``).

Canonical rationale: Macro-panel factors (state variables that vary
by country / region, e.g. CPI) are evaluated via the Fama-MacBeth
(1973) procedure. The λ series is autocorrelated by construction, so
we use Newey-West HAC standard errors (Newey & West 1987). The pooled
OLS check provides a sign-consistency cross-validation and is exposed
as ``pooled_beta`` for diagnose(), not as the primary test.
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
    _verdict_with_warnings,
    register_profile,
)

if TYPE_CHECKING:
    from factorlib.evaluation._protocol import Artifacts


@register_profile(FactorType.MACRO_PANEL)
@dataclass(frozen=True, slots=True)
class MacroPanelProfile:
    """Typed profile for a macro-panel factor.

    ``n_periods`` is the length of the FM β time series — one observation
    per date with a valid cross-sectional regression.
    """

    # Identity
    factor_name: str
    n_periods: int
    median_cross_section_n: int
    cross_section_below_min: bool

    # Fama-MacBeth (canonical)
    fm_beta_mean: float
    fm_beta_tstat: float
    fm_beta_p: PValue
    fm_newey_west_lags: int

    # Pooled OLS (confirmatory)
    pooled_beta: float
    pooled_beta_tstat: float
    pooled_beta_p: PValue

    # Sign consistency
    beta_sign_consistency: float

    # Stability
    oos_survival_ratio: float
    oos_sign_flipped: bool
    beta_trend: float
    beta_trend_p: PValue

    # Portfolio (tercile quantile spread)
    quantile_spread: float
    spread_tstat: float
    spread_p: PValue
    turnover: float
    breakeven_cost: float
    net_spread: float

    insufficient_metrics: tuple[str, ...]  # see _base._insufficient_metrics

    CANONICAL_P_FIELD: ClassVar[str] = "fm_beta_p"
    P_VALUE_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "fm_beta_p", "pooled_beta_p", "beta_trend_p", "spread_p",
    })

    @property
    def canonical_p(self) -> PValue:
        return getattr(self, self.CANONICAL_P_FIELD)

    def verdict(self, threshold: float = 2.0) -> Verdict:
        return _verdict_with_warnings(self, threshold)

    def diagnose(self) -> list[Diagnostic]:
        return _diagnose(self)

    @classmethod
    def from_artifacts(
        cls, artifacts: "Artifacts",
    ) -> tuple[Self, dict[str, MetricOutput]]:
        from factorlib.config import MacroPanelConfig
        from factorlib.metrics._helpers import _median_universe_size
        from factorlib.metrics.fama_macbeth import (
            beta_sign_consistency,
            fama_macbeth,
            pooled_ols,
        )
        from factorlib.metrics.oos import multi_split_oos_decay
        from factorlib.metrics.quantile import quantile_spread
        from factorlib.metrics.tradability import (
            breakeven_cost, net_spread, turnover,
        )
        from factorlib.metrics.trend import ic_trend

        config = artifacts.config
        if not isinstance(config, MacroPanelConfig):
            raise TypeError(
                f"MacroPanelProfile.from_artifacts expects MacroPanelConfig; "
                f"got {type(config).__name__}."
            )

        outputs: dict[str, MetricOutput] = dict(artifacts.metric_outputs)
        fp = config.forward_periods
        beta_series = artifacts.get("beta_series")
        beta_values = artifacts.get("beta_values")
        spread_series = artifacts.get("spread_series")
        median_xs_n = int(_median_universe_size(artifacts.prepared))

        fm_m = _memoized(outputs, "fm_beta", fama_macbeth, beta_series)
        pooled_m = _memoized(outputs, "pooled_beta", pooled_ols, artifacts.prepared)
        sign_m = _memoized(outputs, "beta_sign_consistency", beta_sign_consistency, beta_series)
        oos_m = _memoized(outputs, "oos_decay", multi_split_oos_decay, beta_values)
        trend_m = _memoized(outputs, "beta_trend", ic_trend, beta_values, name="beta_trend")
        spread_m = _memoized(
            outputs, "quantile_spread", quantile_spread,
            artifacts.prepared,
            forward_periods=fp,
            n_groups=config.n_groups,
            _precomputed_series=spread_series,
        )
        turn_m = _memoized(outputs, "turnover", turnover, artifacts.prepared)
        be_m = _memoized(
            outputs, "breakeven_cost", breakeven_cost,
            spread_m.value, turn_m.value,
        )
        ns_m = _memoized(
            outputs, "net_spread", net_spread,
            spread_m.value, turn_m.value, config.estimated_cost_bps,
        )

        insufficient = _insufficient_metrics({
            "fm_beta_mean": fm_m,
            "pooled_beta": pooled_m,
            "beta_sign_consistency": sign_m,
            "beta_trend": trend_m,
            "quantile_spread": spread_m,
            "turnover": turn_m,
        })

        profile = cls(
            factor_name=artifacts.factor_name,
            n_periods=int(fm_m.metadata.get("n_periods", len(beta_series))),
            median_cross_section_n=median_xs_n,
            cross_section_below_min=median_xs_n < config.min_cross_section,
            fm_beta_mean=float(fm_m.value),
            fm_beta_tstat=float(fm_m.stat or 0.0),
            fm_beta_p=_pv(fm_m),
            fm_newey_west_lags=int(fm_m.metadata.get("newey_west_lags", 0)),
            pooled_beta=float(pooled_m.value),
            pooled_beta_tstat=float(pooled_m.stat or 0.0),
            pooled_beta_p=_pv(pooled_m),
            beta_sign_consistency=float(sign_m.value),
            oos_survival_ratio=float(oos_m.value),
            oos_sign_flipped=bool(oos_m.metadata["sign_flipped"]),
            beta_trend=float(trend_m.value),
            beta_trend_p=_pv(trend_m),
            quantile_spread=float(spread_m.value),
            spread_tstat=float(spread_m.stat or 0.0),
            spread_p=_pv(spread_m),
            turnover=float(turn_m.value),
            breakeven_cost=float(be_m.value),
            net_spread=float(ns_m.value),
            insufficient_metrics=insufficient,
        )
        return profile, outputs
