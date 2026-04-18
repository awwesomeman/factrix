"""Cross-sectional factor profile.

Canonical test: IC non-overlapping t-test (``ic_p``).

Canonical rationale: IC measures signal strength per-date and is the
workhorse statistic for cross-sectional strategies (Grinold & Kahn 2000).
Q1-Q5 spread exposes a different hypothesis (long-short portfolio alpha
> 0) and belongs in diagnose() as secondary evidence, not in verdict
— mixing them recreates the OR logic we deliberately retired with gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Self, TYPE_CHECKING

from factorlib._types import Diagnostic, FactorType, PValue, Verdict
from factorlib.evaluation.profiles._base import (
    _diagnose,
    _insufficient_metrics,
    _pv,
    _verdict_from_p,
    register_profile,
)

if TYPE_CHECKING:
    from factorlib.evaluation._protocol import Artifacts


@register_profile(FactorType.CROSS_SECTIONAL)
@dataclass(frozen=True, slots=True)
class CrossSectionalProfile:
    """Typed profile for a cross-sectional factor.

    Fields are grouped by concept via naming prefix (ic_*, spread_*,
    oos_*, q1_*). Flat layout keeps filter expressions ergonomic
    (``pl.col("ic_tstat") >= 2.0``) and polars interop trivial.
    """

    # Identity
    factor_name: str
    n_periods: int

    # IC family
    ic_mean: float
    ic_tstat: float
    ic_p: PValue
    ic_ir: float
    hit_rate: float
    hit_rate_p: PValue
    ic_trend: float
    ic_trend_p: PValue

    # Monotonicity
    monotonicity: float

    # Q1-Q5 spread
    q1_q5_spread: float
    spread_tstat: float
    spread_p: PValue

    # Concentration
    q1_concentration: float
    q1_concentration_eff_ratio: float

    # OOS stability
    oos_survival_ratio: float
    oos_sign_flipped: bool

    # Data quality context (for diagnose)
    median_universe_n: int

    # Orthogonalization (Step 6, opt-in; None when not applied)
    # r2_mean is the mean R² of the per-date residualization regression;
    # n_base is the number of basis factors used. Zero base = not applied.
    orthogonalize_r2_mean: float | None
    orthogonalize_n_base: int

    # Regime IC (T3.S2, opt-in; None when config.regime_labels is None)
    # min_tstat is the |t| of the weakest regime (conservative: if even
    # the weakest regime shows signal, all regimes do). consistent is
    # True when IC direction agrees across regimes.
    regime_ic_min_tstat: float | None
    regime_ic_consistent: bool | None

    # Multi-horizon IC (T3.S2, opt-in via config.multi_horizon_periods)
    # retention = IC(longest_horizon) / IC(shortest_horizon); None when
    # |IC(shortest)| < 1e-4 (no signal at the shortest horizon to
    # retain) or when the metric is not enabled. monotonic = |IC|
    # non-increasing as horizon grows. See docs/spike_level2_profile_integration.md
    # §3.4.2 — a single decay ratio can't distinguish sign-flip from
    # monotonic decay, so both shape and magnitude are needed.
    multi_horizon_ic_retention: float | None
    multi_horizon_ic_monotonic: bool | None

    # Spanning alpha (T3.S2, opt-in; None when config.spanning_base_spreads is None)
    # Tests whether the factor's per-date Q1-Q5 spread has alpha after
    # controlling for a user-supplied set of base-factor spreads. p is
    # NOT added to P_VALUE_FIELDS — canonical_p must remain singular.
    spanning_alpha_t: float | None
    spanning_alpha_p: PValue | None

    # Implementation
    turnover: float
    breakeven_cost: float
    net_spread: float

    insufficient_metrics: tuple[str, ...]  # see _base._insufficient_metrics

    CANONICAL_P_FIELD: ClassVar[str] = "ic_p"
    P_VALUE_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "ic_p", "hit_rate_p", "ic_trend_p", "spread_p",
    })

    @property
    def canonical_p(self) -> PValue:
        return getattr(self, self.CANONICAL_P_FIELD)

    def verdict(self, threshold: float = 2.0) -> Verdict:
        return _verdict_from_p(self.canonical_p, threshold, self.n_periods)

    def diagnose(self) -> list[Diagnostic]:
        return _diagnose(self)

    @classmethod
    def from_artifacts(cls, artifacts: "Artifacts") -> Self:
        # WHY: lazy imports — match existing pipeline style and avoid
        # pulling every metric module at package import time.
        from factorlib.config import CrossSectionalConfig
        from factorlib.metrics.ic import ic as ic_metric, ic_ir as ic_ir_metric
        from factorlib.metrics.quantile import quantile_spread
        from factorlib.metrics.monotonicity import monotonicity
        from factorlib.metrics.concentration import q1_concentration
        from factorlib.metrics.hit_rate import hit_rate
        from factorlib.metrics.trend import ic_trend
        from factorlib.metrics.oos import multi_split_oos_decay
        from factorlib.metrics._helpers import _median_universe_size
        from factorlib.metrics.tradability import (
            breakeven_cost, net_spread, turnover,
        )

        config = artifacts.config
        if not isinstance(config, CrossSectionalConfig):
            raise TypeError(
                f"CrossSectionalProfile.from_artifacts expects "
                f"CrossSectionalConfig; got {type(config).__name__}."
            )

        fp = config.forward_periods
        ic_series = artifacts.get("ic_series")
        ic_values = artifacts.get("ic_values")
        spread_series = artifacts.get("spread_series")

        ic_m = ic_metric(ic_series, forward_periods=fp)
        ic_ir_m = ic_ir_metric(ic_series)
        hit_m = hit_rate(ic_values, forward_periods=fp)
        trend_m = ic_trend(ic_values)
        mono_m = monotonicity(
            artifacts.prepared, forward_periods=fp, n_groups=config.n_groups,
        )
        oos = multi_split_oos_decay(ic_values)
        spread_m = quantile_spread(
            artifacts.prepared,
            forward_periods=fp,
            n_groups=config.n_groups,
            _precomputed_series=spread_series,
        )
        turn_m = turnover(artifacts.prepared)
        be_m = breakeven_cost(spread_m.value, turn_m.value)
        ns_m = net_spread(
            spread_m.value, turn_m.value, config.estimated_cost_bps,
        )
        conc_m = q1_concentration(
            artifacts.prepared, forward_periods=fp, q_top=config.q_top,
        )

        ortho_stats = artifacts.intermediates.get("ortho_stats")
        if ortho_stats is not None:
            ortho_r2 = float(ortho_stats["r2_mean"][0])
            ortho_n_base = int(ortho_stats["n_base"][0])
        else:
            ortho_r2 = None
            ortho_n_base = 0

        regime_stats = artifacts.intermediates.get("regime_stats")
        if regime_stats is not None:
            regime_min_t = float(regime_stats["min_tstat"][0])
            regime_consistent = bool(regime_stats["consistent"][0])
        else:
            regime_min_t = None
            regime_consistent = None

        mh_stats = artifacts.intermediates.get("multi_horizon_stats")
        if mh_stats is not None:
            retention_val = mh_stats["retention"][0]
            mh_retention = float(retention_val) if retention_val is not None else None
            mh_monotonic = bool(mh_stats["monotonic"][0])
        else:
            mh_retention = None
            mh_monotonic = None

        sp_stats = artifacts.intermediates.get("spanning_stats")
        if sp_stats is not None:
            spanning_t = float(sp_stats["t"][0])
            spanning_p: "PValue | None" = float(sp_stats["p"][0])
        else:
            spanning_t = None
            spanning_p = None

        insufficient = _insufficient_metrics({
            "ic_mean": ic_m,
            "ic_ir": ic_ir_m,
            "hit_rate": hit_m,
            "ic_trend": trend_m,
            "monotonicity": mono_m,
            "q1_q5_spread": spread_m,
            "q1_concentration": conc_m,
            "turnover": turn_m,
        })
        return cls(
            factor_name=artifacts.factor_name,
            n_periods=int(ic_m.metadata.get("n_periods", len(ic_series))),
            ic_mean=float(ic_m.value),
            ic_tstat=float(ic_m.stat or 0.0),
            ic_p=_pv(ic_m),
            ic_ir=float(ic_ir_m.value),
            hit_rate=float(hit_m.value),
            hit_rate_p=_pv(hit_m),
            ic_trend=float(trend_m.value),
            ic_trend_p=_pv(trend_m),
            monotonicity=float(mono_m.value),
            q1_q5_spread=float(spread_m.value),
            spread_tstat=float(spread_m.stat or 0.0),
            spread_p=_pv(spread_m),
            q1_concentration=float(conc_m.value),
            q1_concentration_eff_ratio=float(
                conc_m.metadata.get("ratio_eff_to_total", 1.0)
            ),
            oos_survival_ratio=float(oos.survival_ratio),
            oos_sign_flipped=bool(oos.sign_flipped),
            median_universe_n=int(_median_universe_size(artifacts.prepared)),
            orthogonalize_r2_mean=ortho_r2,
            orthogonalize_n_base=ortho_n_base,
            regime_ic_min_tstat=regime_min_t,
            regime_ic_consistent=regime_consistent,
            multi_horizon_ic_retention=mh_retention,
            multi_horizon_ic_monotonic=mh_monotonic,
            spanning_alpha_t=spanning_t,
            spanning_alpha_p=spanning_p,
            turnover=float(turn_m.value),
            breakeven_cost=float(be_m.value),
            net_spread=float(ns_m.value),
            insufficient_metrics=insufficient,
        )
