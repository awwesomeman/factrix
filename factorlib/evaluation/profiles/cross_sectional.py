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
    oos_decay: float
    oos_sign_flipped: bool

    # Data quality context (for diagnose)
    median_universe_n: int

    # Implementation
    turnover: float
    breakeven_cost: float
    net_spread: float

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
            oos_decay=float(oos.decay_ratio),
            oos_sign_flipped=bool(oos.sign_flipped),
            median_universe_n=int(_median_universe_size(artifacts.prepared)),
            turnover=float(turn_m.value),
            breakeven_cost=float(be_m.value),
            net_spread=float(ns_m.value),
        )
