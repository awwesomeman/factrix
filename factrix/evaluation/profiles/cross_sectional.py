"""Cross-sectional factor profile.

Canonical test: IC non-overlapping t-test (``ic_p``).

Canonical rationale: IC measures signal strength per-date and is the
workhorse statistic for cross-sectional strategies (Grinold & Kahn 2000).
long-short spread exposes a different hypothesis (long-short portfolio alpha
> 0) and belongs in diagnose() as secondary evidence, not in verdict
— mixing them recreates the OR logic we deliberately retired with gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Self, TYPE_CHECKING

from factrix._types import Diagnostic, FactorType, MetricOutput, PValue, Verdict
from factrix.evaluation.profiles._base import (
    _diagnose,
    _insufficient_metrics,
    _memoized,
    _pv,
    _verdict_with_warnings,
    register_profile,
)

if TYPE_CHECKING:
    from factrix.evaluation._protocol import Artifacts


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

    # Quantile long-short spread
    quantile_spread: float
    spread_tstat: float
    spread_p: PValue

    # Concentration
    top_concentration: float
    top_concentration_eff_ratio: float

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
    # Tests whether the factor's per-date long-short spread has alpha after
    # controlling for a user-supplied set of base-factor spreads. p is
    # NOT added to P_VALUE_FIELDS — canonical_p must remain singular.
    spanning_alpha_t: float | None
    spanning_alpha_p: PValue | None

    # Implementation
    #
    # turnover: rank-stability (1 − mean Spearman ρ), mid-rank sensitive —
    #   diagnostic only, NOT the cost driver.
    # turnover_jaccard: fraction of Q1/Q_n membership replaced per
    #   rebalance (Novy-Marx & Velikov τ) — this is what drives the bps
    #   arithmetic in breakeven_cost / net_spread.
    turnover: float
    turnover_jaccard: float
    breakeven_cost: float
    net_spread: float

    insufficient_metrics: tuple[str, ...]  # see _base._insufficient_metrics

    # Newey-West HAC IC p-value on the overlapping series. Same null
    # (mean IC = 0) as ``ic_p``; the HAC correction absorbs autocorrelation
    # from overlapping forward returns rather than dropping samples. kw-only
    # default of PValue(1.0) keeps old kwarg-construction paths working
    # without modification; reaches P_VALUE_FIELDS so users can
    # ``multiple_testing_correct(p_source="ic_nw_p")``.
    ic_nw_p: PValue = PValue(1.0)

    CANONICAL_P_FIELD: ClassVar[str] = "ic_p"
    P_VALUE_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "ic_p", "ic_nw_p", "hit_rate_p", "ic_trend_p", "spread_p",
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
        # WHY: lazy imports — match existing pipeline style and avoid
        # pulling every metric module at package import time.
        from factrix.config import CrossSectionalConfig
        from factrix.metrics.ic import (
            ic as ic_metric,
            ic_ir as ic_ir_metric,
            ic_newey_west,
        )
        from factrix.metrics.quantile import quantile_spread
        from factrix.metrics.monotonicity import monotonicity
        from factrix.metrics.concentration import top_concentration
        from factrix.metrics.hit_rate import hit_rate
        from factrix.metrics.trend import ic_trend
        from factrix.metrics.oos import multi_split_oos_decay
        from factrix.metrics._helpers import _median_universe_size
        from factrix.metrics.tradability import (
            breakeven_cost, net_spread, turnover, turnover_jaccard,
        )

        config = artifacts.config
        if not isinstance(config, CrossSectionalConfig):
            raise TypeError(
                f"CrossSectionalProfile.from_artifacts expects "
                f"CrossSectionalConfig; got {type(config).__name__}."
            )

        # Start from what the pipeline already stashed (Level 2 opt-in
        # metrics live there). Copy so we don't mutate the input artifacts.
        outputs: dict[str, MetricOutput] = dict(artifacts.metric_outputs)
        fp = config.forward_periods
        ic_series = artifacts.get("ic_series")
        ic_values = artifacts.get("ic_values")
        spread_series = artifacts.get("spread_series")

        ic_m = _memoized(outputs, "ic", ic_metric, ic_series, forward_periods=fp)
        ic_nw_m = _memoized(
            outputs, "ic_newey_west", ic_newey_west, ic_series, forward_periods=fp,
        )
        ic_ir_m = _memoized(outputs, "ic_ir", ic_ir_metric, ic_series)
        hit_m = _memoized(outputs, "hit_rate", hit_rate, ic_values, forward_periods=fp)
        trend_m = _memoized(outputs, "ic_trend", ic_trend, ic_values)
        mono_m = _memoized(
            outputs, "monotonicity", monotonicity,
            artifacts.prepared, forward_periods=fp, n_groups=config.n_groups,
        )
        oos_m = _memoized(outputs, "oos_decay", multi_split_oos_decay, ic_values)
        spread_m = _memoized(
            outputs, "quantile_spread", quantile_spread,
            artifacts.prepared,
            forward_periods=fp,
            n_groups=config.n_groups,
            _precomputed_series=spread_series,
        )
        # Only turnover_jaccard's units align with bps cost arithmetic;
        # turnover is kept for diagnostic side-by-side reporting.
        turn_m = _memoized(
            outputs, "turnover", turnover,
            artifacts.prepared, forward_periods=fp,
        )
        turn_jac_m = _memoized(
            outputs, "turnover_jaccard", turnover_jaccard,
            artifacts.prepared,
            forward_periods=fp, n_groups=config.n_groups,
        )
        be_m = _memoized(
            outputs, "breakeven_cost", breakeven_cost,
            spread_m.value, turn_jac_m.value,
        )
        ns_m = _memoized(
            outputs, "net_spread", net_spread,
            spread_m.value, turn_jac_m.value, config.estimated_cost_bps,
        )
        # Q1 = top 1/n_groups — mirrors the quantile_spread Q1 definition
        # so top_concentration and quantile_spread report on the same bucket.
        conc_m = _memoized(
            outputs, "top_concentration", top_concentration,
            artifacts.prepared, forward_periods=fp, q_top=1.0 / config.n_groups,
        )

        ortho_stats = artifacts.intermediates.get("ortho_stats")
        if ortho_stats is not None:
            ortho_r2 = float(ortho_stats["r2_mean"][0])
            ortho_n_base = int(ortho_stats["n_base"][0])
        else:
            ortho_r2 = None
            ortho_n_base = 0

        # Level 2 opt-in MetricOutputs (regime_ic / multi_horizon_ic /
        # spanning_alpha) are already in `outputs` via the pipeline's
        # _augment_level2_intermediates. Here we just read the summary
        # DataFrames for Profile scalar fields.
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
            "quantile_spread": spread_m,
            "top_concentration": conc_m,
            "turnover": turn_m,
            "turnover_jaccard": turn_jac_m,
        })
        profile = cls(
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
            quantile_spread=float(spread_m.value),
            spread_tstat=float(spread_m.stat or 0.0),
            spread_p=_pv(spread_m),
            top_concentration=float(conc_m.value),
            top_concentration_eff_ratio=float(
                conc_m.metadata.get("ratio_eff_to_total", 1.0)
            ),
            oos_survival_ratio=float(oos_m.value),
            oos_sign_flipped=bool(oos_m.metadata["sign_flipped"]),
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
            turnover_jaccard=float(turn_jac_m.value),
            breakeven_cost=float(be_m.value),
            net_spread=float(ns_m.value),
            insufficient_metrics=insufficient,
            ic_nw_p=_pv(ic_nw_m),
        )
        return profile, outputs
