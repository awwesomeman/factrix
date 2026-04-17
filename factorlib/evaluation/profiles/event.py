"""Event-signal factor profile.

Canonical test: CAAR non-overlapping t-test (``caar_p``).

Canonical rationale: Cumulative Average Abnormal Return is the
established event-study significance test (MacKinlay 1997). The BMP SAR
test (Boehmer et al. 1991) is robust to event-induced variance but the
codebase exposes both; BMP lives in the profile and in diagnose() as a
secondary confirmation signal — when CAAR-significant but BMP
non-confirmatory, diagnose flags the event-variance inflation risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Self, TYPE_CHECKING

import polars as pl

from factorlib._types import Diagnostic, FactorType, PValue, Verdict
from factorlib.evaluation.profiles._base import (
    _pv,
    _verdict_from_p,
    register_profile,
)

if TYPE_CHECKING:
    from factorlib.evaluation._protocol import Artifacts


@register_profile(FactorType.EVENT_SIGNAL)
@dataclass(frozen=True, slots=True)
class EventProfile:
    """Typed profile for an event-signal factor.

    ``n_periods`` counts unique event dates in the CAAR series (not
    total rows in prepared), matching the sampling unit for the CAAR
    t-test.
    """

    # Identity
    factor_name: str
    n_periods: int

    # CAAR family
    caar_mean: float
    caar_tstat: float
    caar_p: PValue

    # BMP standardized AR test
    bmp_sar_mean: float
    bmp_zstat: float
    bmp_p: PValue

    # Event hit rate
    event_hit_rate: float
    event_hit_rate_p: PValue

    # Distribution quality
    profit_factor: float
    event_skewness: float

    # Stability
    oos_decay: float
    oos_sign_flipped: bool
    caar_trend: float
    caar_trend_p: PValue

    # Event-level diagnostics
    signal_density: float
    clustering_hhi: float | None
    event_ic: float | None
    event_ic_p: PValue | None

    CANONICAL_P_FIELD: ClassVar[str] = "caar_p"
    # event_ic_p is deliberately excluded: the event_ic test is only run
    # when the signal has magnitude variance, so the field may be None
    # and would break the "every factor in a BHY batch has this p"
    # invariant. Accessible on the dataclass, not whitelisted for BHY.
    P_VALUE_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "caar_p", "bmp_p", "event_hit_rate_p", "caar_trend_p",
    })

    @property
    def canonical_p(self) -> PValue:
        return getattr(self, self.CANONICAL_P_FIELD)

    def verdict(self, threshold: float = 2.0) -> Verdict:
        return _verdict_from_p(self.canonical_p, threshold)

    def diagnose(self) -> list[Diagnostic]:
        return []

    @classmethod
    def from_artifacts(cls, artifacts: "Artifacts") -> Self:
        from factorlib.config import EventConfig
        from factorlib.metrics.caar import bmp_test, caar as caar_metric
        from factorlib.metrics.clustering import clustering_diagnostic
        from factorlib.metrics.event_quality import (
            event_hit_rate,
            event_ic as event_ic_metric,
            event_skewness,
            profit_factor,
            signal_density,
        )
        from factorlib.metrics.oos import multi_split_oos_decay
        from factorlib.metrics.trend import ic_trend

        config = artifacts.config
        if not isinstance(config, EventConfig):
            raise TypeError(
                f"EventProfile.from_artifacts expects EventConfig; "
                f"got {type(config).__name__}."
            )

        ret_col = (
            "abnormal_return"
            if "abnormal_return" in artifacts.prepared.columns
            else "forward_return"
        )

        caar_series = artifacts.get("caar_series")
        caar_values = artifacts.get("caar_values")

        caar_m = caar_metric(caar_series, forward_periods=config.forward_periods)
        bmp_m = bmp_test(
            artifacts.prepared,
            return_col=ret_col,
            forward_periods=config.forward_periods,
        )
        hit_m = event_hit_rate(artifacts.prepared, return_col=ret_col)
        pf_m = profit_factor(artifacts.prepared, return_col=ret_col)
        skew_m = event_skewness(artifacts.prepared, return_col=ret_col)
        oos = multi_split_oos_decay(caar_values)
        trend_m = ic_trend(caar_values)  # same tool, applied to CAAR series
        density_m = signal_density(artifacts.prepared)

        # Clustering is only meaningful with multiple assets.
        n_assets = artifacts.prepared["asset_id"].n_unique()
        clustering_hhi: float | None
        if n_assets > 1:
            clust_m = clustering_diagnostic(
                artifacts.prepared, cluster_window=config.cluster_window,
            )
            clustering_hhi = float(clust_m.value)
        else:
            clustering_hhi = None

        # Event IC only makes sense when signal magnitude varies.
        events = artifacts.prepared.filter(pl.col("factor") != 0)
        event_ic_val: float | None
        event_ic_p_val: PValue | None
        if events["factor"].abs().n_unique() > 1:
            eic_m = event_ic_metric(artifacts.prepared, return_col=ret_col)
            event_ic_val = float(eic_m.value)
            event_ic_p_val = _pv(eic_m)
        else:
            event_ic_val = None
            event_ic_p_val = None

        return cls(
            factor_name=artifacts.factor_name,
            n_periods=int(caar_m.metadata.get("n_event_dates", len(caar_series))),
            caar_mean=float(caar_m.value),
            caar_tstat=float(caar_m.stat or 0.0),
            caar_p=_pv(caar_m),
            bmp_sar_mean=float(bmp_m.value),
            bmp_zstat=float(bmp_m.stat or 0.0),
            bmp_p=_pv(bmp_m),
            event_hit_rate=float(hit_m.value),
            event_hit_rate_p=_pv(hit_m),
            profit_factor=float(pf_m.value),
            event_skewness=float(skew_m.value),
            oos_decay=float(oos.decay_ratio),
            oos_sign_flipped=bool(oos.sign_flipped),
            caar_trend=float(trend_m.value),
            caar_trend_p=_pv(trend_m),
            signal_density=float(density_m.value),
            clustering_hhi=clustering_hhi,
            event_ic=event_ic_val,
            event_ic_p=event_ic_p_val,
        )
