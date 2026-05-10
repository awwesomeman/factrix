"""IC PANEL + FM PANEL procedures emit ``StatCode.P_HH`` (#184).

Procedure-side wiring: when ``forward_periods > 1`` the IC / FM
procedures populate ``P_HH`` alongside ``P``, set the kernel + clamp
metadata, and ``HansenHodrick`` dispatched via ``bhy(estimator=...)``
reads it. ``forward_periods=1`` (no overlap) skips the emission so the
estimator lands on a missing-stat error rather than aliasing NW.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest
from factrix import AnalysisConfig, Metric, evaluate
from factrix._codes import StatCode, WarningCode
from factrix._errors import UserInputError
from factrix._multi_factor import bhy
from factrix.stats import HansenHodrick


def _build_panel(
    *,
    n_dates: int,
    n_assets: int,
    seed: int,
    factor_strength: float = 0.0,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = dt.date(2024, 1, 1)
    rows: list[dict[str, object]] = []
    for i in range(n_dates):
        d = start + dt.timedelta(days=i)
        fwd = rng.standard_normal(n_assets)
        noise = rng.standard_normal(n_assets)
        factor = factor_strength * fwd + (1.0 - factor_strength) * noise
        for j in range(n_assets):
            rows.append(
                {
                    "date": d,
                    "asset_id": f"A{j:03d}",
                    "factor": float(factor[j]),
                    "forward_return": float(fwd[j]),
                }
            )
    return pl.DataFrame(rows)


@pytest.mark.parametrize("metric", [Metric.IC, Metric.FM])
class TestProcedureEmitsPHh:
    def test_emits_p_hh_when_overlap(self, metric: Metric) -> None:
        panel = _build_panel(n_dates=80, n_assets=15, seed=1)
        cfg = AnalysisConfig.individual_continuous(metric=metric, forward_periods=5)
        profile = evaluate(panel, cfg)
        assert StatCode.P_HH in profile.stats
        assert StatCode.T_HH in profile.stats
        p_hh = profile.stats[StatCode.P_HH]
        t_hh = profile.stats[StatCode.T_HH]
        assert 0.0 <= p_hh <= 1.0
        assert np.isfinite(t_hh)
        meta = profile.metadata[StatCode.P_HH]
        assert meta["kernel"] == "rectangular"
        assert meta["variance_clamped"] is False
        # Same metadata mirrored under T_HH so single-key lookup stays honest.
        assert profile.metadata[StatCode.T_HH] == meta

    def test_no_emission_when_forward_periods_one(self, metric: Metric) -> None:
        panel = _build_panel(n_dates=80, n_assets=15, seed=2)
        cfg = AnalysisConfig.individual_continuous(metric=metric, forward_periods=1)
        profile = evaluate(panel, cfg)
        assert StatCode.P_HH not in profile.stats
        assert StatCode.T_HH not in profile.stats
        assert StatCode.P in profile.stats

    def test_bhy_dispatches_hh_p_value(self, metric: Metric) -> None:
        panel = _build_panel(n_dates=80, n_assets=15, seed=3)
        cfg = AnalysisConfig.individual_continuous(metric=metric, forward_periods=5)
        profile = evaluate(panel, cfg)
        family = bhy([profile], estimator=HansenHodrick())
        # The single-profile family must read p from StatCode.P_HH, not the
        # NW-canonical primary_p — confirm by comparing to the source value.
        assert family.profiles[0].stats[StatCode.P_HH] == profile.stats[StatCode.P_HH]

    def test_bhy_missing_p_hh_raises_user_input_error(self, metric: Metric) -> None:
        panel = _build_panel(n_dates=80, n_assets=15, seed=4)
        cfg = AnalysisConfig.individual_continuous(metric=metric, forward_periods=1)
        profile = evaluate(panel, cfg)
        with pytest.raises(UserInputError) as exc:
            bhy([profile], estimator=HansenHodrick())
        assert exc.value.field == "estimator"
        assert "P_HH" in (exc.value.expected or "")


class TestNegativeVarianceWarning:
    def test_clamp_emits_warning(self) -> None:
        # Construct a panel where the IC series is strongly anti-correlated
        # at lag 1 by alternating factor sign per date — γ₀ + 2γ₁ < 0 for h=2.
        rng = np.random.default_rng(0)
        n_dates, n_assets = 60, 12
        start = dt.date(2024, 1, 1)
        rows: list[dict[str, object]] = []
        for i in range(n_dates):
            d = start + dt.timedelta(days=i)
            fwd = rng.standard_normal(n_assets)
            sign = 1.0 if i % 2 == 0 else -1.0
            factor = sign * fwd + 0.01 * rng.standard_normal(n_assets)
            for j in range(n_assets):
                rows.append(
                    {
                        "date": d,
                        "asset_id": f"A{j:03d}",
                        "factor": float(factor[j]),
                        "forward_return": float(fwd[j]),
                    }
                )
        panel = pl.DataFrame(rows)
        cfg = AnalysisConfig.individual_continuous(metric=Metric.IC, forward_periods=2)
        profile = evaluate(panel, cfg)
        assert WarningCode.RECT_KERNEL_NEGATIVE_VARIANCE in profile.warnings
        assert profile.metadata[StatCode.P_HH]["variance_clamped"] is True
        assert profile.stats[StatCode.P_HH] == 1.0
        assert profile.stats[StatCode.T_HH] == 0.0
