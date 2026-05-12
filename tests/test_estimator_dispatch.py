"""Cell-procedure dispatch through ``cfg.estimator.compute`` (#163 batch 3).

Verifies the BREAKING behavior changes:
- D9: default cfg no longer emits ``P_HH`` / ``T_HH`` as a side-stat
- D10: ``primary_p`` / ``primary_stat_name`` reflect the chosen estimator
- ``profile.context["estimator"]`` records provenance on every cell

Bit-equal check: default ``NeweyWest()`` cfg produces the same
``primary_p`` as the v0.11 hardcoded NW path (via ``compute`` delegating
to the same primitive); explicit ``HansenHodrick()`` produces the HH
result that the v0.11 side-emit used to expose.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest
from factrix import AnalysisConfig, Metric, evaluate
from factrix._codes import StatCode
from factrix._stats import _hansen_hodrick_t_test, _newey_west_t_test, _resolve_nw_lags
from factrix._stats.constants import auto_bartlett
from factrix.metrics.ic import compute_ic
from factrix.stats import HansenHodrick, NeweyWest


def _panel(*, n_dates: int = 80, n_assets: int = 15, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = dt.date(2024, 1, 1)
    rows: list[dict[str, object]] = []
    for i in range(n_dates):
        d = start + dt.timedelta(days=i)
        fwd = rng.standard_normal(n_assets)
        noise = rng.standard_normal(n_assets)
        for j in range(n_assets):
            rows.append(
                {
                    "date": d,
                    "asset_id": f"A{j:03d}",
                    "factor": float(noise[j]),
                    "forward_return": float(fwd[j]),
                }
            )
    return pl.DataFrame(rows)


class TestDefaultNWBitEqual:
    """Default cfg (``NeweyWest()``) is bit-equal to the v0.11 hardcoded path."""

    def test_ic_panel_primary_p_matches_primitive(self) -> None:
        panel = _panel(seed=1)
        cfg = AnalysisConfig.individual_continuous(metric=Metric.IC, forward_periods=5)
        profile = evaluate(panel, cfg)

        ic_values = compute_ic(panel)["ic"].drop_nulls().to_numpy()
        n = len(ic_values)
        nw_lags = _resolve_nw_lags(n, auto_bartlett(n), 5)
        t_expected, p_expected, _ = _newey_west_t_test(ic_values, lags=nw_lags)

        assert profile.primary_p == p_expected
        assert profile.primary_stat == t_expected
        assert profile.primary_stat_name is StatCode.T_NW


class TestDefaultDoesNotEmitHH:
    """D9: default ``NeweyWest()`` cfg no longer side-emits ``P_HH`` / ``T_HH``."""

    @pytest.mark.parametrize("metric", [Metric.IC, Metric.FM])
    def test_p_hh_absent(self, metric: Metric) -> None:
        panel = _panel(seed=2)
        cfg = AnalysisConfig.individual_continuous(metric=metric, forward_periods=5)
        profile = evaluate(panel, cfg)
        assert StatCode.P_HH not in profile.stats
        assert StatCode.T_HH not in profile.stats


class TestHHDispatch:
    """Explicit ``HansenHodrick()`` cfg routes the HH path through dispatch."""

    def test_primary_p_matches_hh_primitive(self) -> None:
        panel = _panel(seed=3)
        cfg = AnalysisConfig.individual_continuous(
            metric=Metric.IC, forward_periods=5, estimator=HansenHodrick()
        )
        profile = evaluate(panel, cfg)

        ic_values = compute_ic(panel)["ic"].drop_nulls().to_numpy()
        t_expected, p_expected, _, _ = _hansen_hodrick_t_test(
            ic_values, forward_periods=5
        )

        assert profile.primary_p == p_expected
        assert profile.primary_stat == t_expected
        assert profile.primary_stat_name is StatCode.T_HH


class TestContextProvenance:
    """``profile.context["estimator"]`` records the cfg-driven estimator."""

    @pytest.mark.parametrize(
        ("estimator", "expected_name"),
        [(NeweyWest(), "NeweyWest"), (HansenHodrick(), "HansenHodrick")],
    )
    def test_ic_panel_context(self, estimator, expected_name: str) -> None:
        panel = _panel(seed=4)
        cfg = AnalysisConfig.individual_continuous(
            metric=Metric.IC, forward_periods=5, estimator=estimator
        )
        profile = evaluate(panel, cfg)
        assert profile.context["estimator"] == expected_name

    def test_common_continuous_default_records_nw(self) -> None:
        # COMMON cells don't dispatch through estimator (no HAC-on-mean),
        # but provenance is still written so all profiles carry the key.
        panel = _panel(seed=5)
        cfg = AnalysisConfig.common_continuous(forward_periods=5)
        profile = evaluate(panel, cfg)
        assert profile.context["estimator"] == "NeweyWest"
