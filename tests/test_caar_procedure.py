"""v0.5 ``_CAARSparsePanelProcedure.compute`` end-to-end tests.

Plan §4.3: per-event signed AR aggregated to a CAAR series across
event dates, then NW HAC t-test (with the same HH overlap floor as
IC / FM PANEL — h-period forward returns induce identical MA(h-1)).
Synthetic panels carry sparse ``{-1, 0, +1}`` triggers per (date,
asset); ``forward_return = β · factor + noise`` produces a CAAR
series with mean ≈ β under the strong scenario.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from factrix._analysis_config import AnalysisConfig
from factrix._axis import FactorScope, Mode, Signal
from factrix._codes import StatCode, Verdict
from factrix._evaluate import _evaluate
from factrix._procedures import InputSchema, _CAARSparsePanelProcedure
from factrix._profile import FactorProfile
from factrix._registry import _DISPATCH_REGISTRY, _DispatchKey
from factrix._stats.constants import auto_bartlett


def _make_event_panel(
    *,
    n_dates: int,
    n_assets: int,
    seed: int,
    beta: float,
    event_prob: float = 0.05,
) -> pl.DataFrame:
    """Build a (date, asset) panel with sparse {-1, 0, +1} triggers.

    Each cell is independently drawn: ``P(factor=±1) = event_prob``,
    ``P(factor=0) = 1 - 2*event_prob``. Forward returns follow
    ``β·factor + N(0,1)`` so per-event signed AR has mean ≈ β.
    """
    rng = np.random.default_rng(seed)
    factor = rng.choice(
        [-1, 0, 1],
        size=(n_dates, n_assets),
        p=[event_prob, 1.0 - 2 * event_prob, event_prob],
    ).astype(np.float64)
    fwd = beta * factor + rng.standard_normal((n_dates, n_assets))

    start = dt.date(2024, 1, 1)
    rows: list[dict[str, object]] = []
    for t in range(n_dates):
        d = start + dt.timedelta(days=t)
        for j in range(n_assets):
            rows.append({
                "date": d,
                "asset_id": f"A{j:03d}",
                "factor": float(factor[t, j]),
                "forward_return": float(fwd[t, j]),
            })
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def cfg() -> AnalysisConfig:
    return AnalysisConfig.individual_sparse()


class TestRegistryWiring:
    def test_registered_procedure_is_real(self) -> None:
        key = _DispatchKey(
            FactorScope.INDIVIDUAL, Signal.SPARSE, None, Mode.PANEL,
        )
        assert isinstance(
            _DISPATCH_REGISTRY[key].procedure,
            _CAARSparsePanelProcedure,
        )

    def test_input_schema_columns(self) -> None:
        schema = _CAARSparsePanelProcedure.INPUT_SCHEMA
        assert isinstance(schema, InputSchema)
        assert set(schema.required_columns) == {
            "date", "asset_id", "factor", "forward_return",
        }


class TestStrongCaar:
    @pytest.fixture(scope="class")
    def profile(self, cfg: AnalysisConfig) -> FactorProfile:
        panel = _make_event_panel(
            n_dates=80, n_assets=20, seed=42, beta=1.0,
        )
        return _CAARSparsePanelProcedure().compute(panel, cfg)

    def test_mode_panel(self, profile: FactorProfile) -> None:
        assert profile.mode is Mode.PANEL

    def test_passes_verdict(self, profile: FactorProfile) -> None:
        assert profile.verdict() is Verdict.PASS

    def test_low_primary_p(self, profile: FactorProfile) -> None:
        assert profile.primary_p < 0.001

    def test_caar_mean_close_to_beta(self, profile: FactorProfile) -> None:
        # signed_car ≈ β + noise*sign(factor); cross-event mean ≈ β=1.0.
        assert 0.7 < profile.stats[StatCode.CAAR_MEAN] < 1.3

    def test_required_stats(self, profile: FactorProfile) -> None:
        for key in (
            StatCode.CAAR_MEAN,
            StatCode.CAAR_T_NW,
            StatCode.CAAR_P,
            StatCode.NW_LAGS_USED,
        ):
            assert key in profile.stats

    def test_primary_p_matches_caar_p_stat(
        self, profile: FactorProfile,
    ) -> None:
        assert profile.primary_p == profile.stats[StatCode.CAAR_P]

    def test_n_obs_equals_event_dates(self, profile: FactorProfile) -> None:
        # n_obs = number of distinct event-dates feeding the NW test;
        # should be > 0 and bounded above by n_dates=80.
        assert 0 < profile.n_obs <= 80


class TestRandomCaar:
    def test_zero_beta_fails_at_default_threshold(
        self, cfg: AnalysisConfig,
    ) -> None:
        panel = _make_event_panel(
            n_dates=80, n_assets=20, seed=10, beta=0.0,
        )
        profile = _CAARSparsePanelProcedure().compute(panel, cfg)
        assert profile.verdict() is Verdict.FAIL


class TestNwLagFloor:
    def test_nw_lags_floor_at_forward_periods_minus_one(
        self, cfg: AnalysisConfig,
    ) -> None:
        panel = _make_event_panel(
            n_dates=80, n_assets=20, seed=4, beta=0.5, event_prob=0.10,
        )
        profile = _CAARSparsePanelProcedure().compute(panel, cfg)
        T = profile.n_obs
        expected = float(max(auto_bartlett(T), cfg.forward_periods - 1))
        assert profile.stats[StatCode.NW_LAGS_USED] == expected


class TestEndToEndViaEvaluate:
    def test_evaluate_dispatches_to_caar(self, cfg: AnalysisConfig) -> None:
        panel = _make_event_panel(
            n_dates=80, n_assets=20, seed=99, beta=0.8,
        )
        profile = _evaluate(panel, cfg)
        assert profile.mode is Mode.PANEL
        assert StatCode.CAAR_P in profile.stats
        assert profile.verdict() is Verdict.PASS
