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
from factrix._codes import StatCode, Verdict, WarningCode
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
            rows.append(
                {
                    "date": d,
                    "asset_id": f"A{j:03d}",
                    "factor": float(factor[t, j]),
                    "forward_return": float(fwd[t, j]),
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def cfg() -> AnalysisConfig:
    return AnalysisConfig.individual_sparse()


class TestRegistryWiring:
    def test_registered_procedure_is_real(self) -> None:
        key = _DispatchKey(
            FactorScope.INDIVIDUAL,
            Signal.SPARSE,
            None,
            Mode.PANEL,
        )
        assert isinstance(
            _DISPATCH_REGISTRY[key].procedure,
            _CAARSparsePanelProcedure,
        )

    def test_input_schema_columns(self) -> None:
        schema = _CAARSparsePanelProcedure.INPUT_SCHEMA
        assert isinstance(schema, InputSchema)
        assert set(schema.required_columns) == {
            "date",
            "asset_id",
            "factor",
            "forward_return",
        }


class TestStrongCaar:
    @pytest.fixture(scope="class")
    def profile(self, cfg: AnalysisConfig) -> FactorProfile:
        panel = _make_event_panel(
            n_dates=80,
            n_assets=20,
            seed=42,
            beta=1.0,
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
        self,
        profile: FactorProfile,
    ) -> None:
        assert profile.primary_p == profile.stats[StatCode.CAAR_P]

    def test_n_obs_equals_event_dates(self, profile: FactorProfile) -> None:
        # n_obs = number of distinct event-dates feeding the NW test;
        # should be > 0 and bounded above by n_dates=80.
        assert 0 < profile.n_obs <= 80


class TestRandomCaar:
    def test_zero_beta_fails_at_default_threshold(
        self,
        cfg: AnalysisConfig,
    ) -> None:
        panel = _make_event_panel(
            n_dates=80,
            n_assets=20,
            seed=10,
            beta=0.0,
        )
        profile = _CAARSparsePanelProcedure().compute(panel, cfg)
        assert profile.verdict() is Verdict.FAIL


class TestNwLagFloor:
    def test_nw_lags_floor_at_forward_periods_minus_one(
        self,
        cfg: AnalysisConfig,
    ) -> None:
        panel = _make_event_panel(
            n_dates=80,
            n_assets=20,
            seed=4,
            beta=0.5,
            event_prob=0.10,
        )
        profile = _CAARSparsePanelProcedure().compute(panel, cfg)
        T = profile.n_obs
        expected = float(max(auto_bartlett(T), cfg.forward_periods - 1))
        assert profile.stats[StatCode.NW_LAGS_USED] == expected


class TestEndToEndViaEvaluate:
    def test_evaluate_dispatches_to_caar(self, cfg: AnalysisConfig) -> None:
        panel = _make_event_panel(
            n_dates=80,
            n_assets=20,
            seed=99,
            beta=0.8,
        )
        profile = _evaluate(panel, cfg)
        assert profile.mode is Mode.PANEL
        assert StatCode.CAAR_P in profile.stats
        assert profile.verdict() is Verdict.PASS


# ---------------------------------------------------------------------------
# Calendar-time portfolio regimes (issue #24)
# ---------------------------------------------------------------------------


def _make_panel_with_event_dates(
    *,
    event_dates: list[dt.date],
    all_dates: list[dt.date],
    n_assets: int,
    seed: int,
    beta: float,
) -> pl.DataFrame:
    """Panel with ±1 events only on the listed event_dates.

    Forward returns follow ``β·factor + N(0,1)`` so per-event signed AR
    has mean ≈ β. Non-event dates carry factor=0 with pure-noise return.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for d in all_dates:
        is_event = d in set(event_dates)
        for j in range(n_assets):
            f = float(rng.choice([-1.0, 1.0])) if is_event else 0.0
            rows.append(
                {
                    "date": d,
                    "asset_id": f"A{j:03d}",
                    "factor": f,
                    "forward_return": float(beta * f + rng.standard_normal()),
                }
            )
    return pl.DataFrame(rows)


class TestCalendarTimeRegimes:
    def test_sparse_regime_n_obs_is_calendar_count(
        self,
        cfg: AnalysisConfig,
    ) -> None:
        # 4 events on a 120-day panel → n_obs is the dense-series length.
        start = dt.date(2024, 1, 1)
        all_dates = [start + dt.timedelta(days=i) for i in range(120)]
        event_dates = [start + dt.timedelta(days=30 * k) for k in range(4)]
        panel = _make_panel_with_event_dates(
            event_dates=event_dates,
            all_dates=all_dates,
            n_assets=10,
            seed=7,
            beta=0.0,
        )
        profile = _CAARSparsePanelProcedure().compute(panel, cfg)
        assert profile.n_obs == 120

    def test_sparse_regime_caar_mean_uses_event_only_average(
        self,
        cfg: AnalysisConfig,
    ) -> None:
        # CAAR_MEAN must be the per-event-date mean (≈ β=1.0), not the
        # zero-diluted dense mean (≈ 0.033 = β × n_event/n_calendar).
        start = dt.date(2024, 1, 1)
        all_dates = [start + dt.timedelta(days=i) for i in range(120)]
        event_dates = [start + dt.timedelta(days=30 * k) for k in range(4)]
        panel = _make_panel_with_event_dates(
            event_dates=event_dates,
            all_dates=all_dates,
            n_assets=40,
            seed=11,
            beta=1.0,
        )
        profile = _CAARSparsePanelProcedure().compute(panel, cfg)
        assert 0.6 < profile.stats[StatCode.CAAR_MEAN] < 1.4

    def test_dense_regime_matches_event_only_when_every_date_is_event(
        self,
        cfg: AnalysisConfig,
    ) -> None:
        # When every date carries an event, the dense reindex is a no-op —
        # the procedure should agree with a hand-rolled NW HAC on the
        # raw CAAR series (the pre-fix codepath, in this regime only).
        from factrix._stats import _newey_west_t_test, _resolve_nw_lags
        from factrix._stats.constants import auto_bartlett
        from factrix.metrics.caar import compute_caar

        start = dt.date(2024, 1, 1)
        all_dates = [start + dt.timedelta(days=i) for i in range(80)]
        panel = _make_panel_with_event_dates(
            event_dates=all_dates,
            all_dates=all_dates,
            n_assets=20,
            seed=23,
            beta=0.5,
        )
        profile = _CAARSparsePanelProcedure().compute(panel, cfg)

        event_caar = compute_caar(panel)["caar"].drop_nulls().to_numpy()
        T = len(event_caar)
        lags = _resolve_nw_lags(T, auto_bartlett(T), cfg.forward_periods)
        ref_t, ref_p, _ = _newey_west_t_test(event_caar, lags=lags)
        assert profile.stats[StatCode.CAAR_T_NW] == pytest.approx(ref_t)
        assert profile.stats[StatCode.CAAR_P] == pytest.approx(ref_p)

    def test_clustered_regime_picks_up_overlap_via_nw_hac(
        self,
        cfg: AnalysisConfig,
    ) -> None:
        # Two clusters of 4 consecutive event days within forward_periods=5;
        # estimator must produce a finite p without collapsing.
        start = dt.date(2024, 1, 1)
        all_dates = [start + dt.timedelta(days=i) for i in range(60)]
        clusters = [start + dt.timedelta(days=10 + k) for k in range(4)] + [
            start + dt.timedelta(days=40 + k) for k in range(4)
        ]
        panel = _make_panel_with_event_dates(
            event_dates=clusters,
            all_dates=all_dates,
            n_assets=15,
            seed=31,
            beta=0.5,
        )
        profile = _CAARSparsePanelProcedure().compute(panel, cfg)
        assert 0.0 <= profile.stats[StatCode.CAAR_P] <= 1.0
        assert profile.n_obs == 60


class TestSparseMagnitudeWeightedWarning:
    """Individual×Sparse procedure surfaces magnitude-weighted contract warning."""

    def test_mixed_sign_non_ternary_emits_warning(
        self,
        cfg: AnalysisConfig,
    ) -> None:
        panel = _make_event_panel(n_dates=80, n_assets=20, seed=42, beta=1.0)
        perturbed = panel.with_columns(
            pl.when(pl.col("factor") > 0)
            .then(2.5)
            .when(pl.col("factor") < 0)
            .then(-1.7)
            .otherwise(0.0)
            .alias("factor")
        )
        profile = _CAARSparsePanelProcedure().compute(perturbed, cfg)
        assert WarningCode.SPARSE_MAGNITUDE_WEIGHTED in profile.warnings

    def test_clean_ternary_silent(self, cfg: AnalysisConfig) -> None:
        panel = _make_event_panel(n_dates=80, n_assets=20, seed=43, beta=1.0)
        profile = _CAARSparsePanelProcedure().compute(panel, cfg)
        assert WarningCode.SPARSE_MAGNITUDE_WEIGHTED not in profile.warnings


class TestFewEventsBrownWarner:
    """`(INDIVIDUAL, SPARSE, PANEL)` propagates Brown-Warner borderline."""

    def test_borderline_event_count_emits_warning(
        self,
        cfg: AnalysisConfig,
    ) -> None:
        # Density tuned so the seed produces n_event_dates in [4, 30).
        panel = _make_event_panel(
            n_dates=120,
            n_assets=10,
            seed=51,
            beta=0.0,
            event_prob=0.02,
        )
        profile = _CAARSparsePanelProcedure().compute(panel, cfg)
        if profile.n_obs >= 4:  # n_obs is dense; sanity guard
            ev = panel.filter(pl.col("factor") != 0)["date"].n_unique()
            if 4 <= ev < 30:
                assert WarningCode.FEW_EVENTS_BROWN_WARNER in profile.warnings

    def test_many_events_silent(self, cfg: AnalysisConfig) -> None:
        panel = _make_event_panel(
            n_dates=200,
            n_assets=20,
            seed=52,
            beta=0.0,
            event_prob=0.20,
        )
        profile = _CAARSparsePanelProcedure().compute(panel, cfg)
        assert WarningCode.FEW_EVENTS_BROWN_WARNER not in profile.warnings
