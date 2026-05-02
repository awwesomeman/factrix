"""v0.5 ``_CommonContPanelProcedure`` + ``_CommonSparsePanelProcedure``.

Both PANEL cells follow the same shape: per-asset OLS β on a
broadcast regressor (continuous factor or sparse dummy), aggregated
into a cross-asset t-test on ``E[β]``. Synthetic panels build a
broadcast time series shared by every asset, then draw heterogeneous
true betas per asset to exercise the cross-asset variance.
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
from factrix._procedures import (
    InputSchema,
    _CommonContPanelProcedure,
    _CommonSparsePanelProcedure,
)
from factrix._profile import FactorProfile
from factrix._registry import _DISPATCH_REGISTRY, _DispatchKey


def _make_common_panel(
    *,
    n_dates: int,
    n_assets: int,
    seed: int,
    true_beta: float,
    factor_kind: str = "iid",
    sparse_event_density: float = 0.05,
    beta_dispersion: float = 0.1,
) -> pl.DataFrame:
    """Build a broadcast-factor panel with heterogeneous per-asset β.

    ``factor_kind="iid"``: factor ~ N(0, 1)
    ``factor_kind="rw"``: factor = cumsum(N(0, 1))
    ``factor_kind="sparse"``: factor ∈ {-1, 0, +1} with given density
    """
    rng = np.random.default_rng(seed)
    if factor_kind == "iid":
        factor = rng.standard_normal(n_dates)
    elif factor_kind == "rw":
        factor = np.cumsum(rng.standard_normal(n_dates))
    elif factor_kind == "sparse":
        factor = np.zeros(n_dates)
        n_events = max(2, int(n_dates * sparse_event_density))
        positions = rng.choice(n_dates, size=n_events, replace=False)
        factor[positions] = rng.choice([-1.0, 1.0], size=n_events)
    else:  # pragma: no cover
        raise ValueError(f"unknown factor_kind: {factor_kind}")

    asset_betas = rng.normal(loc=true_beta, scale=beta_dispersion, size=n_assets)
    start = dt.date(2024, 1, 1)
    rows: list[dict[str, object]] = []
    for t in range(n_dates):
        d = start + dt.timedelta(days=t)
        for j in range(n_assets):
            noise = float(rng.standard_normal())
            rows.append({
                "date": d,
                "asset_id": f"A{j:03d}",
                "factor": float(factor[t]),
                "forward_return": float(asset_betas[j] * factor[t] + noise),
            })
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def cfg_continuous() -> AnalysisConfig:
    return AnalysisConfig.common_continuous()


@pytest.fixture(scope="module")
def cfg_sparse() -> AnalysisConfig:
    return AnalysisConfig.common_sparse()


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


class TestRegistryWiring:
    def test_continuous_panel_registered(self) -> None:
        key = _DispatchKey(
            FactorScope.COMMON, Signal.CONTINUOUS, None, Mode.PANEL,
        )
        assert isinstance(
            _DISPATCH_REGISTRY[key].procedure, _CommonContPanelProcedure,
        )

    def test_sparse_panel_registered(self) -> None:
        key = _DispatchKey(
            FactorScope.COMMON, Signal.SPARSE, None, Mode.PANEL,
        )
        assert isinstance(
            _DISPATCH_REGISTRY[key].procedure, _CommonSparsePanelProcedure,
        )

    def test_continuous_input_schema(self) -> None:
        assert isinstance(
            _CommonContPanelProcedure.INPUT_SCHEMA, InputSchema,
        )
        assert "asset_id" in _CommonContPanelProcedure.INPUT_SCHEMA.required_columns


# ---------------------------------------------------------------------------
# COMMON × CONTINUOUS PANEL
# ---------------------------------------------------------------------------


class TestContinuousStrong:
    @pytest.fixture(scope="class")
    def profile(self, cfg_continuous: AnalysisConfig) -> FactorProfile:
        panel = _make_common_panel(
            n_dates=80, n_assets=20, seed=42,
            true_beta=0.6, factor_kind="iid",
        )
        return _CommonContPanelProcedure().compute(panel, cfg_continuous)

    def test_mode_panel(self, profile: FactorProfile) -> None:
        assert profile.mode is Mode.PANEL

    def test_n_obs_equals_n_assets(self, profile: FactorProfile) -> None:
        assert profile.n_obs == 20

    def test_passes_verdict(self, profile: FactorProfile) -> None:
        assert profile.verdict() is Verdict.PASS

    def test_low_primary_p(self, profile: FactorProfile) -> None:
        assert profile.primary_p < 0.001

    def test_beta_mean_close_to_truth(self, profile: FactorProfile) -> None:
        # true β=0.6, dispersion 0.1, N=20 — sample mean SE ≈ 0.022.
        assert 0.45 < profile.stats[StatCode.TS_BETA] < 0.75

    def test_required_stats(self, profile: FactorProfile) -> None:
        for key in (
            StatCode.TS_BETA,
            StatCode.TS_BETA_T_NW,
            StatCode.TS_BETA_P,
            StatCode.FACTOR_ADF_P,
        ):
            assert key in profile.stats


class TestContinuousRandom:
    def test_zero_true_beta_fails(
        self, cfg_continuous: AnalysisConfig,
    ) -> None:
        panel = _make_common_panel(
            n_dates=80, n_assets=20, seed=10,
            true_beta=0.0, beta_dispersion=0.05, factor_kind="iid",
        )
        profile = _CommonContPanelProcedure().compute(panel, cfg_continuous)
        assert profile.verdict() is Verdict.FAIL


class TestContinuousPersistentFactor:
    def test_random_walk_factor_warns(
        self, cfg_continuous: AnalysisConfig,
    ) -> None:
        panel = _make_common_panel(
            n_dates=80, n_assets=15, seed=7,
            true_beta=0.0, factor_kind="rw",
        )
        profile = _CommonContPanelProcedure().compute(panel, cfg_continuous)
        assert WarningCode.PERSISTENT_REGRESSOR in profile.warnings
        assert profile.stats[StatCode.FACTOR_ADF_P] > 0.10


# ---------------------------------------------------------------------------
# COMMON × SPARSE PANEL
# ---------------------------------------------------------------------------


class TestSparseStrong:
    @pytest.fixture(scope="class")
    def profile(self, cfg_sparse: AnalysisConfig) -> FactorProfile:
        panel = _make_common_panel(
            n_dates=80, n_assets=20, seed=43,
            true_beta=1.5, factor_kind="sparse",
            sparse_event_density=0.10,
        )
        return _CommonSparsePanelProcedure().compute(panel, cfg_sparse)

    def test_mode_panel(self, profile: FactorProfile) -> None:
        assert profile.mode is Mode.PANEL

    def test_passes_verdict(self, profile: FactorProfile) -> None:
        assert profile.verdict() is Verdict.PASS

    def test_low_primary_p(self, profile: FactorProfile) -> None:
        assert profile.primary_p < 0.001

    def test_beta_mean_close_to_truth(self, profile: FactorProfile) -> None:
        assert 1.0 < profile.stats[StatCode.TS_BETA] < 2.0

    def test_no_adf_stat_for_sparse(self, profile: FactorProfile) -> None:
        # I6: ADF persistence diagnostic is CONTINUOUS-only.
        assert StatCode.FACTOR_ADF_P not in profile.stats


class TestSparseRandom:
    def test_zero_true_beta_fails(
        self, cfg_sparse: AnalysisConfig,
    ) -> None:
        panel = _make_common_panel(
            n_dates=80, n_assets=20, seed=11,
            true_beta=0.0, beta_dispersion=0.05, factor_kind="sparse",
            sparse_event_density=0.08,
        )
        profile = _CommonSparsePanelProcedure().compute(panel, cfg_sparse)
        assert profile.verdict() is Verdict.FAIL


# ---------------------------------------------------------------------------
# End-to-end via _evaluate (panel sparse must NOT collapse — §5.4.1
# collapse only fires at N=1)
# ---------------------------------------------------------------------------


class TestEndToEndViaEvaluate:
    def test_continuous_e2e(self, cfg_continuous: AnalysisConfig) -> None:
        panel = _make_common_panel(
            n_dates=60, n_assets=15, seed=99,
            true_beta=0.5, factor_kind="iid",
        )
        profile = _evaluate(panel, cfg_continuous)
        assert profile.mode is Mode.PANEL
        assert profile.verdict() is Verdict.PASS

    def test_sparse_panel_does_not_collapse(
        self, cfg_sparse: AnalysisConfig,
    ) -> None:
        from factrix._codes import InfoCode

        panel = _make_common_panel(
            n_dates=60, n_assets=15, seed=88,
            true_beta=1.2, factor_kind="sparse",
            sparse_event_density=0.10,
        )
        profile = _evaluate(panel, cfg_sparse)
        assert profile.mode is Mode.PANEL
        # SCOPE_AXIS_COLLAPSED only fires at N=1; PANEL routing keeps
        # COMMON scope intact.
        assert InfoCode.SCOPE_AXIS_COLLAPSED not in profile.info_notes


# ---------------------------------------------------------------------------
# Review fix TC-1: empty-panel fallback (N == 0)
# ---------------------------------------------------------------------------


class TestEmptyPanelFallback:
    """When ``compute_ts_betas`` yields zero betas (empty panel after
    drop_nulls), the procedure must return a finite profile with
    primary_p == 1.0 rather than raising or producing NaN. Pins the
    N == 0 branch in ``_compute_common_panel``."""

    def test_continuous_empty_returns_p_one(
        self, cfg_continuous: AnalysisConfig,
    ) -> None:
        empty = pl.DataFrame(
            schema={
                "date": pl.Date,
                "asset_id": pl.Utf8,
                "factor": pl.Float64,
                "forward_return": pl.Float64,
            },
        )
        profile = _CommonContPanelProcedure().compute(empty, cfg_continuous)
        assert profile.primary_p == 1.0
        assert profile.n_obs == 0
        assert profile.stats[StatCode.TS_BETA] == 0.0

    def test_sparse_empty_returns_p_one(
        self, cfg_sparse: AnalysisConfig,
    ) -> None:
        empty = pl.DataFrame(
            schema={
                "date": pl.Date,
                "asset_id": pl.Utf8,
                "factor": pl.Float64,
                "forward_return": pl.Float64,
            },
        )
        profile = _CommonSparsePanelProcedure().compute(empty, cfg_sparse)
        assert profile.primary_p == 1.0
        assert profile.n_obs == 0
