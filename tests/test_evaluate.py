"""v0.5 ``_evaluate`` — mode derivation + registry dispatch + collapse."""

from __future__ import annotations

import datetime as dt

import factrix as fx
import numpy as np
import polars as pl
import pytest
from factrix._analysis_config import AnalysisConfig
from factrix._axis import FactorScope, Metric, Mode, Signal
from factrix._codes import InfoCode, StatCode
from factrix._errors import ModeAxisError, UserInputError
from factrix._evaluate import _derive_mode, _evaluate
from factrix._metric_index import spec_by_name
from factrix._profile import FactorProfile
from factrix._results import EvaluationResult


def _build_panel(
    *,
    n_dates: int,
    n_assets: int,
    seed: int,
    factor_strength: float = 0.0,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = dt.date(2024, 1, 1)
    dates = [start + dt.timedelta(days=i) for i in range(n_dates)]
    rows: list[dict[str, object]] = []
    for d in dates:
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


def _build_timeseries(*, n_dates: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = dt.date(2024, 1, 1)
    dates = [start + dt.timedelta(days=i) for i in range(n_dates)]
    rows = [
        {
            "date": d,
            "asset_id": "SPY",
            "factor": float(rng.standard_normal()),
            "forward_return": float(rng.standard_normal()),
        }
        for d in dates
    ]
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Mode derivation
# ---------------------------------------------------------------------------


class TestDeriveMode:
    def test_panel_when_multi_asset(self) -> None:
        panel = _build_panel(n_dates=10, n_assets=15, seed=1)
        assert _derive_mode(panel) is Mode.PANEL

    def test_timeseries_when_single_asset(self) -> None:
        ts = _build_timeseries(n_dates=30, seed=1)
        assert _derive_mode(ts) is Mode.TIMESERIES

    def test_timeseries_when_zero_rows(self) -> None:
        empty = pl.DataFrame(
            schema={
                "date": pl.Date,
                "asset_id": pl.Utf8,
                "factor": pl.Float64,
                "forward_return": pl.Float64,
            }
        )
        assert _derive_mode(empty) is Mode.TIMESERIES


# ---------------------------------------------------------------------------
# Happy path — IC PANEL end-to-end
# ---------------------------------------------------------------------------


class TestIcPanelEndToEnd:
    @pytest.fixture(scope="class")
    def profile(self) -> FactorProfile:
        panel = _build_panel(
            n_dates=60,
            n_assets=30,
            seed=42,
            factor_strength=0.95,
        )
        cfg = AnalysisConfig.individual_continuous(metric=Metric.IC)
        return _evaluate(panel, cfg)["factor"]

    def test_returns_factor_profile(self, profile: FactorProfile) -> None:
        assert isinstance(profile, FactorProfile)

    def test_mode_is_panel(self, profile: FactorProfile) -> None:
        assert profile.mode is Mode.PANEL

    def test_low_primary_p_on_strong_factor(self, profile: FactorProfile) -> None:
        assert profile.primary_p < 0.05

    def test_no_collapse_info_note(self, profile: FactorProfile) -> None:
        assert InfoCode.SCOPE_AXIS_COLLAPSED not in profile.info_notes

    def test_ic_p_populated(self, profile: FactorProfile) -> None:
        assert StatCode.P_NW in profile.stats


# ---------------------------------------------------------------------------
# ModeAxisError on undefined cell (§5.5)
# ---------------------------------------------------------------------------


class TestModeAxisError:
    def test_individual_continuous_n1_raises(self) -> None:
        ts = _build_timeseries(n_dates=40, seed=2)
        cfg = AnalysisConfig.individual_continuous(metric=Metric.IC)
        with pytest.raises(ModeAxisError) as exc_info:
            _evaluate(ts, cfg)
        # Suggested fix points at the only legal CONTINUOUS path at N=1.
        assert exc_info.value.suggested_fix == AnalysisConfig.common_continuous()

    def test_individual_continuous_fm_n1_raises(self) -> None:
        ts = _build_timeseries(n_dates=40, seed=3)
        cfg = AnalysisConfig.individual_continuous(metric=Metric.FM)
        with pytest.raises(ModeAxisError):
            _evaluate(ts, cfg)


# ---------------------------------------------------------------------------
# Sparse N=1 collapse routing (§5.4.1)
# ---------------------------------------------------------------------------


class TestSparseCollapse:
    def test_individual_sparse_n1_routes_to_sentinel_entry(self) -> None:
        # Sentinel cell is now wired — verify the routing reaches a real
        # FactorProfile and the collapse InfoCode is attached.
        ts = _build_timeseries(n_dates=40, seed=4)
        profile = _evaluate(ts, AnalysisConfig.individual_sparse())["factor"]
        assert profile.mode is Mode.TIMESERIES
        assert InfoCode.SCOPE_AXIS_COLLAPSED in profile.info_notes

    def test_common_sparse_n1_routes_to_same_sentinel_entry(self) -> None:
        ts = _build_timeseries(n_dates=40, seed=5)
        profile = _evaluate(ts, AnalysisConfig.common_sparse())["factor"]
        assert profile.mode is Mode.TIMESERIES
        assert InfoCode.SCOPE_AXIS_COLLAPSED in profile.info_notes

    def test_panel_sparse_does_not_collapse(self) -> None:
        # PANEL routing keeps the user-facing scope intact; the CAAR
        # cell handles individual_sparse without going through the
        # _SCOPE_COLLAPSED sentinel (collapse only fires at N=1).
        panel = _build_panel(n_dates=30, n_assets=20, seed=6)
        sparse_factor = (np.arange(len(panel)) % 11 == 0).astype(np.float64)
        panel = panel.with_columns(pl.Series("factor", sparse_factor))
        profile = _evaluate(panel, AnalysisConfig.individual_sparse())["factor"]
        assert profile.mode is Mode.PANEL
        assert InfoCode.SCOPE_AXIS_COLLAPSED not in profile.info_notes


# ---------------------------------------------------------------------------
# Review fix TC-FALLBACK-None: monkey-patch the registry so a legal
# AnalysisConfig produces a registry miss with no _FALLBACK_MAP entry,
# verifying _evaluate's `suggested_fix=None` branch (the suffix="" path
# in the ModeAxisError construction) is exercised.
# ---------------------------------------------------------------------------


class TestFallbackNoneBranch:
    def test_unregistered_cell_with_no_fallback_raises_clean_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from factrix._registry import _DISPATCH_REGISTRY, _DispatchKey

        # Pop the live (COMMON, CONTINUOUS, PANEL) entry so a real
        # call with that config produces a registry miss. _FALLBACK_MAP
        # has no entry for (COMMON, CONTINUOUS, PANEL) — N≥2 paths are
        # always supposed to be wired — so the lookup yields None and
        # ModeAxisError is raised with suggested_fix=None.
        key = _DispatchKey(
            FactorScope.COMMON,
            Signal.CONTINUOUS,
            None,
            Mode.PANEL,
        )
        original = _DISPATCH_REGISTRY.pop(key)
        try:
            panel = _build_panel(n_dates=30, n_assets=20, seed=11)
            with pytest.raises(ModeAxisError) as exc:
                _evaluate(
                    panel,
                    AnalysisConfig.common_continuous(),
                )
            assert exc.value.suggested_fix is None
            assert "Suggested fix:" not in str(exc.value)
        finally:
            _DISPATCH_REGISTRY[key] = original


# ---------------------------------------------------------------------------
# n_assets exposure: every procedure populates the cross-section width
# ---------------------------------------------------------------------------


class TestNAssetsExposure:
    """Every cell should carry the panel's cross-section width on the
    profile so users can disambiguate "small n_obs from short series"
    vs "small n_obs from thin cross-section". Per-procedure derivation
    keeps the value correct even for direct ``procedure.compute(...)``
    calls that bypass ``_evaluate``."""

    def test_panel_ic_carries_n_assets(self) -> None:
        panel = _build_panel(n_dates=60, n_assets=25, seed=7)
        profile = _evaluate(
            panel,
            AnalysisConfig.individual_continuous(metric=Metric.IC),
        )["factor"]
        assert profile.n_assets == 25
        assert profile.n_obs != profile.n_assets  # T vs N differ

    def test_timeseries_n_assets_is_one(self) -> None:
        panel = _build_panel(n_dates=60, n_assets=25, seed=8)
        single = panel.filter(pl.col("asset_id") == panel["asset_id"][0])
        profile = _evaluate(single, AnalysisConfig.common_continuous())["factor"]
        assert profile.mode is Mode.TIMESERIES
        assert profile.n_assets == 1

    def test_diagnose_includes_n_assets(self) -> None:
        panel = _build_panel(n_dates=60, n_assets=15, seed=9)
        profile = _evaluate(
            panel,
            AnalysisConfig.individual_continuous(metric=Metric.IC),
        )["factor"]
        d = profile.diagnose()
        assert d["n_assets"] == 15
        assert d["n_obs"] != d["n_assets"]


# ---------------------------------------------------------------------------
# factor_cols — replaces the old factor_col=str surface (#421)
# ---------------------------------------------------------------------------


class TestFactorColsOption:
    def test_default_factor_cols_canonical_panel(self) -> None:
        # Default factor_cols=("factor",) goes through the existing happy
        # path — sanity that the canonical-name single-factor case still
        # produces the expected dict keyed by "factor".
        panel = _build_panel(n_dates=60, n_assets=15, seed=10)
        profiles = _evaluate(
            panel,
            AnalysisConfig.individual_continuous(metric=Metric.IC),
        )
        assert set(profiles) == {"factor"}
        assert profiles["factor"].factor_id == "factor"
        assert StatCode.MEAN in profiles["factor"].stats

    def test_named_factor_col_returned_under_its_name(self) -> None:
        # factor_cols=["alpha"] on a renamed panel returns a dict keyed
        # by "alpha"; the profile's factor_id is also stamped to match.
        panel = _build_panel(n_dates=60, n_assets=15, seed=11)
        renamed = panel.rename({"factor": "alpha"})
        profiles = _evaluate(
            renamed,
            AnalysisConfig.individual_continuous(metric=Metric.IC),
            factor_cols=["alpha"],
        )
        assert set(profiles) == {"alpha"}
        assert profiles["alpha"].factor_id == "alpha"
        assert StatCode.MEAN in profiles["alpha"].stats

    def test_batch_two_factors_returns_dict_keyed_by_name(self) -> None:
        # The headline #421 capability: pass a list, get back a dict —
        # both keys present, factor_id matches, each profile independent.
        panel = _build_panel(n_dates=60, n_assets=15, seed=12)
        wide = panel.with_columns(
            pl.col("factor").alias("alpha"),
            (pl.col("factor") * -1.0).alias("beta"),
        ).drop("factor")
        profiles = _evaluate(
            wide,
            AnalysisConfig.individual_continuous(metric=Metric.IC),
            factor_cols=["alpha", "beta"],
        )
        assert list(profiles) == ["alpha", "beta"]
        assert profiles["alpha"].factor_id == "alpha"
        assert profiles["beta"].factor_id == "beta"

    def test_batch_single_factor_equals_named_single_call(self) -> None:
        # factor_cols=["alpha"] inside a batch must produce a profile
        # bit-equivalent to the single-factor named call — pins that the
        # per-factor projection does not leak sibling columns into the
        # procedure.
        panel = _build_panel(n_dates=60, n_assets=15, seed=13)
        wide = panel.with_columns(
            pl.col("factor").alias("alpha"),
            (pl.col("factor") * 0.5).alias("beta"),
        ).drop("factor")
        cfg = AnalysisConfig.individual_continuous(metric=Metric.IC)
        batched = _evaluate(wide, cfg, factor_cols=["alpha", "beta"])["alpha"]
        alone = _evaluate(wide.drop("beta"), cfg, factor_cols=["alpha"])["alpha"]
        assert batched.primary_p == alone.primary_p
        assert batched.stats[StatCode.MEAN] == alone.stats[StatCode.MEAN]

    def test_empty_factor_cols_raises_user_input_error(self) -> None:
        panel = _build_panel(n_dates=30, n_assets=10, seed=14)
        with pytest.raises(UserInputError, match="factor_cols"):
            _evaluate(
                panel,
                AnalysisConfig.individual_continuous(metric=Metric.IC),
                factor_cols=[],
            )

    def test_duplicate_factor_cols_raises_user_input_error(self) -> None:
        panel = _build_panel(n_dates=30, n_assets=10, seed=15)
        with pytest.raises(UserInputError, match="duplicates"):
            _evaluate(
                panel,
                AnalysisConfig.individual_continuous(metric=Metric.IC),
                factor_cols=["factor", "factor"],
            )

    def test_missing_factor_col_raises_user_input_error(self) -> None:
        panel = _build_panel(n_dates=30, n_assets=10, seed=16)
        with pytest.raises(UserInputError, match="alpha"):
            _evaluate(
                panel,
                AnalysisConfig.individual_continuous(metric=Metric.IC),
                factor_cols=["alpha"],
            )


class TestPublicEvaluateNewSignature:
    """``factrix.evaluate`` — DAG-executor-backed signature (#445)."""

    def test_single_factor_returns_one_element_list(self) -> None:
        panel = _build_panel(n_dates=80, n_assets=15, seed=21)
        ic = spec_by_name()["ic"]
        results = fx.evaluate(
            panel, metrics=[ic], factor_cols=["factor"], forward_periods=5
        )
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], EvaluationResult)
        assert results[0].factor == "factor"
        assert "ic" in results[0].metrics

    def test_multi_factor_preserves_order(self) -> None:
        panel = _build_panel(n_dates=80, n_assets=15, seed=22)
        panel = panel.with_columns(
            pl.col("factor").alias("alpha"),
            pl.col("factor").alias("beta"),
        )
        ic = spec_by_name()["ic"]
        results = fx.evaluate(
            panel,
            metrics=[ic],
            factor_cols=["beta", "alpha"],
            forward_periods=5,
        )
        assert [r.factor for r in results] == ["beta", "alpha"]

    def test_metrics_must_be_list(self) -> None:
        panel = _build_panel(n_dates=40, n_assets=10, seed=23)
        ic = spec_by_name()["ic"]
        with pytest.raises(UserInputError, match="list"):
            fx.evaluate(panel, metrics=ic, factor_cols=["factor"], forward_periods=5)  # type: ignore[arg-type]

    def test_metrics_elements_must_be_metric_spec(self) -> None:
        panel = _build_panel(n_dates=40, n_assets=10, seed=24)
        with pytest.raises(UserInputError, match="MetricSpec"):
            fx.evaluate(
                panel,
                metrics=["ic"],
                factor_cols=["factor"],
                forward_periods=5,  # type: ignore[list-item]
            )

    def test_metrics_empty_list_rejected(self) -> None:
        panel = _build_panel(n_dates=40, n_assets=10, seed=25)
        with pytest.raises(UserInputError, match="non-empty"):
            fx.evaluate(panel, metrics=[], factor_cols=["factor"], forward_periods=5)

    def test_factor_cols_single_str_rejected(self) -> None:
        panel = _build_panel(n_dates=40, n_assets=10, seed=26)
        ic = spec_by_name()["ic"]
        with pytest.raises(UserInputError, match="single str is rejected"):
            fx.evaluate(panel, metrics=[ic], factor_cols="factor", forward_periods=5)  # type: ignore[arg-type]

    def test_missing_metrics_kwarg_raises_typeerror(self) -> None:
        panel = _build_panel(n_dates=40, n_assets=10, seed=27)
        with pytest.raises(TypeError, match="metrics"):
            fx.evaluate(panel)  # type: ignore[call-arg]

    def test_panel_missing_forward_return_raises(self) -> None:
        panel = _build_panel(n_dates=40, n_assets=10, seed=28)
        panel_no_fwd = panel.drop("forward_return")
        ic = spec_by_name()["ic"]
        with pytest.raises(UserInputError, match="forward_return"):
            fx.evaluate(
                panel_no_fwd, metrics=[ic], factor_cols=["factor"], forward_periods=5
            )

    def test_panel_missing_baseline_column_raises(self) -> None:
        panel = _build_panel(n_dates=40, n_assets=10, seed=29)
        panel_no_asset = panel.drop("asset_id")
        ic = spec_by_name()["ic"]
        with pytest.raises(UserInputError, match="asset_id"):
            fx.evaluate(
                panel_no_asset,
                metrics=[ic],
                factor_cols=["factor"],
                forward_periods=5,
            )

    def test_forward_periods_missing_when_required_raises(self) -> None:
        panel = _build_panel(n_dates=40, n_assets=10, seed=30)
        ic = spec_by_name()["ic"]
        with pytest.raises(UserInputError, match="forward_periods"):
            fx.evaluate(panel, metrics=[ic], factor_cols=["factor"])
