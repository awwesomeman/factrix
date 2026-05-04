"""v0.5 introspection helpers — describe_analysis_modes + suggest_config."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from factrix._analysis_config import AnalysisConfig
from factrix._axis import FactorScope, Metric, Mode, Signal
from factrix._codes import WarningCode
from factrix._describe import (
    SuggestConfigResult,
    describe_analysis_modes,
    suggest_config,
)
from factrix._stats.constants import MIN_PERIODS_HARD, MIN_PERIODS_RELIABLE


# ---------------------------------------------------------------------------
# describe_analysis_modes
# ---------------------------------------------------------------------------


class TestDescribeAnalysisModes:
    def test_text_format_returns_string(self) -> None:
        out = describe_analysis_modes(format="text")
        assert isinstance(out, str)
        assert "Cell:" in out
        assert "PANEL" in out and "TIMESERIES" in out

    def test_json_format_returns_five_user_tuples(self) -> None:
        rows = describe_analysis_modes(format="json")
        assert isinstance(rows, list)
        assert len(rows) == 5  # plan §4.3 — exactly five legal user-facing tuples

    def test_json_row_shape(self) -> None:
        rows = describe_analysis_modes(format="json")
        row = rows[0]
        assert set(row.keys()) == {
            "scope", "signal", "metric",
            "panel", "timeseries",
        }

    def test_individual_continuous_ic_routing(self) -> None:
        rows = describe_analysis_modes(format="json")
        ic_row = next(
            r for r in rows
            if r["scope"] == "individual"
            and r["signal"] == "continuous"
            and r["metric"] == "ic"
        )
        # PANEL entry exists; TIMESERIES entry does NOT (raises ModeAxisError).
        assert ic_row["panel"] is not None
        assert isinstance(ic_row["timeseries"], str)
        assert "ModeAxisError" in ic_row["timeseries"]

    def test_common_continuous_has_both_modes(self) -> None:
        rows = describe_analysis_modes(format="json")
        cc_row = next(
            r for r in rows
            if r["scope"] == "common" and r["signal"] == "continuous"
        )
        assert cc_row["panel"] is not None
        assert isinstance(cc_row["timeseries"], dict)
        # No collapse on common × continuous.
        assert cc_row["timeseries"]["scope_collapsed"] is False

    def test_sparse_rows_flag_scope_collapse_at_n1(self) -> None:
        rows = describe_analysis_modes(format="json")
        for r in rows:
            if r["signal"] != "sparse":
                continue
            ts = r["timeseries"]
            assert isinstance(ts, dict)
            assert ts["scope_collapsed"] is True

    def test_references_present_for_panel_entries(self) -> None:
        rows = describe_analysis_modes(format="json")
        for r in rows:
            panel = r["panel"]
            assert panel is not None
            assert isinstance(panel["references"], list)
            assert all(isinstance(ref, str) for ref in panel["references"])

    def test_text_distinguishes_collapse_vs_single_series(self) -> None:
        """A-8 review fix: rendered text names what TIMESERIES actually tests
        (single-series null) instead of implying parity with PANEL."""
        out = describe_analysis_modes(format="text")
        # Sparse cells get the scope-collapse note (true collapse).
        assert "scope axis collapsed at N=1" in out
        # COMMON × CONTINUOUS TIMESERIES keeps a different annotation that
        # warns the reader the null is not the cross-asset E[β]=0 of PANEL.
        assert "single-series test" in out

    def test_text_includes_factory_call_per_row(self) -> None:
        """UX-5 review fix: each cell row prints the factory call that
        constructs the corresponding AnalysisConfig — answers
        "which factory do I call?" without README cross-reference."""
        out = describe_analysis_modes(format="text")
        for call in (
            "AnalysisConfig.individual_continuous(metric=Metric.IC)",
            "AnalysisConfig.individual_continuous(metric=Metric.FM)",
            "AnalysisConfig.individual_sparse()",
            "AnalysisConfig.common_continuous()",
            "AnalysisConfig.common_sparse()",
        ):
            assert call in out, f"missing factory call: {call}"


# ---------------------------------------------------------------------------
# suggest_config — synthetic data fixtures
# ---------------------------------------------------------------------------


def _make_individual_continuous_panel_n(
    n_assets: int, *, n_dates: int = 60, seed: int = 17,
) -> pl.DataFrame:
    """Factor varies across assets at each date; ``n_assets`` is parametric."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for t in range(n_dates):
        d = dt.date(2024, 1, 1) + dt.timedelta(days=t)
        for j in range(n_assets):
            rows.append({
                "date": d, "asset_id": f"A{j:03d}",
                "factor": float(rng.standard_normal()),
                "forward_return": float(rng.standard_normal()),
            })
    return pl.DataFrame(rows)


def _make_individual_continuous_panel(seed: int = 1) -> pl.DataFrame:
    """Factor varies across assets at each date (fixed n_assets=20)."""
    return _make_individual_continuous_panel_n(20, seed=seed)


def _make_common_continuous_panel(seed: int = 2) -> pl.DataFrame:
    """Broadcast factor — same value for every asset on a given date."""
    rng = np.random.default_rng(seed)
    n_dates, n_assets = 60, 15
    rows: list[dict[str, object]] = []
    for t in range(n_dates):
        d = dt.date(2024, 1, 1) + dt.timedelta(days=t)
        f_t = float(rng.standard_normal())  # SAME for every asset
        for j in range(n_assets):
            rows.append({
                "date": d, "asset_id": f"A{j:03d}",
                "factor": f_t,
                "forward_return": float(rng.standard_normal()),
            })
    return pl.DataFrame(rows)


def _make_sparse_panel(seed: int = 3) -> pl.DataFrame:
    """Sparse triggers: most factor values are 0."""
    rng = np.random.default_rng(seed)
    n_dates, n_assets = 60, 15
    factor = rng.choice([-1.0, 0.0, 1.0], size=(n_dates, n_assets), p=[0.04, 0.92, 0.04])
    rows: list[dict[str, object]] = []
    for t in range(n_dates):
        d = dt.date(2024, 1, 1) + dt.timedelta(days=t)
        for j in range(n_assets):
            rows.append({
                "date": d, "asset_id": f"A{j:03d}",
                "factor": float(factor[t, j]),
                "forward_return": float(rng.standard_normal()),
            })
    return pl.DataFrame(rows)


def _make_timeseries(*, n_dates: int, sparse: bool, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    if sparse:
        factor = rng.choice([-1.0, 0.0, 1.0], size=n_dates, p=[0.04, 0.92, 0.04])
    else:
        factor = rng.standard_normal(n_dates)
    rows = [
        {
            "date": dt.date(2024, 1, 1) + dt.timedelta(days=t),
            "asset_id": "SPY",
            "factor": float(factor[t]),
            "forward_return": float(rng.standard_normal()),
        }
        for t in range(n_dates)
    ]
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# suggest_config — routing
# ---------------------------------------------------------------------------


class TestSuggestConfigRouting:
    def test_individual_continuous_panel_suggests_individual_continuous_ic(self) -> None:
        result = suggest_config(_make_individual_continuous_panel())
        assert result.suggested == AnalysisConfig.individual_continuous(metric=Metric.IC)

    def test_common_continuous_panel_suggests_common_continuous(self) -> None:
        result = suggest_config(_make_common_continuous_panel())
        assert result.suggested == AnalysisConfig.common_continuous()

    def test_sparse_panel_suggests_individual_sparse(self) -> None:
        # Sparse triggers vary per asset → individual_sparse.
        result = suggest_config(_make_sparse_panel())
        assert result.suggested == AnalysisConfig.individual_sparse()

    def test_continuous_timeseries_suggests_common_continuous(self) -> None:
        ts = _make_timeseries(n_dates=80, sparse=False, seed=11)
        result = suggest_config(ts)
        assert result.suggested == AnalysisConfig.common_continuous()

    def test_sparse_timeseries_suggests_common_sparse(self) -> None:
        ts = _make_timeseries(n_dates=80, sparse=True, seed=12)
        result = suggest_config(ts)
        # At N=1, scope axis is trivially COMMON (and would collapse anyway).
        assert result.suggested == AnalysisConfig.common_sparse()


# ---------------------------------------------------------------------------
# suggest_config — reasoning + warnings
# ---------------------------------------------------------------------------


class TestSuggestConfigReasoning:
    def test_reasoning_has_four_invariant_keys(self) -> None:
        result = suggest_config(_make_individual_continuous_panel())
        assert set(result.reasoning.keys()) == {"scope", "signal", "metric", "mode"}

    def test_reasoning_values_are_strings(self) -> None:
        result = suggest_config(_make_common_continuous_panel())
        for v in result.reasoning.values():
            assert isinstance(v, str)
            assert v  # non-empty

    def test_metric_reasoning_explains_collapse_for_common(self) -> None:
        result = suggest_config(_make_common_continuous_panel())
        assert "collapsed" in result.reasoning["metric"]

    def test_mode_reasoning_picks_panel_when_multi_asset(self) -> None:
        result = suggest_config(_make_individual_continuous_panel())
        assert "PANEL" in result.reasoning["mode"]

    def test_mode_reasoning_picks_timeseries_when_n1(self) -> None:
        ts = _make_timeseries(n_dates=80, sparse=False, seed=13)
        result = suggest_config(ts)
        assert "TIMESERIES" in result.reasoning["mode"]


class TestSuggestConfigWarnings:
    def test_warnings_is_list_of_warning_codes(self) -> None:
        result = suggest_config(_make_individual_continuous_panel())
        assert isinstance(result.warnings, list)
        assert all(isinstance(w, WarningCode) for w in result.warnings)

    def test_short_timeseries_emits_unreliable_se_warning(self) -> None:
        ts = _make_timeseries(
            n_dates=MIN_PERIODS_HARD + 2, sparse=False, seed=14,
        )
        result = suggest_config(ts)
        assert WarningCode.UNRELIABLE_SE_SHORT_SERIES in result.warnings

    def test_long_timeseries_no_warning(self) -> None:
        ts = _make_timeseries(
            n_dates=MIN_PERIODS_RELIABLE + 50, sparse=False, seed=15,
        )
        result = suggest_config(ts)
        assert WarningCode.UNRELIABLE_SE_SHORT_SERIES not in result.warnings


# ---------------------------------------------------------------------------
# suggest_config — sparse magnitude drop warning (issue #8)
# ---------------------------------------------------------------------------


def _make_sparse_weighted_panel(seed: int = 21) -> pl.DataFrame:
    """Sparse layout but non-zero values are continuous magnitudes (SUE-like)."""
    rng = np.random.default_rng(seed)
    n_dates, n_assets = 60, 15
    is_event = rng.choice([0.0, 1.0], size=(n_dates, n_assets), p=[0.92, 0.08])
    magnitudes = rng.standard_normal((n_dates, n_assets)) * 2.0
    factor = is_event * magnitudes
    rows: list[dict[str, object]] = []
    for t in range(n_dates):
        d = dt.date(2024, 1, 1) + dt.timedelta(days=t)
        for j in range(n_assets):
            rows.append({
                "date": d, "asset_id": f"A{j:03d}",
                "factor": float(factor[t, j]),
                "forward_return": float(rng.standard_normal()),
            })
    return pl.DataFrame(rows)


class TestSparseMagnitudeWarning:
    def test_pure_ternary_sparse_no_magnitude_warning(self) -> None:
        result = suggest_config(_make_sparse_panel())
        assert WarningCode.SPARSE_MAGNITUDE_DROPPED not in result.warnings

    def test_continuous_sparse_emits_magnitude_warning(self) -> None:
        result = suggest_config(_make_sparse_weighted_panel())
        assert result.suggested.signal is Signal.SPARSE
        assert WarningCode.SPARSE_MAGNITUDE_DROPPED in result.warnings

    def test_continuous_dense_no_magnitude_warning(self) -> None:
        result = suggest_config(_make_individual_continuous_panel())
        assert WarningCode.SPARSE_MAGNITUDE_DROPPED not in result.warnings

    def test_signal_reasoning_mentions_coercion_when_dropped(self) -> None:
        result = suggest_config(_make_sparse_weighted_panel())
        assert ".sign()" in result.reasoning["signal"]


# ---------------------------------------------------------------------------
# suggest_config — n_assets two-tier guard (issue #15)
# ---------------------------------------------------------------------------


class TestSuggestConfigCrossSectionNWarnings:
    def test_panel_n5_emits_small(self) -> None:
        result = suggest_config(_make_individual_continuous_panel_n(5))
        assert WarningCode.SMALL_CROSS_SECTION_N in result.warnings
        assert WarningCode.BORDERLINE_CROSS_SECTION_N not in result.warnings

    def test_panel_n15_emits_borderline(self) -> None:
        result = suggest_config(_make_individual_continuous_panel_n(15))
        assert WarningCode.BORDERLINE_CROSS_SECTION_N in result.warnings
        assert WarningCode.SMALL_CROSS_SECTION_N not in result.warnings

    def test_panel_n35_no_n_warning(self) -> None:
        result = suggest_config(_make_individual_continuous_panel_n(35))
        assert WarningCode.SMALL_CROSS_SECTION_N not in result.warnings
        assert WarningCode.BORDERLINE_CROSS_SECTION_N not in result.warnings

    def test_n1_no_panel_warning(self) -> None:
        # N=1 routes to TIMESERIES, so PANEL guards must not fire.
        ts = _make_timeseries(n_dates=80, sparse=False, seed=33)
        result = suggest_config(ts)
        assert WarningCode.SMALL_CROSS_SECTION_N not in result.warnings
        assert WarningCode.BORDERLINE_CROSS_SECTION_N not in result.warnings

    def test_mode_reasoning_mentions_warning_at_small_n(self) -> None:
        result = suggest_config(_make_individual_continuous_panel_n(5))
        assert "SMALL_CROSS_SECTION_N" in result.reasoning["mode"]


# ---------------------------------------------------------------------------
# Frozen result type
# ---------------------------------------------------------------------------


class TestSuggestConfigResultImmutability:
    def test_frozen(self) -> None:
        import dataclasses

        result = suggest_config(_make_common_continuous_panel())
        assert isinstance(result, SuggestConfigResult)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.suggested = AnalysisConfig.common_continuous()  # type: ignore[misc]
