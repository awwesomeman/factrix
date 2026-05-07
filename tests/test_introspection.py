"""v0.5 introspection helpers — describe_analysis_modes + suggest_config."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest
from factrix._analysis_config import AnalysisConfig
from factrix._axis import Metric
from factrix._codes import WarningCode
from factrix._describe import (
    DETECTED_KEYS,
    SuggestConfigResult,
    describe_analysis_modes,
    suggest_config,
)
from factrix._stats.constants import MIN_PERIODS_HARD, MIN_PERIODS_WARN

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
            "scope",
            "signal",
            "metric",
            "panel",
            "timeseries",
        }

    def test_individual_continuous_ic_routing(self) -> None:
        rows = describe_analysis_modes(format="json")
        ic_row = next(
            r
            for r in rows
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
            r for r in rows if r["scope"] == "common" and r["signal"] == "continuous"
        )
        assert cc_row["panel"] is not None
        assert isinstance(cc_row["timeseries"], dict)
        # No collapse on common × continuous.
        assert cc_row["timeseries"]["scope_collapsed"] is False

    def test_stats_keys_present_per_mode(self) -> None:
        # Every PANEL / TIMESERIES sub-row carries a sorted list of
        # StatCode .value strings; agents pre-validate verdict(gate=...)
        # / bhy(gate=...) reachability against this set.
        rows = describe_analysis_modes(format="json")
        for r in rows:
            for mode_dict in (r["panel"], r["timeseries"]):
                if not isinstance(mode_dict, dict):
                    continue
                keys = mode_dict["stats_keys"]
                assert isinstance(keys, list)
                assert keys == sorted(keys)
                assert all(isinstance(k, str) for k in keys)
                assert keys, "stats_keys should not be empty"

    def test_stats_keys_match_emits_stats(self) -> None:
        # JSON output must reflect the procedure's declared EMITS_STATS
        # — round-trip via describe_analysis_modes against the registry.
        from factrix._registry import _DISPATCH_REGISTRY

        rows = describe_analysis_modes(format="json")
        cc_row = next(
            r for r in rows if r["scope"] == "common" and r["signal"] == "continuous"
        )
        # PANEL entry of (COMMON, CONTINUOUS) is _CommonContPanelProcedure
        # which emits TS_BETA / TS_BETA_T_NW / TS_BETA_P / FACTOR_ADF_P.
        assert "factor_adf_p" in cc_row["panel"]["stats_keys"]
        assert "ts_beta_p" in cc_row["panel"]["stats_keys"]
        # The (COMMON, CONTINUOUS, TIMESERIES) procedure also emits
        # NW_LAGS_USED in addition to FACTOR_ADF_P.
        assert "nw_lags_used" in cc_row["timeseries"]["stats_keys"]
        del _DISPATCH_REGISTRY  # reference imported above; satisfies linter

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
    n_assets: int,
    *,
    n_dates: int = 60,
    seed: int = 17,
) -> pl.DataFrame:
    """Factor varies across assets at each date; ``n_assets`` is parametric."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for t in range(n_dates):
        d = dt.date(2024, 1, 1) + dt.timedelta(days=t)
        for j in range(n_assets):
            rows.append(
                {
                    "date": d,
                    "asset_id": f"A{j:03d}",
                    "factor": float(rng.standard_normal()),
                    "forward_return": float(rng.standard_normal()),
                }
            )
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
            rows.append(
                {
                    "date": d,
                    "asset_id": f"A{j:03d}",
                    "factor": f_t,
                    "forward_return": float(rng.standard_normal()),
                }
            )
    return pl.DataFrame(rows)


def _make_sparse_panel(seed: int = 3) -> pl.DataFrame:
    """Sparse triggers: most factor values are 0."""
    rng = np.random.default_rng(seed)
    n_dates, n_assets = 60, 15
    factor = rng.choice(
        [-1.0, 0.0, 1.0], size=(n_dates, n_assets), p=[0.04, 0.92, 0.04]
    )
    rows: list[dict[str, object]] = []
    for t in range(n_dates):
        d = dt.date(2024, 1, 1) + dt.timedelta(days=t)
        for j in range(n_assets):
            rows.append(
                {
                    "date": d,
                    "asset_id": f"A{j:03d}",
                    "factor": float(factor[t, j]),
                    "forward_return": float(rng.standard_normal()),
                }
            )
    return pl.DataFrame(rows)


def _make_common_sparse_panel(seed: int = 4) -> pl.DataFrame:
    """Broadcast sparse — same {-1, 0, +1} value for every asset on a date."""
    rng = np.random.default_rng(seed)
    n_dates, n_assets = 60, 15
    factor = rng.choice([-1.0, 0.0, 1.0], size=n_dates, p=[0.10, 0.80, 0.10])
    rows: list[dict[str, object]] = []
    for t in range(n_dates):
        d = dt.date(2024, 1, 1) + dt.timedelta(days=t)
        f_t = float(factor[t])
        for j in range(n_assets):
            rows.append(
                {
                    "date": d,
                    "asset_id": f"A{j:03d}",
                    "factor": f_t,
                    "forward_return": float(rng.standard_normal()),
                }
            )
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
    def test_individual_continuous_panel_suggests_individual_continuous_ic(
        self,
    ) -> None:
        result = suggest_config(_make_individual_continuous_panel())
        assert result.suggested == AnalysisConfig.individual_continuous(
            metric=Metric.IC
        )

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
            n_dates=MIN_PERIODS_HARD + 2,
            sparse=False,
            seed=14,
        )
        result = suggest_config(ts)
        assert WarningCode.UNRELIABLE_SE_SHORT_PERIODS in result.warnings

    def test_long_timeseries_no_warning(self) -> None:
        ts = _make_timeseries(
            n_dates=MIN_PERIODS_WARN + 50,
            sparse=False,
            seed=15,
        )
        result = suggest_config(ts)
        assert WarningCode.UNRELIABLE_SE_SHORT_PERIODS not in result.warnings


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


class TestSuggestConfigCommonPanelInferenceN:
    """Issue #83 — COMMON × * PANEL warning must reflect the post-filter
    inference-stage N, not the panel-union ``n_assets``.

    ``compute_ts_betas`` drops assets with fewer than ``MIN_TS_OBS = 20``
    non-null factor / forward-return rows. A panel can have a wide union
    (say 50 assets) but a small inference-stage cross-section if most
    assets only enter for a short tail. The pre-fix preview optimistically
    cleared the warning; the fix routes the filtered count to
    ``cross_section_tier``.
    """

    @staticmethod
    def _short_history_panel(
        *,
        n_full_history_assets: int,
        n_short_history_assets: int,
        n_dates: int = 60,
        seed: int = 91,
    ) -> pl.DataFrame:
        """Broadcast factor; only the first ``n_full_history_assets`` have
        ≥MIN_TS_OBS rows of valid (factor, forward_return). The rest only
        enter on the final 5 dates — well below the per-asset filter."""
        rng = np.random.default_rng(seed)
        rows: list[dict[str, object]] = []
        for t in range(n_dates):
            d = dt.date(2024, 1, 1) + dt.timedelta(days=t)
            f_t = float(rng.standard_normal())
            for j in range(n_full_history_assets):
                rows.append(
                    {
                        "date": d,
                        "asset_id": f"FULL{j:03d}",
                        "factor": f_t,
                        "forward_return": float(rng.standard_normal()),
                    }
                )
            if t >= n_dates - 5:
                for j in range(n_short_history_assets):
                    rows.append(
                        {
                            "date": d,
                            "asset_id": f"SHORT{j:03d}",
                            "factor": f_t,
                            "forward_return": float(rng.standard_normal()),
                        }
                    )
        return pl.DataFrame(rows)

    def test_optimistic_warning_caught_when_union_is_large(self) -> None:
        # Union n_assets = 5 + 40 = 45 (would be clean at MIN_ASSETS_WARN=30);
        # inference-stage N = 5 (only FULL assets clear MIN_TS_OBS).
        panel = self._short_history_panel(
            n_full_history_assets=5,
            n_short_history_assets=40,
        )
        result = suggest_config(panel)
        assert result.detected["n_assets"] == 45
        assert WarningCode.SMALL_CROSS_SECTION_N in result.warnings

    def test_borderline_inference_n_emits_borderline(self) -> None:
        # Union = 50, inference-stage = 15 → BORDERLINE.
        panel = self._short_history_panel(
            n_full_history_assets=15,
            n_short_history_assets=35,
        )
        result = suggest_config(panel)
        assert WarningCode.BORDERLINE_CROSS_SECTION_N in result.warnings
        assert WarningCode.SMALL_CROSS_SECTION_N not in result.warnings

    def test_clean_when_inference_n_is_high(self) -> None:
        # Union = 35, inference-stage = 35 → no tier warning.
        panel = self._short_history_panel(
            n_full_history_assets=35,
            n_short_history_assets=0,
        )
        result = suggest_config(panel)
        assert WarningCode.SMALL_CROSS_SECTION_N not in result.warnings
        assert WarningCode.BORDERLINE_CROSS_SECTION_N not in result.warnings

    def test_mode_reason_flags_filter_when_union_diverges(self) -> None:
        panel = self._short_history_panel(
            n_full_history_assets=5,
            n_short_history_assets=40,
        )
        result = suggest_config(panel)
        assert "post-MIN_TS_OBS filter" in result.reasoning["mode"]


# ---------------------------------------------------------------------------
# suggest_config — detected (issue #20)
# ---------------------------------------------------------------------------


class TestSuggestConfigDetected:
    """`detected` carries the structured panel observations behind the
    suggestion. Same data the reasoning strings narrate, but
    machine-readable for AI agents / pipeline gates."""

    @pytest.mark.parametrize(
        "fixture_factory",
        [
            lambda: _make_individual_continuous_panel(),
            lambda: _make_sparse_panel(),
            lambda: _make_timeseries(n_dates=80, sparse=False, seed=41),
        ],
        ids=["individual_continuous", "sparse_panel", "timeseries"],
    )
    def test_keys_match_canonical_set(self, fixture_factory) -> None:
        result = suggest_config(fixture_factory())
        assert set(result.detected.keys()) == DETECTED_KEYS

    def test_individual_continuous_values(self) -> None:
        result = suggest_config(_make_individual_continuous_panel())
        d = result.detected
        assert d["scope"] == "individual"
        assert d["signal"] == "continuous"
        assert d["mode"] == "panel"
        assert d["n_assets"] == 20
        assert d["n_periods"] == 60
        assert 0.0 <= d["sparsity"] < 0.5

    def test_sparse_values(self) -> None:
        result = suggest_config(_make_sparse_panel())
        d = result.detected
        assert d["signal"] == "sparse"
        assert d["sparsity"] >= 0.5

    def test_timeseries_mode(self) -> None:
        ts = _make_timeseries(n_dates=80, sparse=False, seed=42)
        d = suggest_config(ts).detected
        assert d["mode"] == "timeseries"
        assert d["n_assets"] == 1
        assert d["n_periods"] == 80

    def test_consistency_with_suggested_config(self) -> None:
        # detected.scope/signal must round-trip to the suggested config's axes.
        for fixture in (
            _make_individual_continuous_panel(),
            _make_common_continuous_panel(),
            _make_sparse_panel(),
        ):
            result = suggest_config(fixture)
            assert result.detected["scope"] == result.suggested.scope.value
            assert result.detected["signal"] == result.suggested.signal.value


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


# ---------------------------------------------------------------------------
# EMITS_STATS drift guard — every procedure's actually-emitted stats keys
# must be a subset of its declared EMITS_STATS, otherwise describe_*'s
# stats_keys field lies to agents pre-validating verdict()/bhy() gates.
# ---------------------------------------------------------------------------


def _emits_stats_cases() -> list[tuple[str, AnalysisConfig, pl.DataFrame]]:
    """One (cell-label, config, panel) per registered procedure."""
    return [
        (
            "ind_cont_ic_panel",
            AnalysisConfig.individual_continuous(metric=Metric.IC),
            _make_individual_continuous_panel(),
        ),
        (
            "ind_cont_fm_panel",
            AnalysisConfig.individual_continuous(metric=Metric.FM),
            _make_individual_continuous_panel(),
        ),
        (
            "ind_sparse_panel",
            AnalysisConfig.individual_sparse(),
            _make_sparse_panel(),
        ),
        (
            "com_cont_panel",
            AnalysisConfig.common_continuous(),
            _make_common_continuous_panel(),
        ),
        (
            "com_sparse_panel",
            AnalysisConfig.common_sparse(),
            _make_common_sparse_panel(),
        ),
        (
            "com_cont_timeseries",
            AnalysisConfig.common_continuous(),
            _make_timeseries(n_dates=80, sparse=False, seed=11),
        ),
        (
            "sparse_timeseries_collapsed",
            AnalysisConfig.common_sparse(),
            _make_timeseries(n_dates=80, sparse=True, seed=12),
        ),
    ]


class TestEmitsStatsDrift:
    @pytest.mark.parametrize(
        "label,config,panel",
        _emits_stats_cases(),
        ids=lambda v: v if isinstance(v, str) else "",
    )
    def test_actual_stats_subset_of_declared(
        self,
        label: str,
        config: AnalysisConfig,
        panel: pl.DataFrame,
    ) -> None:
        # Dispatch through the registry, then verify the procedure's
        # populated profile.stats keys are a subset of its declared
        # EMITS_STATS. Catches drift if a new stats[StatCode.X] = ...
        # is added in compute() without updating the class constant.
        from factrix._evaluate import _derive_mode
        from factrix._registry import _DISPATCH_REGISTRY, _dispatch_key_for

        mode = _derive_mode(panel)
        key = _dispatch_key_for(config.scope, config.signal, config.metric, mode)
        entry = _DISPATCH_REGISTRY[key]

        profile = entry.procedure.compute(panel, config)
        actual = set(profile.stats.keys())
        declared = entry.procedure.EMITS_STATS
        assert actual <= declared, (
            f"{label}: actual stats {sorted(s.value for s in actual - declared)} "
            f"not in declared EMITS_STATS"
        )


class TestSuggestConfigResultDiagnose:
    def test_diagnose_shape(self) -> None:
        # Symmetric with FactorProfile.diagnose(): JSON-shape dict with
        # str-valued warnings for wire / log / agent consumption.
        result = suggest_config(_make_individual_continuous_panel())
        d = result.diagnose()
        assert set(d.keys()) == {"suggested", "detected", "reasoning", "warnings"}
        assert d["suggested"] == result.suggested.to_dict()
        assert d["detected"] == result.detected
        assert d["reasoning"] == result.reasoning
        assert d["warnings"] == sorted(w.value for w in result.warnings)

    def test_diagnose_warnings_are_strings(self) -> None:
        # Even when warnings list is non-empty, diagnose() emits .value
        # strings, not enum members — agents reading JSON should never
        # see Python enum repr.
        ts = _make_timeseries(n_dates=80, sparse=False, seed=11)
        d = suggest_config(ts).diagnose()
        for w in d["warnings"]:
            assert isinstance(w, str)

    def test_diagnose_is_jsonable(self) -> None:
        # End-to-end: diagnose() output round-trips through json.dumps
        # without a custom encoder (this is the canonical wire format).
        import json

        result = suggest_config(_make_common_continuous_panel())
        json.dumps(result.diagnose())
