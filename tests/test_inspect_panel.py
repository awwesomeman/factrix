"""``fx.inspect_panel`` — typed PanelInspection + per-metric verdict (#443)."""

from __future__ import annotations

import math

import factrix as fx
import polars as pl
from factrix._inspect import (
    MetricApplicability,
    PanelInspection,
    PanelProperties,
    PanelReasoning,
    inspect_panel,
)


def _single_asset_panel(n_dates: int = 80) -> pl.DataFrame:
    raw = fx.datasets.make_cs_panel(n_assets=4, n_dates=n_dates)
    first = raw["asset_id"].unique()[0]
    return raw.filter(pl.col("asset_id") == first)


def _common_continuous_panel(n_assets: int = 20, n_dates: int = 80) -> pl.DataFrame:
    raw = fx.datasets.make_cs_panel(n_assets=n_assets, n_dates=n_dates)
    one_per_date = raw.group_by("date").agg(pl.col("factor").first())
    return raw.drop("factor").join(one_per_date, on="date")


def _by_name(info: PanelInspection, name: str) -> MetricApplicability:
    for m in info.metrics:
        if m.spec.name == name:
            return m
    raise KeyError(name)


class TestPanelPropertiesDetection:
    def test_individual_continuous_cs_panel(self):
        info = inspect_panel(fx.datasets.make_cs_panel(n_assets=20, n_dates=80))
        assert info.detected.scope is fx.FactorScope.INDIVIDUAL
        assert info.detected.density is fx.FactorDensity.DENSE
        assert info.detected.structure is fx.DataStructure.PANEL
        assert info.detected.n_assets == 20
        assert info.detected.n_periods == 80
        assert info.detected.n_pairs == 20 * 80
        assert 0.0 <= info.detected.sparse_ratio < 0.5

    def test_n1_routes_to_timeseries(self):
        info = inspect_panel(_single_asset_panel(n_dates=80))
        assert info.detected.structure is fx.DataStructure.TIMESERIES
        assert info.detected.n_assets == 1
        assert "TIMESERIES" in info.reasoning.structure_reason

    def test_common_continuous_detection(self):
        info = inspect_panel(_common_continuous_panel())
        assert info.detected.scope is fx.FactorScope.COMMON
        assert info.detected.density is fx.FactorDensity.DENSE

    def test_empty_panel_sparse_ratio_is_nan(self):
        empty = fx.datasets.make_cs_panel(n_assets=4, n_dates=10).head(0)
        info = inspect_panel(empty)
        assert math.isnan(info.detected.sparse_ratio)
        assert info.detected.n_pairs == 0

    def test_n_pairs_counts_non_null_factor_only(self):
        raw = fx.datasets.make_cs_panel(n_assets=4, n_dates=10)
        with_nulls = raw.with_columns(
            pl.when(pl.int_range(0, raw.height) % 3 == 0)
            .then(None)
            .otherwise(pl.col("factor"))
            .alias("factor")
        )
        info = inspect_panel(with_nulls)
        assert info.detected.n_pairs == with_nulls.drop_nulls("factor").height
        assert info.detected.n_pairs < raw.height


class TestPanelReasoning:
    def test_three_axis_fields_populated(self):
        info = inspect_panel(fx.datasets.make_cs_panel(n_assets=20, n_dates=80))
        assert "INDIVIDUAL" in info.reasoning.scope_reason
        assert "DENSE" in info.reasoning.density_reason
        assert "PANEL" in info.reasoning.structure_reason


class TestCellMatchGate:
    def test_panel_mode_metric_unusable_under_timeseries(self):
        # IC's cell declares structure=PANEL; single-asset panel must reject it.
        info = inspect_panel(_single_asset_panel(n_dates=80))
        ic = _by_name(info, "ic")
        assert ic.usable is False
        assert any("cell mismatch" in b for b in ic.blockers)

    def test_sparse_metric_unusable_under_continuous(self):
        info = inspect_panel(fx.datasets.make_cs_panel(n_assets=20, n_dates=80))
        caar = _by_name(info, "caar")
        assert caar.usable is False
        assert any("cell mismatch" in b for b in caar.blockers)

    def test_only_public_specs_appear(self):
        info = inspect_panel(fx.datasets.make_cs_panel(n_assets=20, n_dates=80))
        names = {m.spec.name for m in info.metrics}
        assert "compute_ic" not in names  # visibility=INTERNAL
        assert "ic" in names


class TestSampleThresholdGate:
    def test_below_min_periods_is_unusable(self):
        info = inspect_panel(fx.datasets.make_cs_panel(n_assets=20, n_dates=15))
        nw = _by_name(info, "ic_newey_west")
        assert nw.usable is False
        assert any("min_periods" in b for b in nw.blockers)

    def test_between_min_and_warn_is_usable_with_warning(self):
        info = inspect_panel(fx.datasets.make_cs_panel(n_assets=20, n_dates=25))
        nw = _by_name(info, "ic_newey_west")
        assert nw.usable is True
        codes = [w.code.value for w in nw.warnings]
        assert "unreliable_se_short_periods" in codes

    def test_above_warn_floor_is_clean(self):
        info = inspect_panel(fx.datasets.make_cs_panel(n_assets=20, n_dates=120))
        nw = _by_name(info, "ic_newey_west")
        assert nw.usable is True
        assert nw.warnings == []

    def test_below_min_pairs_is_unusable(self):
        from factrix._metric_index import SampleThreshold

        info = inspect_panel(fx.datasets.make_cs_panel(n_assets=20, n_dates=120))
        floor = SampleThreshold(min_pairs=info.detected.n_pairs + 1)
        spec = next(m.spec for m in info.metrics if m.spec.name == "ic_newey_west")
        from dataclasses import replace

        from factrix._inspect import _evaluate_applicability

        verdict = _evaluate_applicability(
            replace(spec, sample_floor=floor), info.detected
        )
        assert verdict.usable is False
        assert any("min_pairs" in b for b in verdict.blockers)

    def test_between_min_and_warn_pairs_is_usable_with_warning(self):
        from dataclasses import replace

        from factrix._inspect import _evaluate_applicability
        from factrix._metric_index import SampleThreshold

        info = inspect_panel(fx.datasets.make_cs_panel(n_assets=20, n_dates=120))
        n = info.detected.n_pairs
        floor = SampleThreshold(min_pairs=n - 1, warn_pairs=n + 1)
        spec = next(m.spec for m in info.metrics if m.spec.name == "ic_newey_west")
        verdict = _evaluate_applicability(
            replace(spec, sample_floor=floor), info.detected
        )
        assert verdict.usable is True
        assert any("n_pairs" in w.message for w in verdict.warnings)

    def test_metric_without_sample_floor_only_cell_checked(self):
        info = inspect_panel(fx.datasets.make_cs_panel(n_assets=20, n_dates=15))
        ic = _by_name(info, "ic")
        # ic has no sample_floor declared (only cell match); cell matches, so usable
        assert ic.usable is True
        assert ic.warnings == []


class TestTierPartition:
    def test_three_tiers_partition_metrics_disjointly(self):
        # n_dates=25 puts ic_newey_west between min and warn -> degraded.
        info = inspect_panel(fx.datasets.make_cs_panel(n_assets=20, n_dates=25))
        usable = {m.spec.name for m in info.usable}
        degraded = {m.spec.name for m in info.degraded}
        unusable = {m.spec.name for m in info.unusable}
        # Mutually exclusive.
        assert usable & degraded == set()
        assert usable & unusable == set()
        assert degraded & unusable == set()
        # Exhaustive: union covers every verdict exactly once.
        assert len(info.usable) + len(info.degraded) + len(info.unusable) == len(
            info.metrics
        )
        assert usable | degraded | unusable == {m.spec.name for m in info.metrics}

    def test_usable_excludes_degraded(self):
        info = inspect_panel(fx.datasets.make_cs_panel(n_assets=20, n_dates=25))
        nw = _by_name(info, "ic_newey_west")
        assert nw.usable is True and nw.warnings  # the degraded case
        assert nw not in info.usable
        assert nw in info.degraded

    def test_usable_is_clean_only(self):
        info = inspect_panel(fx.datasets.make_cs_panel(n_assets=20, n_dates=120))
        assert all(m.usable and not m.warnings for m in info.usable)

    def test_degraded_is_usable_with_warnings(self):
        info = inspect_panel(fx.datasets.make_cs_panel(n_assets=20, n_dates=25))
        assert all(m.usable and m.warnings for m in info.degraded)

    def test_unusable_is_not_usable(self):
        info = inspect_panel(_single_asset_panel(n_dates=80))
        assert all(not m.usable for m in info.unusable)
        # ic (cell=PANEL) is unusable on a single-asset panel.
        assert "ic" in {m.spec.name for m in info.unusable}


class TestPanelLevelWarnings:
    def test_thin_panel_short_n_periods_emits_panel_warning(self):
        info = inspect_panel(_single_asset_panel(n_dates=25))
        codes = [w.code.value for w in info.warnings]
        assert "unreliable_se_short_periods" in codes
        assert all(w.source is None for w in info.warnings)

    def test_thin_cross_section_emits_tier(self):
        info = inspect_panel(fx.datasets.make_cs_panel(n_assets=5, n_dates=120))
        codes = [w.code.value for w in info.warnings]
        assert any("cross_section" in c for c in codes)


class TestToDict:
    def test_round_trips_through_json(self):
        import json

        info = inspect_panel(fx.datasets.make_cs_panel(n_assets=20, n_dates=25))
        d = info.to_dict()
        back = json.loads(json.dumps(d))
        assert back["detected"]["scope"] == "individual"
        assert back["detected"]["structure"] == "panel"
        assert back["detected"]["n_assets"] == 20
        assert back["reasoning"]["scope"]
        assert any(m["name"] == "ic_newey_west" for m in back["metrics"])
        nw = next(m for m in back["metrics"] if m["name"] == "ic_newey_west")
        assert nw["usable"] is True
        assert any(w["code"] == "unreliable_se_short_periods" for w in nw["warnings"])

    def test_nan_sparse_ratio_becomes_null(self):
        empty = fx.datasets.make_cs_panel(n_assets=4, n_dates=10).head(0)
        d = inspect_panel(empty).to_dict()
        assert d["detected"]["sparse_ratio"] is None
        assert d["detected"]["n_pairs"] == 0

    def test_panel_level_warnings_serialised(self):
        info = inspect_panel(fx.datasets.make_cs_panel(n_assets=5, n_dates=120))
        d = info.to_dict()
        assert any(w["source"] is None for w in d["warnings"])


class TestReprHtml:
    def test_smoke(self):
        info = inspect_panel(fx.datasets.make_cs_panel(n_assets=20, n_dates=80))
        out = info._repr_html_()
        assert "PanelInspection" in out
        assert "usable" in out


class TestPublicSurface:
    def test_dataclass_types_exported(self):
        assert fx.PanelInspection is PanelInspection
        assert fx.PanelProperties is PanelProperties
        assert fx.PanelReasoning is PanelReasoning
        assert fx.MetricApplicability is MetricApplicability
        assert fx.inspect_panel is inspect_panel
        assert fx.SampleThreshold is not None


class TestCellMatchesSignature:
    def test_matches_skips_mode_when_omitted(self):
        from factrix._metric_index import cell

        c = cell(
            fx.FactorScope.INDIVIDUAL,
            fx.FactorDensity.DENSE,
            structure=fx.DataStructure.PANEL,
        )
        assert c.matches(fx.FactorScope.INDIVIDUAL, fx.FactorDensity.DENSE) is True

    def test_matches_with_mode_rejects_mismatch(self):
        from factrix._metric_index import cell

        c = cell(
            fx.FactorScope.INDIVIDUAL,
            fx.FactorDensity.DENSE,
            structure=fx.DataStructure.PANEL,
        )
        assert (
            c.matches(
                fx.FactorScope.INDIVIDUAL,
                fx.FactorDensity.DENSE,
                structure=fx.DataStructure.TIMESERIES,
            )
            is False
        )

    def test_matches_wildcard_mode_accepts_anything(self):
        from factrix._metric_index import cell

        c = cell(fx.FactorScope.INDIVIDUAL, fx.FactorDensity.DENSE)
        assert (
            c.matches(
                fx.FactorScope.INDIVIDUAL,
                fx.FactorDensity.DENSE,
                structure=fx.DataStructure.TIMESERIES,
            )
            is True
        )
