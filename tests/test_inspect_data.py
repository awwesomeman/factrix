"""``fx.inspect_data`` -- typed DataInspection + per-metric verdict."""

from __future__ import annotations

import math
import warnings

import factrix as fx
import polars as pl
import pytest
from factrix._inspect import (
    DataInspection,
    DataProperties,
    MetricApplicability,
    inspect_data,
)
from factrix.preprocess import compute_forward_return


def _single_asset_data(n_dates: int = 80) -> pl.DataFrame:
    raw = fx.datasets.make_cs_panel(n_assets=4, n_dates=n_dates)
    first = raw["asset_id"].unique()[0]
    return raw.filter(pl.col("asset_id") == first)


def _common_continuous_data(n_assets: int = 20, n_dates: int = 80) -> pl.DataFrame:
    raw = fx.datasets.make_cs_panel(n_assets=n_assets, n_dates=n_dates)
    one_per_date = raw.group_by("date").agg(pl.col("factor").first())
    return raw.drop("factor").join(one_per_date, on="date")


def _by_name(info: DataInspection, name: str) -> MetricApplicability:
    for m in info.metrics:
        if m.spec.name == name:
            return m
    raise KeyError(name)


class TestDataPropertiesDetection:
    def test_individual_continuous_cs_panel(self):
        info = inspect_data(fx.datasets.make_cs_panel(n_assets=20, n_dates=80))
        assert info.properties.scope is fx.FactorScope.INDIVIDUAL
        assert info.properties.density is fx.FactorDensity.DENSE
        assert info.properties.structure is fx.DataStructure.PANEL
        assert info.properties.n_assets == 20
        assert info.properties.n_periods == 80
        assert info.properties.n_pairs == 20 * 80
        assert 0.0 <= info.properties.sparse_ratio < 0.5

    def test_n1_routes_to_timeseries(self):
        info = inspect_data(_single_asset_data(n_dates=80))
        assert info.properties.structure is fx.DataStructure.TIMESERIES
        assert info.properties.n_assets == 1
        assert "TIMESERIES" in info.properties.structure_reason

    def test_common_continuous_detection(self):
        info = inspect_data(_common_continuous_data())
        assert info.properties.scope is fx.FactorScope.COMMON
        assert info.properties.density is fx.FactorDensity.DENSE

    def test_empty_panel_sparse_ratio_is_nan(self):
        empty = fx.datasets.make_cs_panel(n_assets=4, n_dates=10).head(0)
        info = inspect_data(empty)
        assert math.isnan(info.properties.sparse_ratio)
        assert info.properties.n_pairs == 0

    def test_n_pairs_counts_non_null_factor_only(self):
        raw = fx.datasets.make_cs_panel(n_assets=4, n_dates=10)
        with_nulls = raw.with_columns(
            pl.when(pl.int_range(0, raw.height) % 3 == 0)
            .then(None)
            .otherwise(pl.col("factor"))
            .alias("factor")
        )
        info = inspect_data(with_nulls)
        assert info.properties.n_pairs == with_nulls.drop_nulls("factor").height
        assert info.properties.n_pairs < raw.height

    def test_sparse_ratio_uses_non_null_factor_denominator(self):
        raw = fx.datasets.make_cs_panel(n_assets=10, n_dates=10)
        data = raw.with_columns(
            pl.when(pl.int_range(0, raw.height) < 60)
            .then(None)
            .when(pl.int_range(0, raw.height) < 99)
            .then(0.0)
            .otherwise(1.0)
            .alias("factor")
        )
        info = inspect_data(data)
        assert info.properties.density is fx.FactorDensity.SPARSE
        assert info.properties.n_pairs == 40
        assert info.properties.n_events == 1
        assert math.isclose(info.properties.sparse_ratio, 39 / 40)

    def test_low_cardinality_dense_signal_warns_without_sparse_routing(self):
        raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=80)
        data = raw.with_columns(
            pl.when(pl.col("factor") >= 0).then(1.0).otherwise(-1.0).alias("factor")
        )
        info = inspect_data(data)

        assert info.properties.density is fx.FactorDensity.DENSE
        assert info.properties.sparse_ratio == 0.0
        warn = [
            w
            for w in info.warnings
            if w.code is fx.WarningCode.LOW_CARDINALITY_DENSE_SIGNAL
        ]
        assert len(warn) == 1
        assert warn[0].source is None
        assert "{-1, +1}" in warn[0].message
        assert "non-events as 0" in warn[0].message

    def test_low_zero_ratio_event_signal_warns_but_keeps_sparse_metrics_runnable(
        self,
    ):
        raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=80)
        data = raw.with_columns(
            pl.when(pl.int_range(0, pl.len()) % 5 < 2)
            .then(0.0)
            .otherwise(1.0)
            .alias("factor")
        )
        info = inspect_data(data)
        caar = _by_name(info, "caar")

        assert info.properties.density is fx.FactorDensity.DENSE
        assert math.isclose(info.properties.sparse_ratio, 0.4)
        assert caar.usable is True
        assert caar.blockers == []
        assert caar in info.degraded
        assert any(
            w.code is fx.WarningCode.FREQUENT_EVENT_SIGNAL for w in caar.warnings
        )


class TestDataReasoning:
    def test_three_axis_fields_populated(self):
        info = inspect_data(fx.datasets.make_cs_panel(n_assets=20, n_dates=80))
        assert "INDIVIDUAL" in info.properties.scope_reason
        assert "DENSE" in info.properties.density_reason
        assert "PANEL" in info.properties.structure_reason


class TestCellMatchGate:
    def test_panel_mode_metric_unusable_under_timeseries(self):
        # IC's cell declares structure=PANEL; single-asset data must reject it.
        info = inspect_data(_single_asset_data(n_dates=80))
        ic = _by_name(info, "ic")
        assert ic.usable is False
        assert any("cell mismatch" in b for b in ic.blockers)

    def test_sparse_metric_unusable_under_continuous(self):
        info = inspect_data(fx.datasets.make_cs_panel(n_assets=20, n_dates=80))
        caar = _by_name(info, "caar")
        assert caar.usable is False
        assert any("cell mismatch" in b for b in caar.blockers)
        assert any("zero non-event" in b for b in caar.blockers)

    def test_only_public_specs_appear(self):
        info = inspect_data(fx.datasets.make_cs_panel(n_assets=20, n_dates=80))
        names = {m.spec.name for m in info.metrics}
        assert "compute_ic" not in names  # visibility=INTERNAL
        assert "ic" in names


class TestMetricApplicabilityIdentity:
    def test_metric_and_name_resolve_to_registry_class(self):
        from factrix.metrics._base import MetricBase
        from factrix.metrics._registry import REGISTRY

        info = inspect_data(fx.datasets.make_cs_panel(n_assets=20, n_dates=80))
        for m in info.metrics:
            assert issubclass(m.metric, MetricBase)
            # name is the registry key and agrees with both the class and spec
            assert m.name == m.spec.name
            assert m.name == m.metric.__name__
            assert REGISTRY[m.name] is m.metric

    def test_name_lets_caller_key_without_spec(self):
        info = inspect_data(fx.datasets.make_cs_panel(n_assets=20, n_dates=80))
        ic = _by_name(info, "ic")
        assert ic.name == "ic"
        assert ic.metric is _by_name(info, "ic").metric


class TestSampleThresholdGate:
    def test_below_min_periods_is_unusable(self):
        info = inspect_data(fx.datasets.make_cs_panel(n_assets=20, n_dates=15))
        nw = _by_name(info, "ic_ir")
        assert nw.usable is False
        assert any("min_periods" in b for b in nw.blockers)

    def test_between_min_and_warn_is_usable_with_warning(self):
        info = inspect_data(fx.datasets.make_cs_panel(n_assets=20, n_dates=25))
        nw = _by_name(info, "ic_ir")
        assert nw.usable is True
        codes = [w.code.value for w in nw.warnings]
        assert "unreliable_se_short_periods" in codes

    def test_above_warn_floor_is_clean(self):
        info = inspect_data(fx.datasets.make_cs_panel(n_assets=20, n_dates=120))
        nw = _by_name(info, "ic_ir")
        assert nw.usable is True
        assert nw.warnings == []

    def test_below_min_pairs_is_unusable(self):
        from factrix._metric_index import SampleThreshold

        info = inspect_data(fx.datasets.make_cs_panel(n_assets=20, n_dates=120))
        floor = SampleThreshold(min_pairs=info.properties.n_pairs + 1)
        spec = next(m.spec for m in info.metrics if m.spec.name == "ic_ir")
        from dataclasses import replace

        from factrix._inspect import _evaluate_applicability

        verdict = _evaluate_applicability(
            replace(spec, sample_threshold=floor),
            info.properties,
            signal_discrete=False,
        )
        assert verdict.usable is False
        assert any("min_pairs" in b for b in verdict.blockers)

    def test_between_min_and_warn_pairs_is_usable_with_warning(self):
        from dataclasses import replace

        from factrix._inspect import _evaluate_applicability
        from factrix._metric_index import SampleThreshold

        info = inspect_data(fx.datasets.make_cs_panel(n_assets=20, n_dates=120))
        n = info.properties.n_pairs
        floor = SampleThreshold(min_pairs=n - 1, warn_pairs=n + 1)
        spec = next(m.spec for m in info.metrics if m.spec.name == "ic_ir")
        verdict = _evaluate_applicability(
            replace(spec, sample_threshold=floor),
            info.properties,
            signal_discrete=False,
        )
        assert verdict.usable is True
        assert any("n_pairs" in w.message for w in verdict.warnings)

    def test_rank_turnover_below_dynamic_floor_unusable(self):
        raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=7)
        dates = raw.select("date").unique().sort("date").head(2)
        info = inspect_data(raw.join(dates, on="date", how="inner"))
        rank_turnover = _by_name(info, "rank_turnover")
        assert rank_turnover.usable is False
        assert any("min_periods" in b for b in rank_turnover.blockers)


class TestICStageOneFeasibility:
    def test_ic_family_blocked_when_cross_section_never_reaches_ic_floor(self):
        raw = compute_forward_return(
            fx.datasets.make_cs_panel(n_assets=20, n_dates=120), forward_periods=3
        )
        keepers = raw["asset_id"].unique().sort().head(1).to_list()
        data = raw.with_columns(
            pl.when(pl.col("asset_id").is_in(keepers))
            .then(pl.col("factor"))
            .otherwise(None)
            .alias("factor")
        )
        info = inspect_data(data)

        for name in ("ic", "ic_ir"):
            verdict = _by_name(info, name)
            assert verdict.usable is False
            assert any("MIN_IC_ASSETS_HARD=2" in b for b in verdict.blockers)

        for name in ("fm_beta", "fm_beta_sign_consistency"):
            verdict = _by_name(info, name)
            assert verdict.usable is False
            assert any("MIN_FM_ASSETS_HARD=3" in b for b in verdict.blockers)

    def test_ic_family_uses_pairwise_complete_assets_not_raw_asset_count(self):
        raw = compute_forward_return(
            fx.datasets.make_cs_panel(n_assets=20, n_dates=120), forward_periods=3
        )
        keepers = raw["asset_id"].unique().sort().head(8).to_list()
        data = raw.with_columns(
            pl.when(pl.col("asset_id").is_in(keepers))
            .then(pl.col("factor"))
            .otherwise(None)
            .alias("factor")
        )

        info = inspect_data(data)
        assert info.properties.n_assets == 20

        for name in ("ic", "ic_ir"):
            verdict = _by_name(info, name)
            assert verdict.usable is True
            assert verdict.blockers == []
            assert "few_assets" in [w.code.value for w in verdict.warnings]
            assert any("MIN_IC_ASSETS_WARN=10" in w.message for w in verdict.warnings)

        for name in ("fm_beta", "fm_beta_sign_consistency"):
            verdict = _by_name(info, name)
            assert verdict.usable is True
            assert verdict.blockers == []
            assert "few_assets" in [w.code.value for w in verdict.warnings]
            assert any("MIN_FM_ASSETS_WARN=10" in w.message for w in verdict.warnings)

    def test_ic_family_period_floor_uses_surviving_ic_series_length(self):
        raw = compute_forward_return(
            fx.datasets.make_cs_panel(n_assets=20, n_dates=40), forward_periods=3
        )
        full_width_cutoff = raw["date"].unique().sort()[14]
        keepers = raw["asset_id"].unique().sort().head(1).to_list()
        data = raw.with_columns(
            pl.when(
                (pl.col("date") <= full_width_cutoff)
                | pl.col("asset_id").is_in(keepers)
            )
            .then(pl.col("factor"))
            .otherwise(None)
            .alias("factor")
        )

        info = inspect_data(data)
        ic_ir = _by_name(info, "ic_ir")

        assert info.properties.n_periods == raw["date"].n_unique()
        assert ic_ir.usable is False
        assert any("n_periods=15 < min_periods=20" in b for b in ic_ir.blockers)
        assert not any("MIN_IC_ASSETS" in b for b in ic_ir.blockers)


class TestDeclaredPeriodsFloorsVisible:
    """Metrics that gate on a periods floor must declare it on their spec so
    ``inspect_data`` can pre-flight it -- previously these enforced the floor in
    the body while declaring an empty ``SampleThreshold()``, hiding it from the
    pre-flight verdict.
    """

    def test_directional_hit_rate_declares_pairs_floors(self):
        from factrix._types import (
            MIN_DIRECTIONAL_PAIRS_HARD,
            MIN_DIRECTIONAL_PAIRS_WARN,
        )
        from factrix.metrics.directional_hit_rate import directional_hit_rate

        # directional_hit_rate gates on pooled (date, asset) directional
        # trials, so its floor lives on the pairs axis -- not periods.
        st = directional_hit_rate.spec().sample_threshold
        assert st.min_periods is None
        assert st.min_pairs == MIN_DIRECTIONAL_PAIRS_HARD
        assert st.warn_pairs == MIN_DIRECTIONAL_PAIRS_WARN

    def test_directional_hit_rate_usable_on_wide_short_panel(self):
        # n_periods (7) < MIN_DIRECTIONAL_PAIRS_HARD but n_pairs (7 * 40 = 280)
        # clears the WARN floor -- the pairs-axis pre-flight must not flag it
        # UNUSABLE the way the old periods-axis floor did.
        info = inspect_data(fx.datasets.make_cs_panel(n_assets=40, n_dates=7))
        da = _by_name(info, "directional_hit_rate")
        assert da.usable is True
        assert da.warnings == []

    def test_directional_hit_rate_advises_on_a_one_sided_factor_sign(self):
        # The run keeps the hit rate and withholds the test, so pre-flight
        # advises rather than refuses (#1116); it used to blocker this shape
        # while strict=True ran it.
        raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=80)
        data = raw.with_columns((pl.col("factor").abs() + 0.1).alias("factor"))

        info = inspect_data(data)
        da = _by_name(info, "directional_hit_rate")

        assert da.usable is True
        assert da not in info.unusable
        assert da.blockers == []
        advisory = next(
            w for w in da.warnings if w.code is fx.WarningCode.DEGENERATE_VARIANCE
        )
        assert "one-sided directional signal" in advisory.message

    def test_top_concentration_declares_periods_floors(self):
        from factrix._types import (
            MIN_PORTFOLIO_PERIODS_HARD,
            MIN_PORTFOLIO_PERIODS_WARN,
        )
        from factrix.metrics._helpers import _scaled_min_periods
        from factrix.metrics.concentration import top_concentration

        # Floor scales with the non-overlap stride; spec() resolves at the
        # default config (overlap_periods=5).
        st = top_concentration.spec().sample_threshold
        assert st.min_periods == _scaled_min_periods(MIN_PORTFOLIO_PERIODS_HARD, 5)
        assert st.warn_periods == _scaled_min_periods(MIN_PORTFOLIO_PERIODS_WARN, 5)

    def test_pooled_beta_declares_periods_floors_alongside_pairs(self):
        from factrix._stats.constants import MIN_PERIODS_WARN
        from factrix.metrics.fm_beta import _MIN_DK_PERIODS_HARD, pooled_beta

        st = pooled_beta.spec().sample_threshold
        assert st.min_pairs == 10
        assert st.min_periods == _MIN_DK_PERIODS_HARD
        assert st.warn_periods == MIN_PERIODS_WARN

    def test_rank_turnover_declares_dynamic_periods_floor(self):
        from factrix.metrics.tradability import (
            _rank_turnover_min_dates,
            rank_turnover,
        )

        # Hook resolves against the default config (overlap_periods=1).
        st = rank_turnover.spec().sample_threshold
        assert st.min_periods == _rank_turnover_min_dates(1)


class TestDeclaredEventFloorsVisible:
    """Event-driven metrics must declare their event floor on the spec so
    ``inspect_data`` can pre-flight it -- previously these enforced the floor in
    the body while declaring an empty ``SampleThreshold()``, hiding it from the
    pre-flight verdict.
    """

    def test_static_event_metrics_declare_min_events(self):
        from factrix._types import MIN_EVENTS_HARD
        from factrix.metrics.caar import bmp_z
        from factrix.metrics.clustering_hhi import clustering_hhi
        from factrix.metrics.corrado_rank import corrado_rank
        from factrix.metrics.event_quality import (
            event_hit_rate,
            event_ic,
            event_skewness,
            profit_factor,
        )
        from factrix.metrics.mfe_mae import mfe_mae

        for m in (
            corrado_rank,
            clustering_hhi,
            mfe_mae,
            bmp_z,
            event_hit_rate,
            event_ic,
            profit_factor,
            event_skewness,
        ):
            assert m.spec().sample_threshold.min_events == MIN_EVENTS_HARD

    def test_caar_declares_scaled_event_floor(self):
        from factrix._types import MIN_EVENTS_HARD, MIN_EVENTS_WARN
        from factrix.metrics._helpers import _scaled_min_periods
        from factrix.metrics.caar import caar

        # Hook resolves against the default config (overlap_periods=5).
        st = caar.spec().sample_threshold
        assert st.min_events == _scaled_min_periods(MIN_EVENTS_HARD, 5)
        assert st.warn_events == _scaled_min_periods(MIN_EVENTS_WARN, 5)


class TestEventAxisPreflight:
    def _event_panel(self):
        return fx.datasets.make_event_panel(n_assets=50, n_dates=400, rng=0)

    def test_n_events_counts_nonzero_factor_rows(self):
        import polars as pl

        panel = self._event_panel()
        info = inspect_data(panel)
        assert info.properties.n_events == panel.filter(pl.col("factor") != 0).height

    def test_below_min_events_is_unusable(self):
        from dataclasses import replace

        from factrix._inspect import _evaluate_applicability
        from factrix._metric_index import SampleThreshold

        info = inspect_data(self._event_panel())
        floor = SampleThreshold(min_events=info.properties.n_events + 1)
        spec = next(m.spec for m in info.metrics if m.spec.name == "corrado_rank")
        verdict = _evaluate_applicability(
            replace(spec, sample_threshold=floor),
            info.properties,
            signal_discrete=False,
        )
        assert verdict.usable is False
        assert any("min_events" in b for b in verdict.blockers)

    def test_between_min_and_warn_events_emits_few_events(self):
        from dataclasses import replace

        from factrix._inspect import _evaluate_applicability
        from factrix._metric_index import SampleThreshold

        info = inspect_data(self._event_panel())
        n = info.properties.n_events
        floor = SampleThreshold(min_events=n - 1, warn_events=n + 1)
        spec = next(m.spec for m in info.metrics if m.spec.name == "corrado_rank")
        verdict = _evaluate_applicability(
            replace(spec, sample_threshold=floor),
            info.properties,
            signal_discrete=False,
        )
        assert verdict.usable is True
        assert "few_events" in [w.code.value for w in verdict.warnings]


class TestContinuousMagnitudePreflight:
    """event_ic declares requires_continuous_magnitude: the pre-flight must
    block it on a discrete +/-k signal, matching its run-time short-circuit."""

    def _ternary_event_panel(self):
        raw = fx.datasets.make_event_panel(n_assets=50, n_dates=400, rng=0)
        return fx.preprocess.compute_forward_return(raw, forward_periods=5)

    def _continuous_magnitude_panel(self):
        # Scale the +/-1 events by a random positive magnitude so |factor| varies.
        import numpy as np

        df = self._ternary_event_panel()
        rng = np.random.default_rng(0)
        scale = pl.Series(rng.uniform(0.5, 2.0, len(df)))
        return df.with_columns(
            pl.when(pl.col("factor") != 0)
            .then(pl.col("factor") * scale)
            .otherwise(0.0)
            .alias("factor")
        )

    def test_event_ic_blocked_on_discrete_signal(self):
        info = inspect_data(self._ternary_event_panel())
        eic = _by_name(info, "event_ic")
        assert eic.usable is False
        assert any("discrete signal" in b for b in eic.blockers)

    def test_event_ic_usable_on_continuous_magnitude(self):
        info = inspect_data(self._continuous_magnitude_panel())
        eic = _by_name(info, "event_ic")
        assert eic.usable is True
        assert not any("discrete" in b for b in eic.blockers)

    def test_preflight_verdict_matches_evaluate(self):
        # The pre-flight gate and the run-time short-circuit share one predicate
        # (_event_signal_is_discrete), so a discrete signal must agree on both
        # paths: inspect_data says unusable AND evaluate short-circuits to NaN.
        panel = self._ternary_event_panel()
        from factrix.metrics import event_ic

        eic = _by_name(inspect_data(panel), "event_ic")
        assert eic.usable is False

        res = fx.evaluate(
            panel,
            metrics={"eic": event_ic()},
            factor_cols=["factor"],
            strict=False,
        )["factor"].metrics["eic"]
        assert math.isnan(res.value)
        assert res.metadata["reason"] == "not_applicable_discrete_signal"


class TestTierPartition:
    def test_three_tiers_partition_metrics_disjointly(self):
        # n_dates=25 puts ic_ir between min and warn -> degraded.
        info = inspect_data(fx.datasets.make_cs_panel(n_assets=20, n_dates=25))
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
        info = inspect_data(fx.datasets.make_cs_panel(n_assets=20, n_dates=25))
        nw = _by_name(info, "ic_ir")
        assert nw.usable is True and nw.warnings  # the degraded case
        assert nw not in info.usable
        assert nw in info.degraded

    def test_usable_is_clean_only(self):
        info = inspect_data(fx.datasets.make_cs_panel(n_assets=20, n_dates=120))
        assert all(m.usable and not m.warnings for m in info.usable)

    def test_degraded_is_usable_with_warnings(self):
        info = inspect_data(fx.datasets.make_cs_panel(n_assets=20, n_dates=25))
        assert all(m.usable and m.warnings for m in info.degraded)

    def test_unusable_is_not_usable(self):
        info = inspect_data(_single_asset_data(n_dates=80))
        assert all(not m.usable for m in info.unusable)
        # ic (cell=PANEL) is unusable on a single-asset data.
        assert "ic" in {m.spec.name for m in info.unusable}


class TestDataLevelWarnings:
    def test_preflight_warnings_echo_and_can_be_declared(self):
        panel = fx.datasets.make_cs_panel(n_assets=5, n_dates=25)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            plain = inspect_data(panel)

        records = [
            *plain.warnings,
            *(warning for metric in plain.metrics for warning in metric.warnings),
        ]
        assert records
        messages = [str(warning.message) for warning in caught]
        for code in {warning.code.value for warning in records}:
            assert any(f"({code};" in message for message in messages), code

        declared = tuple(code.value for code in fx.WarningCode)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            quiet = inspect_data(panel, expected_warnings=declared)
        quiet_records = [
            *quiet.warnings,
            *(warning for metric in quiet.metrics for warning in metric.warnings),
        ]
        assert quiet_records
        assert all(warning.expected for warning in quiet_records)

    def test_expected_warnings_rejects_an_unknown_code(self):
        with pytest.raises(ValueError, match="expected_warnings"):
            inspect_data(
                fx.datasets.make_cs_panel(n_assets=5, n_dates=25),
                expected_warnings=("few_asset",),
            )

    def test_thin_data_short_n_periods_emits_data_warning(self):
        info = inspect_data(_single_asset_data(n_dates=25))
        codes = [w.code.value for w in info.warnings]
        assert "unreliable_se_short_periods" in codes
        assert all(w.source is None for w in info.warnings)

    def test_thin_cross_section_emits_tier(self):
        info = inspect_data(fx.datasets.make_cs_panel(n_assets=5, n_dates=120))
        codes = [w.code.value for w in info.warnings]
        assert "few_assets" in codes

    def test_single_asset_event_data_emits_guidance(self):
        # n_assets=1 event panel -- TIMESERIES + SPARSE: event-axis metrics are
        # usable, but the cross-sectional clustering_hhi stays blocked. The
        # data-level warning names it so its absence from `usable` is explained.
        info = inspect_data(
            fx.datasets.make_event_panel(n_assets=1, n_dates=400, rng=0)
        )
        warn = [w for w in info.warnings if w.code.value == "single_asset_event_data"]
        assert len(warn) == 1
        assert warn[0].source is None
        assert "clustering_hhi" in warn[0].message
        # event-axis metrics run on a single name; the cross-sectional one does not
        assert len(info.usable) > 0
        usable_names = {m.name for m in info.usable}
        assert "clustering_hhi" not in usable_names

    def test_multi_asset_event_panel_no_single_asset_warning(self):
        info = inspect_data(
            fx.datasets.make_event_panel(n_assets=50, n_dates=400, rng=0)
        )
        codes = [w.code.value for w in info.warnings]
        assert "single_asset_event_data" not in codes

    def test_dense_single_asset_no_event_warning(self):
        # A dense single-asset timeseries is not event-shaped; must not fire.
        info = inspect_data(_single_asset_data(n_dates=120))
        codes = [w.code.value for w in info.warnings]
        assert "single_asset_event_data" not in codes


class TestEventAxisSingleAsset:
    """Event metrics whose inference unit is the event cross-section run on
    single-asset (TIMESERIES) data; the cross-sectional clustering_hhi does not.
    """

    @staticmethod
    def _panel() -> pl.DataFrame:
        return fx.datasets.make_event_panel(n_assets=1, n_dates=400, rng=0)

    def test_event_axis_metrics_usable_on_single_asset(self):
        info = inspect_data(self._panel())
        # representative event-axis metrics: nothing about a single asset
        # blocks them. The significance tests among them land in the degraded
        # tier on this panel because its 8 events sit below the stride-scaled
        # FEW_EVENTS floor, not because of the cell.
        runnable = {m.name for m in info.usable} | {m.name for m in info.degraded}
        assert {"bmp_z", "corrado_rank", "mfe_mae", "event_hit_rate"} <= runnable
        assert {m.name for m in info.usable} >= {"mfe_mae"}

    def test_clustering_hhi_stays_cross_sectional(self):
        info = inspect_data(self._panel())
        hhi = _by_name(info, "clustering_hhi")
        assert hhi.usable is False
        assert any("cell mismatch" in b for b in hhi.blockers)

    def test_event_axis_metric_computes_finite_value(self):
        # applicability is not enough -- the primitive must produce a real value
        # on single-asset data, not a structure short-circuit.
        panel = fx.preprocess.compute_forward_return(self._panel(), forward_periods=5)
        res = fx.evaluate(
            panel, metrics={"bmp_z": fx.metrics.bmp_z()}, factor_cols=["factor"]
        )
        value = res["factor"].metrics["bmp_z"].value
        assert value is not None and not math.isnan(value)


class TestWarningSourceConvention:
    """Guard the `Warning.source` convention across every emission point:
    per-metric warnings carry `source == <metric name>`; data-level
    warnings carry `source is None`.
    """

    def test_per_metric_warnings_carry_their_metric_name(self):
        # n_periods=25 puts NW-SE metrics in the degraded tier, n_assets=5
        # trips the cross-section tier -- both per-metric warning paths fire.
        info = inspect_data(fx.datasets.make_cs_panel(n_assets=5, n_dates=25))
        emitted = 0
        for m in info.metrics:
            for w in m.warnings:
                assert w.source == m.name
                emitted += 1
        assert emitted > 0  # the fixture actually exercises the per-metric path

    def test_data_level_warnings_carry_no_source(self):
        info = inspect_data(fx.datasets.make_cs_panel(n_assets=5, n_dates=25))
        assert info.warnings  # fixture produces data-level diagnostics
        assert all(w.source is None for w in info.warnings)


class TestToDict:
    def test_round_trips_through_json(self):
        import json

        info = inspect_data(fx.datasets.make_cs_panel(n_assets=20, n_dates=25))
        d = info.to_dict()
        back = json.loads(json.dumps(d))
        assert back["properties"]["scope"] == "individual"
        assert back["properties"]["structure"] == "panel"
        assert back["properties"]["n_assets"] == 20
        assert back["reasoning"]["scope"]
        assert any(m["name"] == "ic_ir" for m in back["metrics"])
        nw = next(m for m in back["metrics"] if m["name"] == "ic_ir")
        assert nw["usable"] is True
        assert any(w["code"] == "unreliable_se_short_periods" for w in nw["warnings"])

    def test_nan_sparse_ratio_becomes_null(self):
        empty = fx.datasets.make_cs_panel(n_assets=4, n_dates=10).head(0)
        d = inspect_data(empty).to_dict()
        assert d["properties"]["sparse_ratio"] is None
        assert d["properties"]["n_pairs"] == 0

    def test_data_level_warnings_serialised(self):
        info = inspect_data(fx.datasets.make_cs_panel(n_assets=5, n_dates=120))
        d = info.to_dict()
        assert any(w["source"] is None for w in d["warnings"])


class TestReprHtml:
    def test_smoke(self):
        info = inspect_data(fx.datasets.make_cs_panel(n_assets=20, n_dates=80))
        out = info._repr_html_()
        assert "DataInspection" in out
        assert "usable" in out


class TestPublicSurface:
    def test_dataclass_types_exported(self):
        assert fx.DataInspection is DataInspection
        assert fx.DataProperties is DataProperties
        assert fx.MetricApplicability is MetricApplicability
        assert fx.inspect_data is inspect_data
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


class TestMetricApplicabilityGroup:
    def _info(self):
        return inspect_data(fx.datasets.make_cs_panel(n_assets=20, n_dates=80))

    def test_partitions_are_groups_and_lists(self):
        info = self._info()
        for tier in (info.usable, info.degraded, info.unusable):
            assert isinstance(tier, fx.MetricApplicabilityGroup)
            assert isinstance(tier, list)  # every list operation still works

    def test_names_property(self):
        info = self._info()
        assert info.usable.names == [m.name for m in info.usable]

    def test_to_metrics_dict_shape_and_flags_scalar_utilities(self):
        from factrix.metrics._base import MetricBase

        info = self._info()
        md = info.usable.to_metrics_dict()
        assert md, "fixture should have usable metrics"
        assert all(isinstance(v, MetricBase) for v in md.values())
        assert set(md) <= set(info.usable.names)
        assert "breakeven_cost" not in md
        assert "net_spread" not in md
        scalar_blocked = {
            m.name: m.blockers
            for m in info.unusable
            if m.name in {"breakeven_cost", "net_spread"}
        }
        assert set(scalar_blocked) == {"breakeven_cost", "net_spread"}
        assert all(
            "scalar input utility" in blockers[0]
            for blockers in scalar_blocked.values()
        )

    def test_slice_preserves_type(self):
        info = self._info()
        sliced = info.usable[:2]
        assert isinstance(sliced, fx.MetricApplicabilityGroup)
        assert sliced.names == info.usable.names[:2]

    def test_add_preserves_type(self):
        info = self._info()
        combined = info.usable + info.degraded
        assert isinstance(combined, fx.MetricApplicabilityGroup)
        assert combined.names == info.usable.names + info.degraded.names

    def test_to_metrics_dict_feeds_evaluate(self):
        # The discovery bridge: a usable metric round-trips into evaluate().
        raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=80)
        panel = fx.preprocess.compute_forward_return(raw, forward_periods=5)
        md = inspect_data(panel).usable.to_metrics_dict()
        results = fx.evaluate(
            panel,
            metrics=md,
            factor_cols=["factor"],
            forward_periods=5,
            strict=False,
        )
        for name in md:
            assert name in results["factor"].metrics


class TestCrossFactorConsistency:
    def test_single_factor_no_warning(self):
        # Only 1 factor column, should have no cross-factor mismatch warnings
        data = fx.datasets.make_cs_panel(n_assets=10, n_dates=30)
        info = inspect_data(data)
        mismatch_codes = {
            w.code.value for w in info.warnings if "cross_factor" in str(w.code.value)
        }
        assert mismatch_codes == set()

    def test_multi_factor_consistent_no_warning(self):
        # Two factor columns, both individual dense
        raw = fx.datasets.make_cs_panel(n_assets=10, n_dates=30, rng=1)
        raw2 = fx.datasets.make_cs_panel(n_assets=10, n_dates=30, rng=2)
        # Merge them
        data = raw.join(
            raw2.select("date", "asset_id", pl.col("factor").alias("factor2")),
            on=["date", "asset_id"],
        )
        info = inspect_data(data, factor_cols=["factor", "factor2"])
        mismatch_codes = {
            w.code.value for w in info.warnings if "cross_factor" in str(w.code.value)
        }
        assert mismatch_codes == set()

    def test_multi_factor_inconsistent_warnings(self):
        # factor: individual dense
        raw = fx.datasets.make_cs_panel(n_assets=10, n_dates=30, rng=1)

        # factor2: common dense (scope mismatch)
        one_per_date = raw.group_by("date").agg(pl.col("factor").first())
        common = (
            raw.drop("factor")
            .join(one_per_date, on="date")
            .select("date", "asset_id", pl.col("factor").alias("factor2"))
        )

        # factor3: individual sparse (density mismatch)
        sparse_factor = raw.with_columns(
            pl.when(pl.int_range(0, raw.height) % 3 == 0)
            .then(pl.col("factor"))
            .otherwise(0.0)
            .alias("factor3")
        )

        # Merge them
        data = raw.join(common, on=["date", "asset_id"]).join(
            sparse_factor.select("date", "asset_id", "factor3"), on=["date", "asset_id"]
        )

        # Test full mismatch
        info = inspect_data(data, factor_cols=["factor", "factor2", "factor3"])
        mismatch_codes = {
            w.code.value for w in info.warnings if "cross_factor" in str(w.code.value)
        }
        assert "cross_factor_density_mismatch" in mismatch_codes
        assert "cross_factor_scope_mismatch" in mismatch_codes

        # Verify message contents carry inconsistent details
        density_warning = next(
            w for w in info.warnings if w.code.value == "cross_factor_density_mismatch"
        )
        assert "Basis 'factor' (dense)" in density_warning.message
        assert "disagreeing columns: 'factor3' (sparse)" in density_warning.message
        assert "DataInspection.factors[<col>]" in density_warning.message
        assert "separate evaluate batches" in density_warning.message

        scope_warning = next(
            w for w in info.warnings if w.code.value == "cross_factor_scope_mismatch"
        )
        assert "Basis 'factor' (individual)" in scope_warning.message
        assert "disagreeing columns: 'factor2' (common)" in scope_warning.message
        assert "DataInspection.factors[<col>]" in scope_warning.message
        assert "asset-specific and common macro factors" in scope_warning.message

    def test_factor_cols_restricts_scope(self):
        raw = fx.datasets.make_cs_panel(n_assets=10, n_dates=30, rng=1)
        # factor2: common dense
        one_per_date = raw.group_by("date").agg(pl.col("factor").first())
        common = (
            raw.drop("factor")
            .join(one_per_date, on="date")
            .select("date", "asset_id", pl.col("factor").alias("factor2"))
        )
        # factor3: individual dense (consistent with factor)
        raw3 = fx.datasets.make_cs_panel(n_assets=10, n_dates=30, rng=3)

        data = raw.join(common, on=["date", "asset_id"]).join(
            raw3.select("date", "asset_id", pl.col("factor").alias("factor3")),
            on=["date", "asset_id"],
        )

        # If we only inspect ["factor", "factor3"], there should be no mismatch warning
        info = inspect_data(data, factor_cols=["factor", "factor3"])
        mismatch_codes = {
            w.code.value for w in info.warnings if "cross_factor" in str(w.code.value)
        }
        assert mismatch_codes == set()

    def test_auto_detect_excludes_reserved_columns(self):
        raw = fx.datasets.make_cs_panel(n_assets=10, n_dates=30, rng=1)
        data = raw.with_columns(pl.lit(1.0).alias("forward_return")).with_columns(
            pl.col("factor").alias("factor2")
        )

        # If factor_cols is None, it should auto-detect: ['factor', 'factor2'] (no mismatch)
        info = inspect_data(data)
        mismatch_codes = {
            w.code.value for w in info.warnings if "cross_factor" in str(w.code.value)
        }
        assert mismatch_codes == set()


def _common_factor_panel(n_assets: int = 6, n_dates: int = 30) -> pl.DataFrame:
    """Panel whose factor is broadcast: one value per period, shared by every asset."""
    raw = fx.datasets.make_cs_panel(n_assets=n_assets, n_dates=n_dates, rng=7)
    one_per_date = raw.group_by("date").agg(pl.col("factor").first())
    return raw.drop("factor").join(one_per_date, on="date").sort("date", "asset_id")


def _hole_on_first_period(data: pl.DataFrame, value: float | None) -> pl.DataFrame:
    """Replace one asset's factor cell on the first period with ``value``."""
    first_date = data["date"].min()
    first_asset = data["asset_id"].min()
    return data.with_columns(
        pl.when((pl.col("date") == first_date) & (pl.col("asset_id") == first_asset))
        .then(pl.lit(value, dtype=pl.Float64))
        .otherwise(pl.col("factor"))
        .alias("factor")
    )


def _blank_first_period(data: pl.DataFrame) -> pl.DataFrame:
    first_date = data["date"].min()
    return data.with_columns(
        pl.when(pl.col("date") == first_date)
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("factor"))
        .alias("factor")
    )


class TestScopeIgnoresMissingCells:
    """Missing cells must not manufacture cross-sectional variation (#1055)."""

    def test_dense_common_factor_is_common(self):
        info = inspect_data(_common_factor_panel())
        assert info.properties.scope is fx.FactorScope.COMMON

    def test_common_factor_with_one_null_cell_stays_common(self):
        info = inspect_data(_hole_on_first_period(_common_factor_panel(), None))
        assert info.properties.scope is fx.FactorScope.COMMON

    def test_common_factor_with_one_nan_cell_stays_common(self):
        info = inspect_data(_hole_on_first_period(_common_factor_panel(), math.nan))
        assert info.properties.scope is fx.FactorScope.COMMON

    def test_common_factor_with_one_infinite_cell_stays_common(self):
        info = inspect_data(_hole_on_first_period(_common_factor_panel(), math.inf))
        assert info.properties.scope is fx.FactorScope.COMMON

    def test_ragged_common_factor_stays_common(self):
        """An asset with no row at all on a period is not variation either."""
        data = _common_factor_panel()
        first_date = data["date"].min()
        first_asset = data["asset_id"].min()
        ragged = data.filter(
            ~((pl.col("date") == first_date) & (pl.col("asset_id") == first_asset))
        )
        info = inspect_data(ragged)
        assert info.properties.scope is fx.FactorScope.COMMON

    def test_period_with_a_single_finite_cell_stays_common(self):
        """One finite observation cannot contradict a broadcast structure."""
        data = _common_factor_panel()
        first_date = data["date"].min()
        keep = data["asset_id"].min()
        one_finite = data.with_columns(
            pl.when((pl.col("date") == first_date) & (pl.col("asset_id") != keep))
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col("factor"))
            .alias("factor")
        )
        info = inspect_data(one_finite)
        assert info.properties.scope is fx.FactorScope.COMMON

    def test_fully_missing_period_is_ignored(self):
        """A period with no finite cell carries no scope evidence either way."""
        info = inspect_data(_blank_first_period(_common_factor_panel()))
        assert info.properties.scope is fx.FactorScope.COMMON
        assert not [
            w
            for w in info.warnings
            if w.code is fx.WarningCode.FACTOR_SCOPE_UNIDENTIFIABLE
        ]

    def test_all_periods_missing_is_unidentifiable_and_warns(self):
        data = _common_factor_panel().with_columns(
            pl.lit(None, dtype=pl.Float64).alias("factor")
        )
        info = inspect_data(data)
        assert info.properties.scope is fx.FactorScope.INDIVIDUAL
        warns = [
            w
            for w in info.warnings
            if w.code is fx.WarningCode.FACTOR_SCOPE_UNIDENTIFIABLE
        ]
        assert len(warns) == 1
        assert "0 of 30 periods" in warns[0].message
        assert "individual" in warns[0].message

    def test_genuinely_varying_factor_stays_individual_despite_missing_cells(self):
        raw = fx.datasets.make_cs_panel(n_assets=6, n_dates=30, rng=7)
        info = inspect_data(_hole_on_first_period(raw, None))
        assert info.properties.scope is fx.FactorScope.INDIVIDUAL

    def test_scope_reason_reports_the_periods_actually_read(self):
        info = inspect_data(_blank_first_period(_common_factor_panel()))
        assert "29 of 30 periods" in info.properties.scope_reason


def _heterogeneous_panel(n_assets: int = 12, n_dates: int = 120) -> pl.DataFrame:
    """One panel carrying four deliberately different factor columns.

    ``factor`` individual dense; ``common`` broadcast dense; ``sparse``
    individual zero-encoded event; ``ternary`` individual dense {-1, +1} with
    a hole, so the columns differ on scope, density, missingness, cardinality
    and event count at once.
    """
    raw = fx.datasets.make_cs_panel(n_assets=n_assets, n_dates=n_dates, rng=11)
    one_per_period = raw.group_by("date").agg(pl.col("factor").first().alias("common"))
    data = (
        raw.join(one_per_period, on="date")
        .with_columns(
            pl.when(pl.int_range(0, pl.len()) % 4 == 0)
            .then(pl.col("factor"))
            .otherwise(0.0)
            .alias("sparse"),
            pl.when(pl.int_range(0, pl.len()) % 10 == 0)
            .then(pl.lit(None, dtype=pl.Float64))
            .when(pl.col("factor") >= 0)
            .then(1.0)
            .otherwise(-1.0)
            .alias("ternary"),
        )
        .sort("date", "asset_id")
    )
    return data


_HETEROGENEOUS_COLS = ["factor", "common", "sparse", "ternary"]


class TestPerFactorInspection:
    """Multi-factor input gets an auditable per-column result (#1056)."""

    def test_factors_covers_every_inspected_column_in_order(self):
        info = inspect_data(_heterogeneous_panel(), factor_cols=_HETEROGENEOUS_COLS)
        assert list(info.factors) == _HETEROGENEOUS_COLS
        assert all(info.factors[c].factor == c for c in _HETEROGENEOUS_COLS)

    def test_single_factor_path_stays_simple(self):
        data = fx.datasets.make_cs_panel(n_assets=12, n_dates=120, rng=11)
        info = inspect_data(data)
        assert list(info.factors) == ["factor"]
        only = info.factors["factor"]
        assert only.properties == info.properties
        assert only.metrics == info.metrics
        assert only.warnings == info.warnings

    def test_aggregate_still_describes_the_first_column(self):
        info = inspect_data(_heterogeneous_panel(), factor_cols=_HETEROGENEOUS_COLS)
        assert info.properties == info.factors["factor"].properties
        assert info.metrics == info.factors["factor"].metrics

    def test_scope_is_reported_per_factor(self):
        info = inspect_data(_heterogeneous_panel(), factor_cols=_HETEROGENEOUS_COLS)
        assert info.factors["factor"].properties.scope is fx.FactorScope.INDIVIDUAL
        assert info.factors["common"].properties.scope is fx.FactorScope.COMMON

    def test_density_is_reported_per_factor(self):
        info = inspect_data(_heterogeneous_panel(), factor_cols=_HETEROGENEOUS_COLS)
        assert info.factors["factor"].properties.density is fx.FactorDensity.DENSE
        assert info.factors["sparse"].properties.density is fx.FactorDensity.SPARSE
        assert info.factors["sparse"].properties.sparse_ratio >= 0.5

    def test_missingness_is_reported_per_factor(self):
        info = inspect_data(_heterogeneous_panel(), factor_cols=_HETEROGENEOUS_COLS)
        assert (
            info.factors["ternary"].properties.n_pairs
            < info.factors["factor"].properties.n_pairs
        )

    def test_event_count_is_reported_per_factor(self):
        info = inspect_data(_heterogeneous_panel(), factor_cols=_HETEROGENEOUS_COLS)
        assert (
            info.factors["sparse"].properties.n_events
            < info.factors["factor"].properties.n_events
        )

    def test_cardinality_verdict_is_reported_per_factor(self):
        """A low-cardinality column's advisory rides on that column, not the first."""
        info = inspect_data(_heterogeneous_panel(), factor_cols=_HETEROGENEOUS_COLS)
        codes = {w.code for w in info.factors["ternary"].warnings}
        assert fx.WarningCode.LOW_CARDINALITY_DENSE_SIGNAL in codes
        assert fx.WarningCode.LOW_CARDINALITY_DENSE_SIGNAL not in {
            w.code for w in info.factors["factor"].warnings
        }

    def test_metric_verdicts_differ_per_factor(self):
        info = inspect_data(_heterogeneous_panel(), factor_cols=_HETEROGENEOUS_COLS)
        individual = info.factors["factor"]
        common = info.factors["common"]
        assert "ic" in (individual.usable + individual.degraded).names
        assert "ic" in common.unusable.names
        assert "common_beta" in common.usable.names + common.degraded.names
        assert "common_beta" in individual.unusable.names

    def test_per_factor_stage_one_profile_gates_its_own_column(self):
        """A column whose IC stage-one profile is thin blocks `ic` on itself only.

        ``thin`` carries a usable cross-section on 20 periods and a single
        asset on the rest, so ``compute_ic`` would keep 20 periods — below
        ``ic``'s 50-period floor — while the sibling column keeps 118. The
        panel-level ``n_periods`` is identical for both, so only a per-column
        stage-one profile can separate them.
        """
        raw = compute_forward_return(
            fx.datasets.make_cs_panel(n_assets=12, n_dates=120, rng=11),
            forward_periods=1,
        )
        early = raw["date"].unique().sort()[:20]
        lone_asset = raw["asset_id"].unique().sort()[0]
        thin = raw.with_columns(
            pl.when(
                pl.col("date").is_in(early.implode())
                | (pl.col("asset_id") == lone_asset)
            )
            .then(pl.col("factor"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("thin")
        )
        info = inspect_data(thin, factor_cols=["factor", "thin"])

        assert info.factors["thin"].properties.n_periods == (
            info.factors["factor"].properties.n_periods
        )
        assert "ic" in info.factors["thin"].unusable.names
        assert "ic" not in info.factors["factor"].unusable.names
        thin_ic = next(m for m in info.factors["thin"].metrics if m.name == "ic")
        assert "n_periods=20 < min_periods=50" in thin_ic.blockers

    def test_aggregate_warning_names_the_disagreeing_columns(self):
        info = inspect_data(_heterogeneous_panel(), factor_cols=_HETEROGENEOUS_COLS)
        scope_warning = next(
            w
            for w in info.warnings
            if w.code is fx.WarningCode.CROSS_FACTOR_SCOPE_MISMATCH
        )
        assert "'common' (common)" in scope_warning.message
        assert "'factor' (individual)" in scope_warning.message
        assert "DataInspection.factors" in scope_warning.message

        density_warning = next(
            w
            for w in info.warnings
            if w.code is fx.WarningCode.CROSS_FACTOR_DENSITY_MISMATCH
        )
        assert "'sparse' (sparse)" in density_warning.message
        assert "'factor' (dense)" in density_warning.message

    def test_to_dict_carries_every_factor(self):
        info = inspect_data(_heterogeneous_panel(), factor_cols=_HETEROGENEOUS_COLS)
        back = info.to_dict()
        assert list(back["factors"]) == _HETEROGENEOUS_COLS
        assert back["factors"]["common"]["properties"]["scope"] == "common"
        assert back["factors"]["sparse"]["properties"]["density"] == "sparse"
        assert back["factors"]["factor"]["properties"] == back["properties"]

    def test_repr_html_lists_each_factor_with_its_own_axes(self):
        info = inspect_data(_heterogeneous_panel(), factor_cols=_HETEROGENEOUS_COLS)
        html_out = info._repr_html_()
        assert "per-factor" in html_out
        # Whole rows, not loose substrings: the column name must travel with the
        # axes detected for that column.
        assert "<td>common</td><td>common</td><td>dense</td>" in html_out
        assert "<td>sparse</td><td>individual</td><td>sparse</td>" in html_out


def _zero_variance_common_panel(n_assets: int = 20, n_dates: int = 120) -> pl.DataFrame:
    """COMMON-scope panel whose broadcast factor never moves in the time series."""
    raw = fx.datasets.make_cs_panel(n_assets=n_assets, n_dates=n_dates, rng=17)
    return compute_forward_return(
        raw.with_columns(pl.lit(1.0).alias("factor")), forward_periods=5
    )


class TestCommonQuantileFactorVariation:
    """Pre-flight applies the distinct-value gate the metric applies (#1115)."""

    def test_zero_variance_common_factor_is_unusable(self):
        info = inspect_data(_zero_variance_common_panel())
        verdict = _by_name(info, "common_quantile_spread")
        assert not verdict.usable
        blocker = next(
            b for b in verdict.blockers if "insufficient_factor_variation" in b
        )
        # One token, not two `in` checks: the count and the bound it is
        # compared against have to travel together for the message to be
        # reproducible by a reader.
        assert "n_distinct=1 < n_groups * 2 = 10" in blocker

    def test_verdict_matches_the_run_on_a_zero_variance_common_factor(self):
        panel = _zero_variance_common_panel()
        assert not _by_name(inspect_data(panel), "common_quantile_spread").usable
        out = fx.metrics.common_quantile.common_quantile_spread(panel)
        assert out.metadata["reason"] == "insufficient_factor_variation"

    def test_a_varying_common_factor_stays_usable(self):
        raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=120, rng=17)
        one_per_date = raw.group_by("date").agg(pl.col("factor").first())
        panel = compute_forward_return(
            raw.drop("factor").join(one_per_date, on="date").sort("date", "asset_id"),
            forward_periods=5,
        )
        verdict = _by_name(inspect_data(panel), "common_quantile_spread")
        assert verdict.usable, verdict.blockers

    def test_period_floor_uses_the_aggregated_series(self):
        raw = fx.datasets.make_cs_panel(n_assets=20, n_dates=20, rng=17)
        one_per_date = raw.group_by("date").agg(pl.col("factor").first())
        panel = compute_forward_return(
            raw.drop("factor").join(one_per_date, on="date").sort("date", "asset_id"),
            forward_periods=1,
        )
        usable_dates = panel["date"].unique().sort().head(2)
        panel = panel.with_columns(
            pl.when(pl.col("date").is_in(usable_dates.implode()))
            .then(pl.col("forward_return"))
            .otherwise(None)
            .alias("forward_return")
        )

        verdict = _by_name(inspect_data(panel), "common_quantile_spread")
        assert any(
            "n_periods=2 < min_periods=3" in blocker for blocker in verdict.blockers
        )
        out = fx.metrics.common_quantile.common_quantile_spread(panel)
        assert out.metadata["reason"] == "insufficient_portfolio_periods"


def _one_sided_sign_panel(
    n_assets: int = 20, n_dates: int = 120, *, side: float = 1.0
) -> pl.DataFrame:
    """Panel whose factor has one non-zero sign, so sign(factor) never varies."""
    raw = fx.datasets.make_cs_panel(n_assets=n_assets, n_dates=n_dates, rng=17)
    return compute_forward_return(
        raw.with_columns(((pl.col("factor").abs() + 1.0) * side).alias("factor")),
        forward_periods=5,
    )


class TestDirectionalHitRateOneSidedSignal:
    """Pre-flight reports the outcome the run produces, not a short-circuit (#1116)."""

    @pytest.mark.parametrize("side", [1.0, -1.0])
    def test_one_sided_signal_is_usable_with_an_advisory(self, side: float):
        verdict = _by_name(
            inspect_data(_one_sided_sign_panel(side=side)), "directional_hit_rate"
        )
        assert verdict.usable, verdict.blockers
        advisory = next(
            w for w in verdict.warnings if w.code is fx.WarningCode.DEGENERATE_VARIANCE
        )
        assert "one-sided directional signal" in advisory.message
        assert "sole predicted sign" in advisory.message
        assert "unconditional positive-return rate" not in advisory.message

    @pytest.mark.parametrize("side", [1.0, -1.0])
    def test_the_run_returns_a_hit_rate_with_that_warning(self, side: float):
        panel = _one_sided_sign_panel(side=side)
        assert _by_name(inspect_data(panel), "directional_hit_rate").usable
        from factrix.metrics.directional_hit_rate import directional_hit_rate

        out = directional_hit_rate(panel, overlap_periods=5)
        assert out.metadata.get("reason") is None
        assert "degenerate_variance" in out.warning_codes
        assert out.value == pytest.approx(out.metadata["p_correct"])

    @pytest.mark.parametrize("nonfinite", [math.nan, -math.inf])
    def test_nonfinite_cells_do_not_manufacture_a_second_sign(self, nonfinite: float):
        panel = _one_sided_sign_panel().with_columns(
            pl.when(pl.int_range(0, pl.len()) == 0)
            .then(pl.lit(nonfinite))
            .otherwise(pl.col("factor"))
            .alias("factor")
        )
        verdict = _by_name(inspect_data(panel), "directional_hit_rate")
        assert any(
            warning.code is fx.WarningCode.DEGENERATE_VARIANCE
            for warning in verdict.warnings
        )

    def test_a_two_sided_signal_carries_no_advisory(self):
        panel = compute_forward_return(
            fx.datasets.make_cs_panel(n_assets=20, n_dates=120, rng=17),
            forward_periods=5,
        )
        verdict = _by_name(inspect_data(panel), "directional_hit_rate")
        assert verdict.usable, verdict.blockers
        assert not [
            w for w in verdict.warnings if w.code is fx.WarningCode.DEGENERATE_VARIANCE
        ]
