"""Tests for ``MetricSpec.requires`` / ``batchable`` / ``visibility``.

The fields are spec-level contract; consumers read them via
:func:`spec_by_name` / :func:`public_specs`. These tests pin the
spec-side invariants: default values, the visibility / batchable
flags on known metrics, and that every ``requires`` value is a
callable.
"""

from __future__ import annotations

import pytest
from factrix._axis import Aggregation, FactorDensity, FactorScope, SEMethod, SpecRole, TestMethod
from factrix._metric_index import (
    MetricSpec,
    SampleThreshold,
    _all_specs,
    cell,
    public_specs,
    spec_by_name,
)


class TestSampleThresholdInvariant:
    def test_min_above_warn_rejected_per_axis(self) -> None:
        for axis in ("periods", "assets", "pairs"):
            with pytest.raises(ValueError, match=f"{axis}: min"):
                SampleThreshold(**{f"min_{axis}": 30, f"warn_{axis}": 10})

    def test_min_equal_warn_allowed(self) -> None:
        assert SampleThreshold(min_periods=20, warn_periods=20).min_periods == 20

    def test_min_below_warn_allowed(self) -> None:
        st = SampleThreshold(min_periods=20, warn_periods=30)
        assert (st.min_periods, st.warn_periods) == (20, 30)

    def test_one_sided_floor_allowed(self) -> None:
        assert SampleThreshold(min_pairs=100).warn_pairs is None
        assert SampleThreshold(warn_assets=5).min_assets is None


class TestDefaults:
    def test_requires_defaults_to_empty_dict(self) -> None:
        spec = MetricSpec(
            name="probe",
            cell=cell(FactorScope.INDIVIDUAL, FactorDensity.DENSE),
            aggregation=Aggregation.CS_THEN_TS,
            test_method=TestMethod.T, se_method=SEMethod.HAC,
        )
        assert spec.requires == {}

    def test_batchable_defaults_false(self) -> None:
        spec = MetricSpec(
            name="probe",
            cell=cell(FactorScope.INDIVIDUAL, FactorDensity.DENSE),
            aggregation=Aggregation.CS_THEN_TS,
            test_method=TestMethod.T, se_method=SEMethod.HAC,
        )
        assert spec.batchable is False

    def test_visibility_defaults_public(self) -> None:
        spec = MetricSpec(
            name="probe",
            cell=cell(FactorScope.INDIVIDUAL, FactorDensity.DENSE),
            aggregation=Aggregation.CS_THEN_TS,
            test_method=TestMethod.T, se_method=SEMethod.HAC,
        )
        assert spec.role is SpecRole.METRIC


class TestVisibility:
    def test_internal_specs_excluded_from_public(self) -> None:
        all_pipeline = {
            spec.name
            for _, spec in _all_specs()
            if spec.role is SpecRole.PIPELINE
        }
        public_names = {spec.name for _, spec in public_specs()}
        assert all_pipeline.isdisjoint(public_names)

    def test_known_stage1_producers_are_internal(self) -> None:
        specs = spec_by_name()
        stage1_producers = (
            "compute_ic",
            "compute_caar",
            "compute_fm_betas",
            "compute_mfe_mae",
            "compute_event_returns",
            "compute_ts_betas",
            "ts_beta_single_asset_fallback",
            "compute_spread_series",
            "compute_group_returns",
        )
        for name in stage1_producers:
            assert specs[name].role is SpecRole.PIPELINE, name


class TestBatchable:
    def test_known_batchable_specs(self) -> None:
        specs = spec_by_name()
        for name in ("compute_ic", "quantile_spread", "monotonicity"):
            assert specs[name].batchable is True, name

    def test_non_batchable_specs(self) -> None:
        specs = spec_by_name()
        for name in ("top_concentration", "turnover", "compute_fm_betas"):
            assert specs[name].batchable is False, name


class TestRequires:
    def test_ic_consumers_require_compute_ic(self) -> None:
        import factrix.metrics as metrics_pkg

        specs = spec_by_name()
        for name in ("ic", "ic_newey_west", "ic_ir"):
            assert specs[name].requires == {"ic_df": metrics_pkg.compute_ic}, name

    def test_caar_consumer_requires_compute_caar(self) -> None:
        specs = spec_by_name()
        assert specs["caar"].requires["caar_df"].__name__ == "compute_caar"

    def test_fama_macbeth_consumers_require_compute_fm_betas(self) -> None:
        specs = spec_by_name()
        for name in ("fm_beta", "beta_sign_consistency"):
            assert specs[name].requires["beta_df"].__name__ == "compute_fm_betas", name

    def test_ts_beta_consumers_require_compute_ts_betas(self) -> None:
        specs = spec_by_name()
        for name in ("ts_beta", "mean_r_squared", "ts_beta_sign_consistency"):
            assert specs[name].requires["ts_betas_df"].__name__ == "compute_ts_betas", (
                name
            )

    def test_mfe_mae_consumer_requires_compute_mfe_mae(self) -> None:
        specs = spec_by_name()
        assert (
            specs["mfe_mae_summary"].requires["mfe_mae_df"].__name__
            == "compute_mfe_mae"
        )

    def test_requires_values_are_callables(self) -> None:
        for _, spec in _all_specs():
            for key, value in spec.requires.items():
                assert callable(value), f"{spec.name}.requires[{key!r}] is not callable"
