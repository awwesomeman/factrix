"""Tests for ``MetricSpec.requires`` / ``batchable`` / ``visibility``.

The fields are spec-level contract; consumers read them via
:func:`spec_by_name` / :func:`public_specs`. These tests pin the
spec-side invariants: default values, the visibility / batchable
flags on known metrics, and that every ``requires`` value is a
callable.
"""

from __future__ import annotations

import pytest
from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    FactorScope,
    SEMethod,
    SpecRole,
    TestMethod,
    Tier,
)
from factrix._inspect import DataProperties
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


class TestSampleThresholdHelpers:
    def test_verdict_and_per_axis_verdict(self) -> None:
        st = SampleThreshold(
            min_periods=10,
            warn_periods=30,
            min_assets=5,
            warn_assets=15,
            min_pairs=20,
            warn_pairs=50,
        )

        props_clean = DataProperties(
            scope=FactorScope.INDIVIDUAL,
            scope_reason="",
            density=FactorDensity.DENSE,
            density_reason="",
            structure=DataStructure.PANEL,
            structure_reason="",
            n_periods=30,
            n_assets=15,
            n_pairs=50,
            sparse_ratio=0.0,
        )
        assert st.per_axis_verdict(props_clean) == {
            "periods": Tier.CLEAN,
            "assets": Tier.CLEAN,
            "pairs": Tier.CLEAN,
        }
        assert st.verdict(props_clean) is Tier.CLEAN

        props_degraded = DataProperties(
            scope=FactorScope.INDIVIDUAL,
            scope_reason="",
            density=FactorDensity.DENSE,
            density_reason="",
            structure=DataStructure.PANEL,
            structure_reason="",
            n_periods=20,
            n_assets=15,
            n_pairs=50,
            sparse_ratio=0.0,
        )
        assert st.per_axis_verdict(props_degraded) == {
            "periods": Tier.DEGRADED,
            "assets": Tier.CLEAN,
            "pairs": Tier.CLEAN,
        }
        assert st.verdict(props_degraded) is Tier.DEGRADED

        props_unusable = DataProperties(
            scope=FactorScope.INDIVIDUAL,
            scope_reason="",
            density=FactorDensity.DENSE,
            density_reason="",
            structure=DataStructure.PANEL,
            structure_reason="",
            n_periods=20,
            n_assets=4,
            n_pairs=50,
            sparse_ratio=0.0,
        )
        assert st.per_axis_verdict(props_unusable) == {
            "periods": Tier.DEGRADED,
            "assets": Tier.UNUSABLE,
            "pairs": Tier.CLEAN,
        }
        assert st.verdict(props_unusable) is Tier.UNUSABLE

        st_empty = SampleThreshold()
        assert st_empty.per_axis_verdict(props_unusable) == {
            "periods": Tier.CLEAN,
            "assets": Tier.CLEAN,
            "pairs": Tier.CLEAN,
        }
        assert st_empty.verdict(props_unusable) is Tier.CLEAN


class TestDefaults:
    def test_requires_defaults_to_empty_dict(self) -> None:
        spec = MetricSpec(
            name="probe",
            cell=cell(FactorScope.INDIVIDUAL, FactorDensity.DENSE),
            aggregation=Aggregation.CS_THEN_TS,
            test_method=TestMethod.T,
            se_method=SEMethod.HAC,
        )
        assert spec.requires == {}

    def test_batchable_defaults_false(self) -> None:
        spec = MetricSpec(
            name="probe",
            cell=cell(FactorScope.INDIVIDUAL, FactorDensity.DENSE),
            aggregation=Aggregation.CS_THEN_TS,
            test_method=TestMethod.T,
            se_method=SEMethod.HAC,
        )
        assert spec.batchable is False

    def test_visibility_defaults_public(self) -> None:
        spec = MetricSpec(
            name="probe",
            cell=cell(FactorScope.INDIVIDUAL, FactorDensity.DENSE),
            aggregation=Aggregation.CS_THEN_TS,
            test_method=TestMethod.T,
            se_method=SEMethod.HAC,
        )
        assert spec.role is SpecRole.METRIC


class TestVisibility:
    def test_internal_specs_excluded_from_public(self) -> None:
        all_pipeline = {
            spec.name for _, spec in _all_specs() if spec.role is SpecRole.PIPELINE
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
            "compute_spread_series",
            "compute_group_returns",
        )
        for name in stage1_producers:
            assert specs[name].role is SpecRole.PIPELINE, name


class TestBatchable:
    def test_known_batchable_specs(self) -> None:
        specs = spec_by_name()
        for name in (
            "compute_ic",
            "compute_fm_betas",
            "compute_ts_betas",
            "quantile_spread",
            "monotonicity",
        ):
            assert specs[name].batchable is True, name

    def test_non_batchable_specs(self) -> None:
        specs = spec_by_name()
        for name in ("top_concentration", "turnover"):
            assert specs[name].batchable is False, name


class TestRequires:
    def test_ic_consumers_require_compute_ic(self) -> None:
        from factrix.metrics._primitives import compute_ic

        specs = spec_by_name()
        for name in ("ic", "ic_ir"):
            assert specs[name].requires == {"ic_df": compute_ic}, name

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


def test_pipeline_naming_lint():
    """Lint: `compute_*` prefix <-> role=PIPELINE."""
    from factrix._axis import SpecRole
    from factrix.metrics._registry import REGISTRY

    for name, cls in REGISTRY.items():
        spec = cls.spec()
        if name.startswith("compute_"):
            assert spec.role is SpecRole.PIPELINE, (
                f"FX001: {name} starts with 'compute_' but is not PIPELINE"
            )
        else:
            assert spec.role is not SpecRole.PIPELINE, (
                f"FX002: {name} is PIPELINE but does not start with 'compute_'"
            )
