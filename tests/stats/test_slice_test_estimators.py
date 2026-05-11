"""Tests for slice-test Estimators (#153): WaldNWCluster / WaldTwoWayCluster / BlockBootstrap."""

from __future__ import annotations

import pytest
from factrix._axis import FactorScope, Metric, Signal
from factrix._codes import StatCode
from factrix.stats import (
    _ESTIMATOR_REGISTRY,
    BlockBootstrap,
    Estimator,
    WaldNWCluster,
    WaldTwoWayCluster,
)


class TestEstimatorProtocol:
    def test_all_three_satisfy_protocol(self):
        for est in (WaldNWCluster(), WaldTwoWayCluster(), BlockBootstrap()):
            assert isinstance(est, Estimator)

    def test_names_distinct(self):
        names = {e.name for e in _ESTIMATOR_REGISTRY}
        assert names == {
            "NeweyWest",
            "HansenHodrick",
            "WaldNWCluster",
            "WaldTwoWayCluster",
            "BlockBootstrap",
        }


class TestWaldNWCluster:
    def test_emits_p_wald_nwcl(self):
        est = WaldNWCluster()
        code = est.emits_for(FactorScope.INDIVIDUAL, Signal.CONTINUOUS, Metric.IC)
        assert code is StatCode.P_WALD_NWCL

    def test_applicable_to_individual_continuous(self):
        est = WaldNWCluster()
        assert est.applicable_to(FactorScope.INDIVIDUAL, Signal.CONTINUOUS)

    def test_not_applicable_to_common(self):
        est = WaldNWCluster()
        assert not est.applicable_to(FactorScope.COMMON, Signal.CONTINUOUS)

    def test_not_applicable_to_sparse(self):
        est = WaldNWCluster()
        assert not est.applicable_to(FactorScope.INDIVIDUAL, Signal.SPARSE)

    def test_description_mentions_cluster_and_nw(self):
        d = WaldNWCluster().description.lower()
        assert "cluster" in d
        assert "nw" in d or "newey" in d


class TestWaldTwoWayCluster:
    def test_emits_p_wald_twoway(self):
        est = WaldTwoWayCluster()
        code = est.emits_for(FactorScope.INDIVIDUAL, Signal.CONTINUOUS, Metric.IC)
        assert code is StatCode.P_WALD_TWOWAY

    def test_applicable_to_individual_continuous(self):
        est = WaldTwoWayCluster()
        assert est.applicable_to(FactorScope.INDIVIDUAL, Signal.CONTINUOUS)

    def test_description_mentions_two_way(self):
        d = WaldTwoWayCluster().description.lower()
        assert "two-way" in d or "double" in d or "cgm" in d


class TestBlockBootstrap:
    def test_default_ctor(self):
        est = BlockBootstrap()
        assert est.block_length == "auto"
        assert est.n_resamples == 999
        assert est.scheme == "stationary"
        assert est.rng_seed is None

    def test_emits_p_boot(self):
        est = BlockBootstrap()
        assert (
            est.emits_for(FactorScope.INDIVIDUAL, Signal.CONTINUOUS, Metric.IC)
            is StatCode.P_BOOT
        )

    def test_explicit_config_stored(self):
        est = BlockBootstrap(
            block_length=20,
            n_resamples=499,
            scheme="fixed",
            rng_seed=42,
        )
        assert est.block_length == 20
        assert est.n_resamples == 499
        assert est.scheme == "fixed"
        assert est.rng_seed == 42

    def test_description_reflects_config(self):
        est = BlockBootstrap(block_length=10, scheme="fixed", n_resamples=199)
        d = est.description
        assert "fixed" in d
        assert "L=10" in d
        assert "B=199" in d

    def test_rejects_bad_block_length(self):
        with pytest.raises(ValueError, match="block_length must be"):
            BlockBootstrap(block_length=0)

    def test_rejects_bad_n_resamples(self):
        with pytest.raises(ValueError, match="n_resamples must be >= 1"):
            BlockBootstrap(n_resamples=0)

    def test_rejects_bad_scheme(self):
        with pytest.raises(ValueError, match="scheme must be"):
            BlockBootstrap(scheme="rolling")  # type: ignore[arg-type]

    def test_two_instances_distinct(self):
        # Same class, different config — caller can pass an explicitly-
        # constructed instance to override the default in the registry.
        a = BlockBootstrap(scheme="stationary")
        b = BlockBootstrap(scheme="fixed")
        assert a.scheme != b.scheme
        # Both still emit the same StatCode (scheme is metadata, not
        # a separate StatCode key).
        assert a.emits_for(
            FactorScope.INDIVIDUAL, Signal.CONTINUOUS, Metric.IC
        ) == b.emits_for(FactorScope.INDIVIDUAL, Signal.CONTINUOUS, Metric.IC)


class TestRegistryIntegration:
    def test_layer_b_estimators_in_registry(self):
        types_in_registry = {type(e).__name__ for e in _ESTIMATOR_REGISTRY}
        assert {
            "WaldNWCluster",
            "WaldTwoWayCluster",
            "BlockBootstrap",
        } <= types_in_registry

    def test_existing_estimators_still_in_registry(self):
        # Regression guard: NW/HH must remain after the slice-test Estimator append.
        types_in_registry = {type(e).__name__ for e in _ESTIMATOR_REGISTRY}
        assert "NeweyWest" in types_in_registry
        assert "HansenHodrick" in types_in_registry

    def test_list_estimators_surfaces_slice_test_estimators(self):
        from factrix import list_estimators

        names = list_estimators(FactorScope.INDIVIDUAL, Signal.CONTINUOUS)
        assert "WaldNWCluster" in names
        assert "WaldTwoWayCluster" in names
        assert "BlockBootstrap" in names

    def test_list_estimators_excludes_slice_test_estimators_for_common(self):
        from factrix import list_estimators

        names = list_estimators(FactorScope.COMMON, Signal.CONTINUOUS)
        # NW applies universally; slice-test Estimators + HH restricted to (INDIVIDUAL, CONTINUOUS).
        assert "NeweyWest" in names
        assert "WaldNWCluster" not in names
        assert "WaldTwoWayCluster" not in names
        assert "BlockBootstrap" not in names
        assert "HansenHodrick" not in names
