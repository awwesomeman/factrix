"""Tests for slice-test Estimators: WaldNWCluster / WaldTwoWayCluster / BlockBootstrap."""

from __future__ import annotations

import pytest
from factrix.stats import (
    BlockBootstrap,
    DriscollKraay,
    WaldNWCluster,
    WaldTwoWayCluster,
)

_ALL_ESTIMATORS = (
    WaldNWCluster(),
    WaldTwoWayCluster(),
    BlockBootstrap(),
    DriscollKraay(),
)


class TestEstimatorProtocol:
    def test_names_distinct(self):
        names = {e.name for e in _ALL_ESTIMATORS}
        assert names == {
            "WaldNWCluster",
            "WaldTwoWayCluster",
            "BlockBootstrap",
            "DriscollKraay",
        }


class TestWaldNWCluster:
    def test_description_mentions_cluster_and_nw(self):
        d = WaldNWCluster().description.lower()
        assert "cluster" in d
        assert "nw" in d or "newey" in d


class TestWaldTwoWayCluster:
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
        # Same class, different config — caller passes an explicitly-
        # constructed instance to make the inference choice explicit.
        a = BlockBootstrap(scheme="stationary")
        b = BlockBootstrap(scheme="fixed")
        assert a.scheme != b.scheme
