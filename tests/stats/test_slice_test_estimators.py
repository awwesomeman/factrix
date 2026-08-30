"""Tests for the selection-only Estimator handles: WaldNWCluster / WaldTwoWayCluster / DriscollKraay."""

from __future__ import annotations

from factrix.stats import (
    DriscollKraay,
    WaldNWCluster,
    WaldTwoWayCluster,
)

_ALL_ESTIMATORS = (
    WaldNWCluster(),
    WaldTwoWayCluster(),
    DriscollKraay(),
)


class TestEstimatorProtocol:
    def test_names_distinct(self):
        names = {e.name for e in _ALL_ESTIMATORS}
        assert names == {
            "WaldNWCluster",
            "WaldTwoWayCluster",
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
