"""Tests for ``factrix.list_estimators`` (#255).

No-arg overview API — returns every registered estimator regardless of
cell context, matching the ``list_metrics`` pattern established in #498.
"""

from __future__ import annotations

from factrix import list_estimators

ALL_NAMES = [
    "BlockBootstrap",
    "DriscollKraay",
    "GMM",
    "HansenHodrick",
    "NeweyWest",
    "WaldNWCluster",
    "WaldTwoWayCluster",
]


def test_text_format_returns_all_names() -> None:
    assert list_estimators() == ALL_NAMES


def test_json_format_includes_metadata_keys() -> None:
    rows = list_estimators(format="json")
    by_name = {row["name"]: row for row in rows}
    assert "Bartlett" in by_name["NeweyWest"]["description"]
    assert by_name["NeweyWest"]["import_path"] == "factrix.stats.NeweyWest"
    assert "Hansen-Hodrick" in by_name["HansenHodrick"]["description"]
    assert by_name["HansenHodrick"]["import_path"] == "factrix.stats.HansenHodrick"


def test_with_import_returns_two_column_lines() -> None:
    rows = list_estimators(with_import=True)
    assert rows == [
        "BlockBootstrap    → factrix.stats.BlockBootstrap",
        "DriscollKraay     → factrix.stats.DriscollKraay",
        "GMM               → factrix.stats.GMM",
        "HansenHodrick     → factrix.stats.HansenHodrick",
        "NeweyWest         → factrix.stats.NeweyWest",
        "WaldNWCluster     → factrix.stats.WaldNWCluster",
        "WaldTwoWayCluster → factrix.stats.WaldTwoWayCluster",
    ]


def test_with_import_ignored_under_json() -> None:
    json_rows = list_estimators(format="json", with_import=True)
    paths = {row["name"]: row["import_path"] for row in json_rows}
    assert paths["NeweyWest"] == "factrix.stats.NeweyWest"
    assert paths["HansenHodrick"] == "factrix.stats.HansenHodrick"
