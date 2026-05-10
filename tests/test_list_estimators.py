"""Tests for ``factrix.list_estimators`` (#170, #184).

Mirrors the `list_metrics` listing pattern: text-format names, JSON
dict rows, `with_import` two-column form, and `IncompatibleAxisError`
on the empty cell.
"""

from __future__ import annotations

import pytest
from factrix import FactorScope, Signal, list_estimators
from factrix._errors import IncompatibleAxisError


@pytest.mark.parametrize(
    ("scope", "signal", "expected"),
    [
        # HansenHodrick applies only to (INDIVIDUAL, CONTINUOUS).
        (FactorScope.INDIVIDUAL, Signal.CONTINUOUS, ["HansenHodrick", "NeweyWest"]),
        (FactorScope.INDIVIDUAL, Signal.SPARSE, ["NeweyWest"]),
        (FactorScope.COMMON, Signal.CONTINUOUS, ["NeweyWest"]),
        (FactorScope.COMMON, Signal.SPARSE, ["NeweyWest"]),
    ],
)
def test_text_format_returns_name_list(
    scope: FactorScope, signal: Signal, expected: list[str]
) -> None:
    assert list_estimators(scope, signal) == expected


def test_json_format_includes_metadata_keys() -> None:
    rows = list_estimators(FactorScope.INDIVIDUAL, Signal.CONTINUOUS, format="json")
    by_name = {row["name"]: row for row in rows}
    assert "Bartlett" in by_name["NeweyWest"]["description"]
    assert by_name["NeweyWest"]["import_path"] == "factrix.stats.NeweyWest"
    assert "Hansen-Hodrick" in by_name["HansenHodrick"]["description"]
    assert by_name["HansenHodrick"]["import_path"] == "factrix.stats.HansenHodrick"


def test_with_import_returns_two_column_lines() -> None:
    rows = list_estimators(FactorScope.INDIVIDUAL, Signal.CONTINUOUS, with_import=True)
    assert rows == [
        "HansenHodrick → factrix.stats.HansenHodrick",
        "NeweyWest     → factrix.stats.NeweyWest",
    ]


def test_with_import_ignored_under_json() -> None:
    json_rows = list_estimators(
        FactorScope.INDIVIDUAL,
        Signal.CONTINUOUS,
        format="json",
        with_import=True,
    )
    paths = {row["name"]: row["import_path"] for row in json_rows}
    assert paths["NeweyWest"] == "factrix.stats.NeweyWest"
    assert paths["HansenHodrick"] == "factrix.stats.HansenHodrick"


def test_empty_match_raises_incompatible_axis_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force an empty registry so we exercise the defensive raise path
    # even with NeweyWest's universal applicability.
    import factrix.stats as stats_pkg

    monkeypatch.setattr(stats_pkg, "_ESTIMATOR_REGISTRY", ())
    with pytest.raises(IncompatibleAxisError):
        list_estimators(FactorScope.INDIVIDUAL, Signal.CONTINUOUS)
