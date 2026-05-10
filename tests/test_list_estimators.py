"""Tests for ``factrix.list_estimators`` (#170).

Mirrors the `list_metrics` listing pattern: text-format names, JSON
dict rows, `with_import` two-column form, and `IncompatibleAxisError`
on the empty cell. v0.11 ships only ``NeweyWest`` so every cell that
emits a primary p-value resolves to that single entry.
"""

from __future__ import annotations

import pytest
from factrix import FactorScope, Signal, list_estimators
from factrix._errors import IncompatibleAxisError


@pytest.mark.parametrize(
    ("scope", "signal"),
    [
        (FactorScope.INDIVIDUAL, Signal.CONTINUOUS),
        (FactorScope.INDIVIDUAL, Signal.SPARSE),
        (FactorScope.COMMON, Signal.CONTINUOUS),
        (FactorScope.COMMON, Signal.SPARSE),
    ],
)
def test_text_format_returns_name_list(scope: FactorScope, signal: Signal) -> None:
    rows = list_estimators(scope, signal)
    assert rows == ["NeweyWest"]


def test_json_format_includes_metadata_keys() -> None:
    [row] = list_estimators(FactorScope.INDIVIDUAL, Signal.CONTINUOUS, format="json")
    assert row["name"] == "NeweyWest"
    assert "Bartlett" in row["description"]
    assert row["import_path"] == "factrix.stats.NeweyWest"


def test_with_import_returns_two_column_lines() -> None:
    rows = list_estimators(FactorScope.INDIVIDUAL, Signal.CONTINUOUS, with_import=True)
    assert rows == ["NeweyWest → factrix.stats.NeweyWest"]


def test_with_import_ignored_under_json() -> None:
    json_rows = list_estimators(
        FactorScope.INDIVIDUAL,
        Signal.CONTINUOUS,
        format="json",
        with_import=True,
    )
    assert isinstance(json_rows, list)
    assert json_rows[0]["import_path"] == "factrix.stats.NeweyWest"


def test_empty_match_raises_incompatible_axis_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force an empty registry so we exercise the defensive raise path
    # even with NeweyWest's universal applicability in v0.11.
    import factrix.stats as stats_pkg

    monkeypatch.setattr(stats_pkg, "_ESTIMATOR_REGISTRY", ())
    with pytest.raises(IncompatibleAxisError):
        list_estimators(FactorScope.INDIVIDUAL, Signal.CONTINUOUS)
