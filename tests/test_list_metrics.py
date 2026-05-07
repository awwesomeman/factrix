"""Tests for ``factrix.list_metrics`` and the shared ``_metric_index`` parser.

Asserts the runtime API agrees with the hand-curated applicability table
in ``docs/reference/metric-applicability.md`` — the test catches drift
in either direction (a new metric module that forgets to update the
doc, or vice versa).
"""

from __future__ import annotations

import json
import pathlib
import re
from unittest.mock import patch

import factrix as fl
import pytest
from factrix._axis import FactorScope, Signal
from factrix._errors import IncompatibleAxisError
from factrix._metric_index import _STAGE1_HELPERS, MetricRow, user_facing_rows

_APPLICABILITY_DOC = pathlib.Path("docs/reference/metric-applicability.md")


def _parse_applicability_doc() -> dict[tuple[FactorScope, Signal], set[str]]:
    """Build per-cell expected name sets from the applicability matrix."""
    expected: dict[tuple[FactorScope, Signal], set[str]] = {
        (FactorScope.INDIVIDUAL, Signal.CONTINUOUS): set(),
        (FactorScope.COMMON, Signal.CONTINUOUS): set(),
        (FactorScope.INDIVIDUAL, Signal.SPARSE): set(),
        (FactorScope.COMMON, Signal.SPARSE): set(),
    }
    text = _APPLICABILITY_DOC.read_text(encoding="utf-8")
    row_re = re.compile(r"^\|\s*\[`([^`]+)`\][^|]*\|\s*([^|]+?)\s*\|", re.MULTILINE)
    for match in row_re.finditer(text):
        name, cell = match.group(1), match.group(2)
        # Drop the leading symbol on the metric name when present
        # (table syntax already strips backticks).
        targets: list[tuple[FactorScope, Signal]] = []
        if cell.startswith("Individual × Continuous") or cell.startswith(
            "Spread-series consumer"
        ):
            targets = [(FactorScope.INDIVIDUAL, Signal.CONTINUOUS)]
        elif cell.startswith("Common × Continuous"):
            targets = [(FactorScope.COMMON, Signal.CONTINUOUS)]
        elif cell.startswith("Individual × Sparse"):
            # Matrix-row tags use ``(*, SPARSE, *, PANEL)``: wildcard scope.
            targets = [
                (FactorScope.INDIVIDUAL, Signal.SPARSE),
                (FactorScope.COMMON, Signal.SPARSE),
            ]
        elif cell.startswith("Series-tools"):
            # ``(*, CONTINUOUS, *, TIMESERIES)``: wildcard scope.
            targets = [
                (FactorScope.INDIVIDUAL, Signal.CONTINUOUS),
                (FactorScope.COMMON, Signal.CONTINUOUS),
            ]
        for t in targets:
            expected[t].add(name)
    return expected


# ---------------------------------------------------------------------------
# applicability-doc parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scope", "signal"),
    [
        (FactorScope.INDIVIDUAL, Signal.CONTINUOUS),
        (FactorScope.COMMON, Signal.CONTINUOUS),
        (FactorScope.INDIVIDUAL, Signal.SPARSE),
        (FactorScope.COMMON, Signal.SPARSE),
    ],
)
def test_list_metrics_matches_applicability_doc(
    scope: FactorScope, signal: Signal
) -> None:
    expected = _parse_applicability_doc()[(scope, signal)]
    assert expected, "applicability doc parser found no rows for this cell"
    actual = set(fl.list_metrics(scope, signal))
    assert actual == expected, (
        f"({scope.value}, {signal.value}) drift\n"
        f"  only in list_metrics: {sorted(actual - expected)}\n"
        f"  only in doc table:    {sorted(expected - actual)}"
    )


# ---------------------------------------------------------------------------
# stage-1 helper exclusion
# ---------------------------------------------------------------------------


def test_stage1_helpers_excluded_from_user_facing_rows() -> None:
    names = {row.name for row in user_facing_rows()}
    assert _STAGE1_HELPERS.isdisjoint(names)


def test_compute_rolling_mean_beta_remains_user_facing() -> None:
    # Sanity: it starts with ``compute_`` but is a real metric per the doc;
    # exclusion must be by name set, not prefix.
    names = {row.name for row in user_facing_rows()}
    assert "compute_rolling_mean_beta" in names


# ---------------------------------------------------------------------------
# sort stability + format
# ---------------------------------------------------------------------------


def test_text_output_is_sorted_by_module_then_name() -> None:
    out = fl.list_metrics(FactorScope.INDIVIDUAL, Signal.CONTINUOUS)
    json_rows = fl.list_metrics(
        FactorScope.INDIVIDUAL, Signal.CONTINUOUS, format="json"
    )
    expected_order = [r["name"] for r in json_rows]
    assert out == expected_order
    keys = [(r["module"], r["name"]) for r in json_rows]
    assert keys == sorted(keys)


def test_json_format_round_trips() -> None:
    rows = fl.list_metrics(FactorScope.INDIVIDUAL, Signal.CONTINUOUS, format="json")
    text = json.dumps(rows)
    decoded = json.loads(text)
    assert decoded == rows
    sample = rows[0]
    assert set(sample) == {
        "name",
        "module",
        "cell",
        "agg_order",
        "inference_se",
        "import_path",
        "input_kind",
        "docs_anchor",
        "emitted_name",
    }


def test_json_carries_import_path_and_input_kind() -> None:
    rows = fl.list_metrics(FactorScope.INDIVIDUAL, Signal.CONTINUOUS, format="json")
    by_name = {r["name"]: r for r in rows}

    # ``ic`` is the canonical panel-input metric.
    assert by_name["ic"]["import_path"] == "factrix.metrics.ic"
    assert by_name["ic"]["input_kind"] == "panel"
    assert by_name["ic"]["docs_anchor"] == "api/metrics/ic.md#factrix.metrics.ic.ic"
    assert by_name["ic"]["emitted_name"] == "ic"

    # Function name and emitted name diverge for the historical
    # exceptions tracked in _EMITTED_NAME_OVERRIDES.
    assert by_name["fama_macbeth"]["emitted_name"] == "fm_beta"
    assert by_name["pooled_ols"]["emitted_name"] == "pooled_beta"

    # ``breakeven_cost`` / ``net_spread`` are the scalar-input utilities.
    assert by_name["breakeven_cost"]["import_path"] == "factrix.metrics.tradability"
    assert by_name["breakeven_cost"]["input_kind"] == "scalar"
    assert by_name["net_spread"]["input_kind"] == "scalar"

    # Every other row is panel-input.
    panels = {r["name"] for r in rows if r["input_kind"] == "panel"}
    scalars = {r["name"] for r in rows if r["input_kind"] == "scalar"}
    assert scalars == {"breakeven_cost", "net_spread"}
    assert panels.isdisjoint(scalars)


def test_import_path_resolves_for_every_row() -> None:
    # Walk every cell so the assertion covers cross-cell rows too.
    import importlib

    seen: set[str] = set()
    for scope in FactorScope:
        for signal in Signal:
            for r in fl.list_metrics(scope, signal, format="json"):
                if r["name"] in seen:
                    continue
                seen.add(r["name"])
                module = importlib.import_module(r["import_path"])
                assert hasattr(module, r["name"]), (
                    f"{r['import_path']} is missing {r['name']}"
                )


def test_with_import_renders_two_column_text() -> None:
    rendered = fl.list_metrics(
        FactorScope.INDIVIDUAL, Signal.CONTINUOUS, with_import=True
    )
    assert all(" → factrix.metrics." in line for line in rendered)
    # Same row order as the plain text output — only the rendering changes.
    plain = fl.list_metrics(FactorScope.INDIVIDUAL, Signal.CONTINUOUS)
    assert [line.split(" → ", 1)[0].rstrip() for line in rendered] == plain


def test_with_import_is_ignored_under_json_format() -> None:
    # ``with_import`` is a text-only knob; JSON always carries the field.
    a = fl.list_metrics(FactorScope.INDIVIDUAL, Signal.CONTINUOUS, format="json")
    b = fl.list_metrics(
        FactorScope.INDIVIDUAL, Signal.CONTINUOUS, format="json", with_import=True
    )
    assert a == b


def test_json_output_is_stable_across_calls() -> None:
    a = fl.list_metrics(FactorScope.COMMON, Signal.CONTINUOUS, format="json")
    b = fl.list_metrics(FactorScope.COMMON, Signal.CONTINUOUS, format="json")
    assert a == b


# ---------------------------------------------------------------------------
# unrepresented (scope, signal) pair
# ---------------------------------------------------------------------------


def test_incompatible_axis_pair_raises() -> None:
    # All four real (scope, signal) combos are populated; simulate a
    # genuinely unrepresented pair by monkeypatching the index empty.
    with (
        patch("factrix._describe.user_facing_rows", return_value=[]),
        pytest.raises(IncompatibleAxisError),
    ):
        fl.list_metrics(FactorScope.INDIVIDUAL, Signal.CONTINUOUS)


# ---------------------------------------------------------------------------
# spanning special-case
# ---------------------------------------------------------------------------


def test_spanning_metrics_appear_for_individual_continuous_only() -> None:
    ind = set(fl.list_metrics(FactorScope.INDIVIDUAL, Signal.CONTINUOUS))
    com = set(fl.list_metrics(FactorScope.COMMON, Signal.CONTINUOUS))
    assert {"spanning_alpha", "greedy_forward_selection"}.issubset(ind)
    assert {"spanning_alpha", "greedy_forward_selection"}.isdisjoint(com)


# ---------------------------------------------------------------------------
# public surface
# ---------------------------------------------------------------------------


def test_list_metrics_in_public_namespace() -> None:
    assert "list_metrics" in fl.__all__
    assert callable(fl.list_metrics)


def test_metric_row_dataclass_fields_are_serialisable() -> None:
    row = user_facing_rows()[0]
    assert isinstance(row, MetricRow)
    serialised = json.dumps(
        {
            "name": row.name,
            "module": row.module,
            "cell": row.cell.raw,
            "agg_order": row.agg_order,
            "inference_se": row.inference_se,
            "import_path": row.import_path,
            "input_kind": row.input_kind,
        }
    )
    assert json.loads(serialised)["name"] == row.name
