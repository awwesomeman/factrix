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

import factrix as fx
import polars as pl
import pytest
from factrix._axis import FactorScope, FactorDensity, Visibility
from factrix._errors import IncompatibleAxisError
from factrix._metric_index import (
    MetricSpec,
    _all_specs,
    import_path_for,
    public_specs,
)

_APPLICABILITY_DOC = pathlib.Path("docs/reference/metric-applicability.md")


# Cell-canonical primaries: SSOT moved to the auto-generated dispatch
# table at the top of metric-applicability.md (#142), whose tuple keys
# are dispatch-keyed and broader than authoring scope (e.g. ts_beta
# dispatches on (COMMON, SPARSE, *, PANEL) but its Matrix-row authoring
# cell is (COMMON, DENSE, *, PANEL); list_metrics returns the
# authoring set). Inject primaries by their authoring cell rather than
# parse the dispatch table — adding a primary requires updating one
# literal here, the same cadence as updating Matrix-row tags.
_PRIMARY_METRIC_CELLS: dict[str, list[tuple[FactorScope, FactorDensity]]] = {
    "ic": [(FactorScope.INDIVIDUAL, FactorDensity.DENSE)],
    "fm_beta": [(FactorScope.INDIVIDUAL, FactorDensity.DENSE)],
    "caar": [
        (FactorScope.INDIVIDUAL, FactorDensity.SPARSE),
        (FactorScope.COMMON, FactorDensity.SPARSE),
    ],
    "ts_beta": [(FactorScope.COMMON, FactorDensity.DENSE)],
}

# Family-subsection cell-name → (scope, density) cells.
_CELL_HEADING_MAP: dict[str, list[tuple[FactorScope, FactorDensity]]] = {
    "Individual × Continuous": [(FactorScope.INDIVIDUAL, FactorDensity.DENSE)],
    "Common × Continuous": [(FactorScope.COMMON, FactorDensity.DENSE)],
    # Matrix-row tags use ``(*, SPARSE, *, PANEL)``: wildcard scope.
    "Individual × Sparse": [
        (FactorScope.INDIVIDUAL, FactorDensity.SPARSE),
        (FactorScope.COMMON, FactorDensity.SPARSE),
    ],
    "Common × Sparse": [(FactorScope.COMMON, FactorDensity.SPARSE)],
}

# Not-cell-bound family-name → cells. Spread-series consumers operate on
# the Individual × Continuous quantile spread output; series-tools are
# wildcard-scope, ``(*, DENSE, *, TIMESERIES)``.
_NOT_CELL_BOUND_MAP: dict[str, list[tuple[FactorScope, FactorDensity]]] = {
    "Spread-series consumers": [(FactorScope.INDIVIDUAL, FactorDensity.DENSE)],
    "Series-tools": [
        (FactorScope.INDIVIDUAL, FactorDensity.DENSE),
        (FactorScope.COMMON, FactorDensity.DENSE),
    ],
}


def _parse_applicability_doc() -> dict[tuple[FactorScope, FactorDensity], set[str]]:
    """Build per-cell expected name sets from the applicability matrix.

    Walks the family subsections under ``## Other metrics by family``:
    each ``### <Family> — Cell: <CellName>`` heading sets the active
    cell list; subsequent ``| [`name`]...`` table rows are assigned to
    it. ``### <Family> — not cell-bound`` activates the wildcard
    families (spread-series / series-tools). Cell-canonical primaries
    are injected from :data:`_PRIMARY_METRIC_CELLS`. Each new ``##``
    heading clears the active list so rows in unrelated sections
    (event-study contracts, sample-size constants) are ignored.
    """
    expected: dict[tuple[FactorScope, FactorDensity], set[str]] = {
        (FactorScope.INDIVIDUAL, FactorDensity.DENSE): set(),
        (FactorScope.COMMON, FactorDensity.DENSE): set(),
        (FactorScope.INDIVIDUAL, FactorDensity.SPARSE): set(),
        (FactorScope.COMMON, FactorDensity.SPARSE): set(),
    }
    for name, cells in _PRIMARY_METRIC_CELLS.items():
        for cell in cells:
            expected[cell].add(name)

    text = _APPLICABILITY_DOC.read_text(encoding="utf-8")
    cell_re = re.compile(r"^###\s+(.+?)\s+—\s+Cell:\s+(.+?)\s*$")
    nocell_re = re.compile(r"^###\s+(.+?)\s+—\s+not cell-bound\s*$")
    row_re = re.compile(r"^\|\s*\[`([^`]+)`\]")
    h2_re = re.compile(r"^##\s")

    active: list[tuple[FactorScope, FactorDensity]] = []
    for line in text.splitlines():
        if h2_re.match(line):
            active = []
            continue
        if m := cell_re.match(line):
            active = _CELL_HEADING_MAP.get(m.group(2).strip(), [])
            continue
        if m := nocell_re.match(line):
            active = _NOT_CELL_BOUND_MAP.get(m.group(1).strip(), [])
            continue
        if active and (m := row_re.match(line)):
            for cell in active:
                expected[cell].add(m.group(1))
    return expected


# ---------------------------------------------------------------------------
# applicability-doc parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scope", "density"),
    [
        (FactorScope.INDIVIDUAL, FactorDensity.DENSE),
        (FactorScope.COMMON, FactorDensity.DENSE),
        (FactorScope.INDIVIDUAL, FactorDensity.SPARSE),
        (FactorScope.COMMON, FactorDensity.SPARSE),
    ],
)
def test_list_metrics_matches_applicability_doc(
    scope: FactorScope, density: FactorDensity
) -> None:
    expected = _parse_applicability_doc()[(scope, density)]
    assert expected, "applicability doc parser found no rows for this cell"
    actual = set(fx.list_metrics(scope, density))
    assert actual == expected, (
        f"({scope.value}, {density.value}) drift\n"
        f"  only in list_metrics: {sorted(actual - expected)}\n"
        f"  only in doc table:    {sorted(expected - actual)}"
    )


# ---------------------------------------------------------------------------
# stage-1 helper exclusion
# ---------------------------------------------------------------------------


def test_internal_specs_excluded_from_public_specs() -> None:
    public_names = {spec.name for _, spec in public_specs()}
    internal_names = {
        spec.name for _, spec in _all_specs() if spec.visibility is Visibility.INTERNAL
    }
    assert internal_names.isdisjoint(public_names)


def test_compute_rolling_mean_beta_remains_user_facing() -> None:
    # Sanity: it starts with ``compute_`` but is a real metric per the doc;
    # exclusion is by visibility, not prefix.
    names = {spec.name for _, spec in public_specs()}
    assert "compute_rolling_mean_beta" in names


# ---------------------------------------------------------------------------
# sort stability + format
# ---------------------------------------------------------------------------


def test_text_output_is_sorted_by_module_then_name() -> None:
    out = fx.list_metrics(FactorScope.INDIVIDUAL, FactorDensity.DENSE)
    json_rows = fx.list_metrics(
        FactorScope.INDIVIDUAL, FactorDensity.DENSE, format="json"
    )
    expected_order = [r["name"] for r in json_rows]
    assert out == expected_order
    keys = [(r["module"], r["name"]) for r in json_rows]
    assert keys == sorted(keys)


def test_json_format_round_trips() -> None:
    rows = fx.list_metrics(
        FactorScope.INDIVIDUAL, FactorDensity.DENSE, format="json"
    )
    text = json.dumps(rows)
    decoded = json.loads(text)
    assert decoded == rows
    sample = rows[0]
    assert set(sample) == {
        "name",
        "module",
        "family",
        "cell",
        "agg_order",
        "inference_se",
        "import_path",
        "input_kind",
        "docs_anchor",
    }


def test_json_carries_import_path_and_input_kind() -> None:
    rows = fx.list_metrics(
        FactorScope.INDIVIDUAL, FactorDensity.DENSE, format="json"
    )
    by_name = {r["name"]: r for r in rows}

    # ``ic`` is the canonical panel-input metric.
    assert by_name["ic"]["import_path"] == "factrix.metrics.ic"
    assert by_name["ic"]["input_kind"] == "panel"
    assert by_name["ic"]["docs_anchor"] == "api/metrics/ic.md#factrix.metrics.ic.ic"

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
        for density in FactorDensity:
            for r in fx.list_metrics(scope, density, format="json"):
                if r["name"] in seen:
                    continue
                seen.add(r["name"])
                module = importlib.import_module(r["import_path"])
                assert hasattr(module, r["name"]), (
                    f"{r['import_path']} is missing {r['name']}"
                )


def test_with_import_renders_two_column_text() -> None:
    rendered = fx.list_metrics(
        FactorScope.INDIVIDUAL, FactorDensity.DENSE, with_import=True
    )
    assert all(" → factrix.metrics." in line for line in rendered)
    # Same row order as the plain text output — only the rendering changes.
    plain = fx.list_metrics(FactorScope.INDIVIDUAL, FactorDensity.DENSE)
    assert [line.split(" → ", 1)[0].rstrip() for line in rendered] == plain


def test_with_import_is_ignored_under_json_format() -> None:
    # ``with_import`` is a text-only knob; JSON always carries the field.
    a = fx.list_metrics(FactorScope.INDIVIDUAL, FactorDensity.DENSE, format="json")
    b = fx.list_metrics(
        FactorScope.INDIVIDUAL, FactorDensity.DENSE, format="json", with_import=True
    )
    assert a == b


def test_json_output_is_stable_across_calls() -> None:
    a = fx.list_metrics(FactorScope.COMMON, FactorDensity.DENSE, format="json")
    b = fx.list_metrics(FactorScope.COMMON, FactorDensity.DENSE, format="json")
    assert a == b


# ---------------------------------------------------------------------------
# unrepresented (scope, density) pair
# ---------------------------------------------------------------------------


def test_incompatible_axis_pair_raises() -> None:
    # All four real (scope, density) combos are populated; simulate a
    # genuinely unrepresented pair by monkeypatching the index empty.
    with (
        patch("factrix._metric_index.public_specs", return_value=()),
        pytest.raises(IncompatibleAxisError),
    ):
        fx.list_metrics(FactorScope.INDIVIDUAL, FactorDensity.DENSE)


# ---------------------------------------------------------------------------
# spanning special-case
# ---------------------------------------------------------------------------


def test_spanning_metrics_appear_for_individual_continuous_only() -> None:
    ind = set(fx.list_metrics(FactorScope.INDIVIDUAL, FactorDensity.DENSE))
    com = set(fx.list_metrics(FactorScope.COMMON, FactorDensity.DENSE))
    assert {"spanning_alpha", "greedy_forward_selection"}.issubset(ind)
    assert {"spanning_alpha", "greedy_forward_selection"}.isdisjoint(com)


# ---------------------------------------------------------------------------
# public surface
# ---------------------------------------------------------------------------


def test_list_metrics_in_public_namespace() -> None:
    assert "list_metrics" in fx.__all__
    assert callable(fx.list_metrics)


def test_public_spec_fields_are_serialisable() -> None:
    stem, spec = public_specs()[0]
    serialised = json.dumps(
        {
            "name": spec.name,
            "module": stem,
            "family": stem,
            "cell": spec.cell.raw,
            "agg_order": spec.agg_order,
            "inference_se": spec.inference,
            "import_path": import_path_for(stem),
            "input_kind": spec.input_kind,
        }
    )
    assert json.loads(serialised)["name"] == spec.name


class TestNoArgOverview:
    def test_no_arg_returns_family_grouped_dict(self) -> None:
        overview = fx.list_metrics()
        assert isinstance(overview, dict)
        # Keys are concept families (module stems); ic family present.
        assert "ic" in overview
        assert {s.name for s in overview["ic"]} >= {"ic", "ic_newey_west", "ic_ir"}

    def test_overview_keys_match_public_spec_stems(self) -> None:
        overview = fx.list_metrics()
        assert set(overview) == {stem for stem, _ in public_specs()}

    def test_overview_values_are_specs_not_callables(self) -> None:
        overview = fx.list_metrics()
        assert all(
            isinstance(spec, MetricSpec)
            for specs in overview.values()
            for spec in specs
        )

    def test_overview_covers_every_public_spec_once(self) -> None:
        overview = fx.list_metrics()
        flattened = [spec for specs in overview.values() for spec in specs]
        assert len(flattened) == len(public_specs())

    def test_overview_keys_are_sorted(self) -> None:
        overview = fx.list_metrics()
        assert list(overview) == sorted(overview)

    def test_single_axis_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="exactly one axis"):
            fx.list_metrics(FactorScope.INDIVIDUAL)  # type: ignore[call-overload]
        with pytest.raises(ValueError, match="exactly one axis"):
            fx.list_metrics(density=FactorDensity.DENSE)  # type: ignore[call-overload]


class TestOverviewNotRunnable:
    """The §6 overview is a catalog; evaluate must reject it with guidance."""

    def test_is_metrics_overview_recognises_overview(self) -> None:
        from factrix import _is_metrics_overview

        assert _is_metrics_overview(fx.list_metrics()) is True

    def test_is_metrics_overview_rejects_runnable_and_empty(self) -> None:
        from factrix import _is_metrics_overview

        _, spec = public_specs()[0]
        assert _is_metrics_overview([spec]) is False  # a runnable list
        assert _is_metrics_overview({}) is False  # empty dict
        assert _is_metrics_overview({"x": [1, 2]}) is False  # not specs

    def test_evaluate_rejects_overview_with_guidance(self) -> None:
        from factrix._errors import UserInputError

        with pytest.raises(UserInputError, match="overview catalog"):
            fx.evaluate(
                pl.DataFrame(),
                metrics=fx.list_metrics(),  # type: ignore[arg-type]
                factor_cols=["alpha"],
                forward_periods=5,
            )


def test_json_agg_order_and_family_are_distinct_fields() -> None:
    rows = fx.list_metrics(
        FactorScope.INDIVIDUAL, FactorDensity.DENSE, format="json"
    )
    by_name = {r["name"]: r for r in rows}
    # agg_order is the reduction order; family is the concept group.
    assert by_name["ic"]["agg_order"] == "cs-first"
    assert by_name["ic"]["family"] == "ic"
