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

import factrix as fx
import polars as pl
import pytest
from factrix._axis import FactorDensity, FactorScope, SpecRole
from factrix._metric_index import (
    MetricSpec,
    _all_specs,
    import_path_for,
    public_specs,
)

_APPLICABILITY_DOC = pathlib.Path("docs/reference/metric-applicability.md")


def _names_for_cell(scope: FactorScope, density: FactorDensity) -> set[str]:
    """Public metric names applicable to a ``(scope, density)`` cell.

    The cell filter ``list_metrics(scope, density)`` exposed was retired
    the underlying mechanism — ``spec.cell.matches`` over
    ``public_specs()`` — is what drives both this drift guard and
    ``inspect_data``'s per-metric verdict.
    """
    return {
        spec.name for _, spec in public_specs() if spec.cell.matches(scope, density)
    }


# Cell-canonical primaries: SSOT moved to the auto-generated dispatch
# table at the top of metric-applicability.md. Some public metrics have
# wildcard cells in the registry (for example caar applies to both sparse
# scopes), while this test needs the user-facing cell groups shown in the docs.
# Inject primaries by their documented cell rather than parsing the table —
# adding a primary requires updating one literal here, the same cadence as
# updating the registered specs.
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
    # Sparse specs use wildcard scope.
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
    actual = _names_for_cell(scope, density)
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
        spec.name for _, spec in _all_specs() if spec.role is SpecRole.PIPELINE
    }
    assert internal_names.isdisjoint(public_names)


# ---------------------------------------------------------------------------
# import-path resolution
# ---------------------------------------------------------------------------


def test_import_path_resolves_for_every_public_spec() -> None:
    import importlib

    for stem, spec in public_specs():
        module = importlib.import_module(import_path_for(stem))
        assert hasattr(module, spec.name), (
            f"{import_path_for(stem)} missing {spec.name}"
        )


# ---------------------------------------------------------------------------
# spanning special-case
# ---------------------------------------------------------------------------


def test_spanning_metrics_appear_for_individual_continuous_only() -> None:
    ind = _names_for_cell(FactorScope.INDIVIDUAL, FactorDensity.DENSE)
    com = _names_for_cell(FactorScope.COMMON, FactorDensity.DENSE)
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
            "aggregation": spec.aggregation.value,
            "import_path": import_path_for(stem),
            "input_shape": spec.input_shape.value,
        }
    )
    assert json.loads(serialised)["name"] == spec.name


class TestNoArgOverview:
    def test_no_arg_returns_family_grouped_dict(self) -> None:
        overview = fx.list_metrics()
        assert isinstance(overview, dict)
        # Keys are concept families (module stems); ic family present.
        assert "ic" in overview
        assert {s.name for s in overview["ic"]} >= {"ic", "ic_ir"}

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

    def test_positional_args_are_rejected(self) -> None:
        # The retired cell filter took (scope, density); the no-arg form
        # takes none, so any positional argument is now a TypeError.
        with pytest.raises(TypeError):
            fx.list_metrics(FactorScope.INDIVIDUAL)  # type: ignore[call-arg]


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


class TestMetricsSummary:
    def test_columns_and_membership(self) -> None:
        summary = fx.metrics_summary()
        assert summary.columns == ["family", "metric", "summary"]
        metrics = summary["metric"].to_list()
        # Public metric names appear; pipeline producers (compute_ic) do not.
        assert "ic" in metrics
        assert "quantile_spread" in metrics
        assert "compute_ic" not in metrics

    def test_summary_is_first_docstring_line(self) -> None:
        from factrix.metrics.ic import ic

        summary = fx.metrics_summary()
        row = summary.filter(pl.col("metric") == "ic").row(0, named=True)
        expected = (ic.__doc__ or "").strip().split("\n", 1)[0].strip()
        assert row["summary"] == expected
        assert row["family"] == "ic"

    def test_one_row_per_public_spec(self) -> None:
        assert fx.metrics_summary().height == len(public_specs())


class TestMetricFactorySignature:
    def test_signature_exposes_real_params_not_varargs(self) -> None:
        import inspect

        from factrix.metrics import ic, quantile_spread

        params = inspect.signature(ic).parameters
        assert "inference" in params
        assert "ic_df" in params
        # The metaclass-shadowed ``(*args, **kwargs)`` must be gone.
        assert not any(
            p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params.values()
        )
        # Dispatch-internal underscore params are hidden.
        assert (
            "_precomputed_series" not in inspect.signature(quantile_spread).parameters
        )
