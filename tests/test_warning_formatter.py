"""One emit chokepoint, one warning frame.

Every :class:`~factrix.WarningCode` advisory the library echoes goes through
:func:`factrix._codes._emit_warning` and comes out as
``<label>: <message> (<code>; declare it in expected_warnings=)``. Two things
are locked here:

* the *structural* guard — no ``warnings.warn`` call for a ``WarningCode``
  exists anywhere outside the chokepoint, so a new advisory cannot quietly
  invent its own wording; and
* the *behavioural* guard — the panel from the report that motivated this
  (8 assets, 90 periods, ``n_groups=3``) reaches stderr with every code it
  records, each carrying its metric label and its code token.

The second half is the one that would have caught the original defect: the
codes were on ``MetricResult.warning_codes`` all along and nothing printed.
"""

from __future__ import annotations

import ast
import pathlib
import warnings

import factrix as fx
import polars as pl
import pytest
from factrix._codes import _DECLARE_HINT, WarningCode, _emit_warning, _format_warning
from factrix.metrics import caar, notional_turnover, quantile_spread, spanning_alpha
from factrix.preprocess import compute_forward_return

_PACKAGE = pathlib.Path(fx.__file__).parent
#: The chokepoint itself is the one module allowed to call ``warnings.warn``
#: for a code.
_CHOKEPOINT = _PACKAGE / "_codes.py"


def _warn_calls(tree: ast.AST) -> list[ast.Call]:
    """Every ``warnings.warn(...)`` / ``_warnings.warn(...)`` call in a module."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "warn"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in {"warnings", "_warnings"}
    ]


def _warning_category(call: ast.Call) -> str | None:
    """Return the explicit warning category named by a direct warn call."""
    category = call.args[1] if len(call.args) > 1 else next(
        (kw.value for kw in call.keywords if kw.arg == "category"), None
    )
    return category.id if isinstance(category, ast.Name) else None


def _enclosing_function(tree: ast.AST, target: ast.AST) -> str:
    """Find the nearest function containing ``target`` without line heuristics."""
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    node = target
    while id(node) in parents:
        node = parents[id(node)]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return "<module>"


class TestOneEmitChokepoint:
    """No ``WarningCode`` warning is raised outside :func:`_emit_warning`."""

    def test_direct_warning_sites_are_deliberate_contract_exceptions(self):
        direct_sites: dict[str | None, set[tuple[str, str]]] = {}
        for path in sorted(_PACKAGE.rglob("*.py")):
            if path == _CHOKEPOINT:
                continue
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            for call in _warn_calls(tree):
                site = (
                    path.relative_to(_PACKAGE).as_posix(),
                    _enclosing_function(tree, call),
                )
                direct_sites.setdefault(_warning_category(call), set()).add(site)

        # This unconditional method caveat predates the structured result
        # channel and has its own explicit suppression keyword.
        assert direct_sites.pop("UserWarning", set()) == {
            ("metrics/spanning.py", "greedy_forward_selection")
        }
        # These call-wide family-construction diagnostics return no
        # EvaluationResult that could own a Warning record. They deliberately
        # remain RuntimeWarning and are outside expected_warnings=.
        assert direct_sites.pop("RuntimeWarning", set()) == {
            ("_multi_factor.py", "bhy"),
            ("_multi_factor.py", "bhy_across_metrics"),
            ("_multi_factor.py", "_partial_conjunction_one"),
            ("_multi_factor.py", "_bhy_hierarchical_one"),
            ("_multi_factor.py", "_warn_on_mixed_horizons"),
        }
        assert direct_sites == {}, (
            "new direct warnings.warn site: route a UserWarning through "
            "factrix._codes._emit_warning, or document and classify a true "
            f"contract exception: {direct_sites}"
        )

    def test_warning_record_sites_are_routed_to_an_echo_surface(self):
        """Converse guard: a new structured record needs an echo route."""
        record_sites: set[tuple[str, str]] = set()
        for path in sorted(_PACKAGE.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            for call in (
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "Warning"
            ):
                record_sites.add(
                    (
                        path.relative_to(_PACKAGE).as_posix(),
                        _enclosing_function(tree, call),
                    )
                )

        # _inspect records are all rebuilt by _surface_factor_inspection;
        # _dag and __init__ assembly records all pass through _relabel_result;
        # the slicing record co-emits inside _warn_date_axis_truncation.
        assert record_sites == {
            ("_dag.py", "_assemble"),
            ("_inspect.py", "_cross_factor_warnings"),
            ("_inspect.py", "_single_asset_event_warning"),
            ("_inspect.py", "_evaluate_applicability"),
            ("_inspect.py", "_data_level_warnings"),
            ("_inspect.py", "_scope_unidentifiable_warning"),
            ("_inspect.py", "_dense_factor_advisory_warnings"),
            ("__init__.py", "_relabel_result"),
            ("__init__.py", "_cell_compatibility_warnings"),
            ("slicing/dispatcher.py", "_warn_date_axis_truncation"),
        }, (
            "new Warning record site: route it through _emit_warning or a "
            f"documented central assembly surface: {record_sites}"
        )

    def test_chokepoint_frames_label_message_and_code(self):
        text = _format_warning(
            WarningCode.FEW_ASSETS, "n_assets=8 below the floor", label="ic"
        )
        assert text == (f"ic: n_assets=8 below the floor (few_assets; {_DECLARE_HINT})")

    def test_record_survives_a_declaration_and_the_echo_stops(self):
        codes: list[str] = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            returned = _emit_warning(
                WarningCode.FEW_ASSETS,
                "body",
                label="ic",
                expected_warnings=("few_assets",),
                warning_codes=codes,
            )
        assert returned == "few_assets"
        assert codes == ["few_assets"], "a declaration must not drop the record"
        assert caught == []

    def test_undeclared_echo_carries_the_frame(self):
        codes: list[str] = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _emit_warning(
                WarningCode.FEW_ASSETS, "body", label="ic", warning_codes=codes
            )
        assert codes == ["few_assets"]
        assert len(caught) == 1
        assert str(caught[0].message) == (f"ic: body (few_assets; {_DECLARE_HINT})")

    def test_the_code_is_recorded_once_on_a_repeated_emit(self):
        codes: list[str] = ["few_assets"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _emit_warning(
                WarningCode.FEW_ASSETS, "body", label="ic", warning_codes=codes
            )
        assert codes == ["few_assets"]


@pytest.fixture(scope="module")
def thin_panel():
    """The panel from the report: 8 assets over 90 periods, h=5."""
    return compute_forward_return(
        fx.datasets.make_cs_panel(n_assets=8, n_dates=90, rng=0), forward_periods=5
    )


class TestThinPanelReachesStderr:
    """Every code the thin panel records is also echoed, framed."""

    def _run(self, panel, **kwargs):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fx.evaluate(
                panel,
                metrics={
                    "quantile_spread": quantile_spread(n_groups=3),
                    "notional_turnover": notional_turnover(n_groups=3),
                },
                factor_cols=["factor"],
                strict=False,
                **kwargs,
            )
        return result["factor"], [str(w.message) for w in caught]

    def test_all_three_codes_echo_with_label_and_token(self, thin_panel):
        result, messages = self._run(thin_panel)
        recorded = set(result.metrics["quantile_spread"].warning_codes)
        assert recorded == {
            "unreliable_se_short_periods",
            "few_assets",
            "thin_quantile_groups",
        }
        for code in recorded:
            hits = [m for m in messages if f"({code}; {_DECLARE_HINT})" in m]
            assert hits, f"{code} was recorded but never echoed: {messages}"
            # Anatomy: every echo opens with the label that raised it.
            for hit in hits:
                label = hit.split(":", 1)[0]
                assert label and " " not in label

    def test_the_thin_group_advisory_names_the_metric_that_raised_it(self, thin_panel):
        _, messages = self._run(thin_panel)
        labels = {m.split(":", 1)[0] for m in messages if "thin_quantile_groups" in m}
        # The turnover metric buckets the same cross-section and used to
        # report clean where its spread sibling warned.
        assert "notional_turnover" in labels

    def test_notional_turnover_records_the_thin_group_code(self, thin_panel):
        result, _ = self._run(thin_panel)
        assert result.metrics["notional_turnover"].warning_codes == (
            "thin_quantile_groups",
        )

    def test_a_declaration_silences_every_echo_and_keeps_every_record(self, thin_panel):
        declared = (
            "unreliable_se_short_periods",
            "few_assets",
            "thin_quantile_groups",
        )
        result, messages = self._run(thin_panel, expected_warnings=declared)
        assert messages == []
        assert set(result.metrics["quantile_spread"].warning_codes) == set(declared)
        assert all(w.expected for w in result.warnings if w.code.value in declared)


class TestEvaluationAssemblyWarningsReachStderr:
    """Warnings created after metric execution obey the same echo contract."""

    @staticmethod
    def _panel(n_assets: int = 20, n_dates: int = 120):
        return compute_forward_return(
            fx.datasets.make_cs_panel(n_assets=n_assets, n_dates=n_dates, rng=7),
            forward_periods=5,
        )

    def test_metric_unavailable_echo_uses_the_callers_label(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fx.evaluate(
                self._panel(),
                metrics={"incremental_alpha": spanning_alpha()},
                factor_cols=["factor"],
                strict=False,
            )["factor"]
        records = [
            warning
            for warning in result.warnings
            if warning.code is WarningCode.METRIC_UNAVAILABLE
        ]
        assert len(records) == 1
        assert records[0].source == "incremental_alpha"
        assert any(
            str(warning.message).startswith("incremental_alpha:")
            and "(metric_unavailable;" in str(warning.message)
            for warning in caught
        )

    def test_structure_mismatch_echo_can_be_declared(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fx.evaluate(
                self._panel(),
                metrics={"event_test": caar()},
                factor_cols=["factor"],
                strict=False,
            )["factor"]
        assert any(
            "event_test:" in str(warning.message)
            and "(structure_mismatch;" in str(warning.message)
            for warning in caught
        )
        assert any(
            warning.code is WarningCode.STRUCTURE_MISMATCH
            for warning in result.warnings
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            quiet = fx.evaluate(
                self._panel(),
                metrics={"event_test": caar()},
                factor_cols=["factor"],
                strict=False,
                expected_warnings=("structure_mismatch",),
            )["factor"]
        record = next(
            warning
            for warning in quiet.warnings
            if warning.code is WarningCode.STRUCTURE_MISMATCH
        )
        assert record.expected is True

    def test_frequent_event_override_echoes_its_record(self):
        panel = self._panel(n_dates=220).with_columns(
            pl.when(pl.int_range(0, pl.len()) % 5 < 2)
            .then(0.0)
            .otherwise(1.0)
            .alias("factor")
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fx.evaluate(
                panel,
                metrics={"event_test": caar()},
                factor_cols=["factor"],
            )["factor"]
        assert any(
            warning.code is WarningCode.FREQUENT_EVENT_SIGNAL
            for warning in result.warnings
        )
        assert any(
            "event_test:" in str(warning.message)
            and "(frequent_event_signal;" in str(warning.message)
            for warning in caught
        )
