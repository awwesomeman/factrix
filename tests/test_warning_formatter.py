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
import pytest
from factrix._codes import _DECLARE_HINT, WarningCode, _emit_warning, _format_warning
from factrix.metrics import notional_turnover, quantile_spread
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


class TestOneEmitChokepoint:
    """No ``WarningCode`` warning is raised outside :func:`_emit_warning`."""

    def test_no_warning_code_warn_call_outside_the_chokepoint(self):
        offenders: list[str] = []
        for path in sorted(_PACKAGE.rglob("*.py")):
            if path == _CHOKEPOINT:
                continue
            source = path.read_text()
            tree = ast.parse(source)
            lines = source.splitlines()
            for call in _warn_calls(tree):
                # A code-bearing site names WarningCode in the enclosing
                # block; the surviving non-code advisories (bhy singletons,
                # romano_wolf resample counts, the greedy-selection snooping
                # note) name none.
                context = "\n".join(
                    lines[max(0, call.lineno - 25) : (call.end_lineno or call.lineno)]
                )
                if "WarningCode" in context:
                    offenders.append(f"{path.relative_to(_PACKAGE)}:{call.lineno}")
        assert offenders == [], (
            "these warnings.warn sites raise a WarningCode without going "
            f"through factrix._codes._emit_warning: {offenders}"
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
