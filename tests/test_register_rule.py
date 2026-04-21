"""register_rule: external diagnose-rule injection."""

from __future__ import annotations

import pytest

import factrix as fl
from factrix._types import FactorType
from factrix.evaluation.diagnostics import (
    Rule,
    clear_custom_rules,
    register_rule,
)


@pytest.fixture(autouse=True)
def _isolate_custom_rules():
    clear_custom_rules()
    yield
    clear_custom_rules()


def test_register_rule_appends_to_diagnose_output(cs_profile_strong):
    register_rule(
        "cross_sectional",
        Rule(
            code="custom.always",
            severity="info",
            message="Always fires",
            predicate=lambda _p: True,
        ),
    )
    codes = [d.code for d in cs_profile_strong.diagnose()]
    assert "custom.always" in codes


def test_register_rule_accepts_enum_factor_type(cs_profile_strong):
    register_rule(
        FactorType.CROSS_SECTIONAL,
        Rule(
            code="custom.enum_key",
            severity="info",
            message="Enum-keyed",
            predicate=lambda _p: True,
        ),
    )
    assert any(
        d.code == "custom.enum_key" for d in cs_profile_strong.diagnose()
    )


def test_register_rule_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown factor_type"):
        register_rule(
            "not_a_type",
            Rule(code="c", severity="info", message="m", predicate=lambda _p: True),
        )


def test_custom_rules_run_after_builtins(cs_profile_strong):
    register_rule(
        "cross_sectional",
        Rule(
            code="custom.last",
            severity="info",
            message="Last",
            predicate=lambda _p: True,
        ),
    )
    codes = [d.code for d in cs_profile_strong.diagnose()]
    assert codes[-1] == "custom.last"


def test_clear_custom_rules_by_type_is_scoped(cs_profile_strong):
    register_rule(
        "cross_sectional",
        Rule(code="custom.cs", severity="info", message="m",
             predicate=lambda _p: True),
    )
    register_rule(
        "event_signal",
        Rule(code="custom.es", severity="info", message="m",
             predicate=lambda _p: True),
    )
    clear_custom_rules("cross_sectional")

    codes = [d.code for d in cs_profile_strong.diagnose()]
    assert "custom.cs" not in codes

    from factrix.evaluation.diagnostics._rules import _CUSTOM_RULES

    assert FactorType.EVENT_SIGNAL in _CUSTOM_RULES
    assert FactorType.CROSS_SECTIONAL not in _CUSTOM_RULES


def test_top_level_reexport():
    assert fl.register_rule is register_rule
    assert fl.clear_custom_rules is clear_custom_rules
    assert fl.Rule is Rule
