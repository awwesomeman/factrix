"""register_rule: external diagnose-rule injection."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

import factorlib as fl
from factorlib._types import FactorType
from factorlib.evaluation.diagnostics import (
    Rule,
    clear_custom_rules,
    register_rule,
)


@pytest.fixture(autouse=True)
def _isolate_custom_rules():
    # Ensure no test leaks into another.
    clear_custom_rules()
    yield
    clear_custom_rules()


def _cs_panel(signal: float = 0.3, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n_dates, n_assets = 80, 30
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    prices = {f"a{i}": 100.0 for i in range(n_assets)}
    rows: list[dict] = []
    for d in dates:
        f = rng.standard_normal(n_assets)
        r = signal * f * 0.01 + (1 - abs(signal)) * 0.01 * rng.standard_normal(n_assets)
        for i in range(n_assets):
            prices[f"a{i}"] *= (1 + r[i])
            rows.append({
                "date": d, "asset_id": f"a{i}",
                "factor": float(f[i]), "price": float(prices[f"a{i}"]),
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("ms")))


def test_register_rule_appends_to_diagnose_output():
    always_fire = Rule(
        code="custom.always",
        severity="info",
        message="Always fires",
        predicate=lambda _p: True,
    )
    register_rule("cross_sectional", always_fire)

    profile = fl.evaluate(_cs_panel(), "x", factor_type="cross_sectional")
    codes = [d.code for d in profile.diagnose()]
    assert "custom.always" in codes


def test_register_rule_accepts_enum_factor_type():
    register_rule(
        FactorType.CROSS_SECTIONAL,
        Rule(
            code="custom.enum_key",
            severity="info",
            message="Enum-keyed",
            predicate=lambda _p: True,
        ),
    )
    profile = fl.evaluate(_cs_panel(seed=43), "x", factor_type="cross_sectional")
    assert any(d.code == "custom.enum_key" for d in profile.diagnose())


def test_register_rule_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown factor_type"):
        register_rule(
            "not_a_type",
            Rule(code="c", severity="info", message="m", predicate=lambda _p: True),
        )


def test_custom_rules_run_after_builtins():
    # Custom predicate that depends on being the LAST to fire — sanity
    # check that built-ins already populated the diagnose list.
    register_rule(
        "cross_sectional",
        Rule(
            code="custom.last",
            severity="info",
            message="Last",
            predicate=lambda _p: True,
        ),
    )
    profile = fl.evaluate(_cs_panel(seed=44), "x", factor_type="cross_sectional")
    codes = [d.code for d in profile.diagnose()]
    assert codes[-1] == "custom.last"


def test_clear_custom_rules_by_type_is_scoped():
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

    profile = fl.evaluate(_cs_panel(seed=45), "x", factor_type="cross_sectional")
    codes = [d.code for d in profile.diagnose()]
    assert "custom.cs" not in codes
    # ES rule still registered (can't run without an event panel here,
    # but the internal dict should retain it).
    from factorlib.evaluation.diagnostics._rules import _CUSTOM_RULES

    assert FactorType.EVENT_SIGNAL in _CUSTOM_RULES
    assert FactorType.CROSS_SECTIONAL not in _CUSTOM_RULES


def test_top_level_reexport():
    # fl.register_rule / fl.Rule / fl.clear_custom_rules usable directly
    assert fl.register_rule is register_rule
    assert fl.clear_custom_rules is clear_custom_rules
    assert fl.Rule is Rule
