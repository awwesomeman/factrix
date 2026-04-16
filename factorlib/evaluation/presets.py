"""Pre-built gate combinations and per-type defaults."""

from __future__ import annotations

from factorlib._types import FactorType
from factorlib.config import BaseConfig
from factorlib.evaluation._protocol import GateFn
from functools import partial

from factorlib.evaluation.gates.significance import significance_gate
from factorlib.evaluation.gates.oos_persistence import oos_persistence_gate
from factorlib.evaluation.gates.fm_significance import fm_significance_gate
from factorlib.evaluation.gates.ts_significance import ts_significance_gate
from factorlib.evaluation.gates.event_significance import event_significance_gate

CROSS_SECTIONAL_GATES: list[GateFn] = [
    significance_gate,
    oos_persistence_gate,
]

EVENT_SIGNAL_GATES: list[GateFn] = [
    event_significance_gate,
    partial(oos_persistence_gate, value_key="caar_values"),
]

MACRO_PANEL_GATES: list[GateFn] = [
    fm_significance_gate,
    partial(oos_persistence_gate, value_key="beta_values"),
]

MACRO_COMMON_GATES: list[GateFn] = [
    ts_significance_gate,
    partial(oos_persistence_gate, value_key="beta_values"),
]

_DEFAULT_GATES: dict[FactorType, list[GateFn]] = {
    FactorType.CROSS_SECTIONAL: CROSS_SECTIONAL_GATES,
    FactorType.EVENT_SIGNAL: EVENT_SIGNAL_GATES,
    FactorType.MACRO_PANEL: MACRO_PANEL_GATES,
    FactorType.MACRO_COMMON: MACRO_COMMON_GATES,
}


def default_gates_for(config: BaseConfig) -> list[GateFn]:
    """Return default gate list for the given config's factor type."""
    ft = type(config).factor_type
    gates = _DEFAULT_GATES.get(ft)
    if gates is None:
        raise NotImplementedError(
            f"No default gates defined for {ft}. "
            f"Pass gates= explicitly."
        )
    return list(gates)
