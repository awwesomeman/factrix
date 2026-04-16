"""Pre-built gate combinations and per-type defaults."""

from __future__ import annotations

from factorlib._types import FactorType
from factorlib.config import BaseConfig
from factorlib.evaluation._protocol import GateFn
from factorlib.evaluation.gates.significance import significance_gate
from factorlib.evaluation.gates.oos_persistence import oos_persistence_gate

CROSS_SECTIONAL_GATES: list[GateFn] = [
    significance_gate,
    oos_persistence_gate,
]

_DEFAULT_GATES: dict[FactorType, list[GateFn]] = {
    FactorType.CROSS_SECTIONAL: CROSS_SECTIONAL_GATES,
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
