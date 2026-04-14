"""Pre-built gate combinations.

Users can use these directly or compose their own gate lists.

Usage::

    from factorlib.gates.presets import CROSS_SECTIONAL_GATES
    from factorlib.gates.pipeline import evaluate_factor

    result = evaluate_factor(df, "Mom_20D", CROSS_SECTIONAL_GATES, config)
"""

from __future__ import annotations

from factorlib.gates._protocol import GateFn
from factorlib.gates.significance import significance_gate
from factorlib.gates.oos_persistence import oos_persistence_gate

CROSS_SECTIONAL_GATES: list[GateFn] = [
    significance_gate,
    oos_persistence_gate,
]
