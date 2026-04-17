"""Diagnostic rules for FactorProfile.

Each rule is a ``Rule(predicate, severity, message, code)``. Profile's
``diagnose()`` dispatches to the rule list for its factor type via
``diagnose_profile``.

Separating rules from the Profile classes keeps the rule set editable
without touching the dataclass schema, and lets rules be loaded /
filtered programmatically (e.g. an AI agent suppressing 'info'-level
hints while acting on 'veto').
"""

from factorlib.evaluation.diagnostics._rules import (
    Rule,
    diagnose_profile,
)

__all__ = ["Rule", "diagnose_profile"]
