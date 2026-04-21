"""Diagnostic rules for FactorProfile.

Each rule is a ``Rule(predicate, severity, message, code)``. Profile's
``diagnose()`` dispatches to the rule list for its factor type via
``diagnose_profile``.

Separating rules from the Profile classes keeps the rule set editable
without touching the dataclass schema, and lets rules be loaded /
filtered programmatically (e.g. an AI agent suppressing 'info'-level
hints while acting on 'veto').
"""

from factrix._types import DiagnosticSeverity
from factrix.evaluation.diagnostics._rules import (
    Rule,
    clear_custom_rules,
    diagnose_profile,
    register_rule,
)

__all__ = [
    "Rule",
    "DiagnosticSeverity",
    "diagnose_profile",
    "register_rule",
    "clear_custom_rules",
]
