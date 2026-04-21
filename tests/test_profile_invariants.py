"""Structural invariants on Profile classes and Rule definitions.

Catches developer-authoring errors that would otherwise surface only at
runtime:

- ``CANONICAL_P_FIELD`` declared but not listed in ``P_VALUE_FIELDS``
  (already asserted in ``ProfileSet.multiple_testing_correct`` but raised
  only when BHY runs; this test fails at import time instead).
- An entry in ``P_VALUE_FIELDS`` is not a declared dataclass field on
  the Profile, or its annotation is not the ``PValue`` NewType — catches
  typos / stale whitelist entries after field renames.
- A ``Rule.recommended_p_source`` names a field outside the matching
  Profile's ``P_VALUE_FIELDS`` — ``_verdict_with_warnings`` would raise
  at the first PASS evaluation; this test fails at definition time.

These are developer errors, not user-data pathologies. Failing here
keeps `fl.evaluate` / BHY runtime errors surfaced via code review, not
via silent production breakage.
"""

from __future__ import annotations

import dataclasses
import typing

import pytest

from factorlib._types import FactorType, PValue
from factorlib.evaluation.diagnostics._rules import (
    CROSS_SECTIONAL_RULES,
    CROSS_TYPE_RULES,
    EVENT_RULES,
    MACRO_COMMON_RULES,
    MACRO_PANEL_RULES,
    _CUSTOM_RULES,
)
from factorlib.evaluation.profiles import _PROFILE_REGISTRY


_RULE_LISTS_BY_TYPE = {
    FactorType.CROSS_SECTIONAL: CROSS_SECTIONAL_RULES,
    FactorType.EVENT_SIGNAL: EVENT_RULES,
    FactorType.MACRO_PANEL: MACRO_PANEL_RULES,
    FactorType.MACRO_COMMON: MACRO_COMMON_RULES,
}


def _annotation_is_pvalue(ann: object) -> bool:
    """True when ``ann`` is ``PValue`` (possibly wrapped in Optional).

    ``PValue`` is a ``NewType`` over ``float``; ``Optional[PValue]`` is
    ``PValue | None`` → we accept either. We purposefully reject bare
    ``float`` — ``P_VALUE_FIELDS`` must only hold PValue-typed entries
    so the NewType is load-bearing for the BHY whitelist's intent.
    """
    if ann is PValue:
        return True
    origin = typing.get_origin(ann)
    args = typing.get_args(ann)
    if origin is typing.Union or (args and type(None) in args):
        return any(a is PValue for a in args if a is not type(None))
    return False


class TestProfileWhitelistInvariants:
    """Per-Profile class: CANONICAL_P_FIELD and P_VALUE_FIELDS."""

    @pytest.mark.parametrize(
        "factor_type", list(_PROFILE_REGISTRY.keys()),
        ids=lambda ft: ft.value,
    )
    def test_canonical_p_field_is_in_whitelist(self, factor_type: FactorType):
        cls = _PROFILE_REGISTRY[factor_type]
        assert cls.CANONICAL_P_FIELD in cls.P_VALUE_FIELDS, (
            f"{cls.__name__}.CANONICAL_P_FIELD={cls.CANONICAL_P_FIELD!r} "
            f"is not listed in P_VALUE_FIELDS={sorted(cls.P_VALUE_FIELDS)}. "
            f"Add it to P_VALUE_FIELDS, or fix the canonical field name."
        )

    @pytest.mark.parametrize(
        "factor_type", list(_PROFILE_REGISTRY.keys()),
        ids=lambda ft: ft.value,
    )
    def test_p_value_fields_are_declared_and_pvalue_typed(
        self, factor_type: FactorType,
    ):
        cls = _PROFILE_REGISTRY[factor_type]
        field_names = {f.name for f in dataclasses.fields(cls)}
        hints = typing.get_type_hints(cls)
        for name in cls.P_VALUE_FIELDS:
            assert name in field_names, (
                f"{cls.__name__}.P_VALUE_FIELDS entry {name!r} is not a "
                f"declared dataclass field. Likely a typo or a stale "
                f"whitelist entry after a rename."
            )
            ann = hints.get(name)
            assert _annotation_is_pvalue(ann), (
                f"{cls.__name__}.{name} has annotation {ann!r}; expected "
                f"PValue (or PValue | None). P_VALUE_FIELDS is the BHY "
                f"input whitelist — each entry must be a genuine p-value "
                f"(use the PValue NewType from factorlib._types)."
            )


class TestRuleRecommendedPSourceInvariants:
    """Each Rule's recommended_p_source must be in the target Profile's
    whitelist. Otherwise ``_verdict_with_warnings`` would raise at the
    first PASS diagnostic that triggers it — a latent defect."""

    @pytest.mark.parametrize(
        "factor_type", list(_RULE_LISTS_BY_TYPE.keys()),
        ids=lambda ft: ft.value,
    )
    def test_type_specific_rules_recommended_p_source_in_whitelist(
        self, factor_type: FactorType,
    ):
        cls = _PROFILE_REGISTRY[factor_type]
        whitelist = cls.P_VALUE_FIELDS
        rules = _RULE_LISTS_BY_TYPE[factor_type]
        # Also include any runtime-registered custom rules for this type.
        custom = _CUSTOM_RULES.get(factor_type, [])
        for rule in list(rules) + list(custom):
            if rule.recommended_p_source is None:
                continue
            assert rule.recommended_p_source in whitelist, (
                f"Rule {rule.code!r} for {factor_type.value} recommends "
                f"p_source={rule.recommended_p_source!r}, which is not in "
                f"{cls.__name__}.P_VALUE_FIELDS={sorted(whitelist)}. "
                f"_verdict_with_warnings would raise on the first PASS "
                f"evaluation that fires this rule."
            )

    def test_cross_type_rules_must_not_recommend_p_source(self):
        """Cross-type rules fire on every profile type, so a recommended
        p_source they name is required (by ``_verdict_with_warnings``) to
        live in the firing profile's whitelist. Rather than demand it
        lives in *every* whitelist (blocks legitimate narrow patterns)
        or allow it in ≥1 (depends on predicate correctness to avoid
        runtime errors), enforce the cleaner contract: cross-type rules
        must not recommend a p_source. Authors wanting a p_source hint
        should write a type-specific rule — the type boundary makes the
        recommendation's scope explicit.
        """
        for rule in CROSS_TYPE_RULES:
            assert rule.recommended_p_source is None, (
                f"Cross-type rule {rule.code!r} sets "
                f"recommended_p_source={rule.recommended_p_source!r}. "
                f"Move this to a type-specific rule list (one per target "
                f"factor_type) so the p_source's scope is explicit and "
                f"_verdict_with_warnings cannot hit a whitelist mismatch "
                f"at runtime."
            )
