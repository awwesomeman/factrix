"""Verify ``_verdict_with_warnings`` upgrade path.

Contract:
- FAILED is never softened.
- PASS stays PASS when no ``warn`` diagnostic carries a
  ``recommended_p_source`` different from ``CANONICAL_P_FIELD``.
- PASS → PASS_WITH_WARNINGS when at least one ``warn`` diagnostic names
  a whitelisted alternative p_source the user has not adopted.
"""

from __future__ import annotations

import pytest

from factrix._types import Diagnostic
from factrix.evaluation.diagnostics import Rule, clear_custom_rules, register_rule
from factrix.evaluation.profiles import CrossSectionalProfile


@pytest.fixture(autouse=True)
def _cleanup_custom_rules():
    yield
    clear_custom_rules()


def test_pass_stays_pass_without_warnings(cs_profile_strong):
    assert cs_profile_strong.verdict() == "PASS"


def test_failed_never_softened_by_warnings(cs_profile_weak):
    register_rule(
        "cross_sectional",
        Rule(
            code="test.fake_risk",
            severity="warn",
            message="fake risk",
            predicate=lambda p: True,
            recommended_p_source="spread_p",
        ),
    )
    assert cs_profile_weak.verdict() == "FAILED"


def test_pass_with_warnings_when_alternative_recommended(cs_profile_strong):
    register_rule(
        "cross_sectional",
        Rule(
            code="test.alt_p_source",
            severity="warn",
            message="prefer spread_p for this profile",
            predicate=lambda p: True,
            recommended_p_source="spread_p",
        ),
    )
    assert cs_profile_strong.verdict() == "PASS_WITH_WARNINGS"


def test_warn_without_recommendation_does_not_upgrade(cs_profile_strong):
    register_rule(
        "cross_sectional",
        Rule(
            code="test.warn_no_rec",
            severity="warn",
            message="informational warn",
            predicate=lambda p: True,
        ),
    )
    assert cs_profile_strong.verdict() == "PASS"


def test_rule_recommending_canonical_is_noop(cs_profile_strong):
    canonical = cs_profile_strong.CANONICAL_P_FIELD
    register_rule(
        "cross_sectional",
        Rule(
            code="test.same_canonical",
            severity="warn",
            message="recommending the already-in-use p_source",
            predicate=lambda p: True,
            recommended_p_source=canonical,
        ),
    )
    assert cs_profile_strong.verdict() == "PASS"


def test_info_severity_never_upgrades(cs_profile_strong):
    register_rule(
        "cross_sectional",
        Rule(
            code="test.info_rec",
            severity="info",
            message="info-level recommendation",
            predicate=lambda p: True,
            recommended_p_source="spread_p",
        ),
    )
    assert cs_profile_strong.verdict() == "PASS"


def test_typo_in_recommended_p_source_raises(cs_profile_strong):
    register_rule(
        "cross_sectional",
        Rule(
            code="test.typo",
            severity="warn",
            message="typo'd p_source",
            predicate=lambda p: True,
            recommended_p_source="bpm_p",  # not in P_VALUE_FIELDS
        ),
    )
    with pytest.raises(ValueError, match="P_VALUE_FIELDS"):
        cs_profile_strong.verdict()


def test_diagnostic_carries_recommended_p_source():
    d = Diagnostic(
        severity="warn",
        message="m",
        code="c",
        recommended_p_source="bmp_p",
    )
    assert d.recommended_p_source == "bmp_p"
    # default preserved for callers that don't populate
    d2 = Diagnostic(severity="info", message="m")
    assert d2.recommended_p_source is None
