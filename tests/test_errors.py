"""Coverage for the user-facing error contract (#165)."""

from __future__ import annotations

import pytest
from factrix._errors import (
    FactrixError,
    UserInputError,
    format_user_error,
)


def test_user_input_error_subclasses_factrix_error() -> None:
    assert issubclass(UserInputError, FactrixError)


def test_candidates_with_close_match_emits_suggestion() -> None:
    msg = format_user_error(
        verb="bhy",
        field="expand_over",
        value="univere_id",
        candidates=["universe_id", "regime_id", "sector"],
        docs_path="api/bhy#expand_over",
    )
    assert "bhy(): unknown expand_over='univere_id'" in msg
    assert 'Did you mean: "universe_id"' in msg
    assert "Available:" in msg
    assert "https://awwesomeman.github.io/factrix/api/bhy#expand_over" in msg


def test_candidates_no_close_match_omits_suggestion_line() -> None:
    msg = format_user_error(
        verb="run_metrics",
        field="metrics",
        value="completely_unrelated_xyz",
        candidates=["ic", "fm_lambda", "caar"],
        docs_path="api/run_metrics",
    )
    assert "Did you mean" not in msg
    assert "Available:" in msg


def test_expected_only_renders_type_branch() -> None:
    msg = format_user_error(
        verb="evaluate",
        field="factor_col",
        value="alpha_x",
        expected="column present in panel",
        docs_path="api/evaluate#factor_col",
    )
    assert "evaluate(): invalid factor_col='alpha_x'" in msg
    assert "Expected: column present in panel" in msg
    assert "Available" not in msg


def test_candidates_take_priority_when_both_passed() -> None:
    msg = format_user_error(
        verb="x",
        field="f",
        value="a",
        candidates=["aa", "bb"],
        expected="should not appear",
        docs_path="api/x",
    )
    assert "Expected" not in msg
    assert "Available" in msg


def test_neither_candidates_nor_expected_raises() -> None:
    with pytest.raises(ValueError, match="candidates= or expected="):
        format_user_error(verb="x", field="f", value=1, docs_path="api/x")


def test_docs_path_leading_slash_normalized() -> None:
    msg = format_user_error(
        verb="x",
        field="f",
        value="a",
        candidates=["aa"],
        docs_path="/api/x",
    )
    assert "https://awwesomeman.github.io/factrix/api/x" in msg
    assert "factrix/.api" not in msg
    assert "factrix//api" not in msg


def test_available_list_is_sorted_for_stable_output() -> None:
    msg = format_user_error(
        verb="x",
        field="f",
        value="a",
        candidates=["zeta", "alpha", "mu"],
        docs_path="api/x",
    )
    assert "['alpha', 'mu', 'zeta']" in msg


def test_userinputerror_can_carry_helper_message() -> None:
    msg = format_user_error(
        verb="bhy",
        field="expand_over",
        value="univere_id",
        candidates=["universe_id", "regime_id"],
        docs_path="api/bhy#expand_over",
    )
    err = UserInputError(msg)
    assert "Did you mean" in str(err)
    assert isinstance(err, FactrixError)
