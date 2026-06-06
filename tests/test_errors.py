"""Coverage for the user-facing error contract."""

from __future__ import annotations

import enum

import pytest
from factrix._errors import FactrixError, UserInputError


def test_subclasses_factrix_error_and_value_error() -> None:
    assert issubclass(UserInputError, FactrixError)
    assert issubclass(UserInputError, ValueError)


def test_other_error_subclasses_inheritance() -> None:
    import factrix
    import factrix._errors
    from factrix._errors import (
        IncompatibleAxisError,
        InsufficientSampleError,
        UnknownEstimatorError,
    )

    assert issubclass(UnknownEstimatorError, FactrixError)
    assert issubclass(UnknownEstimatorError, ValueError)

    assert issubclass(IncompatibleAxisError, FactrixError)
    assert not issubclass(IncompatibleAxisError, ValueError)

    assert issubclass(InsufficientSampleError, FactrixError)
    assert not issubclass(InsufficientSampleError, ValueError)

    # Verify ConfigError is completely removed
    assert not hasattr(factrix, "ConfigError")
    assert not hasattr(factrix._errors, "ConfigError")


def test_caught_by_generic_value_error() -> None:
    with pytest.raises(ValueError):
        raise UserInputError(
            func_name="x",
            field="f",
            value="a",
            candidates=["aa"],
            docs_path="api/x",
        )


def test_candidates_branch_renders_suggestion() -> None:
    err = UserInputError(
        func_name="bhy",
        field="expand_over",
        value="univere_id",
        candidates=["universe_id", "regime_id", "sector"],
        docs_path="api/bhy#expand_over",
    )
    msg = str(err)
    assert "bhy(): unknown expand_over='univere_id'" in msg
    assert 'Did you mean: "universe_id"' in msg
    assert "Available:" in msg
    assert "https://awwesomeman.github.io/factrix/api/bhy#expand_over" in msg


def test_attributes_exposed_for_programmatic_recovery() -> None:
    err = UserInputError(
        func_name="bhy",
        field="expand_over",
        value="univere_id",
        candidates=["universe_id", "regime_id", "sector"],
        docs_path="api/bhy#expand_over",
    )
    assert err.func_name == "bhy"
    assert err.field == "expand_over"
    assert err.value == "univere_id"
    assert err.candidates == ("regime_id", "sector", "universe_id")
    assert err.suggestions == ("universe_id",)
    assert err.expected is None
    assert err.docs_url == ("https://awwesomeman.github.io/factrix/api/bhy#expand_over")


def test_no_close_match_omits_suggestion() -> None:
    err = UserInputError(
        func_name="run_metrics",
        field="metrics",
        value="completely_unrelated_xyz",
        candidates=["ic", "fm_lambda", "caar"],
        docs_path="api/run_metrics",
    )
    assert err.suggestions == ()
    assert "Did you mean" not in str(err)
    assert "Available:" in str(err)


def test_expected_branch_for_type_mismatch() -> None:
    err = UserInputError(
        func_name="evaluate",
        field="factor_cols",
        value=["alpha_x"],
        expected="column present in panel",
        docs_path="api/evaluate#factor_cols",
    )
    assert err.candidates == ()
    assert err.expected == "column present in panel"
    assert "evaluate(): invalid factor_cols=['alpha_x']" in str(err)
    assert "Expected: column present in panel" in str(err)
    assert "Available" not in str(err)


def test_neither_candidates_nor_expected_raises() -> None:
    with pytest.raises(ValueError, match="candidates= or expected="):
        UserInputError(func_name="x", field="f", value=1, docs_path="api/x")


def test_docs_path_leading_slash_normalized() -> None:
    err = UserInputError(
        func_name="x",
        field="f",
        value="a",
        candidates=["aa"],
        docs_path="/api/x",
    )
    assert err.docs_url == "https://awwesomeman.github.io/factrix/api/x"


def test_available_truncates_when_long() -> None:
    cands = [f"m{i:03d}" for i in range(50)]
    err = UserInputError(
        func_name="x",
        field="metric",
        value="bad",
        candidates=cands,
        docs_path="api/x",
    )
    msg = str(err)
    assert "Available (15 of 50, see Docs):" in msg
    assert "m000" in msg
    assert "m049" not in msg


def test_value_repr_truncated_for_huge_value() -> None:
    huge = "x" * 500
    err = UserInputError(
        func_name="x",
        field="f",
        value=huge,
        candidates=["aa"],
        docs_path="api/x",
    )
    msg = str(err)
    assert "..." in msg
    assert len(msg.splitlines()[0]) < 200


def test_non_string_candidates_coerced() -> None:
    class DataStructure(enum.Enum):
        A = "alpha"
        B = "beta"

    err = UserInputError(
        func_name="x",
        field="structure",
        value="alfa",
        candidates=list(DataStructure),
        docs_path="api/x",
    )
    assert err.candidates == tuple(sorted(str(m) for m in DataStructure))
