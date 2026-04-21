"""ProfileSet.diagnose_all() + ProfileSet.with_canonical() contract."""

from __future__ import annotations

import polars as pl
import pytest

from factrix.evaluation.diagnostics import Rule, clear_custom_rules, register_rule
from factrix.evaluation.profile_set import ProfileSet
from factrix.evaluation.profiles import CrossSectionalProfile


@pytest.fixture(autouse=True)
def _cleanup_rules():
    yield
    clear_custom_rules()


def test_diagnose_all_schema(cs_profile_strong):
    """Column contract is stable regardless of how many rules fire."""
    df = ProfileSet([cs_profile_strong]).diagnose_all()
    assert set(df.columns) == {
        "factor_name", "severity", "code", "message", "recommended_p_source",
    }


def test_diagnose_all_empty_when_no_profiles():
    empty = ProfileSet([], profile_cls=CrossSectionalProfile)
    df = empty.diagnose_all()
    assert df.height == 0
    assert set(df.columns) == {
        "factor_name", "severity", "code", "message", "recommended_p_source",
    }


def test_diagnose_all_flattens_across_profiles(cs_profile_strong, cs_profile_weak):
    register_rule(
        "cross_sectional",
        Rule(
            code="test.always",
            severity="warn",
            message="m",
            predicate=lambda p: True,
            recommended_p_source="spread_p",
        ),
    )
    ps = ProfileSet([cs_profile_strong, cs_profile_weak])
    df = ps.diagnose_all()
    assert df.height >= 2
    names = df["factor_name"].unique().to_list()
    assert cs_profile_strong.factor_name in names
    assert cs_profile_weak.factor_name in names
    assert "test.always" in df["code"].to_list()
    assert "spread_p" in df["recommended_p_source"].to_list()


def test_with_canonical_rejects_unknown_field(cs_profile_strong):
    ps = ProfileSet([cs_profile_strong])
    with pytest.raises(ValueError, match="P_VALUE_FIELDS"):
        ps.with_canonical("not_a_field")


def test_with_canonical_rebinds_canonical_p_alias_column(cs_profile_strong):
    ps = ProfileSet([cs_profile_strong]).with_canonical("spread_p")
    df = ps.to_polars()
    assert df["canonical_p"][0] == df["spread_p"][0]


def test_with_canonical_flows_through_multiple_testing_correct(
    cs_profile_strong, cs_profile_weak,
):
    ps = ProfileSet([cs_profile_strong, cs_profile_weak]).with_canonical("spread_p")
    corrected = ps.multiple_testing_correct()  # default p_source="canonical_p"
    mt_source = corrected.to_polars()["mt_p_source"][0]
    assert mt_source == "spread_p"


def test_with_canonical_preserved_across_filter_chain(cs_profile_strong, cs_profile_weak):
    ps = (
        ProfileSet([cs_profile_strong, cs_profile_weak])
        .with_canonical("spread_p")
        .filter(pl.col("factor_name").is_not_null())
    )
    corrected = ps.multiple_testing_correct()
    assert corrected.to_polars()["mt_p_source"][0] == "spread_p"


def test_explicit_p_source_overrides_with_canonical(
    cs_profile_strong, cs_profile_weak,
):
    ps = ProfileSet([cs_profile_strong, cs_profile_weak]).with_canonical("spread_p")
    corrected = ps.multiple_testing_correct(p_source="ic_p")
    assert corrected.to_polars()["mt_p_source"][0] == "ic_p"
