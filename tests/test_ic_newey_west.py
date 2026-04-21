"""ic_newey_west + CrossSectionalProfile.ic_nw_p + overlap rule."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from factrix._types import MIN_IC_PERIODS
from factrix.metrics.ic import ic, ic_newey_west


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def _ic_series(values: list[float]) -> pl.DataFrame:
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(len(values))]
    return pl.DataFrame({"date": dates, "ic": values}).with_columns(
        pl.col("date").cast(pl.Datetime("ms")),
    )


def test_ic_newey_west_short_circuits_below_min():
    df = _ic_series([0.05] * (MIN_IC_PERIODS - 1))
    out = ic_newey_west(df)
    assert out.metadata["reason"] == "insufficient_ic_periods"


def test_ic_newey_west_records_method_and_lags(rng):
    df = _ic_series(rng.normal(0.05, 0.02, 60).tolist())
    out = ic_newey_west(df, forward_periods=5)
    assert "Newey-West" in out.metadata["method"]
    assert out.metadata["newey_west_lags"] >= 4  # max(floor(60^(1/3))=3, 5-1=4)


def test_ic_newey_west_less_significant_than_ic_when_autocorrelated(rng):
    """Overlapping forward returns create positive autocorrelation in the IC
    series; NW should deflate significance relative to the plain t-test."""
    innovations = rng.normal(0.03, 0.02, 100)
    ar_series: list[float] = []
    prev = 0.0
    for eps in innovations:
        val = 0.7 * prev + eps
        ar_series.append(float(val))
        prev = val
    df = _ic_series(ar_series)

    plain = ic(df, forward_periods=1)
    nw = ic_newey_west(df, forward_periods=5)

    assert nw.metadata["p_value"] >= plain.metadata["p_value"]


def test_profile_exposes_ic_nw_p_and_whitelist(cs_profile_strong):
    assert hasattr(cs_profile_strong, "ic_nw_p")
    assert "ic_nw_p" in type(cs_profile_strong).P_VALUE_FIELDS


def test_overlap_rule_fires_when_only_ic_is_significant(cs_profile_strong):
    from factrix.evaluation.diagnostics import (
        Rule, clear_custom_rules, register_rule,
    )

    clear_custom_rules()
    try:
        # Use the same cs profile but synthesize a scenario by replacing
        # ic_p and ic_nw_p via a disposable proxy — easiest to exercise
        # the rule is via the stock built-in with a hand-crafted profile
        # from dataclasses.replace.
        import dataclasses
        p = dataclasses.replace(
            cs_profile_strong, ic_p=0.01, ic_nw_p=0.20,
        )
        codes = {d.code for d in p.diagnose()}
        assert "cs.overlapping_returns_inflates_ic" in codes
        ds = [d for d in p.diagnose() if d.code == "cs.overlapping_returns_inflates_ic"]
        assert ds[0].recommended_p_source == "ic_nw_p"
    finally:
        clear_custom_rules()


def test_overlap_rule_silent_when_nw_also_significant(cs_profile_strong):
    import dataclasses
    p = dataclasses.replace(cs_profile_strong, ic_p=0.01, ic_nw_p=0.02)
    codes = {d.code for d in p.diagnose()}
    assert "cs.overlapping_returns_inflates_ic" not in codes


def test_multiple_testing_accepts_ic_nw_p(cs_profiles_and_artifacts):
    from factrix.evaluation.profile_set import ProfileSet

    profiles, _ = cs_profiles_and_artifacts
    ps = ProfileSet(profiles).multiple_testing_correct(p_source="ic_nw_p")
    df = ps.to_polars()
    assert df["mt_p_source"][0] == "ic_nw_p"
