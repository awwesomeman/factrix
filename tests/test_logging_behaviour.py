"""Logging behaviour at decision (evaluation) and metric layers."""

from __future__ import annotations

import logging

import pytest

from factrix.evaluation.diagnostics import (
    Rule, clear_custom_rules, register_rule,
)
from factrix.evaluation.profile_set import ProfileSet


@pytest.fixture(autouse=True)
def _cleanup_rules():
    yield
    clear_custom_rules()


def test_multiple_testing_correct_emits_info(cs_profiles_and_artifacts, caplog):
    profiles, _ = cs_profiles_and_artifacts
    with caplog.at_level(logging.INFO, logger="factrix.evaluation"):
        ProfileSet(profiles).multiple_testing_correct(fdr=0.05)
    msgs = [r.message for r in caplog.records if r.name == "factrix.evaluation"]
    assert any("multiple_testing_correct" in m and "n_rejected" in m for m in msgs)


def test_pass_with_warnings_emits_warning(cs_profile_strong, caplog):
    register_rule(
        "cross_sectional",
        Rule(
            code="test.rebind_spread",
            severity="warn",
            message="m",
            predicate=lambda p: True,
            recommended_p_source="spread_p",
        ),
    )
    with caplog.at_level(logging.WARNING, logger="factrix.evaluation"):
        verdict = cs_profile_strong.verdict()
    assert verdict == "PASS_WITH_WARNINGS"
    msgs = [r.message for r in caplog.records if r.name == "factrix.evaluation"]
    assert any("PASS_WITH_WARNINGS" in m for m in msgs)


def test_sample_non_overlapping_debug_and_warning(caplog):
    from datetime import datetime, timedelta
    import polars as pl

    from factrix.metrics._helpers import _sample_non_overlapping

    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(12)]
    df = pl.DataFrame({"date": dates}).with_columns(
        pl.col("date").cast(pl.Datetime("ms"))
    )
    with caplog.at_level(logging.DEBUG, logger="factrix.metrics"):
        _sample_non_overlapping(df, forward_periods=5)
    records = [r for r in caplog.records if r.name == "factrix.metrics"]
    assert any(r.levelno == logging.DEBUG for r in records)
    # 12 dates, fp=5 → 3 samples, below MIN_IC_PERIODS*1.5=15
    assert any(r.levelno == logging.WARNING for r in records)


def test_newey_west_warns_when_lags_exceed_sample_ratio(caplog):
    import numpy as np
    from factrix._stats import _newey_west_t_test

    with caplog.at_level(logging.WARNING, logger="factrix.metrics"):
        _newey_west_t_test(np.arange(10, dtype=np.float64), lags=5)
    msgs = [r.message for r in caplog.records if r.name == "factrix.metrics"]
    assert any("poorly conditioned" in m for m in msgs)
