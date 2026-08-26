"""Tests for factrix.metrics.event_quality."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from factrix.metrics.event_quality import (
    event_hit_rate,
    event_ic,
    event_skewness,
    profit_factor,
)

from .conftest import with_estimation_window


def _events_panel(n_events: int, seed: int = 0) -> pl.DataFrame:
    """One asset, one event per date, with a zero-return estimation window.

    The event family subtracts an estimation-window mean, so a frame of bare
    event rows has no sample at all. Warming up with zero returns makes the
    estimated mean zero, leaving the abnormal return equal to the raw return
    these fixtures are written against.
    """
    rng = np.random.default_rng(seed)
    return with_estimation_window(
        pl.DataFrame(
            {
                "date": pl.Series(
                    [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_events)],
                    dtype=pl.Datetime("ms"),
                ),
                "asset_id": ["A"] * n_events,
                "factor": [1.0] * n_events,
                "forward_return": rng.normal(0.01, 0.02, size=n_events),
            }
        )
    )


class TestEventSkewness:
    def test_thin_sample_below_skewtest_floor_returns_null_p_and_alternative(self):
        # scipy.stats.skewtest requires n >= 20; below that event_skewness
        # short-circuits its own significance test (p=None) while still
        # returning a descriptive skewness value.
        result = event_skewness(_events_panel(10), forward_periods=1)
        assert result.p_value is None
        assert result.alternative is None
        assert result.stat is None
        assert np.isfinite(result.value)

    def test_large_sample_returns_two_sided_p_and_alternative(self):
        result = event_skewness(_events_panel(50), forward_periods=1)
        assert result.p_value is not None
        assert result.alternative == "two-sided"
        assert result.stat is not None


# ---------------------------------------------------------------------------
# Non-finite handling across the event-quality family (regression)
# ---------------------------------------------------------------------------


def _event_panel(returns: list, factors: list | None = None) -> pl.DataFrame:
    """One asset, one event on every date of the panel.

    The panel carries no non-event rows, so consecutive events are one calendar
    step apart and the event-axis spacing pass would thin them at the default
    horizon. Callers that are testing the non-finite boundary rather than the
    stride pass ``forward_periods=1`` (a documented no-op) to isolate it.
    """
    n = len(returns)
    return with_estimation_window(
        pl.DataFrame(
            {
                "date": pl.Series(
                    [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)],
                    dtype=pl.Datetime("ms"),
                ),
                "asset_id": ["A"] * n,
                "factor": [1.0] * n if factors is None else factors,
                "forward_return": returns,
            }
        )
    )


@pytest.mark.parametrize("hole", [float("nan"), None])
class TestNonFiniteEventsDropped:
    """Every event-quality metric must exclude non-finite events from the
    sample its headline / stat / p_value / n_obs describe."""

    def test_hit_rate_does_not_score_a_hole_as_a_miss(self, hole):
        # 4 wins, 1 hole. Old behaviour: the hole failed `signed_car > 0`
        # and was counted as a miss -> rate 4/5 instead of 4/4.
        result = event_hit_rate(
            _event_panel([0.01, 0.02, 0.03, 0.04, hole]), forward_periods=1
        )
        assert result.value == pytest.approx(1.0)
        assert result.n_obs == 4
        assert result.metadata["n_events"] == 4
        assert result.metadata["n_hits"] == 4
        assert result.metadata["n_events_dropped_non_finite"] == 1

    def test_profit_factor_n_obs_matches_the_summed_sample(self, hole):
        result = profit_factor(_event_panel([0.03, -0.01, 0.02, -0.04, hole]))
        # Old behaviour: the hole entered neither sum but stayed in n_obs.
        assert result.n_obs == 4
        assert result.metadata["n_events"] == 4
        assert result.metadata["n_wins"] + result.metadata["n_losses"] == 4
        assert result.metadata["n_events_dropped_non_finite"] == 1
        assert result.value == pytest.approx(1.0)

    def test_skewness_does_not_raise_on_a_hole(self, hole):
        rng = np.random.default_rng(0)
        returns = [*rng.normal(0.01, 0.02, 40), hole]
        # Old behaviour: NaN propagated into skewtest, p became NaN and
        # MetricResult raised ValueError.
        result = event_skewness(_event_panel(returns), forward_periods=1)
        assert np.isfinite(result.value)
        assert result.p_value is not None
        assert np.isfinite(result.p_value)
        assert result.n_obs == 40
        assert result.metadata["n_events_dropped_non_finite"] == 1

    def test_event_ic_does_not_return_a_nan_correlation(self, hole):
        rng = np.random.default_rng(1)
        n = 40
        factors = [*rng.uniform(0.5, 2.0, n), 1.5]
        returns = [*rng.normal(0.01, 0.02, n), hole]
        result = event_ic(_event_panel(returns, factors), forward_periods=1)
        assert np.isfinite(result.value)
        assert np.isfinite(result.p_value)
        assert result.n_obs == n
        assert result.metadata["n_events_dropped_non_finite"] == 1


class TestNonFiniteFactorDropped:
    def test_nan_factor_survives_the_event_filter_and_is_dropped(self):
        # `NaN != 0` is True in polars, so a NaN factor reaches the event
        # sample and sign(NaN) poisons signed_car.
        result = event_hit_rate(
            _event_panel(
                [0.01, 0.02, 0.03, 0.04, 0.05],
                [1.0, 1.0, 1.0, 1.0, float("nan")],
            ),
            forward_periods=1,
        )
        assert result.value == pytest.approx(1.0)
        assert result.n_obs == 4
        assert result.metadata["n_events_dropped_non_finite"] == 1


class TestEventHitRateAlwaysExact:
    def test_exact_binomial_reported_at_every_sample_size(self):
        # The normal-approximation branch (n >= 20) is gone: stat is always
        # the hit count and the method label never switches.
        from scipy import stats as sp_stats

        rng = np.random.default_rng(7)
        returns = list(rng.normal(0.01, 0.02, 200))
        result = event_hit_rate(_event_panel(returns), forward_periods=1)

        hits = result.metadata["n_hits"]
        assert result.stat == float(hits)
        assert result.metadata["stat_type"] == "binomial_hits"
        assert result.metadata["method"] == "binomial exact test"
        assert result.p_value == pytest.approx(
            sp_stats.binomtest(hits, result.n_obs, 0.5).pvalue
        )

    def test_clean_sample_reports_zero_drops(self):
        result = event_hit_rate(
            _event_panel([0.01, -0.02, 0.03, -0.04, 0.05]), forward_periods=1
        )
        assert result.metadata["n_events_dropped_non_finite"] == 0
