"""Tests for factrix.metrics.event_quality.event_skewness."""

import numpy as np
import polars as pl
from factrix.metrics.event_quality import event_skewness


def _events_panel(n_events: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "factor": [1.0] * n_events,
            "forward_return": rng.normal(0.01, 0.02, size=n_events),
        }
    )


class TestEventSkewness:
    def test_thin_sample_below_skewtest_floor_returns_null_p_and_alternative(self):
        # scipy.stats.skewtest requires n >= 20; below that event_skewness
        # short-circuits its own significance test (p=None) while still
        # returning a descriptive skewness value.
        result = event_skewness(_events_panel(10))
        assert result.p_value is None
        assert result.alternative is None
        assert result.stat is None
        assert np.isfinite(result.value)

    def test_large_sample_returns_two_sided_p_and_alternative(self):
        result = event_skewness(_events_panel(50))
        assert result.p_value is not None
        assert result.alternative == "two-sided"
        assert result.stat is not None
