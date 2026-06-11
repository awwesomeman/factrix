"""``NonOverlappingSample`` — stride subsample OLS t-test inference unit."""

from __future__ import annotations

import numpy as np
import pytest
from factrix._codes import WarningCode
from factrix._stats import _p_value_from_t, _t_stat_from_array
from factrix.stats._estimator import InferenceResult
from factrix.stats.non_overlapping import NonOverlappingSample


class TestNonOverlappingSample:
    def test_strides_at_forward_periods(self):
        series = np.arange(20.0)
        result = NonOverlappingSample().compute(series, forward_periods=5)
        # series[::5] -> [0, 5, 10, 15]
        assert result.metadata["stride"] == 5
        assert result.metadata["n_obs_original"] == 20
        assert result.metadata["n_obs_sampled"] == 4

    def test_matches_primitive_t_and_p(self):
        rng = np.random.default_rng(0)
        series = rng.standard_normal(120) + 0.1
        result = NonOverlappingSample().compute(series, forward_periods=5)
        sampled = series[::5]
        expected_t = _t_stat_from_array(sampled)
        expected_p = _p_value_from_t(expected_t, len(sampled))
        assert result.stat == expected_t
        assert result.p_value == expected_p

    def test_claims_no_statcode(self):
        """Metric-internal unit: emits no profile StatCode pair."""
        result = NonOverlappingSample().compute(np.arange(60.0), forward_periods=1)
        assert isinstance(result, InferenceResult)
        assert result.stat_name is None
        assert result.p_name is None

    def test_short_sample_warns(self):
        # forward_periods=1 keeps all obs; 10 < MIN_PERIODS_WARN (30) -> warn
        result = NonOverlappingSample().compute(np.arange(10.0), forward_periods=1)
        assert WarningCode.UNRELIABLE_SE_SHORT_PERIODS in result.warnings

    def test_ample_sample_no_warning(self):
        result = NonOverlappingSample().compute(np.arange(60.0), forward_periods=1)
        assert result.warnings == frozenset()

    @pytest.mark.parametrize("attr", ["name", "description", "min_periods"])
    def test_identity_surface(self, attr):
        est = NonOverlappingSample()
        assert getattr(est, attr) is not None
        assert est.name == "NonOverlappingSample"
