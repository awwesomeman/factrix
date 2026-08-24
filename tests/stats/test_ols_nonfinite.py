"""Non-finite contract for the Newey-West OLS kernels in ``factrix._stats.ols``."""

from __future__ import annotations

import numpy as np
import pytest
from factrix._stats.ols import _ols_nw_multivariate, _ols_nw_slope_t


class TestNonFiniteContract:
    """Both kernels refuse non-finite input rather than returning NaN.

    Neither degeneracy branch catches a NaN. ``_ols_nw_slope_t`` guards on
    ``sxx < EPSILON``, and every comparison with NaN is False, so a NaN x
    walks straight past it. ``_ols_nw_multivariate`` relies on
    ``np.linalg.inv`` raising ``LinAlgError`` on a singular ``X'X``, but inv
    does not raise on a NaN-bearing matrix — it returns a NaN inverse, so
    the singularity branch never fires. Both returned a silent NaN.

    Protection previously lived at the call sites; ``predictive_beta.py:115``
    even carries a comment explaining that a NaN "would flow into
    ``_ols_nw_slope_t``". That makes it a property of those callers rather
    than of the kernels. Same fail-loud contract as ``_newey_west_se``.
    """

    @staticmethod
    def _design(n: int = 30):
        rng = np.random.default_rng(0)
        x = np.arange(float(n))
        return x * 0.5 + rng.normal(size=n), x

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_slope_t_rejects_non_finite_x(self, bad):
        y, x = self._design()
        x = x.copy()
        x[3] = bad
        with pytest.raises(ValueError, match="_ols_nw_slope_t: values must be finite"):
            _ols_nw_slope_t(y, x, lags=2)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_slope_t_rejects_non_finite_y(self, bad):
        y, x = self._design()
        y = y.copy()
        y[7] = bad
        with pytest.raises(ValueError, match="_ols_nw_slope_t: values must be finite"):
            _ols_nw_slope_t(y, x, lags=2)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_multivariate_rejects_non_finite_design(self, bad):
        y, x = self._design()
        X = np.column_stack([np.ones(len(x)), x])
        X[5, 1] = bad
        with pytest.raises(
            ValueError, match="_ols_nw_multivariate: values must be finite"
        ):
            _ols_nw_multivariate(y, X, lags=2)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_multivariate_rejects_non_finite_y(self, bad):
        y, x = self._design()
        y = y.copy()
        y[2] = bad
        X = np.column_stack([np.ones(len(x)), x])
        with pytest.raises(
            ValueError, match="_ols_nw_multivariate: values must be finite"
        ):
            _ols_nw_multivariate(y, X, lags=2)

    def test_clean_input_is_unaffected(self):
        y, x = self._design()
        beta, t_stat, p_value, resid = _ols_nw_slope_t(y, x, lags=2)
        assert beta == pytest.approx(0.5, abs=0.1)
        assert np.isfinite([t_stat, p_value]).all()
        assert np.isfinite(resid).all()

    def test_short_sample_path_still_returns_the_degenerate_tuple(self):
        """The n<3 early return must sit AFTER the guard, not be bypassed."""
        beta, t_stat, p_value, resid = _ols_nw_slope_t(
            np.array([1.0, 2.0]), np.array([1.0, 2.0]), lags=1
        )
        assert (beta, t_stat, p_value) == (0.0, 0.0, 1.0)
        assert resid.shape == (2,)
