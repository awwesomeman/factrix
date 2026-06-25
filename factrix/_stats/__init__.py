"""Internal statistical toolkit — index of per-topic submodules.

Public surface lives in ``factrix.stats`` (selection-only inference
instances / Benjamini-Hochberg-Yekutieli (BHY) multiple-testing) and
``factrix.inference`` (series-mean HAC methods); this package holds the
private primitives those façades and the procedure layer call into.

**Submodules:**

- ``constants``   — TIMESERIES / PANEL sample-size thresholds and
  ``auto_bartlett`` Newey-West bandwidth.
- ``core``        — t-statistic, p-value (t / z), significance marker,
  and binomial test primitives. ``_p_value_from_t`` /
  ``_significance_marker`` are shared with the heteroskedasticity-and-autocorrelation-consistent (HAC) t-tests in ``hac``.
- ``hac``         — Newey-West (Bartlett) and Hansen-Hodrick
  (rectangular) HAC SE / t-test for a sample mean; shared
  ``_resolve_nw_lags`` bandwidth picker honouring the overlap horizon.
- ``ols``         — ordinary least squares (OLS) slope-only (``_ols_nw_slope_t``) and full
  multivariate (``_ols_nw_multivariate``) with Newey-West HAC
  covariance.
- ``wald``        — Wald χ² test for linear restrictions on an
  estimated coefficient vector.
- ``unit_root``   — Augmented Dickey-Fuller (constant-only) test with
  [MacKinnon (1996)][mackinnon-1996] interpolated p-values.
- ``diagnostics`` — Residual diagnostics (Ljung-Box portmanteau).

BHY multiple-testing lives in ``factrix.stats.multiple_testing``; it
operates on *p-values*.

All names re-exported below are private (leading underscore); callers
use ``from factrix._stats import X`` and the symbol resolves regardless
of which submodule it physically lives in.
"""

from __future__ import annotations

from factrix._stats.core import (
    _BINOMIAL_EXACT_CUTOFF,
    _binomial_test_method_name,
    _binomial_two_sided_p,
    _calc_t_stat,
    _p_value_from_t,
    _p_value_from_z,
    _significance_marker,
    _t_stat_from_array,
    _t_test_summary,
)
from factrix._stats.diagnostics import _ljung_box
from factrix._stats.hac import (
    _bartlett_lrcov,
    _driscoll_kraay_cov,
    _hansen_hodrick_se,
    _hansen_hodrick_t_test,
    _newey_west_se,
    _newey_west_t_test,
    _resolve_nw_lags,
)
from factrix._stats.ols import _ols_nw_multivariate, _ols_nw_slope_t
from factrix._stats.unit_root import _ADF_CRITS_CONSTANT, _adf, _adf_pvalue_interp
from factrix._stats.wald import _wald_p_linear

__all__ = [
    "_ADF_CRITS_CONSTANT",
    "_BINOMIAL_EXACT_CUTOFF",
    "_adf",
    "_adf_pvalue_interp",
    "_bartlett_lrcov",
    "_binomial_test_method_name",
    "_binomial_two_sided_p",
    "_calc_t_stat",
    "_driscoll_kraay_cov",
    "_hansen_hodrick_se",
    "_hansen_hodrick_t_test",
    "_ljung_box",
    "_newey_west_se",
    "_newey_west_t_test",
    "_ols_nw_multivariate",
    "_ols_nw_slope_t",
    "_p_value_from_t",
    "_p_value_from_z",
    "_resolve_nw_lags",
    "_significance_marker",
    "_t_stat_from_array",
    "_t_test_summary",
    "_wald_p_linear",
]
