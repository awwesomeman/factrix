"""Stationary ([Politis-Romano (1994)][politis-romano-1994]) bootstrap for dependent time series.

Parametric inference (standard t-test, Newey-West heteroskedasticity-and-autocorrelation-consistent (HAC)) breaks down when
the sample is short relative to the dependence horizon or the marginal
distribution is heavy-tailed / skewed. The stationary bootstrap
resamples geometric-length blocks from the input series, preserving
short-range dependence without assuming a specific parametric form —
suitable for event-clustering situations, persistent macro factors, and
non-normal information coefficient (IC) distributions.

References:
    - [Politis & Romano (1994)][politis-romano-1994], "The Stationary
      Bootstrap."
    - [Politis & White (2004)][politis-white-2004], "Automatic Block-
      Length Selection for the Dependent Bootstrap." ``block_length=None``
      runs the same spectral plug-in used by ``factrix.stats.BlockBootstrap``
      (via ``factrix._stats.bootstrap._politis_white_block_length``), so
      "auto" means one calibrated estimate everywhere in the library rather
      than a cruder standalone default.
"""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral

import numpy as np


def _resolve_auto_block_length(values: np.ndarray) -> float:
    """Politis-White (2004) block length, shared with ``BlockBootstrap``.

    Matrix input resamples every column under one shared row-index draw
    (see ``stationary_bootstrap_resamples``), so a single block length must
    serve all columns. Taking the max of the per-column spectral estimates
    is the conservative choice — under-blocking the most persistent column
    would understate its dependence in the joint resample.
    """
    from factrix._stats.bootstrap import _politis_white_block_length

    if values.ndim == 1:
        return _politis_white_block_length(values, scheme="stationary")
    if values.shape[1] == 0:
        return _politis_white_block_length(
            np.zeros(values.shape[0]), scheme="stationary"
        )
    return max(
        _politis_white_block_length(values[:, j], scheme="stationary")
        for j in range(values.shape[1])
    )


def stationary_bootstrap_resamples(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    *,
    block_length: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Draw ``n_bootstrap`` stationary-bootstrap resamples of ``values``.

    Each resample has the same length ``T`` as the input. One-dimensional
    input returns ``(B, T)``; two-dimensional ``(T, m)`` input returns
    ``(B, T, m)`` and applies the same sampled row indices to every column.
    The latter preserves cross-hypothesis dependence for joint bootstrap
    procedures such as Romano-Wolf. Blocks have geometric lengths with mean
    ``block_length`` and sampling is circular.

    Args:
        values: Finite ``(T,)`` time series or aligned ``(T, m)`` matrix.
            Matrix columns are always resampled jointly; do not call the
            function separately per column when cross-column dependence
            matters.
        n_bootstrap: Number of resamples to draw.
        block_length: Mean geometric block length. Defaults to the
            [Politis-White (2004)][politis-white-2004] automatic spectral
            plug-in (falling back to the practical ``1.75 * T^(1/3)`` rule
            when the series is too short or degenerate). Must be ``>= 1``;
            block_length=1 reduces to the ordinary iid bootstrap (Efron).
        seed: Seed for ``np.random.default_rng`` to make the resample
            reproducible.

    Returns:
        ``(n_bootstrap, T)`` array for vector input or
        ``(n_bootstrap, T, m)`` for matrix input.

    References:
        - [Politis & Romano (1994)][politis-romano-1994]. "The Stationary
          Bootstrap." Journal of the American Statistical Association,
          89(428), 1303–1313. Stationary block bootstrap with geometric
          block lengths — the resampling scheme this function implements.
        - [Politis & White (2004)][politis-white-2004]. "Automatic Block-
          Length Selection for the Dependent Bootstrap." Econometric
          Reviews, 23(1), 53–70. Source of the spectral plug-in
          ``block_length=None`` resolves to.
    """
    from factrix._stats.bootstrap import _stationary_block_indices

    values = np.asarray(values, dtype=float)
    if values.ndim not in (1, 2):
        raise ValueError(f"values must have shape (T,) or (T, m); got {values.shape}.")
    if values.size and not np.all(np.isfinite(values)):
        raise ValueError("values must be finite.")
    if (
        isinstance(n_bootstrap, bool)
        or not isinstance(n_bootstrap, Integral)
        or n_bootstrap < 1
    ):
        raise ValueError(
            f"n_bootstrap must be a positive integer; got {n_bootstrap!r}."
        )
    n_bootstrap = int(n_bootstrap)
    n = len(values)
    if n == 0:
        return np.empty((n_bootstrap, *values.shape), dtype=float)

    if block_length is None:
        block_length = _resolve_auto_block_length(values)
    if block_length < 1.0:
        raise ValueError(f"block_length must be >= 1.0, got {block_length!r}")

    rng = np.random.default_rng(seed)
    idx = _stationary_block_indices(n, n_bootstrap, float(block_length), rng)
    return values[idx]


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    block_length: float | None = None,
    seed: int | None = None,
    statistic: Callable[[np.ndarray], float] | None = None,
) -> tuple[float, float, float]:
    """Stationary-bootstrap confidence interval for a statistic.

    Default ``statistic`` is the arithmetic mean (matches the parametric
    t-test's H₀ null). Pass a callable taking a 1-D array and returning
    a scalar to CI other statistics (e.g. Sharpe, median, skewness).

    Args:
        values: 1-D array of the original series.
        n_bootstrap: Resample count.
        ci: Two-sided coverage, e.g. ``0.95`` for a 95% CI. Must be in
            ``(0, 1)``.
        block_length: See ``stationary_bootstrap_resamples``.
        seed: Reproducibility seed.
        statistic: Scalar function applied to each resample. Defaults
            to ``np.mean``.

    Returns:
        ``(ci_low, ci_high, point)`` where ``point`` is the statistic
        on the original sample.

    References:
        - [Politis & Romano (1994)][politis-romano-1994]. "The Stationary
          Bootstrap." Journal of the American Statistical Association,
          89(428), 1303–1313. Underlying resampling scheme; percentile CI
          on the bootstrap distribution of the statistic.
    """
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci!r}")
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(
            f"bootstrap_mean_ci: values must be 1-D; got shape {values.shape}."
        )
    resamples = stationary_bootstrap_resamples(
        values,
        n_bootstrap=n_bootstrap,
        block_length=block_length,
        seed=seed,
    )
    # Fast path for the default (mean): native axis reduction beats
    # apply_along_axis by 10-100× on large resample matrices. Custom
    # callables fall through to the generic loop.
    if statistic is None:
        stats = resamples.mean(axis=1)
        point = float(values.mean())
    else:
        stats = np.apply_along_axis(statistic, 1, resamples)
        point = float(statistic(values))
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(stats, alpha))
    hi = float(np.quantile(stats, 1.0 - alpha))
    return lo, hi, point
