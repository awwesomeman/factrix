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
      Length Selection for the Dependent Bootstrap." Full procedure
      requires spectral-density estimation; this module falls back to
      the practical ``L = round(1.75 * T^(1/3))`` rule when
      ``block_length=None``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _default_block_length(n: int) -> float:
    """Practical fallback per [Politis-White (2004)][politis-white-2004] commentary.

    Full PW picks ``L`` from a plug-in of the spectral density; that
    needs another set of choices we don't want to inherit here. The
    ``1.75 * T^(1/3)`` rule is the widely-cited pragmatic compromise
    (same cube-root cadence as [Newey-West (1994)][newey-west-1994], slightly larger constant
    because the stationary bootstrap's kernel is different).
    """
    if n < 2:
        return 1.0
    return max(1.0, 1.75 * (n ** (1.0 / 3.0)))


def stationary_bootstrap_resamples(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    *,
    block_length: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Draw ``n_bootstrap`` stationary-bootstrap resamples of ``values``.

    Each resample has the same length ``T`` as the input. Blocks have
    geometric lengths with mean ``block_length`` — at each step there is
    probability ``1/block_length`` of starting a new block at a random
    position in the original series. Sampling is circular.

    Args:
        values: 1-D array of the original time series.
        n_bootstrap: Number of resamples to draw.
        block_length: Mean geometric block length. Defaults to
            ``1.75 * T^(1/3)`` ([Politis-White (2004)][politis-white-2004] practical rule).
            Must be ``>= 1``; block_length=1 reduces to the ordinary
            iid bootstrap (Efron).
        seed: Seed for ``np.random.default_rng`` to make the resample
            reproducible.

    Returns:
        ``(n_bootstrap, T)`` numpy array of resampled series.

    References:
        - [Politis & Romano (1994)][politis-romano-1994]. "The Stationary
          Bootstrap." Journal of the American Statistical Association,
          89(428), 1303–1313. Stationary block bootstrap with geometric
          block lengths — the resampling scheme this function implements.
        - [Politis & White (2004)][politis-white-2004]. "Automatic Block-
          Length Selection for the Dependent Bootstrap." Econometric
          Reviews, 23(1), 53–70. Source of the practical ``1.75 * T^(1/3)``
          block-length default.
    """
    from factrix._stats.bootstrap import _stationary_block_indices

    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return np.empty((n_bootstrap, 0), dtype=float)

    if block_length is None:
        block_length = _default_block_length(n)
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


# Target peak budget for the materialised resample tensor
# ``values[k_slice][:, idx]`` in ``bootstrap_mean_ci_batch``. K-chunking
# is bounded by this so memory does not scale linearly with n_factors;
# the chosen 256 MB lands two orders of magnitude under the 32 GB laptop
# preset envelope while still letting a 1000-factor / 250-date / B=999
# batch fit in ~10 K-chunks.
_BATCH_RESAMPLE_BYTES_BUDGET = 256 * 1024 * 1024


def bootstrap_mean_ci_batch(
    values: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    block_length: float | None = None,
    seed: int | None = None,
    chunk_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised stationary-bootstrap CI across ``n_factors`` series.

    Shares one ``(B, T)`` block-index matrix across all factors per
    chunk: every factor sees the same resample positions, so the
    bootstrap distributions are jointly drawn rather than independently
    redrawn per factor. This collapses the per-factor Python loop into
    a single batched advanced-index over ``values``.

    For ``n_factors == 1`` the result equals
    :func:`bootstrap_mean_ci` evaluated on ``values[0]`` with the same
    ``seed`` (the index matrix is drawn identically).

    Args:
        values: ``(n_factors, n_observations)`` array. Each row is an
            independent series; the bootstrap is applied per row but
            shares one set of block indices.
        n_bootstrap: Resample count ``B`` (same for every factor).
        ci: Two-sided coverage (same as single-factor variant).
        block_length: Mean geometric block length. Defaults to
            ``1.75 * T^(1/3)`` (one value shared across factors —
            matches single-factor default).
        seed: Reproducibility seed; the drawn index matrix is identical
            to the single-factor path under the same seed.
        chunk_size: K-chunk for materialising the resample tensor.
            ``None`` (default) picks a chunk that targets a 256 MB
            peak on the resample tensor; pass an explicit value to
            override (smaller = lower peak, slightly higher Python
            overhead).

    Returns:
        ``(ci_low, ci_high, point)`` — each a ``(n_factors,)`` array.

    References:
        Same as :func:`bootstrap_mean_ci`.
    """
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci!r}")
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(
            f"values must be 2-D (n_factors, n_observations); got shape {values.shape!r}"
        )
    n_factors, n = values.shape
    if n_factors == 0:
        empty = np.empty((0,), dtype=float)
        return empty, empty.copy(), empty.copy()

    if n == 0:
        zeros = np.zeros(n_factors, dtype=float)
        return zeros, zeros.copy(), zeros.copy()

    if block_length is None:
        block_length = _default_block_length(n)
    if block_length < 1.0:
        raise ValueError(f"block_length must be >= 1.0, got {block_length!r}")

    from factrix._stats.bootstrap import _stationary_block_indices

    rng = np.random.default_rng(seed)
    # Single shared index matrix — preserves single-factor seed
    # equivalence and bounds the per-chunk memory to the values tensor
    # (the index matrix itself is B*T*8 bytes, ~2 MB at B=999 T=250).
    idx = _stationary_block_indices(n, n_bootstrap, float(block_length), rng)

    if chunk_size is None:
        per_factor_bytes = n_bootstrap * n * 8
        chunk_size = max(1, _BATCH_RESAMPLE_BYTES_BUDGET // max(per_factor_bytes, 1))
    chunk_size = min(chunk_size, n_factors)

    alpha = (1.0 - ci) / 2.0
    lo = np.empty(n_factors, dtype=float)
    hi = np.empty(n_factors, dtype=float)
    for k_start in range(0, n_factors, chunk_size):
        k_stop = min(k_start + chunk_size, n_factors)
        # ``values[k_start:k_stop]`` is (k, T); advanced-indexing with
        # ``idx`` of shape (B, T) broadcasts to (k, B, T).
        chunk_resamples = values[k_start:k_stop][:, idx]
        chunk_means = chunk_resamples.mean(axis=2)  # (k, B)
        # Per-row quantiles; axis=1 over the B dimension.
        q = np.quantile(chunk_means, [alpha, 1.0 - alpha], axis=1)
        lo[k_start:k_stop] = q[0]
        hi[k_start:k_stop] = q[1]

    point = values.mean(axis=1)
    return lo, hi, point
