"""Stationary (Politis-Romano 1994) bootstrap for dependent time series.

Parametric inference (standard t-test, Newey-West HAC) breaks down when
the sample is short relative to the dependence horizon or the marginal
distribution is heavy-tailed / skewed. The stationary bootstrap
resamples geometric-length blocks from the input series, preserving
short-range dependence without assuming a specific parametric form —
suitable for event-clustering situations, persistent macro factors, and
non-normal IC distributions.

References:
    Politis & Romano (1994), "The Stationary Bootstrap."
    Politis & White (2004), "Automatic Block-Length Selection for
        the Dependent Bootstrap." (Full procedure requires spectral-
        density estimation; this module falls back to the practical
        ``L = round(1.75 * T^(1/3))`` rule when ``block_length=None``.)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _default_block_length(n: int) -> float:
    """Practical fallback per Politis-White (2004) commentary.

    Full PW picks ``L`` from a plug-in of the spectral density; that
    needs another set of choices we don't want to inherit here. The
    ``1.75 * T^(1/3)`` rule is the widely-cited pragmatic compromise
    (same cube-root cadence as Newey-West 1994, slightly larger constant
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
            ``1.75 * T^(1/3)`` (Politis-White 2004 practical rule).
            Must be ``>= 1``; block_length=1 reduces to the ordinary
            iid bootstrap (Efron).
        seed: Seed for ``np.random.default_rng`` to make the resample
            reproducible.

    Returns:
        ``(n_bootstrap, T)`` numpy array of resampled series.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return np.empty((n_bootstrap, 0), dtype=float)

    if block_length is None:
        block_length = _default_block_length(n)
    if block_length < 1.0:
        raise ValueError(f"block_length must be >= 1.0, got {block_length!r}")

    rng = np.random.default_rng(seed)
    # Probability of starting a new block at each step = 1/L.
    p_new = 1.0 / block_length
    # Pre-draw all random decisions — much faster than a Python loop.
    starts = rng.integers(0, n, size=(n_bootstrap, n))
    new_block = rng.random(size=(n_bootstrap, n)) < p_new
    # Build index matrix: when new_block[i, t] is True, take a fresh
    # random start; else continue by +1 (modulo n) from the previous.
    idx = np.empty((n_bootstrap, n), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, n):
        prev = (idx[:, t - 1] + 1) % n
        idx[:, t] = np.where(new_block[:, t], starts[:, t], prev)
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
