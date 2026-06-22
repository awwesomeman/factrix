"""Block-bootstrap primitives backing the ``BlockBootstrap`` Estimator.

Two resampling schemes for dependent time series, plus the
[Politis-White (2004)][politis-white-2004] automatic block-length
selector and a paired-diff empirical p-value:

- **Stationary scheme** ([Politis-Romano (1994)][politis-romano-1994]) —
  geometric block lengths with mean ``L``. Resamples are themselves
  stationary processes; preferred when downstream estimators rely on
  stationarity (CI for serially-correlated means, Sharpe).
- **Fixed scheme** ([Künsch (1989)][kunsch-1989]) — deterministic block length ``L``.
  Cleaner for variance estimation; loses stationarity at the join
  points but tighter at small ``B``.

The public ``factrix.stats.bootstrap`` module ships standalone
``stationary_bootstrap_resamples`` / ``bootstrap_mean_ci`` for callers
that want a CI utility outside the Estimator dispatch chain. This
private module is consumed by the procedure that backs the
``BlockBootstrap`` Estimator.

References:
    - Künsch, H. R. (1989). "The jackknife and the bootstrap for
      general stationary observations." Annals of Statistics, 17(3),
      1217–1241.
    - Politis, D. N. & Romano, J. P. (1994). "The Stationary
      Bootstrap." Journal of the American Statistical Association,
      89(428), 1303–1313.
    - Politis, D. N. & White, H. (2004). "Automatic Block-Length
      Selection for the Dependent Bootstrap." Econometric Reviews,
      23(1), 53–70.
"""

from __future__ import annotations

import secrets
from typing import Literal

import numpy as np

from factrix._types import EPSILON

Scheme = Literal["fixed", "stationary"]


def _flat_top_kernel(t: float) -> float:
    """Politis-Romano trapezoidal flat-top kernel.

    ``λ(t) = 1`` for ``|t| ≤ 0.5``; linear taper to 0 over ``0.5 < |t| < 1``;
    0 beyond. The flat top eliminates the small-bias term that hurts
    triangular / Bartlett kernels in spectral-density estimation at
    frequency 0 — the load-bearing input to [Politis-White (2004)][politis-white-2004].
    """
    a = abs(t)
    if a <= 0.5:
        return 1.0
    if a < 1.0:
        return 2.0 * (1.0 - a)
    return 0.0


def _politis_white_block_length(
    values: np.ndarray,
    *,
    scheme: Scheme = "stationary",
) -> float:
    """[Politis-White (2004)][politis-white-2004] automatic block length.

    Implements the spectral plug-in described in PW §3-4:
    ``L̂ = (2 Ĝ² / D̂)^(1/3) · T^(1/3)`` where ``Ĝ`` and ``D̂`` are
    flat-top kernel estimates of, respectively, the first derivative
    and variance of the spectral density at frequency 0. Caller picks
    ``scheme`` because ``D̂`` differs by a factor (``2 g(0)²`` for
    stationary, ``(4/3) g(0)²`` for fixed/circular blocks; PW eq 9 / 12).

    Falls back to ``max(1, 1.75 · T^(1/3))`` (the widely-cited practical
    PW approximation, also used by ``factrix.stats.bootstrap``) when
    the series is too short, autocovariance is degenerate, or the
    spectral estimate yields a non-finite ratio. Returns a ``float``;
    callers that need an integer block size round at the call site.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    fallback = max(1.0, 1.75 * n ** (1.0 / 3.0)) if n >= 1 else 1.0
    if n < 4:
        return fallback

    x = x - float(np.mean(x))
    gamma_0 = float(np.dot(x, x)) / n
    if gamma_0 < EPSILON:
        return fallback

    # Bandwidth search range — PW recommend k_max = ceil(sqrt(log10(T) * T)).
    k_max = min(n - 1, int(np.ceil(np.sqrt(max(np.log10(n), 1.0) * n))))
    k_max = max(k_max, 1)
    rho = np.empty(k_max + 1)
    rho[0] = 1.0
    for k in range(1, k_max + 1):
        rho[k] = float(np.dot(x[k:], x[:-k])) / (n * gamma_0)

    # Pick smallest m such that |ρ̂(m+s)| < 2·sqrt(log10(T)/T) for all
    # s = 1..K_T. Threshold = Stock-Watson (1998) significance bar.
    K_T = max(5, int(np.ceil(np.log10(n))))
    threshold = 2.0 * np.sqrt(np.log10(n) / n)
    m_pick = None
    for m in range(0, max(k_max - K_T, 1)):
        window = rho[m + 1 : m + 1 + K_T]
        if window.size and np.all(np.abs(window) < threshold):
            m_pick = m
            break
    if m_pick is None:
        return fallback
    # PW (2004) §4 doubles the chosen index for the kernel bandwidth.
    M = max(2 * m_pick, 2)
    M = min(M, k_max)

    gamma = rho * gamma_0
    # PW (2004) eq. 9 sums |k| ≤ M with the flat-top kernel; the k=M
    # term vanishes because λ(M/M) = λ(1) = 0 by construction, so
    # `range(1, M)` covers every non-zero contributor.
    g0 = gamma[0] + 2.0 * sum(_flat_top_kernel(k / M) * gamma[k] for k in range(1, M))
    g_deriv = 2.0 * sum(_flat_top_kernel(k / M) * k * gamma[k] for k in range(1, M))
    if scheme == "stationary":
        d_hat = 2.0 * g0 * g0
    elif scheme == "fixed":
        d_hat = (4.0 / 3.0) * g0 * g0
    else:
        raise ValueError(f"scheme must be 'fixed' or 'stationary'; got {scheme!r}")

    if d_hat < EPSILON or not np.isfinite(g_deriv):
        return fallback
    ratio = (2.0 * g_deriv * g_deriv) / d_hat
    if ratio <= 0.0 or not np.isfinite(ratio):
        return fallback
    L = (ratio ** (1.0 / 3.0)) * (n ** (1.0 / 3.0))
    if not np.isfinite(L) or L < 1.0:
        return fallback
    return float(min(L, n / 2.0))


def _stationary_block_indices(
    n: int,
    n_resamples: int,
    mean_block_length: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """[Politis-Romano (1994)][politis-romano-1994] geometric-block index matrix, shape ``(B, n)``.

    Block length at each step is geometric with mean ``mean_block_length``;
    sampling is circular. Mirrors ``factrix.stats.bootstrap`` resampler
    but emits indices rather than values so the same matrix can drive
    multiple statistics (mean / variance / Sharpe) without re-drawing.
    """
    if mean_block_length < 1.0:
        raise ValueError(f"mean_block_length must be >= 1.0; got {mean_block_length!r}")
    if n == 0:
        return np.empty((n_resamples, 0), dtype=np.int64)
    p_new = 1.0 / mean_block_length
    starts = rng.integers(0, n, size=(n_resamples, n))
    new_block = rng.random(size=(n_resamples, n)) < p_new
    idx = np.empty((n_resamples, n), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, n):
        prev = (idx[:, t - 1] + 1) % n
        idx[:, t] = np.where(new_block[:, t], starts[:, t], prev)
    return idx


def _fixed_block_indices(
    n: int,
    n_resamples: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """[Künsch (1989)][kunsch-1989] fixed-block index matrix, shape ``(B, n)``.

    Picks ``ceil(n / L)`` random block starts per resample, lays the
    blocks contiguously (modulo ``n`` for circular wrap), then truncates
    to length ``n``. Each resample is composed of ``ceil(n/L)`` blocks
    of identical length ``L`` — the cleaner variance-estimation path
    when serial correlation has a known horizon.
    """
    if block_length < 1:
        raise ValueError(f"block_length must be >= 1; got {block_length!r}")
    if n == 0:
        return np.empty((n_resamples, 0), dtype=np.int64)
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=(n_resamples, n_blocks))
    offsets = np.arange(block_length, dtype=np.int64)
    # Broadcast (B, n_blocks, 1) + (block_length,) → (B, n_blocks, block_length).
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    return idx.reshape(n_resamples, n_blocks * block_length)[:, :n]


def _block_bootstrap_diff_p(
    diff: np.ndarray,
    *,
    block_length: int | Literal["auto"] = "auto",
    n_resamples: int = 999,
    scheme: Scheme = "stationary",
    rng_seed: int | None = None,
) -> tuple[float, dict[str, float | int | str]]:
    """Two-sided empirical p for ``H₀: E[diff] = 0`` on a paired series.

    Resamples ``diff`` under the centring ``diff - mean(diff)`` (the
    null restricts the mean to zero; the bootstrap distribution must be
    drawn under that restriction to give a calibrated p). Empirical p
    uses Davison-Hinkley ``+1 / (B+1)`` smoothing: keeps p strictly
    positive so log-scale plots and downstream multi-stage adjustments
    don't see a hard zero.

    Args:
        diff: 1-D paired-difference series (already date-aligned by
            caller — the bootstrap does not re-align).
        block_length: ``"auto"`` runs Politis-White; ``int >= 1`` uses
            the supplied length unchanged. Stationary scheme uses the
            mean of the geometric distribution; fixed scheme uses the
            integer directly.
        n_resamples: ``B``. [Politis-White (2004)][politis-white-2004] recommends ≥ 999 for two-sided
            5% tests; default matches.
        scheme: ``"fixed"`` (Künsch) or ``"stationary"`` (Politis-Romano).
        rng_seed: ``None`` draws from system entropy; the resolved seed
            is returned in the metadata dict so the caller can record it.

    Returns:
        ``(p_value, metadata)`` — p in ``[1/(B+1), 1]``; metadata
        records the resolved block length, scheme, n_resamples, and the
        actual seed used (so the run is reproducible from the logged
        metadata even when the caller passed ``rng_seed=None``).
    """
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    if n < 2:
        return 1.0, {
            "block_length": 0,
            "n_resamples": 0,
            "scheme": scheme,
            "rng_seed": rng_seed if rng_seed is not None else -1,
        }

    if block_length == "auto":
        L_float = _politis_white_block_length(diff, scheme=scheme)
        L = max(1, round(L_float))
    else:
        L = int(block_length)
        if L < 1:
            raise ValueError(f"block_length must be >= 1; got {L!r}")

    # Resolve seed up front so it can be reported back even when None.
    # `secrets.randbits(32)` is the purpose-built "give me a random int
    # seed" call — `SeedSequence().entropy` is typed as `int | Sequence[int]
    # | None` and the Sequence branch breaks the bit-mask path.
    seed_used = secrets.randbits(32) if rng_seed is None else int(rng_seed)
    rng = np.random.default_rng(seed_used)

    # Centre under H0 (mean=0) before resampling.
    centred = diff - float(np.mean(diff))
    if scheme == "stationary":
        idx = _stationary_block_indices(n, n_resamples, float(L), rng)
    else:
        idx = _fixed_block_indices(n, n_resamples, L, rng)
    resamples = centred[idx]
    boot_means = resamples.mean(axis=1)

    observed = float(np.mean(diff))
    # Two-sided: count resamples whose |bootstrap mean| ≥ |observed|.
    extreme = int(np.sum(np.abs(boot_means) >= abs(observed)))
    p = (extreme + 1.0) / (n_resamples + 1.0)
    metadata: dict[str, float | int | str] = {
        "block_length": L,
        "n_resamples": int(n_resamples),
        "scheme": scheme,
        "rng_seed": seed_used,
    }
    return float(p), metadata
