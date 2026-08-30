"""Block-bootstrap primitives backing ``StationaryBootstrap`` and the period slice tests.

The stationary resampling scheme for dependent time series
([Politis-Romano (1994)][politis-romano-1994]) — geometric block lengths
with mean ``L``, circular sampling — plus the
[Politis-White (2004)][politis-white-2004] automatic block-length
selector and a paired-diff empirical p-value. Resamples are themselves
stationary processes, which is what downstream estimators relying on
stationarity (CI for serially-correlated means, Sharpe) need.

Only the stationary scheme is implemented. The fixed-length circular
scheme ([Politis-Romano (1992)][politis-romano-1992]) lived here for a
while with no public entry point able to select it, and no size table of
its own; factrix exposes measured paths only, so it was removed rather
than promoted to a knob.

The public ``factrix.stats.bootstrap`` module ships standalone
``stationary_bootstrap_resamples`` / ``bootstrap_mean_ci`` for callers
that want a CI utility outside the inference dispatch chain. This
private module is consumed by ``factrix.inference.StationaryBootstrap``
(``_block_bootstrap_diff_p``) and by the period-family slice tests
(``_stationary_block_indices`` / ``_politis_white_block_length``).

References:
    - Politis, D. N. & Romano, J. P. (1994). "The Stationary
      Bootstrap." Journal of the American Statistical Association,
      89(428), 1303–1313.
    - Politis, D. N. & White, H. (2004). "Automatic Block-Length
      Selection for the Dependent Bootstrap." Econometric Reviews,
      23(1), 53–70.
"""

from __future__ import annotations

import secrets
from numbers import Integral
from typing import Literal

import numpy as np

from factrix._types import EPSILON

#: Refusal floor for every entry point that exposes a user-settable resample
#: count *and* reports an inference drawn from it (``bootstrap_mean_ci``,
#: ``monotonicity``'s MR test, ``StationaryBootstrap``, the period slice
#: tests). A Davison-Hinkley empirical p lives on the ``1/(B+1)`` grid, so
#: ``B`` should be chosen with ``alpha * (B + 1)`` an integer
#: ([Davidson-MacKinnon (2000)][davidson-mackinnon-2000]): 199 / 399 / 999 for
#: the usual levels. 199 is the smallest such grid point, and below it the p
#: resolves no finer than 0.005 while its Monte-Carlo SE at ``p = 0.05`` is
#: ``sqrt(.05*.95/199)`` = 1.5pp — a p printed as 0.05 then straddles the
#: conventional threshold in both directions. The floor refuses the
#: indefensible; it does not certify 199. Politis-White (2004) recommend
#: >= 999 for two-sided 5% work, which is the default everywhere, and
#: ``p_value_mc_se`` is reported so the cost of a lower ``B`` stays visible.
#: ``stationary_bootstrap_resamples`` is deliberately outside the floor: it
#: returns draws, not an inference.
#: Deliberately not named ``MIN_*``: that prefix is reserved (FX003) for
#: sample-size floors on a data axis, and a resample count is an algorithm
#: knob, not an axis.
BOOTSTRAP_RESAMPLES_FLOOR: int = 199


def _check_n_resamples(n_resamples: int, *, func_name: str, docs_path: str) -> None:
    """Reject a resample count below ``BOOTSTRAP_RESAMPLES_FLOOR``.

    The one validator behind every entry point that turns a resample count
    into a reported p or interval, so the floor, the message and the
    exception type cannot drift apart. ``func_name`` / ``docs_path`` must
    name the *public* entry point the caller reached this through — the
    message is user-facing.
    """
    from factrix._errors import UserInputError

    if n_resamples < BOOTSTRAP_RESAMPLES_FLOOR:
        raise UserInputError(
            func_name=func_name,
            field="n_resamples",
            value=n_resamples,
            expected=(
                f"at least {BOOTSTRAP_RESAMPLES_FLOOR} resamples — below that "
                f"the empirical p / interval endpoints are resampling noise "
                f"(p resolves no finer than 1/(B+1) = 0.005)"
            ),
            docs_path=docs_path,
        )


#: Type of every ``seed`` knob in factrix. ``int`` reproduces a run exactly;
#: ``None`` lets the entry point draw one from system entropy and report it
#: back; a ``numpy.random.Generator`` hands the entry point a stream the
#: caller owns and advances (the direction numpy, scipy's ``rng=`` and arch
#: have converged on), which is what nested or large-scale simulation needs.
#: Lives here rather than in ``factrix._types`` because that module is the
#: numpy-free home of numerical constants and ``Literal`` option aliases.
Rng = int | np.random.Generator | None


def _resolve_rng(
    rng: Rng, *, func_name: str, docs_path: str
) -> tuple[np.random.Generator, int | None]:
    """Resolve an ``rng`` knob into ``(generator, reported_seed)``.

    The single place factrix builds a ``Generator``, so the three-way
    contract cannot drift between entry points:

    - ``None`` — draw a 32-bit int from system entropy, build a
      ``default_rng`` on it and report the int, so an unseeded run stays
      reproducible after the fact.
    - ``int`` — ``default_rng(int)``, reported unchanged.
    - ``Generator`` — used as-is and reported as ``None``: the caller owns
      the stream, so only the caller can reproduce the draw, and the
      generator is *advanced* by the call.

    ``secrets.randbits(32)`` is the purpose-built "give me a random int
    seed" call — ``SeedSequence().entropy`` is typed as
    ``int | Sequence[int] | None`` and the Sequence branch breaks the
    bit-mask path.

    ``func_name`` / ``docs_path`` must name the *public* entry point the
    caller reached this through — the rejection message is user-facing.
    """
    from factrix._errors import UserInputError

    if isinstance(rng, np.random.Generator):
        return rng, None
    if rng is None:
        resolved = secrets.randbits(32)
        return np.random.default_rng(resolved), resolved
    if isinstance(rng, bool) or not isinstance(rng, Integral):
        raise UserInputError(
            func_name=func_name,
            field="rng",
            value=rng,
            expected=(
                "an int (reproduces the run), None (one is drawn and "
                "reported back), or a numpy.random.Generator (a stream the "
                "caller owns and advances)"
            ),
            docs_path=docs_path,
        )
    resolved = int(rng)
    return np.random.default_rng(resolved), resolved


def _empirical_p(n_extreme: int, n_resamples: int) -> tuple[float, float]:
    """Davison-Hinkley smoothed empirical p and its Monte-Carlo SE.

    ``p = (n_extreme + 1) / (B + 1)`` — the ``+1`` smoothing keeps the p
    strictly positive, so log-scale plots and downstream multi-stage
    adjustments never see a hard zero, and it is the unbiased form under the
    null (Davison-Hinkley smoothing).

    The second element is ``sqrt(p * (1 - p) / B)``, the binomial SE of the
    resampling draw itself — *not* a statistical SE of the estimate. It says
    how much the reported p would move on a re-run with a different seed,
    which is the quantity a reader needs when a p sits near a decision
    threshold: at ``B = 1000`` and ``p = 0.05`` it is ~0.7pp, so 0.043 and
    0.058 are one draw apart. It shrinks as ``1 / sqrt(B)``; raise ``B`` to
    shrink it, since no amount of data does.

    Args:
        n_extreme: Count of resamples at least as extreme as the observed
            statistic.
        n_resamples: ``B``, the number of resamples the count is out of.

    Returns:
        ``(p_value, p_value_mc_se)``. Callers are gated entry points, so
        ``B >= BOOTSTRAP_RESAMPLES_FLOOR`` and ``p`` is confined to
        ``[1/(B+1), 1]``; no clamping or zero-guard is needed, and none is
        done.
    """
    p = (n_extreme + 1.0) / (n_resamples + 1.0)
    return float(p), float(np.sqrt(p * (1.0 - p) / n_resamples))


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


def _politis_white_block_length(values: np.ndarray) -> float:
    """[Politis-White (2004)][politis-white-2004] automatic block length.

    Implements the spectral plug-in described in PW §3-4:
    ``L̂ = (2 Ĝ² / D̂)^(1/3) · T^(1/3)`` where ``Ĝ`` and ``D̂`` are
    flat-top kernel estimates of, respectively, the first derivative
    and variance of the spectral density at frequency 0, with the
    stationary scheme's ``D̂ = 2 g(0)²`` (PW eq 9).

    Falls back to ``max(1, 1.75 · T^(1/3))`` (the widely-cited practical
    PW approximation, also used by ``factrix.stats.bootstrap``) when
    the series is too short, autocovariance is degenerate, or the
    spectral estimate yields a non-finite ratio. Returns a ``float``;
    callers that need an integer block size round at the call site.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    fallback = max(1.0, 1.75 * n ** (1.0 / 3.0)) if n >= 1 else 1.0
    if n < 4 or not np.all(np.isfinite(x)):
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
    # Politis-White section 4 and Patton's reference ``opt_block_length``
    # use sqrt(log10(T)), not log10(T). The two agree at 5 for every n below
    # 1e5 and diverge above it; the paper's form is the one implemented.
    K_T = max(5, int(np.ceil(np.sqrt(np.log10(n)))))
    threshold = 2.0 * np.sqrt(np.log10(n) / n)
    m_pick = None
    for m in range(0, max(k_max - K_T + 1, 1)):
        window = rho[m + 1 : m + 1 + K_T]
        if window.size and np.all(np.abs(window) < threshold):
            m_pick = m
            break
    if m_pick is None:
        # No insignificant run found (strongly persistent series): Patton's
        # reference code / ``arch`` take the largest significant lag as m̂.
        sig = np.nonzero(np.abs(rho[1:]) >= threshold)[0]
        if sig.size == 0:
            return fallback
        m_pick = int(sig[-1]) + 1
    # PW (2004) §4 doubles the chosen index for the kernel bandwidth.
    M = max(2 * m_pick, 2)
    M = min(M, k_max)

    gamma = rho * gamma_0
    # PW (2004) eq. 9 sums |k| ≤ M with the flat-top kernel; the k=M
    # term vanishes because λ(M/M) = λ(1) = 0 by construction, so
    # `range(1, M)` covers every non-zero contributor.
    g0 = gamma[0] + 2.0 * sum(_flat_top_kernel(k / M) * gamma[k] for k in range(1, M))
    g_deriv = 2.0 * sum(_flat_top_kernel(k / M) * k * gamma[k] for k in range(1, M))
    d_hat = 2.0 * g0 * g0

    if d_hat < EPSILON or not np.isfinite(g_deriv):
        return fallback
    # ``ratio == 0`` means the estimated spectral derivative Ghat is 0, i.e.
    # NO detectable dependence, and ``L = 0 * T^(1/3) = 0`` falls through to
    # the ``max(L, 1.0)`` clamp below for the right answer: one observation
    # per block. An earlier non-positive-ratio early return sent that case
    # to the generic ``1.75 * T^(1/3)`` rule instead - blocks an order of
    # magnitude too long for a series with no dependence at all, and in
    # direct contradiction with the clamp's own reasoning. That early return
    # is gone. ``ratio`` cannot
    # be negative (it is ``2 * g_deriv**2 / d_hat`` with ``d_hat > EPSILON``),
    # so only the non-finite guard is left.
    ratio = (2.0 * g_deriv * g_deriv) / d_hat
    if not np.isfinite(ratio):
        return fallback
    L = (ratio ** (1.0 / 3.0)) * (n ** (1.0 / 3.0))
    if not np.isfinite(L):
        return fallback
    # A plug-in estimate below one block is a *valid* answer (no usable
    # dependence — the iid bootstrap is right), not a degenerate one: clamp
    # to 1 as ``arch`` / Patton do. Substituting the generic
    # ``1.75·T^(1/3)`` rule here (an earlier factrix behaviour) replaced a
    # data-driven L≈1 with blocks an order of magnitude longer on ~40% of
    # iid series and inflated the bootstrap test's size (8.7% at nominal
    # 5% on T=120).
    #
    # The upper bound is ``arch``'s ``b_max = ceil(min(3·√n, n/3))``
    # (``arch.bootstrap._single_optimal_block``, verified against the
    # source), replacing an earlier looser ``n / 2``. Bounding L relative
    # to n is what keeps enough effective blocks (~n/L) for the resample
    # variance to mean anything: at L = n/2 a resample is ~2 blocks and
    # the empirical p is built on coin flips. ``3·√n`` is the binding
    # branch below n = 81 and ``n/3`` above it; at n = 120 the bound
    # tightens from 60 to 33.
    #
    # Measured against the old ``n/2`` over 500 series per cell, the
    # resolved L changes on 6.0% of iid n=20 series, 2.0% of AR(0.6)
    # n=20, and under 1% by n=120, reaching 0% by n=300. Note the
    # direction: the bound bites hardest on SHORT, LOW-persistence
    # series, not on persistent ones — the plug-in estimate is noisiest
    # where there is least dependence to estimate, which is the same
    # failure the lower clamp above addresses from the other side.
    # Strongly persistent series (AR(0.95), random walk) sit inside the
    # bound almost always.
    return float(min(max(L, 1.0), float(_max_block_length(n))))


def _max_block_length(n: int) -> int:
    """Largest admissible block length for a series of ``n`` observations.

    ``ceil(min(3 * sqrt(n), n / 3))`` — ``arch``'s ``b_max``
    (``arch.bootstrap._single_optimal_block``). Bounding ``L`` relative to
    ``n`` is what keeps enough effective blocks (``~n/L``) for the resample
    distribution to carry information: at ``L >= n`` the circular resample
    degenerates to a rotation of the whole series, every resample is
    a permutation of the same values, and the centred bootstrap mean is
    identically zero — so the empirical p is ``1/(B+1)`` on any data
    whatsoever, the strongest possible significance from a parameter typo.
    """
    return max(1, int(np.ceil(min(3.0 * np.sqrt(n), n / 3.0))))


def _validate_block_length(
    block_length: float, n: int, func_name: str, docs_path: str
) -> None:
    """Reject a block length too long for the sample. Raises ``UserInputError``.

    ``func_name`` / ``docs_path`` must name the *public* entry point the
    caller reached this through (``StationaryBootstrap``,
    ``stationary_bootstrap_resamples``), not the private frame that happens
    to call it — the message is user-facing.

    ``L >= 1`` alone is not enough: see :func:`_max_block_length` for the
    degeneracy at ``L >= n``. This is a refusal rather than a silent clamp
    because an out-of-range explicit ``L`` is a validation-parameter mistake,
    and clamping it would return a number under a block length the caller
    never asked for.
    """
    from factrix._errors import UserInputError

    b_max = _max_block_length(n)
    if block_length < 1.0 or block_length > b_max:
        raise UserInputError(
            func_name=func_name,
            field="block_length",
            value=block_length,
            expected=(
                f"a block length in [1, {b_max}] for a series of {n} "
                f"observations (ceil(min(3*sqrt(n), n/3)))"
            ),
            docs_path=docs_path,
        )


def _batch_means_se(values: np.ndarray, block_length: float) -> np.ndarray:
    r"""Block (batch-means) standard error of the mean, vectorised over rows.

    Splits each row into ``floor(n / L)`` contiguous non-overlapping batches
    of length ``L``, and takes the standard error of the batch means:

        $$\widehat{se} = \sqrt{\operatorname{Var}(\bar x_b) / n_b}$$

    This is the block-based variance estimator the bootstrap-t root in
    :func:`_block_bootstrap_diff_p` studentizes by. Using the *same*
    estimator, at the *same* ``L``, on the observed series and on every
    resample is what makes the root asymptotically pivotal
    ([Götze-Künsch (1996)][gotze-kunsch-1996], [Lahiri (2003)][lahiri-2003]).

    Args:
        values: ``(n,)`` series or ``(B, n)`` stack of resamples.
        block_length: ``L``; rounded to the nearest integer >= 1.

    Returns:
        ``(B,)`` array of standard errors (length 1 for vector input). NaN
        where the row admits fewer than two batches — the caller falls back
        to the unstudentized root there rather than divide by noise.
    """
    rows = np.atleast_2d(np.asarray(values, dtype=float))
    n = rows.shape[1]
    L = max(round(block_length), 1)
    n_batches = n // L
    if n_batches < 2:
        return np.full(rows.shape[0], float("nan"))
    trimmed = rows[:, : n_batches * L].reshape(rows.shape[0], n_batches, L)
    batch_means = trimmed.mean(axis=2)
    return np.sqrt(batch_means.var(axis=1, ddof=1) / n_batches)


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
    if n == 0:
        return np.empty((n_resamples, 0), dtype=np.int64)
    _validate_block_length(
        mean_block_length,
        n,
        "stationary_bootstrap_resamples",
        "api/stats#factrix.stats.stationary_bootstrap_resamples",
    )
    p_new = 1.0 / mean_block_length
    starts = rng.integers(0, n, size=(n_resamples, n))
    new_block = rng.random(size=(n_resamples, n)) < p_new
    idx = np.empty((n_resamples, n), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, n):
        prev = (idx[:, t - 1] + 1) % n
        idx[:, t] = np.where(new_block[:, t], starts[:, t], prev)
    return idx


def _block_bootstrap_diff_p(
    diff: np.ndarray,
    *,
    block_length: int | Literal["auto"] = "auto",
    n_resamples: int = 999,
    overlap_periods: int | None = None,
    rng: Rng = None,
) -> tuple[float, dict[str, float | int | str | None]]:
    r"""Two-sided empirical p for ``H₀: E[diff] = 0`` on a paired series.

    Resamples ``diff`` under the centring ``diff - mean(diff)`` (the
    null restricts the mean to zero; the bootstrap distribution must be
    drawn under that restriction to give a calibrated p). Empirical p
    uses Davison-Hinkley ``+1 / (B+1)`` smoothing: keeps p strictly
    positive so log-scale plots and downstream multi-stage adjustments
    don't see a hard zero.

    **Studentized (bootstrap-t) root.** The compared quantity is

        $$t^* = \frac{\bar x^*}{\widehat{se}^*},\qquad
          t = \frac{\bar x}{\widehat{se}}$$

    with $\widehat{se}$ the batch-means block SE (:func:`_batch_means_se`)
    at the resolved block length, computed the same way on the observed
    series and on every resample. This is not decoration: the block
    bootstrap attains an asymptotic refinement over the normal
    approximation *only* for a studentized root
    ([Götze-Künsch (1996)][gotze-kunsch-1996],
    [Lahiri (2003)][lahiri-2003]). A percentile bootstrap of the raw mean
    is first-order only and inherits the block bootstrap's finite-sample
    long-run-variance shortfall, which is exactly the dependence it is
    reached for.

    Measured rejection frequency on an AR(1) null at nominal 5%
    (B=999, ``block_length="auto"``, 400 sims per cell):

    | n   | φ=0.0 | φ=0.5 | φ=0.8 |
    |-----|-------|-------|-------|
    | 30  | 0.075 | 0.152 | 0.265 |
    | 60  | 0.055 | 0.115 | 0.133 |
    | 120 | 0.068 | 0.095 | 0.107 |
    | 500 | 0.048 | 0.050 | 0.050 |

    The unstudentized root measured 0.095 / 0.228 / 0.400 at n=30 and
    0.048 / 0.058 / 0.098 at n=500. Studentizing roughly halves the
    excess but does not remove it at short, strongly persistent samples;
    callers holding a result object flag that regime with
    ``WarningCode.SERIAL_CORRELATION_DETECTED``.

    Args:
        diff: 1-D paired-difference series (already date-aligned by
            caller — the bootstrap does not re-align).
        block_length: ``"auto"`` runs Politis-White; an explicit ``int``
            is used unchanged. Either way the resolved value must land in
            ``[1, ceil(min(3*sqrt(n), n/3))]`` or ``UserInputError`` is
            raised — see :func:`_validate_block_length`. The resolved value
            is the *mean* of the geometric block-length distribution and
            stays fractional.
        n_resamples: ``B``. [Politis-White (2004)][politis-white-2004] recommends ≥ 999 for two-sided
            5% tests; default matches. Validated against
            ``BOOTSTRAP_RESAMPLES_FLOOR`` by the public entry point, not
            here.
        overlap_periods: Overlap horizon ``h`` of the series. When set,
            floors the resolved block length at ``h``: Politis-Romano
            validity needs the mean block length to dominate the
            dependence horizon, and with a *known* MA(h-1) structure the
            horizon is known exactly while the Politis-White plug-in has
            to rediscover it from a short noisy sample and systematically
            under-shoots (measured mean L of 7.95 against a needed 21 at
            T=60, h=21). Mirrors the HAC paths' bandwidth floor.
        rng: ``None`` draws from system entropy and the resolved int is
            returned in the metadata dict so the caller can record it; an
            ``int`` is reported unchanged; a ``numpy.random.Generator`` is
            used as-is and advanced, and the metadata reports ``None``
            because only its owner can reproduce the draw.

    Returns:
        ``(p_value, metadata)`` — p in ``[1/(B+1), 1]``; metadata
        records the resolved (fractional) block length, ``n_resamples``,
        whether the root was studentized, the Monte-Carlo SE of the
        reported p (``p_value_mc_se``), and the resolved seed (so the run
        is reproducible from the logged metadata even when the caller
        passed ``rng=None``) — ``None`` when the caller supplied a
        ``Generator``.

        A series too short to test (``n < 2``) returns ``p = nan`` with
        ``block_length = nan`` and ``n_resamples = 0``. NaN rather than
        the former ``1.0``: a sample that admits no test is not evidence
        for the null, and the former sentinel also broke the documented
        ``block_length >= 1`` invariant for anyone reading the metadata.
    """
    diff = np.asarray(diff, dtype=float)
    # A NaN would make ``centred`` all-NaN, every ``|boot| >= |obs|`` test
    # False and the reported p collapse to ``1 / (B + 1)`` — the *strongest*
    # possible significance from a series that carries no information.
    if diff.size and not np.all(np.isfinite(diff)):
        raise ValueError("_block_bootstrap_diff_p: diff must be finite (no NaN / inf).")
    n = len(diff)
    # Resolve up front so the degenerate path reports the same resolved seed
    # the testable path would, and a bogus seed is refused either way.
    rng, seed_used = _resolve_rng(
        rng, func_name="StationaryBootstrap", docs_path="reference/statistical-methods"
    )
    if n < 2:
        return float("nan"), {
            "block_length": float("nan"),
            "n_resamples": 0,
            "studentized": False,
            "p_value_mc_se": float("nan"),
            "seed": seed_used,
        }

    L: float
    if block_length == "auto":
        L_auto = _politis_white_block_length(diff)
        # L stays fractional: it is the MEAN of a geometric distribution —
        # ``_stationary_block_indices`` only ever reads it as
        # ``p_new = 1 / L`` — so rounding discretizes the renewal probability
        # for nothing: L=3.4 turns p_new 0.294 into 0.333. It would also put
        # this function out of step with the other two consumers of the same
        # estimate, ``stationary_bootstrap_resamples`` and
        # ``slicing.period_inference``, which both pass the float through.
        L = max(1.0, L_auto)
    else:
        L = float(int(block_length))

    # Floor at the KNOWN overlap horizon before validating. The plug-in
    # estimates the dependence horizon from the sample; when the caller
    # already knows it exactly there is no reason to accept a shorter block.
    if overlap_periods is not None and overlap_periods > 1:
        L = max(L, float(min(overlap_periods, _max_block_length(n))))
    _validate_block_length(L, n, "StationaryBootstrap", "reference/statistical-methods")

    # Centre under H0 (mean=0) before resampling.
    centred = diff - float(np.mean(diff))
    idx = _stationary_block_indices(n, n_resamples, L, rng)
    resamples = centred[idx]
    boot_means = resamples.mean(axis=1)
    observed = float(np.mean(diff))

    # Bootstrap-t: studentize both sides by the same block SE estimator at
    # the same L. Fall back to the raw-mean root only when the series is too
    # short to form two batches (n < 2L), where there is no block SE to
    # divide by; that is the one case where the unstudentized comparison is
    # the better of two bad options rather than a silent downgrade.
    se_observed = float(_batch_means_se(diff, L)[0])
    studentized = np.isfinite(se_observed) and se_observed > EPSILON
    if studentized:
        se_boot = _batch_means_se(resamples, L)
        usable = np.isfinite(se_boot) & (se_boot > EPSILON)
        boot_t = np.divide(
            boot_means, se_boot, out=np.zeros_like(boot_means), where=usable
        )
        observed_t = observed / se_observed
        extreme = int(np.sum(np.abs(boot_t[usable]) >= abs(observed_t)))
        n_used = int(usable.sum())
    else:
        # Two-sided: count resamples whose |bootstrap mean| ≥ |observed|.
        extreme = int(np.sum(np.abs(boot_means) >= abs(observed)))
        n_used = int(n_resamples)

    p, p_mc_se = _empirical_p(extreme, n_used)
    metadata: dict[str, float | int | str | None] = {
        "block_length": L,
        "n_resamples": int(n_resamples),
        "studentized": studentized,
        "p_value_mc_se": p_mc_se,
        "seed": seed_used,
    }
    return float(p), metadata
