"""Selection-bias measures for a strategy chosen out of many backtested trials.

An automated search that tries ``N`` strategy variants and keeps the best one
reports a Sharpe ratio inflated by the selection itself. Two published
measures quantify that inflation:

- ``deflated_sharpe_ratio`` — [Bailey & López de Prado (2014)][bailey-lopez-de-prado-2014].
  The probability that the selected strategy's true Sharpe exceeds the
  expected maximum Sharpe of ``N`` zero-skill trials, corrected for sample
  length, skewness and kurtosis.
- ``probability_of_backtest_overfitting`` — combinatorially symmetric
  cross-validation (CSCV) of
  [Bailey, Borwein, López de Prado & Zhu (2017)][bailey-borwein-lopez-de-prado-zhu-2017].
  The share of in-sample/out-of-sample splits in which the in-sample best
  trial ranks at or below the out-of-sample median.

Both work in per-period units on the caller's own return grid; neither
annualizes or reads a calendar.
"""

from __future__ import annotations

import math
from itertools import combinations, islice
from numbers import Integral, Real

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata

from factrix._errors import UserInputError

_DSR_DOCS = "api/stats#factrix.stats.deflated_sharpe_ratio"
_PBO_DOCS = "api/stats#factrix.stats.probability_of_backtest_overfitting"

#: CSCV splits evaluated per vectorised batch; bounds peak memory at
#: ``_CSCV_BATCH * N`` floats per intermediate array.
_CSCV_BATCH = 1024


def _finite_real(value: object, *, func_name: str, field: str, docs: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise UserInputError(
            func_name=func_name,
            field=field,
            value=value,
            expected="a finite real number",
            docs_path=docs,
        )
    out = float(value)
    if not math.isfinite(out):
        raise UserInputError(
            func_name=func_name,
            field=field,
            value=value,
            expected="a finite real number",
            docs_path=docs,
        )
    return out


def _int_at_least(
    value: object, minimum: int, *, func_name: str, field: str, docs: str
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise UserInputError(
            func_name=func_name,
            field=field,
            value=value,
            expected=f"an integer >= {minimum}",
            docs_path=docs,
        )
    return int(value)


def deflated_sharpe_ratio(
    sharpe: float,
    *,
    n_obs: int,
    n_trials: int,
    sharpe_variance: float,
    skew: float,
    kurtosis: float,
) -> float:
    """[Bailey & López de Prado (2014)][bailey-lopez-de-prado-2014] Deflated Sharpe Ratio (DSR).

    The DSR is the Probabilistic Sharpe Ratio evaluated against the Sharpe
    that the best of ``N`` zero-skill trials is expected to reach:

    ``DSR = Φ( (SR - SR_0) * sqrt(T - 1) / sqrt(1 - γ3 * SR + (γ4 - 1) / 4 * SR²) )``

    ``SR_0 = sqrt(V) * ((1 - γ) * Φ⁻¹(1 - 1/N) + γ * Φ⁻¹(1 - 1/(N e)))``

    where ``SR`` is the selected trial's estimated Sharpe ratio, ``T`` the
    number of returns it was estimated from, ``γ3`` / ``γ4`` the skewness and
    (non-excess) kurtosis of those returns, ``V`` the variance of the
    estimated Sharpe ratios across the trials, ``N`` the number of independent
    trials, ``γ`` the Euler-Mascheroni constant and ``Φ`` the standard normal
    CDF. ``SR_0`` is the paper's approximation of the expected maximum of
    ``N`` draws from ``N(0, V)``.

    Args:
        sharpe: Estimated Sharpe ratio of the selected trial, per period —
            mean over standard deviation of its per-period returns, not
            annualized.
        n_obs: Number of return periods ``T`` behind ``sharpe``. At least 2.
        n_trials: Number of independent trials ``N`` the selection was made
            from. At least 2; ``SR_0`` diverges at ``N = 1``.
        sharpe_variance: Variance ``V`` of the per-period Sharpe estimates
            across the trials, in the same units as ``sharpe``. Non-negative.
        skew: Skewness ``γ3`` of the selected trial's returns.
        kurtosis: Kurtosis ``γ4`` of the selected trial's returns on the
            non-excess scale (3 for a normal distribution). Must satisfy
            ``kurtosis >= 1 + skew²``, which every distribution does.

    Returns:
        The DSR, a probability in ``[0, 1]``. Values near 1 say the selected
        Sharpe is unlikely to be the product of the search alone.

    Raises:
        UserInputError: a non-finite or non-numeric input, ``n_obs < 2``,
            ``n_trials < 2``, ``sharpe_variance < 0``, or a skew/kurtosis
            pair no distribution can have.

    Notes:
        Every quantity is per period on the returns' own grid; mixing an
        annualized Sharpe with a per-period ``sharpe_variance`` or ``n_obs``
        gives a meaningless result. Convert both consistently, as the
        paper's numerical example does.

        ``n_trials`` counts *independent* trials. Highly correlated variants
        overstate it and over-deflate; the paper leaves the estimate of the
        effective trial count to the caller.

        Raising ``n_trials`` with everything else fixed raises ``SR_0`` and
        lowers the DSR: more trials demand a higher Sharpe for the same
        confidence.

    References:
        - [Bailey & López de Prado (2014)][bailey-lopez-de-prado-2014]. "The
          Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
          Overfitting, and Non-Normality." Journal of Portfolio Management,
          40(5), 94-107. Equations for ``DSR`` and ``SR_0`` as implemented
          here; its numerical example (``T = 1250``, annualized ``SR = 2.5``
          and ``V = 0.5`` over 250 periods a year, ``N = 100``, ``γ3 = -3``,
          ``γ4 = 10``) gives ``DSR ≈ 0.9004``.

    Examples:
        >>> from factrix.stats import deflated_sharpe_ratio
        >>> round(
        ...     deflated_sharpe_ratio(
        ...         2.5 / 250**0.5,
        ...         n_obs=1250,
        ...         n_trials=100,
        ...         sharpe_variance=0.5 / 250,
        ...         skew=-3.0,
        ...         kurtosis=10.0,
        ...     ),
        ...     4,
        ... )
        0.9004
    """
    func = "deflated_sharpe_ratio"
    sr = _finite_real(sharpe, func_name=func, field="sharpe", docs=_DSR_DOCS)
    t = _int_at_least(n_obs, 2, func_name=func, field="n_obs", docs=_DSR_DOCS)
    n = _int_at_least(n_trials, 2, func_name=func, field="n_trials", docs=_DSR_DOCS)
    var = _finite_real(
        sharpe_variance, func_name=func, field="sharpe_variance", docs=_DSR_DOCS
    )
    if var < 0:
        raise UserInputError(
            func_name=func,
            field="sharpe_variance",
            value=sharpe_variance,
            expected="a non-negative variance",
            docs_path=_DSR_DOCS,
        )
    g3 = _finite_real(skew, func_name=func, field="skew", docs=_DSR_DOCS)
    g4 = _finite_real(kurtosis, func_name=func, field="kurtosis", docs=_DSR_DOCS)
    if g4 < 1.0 + g3 * g3:
        raise UserInputError(
            func_name=func,
            field="kurtosis",
            value=kurtosis,
            expected=(
                f"non-excess kurtosis >= 1 + skew² ({1.0 + g3 * g3:g}); no "
                f"distribution has less. Pass kurtosis on the non-excess scale "
                f"(3 for a normal distribution)"
            ),
            docs_path=_DSR_DOCS,
        )
    # kurtosis >= 1 + skew² makes this >= (1 - skew * SR / 2)² >= 0; it is
    # zero only for a two-point distribution at one specific Sharpe.
    sr_var_scale = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr
    if sr_var_scale <= 0:
        raise UserInputError(
            func_name=func,
            field="kurtosis",
            value=kurtosis,
            expected=(
                "a skew/kurtosis/sharpe combination with a positive Sharpe "
                "estimator variance; this one describes a two-point "
                "distribution whose Sharpe has zero variance"
            ),
            docs_path=_DSR_DOCS,
        )
    sr0 = math.sqrt(var) * (
        (1.0 - np.euler_gamma) * float(ndtri(1.0 - 1.0 / n))
        + np.euler_gamma * float(ndtri(1.0 - 1.0 / (n * math.e)))
    )
    z = (sr - sr0) * math.sqrt(t - 1) / math.sqrt(sr_var_scale)
    return float(ndtr(z))


def probability_of_backtest_overfitting(
    trial_returns: npt.ArrayLike,
    *,
    n_splits: int = 16,
) -> float:
    """Probability of Backtest Overfitting (PBO) by CSCV.

    Combinatorially symmetric cross-validation (CSCV) from
    [Bailey, Borwein, López de Prado & Zhu (2017)][bailey-borwein-lopez-de-prado-zhu-2017]:

    1. Split the ``T`` rows of the ``T × N`` return matrix, in order, into
       ``S`` contiguous blocks of ``T / S`` rows.
    2. For each of the ``C(S, S/2)`` ways to pick ``S/2`` blocks, the picked
       blocks form the in-sample set and the rest the out-of-sample set.
    3. Compute every trial's Sharpe ratio on both sets, take the in-sample
       best trial ``n*``, and its out-of-sample relative rank
       ``ω = rank(n*) / (N + 1)`` (rank 1 = worst).
    4. ``λ = log(ω / (1 - ω))``. The PBO is the share of splits with
       ``λ <= 0`` — the in-sample winner ranking at or below the
       out-of-sample median.

    Args:
        trial_returns: Finite ``(T, N)`` matrix of per-period returns, one
            row per period in time order and one column per trial. ``N >= 2``.
        n_splits: Number of blocks ``S``. Even, ``>= 2``, and dividing ``T``
            exactly. The default 16 is the paper's choice and evaluates
            12,870 splits; the split count grows as ``C(S, S/2)``.

    Returns:
        The PBO, a share in ``[0, 1]``. Near 0.5 when no trial carries real
        signal; near 0 when the in-sample winner reliably stays above the
        out-of-sample median.

    Raises:
        UserInputError: ``trial_returns`` not 2-D or non-finite, fewer than 2
            trials, an odd or non-integer ``n_splits`` below 2, a ``T`` that
            ``n_splits`` does not divide, halves of fewer than 2 periods, or
            a trial with zero variance on some half (its Sharpe is undefined).

    Notes:
        ``T`` must divide into equal blocks as the paper specifies; trim rows
        yourself rather than have them dropped silently.

        Out-of-sample ranks of tied Sharpe ratios are averaged; a tie for the
        in-sample best goes to the lowest column index. Both only matter for
        exactly equal Sharpe ratios.

        Each half holds ``T / 2`` periods, so every Sharpe ratio is estimated
        on half the sample.

        Run time is proportional to ``C(S, S/2) * N``: under half a second for
        ``S = 20`` and ``N = 20``, but ``C(30, 15)`` is 1.6e8 splits. Nothing
        caps ``S``; keep it near the default.

    References:
        - [Bailey, Borwein, López de Prado & Zhu (2017)][bailey-borwein-lopez-de-prado-zhu-2017].
          "The Probability of Backtest Overfitting." Journal of
          Computational Finance, 20(4), 39-69. The CSCV procedure and PBO
          definition implemented here.
    """
    func = "probability_of_backtest_overfitting"
    m = np.asarray(trial_returns, dtype=float)
    if m.ndim != 2:
        raise UserInputError(
            func_name=func,
            field="trial_returns",
            value=m.shape,
            expected="a 2-D (T, N) array: one row per period, one column per trial",
            docs_path=_PBO_DOCS,
        )
    t, n = m.shape
    if n < 2:
        raise UserInputError(
            func_name=func,
            field="trial_returns",
            value=m.shape,
            expected="at least 2 trials (columns)",
            docs_path=_PBO_DOCS,
        )
    if not np.all(np.isfinite(m)):
        raise UserInputError(
            func_name=func,
            field="trial_returns",
            value=f"{int(np.sum(~np.isfinite(m)))} non-finite entries",
            expected="every return finite",
            docs_path=_PBO_DOCS,
        )
    s = _int_at_least(n_splits, 2, func_name=func, field="n_splits", docs=_PBO_DOCS)
    if s % 2:
        raise UserInputError(
            func_name=func,
            field="n_splits",
            value=n_splits,
            expected="an even integer, so the blocks split into two equal halves",
            docs_path=_PBO_DOCS,
        )
    if t % s:
        raise UserInputError(
            func_name=func,
            field="n_splits",
            value=n_splits,
            expected=(
                f"a divisor of the {t} periods, so every block has the same "
                f"length; trim trial_returns to a multiple of n_splits"
            ),
            docs_path=_PBO_DOCS,
        )
    if t // 2 < 2:
        raise UserInputError(
            func_name=func,
            field="trial_returns",
            value=m.shape,
            expected="at least 4 periods, so each half has 2 for a Sharpe ratio",
            docs_path=_PBO_DOCS,
        )

    # A half is constant exactly when S/2 of its blocks are constant at one
    # shared value. Checked on the raw values, because the variance computed
    # from sums below cancels to a small non-zero number instead of 0.
    raw_blocks = m.reshape(s, t // s, n)
    block_min = raw_blocks.min(axis=1)
    const_block = block_min == raw_blocks.max(axis=1)
    for j in np.flatnonzero(const_block.sum(axis=0) >= s // 2):
        _, counts = np.unique(block_min[const_block[:, j], j], return_counts=True)
        if counts.max() >= s // 2:
            raise UserInputError(
                func_name=func,
                field="trial_returns",
                value=f"trial (column) {int(j)}",
                expected=(
                    "every trial with non-zero return variance on every half; "
                    f"this one is constant on {int(counts.max())} of "
                    f"{s} blocks, so some half has no Sharpe ratio"
                ),
                docs_path=_PBO_DOCS,
            )

    # Per-block sums and sums of squares give every split's Sharpe ratios
    # without materialising the concatenated halves. Shifting each column by
    # its first value limits cancellation in the variance.
    shift = m[0]
    blocks = (m - shift).reshape(s, t // s, n)
    block_sum = blocks.sum(axis=1)
    block_sq = np.square(blocks).sum(axis=1)
    total_sum = block_sum.sum(axis=0)
    total_sq = block_sq.sum(axis=0)
    half = t // 2

    def _sharpe(sum_: np.ndarray, sq: np.ndarray) -> np.ndarray:
        mean = sum_ / half
        var = np.maximum(sq - half * mean * mean, 0.0) / (half - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (mean + shift) / np.sqrt(var)

    overfit = 0
    n_combos = 0
    split_iter = combinations(range(s), s // 2)
    while True:
        batch = list(islice(split_iter, _CSCV_BATCH))
        if not batch:
            break
        picks = np.zeros((len(batch), s))
        picks[np.arange(len(batch))[:, None], np.asarray(batch)] = 1.0
        is_sum = picks @ block_sum
        is_sq = picks @ block_sq
        sr_is = _sharpe(is_sum, is_sq)
        sr_oos = _sharpe(total_sum - is_sum, total_sq - is_sq)
        bad = ~(np.isfinite(sr_is).all(axis=1) & np.isfinite(sr_oos).all(axis=1))
        if bad.any():
            raise UserInputError(
                func_name=func,
                field="trial_returns",
                value=f"split of blocks {batch[int(np.argmax(bad))]}",
                expected=(
                    "every trial with return variance on every half large "
                    "enough to resolve in floating point"
                ),
                docs_path=_PBO_DOCS,
            )
        best = np.argmax(sr_is, axis=1)
        oos_rank = rankdata(sr_oos, axis=1)[np.arange(len(batch)), best]
        omega = oos_rank / (n + 1)
        logit = np.log(omega / (1.0 - omega))
        overfit += int(np.sum(logit <= 0))
        n_combos += len(batch)
    return overfit / n_combos
