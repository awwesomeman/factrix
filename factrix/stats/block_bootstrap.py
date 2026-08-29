"""``BlockBootstrap`` — empirical p via dependent-block resampling.

Names the block-bootstrap inference path for the paired-diff slice
test. Numerics — Politis-Romano stationary scheme, Politis-Romano
circular fixed-length scheme, Politis-White automatic block length —
live in
``factrix._stats.bootstrap``; this module is the dispatch handle
exposed to family functions / slice-test functions.

Public ``factrix.stats.bootstrap`` standalone helpers
(``stationary_bootstrap_resamples`` / ``bootstrap_mean_ci``) remain
in place for callers wanting a CI utility outside the Estimator
dispatch chain.
"""

from __future__ import annotations

from typing import Literal

from factrix._stats.bootstrap import BOOTSTRAP_RESAMPLES_FLOOR


class BlockBootstrap:
    """Block-bootstrap empirical p-value Estimator for paired-diff slice tests.

    Resamples a paired-difference series under H₀: ``E[diff] = 0`` using
    one of two dependent-bootstrap schemes:

    - ``"stationary"`` ([Politis-Romano (1994)][politis-romano-1994]) —
      geometric block lengths with mean ``L``; each resample is itself a
      stationary process. Default; preferred when downstream stats (CI,
      Sharpe) rely on stationarity.
    - ``"fixed"`` ([Politis-Romano (1992)][politis-romano-1992]) —
      deterministic block length ``L`` with circular wrap, so every
      observation is resampled with equal weight; cleaner for variance
      estimation; loses stationarity at block joins but tighter at small
      ``B``. This is the circular block bootstrap, not [Künsch
      (1989)][kunsch-1989]'s moving-block scheme — the name ``"fixed"``
      refers to the deterministic block *length*, not to Künsch.

    Block length resolves automatically from the input series via
    [Politis-White (2004)][politis-white-2004] when ``block_length="auto"``;
    pass an integer to fix it. Either way the resolved value must land in
    ``[1, ceil(min(3*sqrt(n), n/3))]``, checked against the *compute-time*
    series length and raising ``UserInputError`` otherwise. The bound is
    not decoration: at ``L >= n`` the circular fixed-block resample is a
    pure rotation of the whole series, so every resample is a permutation
    of the same values, the centred bootstrap mean is identically zero and
    the empirical p is ``1/(B+1)`` on any data at all. The constructor
    cannot enforce an ``n``-relative bound without ``n``, so it validates
    type and ``>= 1`` only and the real check fires on first use.

    The empirical p is computed on a **studentized (bootstrap-t) root**,
    not on the raw mean: the block bootstrap only attains its asymptotic
    refinement for a studentized statistic
    ([Götze-Künsch (1996)][gotze-kunsch-1996],
    [Lahiri (2003)][lahiri-2003]). See
    ``factrix._stats.bootstrap._block_bootstrap_diff_p`` for the measured
    size table.

    Applicability is restricted to ``(INDIVIDUAL, DENSE)`` —
    consistent with the slice-test functions that produce paired per-date
    diffs (slice information coefficient (IC), slice FM-λ, …).

    Constructor parameters are stored on the instance and read by the
    slice-test function procedure when it calls
    ``factrix._stats.bootstrap._block_bootstrap_diff_p``. Two
    ``BlockBootstrap`` instances with different ``scheme`` / block
    length are distinct Estimators from the function's perspective;
    scheme and block length live in the result ``metadata``.
    """

    def __init__(
        self,
        block_length: int | Literal["auto"] = "auto",
        n_resamples: int = 999,
        scheme: Literal["fixed", "stationary"] = "stationary",
        rng_seed: int | None = None,
    ) -> None:
        if block_length != "auto" and block_length < 1:
            raise ValueError(
                f"block_length must be 'auto' or int >= 1; got {block_length!r}."
            )
        if n_resamples < BOOTSTRAP_RESAMPLES_FLOOR:
            raise ValueError(
                f"n_resamples must be >= {BOOTSTRAP_RESAMPLES_FLOOR}; got "
                f"{n_resamples!r}. Below that the Davison-Hinkley empirical p "
                f"this estimator reports is dominated by resampling noise — the "
                f"same floor bootstrap_mean_ci and monotonicity enforce."
            )
        if scheme not in ("fixed", "stationary"):
            raise ValueError(f"scheme must be 'fixed' or 'stationary'; got {scheme!r}.")
        self.block_length = block_length
        self.n_resamples = n_resamples
        self.scheme = scheme
        self.rng_seed = rng_seed

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def description(self) -> str:
        bl = "auto" if self.block_length == "auto" else f"L={self.block_length}"
        return (
            f"Block-bootstrap empirical p-value on a paired-diff series "
            f"({self.scheme} scheme, {bl}, B={self.n_resamples})."
        )
