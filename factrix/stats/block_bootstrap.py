"""``BlockBootstrap`` Estimator — empirical p via dependent-block resampling (#153).

Names the block-bootstrap inference path for the paired-diff slice
test. Numerics — Politis-Romano stationary scheme, Künsch fixed
scheme, Politis-White automatic block length — live in
``factrix._stats.bootstrap``; this module is the dispatch handle
exposed to family functions / slice-test functions (#176).

Public ``factrix.stats.bootstrap`` standalone helpers
(``stationary_bootstrap_resamples`` / ``bootstrap_mean_ci``) remain
in place for callers wanting a CI utility outside the Estimator
dispatch chain.
"""

from __future__ import annotations

from typing import Literal

from factrix._axis import FactorScope, FactorSignal, Metric
from factrix._codes import StatCode


class BlockBootstrap:
    """Block-bootstrap empirical p-value Estimator for paired-diff slice tests.

    Resamples a paired-difference series under H₀: ``E[diff] = 0`` using
    one of two dependent-bootstrap schemes:

    - ``"stationary"`` ([Politis-Romano (1994)][politis-romano-1994]) —
      geometric block lengths with mean ``L``; each resample is itself a
      stationary process. Default; preferred when downstream stats (CI,
      Sharpe) rely on stationarity.
    - ``"fixed"`` ([Künsch (1989)][kunsch-1989]) — deterministic block
      length ``L``; cleaner for variance estimation; loses stationarity
      at block joins but tighter at small ``B``.

    Block length resolves automatically from the input series via
    [Politis-White (2004)][politis-white-2004] when ``block_length="auto"``;
    pass an integer
    to fix it.

    Applicability is restricted to ``(INDIVIDUAL, CONTINUOUS)`` —
    consistent with the slice-test functions that produce paired per-date
    diffs (slice information coefficient (IC), slice FM-λ, …).

    Constructor parameters are stored on the instance and read by the
    slice-test function procedure when it calls
    ``factrix._stats.bootstrap._block_bootstrap_diff_p``. Two
    ``BlockBootstrap`` instances with different ``scheme`` / block
    length are distinct Estimators from the function's perspective; the
    StatCode emitted (``P_BOOT``) does not split on scheme — scheme is
    metadata, not a separate code (parallel to how ``NeweyWest``'s lag
    rule lives in ``metadata`` rather than splitting ``P_NW``).
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
        if n_resamples < 1:
            raise ValueError(f"n_resamples must be >= 1; got {n_resamples!r}.")
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

    def applicable_to(self, scope: FactorScope, signal: FactorSignal) -> bool:
        return scope is FactorScope.INDIVIDUAL and signal is FactorSignal.CONTINUOUS

    def emits_for(
        self,
        _scope: FactorScope,
        _signal: FactorSignal,
        _metric: Metric | None,
    ) -> StatCode:
        return StatCode.P_BOOT
