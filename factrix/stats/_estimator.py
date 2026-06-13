"""Estimator protocols — selection (base), HAC-compute.

``Estimator`` (base): family-function override selects which
already-computed p-value to feed step-up math. Carries no compute
logic — the procedure that populated ``FactorProfile.stats`` did the
math.

``HACEstimator(Estimator)``: adds ``compute(series, *,
forward_periods) -> InferenceResult`` for cell-internal estimator swap.
Cell procedures dispatch to ``cfg.estimator.compute(...)`` instead of
hardcoding the Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent (HAC) path.

Base ``Estimator`` is retained for slice-test instances
(``WaldNWCluster`` / ``BlockBootstrap``) whose compute
path is multivariate and lives outside the family-function axis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

# ``InferenceResult`` now lives in ``factrix.inference._base``; re-exported
# here so the (transitional) ``HACEstimator`` sub-protocol and existing
# ``factrix.stats.InferenceResult`` importers keep resolving.
from factrix.inference._base import InferenceResult

if TYPE_CHECKING:
    import numpy as np

    from factrix._axis import FactorDensity, FactorScope
    from factrix._codes import StatCode


@runtime_checkable
class Estimator(Protocol):
    """Inference-method instance: names the p-value source family functions select.

    Implementations supply identity (``name``), human-readable summary
    (``description``), cell-applicability check (``applicable_to``), and a
    cell → StatCode dispatch (``emits_for``) that ``_resolve_family`` uses to
    look up the relevant entry in ``FactorProfile.stats``.

    The protocol is deliberately silent on how the value was originally
    computed — that lives in the procedure that produced the profile, or
    in the ``HACEstimator.compute`` extension below.
    """

    @property
    def name(self) -> str:
        """Stable identifier used in error messages and diagnostics."""
        ...

    @property
    def description(self) -> str:
        """One-line summary of the inference method (cell-agnostic).

        Cell-specific stat semantics live in
        ``factrix._codes._STAT_DESCRIPTIONS`` keyed by ``StatCode``.
        """
        ...

    def applicable_to(self, scope: FactorScope, density: FactorDensity) -> bool:
        """Whether this estimator applies to the ``(scope, density)`` cell."""
        ...

    def emits_for(
        self,
        scope: FactorScope,
        density: FactorDensity,
    ) -> StatCode:
        """Map a cell to the ``StatCode`` whose value this estimator names.

        Called only after ``applicable_to`` has returned ``True``;
        implementations may assume the cell is in their applicability set.
        """
        ...


@runtime_checkable
class HACEstimator(Estimator, Protocol):
    """``Estimator`` that runs HAC-on-mean inference on a 1-D series.

    Adds ``min_periods`` (sample-size floor below which SE is deemed
    unreliable; estimator surfaces it via ``warnings`` rather than
    silently degrading) and ``compute(series, *, forward_periods)``
    returning an ``InferenceResult``. Cell procedures dispatch to this
    instead of hardcoding the Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent (HAC) path.

    Slice-test estimators (cluster Wald, block bootstrap)
    take different input shapes and live on parallel ``Estimator`` sub-
    protocols rather than overloading this one.
    """

    @property
    def min_periods(self) -> int:
        """Lower bound on ``len(series)`` for SE-validity.

        ``len(series) < min_periods`` should emit
        ``WarningCode.UNRELIABLE_SE_SHORT_PERIODS`` in
        ``InferenceResult.warnings``. The HARD-floor case (series so
        short the test cannot run at all) is the estimator's call to
        raise ``InsufficientSampleError``; ``min_periods`` is the soft
        contract.
        """
        ...

    def compute(
        self,
        series: np.ndarray,
        *,
        forward_periods: int,
    ) -> InferenceResult:
        """Run the inference test on ``series``.

        Args:
            series: 1-D HAC-target series (per-period information coefficient (IC), per-date
                Fama-MacBeth λ, dense CAAR, etc.). The cell procedure
                owns extraction from raw panel; the estimator only
                sees the test target.
            forward_periods: Overlap horizon of the series (MA(h-1)
                structure). Newey-West (NW) uses it to floor the Bartlett bandwidth;
                Hansen-Hodrick (HH) requires it for the rectangular-kernel lag count.

        Returns:
            ``InferenceResult`` with ``stat`` / ``p_value`` and the
            ``StatCode`` keys the procedure should stitch into
            ``FactorProfile.stats`` / ``metadata``.
        """
        ...
