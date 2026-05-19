"""Lowercase callable estimator entry points consumed by metric implementations.

Boundary with :mod:`factrix.stats`:

- :mod:`factrix.stats` exposes the PascalCase :class:`Estimator` protocol
  family (``NeweyWest`` / ``HansenHodrick`` / ``BlockBootstrap``) used
  by the ``AnalysisConfig.estimator`` field and the slice-test
  estimator-dispatch path.
- :mod:`factrix.estimators` exposes lowercase function aliases over the
  same underlying computations, callable directly from inside a metric
  implementation without instantiating an ``Estimator`` first.

The two namespaces are not in tension — :mod:`factrix.stats` retires the
class-based ``Estimator`` selection surface as part of the
``AnalysisConfig`` retirement, and the lowercase callables here are the
forward-compatible internal-consumption surface for metric callables
and, in the future, runtime per-metric estimator override.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np

    from factrix.stats._estimator import InferenceResult


def newey_west(series: np.ndarray, *, forward_periods: int) -> InferenceResult:
    """Newey-West HAC SE on a series mean → t-statistic → two-sided p-value.

    Lowercase callable alias over :class:`factrix.stats.NeweyWest`.
    Bartlett kernel, NW1994 auto-bandwidth, Hansen-Hodrick overlap
    floor. ``forward_periods`` sets the overlap horizon used by the
    floor.
    """
    from factrix.stats.newey_west import NeweyWest

    return NeweyWest().compute(series, forward_periods=forward_periods)


def hansen_hodrick(series: np.ndarray, *, forward_periods: int) -> InferenceResult:
    """Hansen-Hodrick (1980) rectangular-kernel HAC SE on a series mean.

    Lowercase callable alias over :class:`factrix.stats.HansenHodrick`.
    Rectangular kernel matched to MA(h-1) overlap structure from
    h-period forward returns; no PSD guarantee — variance is clamped
    at zero and the result carries
    ``WarningCode.RECT_KERNEL_NEGATIVE_VARIANCE`` when the raw estimate
    is negative.
    """
    from factrix.stats.hansen_hodrick import HansenHodrick

    return HansenHodrick().compute(series, forward_periods=forward_periods)


def block_bootstrap(
    diff: np.ndarray,
    *,
    block_length: int | Literal["auto"] = "auto",
    n_resamples: int = 999,
    scheme: Literal["fixed", "stationary"] = "stationary",
    rng_seed: int | None = None,
) -> tuple[float, dict[str, float | int | str]]:
    """Two-sided empirical p for ``H_0: E[diff] = 0`` on a paired series.

    Lowercase callable alias over the dependent-bootstrap path used by
    :class:`factrix.stats.BlockBootstrap`. Stationary scheme
    (Politis-Romano 1994) by default; ``scheme="fixed"`` runs the
    Kunsch (1989) fixed-block variant. Block length resolves via
    Politis-White (2004) when ``block_length="auto"``.

    Returns ``(p_value, metadata)`` — p in ``[1/(B+1), 1]`` (Davison-
    Hinkley smoothing keeps p strictly positive); metadata records the
    resolved block length, scheme, n_resamples, and the seed actually
    used (so reproducibility survives ``rng_seed=None``).
    """
    if block_length != "auto" and block_length < 1:
        raise ValueError(
            f"block_length must be 'auto' or int >= 1; got {block_length!r}."
        )
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1; got {n_resamples!r}.")
    if scheme not in ("fixed", "stationary"):
        raise ValueError(f"scheme must be 'fixed' or 'stationary'; got {scheme!r}.")
    from factrix._stats.bootstrap import _block_bootstrap_diff_p

    return _block_bootstrap_diff_p(
        diff,
        block_length=block_length,
        n_resamples=n_resamples,
        scheme=scheme,
        rng_seed=rng_seed,
    )


__all__ = [
    "block_bootstrap",
    "hansen_hodrick",
    "newey_west",
]
