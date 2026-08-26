"""Stationary ([Politis-Romano (1994)][politis-romano-1994]) bootstrap for dependent time series.

Parametric inference (standard t-test, Newey-West heteroskedasticity-and-autocorrelation-consistent (HAC)) is unreliable when
the marginal distribution is heavy-tailed / skewed or the sample is
short relative to the dependence horizon. The stationary bootstrap
resamples geometric-length blocks from the input series, preserving
short-range dependence without assuming a specific parametric form —
reached for in event-clustering situations, persistent macro factors,
and non-normal information coefficient (IC) distributions.

It is a different set of assumptions, not a rescue. Under dependence the
block bootstrap carries its own finite-sample long-run-variance
shortfall, so ``bootstrap_mean_ci`` studentizes by default and publishes
its measured coverage rather than claiming the regime is handled: at
AR(0.8), T=100 the studentized interval covers 0.890 against a nominal
0.950 (the percentile interval covers 0.792). Read the table in
``bootstrap_mean_ci`` before using it as a short-sample remedy.

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
from typing import Literal, NamedTuple

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
            when the series is too short or degenerate), clamped to
            ``[1, ceil(min(3*sqrt(T), T/3))]`` — ``arch``'s ``b_max``.
            Bounding ``L`` relative to ``T`` keeps enough effective blocks
            (``~T/L``) for the resample distribution to carry information.
            ``block_length=1`` reduces to the ordinary iid bootstrap
            (Efron). An explicitly supplied value is validated against the
            same bound and raises ``UserInputError`` outside it, rather
            than being silently passed through: at ``L >= T`` a circular
            resample degenerates to a rotation of the input.
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


class BootstrapCI(NamedTuple):
    """Bootstrap confidence interval for a scalar statistic.

    A NamedTuple rather than a bare 3-tuple so ``result.estimate`` reads
    unambiguously; the previous plain tuple was ordered ``(low, high,
    point)``, which invites an ``(estimate, low, high)`` misread at the
    call site. Still unpacks positionally in the same order.
    """

    low: float
    high: float
    estimate: float


#: Below this many resamples a bootstrap quantile is dominated by resampling
#: noise: at B=200 the 2.5% tail is the 5th order statistic. Politis-White
#: (2004) recommend >= 999 for two-sided 5% work; this is the refusal floor,
#: not the recommendation. Deliberately not named ``MIN_*``: that prefix is
#: reserved (FX003) for sample-size floors on a data axis, and a resample
#: count is an algorithm knob, not an axis.
_BOOTSTRAP_RESAMPLES_FLOOR: int = 200


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    block_length: float | None = None,
    seed: int | None = None,
    statistic: Callable[[np.ndarray], float] | None = None,
    method: Literal["studentized", "percentile"] = "studentized",
) -> BootstrapCI:
    r"""Stationary-bootstrap confidence interval for a statistic.

    Default ``statistic`` is the arithmetic mean (matches the parametric
    t-test's H₀ null). Pass a callable taking a 1-D array and returning
    a scalar to CI other statistics (e.g. Sharpe, median, skewness);
    those require ``method="percentile"`` (see below).

    **Studentized by default.** The interval is the bootstrap-t

        $$\left[ar x - t^*_{1-lpha}\,\widehat{se},\;
                ar x - t^*_{lpha}\,\widehat{se}
    ight],\qquad
          t^*_b = rac{ar x^*_b - ar x}{\widehat{se}^*_b}$$

    with $\widehat{se}$ the batch-means block SE
    (:func:`factrix._stats.bootstrap._batch_means_se`) at the resolved
    block length. The percentile interval is first-order accurate only
    ([DiCiccio-Efron (1996)][diciccio-efron-1996]), and under dependence
    it compounds the block bootstrap's finite-sample long-run-variance
    shortfall — which is exactly the regime this module's docstring
    reaches for it.

    Measured two-sided coverage at nominal 95% on an AR(1) mean
    (500 sims, B=999):

    | φ   | T   | percentile | studentized |
    |-----|-----|------------|-------------|
    | 0.0 | 100 | 0.940      | 0.942       |
    | 0.0 | 500 | 0.946      | 0.954       |
    | 0.5 | 100 | 0.862      | 0.916       |
    | 0.5 | 500 | 0.924      | 0.940       |
    | 0.8 | 100 | 0.792      | 0.890       |
    | 0.8 | 500 | 0.902      | 0.934       |

    Studentizing removes roughly half the under-coverage; it does not
    remove it. A short, strongly persistent series still under-covers,
    and no resampling scheme fixes that — read the last row as the
    disclosed limit, not as a residual bug.

    Args:
        values: 1-D array of the original series, at least 2 finite
            observations.
        n_bootstrap: Resample count. Must be at least
            200 resamples; below that the interval
            endpoints are resampling noise. At ``B=1`` the old
            implementation returned a zero-width interval that did not
            even contain its own point estimate.
        ci: Two-sided coverage, e.g. ``0.95`` for a 95% CI. Must be in
            ``(0, 1)``.
        block_length: See ``stationary_bootstrap_resamples``. Validated
            against ``[1, ceil(min(3*sqrt(T), T/3))]``; a length at or
            past ``T`` used to collapse every resample to a rotation of
            the input and return a zero-width interval.
        seed: Reproducibility seed.
        statistic: Scalar function applied to each resample. Defaults
            to ``np.mean``.
        method: ``"studentized"`` (default) or ``"percentile"``. The
            studentized root needs a block SE of the statistic on every
            resample, which this module only has for the mean, so a
            custom ``statistic`` must pass ``method="percentile"``
            explicitly. Refusing rather than silently downgrading keeps
            the accuracy order of the returned interval something the
            caller chose.

    Returns:
        :class:`BootstrapCI` — ``(low, high, estimate)`` where
        ``estimate`` is the statistic on the original sample.

    Raises:
        UserInputError: ``method="studentized"`` with a custom
            ``statistic``.
        ValueError: ``ci`` outside ``(0, 1)``, non-1-D ``values``, fewer
            than 2 observations, or ``n_bootstrap`` below the floor.

    References:
        - [Politis & Romano (1994)][politis-romano-1994]. "The Stationary
          Bootstrap." Journal of the American Statistical Association,
          89(428), 1303–1313. Underlying resampling scheme.
        - [DiCiccio & Efron (1996)][diciccio-efron-1996]. "Bootstrap
          Confidence Intervals." Statistical Science 11(3), 189–228.
          Accuracy ordering that makes the studentized interval the
          default here.
        - [Götze & Künsch (1996)][gotze-kunsch-1996]. Second-order
          correctness of the blockwise bootstrap for a studentized root.
    """
    from factrix._errors import UserInputError
    from factrix._stats.bootstrap import _batch_means_se

    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci!r}")
    if method not in ("studentized", "percentile"):
        raise ValueError(
            f"method must be 'studentized' or 'percentile', got {method!r}"
        )
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(
            f"bootstrap_mean_ci: values must be 1-D; got shape {values.shape}."
        )
    if len(values) < 2:
        raise ValueError(
            f"bootstrap_mean_ci: need at least 2 observations to resample; "
            f"got {len(values)}."
        )
    if n_bootstrap < _BOOTSTRAP_RESAMPLES_FLOOR:
        raise ValueError(
            f"bootstrap_mean_ci: n_bootstrap must be at least "
            f"{_BOOTSTRAP_RESAMPLES_FLOOR}; got {n_bootstrap!r}. Below that the "
            f"interval endpoints are resampling noise."
        )
    if statistic is not None and method == "studentized":
        raise UserInputError(
            func_name="bootstrap_mean_ci",
            field="method",
            value=method,
            expected=(
                "'percentile' when a custom statistic is supplied — the "
                "studentized root needs a block SE of the statistic on every "
                "resample, which is only available for the mean"
            ),
            docs_path="api/stats#bootstrap_mean_ci",
        )

    if block_length is None:
        block_length = _resolve_auto_block_length(values)
    resamples = stationary_bootstrap_resamples(
        values,
        n_bootstrap=n_bootstrap,
        block_length=block_length,
        seed=seed,
    )
    alpha = (1.0 - ci) / 2.0

    if statistic is not None:
        # Fast path for the default (mean) below; custom callables fall
        # through to the generic loop.
        stats = np.apply_along_axis(statistic, 1, resamples)
        point = float(statistic(values))
        return BootstrapCI(
            float(np.quantile(stats, alpha)),
            float(np.quantile(stats, 1.0 - alpha)),
            point,
        )

    stats = resamples.mean(axis=1)
    point = float(values.mean())
    if method == "percentile":
        return BootstrapCI(
            float(np.quantile(stats, alpha)),
            float(np.quantile(stats, 1.0 - alpha)),
            point,
        )

    se_observed = float(_batch_means_se(values, block_length)[0])
    se_boot = _batch_means_se(resamples, block_length)
    usable = np.isfinite(se_boot) & (se_boot > 0.0)
    if not (np.isfinite(se_observed) and se_observed > 0.0) or not usable.any():
        # A zero-dispersion sample has no scale to studentize by. The
        # percentile interval on such a sample is the degenerate point
        # itself, which is the honest answer rather than a fabricated width.
        return BootstrapCI(
            float(np.quantile(stats, alpha)),
            float(np.quantile(stats, 1.0 - alpha)),
            point,
        )
    roots = (stats[usable] - point) / se_boot[usable]
    lo_q, hi_q = np.quantile(roots, [alpha, 1.0 - alpha])
    return BootstrapCI(
        point - float(hi_q) * se_observed,
        point - float(lo_q) * se_observed,
        point,
    )
