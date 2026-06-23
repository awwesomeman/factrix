"""Cross-slice statistical-test functions for **date-disjoint** partitions.

The sibling :mod:`factrix.slicing.inference` pair
(:func:`slice_pairwise_test` / :func:`slice_joint_test`) is
**cross-sectional**: it inner-joins each slice's per-date series on
``date`` and runs a joint Newey-West HAC + slice-cluster Wald. Its maths
assumes the slices share dates (sector, size bucket, liquidity tier) so
cross-slice covariance enters through the joint HAC.

**date-disjoint** partitions — market regime (bull / bear, high-vol /
low-vol), calendar period, in/out-of-sample — have *no* common dates:
the cross-sectional inner-join collapses to 0 rows. Their statistical
nature is fundamentally different: disjoint spans are (approximately)
**independent samples** with **block-diagonal** cross-slice covariance.
This module supplies the matching inference pair —
:func:`slice_period_pairwise_test` / :func:`slice_period_joint_test` —
named ``slice_period_*`` because each slice occupies a distinct span of
time (covering regime, calendar period, and in/out-of-sample alike).
They are **regime analysis**'s inferential entry point: ``by_slice``
gives the descriptive per-regime numbers, this pair gives the calibrated
cross-regime contrast with multiple-testing control.

These are kept as a **separate, explicit** function pair (not folded into
the cross-sectional pair via date-overlap auto-routing) so the two
statistical assumptions never hide behind one name.

Both consume the same per-slice per-date series as the cross-sectional
pair (metric producer → ``per_date_series``) but **do not inner-join** —
each slice keeps its own dates. A two-valued ``method`` flag selects the
standard-error / p estimator:

- ``"bootstrap"`` (default) — each slice's series is **independently**
  block-bootstrapped (stationary blocks, Politis-White automatic block
  length); pairwise multiplicity is **Romano-Wolf** step-down
  (bootstrap-native, exploits the joint dependence through shared
  slices). Never invalid: asymptotically valid *and* small-sample robust
  (the block length absorbs serial autocorrelation, i.e. built-in HAC).
  The right default for short regimes (T ≈ 30-80) where HAC asymptotics
  are unreliable.
- ``"analytic"`` (opt-in) — each slice mean carries a Newey-West HAC
  variance; pairwise contrasts are Welch-style unequal-variance, the
  omnibus is a block-diagonal Wald χ²; pairwise multiplicity is **Holm**
  (no bootstrap distribution, so Romano-Wolf is unavailable). Faster,
  deterministic, closed-form — choose it when T is large enough for HAC
  asymptotics (rule of thumb T ≳ 100): decade sub-samples, pre/post,
  in/out-of-sample.

The ``p_adj`` correction family is decided *internally* by ``method``
(bootstrap → Romano-Wolf, analytic → Holm); there is deliberately no
separate ``multiple_testing`` knob.

**Independence assumption.** Disjoint spans are treated as independent
samples (block-diagonal covariance). Adjacent regimes may carry boundary
serial correlation; that is not auto-detected (no general reliable
"calendar-adjacent" semantics) — the caller owns the partition.

Matrix-row: slice_period_pairwise_test, slice_period_joint_test | (*, *, *, *, *) | inference function | per-pair studentized contrast + Romano-Wolf/Holm / block-diagonal Wald χ² | _build_per_slice_series, _slice_by, resolve_per_date_series
"""

from __future__ import annotations

from itertools import combinations
from typing import Literal

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from factrix._data_input import _read_forward_periods_stamp
from factrix._errors import UserInputError
from factrix._metric_index import SampleThreshold
from factrix._stats.bootstrap import (
    Scheme,
    _politis_white_block_length,
    _stationary_block_indices,
)
from factrix._stats.multiple_testing import holm_step_down, romano_wolf
from factrix._stats.wald import _nw_hac_vector_mean, _wald_p_linear
from factrix._types import EPSILON
from factrix.metrics._base import MetricBase
from factrix.metrics._metric_capabilities import resolve_per_date_series
from factrix.slicing._primitive import _slice_by
from factrix.slicing.inference import (
    _DOCS_SLICE,
    _hac_lags,
    _resolve_producer,
    _run_producer_for_factor,
    _validate_metric_instance,
)

Method = Literal["bootstrap", "analytic"]

_N_RESAMPLES = 999
_SCHEME: Scheme = "stationary"


def _validate_method(method: str, func_name: str) -> None:
    if method not in ("bootstrap", "analytic"):
        raise UserInputError(
            func_name=func_name,
            field="method",
            value=method,
            expected='one of "bootstrap" (default) or "analytic"',
            docs_path=_DOCS_SLICE,
        )


def _resolve_sample_threshold(metric: MetricBase) -> SampleThreshold:
    """Resolve the metric instance's ``SampleThreshold``, honouring config.

    Mirrors :meth:`MetricBase.spec` but evaluates the dynamic
    ``sample_threshold_for`` hook against the **actual instance** (not a
    default-built one), so a metric configured with a non-default
    ``forward_periods`` (e.g. ``caar``) reports the floor for its own
    horizon rather than the default-config floor.
    """
    cls = type(metric)
    hook = cls.sample_threshold_for
    return hook(metric) if hook is not None else cls.sample_threshold


def _require_slice_floor(
    metric: MetricBase,
    labels: list[str],
    series_list: list[np.ndarray],
    *,
    func_name: str,
) -> None:
    """Raise when any slice's per-date series is below the metric's own floor.

    ``by_slice`` short-circuits a thin metric to NaN via the metric body; the
    date-disjoint slice tests build each slice's per-date series directly and
    would otherwise return a calibrated-looking p-value on a sub-floor regime.
    Reuse the metric's own :class:`SampleThreshold` — the single source of
    truth both paths read — so they agree on what counts as a thin sample, and
    refuse (rather than emit) the contrast at that size: the inferential path
    must be at least as protective as the descriptive one. The per-date series
    length is the slice's time-axis sample, so only the time-series floors
    (``min_periods`` / ``min_events``) bind — the cross-section floors
    (``min_assets`` / ``min_pairs``) describe within-date width, which the
    series has already collapsed.
    """
    threshold = _resolve_sample_threshold(metric)
    floor = max(
        (f for f in (threshold.min_periods, threshold.min_events) if f is not None),
        default=None,
    )
    if floor is None:
        return
    thin = [
        (lbl, int(s.shape[0]))
        for lbl, s in zip(labels, series_list, strict=True)
        if s.shape[0] < floor
    ]
    if not thin:
        return
    detail = ", ".join(f"{lbl!r} (n_periods={n})" for lbl, n in thin)
    raise ValueError(
        f"{func_name}: slice(s) {detail} fall below {type(metric).__name__!r}'s "
        f"minimum sample floor ({floor}); by_slice short-circuits this metric "
        f"to NaN at that size, so the date-disjoint tests refuse to return a "
        f"contrast that is not calibrated. Use coarser regimes (each ≥{floor} "
        f"periods) or a metric with a lower sample floor."
    )


def _build_per_slice_series(
    data: pl.DataFrame,
    metric: MetricBase,
    by: str,
    *,
    factor_col: str,
    func_name: str,
) -> tuple[list[str], list[np.ndarray]]:
    """Partition ``data`` by ``by`` and build each slice's per-date series.

    Mirrors the cross-sectional :func:`_build_per_date_panel` front-end
    (producer → ``per_date_series``) but **does not inner-join on date** —
    each slice keeps its own (disjoint) dates. Returns
    ``(labels, [series_k])`` with each ``series_k`` a 1-D ``np.ndarray``
    of that slice's per-date metric values.

    Raises ``UserInputError`` if ``factor_col`` is absent; ``ValueError``
    on <2 slice values or any slice with <2 dates; ``TypeError`` (via
    resolver) if ``metric`` is not slice-test-eligible.
    """
    if factor_col not in data.columns:
        raise UserInputError(
            func_name=func_name,
            field="factor_col",
            value=factor_col,
            expected=f"a column present in data; got columns {data.columns}",
            docs_path=_DOCS_SLICE,
        )
    per_date_fn = resolve_per_date_series(type(metric))
    producer = _resolve_producer(metric, func_name)
    slices = _slice_by(data, by)
    if len(slices) < 2:
        raise ValueError(
            f"{func_name}: need ≥2 slice values on {by!r}; got {len(slices)}."
        )
    labels = list(slices.keys())
    series_list: list[np.ndarray] = []
    for lbl in labels:
        produced = _run_producer_for_factor(producer, slices[lbl], factor_col)
        s = per_date_fn(produced)["value"].to_numpy()
        if s.shape[0] < 2:
            raise ValueError(
                f"{func_name}: slice {lbl!r} has <2 dates ({s.shape[0]}); "
                f"each disjoint slice needs ≥2 per-date observations for a "
                f"within-slice variance estimate."
            )
        series_list.append(np.asarray(s, dtype=float))
    _require_slice_floor(metric, labels, series_list, func_name=func_name)
    return labels, series_list


def _bootstrap_slice_means(
    series_list: list[np.ndarray],
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent stationary-block bootstrap of each slice's mean.

    Each slice is resampled **on its own** (disjoint ⇒ independent), with
    a per-slice Politis-White automatic block length that absorbs the
    series' serial autocorrelation. Returns ``(obs_means[K],
    boot_means[K, B])`` — the same ``B`` draws per slice so every pairwise
    difference is formed from one shared set, preserving the joint
    dependence (through shared slices) that Romano-Wolf exploits.
    """
    k = len(series_list)
    obs_means = np.empty(k)
    boot = np.empty((k, _N_RESAMPLES))
    for i, s in enumerate(series_list):
        obs_means[i] = float(s.mean())
        block_len = _politis_white_block_length(s, scheme=_SCHEME)
        idx = _stationary_block_indices(len(s), _N_RESAMPLES, float(block_len), rng)
        boot[i] = s[idx].mean(axis=1)
    return obs_means, boot


def slice_period_pairwise_test(
    data: pl.DataFrame,
    metric: MetricBase,
    *,
    by: str,
    factor_col: str,
    method: Method = "bootstrap",
    rng_seed: int | None = None,
) -> pl.DataFrame:
    """Pairwise cross-slice contrasts for a **date-disjoint** partition.

    Date-disjoint counterpart of :func:`slice_pairwise_test`: partitions
    a raw panel on ``by``, builds each slice's per-date metric series via
    the metric's producer, and contrasts every slice pair as **independent
    samples** (no date inner-join). The right tool for **regime analysis**
    (bull / bear, high-vol / low-vol) and other time-disjoint splits
    (calendar period, in/out-of-sample), where the cross-sectional pair
    would raise ``<2 aligned dates``.

    Args:
        data: Raw long-format panel — same input contract as
            :func:`factrix.evaluate` (``date, asset_id, <factor_col>,
            forward_return``). Must contain ``by``; compose it upstream
            if needed.
        metric: A metric **instance** whose module declares
            ``per_date_series`` (``ic()`` / ``fm_beta()`` / ``hit_rate()``).
            The bare class is rejected.
        by: Column whose values define the slice partition (regime label,
            calendar bucket, …).
        factor_col: The single factor column to score per slice.
        method: ``"bootstrap"`` (default) runs an independent stationary
            block bootstrap per slice with Romano-Wolf step-down ``p_adj``;
            ``"analytic"`` runs Newey-West HAC per slice with Welch-style
            pairwise contrasts and Holm ``p_adj``. Use ``"bootstrap"`` for
            short regimes (T ≈ 30-80); ``"analytic"`` for long spans
            (T ≳ 100) when you want speed / determinism.
        rng_seed: Reproducibility seed for the ``"bootstrap"`` path
            (ignored by ``"analytic"``). ``None`` draws from system
            entropy. This is plumbing, not a statistical knob — block
            length, ``B``, and scheme are fixed by sensible defaults.

    Returns:
        Long-form ``pl.DataFrame`` with columns ``(slice_a, slice_b,
        n_periods_a, n_periods_b, mean_diff, stat, p_raw, p_adj)``; one row
        per ordered slice pair ``(a, b)``. ``n_periods_*`` are each slice's
        own date counts (disjoint spans differ in length). ``mean_diff`` is
        the signed ``μ_a − μ_b``; ``stat`` the studentized contrast on a
        χ²₁ scale; ``p_adj`` the family-wise correction (Romano-Wolf for
        ``"bootstrap"``, Holm for ``"analytic"``).

    Raises:
        UserInputError: ``metric`` is not a metric instance, ``factor_col``
            is absent, or ``method`` is invalid.
        ValueError: Fewer than two slice values, any slice with fewer than
            two dates, or any slice whose per-date series is below the
            metric's own ``SampleThreshold`` floor (the size at which
            :func:`factrix.by_slice` short-circuits the metric to NaN).
        TypeError: Metric is not slice-test-eligible (no ``per_date_series``
            capability / no producer).
    """
    _validate_metric_instance(metric, "slice_period_pairwise_test")
    _validate_method(method, "slice_period_pairwise_test")
    labels, series_list = _build_per_slice_series(
        data, metric, by, factor_col=factor_col, func_name="slice_period_pairwise_test"
    )
    n_periods = [int(s.shape[0]) for s in series_list]
    k = len(labels)
    pairs = list(combinations(range(k), 2))

    if method == "bootstrap":
        rng = np.random.default_rng(rng_seed)
        obs_means, boot = _bootstrap_slice_means(series_list, rng=rng)
        t_obs: list[float] = []
        boot_cols: list[np.ndarray] = []
        mean_diffs: list[float] = []
        p_raw: list[float] = []
        for i, j in pairs:
            diff_obs = float(obs_means[i] - obs_means[j])
            d = boot[i] - boot[j]
            se = float(d.std(ddof=1))
            if se < EPSILON:
                t = 0.0
                col = np.zeros(_N_RESAMPLES)
            else:
                t = diff_obs / se
                col = (d - d.mean()) / se
            t_obs.append(t)
            boot_cols.append(col)
            mean_diffs.append(diff_obs)
            extreme = int(np.sum(np.abs(col) >= abs(t)))
            p_raw.append((extreme + 1.0) / (_N_RESAMPLES + 1.0))
        boot_matrix = np.column_stack(boot_cols)
        p_adj = romano_wolf(t_obs, boot_matrix, one_sided=False)
        stats = [float(t * t) for t in t_obs]
    else:
        means, variances = _analytic_slice_moments(
            series_list, _read_forward_periods_stamp(data)
        )
        mean_diffs = []
        stats = []
        p_raw = []
        for i, j in pairs:
            diff_obs = float(means[i] - means[j])
            se2 = float(variances[i] + variances[j])
            if se2 <= EPSILON:
                chi = 0.0
                p = 1.0
            else:
                chi = diff_obs * diff_obs / se2
                p = float(sp_stats.chi2.sf(chi, df=1))
            mean_diffs.append(diff_obs)
            stats.append(chi)
            p_raw.append(p)
        p_adj = holm_step_down(p_raw)

    return pl.DataFrame(
        {
            "slice_a": [labels[i] for i, _ in pairs],
            "slice_b": [labels[j] for _, j in pairs],
            "n_periods_a": [n_periods[i] for i, _ in pairs],
            "n_periods_b": [n_periods[j] for _, j in pairs],
            "mean_diff": mean_diffs,
            "stat": stats,
            "p_raw": p_raw,
            "p_adj": list(p_adj),
        }
    )


def _analytic_slice_moments(
    series_list: list[np.ndarray],
    forward_periods: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-slice mean and Newey-West HAC variance of that mean.

    Each slice is treated independently: the HAC bandwidth follows the
    slice's own length (and the data's forward-overlap floor), so a
    short regime and a decade sub-sample get appropriately different
    kernels. Returns ``(means[K], variances[K])`` where ``variances`` is
    the HAC variance *of the mean* (the diagonal of the block-diagonal
    cross-slice covariance).
    """
    means = np.empty(len(series_list))
    variances = np.empty(len(series_list))
    for i, s in enumerate(series_list):
        lags = _hac_lags(forward_periods, len(s))
        mean, var = _nw_hac_vector_mean(s.reshape(-1, 1), lags=lags)
        means[i] = float(mean[0])
        variances[i] = float(var[0, 0])
    return means, variances


def _equality_restriction(k: int) -> np.ndarray:
    """``(K-1, K)`` contrast matrix for ``μ_0 = μ_1 = … = μ_{K-1}``.

    Row ``r`` is ``[1, 0, …, -1 (at r+1), …, 0]`` — each later slice
    contrasted against the first.
    """
    restriction = np.zeros((k - 1, k))
    restriction[:, 0] = 1.0
    for r in range(k - 1):
        restriction[r, r + 1] = -1.0
    return restriction


def _wald_bootstrap_omnibus(
    obs_means: np.ndarray,
    boot: np.ndarray,
    variances: np.ndarray,
    restriction: np.ndarray,
) -> tuple[float, float]:
    """Block-diagonal Wald χ² with a **bootstrap** null reference.

    Computes the observed omnibus Wald ``W = (Rμ)' (R V R')⁻¹ (Rμ)`` with
    block-diagonal ``V = diag(variances)``, then calibrates it against the
    empirical distribution of the same quadratic form over the bootstrap
    draws recentred to H₀ (each slice's draws centred on its own observed
    mean, so the contrasts are null). Keeps the ``"bootstrap"`` omnibus
    bootstrap-native — consistent with the pairwise path — rather than
    falling back to the χ² asymptotics the ``"analytic"`` path uses.
    Returns ``(W, p)``; ``(0.0, 1.0)`` on a singular middle matrix.
    """
    middle = restriction @ np.diag(variances) @ restriction.T
    try:
        middle_inv = np.linalg.inv(middle)
    except np.linalg.LinAlgError:
        return 0.0, 1.0
    contrast = restriction @ obs_means
    stat = float(contrast @ middle_inv @ contrast)
    # Recentre each slice's draws to its own mean → null contrasts, then
    # the Wald quadratic form per draw (einsum over the (r, B) contrasts).
    null_contrasts = restriction @ (boot - obs_means[:, None])
    w_boot = np.einsum("ib,ij,jb->b", null_contrasts, middle_inv, null_contrasts)
    b = boot.shape[1]
    p = (int(np.sum(w_boot >= stat)) + 1.0) / (b + 1.0)
    return stat, float(p)


def slice_period_joint_test(
    data: pl.DataFrame,
    metric: MetricBase,
    *,
    by: str,
    factor_col: str,
    method: Method = "bootstrap",
    rng_seed: int | None = None,
) -> pl.DataFrame:
    """Omnibus block-diagonal Wald χ² that all K disjoint-slice means are equal.

    Date-disjoint counterpart of :func:`slice_joint_test`. The restriction
    is ``μ_0 = μ_1 = … = μ_{K-1}`` (K-1 contrasts against the first slice);
    because the slices are independent samples, the cross-slice covariance
    is **block-diagonal** — ``Var(μ_k)`` on the diagonal, zero off it. Both
    methods share the same Wald quadratic form; they differ in how the null
    is referenced, mirroring the pairwise path: ``"analytic"`` uses the
    χ²_{K-1} asymptotic distribution, while ``"bootstrap"`` calibrates the
    statistic against its own block-bootstrap null (so a short-regime
    omnibus stays small-sample robust instead of leaning on χ²
    asymptotics). Useful for **regime analysis**: a single test of "does
    this factor's edge differ across regimes at all?" before drilling into
    pairs.

    Args:
        data: Raw long-format panel (see :func:`slice_period_pairwise_test`).
        metric: A metric **instance** whose module declares
            ``per_date_series``. The bare class is rejected.
        by: Column whose values define the slice partition.
        factor_col: The single factor column to score per slice.
        method: ``"bootstrap"`` (default) sources each ``Var(μ_k)`` from an
            independent stationary block bootstrap; ``"analytic"`` from a
            per-slice Newey-West HAC. See
            :func:`slice_period_pairwise_test`.
        rng_seed: Reproducibility seed for the ``"bootstrap"`` path
            (ignored by ``"analytic"``).

    Returns:
        Single-row ``pl.DataFrame`` with columns
        ``(k_slices, df, stat, p_value)``. ``df`` is the restriction rank
        (``K-1``); ``stat`` the joint Wald χ²; ``p_value`` from the χ²_{K-1}
        survival function (``"analytic"``) or the bootstrap null
        (``"bootstrap"``). No ``multiple_testing`` — a single omnibus has
        no family-internal correction.

    Raises:
        UserInputError: ``metric`` is not a metric instance, ``factor_col``
            is absent, or ``method`` is invalid.
        ValueError: Fewer than two slice values, any slice with fewer than
            two dates, or any slice whose per-date series is below the
            metric's own ``SampleThreshold`` floor (the size at which
            :func:`factrix.by_slice` short-circuits the metric to NaN).
        TypeError: Metric is not slice-test-eligible.
    """
    _validate_metric_instance(metric, "slice_period_joint_test")
    _validate_method(method, "slice_period_joint_test")
    labels, series_list = _build_per_slice_series(
        data, metric, by, factor_col=factor_col, func_name="slice_period_joint_test"
    )
    k = len(labels)
    restriction = _equality_restriction(k)

    if method == "bootstrap":
        rng = np.random.default_rng(rng_seed)
        obs_means, boot = _bootstrap_slice_means(series_list, rng=rng)
        variances = boot.var(axis=1, ddof=1)
        stat, p = _wald_bootstrap_omnibus(obs_means, boot, variances, restriction)
    else:
        means, variances = _analytic_slice_moments(
            series_list, _read_forward_periods_stamp(data)
        )
        stat, p = _wald_p_linear(means, np.diag(variances), restriction)

    return pl.DataFrame(
        {
            "k_slices": [k],
            "df": [k - 1],
            "stat": [stat],
            "p_value": [p],
        }
    )
