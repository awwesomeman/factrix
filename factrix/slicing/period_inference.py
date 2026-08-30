"""Cross-slice statistical-test functions for **date-disjoint** partitions.

The sibling :mod:`factrix.slicing.inference` pair
(:func:`slice_pairwise_test` / :func:`slice_joint_test`) is
**cross-sectional**: it inner-joins each slice's per-period series on
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

Both consume the same per-slice per-period series as the cross-sectional
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

import math
import warnings
from itertools import combinations
from typing import Literal, NamedTuple

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from factrix._codes import WarningCode, _validate_expected_warnings_arg
from factrix._data_input import _resolve_overlap_periods
from factrix._errors import UserInputError
from factrix._stats.bootstrap import (
    Rng,
    _check_n_resamples,
    _empirical_p,
    _politis_white_block_length,
    _resolve_rng,
    _stationary_block_indices,
)
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
from factrix.stats.multiple_testing import holm_adjusted_p, romano_wolf_adjusted_p

Method = Literal["bootstrap", "analytic"]

# Slice length below which the joint test's realised size exceeds ~1.5× its
# nominal level on a true null with K >= 3 slices. Set from the measured
# K × T grid (500 seeds each, iid per-period IC, nominal 5%): K=5 rejects 9.4% /
# 8.2% / 5.4% / 5.6% and K=3 6.2% / 7.8% / 5.4% / 3.8% at T = 50 / 90 / 150 /
# 250, while K=2 stays within 4.6–5.8% throughout. The excess is the
# per-slice Bartlett HAC variance estimate's small-sample noise (effective
# df ≈ 21 at T=50, not T-1) inverted across K-1 restrictions; the bootstrap
# path shows the same 12% at K=5, T=50, so it is not a routing fix.
_JOINT_SHORT_SLICE_PERIODS = 150


def _validate_method(method: str, func_name: str) -> None:
    if method not in ("bootstrap", "analytic"):
        raise UserInputError(
            func_name=func_name,
            field="method",
            value=method,
            expected='one of "bootstrap" (default) or "analytic"',
            docs_path=_DOCS_SLICE,
        )


_REASON_INSUFFICIENT_PERIODS = "insufficient_periods"
_REASON_DEGENERATE_VARIANCE = WarningCode.DEGENERATE_VARIANCE.value


class _SliceSeries(NamedTuple):
    """Per-slice series plus the floor verdict the slice tests gate on."""

    labels: list[str]
    series: list[np.ndarray]
    min_periods: int | None
    thin: frozenset[str]


def _require_slice_floor(
    metric: MetricBase,
    labels: list[str],
    series_list: list[np.ndarray],
    *,
    overlap_periods: int,
    strict: bool,
    func_name: str,
) -> tuple[int | None, frozenset[str]]:
    """Gate every slice's per-period series on the metric's own floor.

    Returns ``(floor, thin_labels)``. With ``strict=True`` a thin slice
    raises; with ``strict=False`` the caller receives the thin labels and
    emits structured unavailable rows for them instead.

    ``by_slice`` short-circuits a thin metric to NaN via the metric body; the
    date-disjoint slice tests build each slice's per-period series directly and
    would otherwise return a calibrated-looking p-value on a sub-floor regime.
    Reuse the metric's own :class:`SampleThreshold`, resolved at the panel's
    stamped (or, on an unstamped panel, declared) ``overlap_periods`` — the
    single source of truth both paths read — so they agree on what counts as a
    thin sample, and refuse (rather than emit) the contrast at that size: the
    inferential path must be at least as protective as the descriptive one.
    The per-period series length is the slice's
    time-axis sample, so only the time-series floors (``min_periods`` /
    ``min_events``) bind — the cross-section floors (``min_assets`` /
    ``min_pairs``) describe within-period width, which the series has already
    collapsed.
    """
    threshold = metric._resolved_sample_threshold(overlap_periods)
    floor = max(
        (f for f in (threshold.min_periods, threshold.min_events) if f is not None),
        default=None,
    )
    if floor is None:
        return None, frozenset()
    thin = [
        (lbl, int(s.shape[0]))
        for lbl, s in zip(labels, series_list, strict=True)
        if s.shape[0] < floor
    ]
    if not thin or not strict:
        return floor, frozenset(lbl for lbl, _ in thin)
    detail = ", ".join(f"{lbl!r} (n_periods={n})" for lbl, n in thin)
    raise UserInputError(
        func_name=func_name,
        field="by",
        value=detail,
        expected=(
            f"every slice at or above {type(metric).__name__!r}'s minimum "
            f"sample floor ({floor}, resolved at "
            f"overlap_periods={overlap_periods}); by_slice short-circuits this "
            f"metric to NaN at that size, so the date-disjoint tests refuse to "
            f"return a contrast that is not calibrated. Use coarser regimes "
            f"(each ≥{floor} periods) or a metric with a lower sample floor."
        ),
        docs_path=_DOCS_SLICE,
    )


_MAX_TOLERATED_SHARED_DATES = 1


def _require_date_disjoint(
    slices: dict[str, pl.DataFrame],
    labels: list[str],
    *,
    by: str,
    func_name: str,
) -> None:
    """Refuse a partition whose slices share dates, before the metric runs.

    Mirror of the cross-sectional guard: :func:`slice_pairwise_test` /
    :func:`slice_joint_test` refuse a date-disjoint partition because their
    inner-join collapses to <2 aligned rows; this pair must refuse the
    opposite case just as loudly. The period-family maths treats each slice
    as an **independent sample** with block-diagonal cross-slice covariance,
    so slices that share dates (a cross-sectional partition — sector, size
    bucket — carries every date in every slice) would be contrasted as if
    their common shocks were independent, and the p-values would be
    anticonservative with nothing in the result marking it.

    Sharing a single date is tolerated: a date-axis partition truncated at a
    regime boundary can leave one common date, which
    ``slice_boundary_truncation`` already describes. Any pair sharing ≥2
    dates is a date-aligned partition and raises.
    """
    date_sets = {
        lbl: set(slices[lbl].get_column("date").unique().to_list()) for lbl in labels
    }
    worst: tuple[int, str, str] | None = None
    for i, left in enumerate(labels):
        for right in labels[i + 1 :]:
            shared = len(date_sets[left] & date_sets[right])
            if shared > _MAX_TOLERATED_SHARED_DATES and (
                worst is None or shared > worst[0]
            ):
                worst = (shared, left, right)
    if worst is None:
        return
    shared, left, right = worst
    raise UserInputError(
        func_name=func_name,
        field="by",
        value=by,
        expected=(
            f"a column whose slices are date-disjoint; slices {left!r} and "
            f"{right!r} share {shared} dates. These tests are date-disjoint "
            f"— they treat each slice as an independent sample, so slices "
            f"must share ≤1 date (a truncated regime boundary). A "
            f"date-aligned partition (e.g. sector, size bucket) shares its "
            f"dates and is not supported here — use slice_pairwise_test / "
            f"slice_joint_test for date-aligned partitions."
        ),
        docs_path=_DOCS_SLICE,
    )


def _build_per_slice_series(
    data: pl.DataFrame,
    metric: MetricBase,
    by: str,
    *,
    factor_col: str,
    overlap_periods: int,
    strict: bool,
    func_name: str,
) -> _SliceSeries:
    """Partition ``data`` by ``by`` and build each slice's per-period series.

    Mirrors the cross-sectional :func:`_build_per_date_panel` front-end
    (producer → ``per_date_series``) but **does not inner-join on date** —
    each slice keeps its own (disjoint) dates. Returns a :class:`_SliceSeries`
    — ``labels`` and one 1-D ``np.ndarray`` per slice of that slice's
    per-period metric values, plus the resolved floor and the labels below it
    (empty unless ``strict=False`` admitted them).

    Raises ``UserInputError`` if ``factor_col`` is absent or the slices are
    not date-disjoint; ``ValueError`` on <2 slice values or any slice with
    <2 dates; ``TypeError`` (via
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
    slices = _slice_by(data, by, func_name=func_name)
    if len(slices) < 2:
        raise UserInputError(
            func_name=func_name,
            field="by",
            value=by,
            expected=(
                f"a column holding ≥2 distinct slice values; {by!r} holds "
                f"{len(slices)}. A one-value partition has nothing to compare."
            ),
            docs_path=_DOCS_SLICE,
        )
    labels = list(slices.keys())
    _require_date_disjoint(slices, labels, by=by, func_name=func_name)
    series_list: list[np.ndarray] = []
    for lbl in labels:
        produced = _run_producer_for_factor(producer, slices[lbl], factor_col)
        s = per_date_fn(produced)["value"].to_numpy()
        if s.shape[0] < 2:
            raise UserInputError(
                func_name=func_name,
                field="by",
                value=lbl,
                expected=(
                    f"≥2 per-period observations per disjoint slice for a "
                    f"within-slice variance estimate; slice {lbl!r} has "
                    f"{s.shape[0]}."
                ),
                docs_path=_DOCS_SLICE,
            )
        series_list.append(np.asarray(s, dtype=float))
    floor, thin = _require_slice_floor(
        metric,
        labels,
        series_list,
        overlap_periods=overlap_periods,
        strict=strict,
        func_name=func_name,
    )
    return _SliceSeries(labels, series_list, floor, thin)


def _bootstrap_slice_means(
    series_list: list[np.ndarray],
    *,
    n_resamples: int,
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
    boot = np.empty((k, n_resamples))
    for i, s in enumerate(series_list):
        obs_means[i] = float(s.mean())
        block_len = _politis_white_block_length(s)
        idx = _stationary_block_indices(len(s), n_resamples, float(block_len), rng)
        boot[i] = s[idx].mean(axis=1)
    return obs_means, boot


def slice_period_pairwise_test(
    data: pl.DataFrame,
    metric: MetricBase,
    *,
    by: str,
    factor_col: str,
    method: Method = "bootstrap",
    overlap_periods: int | None = None,
    n_resamples: int = 999,
    rng: Rng = None,
    strict: bool = True,
) -> pl.DataFrame:
    """Pairwise cross-slice contrasts for a **date-disjoint** partition.

    Date-disjoint counterpart of :func:`slice_pairwise_test`: partitions
    a raw panel on ``by``, builds each slice's per-period metric series via
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
            ``per_date_series`` (``ic()`` / ``fm_beta()`` / ``positive_rate()``).
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
        overlap_periods: The evaluation-grid overlap the sample floor (and,
            under ``method="analytic"``, the per-slice HAC bandwidth) resolve
            at. Normally omitted — :func:`factrix.preprocess.compute_forward_return`
            stamps it on the panel and it is read from there. Declare it for a
            self-attached ``forward_return`` panel that carries no stamp; the
            contract is :func:`factrix.evaluate`'s — a value disagreeing with
            the stamp is rejected, and an unstamped panel with no declaration
            is an error rather than a silent default.
        n_resamples: ``B`` for the ``"bootstrap"`` path (ignored by
            ``"analytic"``). Must be at least ``BOOTSTRAP_RESAMPLES_FLOOR``
            — the shared floor every factrix entry point reporting an
            inference drawn from resamples enforces. The default 999 is
            [Politis-White (2004)][politis-white-2004]'s recommendation for
            two-sided 5% work.
        rng: Reproducibility seed for the ``"bootstrap"`` path
            (ignored by ``"analytic"``). ``None`` draws from system
            entropy and the resolved int is reported in the ``seed``
            column; an ``int`` is reported unchanged; a
            ``numpy.random.Generator`` is used as-is and advanced by the
            call, and the column is null because only its owner can
            reproduce the draw. Under ``method="analytic"`` the column is
            null throughout.
        strict: ``True`` (default) raises ``ValueError`` when any slice is
            below the metric's sample floor. ``False`` keeps the schema and
            returns every pair touching a thin slice as an unavailable row
            (``stat`` / ``p_raw`` / ``p_adj`` NaN, ``reason=
            "insufficient_periods"``); the remaining pairs are tested and
            their multiplicity family is the tested pairs only. The
            ``strict=False`` counterpart of :func:`factrix.evaluate`'s
            ``metric_unavailable`` short-circuit, for batch regime research
            where one thin regime must not abort the sweep.

    Returns:
        Long-form ``pl.DataFrame`` with columns ``(slice_a, slice_b,
        n_periods_a, n_periods_b, mean_diff, stat, p_raw, p_adj, stat_type,
        reference_dist, df_num, df_denom, multiplicity, min_periods, seed,
        reason)``; one row per ordered slice pair ``(a, b)``. ``n_periods_*``
        are each slice's own date counts (disjoint spans differ in length)
        and ``min_periods`` the floor they were gated on (resolved at the
        panel's stamped or declared ``overlap_periods``). ``reason`` is null on a tested
        pair, ``"degenerate_variance"`` when the contrast variance collapsed
        (NaN ``stat`` / ``p_raw`` / ``p_adj`` — no test, not a
        non-rejection), or ``"insufficient_periods"`` for a pair admitted by
        ``strict=False``; filter on ``reason.is_null()`` before folding rows
        into a wider family. ``mean_diff`` is the
        signed ``μ_a − μ_b``; ``stat`` the studentized contrast
        ``(μ_a − μ_b)² / (v_a + v_b)``. The mechanism columns disclose the
        path: ``stat_type="wald"``, ``df_num=1``; ``reference_dist`` is
        ``"bootstrap_null"`` (``method="bootstrap"``, ``df_denom=None``) or
        ``"f"`` (``method="analytic"``: ``F_{1, ν}`` with the per-pair
        Welch-Satterthwaite ``ν`` in ``df_denom`` — two disjoint slices with
        their own HAC variances is the Welch two-sample setting, and the
        asymptotic χ²₁ used earlier over-rejected at short slices); and
        ``multiplicity`` the family-wise correction (``"romano_wolf"`` for
        ``"bootstrap"``, ``"holm"`` for ``"analytic"``).

    Raises:
        UserInputError: ``metric`` is not a metric instance, ``factor_col``
            is absent, ``method`` is invalid, ``overlap_periods`` is not a
            positive ``int`` / disagrees with the panel's stamp / is missing
            on an unstamped panel, or ``by`` names a **date-aligned**
            partition — any two slices sharing ≥2 dates are not independent
            samples and belong to :func:`factrix.slice_pairwise_test` /
            :func:`factrix.slice_joint_test`.
        ValueError: Fewer than two slice values, any slice with fewer than
            two dates, or (``strict=True``) any slice whose per-period series
            is below the metric's own ``SampleThreshold`` floor resolved at
            the panel's stamped or declared ``overlap_periods`` (the size at
            which :func:`factrix.by_slice` short-circuits the metric to NaN;
            see :func:`factrix.sample_requirements`).
        TypeError: Metric is not slice-test-eligible (no ``per_date_series``
            capability / no producer).
    """
    _validate_metric_instance(metric, "slice_period_pairwise_test")
    _validate_method(method, "slice_period_pairwise_test")
    _check_n_resamples(
        n_resamples, func_name="slice_period_pairwise_test", docs_path=_DOCS_SLICE
    )
    op = _resolve_overlap_periods(
        data, overlap_periods, horizon=None, func_name="slice_period_pairwise_test"
    )
    built = _build_per_slice_series(
        data,
        metric,
        by,
        factor_col=factor_col,
        overlap_periods=op,
        strict=strict,
        func_name="slice_period_pairwise_test",
    )
    labels = built.labels
    n_periods = [int(s.shape[0]) for s in built.series]
    pairs = list(combinations(range(len(labels)), 2))
    # Tested pairs are those whose slices both clear the floor; the contrast
    # machinery below runs on that subset and the multiplicity family is the
    # tested pairs only. Pairs touching a thin slice (``strict=False``) are
    # assembled afterwards as unavailable rows in the same schema.
    # Resolve once for the whole call: every pair's contrast draws from the
    # same stream, and the resolved int is reported so an unseeded run stays
    # reproducible. The analytic path draws nothing and reports null, but the
    # seed is still validated so a bogus type is refused on both paths.
    rng, seed_used = _resolve_rng(
        rng, func_name="slice_period_pairwise_test", docs_path=_DOCS_SLICE
    )
    seed_reported = seed_used if method == "bootstrap" else None
    tested = [i for i, lbl in enumerate(labels) if lbl not in built.thin]
    series_list = [built.series[i] for i in tested]
    tested_pairs = list(combinations(range(len(tested)), 2))
    contrasts = _pairwise_contrasts(
        series_list,
        [n_periods[i] for i in tested],
        tested_pairs,
        method=method,
        overlap_periods=op,
        n_resamples=n_resamples,
        rng=rng,
    )
    by_pair = {
        (tested[i], tested[j]): row
        for (i, j), row in zip(tested_pairs, contrasts, strict=True)
    }
    nan = float("nan")
    rows = [by_pair.get(pair, (nan, nan, nan, nan, None)) for pair in pairs]
    mean_diffs = [row[0] for row in rows]
    stats = [row[1] for row in rows]
    p_raw = [row[2] for row in rows]
    p_adj = [row[3] for row in rows]
    df_denoms = [row[4] for row in rows]
    reasons = [
        _REASON_INSUFFICIENT_PERIODS
        if pair not in by_pair
        else (_REASON_DEGENERATE_VARIANCE if math.isnan(stat) else None)
        for pair, stat in zip(pairs, stats, strict=True)
    ]
    n_pairs = len(pairs)
    return pl.DataFrame(
        {
            "slice_a": [labels[i] for i, _ in pairs],
            "slice_b": [labels[j] for _, j in pairs],
            "n_periods_a": [n_periods[i] for i, _ in pairs],
            "n_periods_b": [n_periods[j] for _, j in pairs],
            "mean_diff": mean_diffs,
            "stat": stats,
            "p_raw": p_raw,
            "p_adj": p_adj,
            "stat_type": ["wald"] * n_pairs,
            "reference_dist": ["bootstrap_null" if method == "bootstrap" else "f"]
            * n_pairs,
            "df_num": [1] * n_pairs,
            "df_denom": df_denoms,
            "multiplicity": ["romano_wolf" if method == "bootstrap" else "holm"]
            * n_pairs,
            "min_periods": [built.min_periods] * n_pairs,
            "seed": [seed_reported] * n_pairs,
            "reason": reasons,
        },
        schema_overrides={
            "df_denom": pl.Float64,
            "min_periods": pl.Int64,
            "seed": pl.Int64,
            "reason": pl.String,
        },
    )


def _pairwise_contrasts(
    series_list: list[np.ndarray],
    n_periods: list[int],
    pairs: list[tuple[int, int]],
    *,
    method: Method,
    overlap_periods: int | None,
    n_resamples: int,
    rng: np.random.Generator,
) -> list[tuple[float, float, float, float, float | None]]:
    """Studentized contrast per pair: ``(mean_diff, stat, p_raw, p_adj, df_denom)``.

    A pair whose contrast variance collapses carries NaN in ``stat`` /
    ``p_raw`` / ``p_adj`` (no test, not a non-rejection — see
    ``_stats.wald._NOT_COMPUTABLE``) and is left out of the multiplicity
    family, which runs over the computable pairs only.
    """
    if method == "bootstrap":
        obs_means, boot = _bootstrap_slice_means(
            series_list, n_resamples=n_resamples, rng=rng
        )
        t_obs: list[float] = []
        boot_cols: list[np.ndarray] = []
        mean_diffs: list[float] = []
        p_raw: list[float] = []
        for i, j in pairs:
            diff_obs = float(obs_means[i] - obs_means[j])
            d = boot[i] - boot[j]
            se = float(d.std(ddof=1))
            if se < EPSILON:
                # No resampling dispersion in the contrast: no test. NaN,
                # not t = 0 / p = 1 — see ``_stats.wald._NOT_COMPUTABLE``.
                t = float("nan")
                col = np.full(n_resamples, float("nan"))
            else:
                t = diff_obs / se
                # Centre on the OBSERVED difference, not the bootstrap mean:
                # the studentised null draw is (θ* − θ̂) / se* (Efron-Tibshirani
                # §16), and it is what ``_wald_bootstrap_omnibus`` already does
                # via ``boot - obs_means``. Centring on ``d.mean()`` was an
                # implicit bias correction the omnibus path did not apply.
                col = (d - diff_obs) / se
            t_obs.append(t)
            boot_cols.append(col)
            mean_diffs.append(diff_obs)
            if np.isnan(t):
                p_raw.append(float("nan"))
            else:
                extreme = int(np.sum(np.abs(col) >= abs(t)))
                # Raw p only; the Romano-Wolf step-down p_adj below is not a
                # single binomial draw, so no MC SE is reported here.
                p_raw.append(_empirical_p(extreme, n_resamples)[0])
        # Romano-Wolf runs over the computable pairs; a collapsed pair stays
        # NaN in stat / p_raw / p_adj.
        t_arr = np.asarray(t_obs, dtype=float)
        computable = np.isfinite(t_arr)
        p_adj = np.full(len(pairs), float("nan"))
        if computable.any():
            boot_matrix = np.column_stack(
                [boot_cols[k] for k in np.flatnonzero(computable)]
            )
            p_adj[computable] = romano_wolf_adjusted_p(
                t_arr[computable], boot_matrix, one_sided=False
            )
        stats = [float(t * t) for t in t_obs]
        df_denoms: list[float | None] = [None] * len(pairs)
    else:
        means, variances = _analytic_slice_moments(series_list, overlap_periods)
        mean_diffs = []
        stats = []
        p_raw = []
        df_denoms = []
        for i, j in pairs:
            diff_obs = float(means[i] - means[j])
            se2 = float(variances[i] + variances[j])
            # Two disjoint slices with their own HAC variances is the Welch
            # two-sample setting, so the finite-sample reference is
            # F_{1, ν} with the Welch-Satterthwaite ν rather than the
            # asymptotic χ²₁. At the smallest slice the metric floors
            # admit (50 periods) the two are within simulation noise
            # (6.4% vs 6.6% null size at nominal 5%); the F reference is
            # kept for consistency with the cross-sectional sibling in
            # ``slicing.inference`` and because the disclosed ν lets a
            # reader recompute p from ``stat``. ν is per pair because the
            # two slices differ in length.
            nu = _satterthwaite_df(
                np.array([variances[i], variances[j]]),
                np.array([n_periods[i], n_periods[j]]),
            )
            if se2 <= EPSILON:
                # Both slices constant: no contrast variance, no test. NaN,
                # not (0, 1) — see ``_stats.wald._NOT_COMPUTABLE``.
                chi = float("nan")
                p = float("nan")
            else:
                chi = diff_obs * diff_obs / se2
                p = float(sp_stats.f.sf(chi, dfn=1, dfd=nu))
            mean_diffs.append(diff_obs)
            stats.append(chi)
            p_raw.append(p)
            df_denoms.append(nu)
        # Holm runs over the computable pairs; a collapsed pair stays NaN.
        p_raw_arr = np.asarray(p_raw, dtype=float)
        computable = np.isfinite(p_raw_arr)
        p_adj = np.full(len(p_raw), float("nan"))
        if computable.any():
            p_adj[computable] = holm_adjusted_p(p_raw_arr[computable])

    return list(
        zip(mean_diffs, stats, p_raw, (float(x) for x in p_adj), df_denoms, strict=True)
    )


def _satterthwaite_df(variances: np.ndarray, n_periods: np.ndarray) -> float:
    """Welch-Satterthwaite denominator df for a contrast of K slice means.

    ``variances`` are the (HAC) variances *of the mean*, so the classic
    ``s²/n`` terms are already folded in. The ratio is computed on the
    variance *shares* ``w = var / Σ var`` so it is scale-free: ``var_*``
    shrinks like ``σ²/n`` and the raw ``Σ var²/(n-1)`` denominator like
    ``σ⁴/n³``, which for a per-period σ ≈ 0.1 drops below any absolute
    tolerance from roughly n ≈ 60 and would pin ν at the floor for
    perfectly healthy data. ``K = 2`` is the pairwise Welch df; ``K > 2``
    is the same Satterthwaite approximation on the diagonal Wald form the
    omnibus uses. At the slice lengths the public API admits (``ic``
    floors at 50 periods) ν lands in the hundreds, where ``F_{K-1, ν}``
    and ``χ²_{K-1}`` are practically the same reference — the point of
    using it is consistency with the pairwise path and disclosing ν, not
    a size correction. Floors at 1.0 so a degenerate
    set (all variances zero) cannot hand ``f.sf`` a zero df.
    """
    v = np.asarray(variances, dtype=float)
    n = np.asarray(n_periods, dtype=float)
    total = float(v.sum())
    if not np.isfinite(total) or total <= 0.0:
        return 1.0
    w = v / total
    den = float(np.sum(w**2 / np.maximum(n - 1.0, 1.0)))
    if not np.isfinite(den) or den <= 0.0:
        return 1.0
    return max(1.0 / den, 1.0)


def _analytic_slice_moments(
    series_list: list[np.ndarray],
    overlap_periods: int | None,
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
        lags = _hac_lags(overlap_periods, len(s))
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
    Returns ``(W, p)``; ``(nan, nan)`` on a singular middle matrix (no
    test — see ``_stats.wald._NOT_COMPUTABLE``).
    """
    middle = restriction @ np.diag(variances) @ restriction.T
    try:
        middle_inv = np.linalg.inv(middle)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    contrast = restriction @ obs_means
    stat = float(contrast @ middle_inv @ contrast)
    # Recentre each slice's draws to its own mean → null contrasts, then
    # the Wald quadratic form per draw (einsum over the (r, B) contrasts).
    null_contrasts = restriction @ (boot - obs_means[:, None])
    w_boot = np.einsum("ib,ij,jb->b", null_contrasts, middle_inv, null_contrasts)
    p, _mc_se = _empirical_p(int(np.sum(w_boot >= stat)), boot.shape[1])
    return stat, p


def slice_period_joint_test(
    data: pl.DataFrame,
    metric: MetricBase,
    *,
    by: str,
    factor_col: str,
    method: Method = "bootstrap",
    overlap_periods: int | None = None,
    n_resamples: int = 999,
    rng: Rng = None,
    strict: bool = True,
    expected_warnings: tuple[str, ...] = (),
) -> pl.DataFrame:
    """Omnibus block-diagonal Wald χ² that all K disjoint-slice means are equal.

    Date-disjoint counterpart of :func:`slice_joint_test`. The restriction
    is ``μ_0 = μ_1 = … = μ_{K-1}`` (K-1 contrasts against the first slice);
    because the slices are independent samples, the cross-slice covariance
    is **block-diagonal** — ``Var(μ_k)`` on the diagonal, zero off it. Both
    methods share the same Wald quadratic form; they differ in how the null
    is referenced, mirroring the pairwise path: ``"analytic"`` uses an
    ``F_{K-1, ν}`` reference with a Satterthwaite ``ν``, while ``"bootstrap"`` calibrates the
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
        overlap_periods: The evaluation-grid overlap the sample floor (and,
            under ``method="analytic"``, the per-slice HAC bandwidth) resolve
            at. Normally omitted — :func:`factrix.preprocess.compute_forward_return`
            stamps it on the panel and it is read from there. Declare it for a
            self-attached ``forward_return`` panel that carries no stamp; the
            contract is :func:`factrix.evaluate`'s — a value disagreeing with
            the stamp is rejected, and an unstamped panel with no declaration
            is an error rather than a silent default.
        n_resamples: ``B`` for the ``"bootstrap"`` path (ignored by
            ``"analytic"``). Must be at least ``BOOTSTRAP_RESAMPLES_FLOOR``
            — the shared floor every factrix entry point reporting an
            inference drawn from resamples enforces. The default 999 is
            [Politis-White (2004)][politis-white-2004]'s recommendation for
            two-sided 5% work.
        rng: Reproducibility seed for the ``"bootstrap"`` path
            (ignored by ``"analytic"``). ``None`` draws from system
            entropy and the resolved int is reported in the ``seed``
            column; an ``int`` is reported unchanged; a
            ``numpy.random.Generator`` is used as-is and advanced by the
            call, and the column is null because only its owner can
            reproduce the draw. Under ``method="analytic"`` the column is
            null throughout.
        strict: ``True`` (default) raises ``ValueError`` when any slice is
            below the metric's sample floor. ``False`` returns the same
            single-row schema as an unavailable row instead (``stat`` /
            ``p_value`` NaN, ``reason="insufficient_periods"``): the omnibus
            restriction spans every slice, so one thin slice makes the whole
            test unavailable. The ``strict=False`` counterpart of
            :func:`factrix.evaluate`'s ``metric_unavailable`` short-circuit.
        expected_warnings: :class:`~factrix.WarningCode` values declaring
            warning regimes that are the study's **design** — the same
            contract as :func:`factrix.evaluate`. A batch of regime tests on
            short samples trips ``short_slice_joint_test`` on every cell by
            construction; declaring it keeps the record (the code stays in
            ``warning_codes``, leaves ``unexpected_warning_codes``) and stops
            the per-call ``UserWarning`` echo. Unknown codes are rejected.

    Returns:
        Single-row ``pl.DataFrame`` with columns ``(k_slices, n_periods_min,
        stat, p_value, stat_type, reference_dist, df_num, df_denom,
        multiplicity, min_periods, seed, short_slice_periods,
        warning_codes, unexpected_warning_codes, reason)``. ``stat`` is the joint Wald
        statistic; ``n_periods_min`` the shortest slice's date count and
        ``min_periods`` the floor it was gated on (resolved at the panel's
        stamped or declared ``overlap_periods``); ``reason`` is null when the test ran
        and ``"insufficient_periods"`` on a ``strict=False`` unavailable
        row. ``warning_codes`` lists every :class:`~factrix.WarningCode`
        the test raised (today only ``short_slice_joint_test``) and
        ``unexpected_warning_codes`` the subset not declared via
        ``expected_warnings`` — the audit pair of
        :attr:`~factrix.EvaluationResult.warnings` /
        :attr:`~factrix.EvaluationResult.unexpected_warnings`;
        ``short_slice_periods`` is the calibration threshold the short-slice
        code is gated on, next to the ``k_slices`` / ``n_periods_min`` it was
        read against. The mechanism
        columns
        disclose the reference: ``stat_type="wald"``, ``df_num=K-1``
        (restriction rank); ``reference_dist`` is ``"f"`` for
        ``method="analytic"`` — ``p_value`` from ``F_{K-1, ν}`` with the
        Satterthwaite ``ν`` in ``df_denom``, the K-slice form of the
        pairwise Welch reference — or ``"bootstrap_null"`` (``df_denom``
        ``None``) for ``method="bootstrap"``. ``multiplicity`` is ``None`` — a single
        omnibus has no family-internal correction.

    Raises:
        UserInputError: ``metric`` is not a metric instance, ``factor_col``
            is absent, ``method`` is invalid, ``overlap_periods`` is not a
            positive ``int`` / disagrees with the panel's stamp / is missing
            on an unstamped panel, or ``by`` names a **date-aligned**
            partition — any two slices sharing ≥2 dates are not independent
            samples and belong to :func:`factrix.slice_pairwise_test` /
            :func:`factrix.slice_joint_test`.
        ValueError: Fewer than two slice values, any slice with fewer than
            two dates, or (``strict=True``) any slice whose per-period series
            is below the metric's own ``SampleThreshold`` floor resolved at
            the panel's stamped or declared ``overlap_periods`` (the size at
            which :func:`factrix.by_slice` short-circuits the metric to NaN;
            see :func:`factrix.sample_requirements`).
        TypeError: Metric is not slice-test-eligible.

    Warns:
        UserWarning: ``K >= 3`` and the shortest slice has fewer than 150
            periods — :class:`~factrix.WarningCode.SHORT_SLICE_JOINT_TEST`,
            recorded in the row's ``warning_codes`` and echoed unless declared
            in ``expected_warnings``. Measured size on a true null (iid
            per-period IC, nominal 5%, 500 seeds per cell), rows ``K``,
            columns ``T``::

                K / T     50     90    150    250
                2        .046   .058   .052   .046
                3        .062   .078   .054   .038
                5        .094   .082   .054   .056

            The excess is not the reference distribution (``F`` and χ²
            agree at these ν) and not the aggregation (with the true
            variances the same statistic rejects 3.8%): each slice's
            Bartlett HAC variance is a noisy small-sample estimate —
            effective df ≈ 21 at ``T = 50`` rather than ``T - 1`` — and
            inverting K-1 of them inflates the Wald. The bootstrap path
            carries the same noise (12% at ``K=5, T=50``), so switching
            method does not help; a longer slice does. Prewhitening,
            plug-in bandwidths and finite-sample F references were each
            measured and none calibrates this case. Pairwise contrasts on
            the same slices sit at 5–6% and are the better-calibrated read.
    """
    _validate_metric_instance(metric, "slice_period_joint_test")
    _validate_method(method, "slice_period_joint_test")
    _check_n_resamples(
        n_resamples, func_name="slice_period_joint_test", docs_path=_DOCS_SLICE
    )
    expected = _validate_expected_warnings_arg(
        expected_warnings, func_name="slice_period_joint_test", docs_path=_DOCS_SLICE
    )
    # Resolved before the unavailable-row short-circuits so every row carries
    # the same column, and so a bogus seed is refused on both methods.
    rng, seed_used = _resolve_rng(
        rng, func_name="slice_period_joint_test", docs_path=_DOCS_SLICE
    )
    seed_reported = seed_used if method == "bootstrap" else None
    op = _resolve_overlap_periods(
        data, overlap_periods, horizon=None, func_name="slice_period_joint_test"
    )
    built = _build_per_slice_series(
        data,
        metric,
        by,
        factor_col=factor_col,
        overlap_periods=op,
        strict=strict,
        func_name="slice_period_joint_test",
    )
    labels, series_list = built.labels, built.series
    k = len(labels)
    shortest = min(len(series) for series in series_list)
    reference_dist = "bootstrap_null" if method == "bootstrap" else "f"

    # The audit pair every entry point that takes ``expected_warnings`` shares:
    # a fired code always lands in ``warning_codes``; declaring it only keeps
    # it out of ``unexpected_warning_codes`` and silences the stderr echo. An
    # unavailable (``strict=False``) row carries empty lists — the test did
    # not run, so it raised nothing.
    warning_codes: list[str] = []

    def _row(
        stat: float, p: float, df_denom: float | None, reason: str | None
    ) -> pl.DataFrame:
        unexpected = [c for c in warning_codes if c not in expected]
        return pl.DataFrame(
            {
                "k_slices": [k],
                "n_periods_min": [shortest],
                "stat": [stat],
                "p_value": [p],
                "stat_type": ["wald"],
                "reference_dist": [reference_dist],
                "df_num": [k - 1],
                "df_denom": [df_denom],
                "multiplicity": [None],
                "min_periods": [built.min_periods],
                "seed": [seed_reported],
                "short_slice_periods": [_JOINT_SHORT_SLICE_PERIODS],
                "warning_codes": [list(warning_codes)],
                "unexpected_warning_codes": [unexpected],
                "reason": [reason],
            },
            schema_overrides={
                "df_denom": pl.Float64,
                "multiplicity": pl.String,
                "min_periods": pl.Int64,
                "seed": pl.Int64,
                "short_slice_periods": pl.Int64,
                "warning_codes": pl.List(pl.String),
                "unexpected_warning_codes": pl.List(pl.String),
                "reason": pl.String,
            },
        )

    if built.thin:
        return _row(float("nan"), float("nan"), None, _REASON_INSUFFICIENT_PERIODS)
    if k >= 3 and shortest < _JOINT_SHORT_SLICE_PERIODS:
        code = WarningCode.SHORT_SLICE_JOINT_TEST
        warning_codes.append(code.value)
        if code.value not in expected:
            warnings.warn(
                f"slice_period_joint_test: {len(series_list)} slices with the "
                f"shortest at {shortest} periods "
                f"(< {_JOINT_SHORT_SLICE_PERIODS}). On a true null the joint "
                f"test over-rejects here — measured 8–9% at a nominal 5% for "
                f"K=5 with 50–90-period slices, under both methods — because "
                f"each slice's HAC variance is a noisy small-sample estimate "
                f"inverted across K-1 restrictions. Read the p-value as "
                f"indicative; pairwise contrasts on the same slices are "
                f"better calibrated ({code.value}; declare it in "
                f"expected_warnings= to keep the record and stop this echo).",
                UserWarning,
                stacklevel=2,
            )
    restriction = _equality_restriction(k)

    if method == "bootstrap":
        obs_means, boot = _bootstrap_slice_means(
            series_list, n_resamples=n_resamples, rng=rng
        )
        variances = boot.var(axis=1, ddof=1)
        stat, p = _wald_bootstrap_omnibus(obs_means, boot, variances, restriction)
        df_denom = None
    else:
        means, variances = _analytic_slice_moments(series_list, op)
        # Same finite-sample reference the pairwise path uses, generalised
        # to K slices: F_{K-1, ν} with the Satterthwaite ν on the diagonal
        # Wald form. Consistency + disclosure, not a size fix: at reachable
        # slice lengths ν is in the hundreds and F ≈ χ². The joint path's
        # residual over-rejection (up to ~9% at nominal 5% on short slices,
        # converging by T ≈ 150) comes from the *noise* of each slice's
        # small-sample HAC variance estimate (effective df ≈ 21 at T = 50),
        # not from bias — see the Warns block on ``slice_period_joint_test``.
        nu = _satterthwaite_df(
            variances, np.array([len(series) for series in series_list])
        )
        stat, p = _wald_p_linear(means, np.diag(variances), restriction, df_denom=nu)
        df_denom = nu

    return _row(stat, p, df_denom, None)
