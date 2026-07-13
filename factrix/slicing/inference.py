"""Cross-slice statistical-test functions (cross-sectional / date-aligned).

``slice_pairwise_test`` reports K(K-1)/2 cross-slice contrasts as a
long-form DataFrame of pairwise ``(mean_diff, stat, p_raw, p_adj)``;
``slice_joint_test`` reports the single omnibus Wald χ² that all K
slice means are equal.

The ``_test`` suffix marks "function whose headline output is a
comparison test result (stat + p)" — distinct from metrics like
``ic`` / ``fm_beta`` whose significance is sidecar to a value,
and from ``compare`` which renders existing stats without
recomputing. Convention aligns with scipy (``ttest_ind`` /
``kruskal``) and R (``t.test`` / ``chisq.test``).

Both are **data-first**: they take a raw panel + metric **instance** +
``by`` + ``factor_col``, matching :func:`factrix.by_slice` /
:func:`factrix.evaluate`. Per slice they run the metric's producer
(declared via ``requires``) to build the per-date series, inner-join on
date, and run the analytic estimator
:class:`~factrix.stats.WaldNWCluster` (Newey-West (NW)
heteroskedasticity-and-autocorrelation-consistent (HAC) Bartlett kernel
+ 1-way cluster on the slice grouping) on the joint per-date K-vector
panel — cross-slice covariance enters through the joint HAC.

These tests are **cross-sectional only**: the inner-join on ``date``
requires the slices to share dates (sector, size bucket, liquidity
tier). date-disjoint partitions (market regime, calendar period) have
no common dates and are a different statistical path (two-sample HAC
mean difference) handled separately.

Matrix-row: slice_pairwise_test, slice_joint_test | (*, *, *, *) | inference function | per-pair Wald χ² + Holm / joint Wald χ² | _build_per_date_panel, _slice_by, resolve_per_date_series
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations

import numpy as np
import polars as pl

from factrix._data_input import _read_forward_periods_stamp
from factrix._errors import UserInputError
from factrix._stats.wald import _wald_nw_cluster_means
from factrix.metrics._base import MetricBase
from factrix.metrics._metric_capabilities import resolve_per_date_series
from factrix.slicing._primitive import _slice_by
from factrix.stats.multiple_testing import holm_adjusted_p

_DOCS_SLICE = "api/slice-test"


def _validate_metric_instance(metric: object, func_name: str) -> None:
    """Reject bare classes / non-metrics, matching ``evaluate`` / ``by_slice``."""
    if isinstance(metric, type) and issubclass(metric, MetricBase):
        raise UserInputError(
            func_name=func_name,
            field="metric",
            value=f"{metric.__name__} (the class)",
            expected=f"a metric instance, not the class — call it: {metric.__name__}()",
            docs_path=_DOCS_SLICE,
        )
    if not isinstance(metric, MetricBase):
        raise UserInputError(
            func_name=func_name,
            field="metric",
            value=type(metric).__name__,
            expected=(
                "a metric instance imported from factrix.metrics, e.g. ic() / fm_beta()"
            ),
            docs_path=_DOCS_SLICE,
        )


def _resolve_producer(metric: MetricBase, func_name: str) -> Callable:
    """Return the metric's single upstream producer from ``requires``.

    Slice tests build each slice's per-date series by running the
    producer (e.g. ``compute_ic``) on the raw panel slice. A metric with
    no producer cannot be driven from a raw panel here.
    """
    requires = type(metric).requires
    if not requires:
        raise TypeError(
            f"{func_name}: metric {type(metric).__name__!r} is not "
            f"slice-test-eligible: it declares no producer (`requires`) to "
            f"build a per-date series from a raw panel."
        )
    return next(iter(requires.values()))


def _run_producer_for_factor(
    producer: Callable,
    data: pl.DataFrame,
    factor_col: str,
) -> pl.DataFrame:
    """Run either a batch producer or a single-factor producer for one factor."""
    param_names = set(getattr(producer, "_param_names", ()))
    if "factor_cols" in param_names:
        produced = producer(data, factor_cols=[factor_col])
        return produced[factor_col]
    if "factor_col" in param_names:
        return producer(data, factor_col=factor_col)
    return producer(data)


def _too_few_aligned_dates_msg(
    func_name: str,
    slices: dict[str, pl.DataFrame],
    aligned_height: int,
) -> str:
    """Branch the ``<2 aligned dates`` error on the raw shared-date count.

    Two distinct failures collapse the joined panel to <2 rows:
    (1) the slices are genuinely date-disjoint (e.g. calendar regime); or
    (2) the slices share dates, but the per-slice metric dropped its per-date
    values — typically too few assets per slice. The cause changes the fix,
    so the message must too. The raw shared-date count (dates common to every
    slice *before* the metric runs) tells the two apart, which
    ``aligned_height`` alone cannot.
    """
    shared: set[object] | None = None
    for sub in slices.values():
        dates = set(sub.get_column("date").unique().to_list())
        shared = dates if shared is None else (shared & dates)
    raw_shared = len(shared) if shared is not None else 0
    if raw_shared < 2:
        return (
            f"{func_name}: <2 aligned dates across slices ({aligned_height}); "
            f"joint HAC inference requires aligned rows. These tests are "
            f"cross-sectional — slices must share ≥2 common dates. A "
            f"date-disjoint partition (e.g. calendar regime) shares no dates "
            f"and is not supported here — use slice_period_pairwise_test / "
            f"slice_period_joint_test for date-disjoint partitions."
        )
    return (
        f"{func_name}: slices share {raw_shared} raw dates but only "
        f"{aligned_height} aligned dates survived metric computation; joint "
        f"HAC inference requires ≥2. The partition is date-aligned — the "
        f"per-slice metric dropped most of its per-date values, typically "
        f"because each slice has too few assets to compute the metric "
        f"cross-sectionally. Widen each slice's universe (more assets per "
        f"slice) or use a coarser partition."
    )


def _build_per_date_panel(
    data: pl.DataFrame,
    metric: MetricBase,
    by: str,
    *,
    factor_col: str,
    func_name: str,
) -> tuple[list[str], np.ndarray, int]:
    """Partition ``data`` by ``by``, build each slice's per-date series via
    the metric's producer, inner-join on date, return
    ``(labels, panel[T, K], n_obs)``.

    Raises ``ValueError`` on <2 slice values or <2 aligned dates;
    ``TypeError`` (via resolver) if ``metric`` is not slice-test-eligible.
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

    def series_for(sub: pl.DataFrame) -> pl.DataFrame:
        produced = _run_producer_for_factor(producer, sub, factor_col)
        return per_date_fn(produced)

    aligned = series_for(slices[labels[0]]).rename({"value": "v_0"})
    for i, lbl in enumerate(labels[1:], start=1):
        aligned = aligned.join(
            series_for(slices[lbl]).rename({"value": f"v_{i}"}),
            on="date",
            how="inner",
        )
    if aligned.height < 2:
        raise ValueError(_too_few_aligned_dates_msg(func_name, slices, aligned.height))
    panel = aligned.drop("date").to_numpy()
    return labels, panel, aligned.height


def _hac_lags(forward_periods: int | None, n_obs: int) -> int:
    """Newey-West Bartlett bandwidth for the slice HAC.

    Defaults to the shared ``auto_bartlett(T)`` rule but floors at
    ``forward_periods - 1``:
    overlapping ``h``-period forward returns make the per-date series
    autocorrelated up to lag ``h - 1`` (MA(h-1) overlap structure), so
    the kernel must cover it or the variance is under-estimated and the
    test over-rejects. ``forward_periods`` is the panel's stamped overlap
    horizon — a property of the data, not the metric. Mirrors ``fm_beta``'s
    own NW-lag floor.
    """
    from factrix._stats import _resolve_nw_lags

    return _resolve_nw_lags(n_obs, lags=None, forward_periods=forward_periods)


def slice_pairwise_test(
    data: pl.DataFrame,
    metric: MetricBase,
    *,
    by: str,
    factor_col: str,
) -> pl.DataFrame:
    """Cross-slice pairwise Wald contrasts on a per-date metric panel.

    Data-first counterpart of :func:`factrix.by_slice`: partitions a raw
    panel on ``by``, builds each slice's per-date metric series via the
    metric's producer, aligns on date, and runs the analytic Newey-West
    HAC + slice-cluster Wald on every slice pair. Cross-sectional only
    (slices must share dates).

    Args:
        data: Raw long-format panel — same input contract as
            :func:`factrix.evaluate` (``date, asset_id, <factor_col>,
            forward_return``). Must contain ``by``; compose it upstream
            if needed.
        metric: A metric **instance** whose module declares
            ``per_date_series`` (``ic()`` / ``fm_beta()`` / ``positive_rate()``).
            The bare class is rejected.
        by: Column whose values define the slice partition.
        factor_col: The single factor column to score per slice.

    Returns:
        Long-form ``pl.DataFrame`` with columns ``(slice_a, slice_b,
        n_obs, mean_diff, stat, p_raw, p_adj, stat_type, reference_dist,
        df_num, df_denom, multiplicity)``; one row per ordered slice pair
        ``(a, b)`` with ``a`` before ``b`` in the partition's iteration
        order. ``mean_diff`` is the signed ``μ_a − μ_b`` (direction /
        effect size), ``stat`` the Wald statistic, and ``p_adj`` the Holm
        step-down family-wise correction across the K(K-1)/2 pairs. The
        trailing five columns disclose the active mechanism (constant
        across rows): ``stat_type="wald"``; ``reference_dist="F"`` with
        ``df_num=1`` (single contrast) and ``df_denom=n_obs-1`` — the
        date-cluster count is ``T=n_obs``, so the finite-sample
        ``F_{1, T-1}`` reference is used in place of the over-rejecting
        asymptotic χ²; ``multiplicity="holm"``.

    Raises:
        UserInputError: ``metric`` is not a metric instance, or
            ``factor_col`` is absent.
        ValueError: Fewer than two slice values, or fewer than two dates
            aligned across all slices (e.g. a date-disjoint partition).
        TypeError: Metric is not slice-test-eligible (no
            ``per_date_series`` capability / no producer).

    Examples:
        Pairwise information coefficient (IC) contrasts across two
        sectors on a synthetic cross-sectional panel — partition on a
        sector column, score ``ic`` per sector, contrast the per-date
        series:

        >>> import polars as pl
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics import ic
        >>> raw = fx.datasets.make_cs_panel(n_assets=100, n_dates=250)
        >>> panel = compute_forward_return(raw, forward_periods=5)
        >>> assets = panel["asset_id"].unique().sort().to_list()
        >>> sector = {a: ("tech" if i % 2 else "fin")
        ...           for i, a in enumerate(assets)}
        >>> panel = panel.with_columns(
        ...     pl.col("asset_id").replace_strict(sector).alias("sector")
        ... )
        >>> pairs = fx.slice_pairwise_test(
        ...     panel, ic(), by="sector", factor_col="factor"
        ... )
        >>> pairs.columns
        ['slice_a', 'slice_b', 'n_obs', 'mean_diff', 'stat', 'p_raw', 'p_adj', 'stat_type', 'reference_dist', 'df_num', 'df_denom', 'multiplicity']
    """
    _validate_metric_instance(metric, "slice_pairwise_test")

    labels, panel, n_obs = _build_per_date_panel(
        data, metric, by, factor_col=factor_col, func_name="slice_pairwise_test"
    )
    k = panel.shape[1]
    lags = _hac_lags(_read_forward_periods_stamp(data), n_obs)
    pairs = list(combinations(range(k), 2))
    col_means = panel.mean(axis=0)

    stats: list[float] = []
    p_raw: list[float] = []
    mean_diffs: list[float] = []
    for i, j in pairs:
        restriction = np.zeros((1, k))
        restriction[0, i] = 1.0
        restriction[0, j] = -1.0
        stat, p = _wald_nw_cluster_means(panel, R=restriction, lags=lags)
        stats.append(stat)
        p_raw.append(p)
        mean_diffs.append(float(col_means[i] - col_means[j]))

    p_adj = holm_adjusted_p(p_raw)

    n_pairs = len(pairs)
    return pl.DataFrame(
        {
            "slice_a": [labels[i] for i, _ in pairs],
            "slice_b": [labels[j] for _, j in pairs],
            "n_obs": [n_obs] * n_pairs,
            "mean_diff": mean_diffs,
            "stat": stats,
            "p_raw": p_raw,
            "p_adj": list(p_adj),
            "stat_type": ["wald"] * n_pairs,
            "reference_dist": ["F"] * n_pairs,
            "df_num": [1] * n_pairs,
            "df_denom": [n_obs - 1] * n_pairs,
            "multiplicity": ["holm"] * n_pairs,
        }
    )


def slice_joint_test(
    data: pl.DataFrame,
    metric: MetricBase,
    *,
    by: str,
    factor_col: str,
) -> pl.DataFrame:
    """Omnibus Wald χ² that all K slice means are equal.

    The joint restriction is ``β_0 = β_1 = … = β_{K-1}``, encoded as
    ``K-1`` contrasts against the first slice; the Wald statistic
    follows χ²_{K-1} under H₀. Same data-first contract and
    cross-sectional (shared-date) limitation as
    :func:`slice_pairwise_test`.

    Args:
        data: Raw long-format panel (see :func:`slice_pairwise_test`).
        metric: A metric **instance** whose module declares
            ``per_date_series``. The bare class is rejected.
        by: Column whose values define the slice partition.
        factor_col: The single factor column to score per slice.

    Returns:
        Single-row ``pl.DataFrame`` with columns ``(n_obs, k_slices, stat,
        p_value, stat_type, reference_dist, df_num, df_denom,
        multiplicity)``. ``stat`` is the joint Wald statistic. The
        mechanism columns disclose the reference: ``stat_type="wald"``,
        ``reference_dist="F"``, ``df_num=K-1`` (restriction rank) and
        ``df_denom=n_obs-1`` — ``p_value`` is the finite-sample
        ``F_{K-1, T-1}`` survival of ``stat / (K-1)`` (the date-cluster
        count is ``T=n_obs``, so the asymptotic χ² reference would
        over-reject). ``multiplicity`` is ``None`` — a single omnibus has
        no family-internal correction to apply.

    Raises:
        UserInputError: ``metric`` is not a metric instance, or
            ``factor_col`` is absent.
        ValueError: Fewer than two slice values, or fewer than two
            dates aligned across all slices.
        TypeError: Metric is not slice-test-eligible.

    Examples:
        Joint omnibus test that mean information coefficient (IC) is
        identical across two sectors (see :func:`slice_pairwise_test`
        for the panel construction):

        >>> import polars as pl
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics import ic
        >>> raw = fx.datasets.make_cs_panel(n_assets=100, n_dates=250)
        >>> panel = compute_forward_return(raw, forward_periods=5)
        >>> assets = panel["asset_id"].unique().sort().to_list()
        >>> sector = {a: ("tech" if i % 2 else "fin")
        ...           for i, a in enumerate(assets)}
        >>> panel = panel.with_columns(
        ...     pl.col("asset_id").replace_strict(sector).alias("sector")
        ... )
        >>> joint = fx.slice_joint_test(
        ...     panel, ic(), by="sector", factor_col="factor"
        ... )
        >>> joint["df_num"][0]
        1
    """
    _validate_metric_instance(metric, "slice_joint_test")

    _, panel, n_obs = _build_per_date_panel(
        data, metric, by, factor_col=factor_col, func_name="slice_joint_test"
    )
    k = panel.shape[1]
    lags = _hac_lags(_read_forward_periods_stamp(data), n_obs)

    # K-1 contrasts against slice 0: rows are [1, -1, 0, …], [1, 0, -1, …], …
    restriction = np.zeros((k - 1, k))
    restriction[:, 0] = 1.0
    for r in range(k - 1):
        restriction[r, r + 1] = -1.0

    stat, p = _wald_nw_cluster_means(panel, R=restriction, lags=lags)

    return pl.DataFrame(
        {
            "n_obs": [n_obs],
            "k_slices": [k],
            "stat": [stat],
            "p_value": [p],
            "stat_type": ["wald"],
            "reference_dist": ["F"],
            "df_num": [k - 1],
            "df_denom": [n_obs - 1],
            "multiplicity": [None],
        }
    )
