"""Cross-slice statistical-test verbs.

``slice_pairwise_test`` reports K(K-1)/2 cross-slice contrasts as a
long-form DataFrame of pairwise ``(stat, p_raw, p_adj)``;
``slice_joint_test`` reports the single omnibus Wald χ² that all K
slice means are equal.

The ``_test`` suffix marks "verb whose headline output is a
comparison test result (stat + p)" — distinct from metrics like
``ic`` / ``fama_macbeth`` whose significance is sidecar to a value,
and from ``compare`` which renders existing stats without
recomputing. Convention aligns with scipy (``ttest_ind`` /
``kruskal``) and R (``t.test`` / ``chisq.test``).

Both consume a metric callable whose module declares
``per_date_series`` (see
:mod:`factrix.metrics._metric_capabilities`); the slice partition
reuses ``_slice_by_label`` so dispatch logic is not duplicated. The
default estimator is :class:`~factrix.stats.WaldNWCluster` (NW HAC
Bartlett kernel + 1-way cluster on the slice grouping), which uses
the joint per-date K-vector panel — cross-slice covariance enters
through the joint HAC, so paired-diff degeneracies do not arise on
this path.

Matrix-row: slice_pairwise_test, slice_joint_test | (*, *, *, *) | inference verb | per-pair Wald χ² + Holm/RW/Bonferroni / joint Wald χ² | _build_per_date_panel, _resolve_estimator, _joint_block_bootstrap_pairwise_distribution
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from typing import Literal

import numpy as np
import polars as pl

from factrix._stats.bootstrap import _joint_block_bootstrap_pairwise_distribution
from factrix._stats.multiple_testing import bonferroni, holm_step_down, romano_wolf
from factrix._stats.wald import _wald_nw_cluster_means
from factrix.metrics._metric_capabilities import resolve_per_date_series
from factrix.slicing._primitive import _slice_by_label
from factrix.stats import BlockBootstrap, Estimator, WaldNWCluster

MultipleTestingMethod = Literal["holm", "bonferroni", "romano_wolf"]


def _resolve_estimator(
    estimator: Estimator | None, func_name: str
) -> WaldNWCluster | BlockBootstrap:
    if estimator is None:
        return WaldNWCluster()
    if not isinstance(estimator, WaldNWCluster | BlockBootstrap):
        raise NotImplementedError(
            f"{func_name}: estimator {type(estimator).__name__!r} not yet "
            f"wired; this function currently supports WaldNWCluster and "
            f"BlockBootstrap."
        )
    return estimator


def _build_per_date_panel(
    metric: Callable,
    df: pl.DataFrame,
    label: str,
    *,
    func_name: str,
) -> tuple[list[str], np.ndarray, int]:
    """Partition ``df`` by ``label``, extract per-date series per slice,
    inner-join on date, return ``(labels, panel[T, K], n_obs)``.

    Raises ``ValueError`` on <2 slice values or <2 aligned dates;
    ``TypeError`` (via resolver) if ``metric`` is not slice-test-
    eligible.
    """
    per_date_fn = resolve_per_date_series(metric)
    slices = _slice_by_label(df, label)
    if len(slices) < 2:
        raise ValueError(
            f"{func_name}: need ≥2 slice values on {label!r}; got {len(slices)}."
        )
    labels = list(slices.keys())
    aligned = per_date_fn(slices[labels[0]]).rename({"value": "v_0"})
    for i, lbl in enumerate(labels[1:], start=1):
        aligned = aligned.join(
            per_date_fn(slices[lbl]).rename({"value": f"v_{i}"}),
            on="date",
            how="inner",
        )
    if aligned.height < 2:
        raise ValueError(
            f"{func_name}: <2 aligned dates across slices ({aligned.height}); "
            f"joint HAC inference requires aligned rows. Check that "
            f"slices share at least 2 common dates."
        )
    panel = aligned.drop("date").to_numpy()
    return labels, panel, aligned.height


def slice_pairwise_test(
    metric: Callable,
    df: pl.DataFrame,
    *,
    label: str,
    estimator: Estimator | None = None,
    multiple_testing: MultipleTestingMethod | None = None,
) -> pl.DataFrame:
    """Cross-slice pairwise Wald contrasts on a per-date metric panel.

    Args:
        metric: Metric callable whose module declares ``per_date_series``.
        df: Input frame for the metric, containing ``label``.
        label: Column whose values define the slice partition.
        estimator: Inference estimator. ``None`` resolves to
            :class:`WaldNWCluster` (analytic NW HAC + 1-way cluster on
            slice). :class:`BlockBootstrap` triggers the joint
            block-bootstrap path and changes the default
            ``multiple_testing`` to ``"romano_wolf"``.
        multiple_testing: P-value adjustment family. ``None`` follows
            the estimator default — ``"holm"`` for ``WaldNWCluster``,
            ``"romano_wolf"`` for ``BlockBootstrap``. ``"romano_wolf"``
            with an analytic estimator raises ``ValueError`` (RW
            requires a bootstrap distribution).

    Returns:
        Long-form ``pl.DataFrame`` with columns
        ``(slice_a, slice_b, n_obs, stat, p_raw, p_adj)``; one row per
        ordered slice pair ``(a, b)`` with ``a`` before ``b`` in the
        partition's iteration order. ``stat`` carries the Wald χ² under
        ``WaldNWCluster`` and the signed mean diff under
        ``BlockBootstrap`` — bootstrap p-values are based on
        ``|mean diff|``.

    Raises:
        ValueError: Fewer than two slice values, fewer than two dates
            aligned across all slices, or an estimator / multiple-
            testing combination that has no calibrated p-value path.
        TypeError: Metric is not slice-test-eligible (no
            ``per_date_series`` capability).
        NotImplementedError: Estimator outside ``WaldNWCluster`` /
            ``BlockBootstrap``.

    Notes:
        BlockBootstrap reproducibility — pass an explicit ``rng_seed``
        on the :class:`BlockBootstrap` instance to fix the bootstrap
        draw. Bootstrap metadata (resolved block length, scheme, seed)
        is not attached to the returned DataFrame in this release;
        callers wanting it can either reconstruct from the estimator
        config or use :func:`factrix._stats.bootstrap._block_bootstrap_diff_p`
        directly per pair.

    Examples:
        Pairwise IC contrasts across two sub-universes. The canonical
        pattern is to compute the per-date metric series per slice
        upstream and concatenate with a label column — slices must
        share dates, so date-disjoint labels (e.g. calendar year)
        do not apply:

        >>> import polars as pl
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics import ic, compute_ic
        >>> raw = fx.datasets.make_cs_panel(n_assets=100, n_dates=500)
        >>> panel = compute_forward_return(raw, forward_periods=5)
        >>> assets = panel["asset_id"].unique().sort().to_list()
        >>> half = len(assets) // 2
        >>> sector_map = {a: ("tech" if i < half else "fin")
        ...               for i, a in enumerate(assets)}
        >>> panel_sec = panel.with_columns(
        ...     pl.col("asset_id").replace_strict(sector_map).alias("sector")
        ... )
        >>> per_sector_ic = pl.concat([
        ...     compute_ic(panel_sec.filter(pl.col("sector") == s))
        ...        .with_columns(pl.lit(s).alias("sector"))
        ...     for s in ("tech", "fin")
        ... ])
        >>> pairs = fx.slice_pairwise_test(ic, per_sector_ic, label="sector")

        Block-bootstrap path (auto-switches to Romano-Wolf
        multiple-testing):

        >>> from factrix.stats import BlockBootstrap
        >>> pairs_bb = fx.slice_pairwise_test(
        ...     ic, per_sector_ic, label="sector",
        ...     estimator=BlockBootstrap(rng_seed=0),
        ... )
    """
    est = _resolve_estimator(estimator, "slice_pairwise_test")

    labels, panel, n_obs = _build_per_date_panel(
        metric, df, label, func_name="slice_pairwise_test"
    )
    k = panel.shape[1]
    pairs = list(combinations(range(k), 2))

    boot: np.ndarray | None
    observed_abs: np.ndarray | None
    if isinstance(est, BlockBootstrap):
        observed_abs, boot, _meta = _joint_block_bootstrap_pairwise_distribution(
            panel,
            pairs=pairs,
            block_length=est.block_length,
            n_resamples=est.n_resamples,
            scheme=est.scheme,
            rng_seed=est.rng_seed,
        )
        col_means = panel.mean(axis=0)
        stats = [float(col_means[i] - col_means[j]) for i, j in pairs]
        b = boot.shape[0]
        p_raw = [
            (int(np.sum(boot[:, p_idx] >= observed_abs[p_idx])) + 1) / (b + 1)
            for p_idx in range(len(pairs))
        ]
        if multiple_testing is None:
            multiple_testing = "romano_wolf"
    else:
        stats = []
        p_raw = []
        for i, j in pairs:
            restriction = np.zeros((1, k))
            restriction[0, i] = 1.0
            restriction[0, j] = -1.0
            stat, p = _wald_nw_cluster_means(panel, R=restriction)
            stats.append(stat)
            p_raw.append(p)
        boot = None
        observed_abs = None
        if multiple_testing is None:
            multiple_testing = "holm"

    if multiple_testing == "holm":
        p_adj = holm_step_down(p_raw)
    elif multiple_testing == "bonferroni":
        p_adj = bonferroni(p_raw)
    elif multiple_testing == "romano_wolf":
        if boot is None or observed_abs is None:
            raise ValueError(
                "slice_pairwise_test: multiple_testing='romano_wolf' "
                "requires a bootstrap distribution; pair it with "
                "estimator=BlockBootstrap(). Analytic estimators "
                "(WaldNWCluster) do not produce one."
            )
        p_adj = romano_wolf(observed_abs.tolist(), boot)
    else:
        raise ValueError(
            f"slice_pairwise_test: multiple_testing="
            f"{multiple_testing!r} not recognized; expected 'holm' / "
            f"'bonferroni' / 'romano_wolf' / None."
        )

    return pl.DataFrame(
        {
            "slice_a": [labels[i] for i, _ in pairs],
            "slice_b": [labels[j] for _, j in pairs],
            "n_obs": [n_obs] * len(pairs),
            "stat": stats,
            "p_raw": p_raw,
            "p_adj": list(p_adj),
        }
    )


def slice_joint_test(
    metric: Callable,
    df: pl.DataFrame,
    *,
    label: str,
    estimator: Estimator | None = None,
) -> pl.DataFrame:
    """Omnibus Wald χ² that all K slice means are equal.

    The joint restriction is ``β_0 = β_1 = … = β_{K-1}``, encoded as
    ``K-1`` contrasts against the first slice; the Wald statistic
    follows χ²_{K-1} under H₀.

    Args:
        metric: Metric callable whose module declares ``per_date_series``.
        df: Input frame for the metric, containing ``label``.
        label: Column whose values define the slice partition.
        estimator: Inference estimator. ``None`` resolves to
            :class:`WaldNWCluster`. Other estimators raise
            ``NotImplementedError`` pending follow-up batches.

    Returns:
        Single-row ``pl.DataFrame`` with columns
        ``(n_obs, k_slices, df, stat, p)``. ``df`` is the restriction
        rank (``K-1``); ``stat`` is the joint Wald χ²; ``p`` is the
        chi-squared survival function. No ``multiple_testing`` kwarg —
        a single omnibus has no family-internal correction to apply.

    Raises:
        ValueError: Fewer than two slice values, or fewer than two
            dates aligned across all slices.
        TypeError: Metric is not slice-test-eligible (no
            ``per_date_series`` capability).
        NotImplementedError: Non-``WaldNWCluster`` estimator passed
            before the block-bootstrap batch lands.

    Examples:
        Joint omnibus test that mean IC is identical across two
        sub-universes (see :func:`slice_pairwise_test` for the
        per-sector ic panel construction):

        >>> import polars as pl
        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics import ic, compute_ic
        >>> raw = fx.datasets.make_cs_panel(n_assets=100, n_dates=500)
        >>> panel = compute_forward_return(raw, forward_periods=5)
        >>> assets = panel["asset_id"].unique().sort().to_list()
        >>> half = len(assets) // 2
        >>> sector_map = {a: ("tech" if i < half else "fin")
        ...               for i, a in enumerate(assets)}
        >>> panel_sec = panel.with_columns(
        ...     pl.col("asset_id").replace_strict(sector_map).alias("sector")
        ... )
        >>> per_sector_ic = pl.concat([
        ...     compute_ic(panel_sec.filter(pl.col("sector") == s))
        ...        .with_columns(pl.lit(s).alias("sector"))
        ...     for s in ("tech", "fin")
        ... ])
        >>> joint = fx.slice_joint_test(ic, per_sector_ic, label="sector")
    """
    est = _resolve_estimator(estimator, "slice_joint_test")
    if isinstance(est, BlockBootstrap):
        raise NotImplementedError(
            "slice_joint_test: BlockBootstrap on the omnibus χ² is not "
            "implemented; the joint Wald χ² has no canonical bootstrap "
            "analogue here. Use WaldNWCluster (default) for the omnibus, "
            "or slice_pairwise_test + BlockBootstrap for pairwise contrasts."
        )

    _, panel, n_obs = _build_per_date_panel(
        metric, df, label, func_name="slice_joint_test"
    )
    k = panel.shape[1]

    # K-1 contrasts against slice 0: rows are [1, -1, 0, …], [1, 0, -1, …], …
    restriction = np.zeros((k - 1, k))
    restriction[:, 0] = 1.0
    for r in range(k - 1):
        restriction[r, r + 1] = -1.0

    stat, p = _wald_nw_cluster_means(panel, R=restriction)

    return pl.DataFrame(
        {
            "n_obs": [n_obs],
            "k_slices": [k],
            "df": [k - 1],
            "stat": [stat],
            "p": [p],
        }
    )
