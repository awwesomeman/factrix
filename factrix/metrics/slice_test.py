"""Cross-slice statistical-test verbs (#176).

``slice_pairwise_test`` reports K(K-1)/2 cross-slice contrasts as a
long-form DataFrame of pairwise ``(stat, p_raw, p_adj)``. The
companion ``slice_joint_test`` (omnibus χ²) lands in a follow-up.

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
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from typing import Literal

import numpy as np
import polars as pl

from factrix._stats.multiple_testing import bonferroni, holm_step_down
from factrix._stats.wald import _wald_nw_cluster_means
from factrix.metrics._metric_capabilities import resolve_per_date_series
from factrix.metrics._slice import _slice_by_label
from factrix.stats import Estimator, WaldNWCluster

MultipleTestingMethod = Literal["holm", "bonferroni"]


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
            :class:`WaldNWCluster`. Other estimators raise
            ``NotImplementedError`` pending follow-up batches.
        multiple_testing: P-value adjustment family. ``None`` falls
            back to ``"holm"`` (conservative under arbitrary
            dependence). Romano-Wolf (block-bootstrap step-down) is
            scheduled for a follow-up batch.

    Returns:
        Long-form ``pl.DataFrame`` with columns
        ``(slice_a, slice_b, n_obs, stat, p_raw, p_adj)``; one row per
        ordered slice pair ``(a, b)`` with ``a`` before ``b`` in the
        partition's iteration order.

    Raises:
        ValueError: Fewer than two slice values, or fewer than two
            dates aligned across all slices.
        TypeError: Metric is not slice-test-eligible (no
            ``per_date_series`` capability).
        NotImplementedError: Non-``WaldNWCluster`` estimator passed
            before the Romano-Wolf / block-bootstrap batch lands.
    """
    if estimator is None:
        estimator = WaldNWCluster()
    if not isinstance(estimator, WaldNWCluster):
        raise NotImplementedError(
            f"slice_pairwise_test: estimator "
            f"{type(estimator).__name__!r} not yet wired; this verb "
            f"currently supports WaldNWCluster only. BlockBootstrap "
            f"(Romano-Wolf path) lands in a follow-up batch."
        )
    if multiple_testing is None:
        multiple_testing = "holm"

    per_date_fn = resolve_per_date_series(metric)

    slices = _slice_by_label(df, label)
    if len(slices) < 2:
        raise ValueError(
            f"slice_pairwise_test: need ≥2 slice values on {label!r}; "
            f"got {len(slices)}."
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
            f"slice_pairwise_test: <2 aligned dates across slices "
            f"({aligned.height}); joint HAC inference requires aligned "
            f"rows. Check that slices share at least 2 common dates."
        )

    panel = aligned.drop("date").to_numpy()
    n_obs = aligned.height
    k = panel.shape[1]

    rows: list[tuple[str, str, float]] = []
    p_raw: list[float] = []
    for i, j in combinations(range(k), 2):
        restriction = np.zeros((1, k))
        restriction[0, i] = 1.0
        restriction[0, j] = -1.0
        stat, p = _wald_nw_cluster_means(panel, R=restriction)
        rows.append((labels[i], labels[j], stat))
        p_raw.append(p)

    if multiple_testing == "holm":
        p_adj = holm_step_down(p_raw)
    elif multiple_testing == "bonferroni":
        p_adj = bonferroni(p_raw)
    else:
        raise ValueError(
            f"slice_pairwise_test: multiple_testing="
            f"{multiple_testing!r} not recognized; expected 'holm' / "
            f"'bonferroni' / None."
        )

    return pl.DataFrame(
        {
            "slice_a": [r[0] for r in rows],
            "slice_b": [r[1] for r in rows],
            "n_obs": [n_obs] * len(rows),
            "stat": [r[2] for r in rows],
            "p_raw": p_raw,
            "p_adj": list(p_adj),
        }
    )
