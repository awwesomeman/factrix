"""Slice-test policy helpers — pre-flight subset detection + ``n_groups`` downscale.

Two private helpers consumed by the slice-test functions
(``slice_pairwise_test`` / ``slice_joint_test``):

- ``_detect_strict_subsets`` — pre-flight check for paired tests.
  When slice A's (date, asset) key-set is strictly contained in slice
  B's, the pairwise diff inherits A's universe entirely; the SE
  estimate has degeneracies the standard cluster-NW formula does not
  warn about. The slice-test function emits a ``UserWarning`` on
  non-empty output and steers the caller toward
  ``slice_joint_test`` (omnibus over all slices) which handles the
  geometry properly.
- ``_downscale_n_groups`` — for slice tests on metrics that bucket
  cross-section assets (quantile spread, monotonicity), shrinks
  ``n_groups`` so each bucket retains at least ``min_assets_per_group``
  assets. The per-metric ``min_assets_per_group`` floor lives on the metric
  module itself; the slice-test function resolves it per
  slice and feeds it here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl


def _detect_strict_subsets(
    slices: Mapping[str, pl.DataFrame],
    *,
    key_cols: Sequence[str] = ("date", "asset"),
) -> list[tuple[str, str]]:
    """Pairs ``(subset_label, superset_label)`` of strictly nested slices.

    Args:
        slices: Per-label slice DataFrames; keys are slice labels,
            values share a common schema and contain ``key_cols``.
        key_cols: Column names whose unique combinations define a
            slice's "membership". Default ``("date", "asset")`` matches
            the canonical panel key.

    Returns:
        List of ordered pairs ``(a, b)`` where ``key_set(a) ⊊
        key_set(b)``. Pairs are emitted once per direction (no double
        report); equal key-sets are not flagged (those are the same
        slice from the test's perspective). Empty list = all pairs are
        either disjoint or partially-overlapping, both safe inputs to
        ``slice_pairwise_test``.

    Raises:
        ValueError: A slice is missing one of ``key_cols``.
    """
    key_cols = tuple(key_cols)
    key_sets: dict[str, frozenset[tuple]] = {}
    for label, data in slices.items():
        missing = [c for c in key_cols if c not in data.columns]
        if missing:
            raise ValueError(
                f"slice {label!r}: missing key columns {missing}; have {data.columns}."
            )
        keys = data.select(list(key_cols)).unique().rows()
        key_sets[label] = frozenset(keys)

    labels = list(slices.keys())
    results: list[tuple[str, str]] = []
    for i, a in enumerate(labels):
        sa = key_sets[a]
        for b in labels[i + 1 :]:
            sb = key_sets[b]
            if sa < sb:
                results.append((a, b))
            elif sb < sa:
                results.append((b, a))
    return results


def _downscale_n_groups(
    base_n_groups: int,
    n_assets: int,
    *,
    min_assets_per_group: int | None,
) -> int:
    """Cap ``n_groups`` so each bucket retains ``≥ min_assets_per_group`` assets.

    Args:
        base_n_groups: Caller's requested bucket count.
        n_assets: Number of distinct assets in the slice.
        min_assets_per_group: Minimum assets per bucket required for the metric
            to be statistically meaningful (e.g. IC quintile spread:
            30; monotonicity rho: 50). ``None`` skips the downscale —
            for metrics that don't bucket cross-section (Fama-MacBeth,
            CAAR), pass ``None`` and ``base_n_groups`` returns unchanged.

    Returns:
        ``min(base_n_groups, max(2, n_assets // min_assets_per_group))`` when
        ``min_assets_per_group`` is set; ``base_n_groups`` otherwise. The lower
        floor of 2 prevents a 1-bucket "spread" (degenerate) on very
        small slices — the slice test should ideally be skipped at
        that point but the helper does not raise.

    Raises:
        ValueError: ``min_assets_per_group < 1``.
    """
    if min_assets_per_group is None:
        return base_n_groups
    if min_assets_per_group < 1:
        raise ValueError(
            f"min_assets_per_group must be >= 1; got {min_assets_per_group!r}."
        )
    capacity = max(2, n_assets // min_assets_per_group)
    return min(base_n_groups, capacity)
