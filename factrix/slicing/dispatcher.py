"""Axis-agnostic slice dispatcher.

Public :func:`by_slice` partitions a raw panel on an existing column and
runs the standard :func:`factrix.evaluate` pipeline independently on each
slice — the cross-slice counterpart of ``evaluate``. It returns the same
``dict[str, EvaluationResult]`` shape as ``evaluate`` (keyed by slice
value rather than factor). Universe-overlap composition is user-side; see
``docs/api/by-slice.md`` for reference patterns.

Matrix-row: by_slice | (*, *, *, *) | dispatcher | none (no cross-slice test) | _slice_by
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import polars as pl

from factrix._codes import WarningCode
from factrix.slicing._primitive import _slice_by

if TYPE_CHECKING:
    from factrix._results import EvaluationResult
    from factrix.metrics._base import MetricBase

# Single-metric label used inside the per-slice evaluate call.
_METRIC_LABEL = "metric"


def by_slice(
    data: pl.DataFrame,
    metric: MetricBase,
    *,
    by: str,
    factor_col: str,
    forward_periods: int | None = None,
    strict: bool = True,
) -> dict[str, EvaluationResult]:
    """Partition ``data`` by ``by`` and run :func:`factrix.evaluate` per slice.

    ``by_slice`` is the cross-slice counterpart of :func:`factrix.evaluate`:
    it partitions a raw panel on a column, evaluates ``metric`` on each
    slice **independently** (the full producer→consumer DAG runs per
    slice, so DAG-consumer metrics work with no pre-computation), and
    returns the per-slice results for comparison. It does no cross-slice
    statistical inference; for paired / omnibus contrasts see
    :func:`factrix.slice_pairwise_test` / :func:`factrix.slice_joint_test`.

    Each slice is evaluated as an independent dataset — it sees only its
    own rows. For **cross-sectional** partitions (sector, size bucket;
    the partition value is constant within an asset) this is exactly the
    intent: each slice is an independent universe with intact per-asset
    history. For **date-axis** partitions (year, regime; the value varies
    within an asset over time) a metric whose aggregation looks across
    dates — rolling-window betas, per-asset time-series regressions, event
    windows (``common_beta``, ``mfe_mae``, ``oos_decay``, …) — sees truncated
    history at slice boundaries, so its per-slice value differs from the
    full-sample value decomposed by period. Per-date metrics (``ic``,
    ``fm_beta``, ``quantile``) are unaffected. A
    :class:`~factrix._codes.WarningCode.SLICE_BOUNDARY_TRUNCATION` warning
    is emitted when a cross-date metric is sliced on a date axis.

    Args:
        data: Raw long-format panel — same input contract as
            :func:`factrix.evaluate` (``date, asset_id, <factor_col>,
            forward_return``; ``forward_return`` already attached via
            :func:`factrix.preprocess.compute_forward_return`). Must
            contain ``by`` as a column; compose it upstream if needed
            (``data.with_columns(...)`` or a join).
        metric: A metric **instance** from :mod:`factrix.metrics` (e.g.
            ``ic()``, ``caar(forward_periods=5)``), consistent with
            :func:`factrix.evaluate`. The bare class (``ic``) is rejected.
        by: Column name in ``data`` whose distinct values define the
            slices. For cross-product slicing (e.g. market × sector)
            compose a single composite column upstream
            (``pl.concat_str([...]).alias("...")``).
        factor_col: The factor column to evaluate. Single-factor by
            design — multi-factor / multi-metric batching is the job of
            :func:`factrix.evaluate`.
        forward_periods: The data's overlap horizon, forwarded to
            ``evaluate`` on every per-slice call. Normally omitted — it is
            read from the panel's ``compute_forward_return`` stamp (which
            survives partitioning). Pass it only to declare the horizon for a
            self-attached ``forward_return`` panel that carries no stamp.
        strict: Forwarded to ``evaluate``. ``True`` (default) raises if
            the metric is inapplicable to a slice; ``False`` surfaces a
            NaN result with a warning.

    Returns:
        ``dict[str, EvaluationResult]`` — the same shape as
        :func:`factrix.evaluate`, keyed by stringified slice value (an
        ``Int64`` decile column yields ``"1".."10"``) rather than factor.
        Iteration order matches polars ``partition_by(as_dict=True)``. For
        a cross-slice comparison table, stack the per-slice frames with
        the standard ``EvaluationResult.to_frame`` idiom and tag each row
        with its slice key::

            pl.concat([
                r.to_frame().with_columns(pl.lit(k).alias("slice"))
                for k, r in result.items()
            ])

        No cross-slice statistical inference — see API page.

    Raises:
        TypeError: ``data`` is not a polars DataFrame.
        ValueError: ``by`` not in ``data.columns``, or ``data`` is empty.
        UserInputError: ``metric`` is not a metric instance, or
            ``factor_col`` is absent / invalid (raised by ``evaluate``).

    Examples:
        Per-sector information coefficient (IC) on a synthetic
        cross-sectional panel — partition on a sector column, evaluate
        ``ic`` independently within each sector:

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
        >>> per_sector = fx.by_slice(panel, ic(), by="sector", factor_col="factor")
        >>> set(per_sector) == {"tech", "fin"}
        True
    """
    import factrix  # local import: evaluate lives at top level (import cycle)

    sliced = _slice_by(data, by)
    _warn_date_axis_truncation(data, metric, by)

    results: dict[str, EvaluationResult] = {}
    for key, sub_df in sliced.items():
        bundle = factrix.evaluate(
            sub_df,
            metrics={_METRIC_LABEL: metric},
            factor_cols=[factor_col],
            forward_periods=forward_periods,
            strict=strict,
        )
        results[key] = bundle[factor_col]
    return results


def _warn_date_axis_truncation(data: pl.DataFrame, metric: MetricBase, by: str) -> None:
    """Warn when a cross-date metric is sliced on a date axis.

    Emits :class:`~factrix._codes.WarningCode.SLICE_BOUNDARY_TRUNCATION`
    only when both hold: (1) the metric declares
    ``MetricSpec.slice_boundary_sensitive``; (2) ``by`` is a date-axis partition —
    its value varies within an asset over time, so partitioning truncates
    each asset's history. A cross-sectional ``by`` (constant within an
    asset) keeps history intact and does not warn.
    """
    try:
        spec = type(metric).spec()
    except (AttributeError, TypeError):
        return  # not a metric instance; evaluate raises the canonical error
    if not spec.slice_boundary_sensitive:
        return
    if "asset_id" not in data.columns:
        return  # cannot classify the axis without an asset dimension
    n_assets = data.select("asset_id").n_unique()
    n_asset_by_pairs = data.select("asset_id", by).n_unique()
    if n_asset_by_pairs <= n_assets:
        return  # by is constant within each asset → cross-sectional
    name = spec.name
    warnings.warn(
        f"by_slice: {name!r} depends on intact date ordering, "
        f"but {by!r} is a date-axis partition (its value varies within an "
        f"asset over time). Each slice is evaluated on its own rows only, so "
        f"rolling windows / per-asset time-series regressions / event windows "
        f"see truncated history at slice boundaries — the per-slice value "
        f"differs from the full-sample value decomposed by period "
        f"({WarningCode.SLICE_BOUNDARY_TRUNCATION.value}). For a cross-"
        f"sectional partition (constant within an asset, e.g. sector) this "
        f"warning does not apply.",
        UserWarning,
        stacklevel=2,
    )
