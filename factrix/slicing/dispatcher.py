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

import dataclasses
from typing import TYPE_CHECKING

import polars as pl

from factrix._codes import WarningCode, _emit_warning, _validate_expected_warnings_arg
from factrix._data_input import DataInput, _coerce_data, _read_horizon_stamps
from factrix._results import Warning
from factrix.slicing._primitive import _slice_by

_DOCS_BY_SLICE = "api/by-slice"

if TYPE_CHECKING:
    from factrix._results import EvaluationResult
    from factrix.metrics._base import MetricBase


def by_slice(
    data: pl.DataFrame,
    metric: MetricBase,
    *,
    by: str,
    factor_col: str,
    price_data: DataInput | None = None,
    forward_periods: int | None = None,
    overlap_periods: int | None = None,
    strict: bool = True,
    expected_warnings: tuple[str, ...] = (),
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
            ``ic()``, ``caar()``), consistent with
            :func:`factrix.evaluate`. The bare class (``ic``) is rejected.
        by: Column name in ``data`` whose distinct values define the
            slices. For cross-product slicing (e.g. market × sector)
            compose a single composite column upstream
            (``pl.concat_str([...]).alias("...")``).
        factor_col: The factor column to evaluate. Single-factor by
            design — multi-factor / multi-metric batching is the job of
            :func:`factrix.evaluate`.
        price_data: Optional complete ``date, asset_id, price`` panel,
            forwarded to every per-slice ``evaluate`` call **restricted to
            that slice's own assets**. The slice's ``data`` rows remain the
            sole owner of event eligibility and the forward-return sample;
            this side panel only supplies the complete price grid for event
            offsets and excursion windows. Consequently, ``offsets=`` and
            ``window=`` count periods on this price grid when it is
            supplied, just as under :func:`factrix.evaluate`.

            The restriction is by asset, never by date: an asset keeps its
            whole price history in every slice it appears in, so a path that
            crosses a slice boundary in time is still walked. Forwarding the
            panel whole instead would hand each slice quantities formed from
            assets it does not contain — the unconditional baseline
            ``event_around_return`` subtracts, and the raggedness verdict
            its warning reports — so passing prices would silently change
            what a slice is measured against.
        forward_periods: The data's return horizon, forwarded to
            ``evaluate`` on every per-slice call. Normally omitted — it is
            read from the panel's ``compute_forward_return`` stamp (which
            survives partitioning). Pass it only to declare the horizon for a
            self-attached ``forward_return`` panel that carries no stamp.
        overlap_periods: The evaluation-grid overlap, forwarded to
            ``evaluate`` alongside ``forward_periods`` with the same
            stamp-first contract; for an unstamped panel it defaults to the
            horizon.
        strict: Forwarded to ``evaluate``. ``True`` (default) raises if
            the metric is inapplicable to a slice; ``False`` surfaces a
            NaN result with a warning.
        expected_warnings: :class:`~factrix.WarningCode` values declaring
            warning regimes that are the study's **design** — the same
            contract as :func:`factrix.evaluate`, and forwarded to it for
            every slice. Also covers this function's own
            ``slice_boundary_truncation`` record: declared codes are kept on
            every slice's ``warnings`` with ``expected=True`` and only the
            ``UserWarning`` echo stops. Unknown codes are rejected.

    Returns:
        ``dict[str, EvaluationResult]`` — the same shape as
        :func:`factrix.evaluate`, keyed by stringified slice value (an
        ``Int64`` decile column yields ``"1".."10"``) rather than factor.
        Inside each bundle the metric is keyed by its **registered spec
        name** (``ic()`` → ``"ic"``), so stacked ``to_frame`` rows from two
        ``by_slice`` runs stay distinguishable.
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

    expected = _validate_expected_warnings_arg(
        expected_warnings, func_name="by_slice", docs_path=_DOCS_BY_SLICE
    )
    # Validate the data-level horizon contract before partitioning. Otherwise
    # each slice can carry one internally constant but mutually different stamp,
    # hiding a mixed-horizon input from the per-slice evaluate calls.
    _read_horizon_stamps(data, func_name="by_slice")
    # Coerce once here rather than per slice: the structural gate is the same
    # for every slice, and the frame has to be eager to be restricted below.
    # Column and dtype validation stays with ``evaluate``, which owns the
    # price-panel contract and reports it against the slice it applies to.
    prices = (
        None
        if price_data is None
        else _coerce_data(price_data, func_name="by_slice", role="price_data")
    )
    sliced = _slice_by(data, by)
    label = _metric_label(metric)
    truncation = _warn_date_axis_truncation(
        data, metric, by, expected_warnings=expected
    )

    results: dict[str, EvaluationResult] = {}
    for key, sub_df in sliced.items():
        bundle = factrix.evaluate(
            sub_df,
            price_data=_slice_price_data(prices, sub_df),
            metrics={label: metric},
            factor_cols=[factor_col],
            forward_periods=forward_periods,
            overlap_periods=overlap_periods,
            strict=strict,
            expected_warnings=expected,
        )
        result = bundle[factor_col]
        if truncation is not None:
            result = dataclasses.replace(
                result, warnings=[*result.warnings, truncation]
            )
        results[key] = result
    return results


def _slice_price_data(
    prices: pl.DataFrame | None, sub_df: pl.DataFrame
) -> pl.DataFrame | None:
    """Restrict a price panel to the assets present in one slice.

    ``price_data`` exists to complete the *price grid* — the periods an
    evaluation panel dropped. It is not a second source of assets. Every
    quantity a metric forms from the price panel rather than from the events
    (the unconditional baseline in ``event_around_return``, the raggedness
    the grid warning reports) would otherwise be formed from assets this
    slice does not contain, and passing prices would change what the slice is
    measured against without saying so.

    Restriction is by asset only. Dropping periods as well would put back the
    truncation the panel was passed to repair.

    A panel with no ``asset_id`` column is forwarded unchanged so that
    ``evaluate`` raises the one canonical price-panel error against the slice
    it applies to, rather than this helper failing first with a polars error.
    """
    if prices is None or "asset_id" not in prices.columns:
        return prices
    return prices.filter(
        pl.col("asset_id").is_in(sub_df["asset_id"].unique().implode())
    )


def _metric_label(metric: MetricBase) -> str:
    """Registered spec name of ``metric`` — the key its results are stored under.

    ``by_slice`` keys the returned ``EvaluationResult`` by slice value, so the
    per-slice ``metrics`` dict is the only place the metric's identity appears.
    Using the spec name (rather than a fixed placeholder) makes stacked
    ``to_frame`` rows from two ``by_slice`` runs distinguishable, and matches
    what ``evaluate`` does when the caller passes no explicit label.
    """
    try:
        return type(metric).spec().name
    except (AttributeError, TypeError):
        # Not a metric instance — ``evaluate`` raises the canonical error below;
        # the placeholder only has to survive until then.
        return "metric"


def _warn_date_axis_truncation(
    data: pl.DataFrame,
    metric: MetricBase,
    by: str,
    *,
    expected_warnings: tuple[str, ...] = (),
) -> Warning | None:
    """Warn when a cross-date metric is sliced on a date axis.

    Emits :class:`~factrix._codes.WarningCode.SLICE_BOUNDARY_TRUNCATION`
    only when both hold: (1) the metric declares
    ``MetricSpec.slice_boundary_sensitive``; (2) ``by`` is a date-axis partition —
    its value varies within an asset over time, so partitioning truncates
    each asset's history. A cross-sectional ``by`` (constant within an
    asset) keeps history intact and does not warn.

    Returns the :class:`~factrix._results.Warning` record ``by_slice`` attaches
    to every slice's :class:`~factrix._results.EvaluationResult` (``None`` when
    the condition does not hold), alongside the ``UserWarning`` echo. The echo
    alone was unreadable programmatically: the condition is a property of the
    ``(metric, by)`` pair, so it applies to every slice, and a caller scanning
    ``result.warnings`` must see it there like any other bundle warning.

    ``expected_warnings`` is the caller's study-level declaration, the same
    contract as :func:`factrix.evaluate`: a declared code keeps its record —
    flagged ``expected=True`` — and only the ``UserWarning`` echo stops.
    """
    try:
        spec = type(metric).spec()
    except (AttributeError, TypeError):
        return None  # not a metric instance; evaluate raises the canonical error
    if not spec.slice_boundary_sensitive:
        return None
    if "asset_id" not in data.columns:
        return None  # cannot classify the axis without an asset dimension
    n_assets = data.select("asset_id").n_unique()
    n_asset_by_pairs = data.select("asset_id", by).n_unique()
    if n_asset_by_pairs <= n_assets:
        return None  # by is constant within each asset → cross-sectional
    name = spec.name
    message = (
        f"{name!r} depends on intact date ordering, "
        f"but {by!r} is a date-axis partition (its value varies within an "
        f"asset over time). Each slice is evaluated on its own rows only, so "
        f"rolling windows / per-asset time-series regressions / event windows "
        f"see truncated history at slice boundaries — the per-slice value "
        f"differs from the full-sample value decomposed by period. For a cross-"
        f"sectional partition (constant within an asset, e.g. sector) this "
        f"warning does not apply."
    )
    expected = WarningCode.SLICE_BOUNDARY_TRUNCATION.value in expected_warnings
    _emit_warning(
        WarningCode.SLICE_BOUNDARY_TRUNCATION,
        message,
        label="by_slice",
        expected_warnings=expected_warnings,
        stacklevel=3,
    )
    return Warning(
        code=WarningCode.SLICE_BOUNDARY_TRUNCATION,
        source=name,
        message=message,
        expected=expected,
    )
