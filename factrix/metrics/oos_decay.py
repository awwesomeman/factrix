"""Out-of-sample (OOS) persistence analysis for any time-indexed series.

This tool is agnostic to what the series represents — it only knows
about IS/OOS splits on a time-indexed numeric sequence.

Notes:
    **Pipeline.** Time-series only, IS/OOS window split on a 1-D series;
    descriptive decay diagnostic (no formal H_0). ``oos_decay`` is the
    single-split primitive; ``oos_decay_splits`` is the robustness
    workflow that runs it over a pre-declared set of split fractions.

    **Input.** DataFrame with ``date, value`` (IC series, CAAR series,
    spread series), one row per period.

    **Output.** MetricResult with ``value`` = survival ratio +
    sign-flip / status detail in ``metadata``.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Literal

import polars as pl

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    FactorScope,
    InputShape,
)
from factrix._codes import WarningCode
from factrix._metric_index import SampleThreshold, cell
from factrix._results import MetricResult
from factrix._types import DEFAULT_FORWARD_PERIODS, EPSILON, MIN_OOS_PERIODS_HARD
from factrix.metrics._base import MetricBase
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    DEGENERATE_SIGNAL_STATUS,
    _enforce_min_floor,
    _finite_expr,
    _finite_values,
    _resolve_series_value_col,
    _short_circuit_output,
    _surface_null_drop,
    _validate_half_open_unit_interval,
    _validate_open_unit_interval,
    _validate_positive_count,
)
from factrix.metrics.ic import compute_ic

__all__ = [
    "oos_decay",
    "oos_decay_splits",
]

GateStatus = Literal["PASS", "VETOED"]

# Minimum observations each side of the split must carry for its mean to be
# a window statistic rather than a single point.
_MIN_SPLIT_OBS = 2

#: Split set ``oos_decay_splits`` declares when the caller declares none.
#: The three fractions the docs have always recommended sweeping — wide
#: enough that a break has to sit near all three cut points to move the
#: median, small enough that each side stays a readable window.
DEFAULT_SPLIT_FRACTIONS: tuple[float, ...] = (0.6, 0.7, 0.8)

#: The aggregate rule. Fixed, not a knob: a choice of aggregate made after
#: seeing the per-split ratios is a model search, which is exactly what this
#: workflow exists to close off. The median is order-invariant and has a 50 %
#: breakdown point, so a break landing inside one cut point cannot carry the
#: verdict; the mean would.
_AGGREGATE_RULE = "median"

#: The sign-flip rule. A flip at any declared split vetoes: an out-of-sample
#: sign reversal says the factor predicts the wrong direction over some
#: contiguous tail, and the single-split primitive already treats that as a
#: hard veto rather than a weak survival ratio. Aggregating direction the way
#: magnitude is aggregated would launder that hard veto into a majority vote.
_SIGN_FLIP_POLICY = "any_flip_vetoes"


def _validate_oos_decay(m: MetricBase) -> None:
    """Both gate knobs are fractions, checked before any data work.

    ``is_ratio`` splits the series, so it is strictly inside ``(0, 1)``;
    ``survival_threshold`` is the share of in-sample magnitude the factor must
    retain, so it is inside ``(0, 1]``.
    """
    _validate_open_unit_interval(
        m.is_ratio,  # type: ignore[attr-defined]
        func_name="oos_decay",
        field="is_ratio",
        detail=(
            "0 leaves no in-sample window and 1 leaves no out-of-sample "
            "window, so no survival ratio is defined."
        ),
        docs_path="api/metrics/oos_decay",
    )
    # WHY: an unvalidated threshold silently forces the gate rather than
    # failing. The survival ratio is a non-negative magnitude ratio, so any
    # value <= 0 PASSes every series a ratio can be computed for (a gate that
    # cannot veto is not a gate), NaN fails every comparison and so VETOES
    # every series, and ``True`` is 1.0 to Python. ``> 1`` asks for
    # out-of-sample *amplification*, which is not the decay this diagnostic
    # gates on; ask for it with an explicit read of ``value`` instead.
    _validate_half_open_unit_interval(
        m.survival_threshold,  # type: ignore[attr-defined]
        func_name="oos_decay",
        field="survival_threshold",
        detail=(
            "It is the share of the in-sample mean magnitude the factor must "
            "retain out of sample; 0 passes every series and a value above 1 "
            "demands out-of-sample amplification rather than survival."
        ),
        docs_path="api/metrics/oos_decay",
    )


def _require_one_row_per_period(series: pl.DataFrame) -> None:
    """Reject a series carrying more than one observation per distinct date.

    The split is a *period* count on the series' own distinct-date grid, taken
    positionally after sorting. A duplicated date has no defined position in
    that sort, so the cut can fall between two observations of the same period
    and put one chronological period on both sides of the IS/OOS boundary —
    the leakage the split exists to prevent. Rejected rather than aggregated:
    the DAG producers (``compute_ic``, ``compute_spread_series``) emit exactly
    one row per period, so no aggregation rule is needed to match them, and
    picking one silently (mean? last?) would apply a statistic the caller
    never asked for. Follows ``spanning``'s treatment of the same input shape.
    """
    n_periods = series["date"].n_unique()
    if n_periods == series.height:
        return
    from factrix._errors import UserInputError

    raise UserInputError(
        func_name="oos_decay",
        field="series",
        value=f"{series.height} rows for {n_periods} distinct periods",
        expected=(
            "one row per period on the series' distinct-date grid. A "
            "duplicated date makes the IS/OOS cut ambiguous, so the same "
            "period can land on both sides of the split. Aggregate to one "
            "observation per period first, e.g. "
            'series.group_by("date").mean().sort("date")'
        ),
        docs_path="api/metrics/oos_decay",
    )


@metric(
    cell=cell(
        FactorScope.INDIVIDUAL, FactorDensity.DENSE, structure=DataStructure.PANEL
    ),
    aggregation=Aggregation.TS_ONLY,
    slice_boundary_sensitive=True,
    input_shape=InputShape.SERIES,
    requires={"series": compute_ic},
    sample_threshold=SampleThreshold(min_periods=MIN_OOS_PERIODS_HARD * 2),
    validate=_validate_oos_decay,
)
def oos_decay(
    series: pl.DataFrame,
    value_col: str = "value",
    is_ratio: float = 0.7,
    survival_threshold: float = 0.5,
    *,
    expected_warnings: tuple[str, ...] = (),
) -> MetricResult:
    """Single-split out-of-sample (OOS) survival ratio with sign-flip detection.

    Splits the sorted series at ``is_ratio`` (IS = first ``is_ratio * n``
    rows, OOS = remainder), computes ``|mean_OOS| / |mean_IS|`` (the
    survival ratio), and checks for an IS/OOS sign flip.

    Args:
        series: DataFrame with ``date`` and ``value_col``, carrying exactly
            **one row per period** on its own distinct-date grid (what every
            producer in the DAG emits). Row order is irrelevant — the series
            is sorted by date here.
        value_col: Numeric column to evaluate. Null, NaN and ±inf
            observations are dropped and the drop is recorded in
            ``metadata``; the periods that remain are the split's grid.
        is_ratio: Fraction of the retained periods allocated to IS (default
            ``0.7``). Must lie strictly inside ``(0, 1)``.
        survival_threshold: Minimum survival ratio for ``status="PASS"``
            (default ``0.5``). It is the share of the in-sample mean
            magnitude the factor must retain out of sample, so its domain is
            the finite half-open interval ``(0, 1]``.

    Raises:
        UserInputError: ``is_ratio`` is not strictly inside ``(0, 1)``, or
            ``survival_threshold`` is not inside ``(0, 1]`` (bool, NaN, ±inf
            and non-numeric included). Raised at construction, before any
            data work.
        UserInputError: ``series`` carries more than one row for some date.
            Raised at call time, before the split.

    Returns:
        MetricResult with:

        - ``value``: survival ratio ``|mean_oos| / |mean_is|``. NaN on a
        short-circuit, and NaN when ``|mean_is| ~ 0``: the ratio is an
        undefined 0/0 there, not a survival of zero. Reporting ``0.0``
        VETOED a factor with a large out-of-sample mean as "fully
        decayed" when the honest reading is "in-sample carried no signal
        to decay from". That case raises
        ``WarningCode.DEGENERATE_VARIANCE`` and keeps ``status="VETOED"``
        — the gate must not read "cannot assess" as "passed".
        - ``stat``: ``None`` — descriptive only (no hypothesis test
        attached; a t-stat at ``MIN_OOS_PERIODS_HARD = 5`` would have power
        ~ 0 and would invite mis-reading the diagnostic as a
        significance test)
        - ``metadata``:

            - ``sign_flipped`` (bool)
            - ``status`` (``"PASS"`` | ``"VETOED"``)
            - ``is_ratio`` (float)
            - ``mean_is`` (float)
            - ``mean_oos`` (float)
            - ``survival_threshold`` (float)
            - ``reason`` (str, short-circuit only):
              ``"insufficient_oos_periods"``

    Notes:
        For multi-fraction sweeps, call ``oos_decay`` per fraction and
        aggregate on the caller side::

            results = {f: oos_decay(series, is_ratio=f) for f in (0.6, 0.7, 0.8)}
            median = statistics.median(r.value for r in results.values())

        Descriptive only — no ``p_value`` is emitted.

        **Split validity.** ``is_ratio`` must be strictly inside ``(0, 1)``
        and is validated up front: ``is_ratio=1.0`` used to produce an empty
        OOS slice whose polars ``mean()`` is ``None``, and ``float(None)``
        then raised a bare ``TypeError`` from deep inside the metric. Beyond
        that, the ``min_periods`` floor bounds the *series length*, not the
        split, so an extreme ratio can still leave one side with fewer than
        two observations; that short-circuits with the usual
        ``reason="insufficient_oos_periods"`` rather than reporting a
        survival ratio computed from a single point.

        **Input contract.** The split is a count of periods on the series'
        own distinct-date grid, taken positionally after sorting by date, so
        the series must carry one row per period. A duplicated date has no
        defined position in that sort and can put the same chronological
        period on both sides of the boundary; it is rejected rather than
        aggregated under a guessed rule (see :class:`UserInputError` above).
        Non-finite observations are dropped first, so ``n_obs`` and the split
        index both count the periods that actually reached the means.

        **Threshold domain.** ``survival_threshold`` is validated as a finite
        fraction inside ``(0, 1]``. Outside that range the knob stops gating
        and starts forcing: ``<= 0`` PASSes every series a ratio exists for,
        ``float("nan")`` VETOES every one of them, and ``True`` silently
        reads as ``1.0``. A threshold above 1 would demand out-of-sample
        amplification rather than survival — read ``value`` directly for
        that question.

    References:
        - [McLean-Pontiff (2016)][mclean-pontiff-2016]: post-publication
          returns ~58% lower than in-sample, with ~32% of that drop
          attributable to publication itself (the remaining ~26% is the
          pure out-of-sample decay).
        - [Lopez-de-Prado (2018)][lopez-de-prado-2018]: CPCV for robust
          train/test split.

    Examples:
        Survival on a per-date information coefficient (IC) series from
        :func:`~factrix.metrics.ic.compute_ic`:

        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.ic import compute_ic
        >>> from factrix.metrics.oos_decay import oos_decay
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=240, rng=0),
        ...     forward_periods=5,
        ... )
        >>> series = compute_ic(panel)["factor"].rename({"ic": "value"}).select("date", "value")
        >>> result = oos_decay(series)
        >>> result.name == ""
        True
    """
    value_col = _resolve_series_value_col(series, value_col)
    _require_one_row_per_period(series)
    sorted_series = series.sort("date")
    # One row per period, so dropping the non-finite observations leaves one
    # value per surviving period: `n` counts periods on the series' own
    # distinct-date grid and the split index below is a period count.
    vals = _finite_values(sorted_series[value_col])
    n = len(vals)

    sc = _enforce_min_floor(
        oos_decay,
        "oos_decay",
        n,
        "insufficient_oos_periods",
        descriptive=True,
        sign_flipped=False,
        status="VETOED",
        is_ratio=is_ratio,
        survival_threshold=survival_threshold,
    )
    if sc is not None:
        return sc

    split_idx = int(n * is_ratio)

    # WHY: the min-periods floor bounds `n`, not the *split*. A lopsided
    # `is_ratio` (0.05, 0.98, ...) can still leave one side with 0 or 1
    # observation, where a mean is either undefined (polars returns None →
    # `float(None)` TypeError) or a single point masquerading as a window.
    if split_idx < _MIN_SPLIT_OBS or n - split_idx < _MIN_SPLIT_OBS:
        return _short_circuit_output(
            "oos_decay",
            "insufficient_oos_periods",
            n_obs=n,
            n_obs_axis="periods",
            descriptive=True,
            sign_flipped=False,
            status="VETOED",
            is_ratio=is_ratio,
            survival_threshold=survival_threshold,
        )

    is_vals = vals[:split_idx]
    oos_vals = vals[split_idx:]

    # Both slices carry >= _MIN_SPLIT_OBS observations, so polars mean()
    # returns a numeric.
    mean_is = float(is_vals.mean())  # type: ignore[arg-type]
    mean_oos = float(oos_vals.mean())  # type: ignore[arg-type]

    sign_flipped = (mean_is > 0 and mean_oos < 0) or (mean_is < 0 and mean_oos > 0)
    # ``mean_is ~ 0`` makes the ratio an undefined 0/0, not a survival of
    # zero. The former ``0.0`` VETOED a factor with a LARGE out-of-sample
    # mean as "fully decayed", when the honest reading is "in-sample had no
    # signal to decay from; the ratio is undefined". NaN + a warning code,
    # matching the repo's degenerate-sample convention.
    degenerate_is = abs(mean_is) < EPSILON
    survival = float("nan") if degenerate_is else abs(mean_oos) / abs(mean_is)

    status: GateStatus
    if degenerate_is:
        # No ratio to compare against the threshold. The gate stays VETOED -
        # it is a gate, and "cannot assess" must not read as "passed" - but
        # ``value`` is NaN and DEGENERATE_VARIANCE says why, matching the
        # ``insufficient_oos_periods`` short circuit above, which is also
        # VETOED-with-a-reason rather than a third status.
        status = "VETOED"
    elif sign_flipped:
        status = "VETOED"
    elif survival >= survival_threshold:
        status = "PASS"
    else:
        status = "VETOED"

    metadata: dict[str, object] = {
        "sign_flipped": sign_flipped,
        "status": status,
        "is_ratio": is_ratio,
        "mean_is": mean_is,
        "mean_oos": mean_oos,
        "survival_threshold": survival_threshold,
    }
    warning_codes: list[str] = []
    if degenerate_is:
        metadata["signal_status"] = DEGENERATE_SIGNAL_STATUS
        warning_codes.append(WarningCode.DEGENERATE_VARIANCE.value)
    _surface_null_drop(
        n_periods_in=sorted_series.height,
        n_periods_out=n,
        drop_reason="null / NaN / +-inf value observations in the series",
        metric_name="oos_decay",
        metadata=metadata,
        warning_codes=warning_codes,
        expected_warnings=expected_warnings,
    )
    return MetricResult(
        value=survival,
        n_obs=n,
        n_obs_axis="periods",
        stat=None,
        metadata=metadata,
        warning_codes=tuple(warning_codes),
    )


def _validate_oos_decay_splits(
    split_fractions: Sequence[float],
    survival_threshold: float,
    forward_periods: int,
) -> tuple[float, ...]:
    """Validate every knob of :func:`oos_decay_splits` at one site.

    The workflow is a plain function, so it has no ``@metric`` constructor
    hook to carry its knob contract; this is the equivalent single chokepoint,
    called before any data work, and the knob validators live here rather than
    scattered through the body for the same reason they live in the hook for
    every registered metric.

    Returns the declared split set in canonical (ascending) order, which is
    what makes the workflow's output independent of the order the caller
    declared them in. A repeated fraction is rejected rather than
    de-duplicated: it would double-weight one cut point inside the median
    without saying so.
    """
    from factrix._errors import UserInputError

    declared = tuple(split_fractions)
    if not declared:
        raise UserInputError(
            func_name="oos_decay_splits",
            field="split_fractions",
            value=declared,
            expected=(
                "a non-empty set of split fractions, declared before the "
                "sweep is run. An empty set has no aggregate to report."
            ),
            docs_path="api/metrics/oos_decay",
        )
    for fraction in declared:
        _validate_open_unit_interval(
            fraction,
            func_name="oos_decay_splits",
            field="split_fractions",
            detail=(
                "0 leaves no in-sample window and 1 leaves no out-of-sample "
                "window, so no survival ratio is defined at that cut point."
            ),
            docs_path="api/metrics/oos_decay",
        )
    if len(set(declared)) != len(declared):
        raise UserInputError(
            func_name="oos_decay_splits",
            field="split_fractions",
            value=declared,
            expected=(
                "distinct split fractions; a repeated fraction double-weights "
                "one cut point in the aggregate without changing the reported "
                "number of splits."
            ),
            docs_path="api/metrics/oos_decay",
        )
    _validate_half_open_unit_interval(
        survival_threshold,
        func_name="oos_decay_splits",
        field="survival_threshold",
        detail=(
            "It is the share of the in-sample mean magnitude the factor must "
            "retain out of sample, applied to the aggregate across splits."
        ),
        docs_path="api/metrics/oos_decay",
    )
    _validate_positive_count(
        forward_periods,
        func_name="oos_decay_splits",
        field="forward_periods",
        detail=(
            "It is the purge gap in periods on the distinct-date grid, taken "
            "from the forward-return window the series values were built "
            "from; a window shorter than one period does not exist."
        ),
        docs_path="api/metrics/oos_decay",
    )
    return tuple(sorted(declared))


def _run_one_split(
    clean: pl.DataFrame,
    value_col: str,
    *,
    fraction: float,
    purge_periods: int,
    survival_threshold: float,
    expected_warnings: tuple[str, ...],
) -> tuple[dict[str, object], tuple[str, ...]]:
    """Run the primitive at one cut point with the purge gap applied.

    *clean* carries one finite observation per period, sorted by date, so
    ``int(n * fraction)`` is a period index on the distinct-date grid. The
    purge drops the ``purge_periods`` periods immediately **before** that
    index — the in-sample tail whose forward-return windows reach into the
    out-of-sample side — and the primitive is then called on the surviving
    frame. Nothing is taken off the out-of-sample side: the window being
    validated is never shortened to protect the window it is validated
    against.

    The synthetic ``is_ratio`` handed to the primitive is
    ``(n_is + 0.5) / height``, which is the cut the primitive's own
    ``int(n * is_ratio)`` reproduces exactly — half a period of slack either
    way, against a floating-point error of a few ulps.
    """
    n = clean.height
    split_idx = int(n * fraction)
    is_end = split_idx - purge_periods
    if is_end < _MIN_SPLIT_OBS or n - split_idx < _MIN_SPLIT_OBS:
        # WHY a distinct reason: the primitive's ``insufficient_oos_periods``
        # means "the series is too short". Here the series may be ample and
        # the *purge* is what emptied the in-sample side, so reusing that
        # sentinel would send a reader looking for more data when what they
        # need is a nearer cut point or a shorter horizon.
        return {
            "split_fraction": fraction,
            "n_is": max(is_end, 0),
            "n_oos": n - split_idx,
            "purge_periods": purge_periods,
            "survival": float("nan"),
            "sign_flipped": False,
            "status": "VETOED",
            "reason": "purged_split_too_short",
        }, ()

    sub = pl.concat([clean.head(is_end), clean.slice(split_idx)])
    is_ratio = (is_end + 0.5) / sub.height
    result = oos_decay(
        sub,
        value_col=value_col,
        is_ratio=is_ratio,
        survival_threshold=survival_threshold,
        expected_warnings=expected_warnings,
    )
    # Report the split the primitive actually cut at, not the one requested.
    n_is = int(sub.height * is_ratio)
    record: dict[str, object] = {
        "split_fraction": fraction,
        "n_is": n_is,
        "n_oos": sub.height - n_is,
        "purge_periods": purge_periods,
        "survival": result.value,
        "sign_flipped": result.metadata["sign_flipped"],
        "status": result.metadata["status"],
        "mean_is": result.metadata.get("mean_is"),
        "mean_oos": result.metadata.get("mean_oos"),
    }
    if "reason" in result.metadata:
        record["reason"] = result.metadata["reason"]
    if "signal_status" in result.metadata:
        record["signal_status"] = result.metadata["signal_status"]
    return record, result.warning_codes


def oos_decay_splits(
    series: pl.DataFrame,
    value_col: str = "value",
    split_fractions: Sequence[float] = DEFAULT_SPLIT_FRACTIONS,
    survival_threshold: float = 0.5,
    forward_periods: int = DEFAULT_FORWARD_PERIODS,
    *,
    expected_warnings: tuple[str, ...] = (),
) -> MetricResult:
    """Purged out-of-sample (OOS) survival over a pre-declared set of splits.

    Robustness validation of :func:`oos_decay`, not a second search over it.
    A single cut point can reverse the single-split gate when a regime change
    lands near it; this runs the same primitive at every fraction in a set
    the caller declares **before** seeing any result, and reports one
    aggregate verdict plus every individual split.

    Args:
        series: DataFrame with ``date`` and ``value_col``, one row per period
            on its own distinct-date grid — the same contract
            :func:`oos_decay` enforces.
        value_col: Numeric column to evaluate.
        split_fractions: The declared cut points, each strictly inside
            ``(0, 1)`` and distinct (default ``(0.6, 0.7, 0.8)``). Order does
            not matter: the set is canonicalised ascending, and the aggregate
            is order-invariant.
        survival_threshold: Minimum **aggregate** survival ratio for
            ``status="PASS"`` (default ``0.5``), in ``(0, 1]``. Passed through
            unchanged to each split, so the per-split statuses read against
            the same bar.
        forward_periods: Length in periods of the forward-return window the
            series values were built from (default
            :data:`~factrix._types.DEFAULT_FORWARD_PERIODS`). This is the
            purge gap: the last ``forward_periods`` periods of each in-sample
            window are dropped, so no in-sample observation's return window
            reaches into the out-of-sample side.
        expected_warnings: Declared warning codes, passed to each split.

    Raises:
        UserInputError: ``split_fractions`` is empty, carries a duplicate, or
            carries a value outside ``(0, 1)``; ``survival_threshold`` is
            outside ``(0, 1]``; ``forward_periods`` is not an integer ``>= 1``;
            or ``series`` carries more than one row for some date.

    Returns:
        MetricResult with:

        - ``value``: the **median** survival ratio across the declared
        splits. NaN when any declared split could not be assessed — the
        aggregate is defined over the set that was declared, not over
        whichever subset happened to compute, and a silently narrowed
        sample would read as a verdict.
        - ``n_obs``: periods that entered the sweep (after the non-finite
        drop), on the ``"periods"`` axis.
        - ``stat``: ``None`` — descriptive, like the primitive. No
        ``p_value`` is emitted, so there is no multiplicity to correct.
        - ``metadata``:

            - ``status`` (``"PASS"`` | ``"VETOED"``)
            - ``splits``: one record per declared fraction, ascending, each
            with ``split_fraction``, ``n_is``, ``n_oos``, ``purge_periods``,
            ``survival``, ``sign_flipped``, ``status``, and (where the
            primitive emitted them) ``mean_is``, ``mean_oos``, ``reason``,
            ``signal_status``
            - ``split_fractions`` (tuple), ``n_splits``, ``n_assessable``,
            ``n_sign_flips``
            - ``aggregate`` (``"median"``), ``sign_flip_policy``
            (``"any_flip_vetoes"``)
            - ``forward_periods``, ``purge_periods``, ``survival_threshold``
            - ``reason`` (str, withheld aggregate only):
            ``"unassessable_splits"``

    Notes:
        **Aggregate rule.** ``status="PASS"`` requires all three of: every
        declared split assessable, **no** split sign-flipped, and the median
        survival ratio at or above ``survival_threshold``. Direction and
        magnitude are aggregated differently on purpose — a sign flip says
        the factor predicts the wrong way over some tail, which a majority
        vote would launder, while a magnitude below the bar at one cut point
        is exactly the arbitrariness the median is here to absorb. The median
        is fixed rather than exposed as a knob: choosing an aggregate after
        seeing the per-split ratios is the search this workflow closes off.

        ``value`` is still reported when the gate vetoes on a sign flip — it
        is the ratio that ran, and the veto is recorded in ``status`` and
        ``n_sign_flips`` rather than by corrupting the number.

        **Purge.** ``forward_periods`` periods are dropped off the end of
        each in-sample window, counted on the panel's distinct-date grid
        (never calendar time). A value stamped at period ``t`` built from a
        ``forward_periods``-period forward return is realised over
        ``(t, t + forward_periods]``, so without the gap the last in-sample
        observations are partly realised inside the out-of-sample window —
        the boundary leakage [Lopez-de-Prado (2018)][lopez-de-prado-2018]
        purges. The gap comes off the in-sample side only; the out-of-sample
        window is what is being validated and is never shortened for it. A
        series with no forward window (a contemporaneous spread) still
        passes ``forward_periods=1``, which drops one period — conservative,
        and cheaper than a knob that turns the guard off.

        **Multiplicity.** This is robustness validation, not an additional
        model search, and the distinction is what keeps it honest. The split
        set is declared up front, every split is reported, the aggregate is
        fixed, and no ``p_value`` is emitted anywhere — so there is nothing
        to correct and no best split to select. Re-running with different
        fraction sets until one passes converts it into an uncorrected
        search; the pre-declaration is the only thing preventing that, and
        the library cannot enforce it for you.

    References:
        - [Lopez-de-Prado (2018)][lopez-de-prado-2018]: purging and
          embargoing overlapping train/test windows; CPCV.
        - [McLean-Pontiff (2016)][mclean-pontiff-2016]: post-publication
          decay the ``survival_threshold`` default is calibrated against.

    Examples:
        >>> import factrix as fx
        >>> from factrix.metrics.ic import compute_ic
        >>> from factrix.metrics.oos_decay import oos_decay_splits
        >>> from factrix.preprocess import compute_forward_return
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=240, rng=0),
        ...     forward_periods=5,
        ... )
        >>> series = compute_ic(panel)["factor"].select("date", "ic")
        >>> result = oos_decay_splits(series, value_col="ic", forward_periods=5)
        >>> result.metadata["split_fractions"]
        (0.6, 0.7, 0.8)
        >>> len(result.metadata["splits"])
        3
    """
    declared = _validate_oos_decay_splits(
        split_fractions, survival_threshold, forward_periods
    )
    value_col = _resolve_series_value_col(series, value_col)
    _require_one_row_per_period(series)
    # Cleaned once, up front: every split then runs on the same period grid,
    # and the drop is surfaced once here rather than re-warned per split.
    clean = series.sort("date").filter(_finite_expr(value_col))
    n = clean.height

    splits: list[dict[str, object]] = []
    warning_codes: list[str] = []
    for fraction in declared:
        record, codes = _run_one_split(
            clean,
            value_col,
            fraction=fraction,
            purge_periods=forward_periods,
            survival_threshold=survival_threshold,
            expected_warnings=expected_warnings,
        )
        splits.append(record)
        warning_codes.extend(c for c in codes if c not in warning_codes)

    survivals = [float(s["survival"]) for s in splits]  # type: ignore[arg-type]
    assessable = [v for v in survivals if math.isfinite(v)]
    n_sign_flips = sum(1 for s in splits if s["sign_flipped"])
    all_assessable = len(assessable) == len(splits)
    aggregate = statistics.median(assessable) if all_assessable else float("nan")

    status: GateStatus = (
        "PASS"
        if all_assessable and n_sign_flips == 0 and aggregate >= survival_threshold
        else "VETOED"
    )
    metadata: dict[str, object] = {
        "status": status,
        "splits": tuple(splits),
        "split_fractions": declared,
        "n_splits": len(splits),
        "n_assessable": len(assessable),
        "n_sign_flips": n_sign_flips,
        "aggregate": _AGGREGATE_RULE,
        "sign_flip_policy": _SIGN_FLIP_POLICY,
        "forward_periods": forward_periods,
        "purge_periods": forward_periods,
        "survival_threshold": survival_threshold,
    }
    if not all_assessable:
        metadata["reason"] = "unassessable_splits"
    _surface_null_drop(
        n_periods_in=series.height,
        n_periods_out=n,
        drop_reason="null / NaN / +-inf value observations in the series",
        metric_name="oos_decay_splits",
        metadata=metadata,
        warning_codes=warning_codes,
        expected_warnings=expected_warnings,
    )
    return MetricResult(
        value=aggregate,
        n_obs=n,
        n_obs_axis="periods",
        stat=None,
        metadata=metadata,
        warning_codes=tuple(warning_codes),
    )
