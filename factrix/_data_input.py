"""Input-type gateway for public entry points.

factrix is polars-native: ``fx.evaluate`` accepts
``pl.DataFrame`` (canonical) or ``pl.LazyFrame`` (collected immediately
at the boundary — no projection or predicate pushdown applied by
factrix, so call ``.select(...)`` / ``.filter(...)`` upstream for
memory efficiency on large sources).

``pd.DataFrame`` is **not** accepted on these entry points by design.
pandas users have two clean paths:

- ``factrix.adapt(data, ...)`` — converts and renames columns in one
  step; the natural entry point when pandas column names are not
  already canonical.
- ``pl.from_pandas(data)`` — when columns are already canonical and
  only the type conversion is needed.

This keeps the type contract honest (factrix's internal pipeline is
polars throughout) and avoids hiding the pd → pl copy inside every
``evaluate()`` call.
"""

from __future__ import annotations

import polars as pl
import polars.selectors as cs

from factrix._errors import UserInputError

type DataInput = pl.DataFrame | pl.LazyFrame
"""Accepted panel input type for every data-consuming entry point.

Either an eager ``pl.DataFrame`` or a ``pl.LazyFrame`` carrying the panel
schema (see [Data schema](../api/data-schema.md)). A ``LazyFrame`` is
collected internally, so passing one is purely an ergonomic convenience —
the validation and dispatch contract is identical.
"""

# Canonical input-data schema — single source of truth shared by evaluate's
# baseline validation and the DAG executor's per-factor projection.
# Required columns every panel must carry; optional columns are passed
# through to metrics when present (e.g. ``price`` for event-study metrics,
# ``market_cap`` for value weighting) but never required. Each name here is
# the *declared* one from the data schema: a metric whose column name is
# configurable (``quantile_spread_vw(weight_col=...)``) is only reachable
# through ``evaluate`` under the declared name, since the per-factor
# projection is built before the metric's kwargs are known.
_BASELINE_COLUMNS: tuple[str, ...] = ("date", "asset_id", "forward_return")
_OPTIONAL_COLUMNS: tuple[str, ...] = ("price", "market_cap")

# The key columns a *direct* call to a raw-panel metric must carry.
# ``evaluate`` validates the full baseline once and projects a view per
# factor; a standalone call skips that gate, so this is the floor it is
# checked against before any polars expression runs — the same
# UserInputError surface, not a ColumnNotFoundError from inside a quantile
# join. ``forward_return`` is deliberately not here: a panel metric that never
# reads returns (``rank_turnover``, ``notional_turnover``) is valid on
# ``(date, asset_id, factor)``, and the metrics that do read it name the
# missing column themselves. Consumers of an upstream producer's output
# (``requires``) are not gated: their input is a derived frame whose schema
# the producer owns, not the panel's.
_PANEL_KEY_COLUMNS: tuple[str, ...] = ("date", "asset_id")

# Reserved columns carrying the two horizon-like facts about a panel:
#
# * ``_forward_periods`` — the economic return horizon, the ``forward_periods``
#   ``compute_forward_return`` built ``forward_return`` with. It names the
#   hypothesis (``EvaluationResult.forward_periods``, the multi-factor identity).
# * ``_overlap_periods`` — the overlap of adjacent observations on the panel's
#   evaluation grid, the quantity inference consumes (HAC bandwidth and
#   effective df, the non-overlapping stride, the stride-scaled sample floors).
#   Equal to the horizon on the full grid; smaller when the panel was built on
#   a coarser evaluation grid via ``compute_forward_return(..., dates=)``.
#
# ``compute_forward_return`` stamps both once; ``evaluate`` reads them and
# strips them before dispatch, so they never reach a metric, a projection, or
# ``EvaluationResult.to_frame``. A constant int column is the one carrier that
# survives the ordinary polars transforms a panel goes through between
# construction and evaluation (``with_columns`` winsorize / abnormal-return,
# ``partition_by`` in ``by_slice``, user ``.with_columns(sector)`` / joins) —
# DataFrame-level metadata does not.
_FORWARD_PERIODS_COL: str = "_forward_periods"
_OVERLAP_PERIODS_COL: str = "_overlap_periods"


def _stamp_horizons(
    data: pl.DataFrame, *, forward_periods: int, overlap_periods: int
) -> pl.DataFrame:
    """Stamp the return horizon and the evaluation-grid overlap as reserved columns."""
    return data.with_columns(
        pl.lit(forward_periods, dtype=pl.Int32).alias(_FORWARD_PERIODS_COL),
        pl.lit(overlap_periods, dtype=pl.Int32).alias(_OVERLAP_PERIODS_COL),
    )


def _read_int_stamp(data: pl.DataFrame, column: str) -> int | None:
    if column not in data.columns or data.height == 0:
        return None
    return int(data[column][0])


def _read_forward_periods_stamp(data: pl.DataFrame) -> int | None:
    """Read the stamped return horizon, or ``None`` when the panel carries none."""
    return _read_int_stamp(data, _FORWARD_PERIODS_COL)


def _read_overlap_periods_stamp(data: pl.DataFrame) -> int | None:
    """Read the stamped evaluation-grid overlap, or ``None`` when absent."""
    return _read_int_stamp(data, _OVERLAP_PERIODS_COL)


_DOCS_OVERLAP_PERIODS = "api/evaluate#forward_periods-and-overlap_periods"
_DOCS_FORWARD_PERIODS = "api/evaluate#forward_periods-and-overlap_periods"


def _validate_overlap_periods(declared: object, *, func_name: str) -> int:
    """Type/range-check a caller-declared evaluation-grid overlap.

    Same shape as ``preprocess.returns._validate_forward_periods``: a
    positive ``int``, ``bool`` rejected. Applied wherever a caller declares
    the overlap (``evaluate``, ``by_slice``, the ``slice_period_*`` tests,
    ``sample_requirements``) so a nonsensical value fails at the boundary
    rather than deep inside a stride computation.
    """
    if not isinstance(declared, int) or isinstance(declared, bool):
        raise UserInputError(
            func_name=func_name,
            field="overlap_periods",
            value=declared,
            expected="a positive int count of periods, e.g. 5",
            docs_path=_DOCS_OVERLAP_PERIODS,
        )
    if declared <= 0:
        raise UserInputError(
            func_name=func_name,
            field="overlap_periods",
            value=declared,
            expected="a positive int count of periods (> 0)",
            docs_path=_DOCS_OVERLAP_PERIODS,
        )
    return declared


def _resolve_forward_periods(
    data: pl.DataFrame, declared: int | None, *, func_name: str = "evaluate"
) -> int:
    """Resolve the panel's return horizon for this evaluation.

    Path A (primary): a panel built by ``compute_forward_return`` carries a
    horizon stamp — the single source of truth. Path B (escape hatch): a
    self-attached ``forward_return`` panel carries no stamp, so the caller must
    declare the horizon once via ``forward_periods=`` (a statement about the
    data's overlap, not a per-metric knob). A declaration that disagrees with
    the stamp is rejected rather than silently resolved.
    """
    stamp = _read_forward_periods_stamp(data)
    if stamp is not None:
        if declared is not None and declared != stamp:
            raise UserInputError(
                func_name=func_name,
                field="forward_periods",
                value=declared,
                expected=(
                    f"forward_periods to match the data's stamped return "
                    f"horizon ({stamp}, set by compute_forward_return). The "
                    f"horizon is a property of the data — omit forward_periods, "
                    f"or rebuild forward_return at horizon {declared}."
                ),
                docs_path=_DOCS_FORWARD_PERIODS,
            )
        return stamp
    if declared is not None:
        return declared
    raise UserInputError(
        func_name=func_name,
        field="forward_periods",
        value=None,
        expected=(
            "the data's return horizon. Either build forward_return via "
            "factrix.preprocess.compute_forward_return(data, forward_periods=<forward_periods>) "
            "(which stamps the horizon), or, for a self-attached forward_return "
            f"column, declare it once with {func_name}(..., forward_periods=<forward_periods>)."
        ),
        docs_path=_DOCS_FORWARD_PERIODS,
    )


def _resolve_overlap_periods(
    data: pl.DataFrame,
    declared: int | None,
    *,
    horizon: int | None,
    func_name: str = "evaluate",
) -> int:
    """Resolve the evaluation-grid overlap inference will consume.

    Same contract as :func:`_resolve_forward_periods`: the stamp left by
    ``compute_forward_return`` is the truth and a disagreeing declaration is
    rejected. Callers that also resolve a horizon (``evaluate``) pass it as
    ``horizon``: an unstamped panel then defaults to it, because a
    self-attached ``forward_return`` on the full grid overlaps by exactly its
    horizon and only a coarser grid needs ``overlap_periods=`` spelled out.
    Callers with no horizon of their own (the ``slice_period_*`` tests) pass
    ``horizon=None``, so an unstamped panel must declare the overlap.
    """
    if declared is not None:
        declared = _validate_overlap_periods(declared, func_name=func_name)
    stamp = _read_overlap_periods_stamp(data)
    if stamp is not None:
        if declared is not None and declared != stamp:
            raise UserInputError(
                func_name=func_name,
                field="overlap_periods",
                value=declared,
                expected=(
                    f"overlap_periods to match the data's stamped evaluation-"
                    f"grid overlap ({stamp}, derived by compute_forward_return). "
                    f"The overlap is a property of the data — omit "
                    f"overlap_periods, or rebuild forward_return on the grid "
                    f"that overlaps by {declared} (compute_forward_return(..., "
                    f"dates=...))."
                ),
                docs_path=_DOCS_OVERLAP_PERIODS,
            )
        return stamp
    if declared is not None:
        return declared
    if horizon is not None:
        return horizon
    raise UserInputError(
        func_name=func_name,
        field="overlap_periods",
        value=None,
        expected=(
            "the evaluation-grid overlap. Build forward_return via "
            "factrix.preprocess.compute_forward_return(data, "
            "forward_periods=<forward_periods>) (which stamps both the horizon "
            "and the evaluation-grid overlap), or declare the overlap on the "
            f"unstamped panel with {func_name}(..., "
            "overlap_periods=<overlap_periods>)."
        ),
        docs_path=_DOCS_OVERLAP_PERIODS,
    )


_DOCS_DATA_SCHEMA = "api/data-schema"


def _validate_panel_key_columns(data: object, *, func_name: str) -> None:
    """Reject a direct raw-panel metric call whose frame lacks the key columns.

    Only a ``pl.DataFrame`` is inspected. Mirrors ``evaluate``'s baseline gate
    in error type, field and docs pointer so the two entry points fail the
    same way on the same mistake.
    """
    if not isinstance(data, pl.DataFrame):
        return
    missing = [c for c in _PANEL_KEY_COLUMNS if c not in data.columns]
    if not missing:
        return
    raise UserInputError(
        func_name=func_name,
        field="data",
        value=list(data.columns),
        expected=(
            f"a panel must carry the key columns {list(_PANEL_KEY_COLUMNS)!r}; "
            f"missing {missing!r}. Rename the source columns "
            f"(factrix.adapt(data, date=..., asset_id=..., price=...)) or pass "
            f"the panel through factrix.evaluate, which validates the full schema"
        ),
        docs_path=_DOCS_DATA_SCHEMA,
    )


def _normalize_panel(data: pl.DataFrame) -> pl.DataFrame:
    """Enforce the panel's structural contract once, at the boundary.

    Three guards that were previously applied per producer — so a metric that
    forgot one, or a path that predated it, diverged silently:

    1. **``date`` must be temporal.** Only column *names* were ever validated,
       so a ``String`` date flowed through ``sort`` / ``shift`` / ``over`` /
       ``group_by`` as text and was ordered lexicographically. With ISO-8601
       strings that accidentally works, which is exactly what makes it
       dangerous — it passes every test and every demo. With ``MM/DD/YYYY`` the
       panel is silently reordered and every forward return is computed against
       the wrong neighbour.
    2. **``(date, asset_id)`` must be unique.** The forward return is a
       positional ``shift`` within an asset, so a duplicated row makes the
       "next period" that same date's twin and manufactures a 0.0 return: a
       four-row panel concatenated with itself came back half fabricated zeros,
       with no error and no warning. A duplicated feed is an ordinary ingestion
       accident, and it biases every downstream mean toward zero.
    3. **Non-finite numerics become null.** NaN and ±Inf are structurally
       unrepresentable downstream, which retires the whole class rather than
       patching instances of it — polars ranks NaN as larger than every real
       value, ``pl.corr(method="spearman")`` silently ranks it, and a ``+Inf``
       denominator turns ``finite / inf`` into a *finite* fabricated return
       that sails through an output-side ``is_finite()`` filter.

    ``_finite_expr`` stays in place as defence in depth; it remains correct and
    free once inputs are pre-normalised. Row order is deliberately **not**
    changed here — the shift-based producers sort themselves, and reordering
    every caller's frame at the gate would be a surprise that buys nothing.

    Raises:
        UserInputError: ``date`` is not a temporal dtype, or ``(date,
            asset_id)`` is not unique.
    """
    columns = set(data.columns)

    if "date" in columns:
        dtype = data.schema["date"]
        if not isinstance(dtype, pl.Date | pl.Datetime):
            raise UserInputError(
                func_name="factrix",
                field="date",
                value=str(dtype),
                expected=(
                    "a Date or Datetime column. A string date sorts "
                    "lexicographically, which silently reorders the panel for "
                    "any non-ISO format; parse it first, e.g. "
                    "pl.col('date').str.to_datetime('%m/%d/%Y')"
                ),
                docs_path=_DOCS_DATA_SCHEMA,
            )

    if {"date", "asset_id"} <= columns:
        keys = data.select("date", "asset_id")
        n_duplicated = int(keys.is_duplicated().sum())
        if n_duplicated:
            raise UserInputError(
                func_name="factrix",
                field="(date, asset_id)",
                value=f"{n_duplicated} duplicated row(s) of {data.height}",
                expected=(
                    "one row per (date, asset_id). The forward return shifts by "
                    "row position within an asset, so a duplicate makes the "
                    "'next period' the same date's twin and fabricates a 0.0 "
                    "return. De-duplicate first, e.g. "
                    "data.unique(subset=['date', 'asset_id'], keep='first')"
                ),
                docs_path=_DOCS_DATA_SCHEMA,
            )

    return data.with_columns(
        # Float columns only: they are the only dtypes that can carry NaN /
        # ±inf, and ``is_finite`` is undefined on ``Decimal`` (the default
        # dtype for a DECIMAL / NUMERIC column read from Parquet or a
        # warehouse), which ``cs.numeric()`` would include.
        pl.when(cs.float().is_finite()).then(cs.float()).otherwise(None)
    )


def _is_pandas_dataframe(obj: object) -> bool:
    """Detect ``pd.DataFrame`` without importing pandas (optional dep)."""
    return type(obj).__module__.split(".", 1)[0] == "pandas"


def _coerce_data(data: DataInput) -> pl.DataFrame:
    """Coerce ``DataInput`` to eager ``pl.DataFrame`` and normalise it.

    ``pl.LazyFrame`` is collected immediately. ``pd.DataFrame`` is
    rejected with a ``TypeError`` that points to the documented
    conversion paths. The result passes through :func:`_normalize_panel`,
    the single structural gate every public entry point shares.
    """
    if isinstance(data, pl.DataFrame):
        return _normalize_panel(data)
    if isinstance(data, pl.LazyFrame):
        return _normalize_panel(data.collect())
    if _is_pandas_dataframe(data):
        raise TypeError(
            "data must be pl.DataFrame or pl.LazyFrame; got pandas DataFrame. "
            "factrix is polars-native — convert with `pl.from_pandas(data)`, "
            "or use `factrix.adapt(data, ...)` if column renaming is needed."
        )
    raise TypeError(
        f"data must be pl.DataFrame or pl.LazyFrame; got {type(data).__name__}."
    )
