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

# Reserved column carrying the panel's single overlap horizon (the
# ``forward_periods`` used to build ``forward_return``). ``compute_forward_return``
# stamps it once; ``evaluate`` reads it and strips it before dispatch, so it
# never reaches a metric, a projection, or ``EvaluationResult.to_frame``. A
# constant int column is the one carrier that survives the ordinary polars
# transforms a panel goes through between construction and evaluation
# (``with_columns`` winsorize / abnormal-return, ``partition_by`` in ``by_slice``,
# user ``.with_columns(sector)`` / joins) — DataFrame-level metadata does not.
_FORWARD_PERIODS_COL: str = "_forward_periods"


def _stamp_forward_periods(data: pl.DataFrame, forward_periods: int) -> pl.DataFrame:
    """Stamp the panel's single overlap horizon as a reserved constant column."""
    return data.with_columns(
        pl.lit(forward_periods, dtype=pl.Int32).alias(_FORWARD_PERIODS_COL)
    )


def _read_forward_periods_stamp(data: pl.DataFrame) -> int | None:
    """Read the stamped overlap horizon, or ``None`` when the panel carries none."""
    if _FORWARD_PERIODS_COL not in data.columns or data.height == 0:
        return None
    return int(data[_FORWARD_PERIODS_COL][0])


_DOCS_DATA_SCHEMA = "api/data-schema"


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
        pl.when(cs.numeric().is_finite()).then(cs.numeric()).otherwise(None)
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
